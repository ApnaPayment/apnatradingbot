"""
Kotak Neo WebSocket Streaming Client
Real-time market data feed — replaces 5-minute polling with sub-second ticks.

Usage (standalone):
    ws = KotakNeoWSClient(kotak_client, data_manager)
    ws.subscribe(["22", "3045"], exchange="nse_cm")   # token list
    ws.start()          # background thread
    ...
    ws.stop()

Usage (from main.py):
    ws = KotakNeoWSClient(client, data)
    ws.subscribe_watchlist(WATCHLIST, data)   # auto-resolve tokens
    ws.start()
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Callable, Optional

try:
    import websocket
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False
    websocket = None  # type: ignore

logger = logging.getLogger(__name__)

# Kotak Neo streaming endpoint — adjust path if the API changes
_WS_HOST = "wss://mlhybrid.kotaksecurities.com"
_WS_PATH = "/websocket/v1/feed"


class KotakNeoWSClient:
    """
    WebSocket client for real-time Kotak Neo market data.

    - Runs in a background daemon thread.
    - On each tick, stores data via DataManager._store_tick().
    - Auto-reconnects on disconnect (exponential back-off, max 60 s).
    - Heartbeat sent every 30 s to keep connection alive.
    """

    HEARTBEAT_INTERVAL = 30   # seconds
    MAX_RECONNECT_WAIT = 60   # seconds
    INITIAL_RECONNECT  = 5    # seconds

    def __init__(
        self,
        kotak_client,
        data_manager,
        on_tick: Optional[Callable[[dict], None]] = None,
    ):
        self.client   = kotak_client
        self.data     = data_manager
        self.on_tick  = on_tick       # optional extra callback per tick

        self._ws:               Optional[websocket.WebSocketApp] = None
        self._thread:           Optional[threading.Thread]       = None
        self._heartbeat_thread: Optional[threading.Thread]       = None
        self._subscriptions:    list[str]                        = []   # "exchange|token"
        self._running           = False
        self._reconnect_wait    = self.INITIAL_RECONNECT
        self._fail_count        = 0
        self._max_fails         = 5   # stop retrying after 5 consecutive DNS/connection failures

    # ─────────────────────────────────────────────────────────────────────────
    # Subscription management
    # ─────────────────────────────────────────────────────────────────────────

    def subscribe(self, tokens: list[str], exchange: str = "nse_cm"):
        """
        Queue tokens for subscription.
        Call before start() or call while running to add more instruments.

        Args:
            tokens:   Instrument tokens as strings, e.g. ["22", "3045"]
            exchange: Exchange segment, e.g. "nse_cm", "nse_fo"
        """
        scrips = [f"{exchange}|{t}" for t in tokens]
        for s in scrips:
            if s not in self._subscriptions:
                self._subscriptions.append(s)

        if self._ws and self._running:
            self._send_subscribe(self._ws)

    def subscribe_watchlist(self, symbols: list[str], exchange: str = "nse_cm"):
        """
        Resolve symbols → tokens using scrip master and subscribe.
        Silently skips symbols not found in the DB.
        """
        tokens = []
        for sym in symbols:
            instrument = self.data.get_instrument(sym, exchange)
            if instrument:
                tokens.append(str(instrument["token"]))
            else:
                logger.warning(f"WS: {sym} not found in scrip master — skipping")
        if tokens:
            self.subscribe(tokens, exchange)
        logger.info(f"WS: subscribed {len(tokens)}/{len(symbols)} symbols")

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        """Start the WebSocket in a background daemon thread."""
        if not _WS_AVAILABLE:
            logger.warning("WS: websocket-client not installed. Run: pip install websocket-client")
            return
        if self._running:
            logger.warning("WS: already running")
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("WS: streaming client started")

    def stop(self):
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        logger.info("WS: streaming client stopped")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal — run loop with reconnect
    # ─────────────────────────────────────────────────────────────────────────

    def _run_loop(self):
        while self._running:
            if self._fail_count >= self._max_fails:
                logger.warning(
                    f"WS: {self._fail_count} consecutive failures — "
                    "live tick stream unavailable, using REST polling. Will retry in 10 min."
                )
                time.sleep(600)
                self._fail_count = 0  # reset and try again after 10 min
                continue

            try:
                self._connect()
            except Exception as e:
                logger.error(f"WS: connection error: {e}")
                self._fail_count += 1

            if not self._running:
                break

            logger.info(f"WS: reconnecting in {self._reconnect_wait}s...")
            time.sleep(self._reconnect_wait)
            self._reconnect_wait = min(self._reconnect_wait * 2, self.MAX_RECONNECT_WAIT)

    def _connect(self):
        if not self.client.is_authenticated():
            logger.warning("WS: not authenticated — skipping connection")
            return

        # Build URL with auth query params
        url = (
            f"{_WS_HOST}{_WS_PATH}"
            f"?sid={self.client.session_sid}"
            f"&auth={self.client.session_token}"
        )

        self._ws = websocket.WebSocketApp(
            url,
            header={
                "Authorization": self.client.consumer_key,
                "Sid":           self.client.session_sid,
                "Auth":          self.client.session_token,
            },
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        logger.info(f"WS: connecting to {_WS_HOST}{_WS_PATH}")
        self._ws.run_forever(ping_interval=self.HEARTBEAT_INTERVAL, ping_timeout=10)

    # ─────────────────────────────────────────────────────────────────────────
    # WebSocket handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_open(self, ws):
        logger.info("WS: connection opened")
        self._reconnect_wait = self.INITIAL_RECONNECT   # reset back-off
        self._fail_count = 0                            # reset failure counter
        self._send_subscribe(ws)

    def _on_message(self, ws, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")

        if msg_type == "quote" or "ltp" in msg:
            self._handle_tick(msg)
        elif msg_type == "pong":
            pass
        elif msg_type == "error":
            logger.error(f"WS server error: {msg.get('message', raw)}")
        else:
            logger.debug(f"WS unhandled message type '{msg_type}'")

    def _on_error(self, ws, error):
        logger.error(f"WS error: {error}")

    def _on_close(self, ws, code, msg):
        logger.warning(f"WS closed: code={code} msg={msg}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tick processing
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_tick(self, msg: dict):
        """Parse a raw tick message and store it."""
        try:
            # Kotak Neo tick format (adapt if field names differ in live feed)
            token    = str(msg.get("tk") or msg.get("token", ""))
            exchange = str(msg.get("e")  or msg.get("exchange", "nse_cm"))
            ltp      = float(msg.get("ltp") or msg.get("last_price", 0))
            volume   = int(msg.get("v")   or msg.get("volume", 0))
            bid      = float(msg.get("bp1") or msg.get("bid", 0))
            ask      = float(msg.get("sp1") or msg.get("ask", 0))

            # Reverse-lookup symbol from token (best effort)
            symbol = self._token_to_symbol(token, exchange)

            tick = {
                "symbol":    symbol or token,
                "exchange":  exchange,
                "ltp":       ltp,
                "volume":    volume,
                "bid":       bid,
                "ask":       ask,
                "timestamp": datetime.now().isoformat(),
            }

            self.data._store_tick(tick)

            if self.on_tick:
                self.on_tick(tick)

        except Exception as e:
            logger.debug(f"WS tick parse error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _send_subscribe(self, ws):
        if not self._subscriptions:
            return
        payload = {
            "type":   "subscribe",
            "scrips": "|".join(self._subscriptions),
        }
        try:
            ws.send(json.dumps(payload))
            logger.info(f"WS: subscribed to {len(self._subscriptions)} instruments")
        except Exception as e:
            logger.error(f"WS: subscribe send failed: {e}")

    def _token_to_symbol(self, token: str, exchange: str) -> Optional[str]:
        """Look up symbol from token in scrip master."""
        try:
            with self.data._get_conn() as conn:
                row = conn.execute(
                    "SELECT symbol FROM scrip_master WHERE token=? AND exchange=? LIMIT 1",
                    (token, exchange)
                ).fetchone()
                return row[0] if row else None
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        return (
            self._running
            and self._ws is not None
            and self._ws.sock is not None
            and self._ws.sock.connected
        )

    def status(self) -> dict:
        return {
            "connected":     self.is_connected(),
            "running":       self._running,
            "subscriptions": len(self._subscriptions),
        }
