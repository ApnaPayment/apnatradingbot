"""
Phase 10 — Telegram Interactive Command Handler
Polls Telegram for incoming messages and dispatches operator commands to
the live AlgoTrader instance. Runs in a background thread.

Commands:
  /help           — list all commands
  /status         — positions, P&L, regime, VIX, calendar
  /pause          — halt new entries for the session
  /resume         — re-enable trading
  /close_all      — emergency exit all open positions
  /veto SYMBOL    — block a symbol from trading for this session
  /unveto SYMBOL  — remove a session veto
  /blocked        — list session-vetoed symbols
  /review         — trigger AI self-review on demand
  /mtf SYMBOL     — run multi-timeframe analysis and reply
  /calendar       — show today's expiry and economic events
  /ml             — show ML ensemble model status
"""

import logging
import threading
import time
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)


class TelegramCommandHandler:
    """
    Long-poll Telegram for updates, parse /commands, and call registered handlers.

    Usage:
        handler = TelegramCommandHandler(token, chat_id)
        handler.register("/status", my_status_fn)   # fn() → str
        handler.start()   # non-blocking — spawns a daemon thread
        ...
        handler.stop()
    """

    POLL_TIMEOUT = 30   # seconds — Telegram long-poll window

    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = str(chat_id)
        self._base   = f"https://api.telegram.org/bot{token}"
        self._handlers: dict[str, Callable[..., str]] = {}
        self._offset  = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────────────

    def register(self, command: str, fn: Callable[..., str]):
        """
        Register a command handler.
        fn receives zero or more string args parsed from the message.
        fn must return a plain-text or HTML string to send back.
        """
        self._handlers[command.lower()] = fn

    # ─────────────────────────────────────────────────────────────────────────
    # Thread lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True, name="TelegramCmd")
        self._thread.start()
        logger.info("Telegram command handler started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Telegram command handler stopped")

    # ─────────────────────────────────────────────────────────────────────────
    # Polling loop
    # ─────────────────────────────────────────────────────────────────────────

    def _poll_loop(self):
        while self._running:
            try:
                updates = self._get_updates()
                for upd in updates:
                    self._dispatch(upd)
            except Exception as e:
                logger.warning(f"Telegram poll error: {e}")
                time.sleep(5)

    def _get_updates(self) -> list[dict]:
        try:
            r = requests.get(
                f"{self._base}/getUpdates",
                params={"offset": self._offset, "timeout": self.POLL_TIMEOUT,
                        "allowed_updates": ["message"]},
                timeout=self.POLL_TIMEOUT + 5,
            )
            r.raise_for_status()
            data = r.json()
            updates = data.get("result", [])
            if updates:
                self._offset = updates[-1]["update_id"] + 1
            return updates
        except Exception as e:
            logger.debug(f"getUpdates failed: {e}")
            return []

    def _dispatch(self, update: dict):
        msg = update.get("message", {})
        # Only accept messages from the configured chat
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if chat_id != self.chat_id:
            return

        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return

        parts   = text.split()
        command = parts[0].lower().split("@")[0]   # strip @botname suffix
        args    = parts[1:]

        handler = self._handlers.get(command)
        if handler:
            try:
                response = handler(*args)
                self._send(response or "✓ Done")
            except Exception as e:
                logger.error(f"Command handler {command} failed: {e}")
                self._send(f"⚠ Error: {str(e)[:200]}")
        else:
            self._send(f"Unknown command: <code>{command}</code>\nSend /help for the list.")

    def _send(self, text: str):
        try:
            requests.post(
                f"{self._base}/sendMessage",
                json={"chat_id": self.chat_id, "text": text[:4096],
                      "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"Telegram reply failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Command factory — wires handlers to the live AlgoTrader instance
# ─────────────────────────────────────────────────────────────────────────────

def build_command_handler(bot) -> Optional[TelegramCommandHandler]:
    """
    Create and register all command handlers bound to `bot` (AlgoTrader instance).
    Returns None if Telegram is not configured.
    """
    import os
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        logger.info("Telegram not configured — command handler disabled")
        return None

    handler = TelegramCommandHandler(token, chat_id)

    # ── /help ─────────────────────────────────────────────────────────────
    handler.register("/help", lambda *_: (
        "🤖 <b>AlgoTrader Commands</b>\n\n"
        "/status        — positions, P&amp;L, regime\n"
        "/pause         — halt new trades\n"
        "/resume        — resume trading\n"
        "/close_all     — emergency exit all positions\n"
        "/cb            — circuit breaker state &amp; loss levels\n"
        "/veto SYMBOL   — block symbol this session\n"
        "/unveto SYMBOL — remove session block\n"
        "/blocked       — list blocked symbols\n"
        "/watchlist     — active watchlist with screener scores\n"
        "/review        — trigger AI self-review now\n"
        "/mtf SYMBOL    — multi-timeframe analysis\n"
        "/calendar      — expiry + events today\n"
        "/ml            — ML model status\n"
    ))

    # ── /status ───────────────────────────────────────────────────────────
    def cmd_status(*_):
        portfolio = bot.risk.get_portfolio_summary()
        positions = portfolio.get("open_positions", {})
        pnl       = portfolio.get("daily_pnl", 0)
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        regime    = bot._market_regime
        paused    = "⛔ PAUSED" if bot.trading_paused else "🟢 ACTIVE"
        stats     = bot.get_session_stats()

        ws_icon  = "🟢" if stats["ws_connected"] else "🔴"
        pf_str   = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float("inf") else "∞"
        dd_str   = f"₹{stats['max_drawdown']:,.0f}" if stats["max_drawdown"] > 0 else "₹0"
        ai_total = stats["ai_approved"] + stats["ai_rejected"]
        ai_rate  = f"{stats['ai_approved']}/{ai_total}" if ai_total > 0 else "0/0"

        lines = [
            f"<b>📊 AlgoTrader Status</b>  [{paused}]",
            f"{'─' * 30}",
            f"🕐 Session start:  {stats['session_start']}",
            f"⏱ Uptime:          {stats['uptime_pct']:.0f}%",
            f"{ws_icon} WebSocket:        {'Connected' if stats['ws_connected'] else 'Disconnected'}",
            f"🔍 Last scan:       {stats['last_scan']}",
            f"{'─' * 30}",
            f"📈 Regime:          {regime.upper()}",
            f"{pnl_emoji} Day P&L:          ₹{pnl:+,.0f}",
            f"📂 Open positions:  {len(positions)}",
        ]

        for sym, pos in positions.items():
            quote = bot.data.get_live_quote(sym, pos.get("exchange", "nse_cm"))
            ltp   = quote["ltp"] if quote else pos["entry_price"]
            upnl  = (ltp - pos["entry_price"]) * pos["quantity"]
            lines.append(f"  • {sym}  entry=₹{pos['entry_price']:.1f}  ltp=₹{ltp:.1f}  uP&L=₹{upnl:+,.0f}")

        lines += [
            f"{'─' * 30}",
            f"📡 Signals today:   {stats['signals_today']}",
            f"🤖 AI decisions:    ✅ {stats['ai_approved']} approved  ❌ {stats['ai_rejected']} rejected",
            f"{'─' * 30}",
            f"🏆 Win rate:        {stats['win_rate']:.1f}%  ({stats['trades_closed']} trades)",
            f"💹 Profit factor:   {pf_str}",
            f"📉 Max drawdown:    {dd_str}",
        ]

        try:
            vix = bot.news.get_india_vix()
            if vix:
                lines.append(f"{'─' * 30}")
                lines.append(f"🌡 VIX:             {vix:.1f}")
        except Exception:
            pass

        cal = bot._calendar_ctx
        if cal:
            lines.append(f"📅 Next expiry:     {cal.get('next_expiry', {}).get('label', '?')}")

        return "\n".join(lines)
    handler.register("/status", cmd_status)

    # ── /pause / /resume ──────────────────────────────────────────────────
    def cmd_pause(*_):
        bot.trading_paused = True
        logger.warning("Trading paused via Telegram command")
        return "⛔ Trading paused. Send /resume to re-enable."
    handler.register("/pause", cmd_pause)

    def cmd_resume(*_):
        bot.trading_paused = False
        logger.info("Trading resumed via Telegram command")
        return "🟢 Trading resumed."
    handler.register("/resume", cmd_resume)

    # ── /close_all ────────────────────────────────────────────────────────
    def cmd_close_all(*_):
        positions = dict(bot.risk.open_positions)
        if not positions:
            return "No open positions to close."
        bot.trading_paused = True   # halt new entries while we close
        closed = []
        for sym, pos in list(positions.items()):
            try:
                quote = bot.data.get_live_quote(sym, pos.get("exchange", "nse_cm"))
                price = quote["ltp"] if quote else pos["entry_price"]
                bot._exit_position(sym, pos, price, "manual_close_all")
                closed.append(sym)
            except Exception as e:
                logger.error(f"close_all failed for {sym}: {e}")
        return (
            f"✅ Closed {len(closed)} position(s): {', '.join(closed)}\n"
            f"Trading paused. Send /resume when ready."
        )
    handler.register("/close_all", cmd_close_all)

    # ── /veto / /unveto / /blocked ────────────────────────────────────────
    if not hasattr(bot, "_session_vetoes"):
        bot._session_vetoes: set[str] = set()

    def cmd_veto(*args):
        if not args:
            return "Usage: /veto SYMBOL  (e.g. /veto RELIANCE-EQ)"
        sym = args[0].upper()
        bot._session_vetoes.add(sym)
        logger.info(f"Session veto added: {sym}")
        return f"🚫 {sym} blocked for this session. /unveto {sym} to remove."
    handler.register("/veto", cmd_veto)

    def cmd_unveto(*args):
        if not args:
            return "Usage: /unveto SYMBOL"
        sym = args[0].upper()
        bot._session_vetoes.discard(sym)
        return f"✅ {sym} unblocked."
    handler.register("/unveto", cmd_unveto)

    def cmd_blocked(*_):
        if not bot._session_vetoes:
            return "No session vetoes active."
        return "🚫 Session vetoes:\n" + "\n".join(f"  • {s}" for s in sorted(bot._session_vetoes))
    handler.register("/blocked", cmd_blocked)

    # ── /review ───────────────────────────────────────────────────────────
    def cmd_review(*_):
        try:
            from ai.feedback_loop import FeedbackLoop
            fb     = FeedbackLoop(bot.ai)
            review = fb.run_weekly_review()
            return f"🧠 <b>AI Self-Review (on-demand)</b>\n\n{review}" if review else "No decisions to review yet."
        except Exception as e:
            return f"Review failed: {e}"
    handler.register("/review", cmd_review)

    # ── /mtf SYMBOL ───────────────────────────────────────────────────────
    def cmd_mtf(*args):
        if not args:
            return "Usage: /mtf SYMBOL  (e.g. /mtf RELIANCE-EQ)"
        sym = args[0].upper()
        try:
            result = bot.mtf.analyze(sym, "BUY")
            if result is None:
                return f"MTF: insufficient data for {sym}"
            d = result.to_dict()
            lines = [
                f"<b>MTF: {sym}</b>",
                f"Verdict: {d['verdict'].upper()}  score={d['score']:.2f}",
            ]
            for label, key in [("5-min", "tf5"), ("15-min", "tf15"), ("30-min", "tf30")]:
                tf = d.get(key, {})
                lines.append(
                    f"  {label}: {tf.get('aligned','?')}  "
                    f"EMA {tf.get('ema_fast','?'):.1f}/{tf.get('ema_slow','?'):.1f}  "
                    f"RSI {tf.get('rsi','?')}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"MTF error: {e}"
    handler.register("/mtf", cmd_mtf)

    # ── /calendar ─────────────────────────────────────────────────────────
    def cmd_calendar(*_):
        try:
            from data.calendar import get_calendar_context
            ctx   = get_calendar_context()
            dte   = ctx.get("days_to_expiry", "?")
            exp   = ctx.get("next_expiry", {}).get("label", "?")
            level = ctx.get("risk_level", "normal").upper()
            lines = [f"📅 <b>Calendar</b>  [{level}]", f"Expiry: {exp}  ({dte}d away)"]
            for ev in ctx.get("economic_events_14d", []):
                lines.append(f"⚡ {ev['label']} — in {ev['days_away']}d")
            for note in ctx.get("trading_notes", []):
                lines.append(f"⚠ {note}")
            return "\n".join(lines) if len(lines) > 1 else "No events in next 14 days."
        except Exception as e:
            return f"Calendar error: {e}"
    handler.register("/calendar", cmd_calendar)

    # ── /cb ───────────────────────────────────────────────────────────────
    def cmd_cb(*_):
        cb = getattr(bot, "cb", None)
        if cb is None:
            return "Circuit breaker not initialised."
        portfolio = bot.risk.get_portfolio_summary()
        positions = portfolio.get("open_positions", {})
        for sym, pos in positions.items():
            q = bot.data.get_live_quote(sym, pos.get("exchange", "nse_cm"))
            if q:
                pos["current_price"] = q["ltp"]
        status = cb.check(portfolio.get("daily_pnl", 0), positions)
        emoji = {"normal": "🟢", "caution": "🟡", "halt": "🔴", "close": "🆘"}.get(
            status.state.value, "⚪"
        )
        return (
            f"{emoji} <b>Circuit Breaker: {status.state.value.upper()}</b>\n"
            f"Effective loss: ₹{status.effective_loss:,.0f}\n"
            f"  Realised: ₹{status.realised_loss:,.0f}\n"
            f"  Unrealised (50%): ₹{status.unrealised_loss * 0.5:,.0f}\n"
            f"Thresholds: L1=₹{status.thresholds['l1']:,.0f}  "
            f"L2=₹{status.thresholds['l2']:,.0f}  "
            f"L3=₹{status.thresholds['l3']:,.0f}\n"
            f"New entries: {'✅' if status.allow_new_entries else '🚫'}"
        )
    handler.register("/cb", cmd_cb)

    # ── /watchlist ────────────────────────────────────────────────────────
    def cmd_watchlist(*_):
        syms = getattr(bot, "active_watchlist", [])
        if not syms:
            return "No active watchlist — screener hasn't run yet."
        try:
            from data.screener import get_latest_screener_results
            top = {r["symbol"]: r for r in get_latest_screener_results(limit=30)}
        except Exception:
            top = {}
        lines = [f"📊 <b>Active Watchlist</b>  ({len(syms)} symbols)"]
        for s in syms:
            info = top.get(s)
            suffix = (
                f"  score={info['score']:.2f}  RSI={info['rsi']:.0f}"
                if info and info.get("rsi") else ""
            )
            lines.append(f"  {'●' if (info or {}).get('promoted') else '·'} {s}{suffix}")
        return "\n".join(lines)
    handler.register("/watchlist", cmd_watchlist)

    # ── /ml ───────────────────────────────────────────────────────────────
    def cmd_ml(*_):
        try:
            from ai.ml_ensemble import get_model_stats
            stats = get_model_stats()
            if stats["status"] == "untrained":
                return f"🤖 ML: untrained  ({stats['reason']})"
            auc_str = f"  AUC={stats['auc']:.3f}" if stats.get("auc") else ""
            top_str = "  ".join(f"{f}={i:.3f}" for f, i in (stats.get("top_features") or []))
            return (
                f"🤖 <b>ML Model</b>  ● READY\n"
                f"Trained on {stats['n_samples']} samples{auc_str}\n"
                f"Top features: {top_str}\n"
                f"Trained: {(stats.get('trained_at') or '')[:16]}"
            )
        except Exception as e:
            return f"ML error: {e}"
    handler.register("/ml", cmd_ml)

    return handler
