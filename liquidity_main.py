"""
Liquidity Engine — Main Orchestrator
Parallel paper-trading engine for BSE equity, MCX commodities, and NSE Currency Derivatives.

Runs as a SEPARATE PROCESS alongside the existing NSE AlgoTrader (main.py).
Zero modification to main.py or any existing code.

Key design constraints:
- No Kotak authentication (avoids invalidating NSE bot's session)
- Uses yfinance for all market data (paper trading accuracy is sufficient)
- Writes only to liq_* DB tables (existing NSE tables untouched)
- Shares same ANTHROPIC_API_KEY for AI decisions

Run:
    python liquidity_main.py
    python liquidity_main.py --segments bse_cm          # BSE only (default Phase 1)
    python liquidity_main.py --segments bse_cm,mcx_fo   # Phase 2
    python liquidity_main.py --dry-run                  # non-market-hours test
"""

import argparse
import logging
import os
import schedule
import signal
import sys
import time
from datetime import date, datetime, time as dtime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ─── Logging ─────────────────────────────────────────────────────────────────
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_dir / f"liq_trader_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── Imports ─────────────────────────────────────────────────────────────────
from core.risk_manager import RiskManager, RiskConfig, TradeSignal
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from ai.decision_engine import AIDecisionEngine
from alerts.telegram_bot import TelegramAlerter
from data.calendar import get_calendar_context
from data.liquidity_data import LiquidityDataFetcher, BSE_WATCHLIST, MCX_TICKERS
from data.liquidity_db import LiquidityDataManager

# ─── Configuration ───────────────────────────────────────────────────────────
LIQ_PAPER_TRADING    = os.getenv("LIQ_PAPER_TRADING",  "true").lower() == "true"
LIQ_MAX_CAPITAL      = float(os.getenv("LIQ_MAX_CAPITAL",       "25000"))
LIQ_MAX_POSITIONS    = int(os.getenv("LIQ_MAX_POSITIONS",        "3"))
LIQ_MAX_DAILY_LOSS   = float(os.getenv("LIQ_MAX_DAILY_LOSS_PCT", "3"))
LIQ_SEGMENTS         = os.getenv("LIQ_SEGMENTS", "bse_cm").split(",")

# MCX commodities require much larger paper capital — a single GOLD lot is ~₹4.3L,
# CRUDE is ~₹8.9k/bbl × 100 bbl = ₹8.9L per lot. ₹10L paper capital covers 1–2 lots.
LIQ_MCX_MAX_CAPITAL  = float(os.getenv("LIQ_MCX_MAX_CAPITAL",   "1000000"))  # ₹10L
LIQ_MCX_MAX_POS      = int(os.getenv("LIQ_MCX_MAX_POSITIONS",   "2"))

# BSE equity — same symbols as NSE watchlist, different exchange suffix for yfinance
BSE_YF_WATCHLIST = BSE_WATCHLIST  # from liquidity_data.py

# MCX commodities — keys match MCX_TICKERS in liquidity_data.py
MCX_WATCHLIST = list(MCX_TICKERS.keys())  # ["GOLD","SILVER","CRUDEOIL","NATURALGAS","COPPER"]

# Symbol map: yfinance ticker → clean display symbol
def _yf_to_symbol(ticker_bo: str) -> str:
    """RELIANCE.BO → RELIANCE"""
    return ticker_bo.replace(".BO", "")

# NSE holidays (same calendar applies to BSE)
NSE_HOLIDAYS: set[date] = {
    date(2025, 10, 2), date(2025, 11, 5), date(2025, 12, 25),
    date(2026, 1, 26), date(2026, 2, 19), date(2026, 3, 31),
    date(2026, 4, 2),  date(2026, 4, 10), date(2026, 4, 14),
    date(2026, 4, 15), date(2026, 5, 1),  date(2026, 8, 15),
    date(2026, 10, 2),
}


class LiquidityEngine:
    """
    Parallel paper-trading engine for non-NSE segments.
    Phase 1: BSE equity via yfinance.
    Phase 2: MCX commodities + NSE CDE (currency).
    """

    def __init__(self, segments: list = None, dry_run: bool = False):
        self.segments  = segments or ["bse_cm"]
        self.dry_run   = dry_run
        self._running  = False
        self._cycle_count = 0

        logger.info("=" * 60)
        logger.info("Liquidity Engine Starting Up")
        logger.info("=" * 60)
        logger.info(f"Segments: {self.segments}")
        logger.info(f"Paper trading: {LIQ_PAPER_TRADING}")
        logger.info(f"Max capital: ₹{LIQ_MAX_CAPITAL:,.0f}")
        logger.info(f"Dry run: {dry_run}")

        # ── Risk Manager for equity segments (BSE/CDE) ──────────────────────
        risk_cfg = RiskConfig(
            max_capital          = LIQ_MAX_CAPITAL,
            max_position_size_pct= 30,      # larger % because capital is small
            max_daily_loss_pct   = LIQ_MAX_DAILY_LOSS,
            max_open_positions   = LIQ_MAX_POSITIONS,
            trailing_stop_pct    = 1.5,
        )
        self.risk = RiskManager(config=risk_cfg)

        # ── Risk Manager for MCX commodities (separate capital pool) ─────────
        # MCX commodity lots are expensive (GOLD ~₹4.3L, CRUDE ~₹8.9L/lot).
        # ₹10L paper capital allows 1–2 commodity lot positions.
        mcx_risk_cfg = RiskConfig(
            max_capital          = LIQ_MCX_MAX_CAPITAL,
            max_position_size_pct= 50,   # up to 50% per commodity position
            max_daily_loss_pct   = LIQ_MAX_DAILY_LOSS,
            max_open_positions   = LIQ_MCX_MAX_POS,
            trailing_stop_pct    = 2.0,  # 2% trail for commodities (wider than equity)
        )
        self.mcx_risk = RiskManager(config=mcx_risk_cfg)

        # ── Strategies (reused, zero modification) ──────────────────────────
        self.momentum       = MomentumStrategy()
        self.mean_reversion = MeanReversionStrategy()

        # ── Commodity strategy (MCX) ─────────────────────────────────────────
        from strategies.commodity import CommodityStrategy
        self.commodity = CommodityStrategy()

        # ── AI Decision Engine (shared ANTHROPIC_API_KEY) ───────────────────
        self.ai = AIDecisionEngine()

        # ── Telegram Alerts ─────────────────────────────────────────────────
        self.telegram = TelegramAlerter()

        # ── Data layer ──────────────────────────────────────────────────────
        self.fetcher = LiquidityDataFetcher()
        self.db      = LiquidityDataManager()

        # In-memory OHLCV cache: symbol → DataFrame
        self._ohlcv: dict = {}

        # ── Shared Kotak session (written by NSE bot, read here) ─────────────
        # Loaded fresh before each MCX cycle; None = use yfinance fallback
        self._kotak_session: dict = None
        self._session_loaded_at: datetime = None

        self.db.update_bot_status(status="starting", segments_active=self.segments)
        logger.info("Liquidity Engine initialised")

    # ─── Position persistence ─────────────────────────────────────────────────

    def _persist_positions(self):
        """Save both risk managers' open positions to DB immediately."""
        all_positions = {**self.risk.open_positions, **self.mcx_risk.open_positions}
        self.db.update_bot_status(open_positions=all_positions)

    def _restore_positions(self):
        """Restore open positions from DB into the correct risk manager on startup."""
        status = self.db.get_bot_status()
        saved = status.get("open_positions", {})
        if not saved:
            return
        restored = 0
        for symbol, pos in saved.items():
            exchange = pos.get("exchange", "")
            if exchange == "mcx_fo":
                self.mcx_risk.open_positions[symbol] = pos
            else:
                self.risk.open_positions[symbol] = pos
            restored += 1
        if restored:
            logger.info(f"LiqRestore: {restored} open position(s) restored from DB: {list(saved.keys())}")

    # ─── Market hours check ──────────────────────────────────────────────────

    def _is_market_open(self, segment: str = "bse_cm") -> bool:
        """
        Return True if the given segment's market is currently open.
        BSE equity: 09:15–15:30, Mon–Fri, excluding NSE holidays.
        """
        today = date.today()
        if today.weekday() >= 5:        # Saturday / Sunday
            return False
        if today in NSE_HOLIDAYS:
            return False
        now = datetime.now().time()
        if segment in ("bse_cm", "nse_cm"):
            return dtime(9, 15) <= now <= dtime(15, 30)
        if segment == "mcx_fo":
            return dtime(9, 0) <= now <= dtime(23, 30)
        if segment == "cde_fo":
            return dtime(9, 0) <= now <= dtime(17, 0)
        return False

    def _is_trading_window(self) -> bool:
        """
        Only enter new positions in the high-quality window (09:15–10:30).
        Mirrors the V2 strategy time gate from the NSE bot.
        """
        now = datetime.now().time()
        return dtime(9, 15) <= now <= dtime(10, 30)

    # ─── Daily setup ─────────────────────────────────────────────────────────

    def job_daily_setup(self):
        """Run at 08:45 each trading day. Bootstrap data for all active segments."""
        today = date.today()
        if today.weekday() >= 5 or today in NSE_HOLIDAYS:
            logger.info("LiqSetup: weekend/holiday — skipping")
            return

        logger.info("LiqSetup: bootstrapping data...")

        if "bse_cm" in self.segments:
            results = self.fetcher.bootstrap_bse(BSE_YF_WATCHLIST)
            total = sum(results.values())
            logger.info(f"LiqSetup: BSE bootstrap complete — {total} candles across {len(results)} symbols")
            # Pre-load into memory cache
            for ticker in BSE_YF_WATCHLIST:
                df = self.fetcher.get_ohlcv_bse(ticker)
                if df is not None:
                    symbol = _yf_to_symbol(ticker)
                    self._ohlcv[symbol] = df

        if "mcx_fo" in self.segments:
            results = self.fetcher.bootstrap_mcx(MCX_WATCHLIST)
            total = sum(results.values())
            logger.info(f"LiqSetup: MCX bootstrap complete — {total} candles across {len(results)} commodities")
            for commodity in MCX_WATCHLIST:
                df = self.fetcher.get_ohlcv_mcx(commodity)
                if df is not None:
                    self._ohlcv[f"MCX_{commodity}"] = df

        self.db.reset_daily()
        # Reset in-memory daily counters for both risk managers
        self.risk.daily_pnl = 0.0
        self.risk.daily_trades = 0
        self.mcx_risk.daily_pnl = 0.0
        self.mcx_risk.daily_trades = 0
        self.db.update_bot_status(status="running", segments_active=self.segments)
        logger.info("LiqSetup: complete — engine ready for trading")

        # Send Telegram morning brief
        try:
            cal = get_calendar_context()
            msg = (
                f"[LIQ] 🌅 Liquidity Engine ready\n"
                f"Segments: {', '.join(self.segments)}\n"
                f"Capital: ₹{LIQ_MAX_CAPITAL:,.0f} | Max positions: {LIQ_MAX_POSITIONS}\n"
                f"Calendar: {cal.get('risk_level','normal').upper()}"
            )
            self.telegram.send(msg)
        except Exception:
            pass

    # ─── Trading cycle ────────────────────────────────────────────────────────

    def run_trading_cycle(self):
        """Main 5-minute cycle. Runs for all active segments."""
        self._cycle_count += 1
        now = datetime.now()

        for segment in self.segments:
            if not (self._is_market_open(segment) or self.dry_run):
                continue
            try:
                if segment == "bse_cm":
                    self._run_bse_cycle()
                elif segment == "mcx_fo":
                    self._run_mcx_cycle()    # Phase 2
                elif segment == "cde_fo":
                    self._run_cde_cycle()    # Phase 2
            except Exception as e:
                logger.error(f"LiqCycle: {segment} error: {e}", exc_info=True)

        # Update DB heartbeat
        # Merge positions from both risk managers for DB heartbeat
        all_positions = {**self.risk.open_positions, **self.mcx_risk.open_positions}
        self.db.update_bot_status(
            daily_pnl    = self.risk.daily_pnl + self.mcx_risk.daily_pnl,
            daily_trades = self.risk.daily_trades + self.mcx_risk.daily_trades,
            open_positions = all_positions,
        )

    def _run_bse_cycle(self):
        """BSE equity 5-minute scan."""
        in_window = self._is_trading_window() or self.dry_run
        signals = []

        for ticker_bo in BSE_YF_WATCHLIST:
            symbol = _yf_to_symbol(ticker_bo)

            # Refresh candles — use cache if fresh, else re-fetch
            df = self.fetcher.get_ohlcv_bse(ticker_bo)
            if df is None or len(df) < 50:
                continue
            self._ohlcv[symbol] = df

            # Monitor open position exits first
            if symbol in self.risk.open_positions:
                self._monitor_bse_position(symbol, ticker_bo, df)
                continue  # one position per symbol; skip signal generation while open

            # Only generate new signals during the trading window
            if not in_window:
                continue

            # Momentum signal
            try:
                sig = self.momentum.generate_signal(symbol, df, exchange="bse_cm")
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug(f"LiqBSE: momentum error {symbol}: {e}")

            # Mean reversion signal
            try:
                sig = self.mean_reversion.generate_signal(symbol, df, exchange="bse_cm")
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug(f"LiqBSE: MR error {symbol}: {e}")

        if signals:
            logger.info(f"LiqBSE: {len(signals)} signal(s) from {len(BSE_YF_WATCHLIST)} symbols")
        else:
            logger.debug(f"LiqBSE: no signals this cycle (window={in_window})")

        for sig in signals:
            self._process_signal(sig)

    def _monitor_bse_position(self, symbol: str, ticker_bo: str, df):
        """Check stop-loss, target, trailing-stop for an open BSE position."""
        pos = self.risk.open_positions.get(symbol)
        if not pos:
            return

        quote = self.fetcher.get_live_quote_bse(ticker_bo)
        current = quote["ltp"] if quote else float(df["close"].iloc[-1])

        # Stop-loss check
        if self.risk.check_stop_loss_hit(symbol, current):
            self._exit_bse_position(symbol, current, "stop_loss")
            return

        # Target check
        if self.risk.check_target_hit(symbol, current):
            self._exit_bse_position(symbol, current, "target")
            return

        # Trailing stop update
        self.risk.update_trailing_stop(symbol, current)

    # ─── Signal processing ────────────────────────────────────────────────────

    def _risk_for(self, exchange: str) -> "RiskManager":
        """Return the appropriate RiskManager for the given exchange."""
        return self.mcx_risk if exchange == "mcx_fo" else self.risk

    def _max_capital_for(self, exchange: str) -> float:
        return LIQ_MCX_MAX_CAPITAL if exchange == "mcx_fo" else LIQ_MAX_CAPITAL

    def _process_signal(self, signal: TradeSignal):
        """
        Risk check → AI evaluation → paper execution.
        Mirrors the NSE bot's _process_signal() but simplified.
        MCX signals use the dedicated mcx_risk manager (larger capital pool).
        """
        risk = self._risk_for(signal.exchange)
        max_cap = self._max_capital_for(signal.exchange)

        # Risk approval
        available = max_cap - risk.get_portfolio_summary()["capital_at_risk"]
        approved, reason = risk.approve_trade(signal, available, {})
        if not approved:
            logger.info(f"LiqRisk: REJECTED {signal.symbol} — {reason}")
            self.db.record_ai_decision(
                symbol=signal.symbol, exchange=signal.exchange,
                action=signal.action, strategy=signal.strategy,
                signal_price=signal.price, signal_conf=signal.confidence,
                approved=False, ai_reasoning="", veto_reason=f"Risk gate: {reason}",
                stop_loss=signal.stop_loss, target=signal.target,
            )
            return

        # Calendar context for AI
        cal_ctx = get_calendar_context()

        # AI evaluation
        portfolio_ctx = risk.get_portfolio_summary()
        ai_result = self.ai.evaluate_signal(
            signal,
            portfolio=portfolio_ctx,
            market_context={
                "segment": signal.exchange,
                "calendar": cal_ctx,
                "paper_trading": LIQ_PAPER_TRADING,
            },
        )

        decision_id = self.db.record_ai_decision(
            symbol       = signal.symbol,
            exchange     = signal.exchange,
            action       = signal.action,
            strategy     = signal.strategy,
            signal_price = signal.price,
            signal_conf  = signal.confidence,
            approved     = ai_result["approved"],
            ai_reasoning = ai_result.get("reasoning", ""),
            veto_reason  = "" if ai_result["approved"] else ai_result.get("reasoning", ""),
            stop_loss    = signal.stop_loss,
            target       = signal.target,
            market_regime= cal_ctx.get("risk_level", "unknown"),
        )

        if not ai_result["approved"]:
            logger.info(f"LiqAI: VETOED {signal.symbol} — {ai_result.get('reasoning','')[:80]}")
            return

        # Execute paper trade
        self._execute_paper_trade(signal, decision_id, risk)

    def _execute_paper_trade(self, signal: TradeSignal, decision_id: int,
                             risk: "RiskManager" = None):
        """
        Simulate order execution. No Kotak API call.
        Records entry in the appropriate risk manager and DB.
        """
        if risk is None:
            risk = self._risk_for(signal.exchange)
        max_cap = self._max_capital_for(signal.exchange)

        order_no = f"PAPER_LIQ_{int(datetime.now().timestamp())}"

        # For MCX commodities, quantity is pre-set to lot size by CommodityStrategy.
        # For BSE equity, calculate quantity using risk-based sizing.
        if signal.exchange == "mcx_fo":
            # MCX: use strategy lot size (already set), just check capital fits
            qty = signal.quantity  # e.g. 1 lot of GOLD
            lot_value = signal.price * qty
            available = max_cap - risk.get_portfolio_summary()["capital_at_risk"]
            if lot_value > available:
                logger.warning(
                    f"LiqExec: {signal.symbol} lot cost ₹{lot_value:,.0f} > "
                    f"available ₹{available:,.0f} — skipping"
                )
                if decision_id:
                    self.db.update_ai_decision_outcome(decision_id, "skipped_no_capital", 0.0)
                return
        else:
            # BSE equity: dynamic sizing
            available = max_cap - risk.get_portfolio_summary()["capital_at_risk"]
            qty = risk.calculate_quantity(
                price=signal.price,
                available_capital=available,
                confidence=signal.confidence,
            )
            if qty <= 0:
                logger.warning(f"LiqExec: zero qty for {signal.symbol} — capital insufficient, skipping")
                if decision_id:
                    self.db.update_ai_decision_outcome(decision_id, "skipped_no_capital", 0.0)
                return

        signal.quantity = qty
        risk.record_entry(signal, order_no)
        # Attach decision_id for outcome linkage on exit
        if signal.symbol in risk.open_positions:
            risk.open_positions[signal.symbol]["_liq_decision_id"] = decision_id

        lot_cost = round(signal.price * qty, 2)
        logger.info(
            f"[LIQ] PAPER {signal.action} {signal.symbol} ({signal.exchange}) "
            f"@ ₹{signal.price:.2f} × {qty} = ₹{lot_cost:,.2f} | "
            f"SL ₹{signal.stop_loss:.2f} | Target ₹{signal.target:.2f} | "
            f"Strategy: {signal.strategy}"
        )

        # Persist positions to DB immediately so a restart doesn't lose this trade
        self._persist_positions()

        try:
            msg = (
                f"[LIQ] 📊 PAPER {signal.action} — {signal.symbol}\n"
                f"Exchange: {signal.exchange}\n"
                f"Entry: ₹{signal.price:.2f} × {qty} = ₹{lot_cost:,.0f}\n"
                f"SL: ₹{signal.stop_loss:.2f} | Target: ₹{signal.target:.2f}\n"
                f"Strategy: {signal.strategy} | Conf: {signal.confidence:.0%}"
            )
            self.telegram.send(msg)
        except Exception:
            pass

    # ─── Position exit ────────────────────────────────────────────────────────

    def _exit_bse_position(self, symbol: str, exit_price: float, reason: str):
        """Record exit for a BSE position."""
        pos = self.risk.open_positions.get(symbol)
        if not pos:
            return

        pnl = self.risk.record_exit(symbol, exit_price)
        decision_id = pos.get("_liq_decision_id")

        trade_id = self.db.record_trade(
            symbol      = symbol,
            exchange    = pos.get("exchange", "bse_cm"),
            action      = pos.get("action", "BUY"),
            strategy    = pos.get("strategy", "unknown"),
            entry_price = pos["entry_price"],
            exit_price  = exit_price,
            quantity    = pos["quantity"],
            pnl         = pnl,
            exit_reason = reason,
            entry_time  = pos.get("entry_time", ""),
            exit_time   = datetime.now().isoformat(),
            order_no    = pos.get("order_no", ""),
        )

        if decision_id:
            outcome = "win" if pnl > 0 else "loss" if pnl < 0 else "breakeven"
            self.db.update_ai_decision_outcome(decision_id, outcome, pnl)

        sign = "✅" if pnl > 0 else "❌"
        logger.info(
            f"[LIQ] EXIT {symbol} @ ₹{exit_price:.2f} "
            f"| P&L: ₹{pnl:+.2f} | Reason: {reason}"
        )
        self._persist_positions()

        try:
            msg = (
                f"[LIQ] {sign} EXIT {symbol}\n"
                f"Price: ₹{exit_price:.2f} | P&L: ₹{pnl:+.2f}\n"
                f"Reason: {reason} | Day P&L: ₹{self.risk.daily_pnl:+.2f}"
            )
            self.telegram.send(msg)
        except Exception:
            pass

    def _squareoff_all(self):
        """EOD squareoff — close all open positions at last known price."""
        eq_positions  = dict(self.risk.open_positions)
        mcx_positions = dict(self.mcx_risk.open_positions)
        total = len(eq_positions) + len(mcx_positions)
        if not total:
            return
        logger.info(f"[LIQ] EOD squareoff: {total} positions")
        for symbol, pos in eq_positions.items():
            df = self._ohlcv.get(symbol)
            price = float(df["close"].iloc[-1]) if df is not None else pos["entry_price"]
            self._exit_bse_position(symbol, price, "eod_squareoff")
        for key, pos in mcx_positions.items():
            commodity = key.replace("MCX_", "")
            df = self._ohlcv.get(key)
            price = float(df["close"].iloc[-1]) if df is not None else pos["entry_price"]
            self._exit_mcx_position(key, commodity, price, "eod_squareoff")

    # ─── MCX Commodity cycle ─────────────────────────────────────────────────

    def _refresh_kotak_session(self):
        """
        Load (or refresh) the shared Kotak session token written by the NSE bot.
        Re-reads every 30 minutes or when session is None.
        Falls back to yfinance if NSE bot hasn't authenticated yet.
        """
        now = datetime.now()
        # Refresh if never loaded or older than 30 min
        if (self._kotak_session is None or self._session_loaded_at is None or
                (now - self._session_loaded_at).total_seconds() > 1800):
            try:
                sess = self.db.get_session_token()
                if sess:
                    self._kotak_session = sess
                    self._session_loaded_at = now
                    logger.info("LiqMCX: Kotak session loaded from shared DB — real MCX quotes active")
                else:
                    if self._kotak_session is not None:
                        logger.warning("LiqMCX: Kotak session expired/missing — falling back to yfinance")
                    self._kotak_session = None
            except Exception as e:
                logger.warning(f"LiqMCX: Failed to load Kotak session: {e} — using yfinance")
                self._kotak_session = None

    def _get_mcx_quote(self, commodity: str, df) -> float:
        """
        Get best available MCX price:
        1. Real Kotak MCX API (if session available)
        2. yfinance fast_info (delayed but free)
        3. Last OHLCV close (always available)
        """
        # Try real Kotak MCX quote
        if self._kotak_session:
            q = self.fetcher.get_live_quote_mcx_kotak(
                commodity,
                session_token = self._kotak_session["token"],
                session_sid   = self._kotak_session["sid"],
                base_url      = self._kotak_session["base_url"],
            )
            if q and q.get("ltp", 0) > 0:
                return float(q["ltp"])
            # Kotak quote failed — mark session for refresh next cycle
            logger.debug(f"LiqMCX: Kotak quote failed for {commodity} — retrying yfinance")

        # Try yfinance fallback
        q = self.fetcher.get_live_quote_mcx(commodity)
        if q and q.get("ltp", 0) > 0:
            return float(q["ltp"])

        # Last resort: latest OHLCV close
        return float(df["close"].iloc[-1])

    def _run_mcx_cycle(self):
        """MCX commodities 5-minute scan. Uses 1h OHLCV from yfinance COMEX proxies."""
        # Refresh shared Kotak session for real MCX quotes
        self._refresh_kotak_session()

        signals = []

        for commodity in MCX_WATCHLIST:
            key = f"MCX_{commodity}"

            # Refresh OHLCV (cached for 4 min inside LiquidityDataFetcher)
            df = self.fetcher.get_ohlcv_mcx(commodity)
            if df is None or len(df) < 60:
                logger.debug(f"LiqMCX: insufficient data for {commodity} ({len(df) if df is not None else 0} bars)")
                continue
            self._ohlcv[key] = df

            # Monitor existing position exits first
            if key in self.mcx_risk.open_positions:
                self._monitor_mcx_position(commodity, key, df)
                continue  # one position per commodity at a time

            # Signal generation
            try:
                sig = self.commodity.generate_signal(commodity, df, exchange="mcx_fo")
                if sig:
                    sig.symbol = key   # normalise symbol to "MCX_GOLD" etc. for position tracking
                    signals.append(sig)
            except Exception as e:
                logger.debug(f"LiqMCX: strategy error {commodity}: {e}")

        if signals:
            logger.info(f"LiqMCX: {len(signals)} signal(s) from {len(MCX_WATCHLIST)} commodities")
        else:
            logger.info(f"LiqMCX: no signals this cycle ({len(MCX_WATCHLIST)} commodities scanned)")

        for sig in signals:
            self._process_signal(sig)

    def _monitor_mcx_position(self, commodity: str, key: str, df):
        """Check stop-loss, target, trailing-stop for an open MCX position."""
        pos = self.mcx_risk.open_positions.get(key)
        if not pos:
            return

        # Get best available price (Kotak → yfinance → last close)
        current = self._get_mcx_quote(commodity, df)

        if self.mcx_risk.check_stop_loss_hit(key, current):
            self._exit_mcx_position(key, commodity, current, "stop_loss")
            return

        if self.mcx_risk.check_target_hit(key, current):
            self._exit_mcx_position(key, commodity, current, "target")
            return

        # Update trailing stop (uses 15% trail for NRML — per Bug 3 fix)
        self.mcx_risk.update_trailing_stop(key, current)

    def _exit_mcx_position(self, key: str, commodity: str,
                           exit_price: float, reason: str):
        """Record exit for an MCX position."""
        pos = self.mcx_risk.open_positions.get(key)
        if not pos:
            return

        pnl = self.mcx_risk.record_exit(key, exit_price)
        decision_id = pos.get("_liq_decision_id")

        self.db.record_trade(
            symbol      = commodity,
            exchange    = "mcx_fo",
            action      = pos.get("action", "BUY"),
            strategy    = pos.get("strategy", "commodity_trend"),
            entry_price = pos["entry_price"],
            exit_price  = exit_price,
            quantity    = pos["quantity"],
            pnl         = pnl,
            exit_reason = reason,
            entry_time  = pos.get("entry_time", ""),
            exit_time   = datetime.now().isoformat(),
            order_no    = pos.get("order_no", ""),
        )

        if decision_id:
            outcome = "win" if pnl > 0 else "loss" if pnl < 0 else "breakeven"
            self.db.update_ai_decision_outcome(decision_id, outcome, pnl)

        sign = "✅" if pnl > 0 else "❌"
        logger.info(
            f"[LIQ] EXIT MCX {commodity} @ ₹{exit_price:.2f} "
            f"| P&L: ₹{pnl:+.2f} | Reason: {reason}"
        )
        self._persist_positions()

        try:
            msg = (
                f"[LIQ] {sign} EXIT MCX {commodity}\n"
                f"Price: ₹{exit_price:.2f} | P&L: ₹{pnl:+.2f}\n"
                f"Reason: {reason} | Day P&L: ₹{self.mcx_risk.daily_pnl:+.2f}"
            )
            self.telegram.send(msg)
        except Exception:
            pass

    def _squareoff_mcx(self):
        """MCX-specific EOD squareoff at 23:25 (5 min before close)."""
        mcx_positions = dict(self.mcx_risk.open_positions)
        if not mcx_positions:
            return
        logger.info(f"[LIQ] MCX EOD squareoff: {len(mcx_positions)} position(s)")
        for key, pos in mcx_positions.items():
            commodity = key.replace("MCX_", "")
            df = self._ohlcv.get(key)
            price = float(df["close"].iloc[-1]) if df is not None else pos["entry_price"]
            self._exit_mcx_position(key, commodity, price, "eod_squareoff")

    # ─── CDE stub [Phase 3] ──────────────────────────────────────────────────

    def _run_cde_cycle(self):
        """Currency derivatives cycle — Phase 3 placeholder."""
        logger.debug("LiqCDE: Phase 3 not yet implemented")

    # ─── EOD summary ─────────────────────────────────────────────────────────

    def job_market_close(self):
        """Run at 15:35. Squareoff remaining positions and send summary."""
        self._squareoff_all()
        stats = self.db.get_today_stats()
        logger.info(
            f"[LIQ] EOD — Trades: {stats['trades']} | "
            f"P&L: ₹{stats['total_pnl']:+.2f} | "
            f"Wins: {stats['wins']} | Losses: {stats['losses']}"
        )
        self.db.update_bot_status(status="closed")
        try:
            wr = f"{stats['wins']}/{stats['trades']}" if stats['trades'] else "0/0"
            msg = (
                f"[LIQ] 🌙 EOD Summary\n"
                f"Trades: {stats['trades']} | W/L: {wr}\n"
                f"Day P&L: ₹{stats['total_pnl']:+.2f}\n"
                f"Segments: {', '.join(self.segments)}"
            )
            self.telegram.send(msg)
        except Exception:
            pass

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def run(self):
        """Start the engine with scheduler. Blocks until SIGINT/SIGTERM."""
        self._running = True

        # Register signal handlers for clean shutdown
        def _shutdown(sig, frame):
            logger.info(f"[LIQ] Received signal {sig}. Shutting down...")
            self._running = False

        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # Schedule jobs
        schedule.every().day.at("08:45").do(self.job_daily_setup)
        schedule.every(5).minutes.do(self.run_trading_cycle)
        schedule.every().day.at("15:35").do(self.job_market_close)
        schedule.every().day.at("23:25").do(self._squareoff_mcx)  # MCX closes 23:30

        logger.info("[LIQ] Scheduler started. Engine running.")
        self.db.update_bot_status(status="running", segments_active=self.segments)

        # If dry-run, run one cycle immediately and exit
        if self.dry_run:
            logger.info("[LIQ] Dry-run mode — running one cycle then exiting")
            self.job_daily_setup()
            self.run_trading_cycle()
            self.db.update_bot_status(status="stopped")
            return

        # Restore any open positions from DB before running daily setup
        # (daily setup resets daily_pnl/trades but must NOT wipe positions)
        self._restore_positions()

        # Always run daily setup on startup (bootstrap data, reset counters).
        # Without this, a restart after 08:45 skips bootstrap entirely — OHLCV cache
        # is empty and strategies have no data to work with.
        logger.info("[LIQ] Running startup bootstrap (daily setup)...")
        self.job_daily_setup()

        while self._running:
            schedule.run_pending()
            time.sleep(1)

        logger.info("[LIQ] Engine stopped cleanly.")
        self.db.update_bot_status(status="stopped")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Liquidity Engine — parallel paper trading")
    parser.add_argument("--segments", default="bse_cm",
                        help="Comma-separated segments: bse_cm,mcx_fo,cde_fo")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run one cycle and exit (for testing outside market hours)")
    args = parser.parse_args()

    segments = [s.strip() for s in args.segments.split(",")]
    engine = LiquidityEngine(segments=segments, dry_run=args.dry_run)
    engine.run()


if __name__ == "__main__":
    main()
