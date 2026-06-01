"""
AlgoTrader — Main Orchestrator
Ties together: Auth → Data → Strategy → Risk → AI → Execution → Alerts → Dashboard

Run this file to start the bot:
    python main.py
"""

import os
import time
import logging
import threading
import schedule
from datetime import datetime, date
from dotenv import load_dotenv
from pathlib import Path

# Always load .env from the project root regardless of working directory
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ─── Logging setup ───────────────────────────────────────────────────────────
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"trader_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ─── Module imports ──────────────────────────────────────────────────────────
from core.kotak_client import KotakNeoClient
from core.risk_manager import RiskManager, RiskConfig, TradeSignal
from data.data_manager import DataManager
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.options import OptionsStrategy
from ai.decision_engine import AIDecisionEngine
from ai.signal_aggregator import SignalAggregator
from alerts.telegram_bot import TelegramAlerter
from dashboard.live_dashboard import LiveDashboard
from data.portfolio_analytics import PortfolioAnalytics
from core.ws_client import KotakNeoWSClient
from data.news_fetcher import NewsFetcher
from ai.tools import ToolExecutor
from ai.optimizer import StrategyOptimizer, load_best_params, MOMENTUM_DEFAULTS, MEAN_REV_DEFAULTS
from ai.feedback_loop import (
    record_ai_decision, update_ai_decision_outcome, link_trade_to_decision,
    FeedbackLoop,
)
from ai.ml_ensemble import predict_win_probability, train_model, reload_model
from data.calendar import get_calendar_context, get_calendar_size_multiplier
from strategies.multi_timeframe import MTFAnalyzer
from alerts.telegram_commands import build_command_handler
from core.kelly_sizer import compute_kelly_multiplier
from data.screener import Screener, NIFTY200_UNIVERSE
from core.circuit_breaker import CircuitBreaker, CBState

# ─── NSE Trading Holidays ─────────────────────────────────────────────────────
# Source: NSE India official holiday calendar. Update each December for the next year.
NSE_HOLIDAYS: set[date] = {
    # 2025
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 19),   # Chhatrapati Shivaji Maharaj Jayanti
    date(2025, 3, 14),   # Holi (Dhuliwashan)
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti / Ugadi
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 10, 2),   # Gandhi Jayanti
    date(2025, 10, 24),  # Dussehra (Vijaya Dashami)
    date(2025, 11, 5),   # Diwali (Laxmi Puja) — Muhurat session only
    date(2025, 11, 14),  # Gurunanak Jayanti
    date(2025, 12, 25),  # Christmas
    # 2026 — update when NSE publishes the official list
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 20),   # Holi (placeholder — confirm with NSE)
    date(2026, 4, 3),    # Good Friday (placeholder)
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 8, 15),   # Independence Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 12, 25),  # Christmas
}

# ─────────────────────────────────────────────────────────────────────────────
# Watchlist — edit this to your preferred stocks
# ─────────────────────────────────────────────────────────────────────────────
WATCHLIST = [
    # Large-cap stocks
    "RELIANCE-EQ", "TCS-EQ", "INFY-EQ", "HDFCBANK-EQ",
    "ICICIBANK-EQ", "WIPRO-EQ", "LT-EQ", "AXISBANK-EQ",
    # Added: high volume / diversification
    "SBIN-EQ",        # Highest volume on NSE, good ATR
    "BHARTIARTL-EQ",  # Telecom — low correlation to IT/banking
    "BAJFINANCE-EQ",  # NBFC — high ATR, different regime behaviour
    # Index ETFs
    "NIFTYBEES-EQ",   # Nifty 50
    "BANKBEES-EQ",    # Bank Nifty
    "HDFCSENSEX-EQ",  # Sensex
    "ITBEES-EQ",      # IT sector
]


class AlgoTrader:
    """Main bot orchestrator."""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("AlgoTrader Starting Up")
        logger.info("=" * 60)

        self.client = KotakNeoClient()
        self.risk   = RiskManager(RiskConfig())
        self.data   = DataManager(self.client)
        self.options_strategies = [
            OptionsStrategy(underlying="NIFTY"),
            OptionsStrategy(underlying="BANKNIFTY"),
            # SENSEX disabled: HDFCSENSEX ETF uses 1/1000 ratio, not 1/100.
            # ETF×100 gives ~8,451 vs real SENSEX ~82,000 → strikes 135 OTM.
            # Re-enable only when live SENSEX index quote is available.
            # OptionsStrategy(underlying="SENSEX"),
        ]

        # Load optimised params from DB (falls back to defaults if no optimisation has run)
        mom_params = load_best_params("momentum")     or MOMENTUM_DEFAULTS
        mr_params  = load_best_params("mean_reversion") or MEAN_REV_DEFAULTS
        self.momentum  = MomentumStrategy(**mom_params)
        self.mean_rev  = MeanReversionStrategy(**mr_params)
        logger.info(f"Momentum params (active): {mom_params}")
        logger.info(f"MeanReversion params (active): {mr_params}")

        self.aggregator = SignalAggregator({
            "momentum":       (self.momentum, 0.6),
            "mean_reversion": (self.mean_rev,  0.4),
        })
        self.ai         = AIDecisionEngine()
        self.news       = NewsFetcher()
        # Phase 3: give Claude live tool access for self-verified signal evaluation
        _tool_executor  = ToolExecutor(self.data, self.risk)
        self.ai.set_tool_executor(_tool_executor)
        # Phase 4: weekly hyperparameter optimiser
        self.optimizer  = StrategyOptimizer(self.data, self.ai)
        self.analytics  = PortfolioAnalytics(self.data)
        self.mtf        = MTFAnalyzer(self.data)       # Phase 9
        self.telegram   = TelegramAlerter()
        self.dashboard  = LiveDashboard(self.risk, self.client, self.data)
        self.ws         = KotakNeoWSClient(self.client, self.data)
        self.running         = False
        self._cycle_lock     = threading.Lock()  # prevent overlapping trading cycles
        self._market_regime  = "unknown"
        self._regime_data    = {}       # Full AI regime dict (regime + confidence + suggestion)
        self.trading_paused  = False    # Set True by morning brief if conditions are extreme
        self._calendar_ctx   = {}       # Refreshed once per trading cycle
        self._session_vetoes: set[str] = set()   # Phase 10: operator-blocked symbols
        self._cmd_handler    = None     # Phase 10: Telegram command handler (started in startup)
        # ── Session-level stats (reset on bot start, used by /status and EOD report) ──
        import datetime as _dt
        self._session_start   = _dt.datetime.now()
        self._first_scan_time = None       # datetime of first trading cycle
        self._last_scan_time  = None       # datetime of last trading cycle
        self._scan_cycles     = 0          # total scan cycles run
        self._signals_today   = 0          # signals generated (passed aggregator)
        self._ai_approved     = 0          # AI approved
        self._ai_rejected     = 0          # AI rejected (veto)
        self._veto_outcomes   = []         # [{symbol, action, entry_price, good_veto}] assessed post-session
        # Phase 12: active watchlist — starts as the hardcoded list, updated daily by screener
        self.active_watchlist: list[str] = list(WATCHLIST)
        self.screener = Screener(universe=NIFTY200_UNIVERSE)
        self.cb = CircuitBreaker()   # Phase 13: intraday circuit breaker

    # ─────────────────────────────────────────────────────────────────────────
    # Startup
    # ─────────────────────────────────────────────────────────────────────────

    def startup(self) -> bool:
        """Authenticate, download scrip master, run startup checks."""
        logger.info("Running startup sequence...")

        if not self.client.authenticate():
            logger.error("Authentication failed. Cannot start.")
            self.telegram.alert_error("Authentication failed", "startup")
            return False

        if not self.data.is_scrip_master_fresh():
            logger.info("Downloading scrip master...")
            self.data.download_scrip_master()

        # Bootstrap OHLCV on every startup so strategies have warm data immediately
        logger.info("Bootstrapping 5m OHLCV from yfinance (60-day intraday limit)...")
        candles = self.data.bootstrap_ohlcv(self.active_watchlist, period="60d", interval="5m")
        logger.info(f"5m OHLCV bootstrap: {candles:,} candles ready")

        # Daily OHLCV — fetch up to 1 year; skips if data is already fresh (< 1 day old)
        with self.data._get_conn() as _c:
            _daily_fresh = _c.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE exchange='nse_cm_daily'"
            ).fetchone()[0]
        if _daily_fresh < len(self.active_watchlist) * 200:
            logger.info("Bootstrapping 1-year daily OHLCV (first-run or stale)...")
            daily_rows = self.data.bootstrap_daily_ohlcv(self.active_watchlist, years=1)
            logger.info(f"Daily OHLCV bootstrap: {daily_rows:,} daily rows ready")
        else:
            logger.info(f"Daily OHLCV already present ({_daily_fresh:,} rows) — skipping")

        stats   = self.data.get_db_stats()
        session = self.client.session_info()
        logger.info(f"DB: {stats}")
        logger.info(f"Session: {session}")

        # Start WebSocket streaming for real-time ticks
        self.ws.subscribe_watchlist(self.active_watchlist)
        self.ws.start()
        logger.info(f"WebSocket: {self.ws.status()}")

        # Phase 10: start Telegram command handler (two-way operator control)
        self._cmd_handler = build_command_handler(self)
        if self._cmd_handler:
            self._cmd_handler.start()
            logger.info("Telegram command handler ready — send /help in chat")

        # Mark bot as running in DB — sleep 3s first so any prior process shutdown
        # (which writes "stopped") completes before we overwrite with "running"
        import threading as _threading
        def _mark_running():
            import time as _t; _t.sleep(3)
            try:
                portfolio = self.risk.get_portfolio_summary()
                self.data.update_bot_status(portfolio, paper_trading=self.client.paper_trading, status="running")
            except Exception:
                pass
        _threading.Thread(target=_mark_running, daemon=True).start()

        # Send restart notification so operator knows bot is live
        from datetime import datetime as _dt
        _mode = "PAPER" if self.client.paper_trading else "LIVE"
        _now  = _dt.now().strftime("%H:%M")
        self.telegram.send(
            f"🔄 *AlgoTrader restarted* — {_mode} mode\n"
            f"📋 Watchlist: {len(self.active_watchlist)} symbols\n"
            f"🕐 {_now} IST — scanning every 5 min"
        )
        # Market-open alert is sent by job_daily_setup at 8:45 AM, not at startup
        logger.info("Startup complete. Bot is ready.")
        return True

    def _build_rich_portfolio_context(self, signal_symbol: str) -> dict:
        """
        Compute a full portfolio health snapshot for Claude's evaluation prompt.
        Includes: sector allocation, correlation pairs, win streak, loss budget.

        This runs BEFORE the AI call so Claude has the full picture in its first
        message — no tool call needed to discover "you already have 2 IT stocks".
        """
        portfolio  = self.risk.get_portfolio_summary()
        positions  = portfolio.get("open_positions", {})
        daily_pnl  = portfolio.get("daily_pnl", 0)
        daily_trades = portfolio.get("daily_trades", 0)

        # Sector allocation of current positions
        sector_alloc   = self.analytics.sector_allocation(positions) if positions else {}
        conc_warnings  = self.analytics.concentration_warnings(positions) if positions else []

        # Sector map for rule-based check: {symbol: sector}
        from data.portfolio_analytics import SECTOR_MAP
        sector_map = {
            sym: SECTOR_MAP.get(sym.split("-")[0], "Other")
            for sym in positions
        }

        # Sector of the incoming signal
        new_base   = signal_symbol.split("-")[0]
        new_sector = SECTOR_MAP.get(new_base, "Other")
        same_sector_held = [s for s, sec in sector_map.items() if sec == new_sector]

        # Correlation among held symbols (only when ≥2 positions)
        held_symbols = list(positions.keys())
        corr_pairs   = []
        if len(held_symbols) >= 2:
            try:
                corr_pairs = self.analytics.high_correlation_pairs(held_symbols, threshold=0.65)
            except Exception:
                corr_pairs = []

        # Recent performance: last 10 completed trades
        try:
            trades_df   = self.data.get_trades(limit=10)
            recent_pnls = trades_df["pnl"].tolist() if not trades_df.empty else []
            win_streak  = 0
            lose_streak = 0
            for p in reversed(recent_pnls):
                if p > 0:
                    if lose_streak > 0:
                        break
                    win_streak += 1
                else:
                    if win_streak > 0:
                        break
                    lose_streak += 1
            recent_win_rate = (
                sum(1 for p in recent_pnls if p > 0) / len(recent_pnls)
                if recent_pnls else 0.5
            )
        except Exception:
            recent_pnls, win_streak, lose_streak, recent_win_rate = [], 0, 0, 0.5

        # Daily loss budget
        max_loss      = self.risk.config.max_capital * (self.risk.config.max_daily_loss_pct / 100)
        loss_used_pct = abs(min(daily_pnl, 0)) / max_loss * 100 if max_loss > 0 else 0

        return {
            "open_position_count": len(positions),
            "open_symbols":        list(positions.keys()),
            "daily_pnl":           daily_pnl,
            "daily_trades":        daily_trades,
            "loss_budget_used_pct": round(loss_used_pct, 1),
            "loss_budget_remaining": round(max_loss - abs(min(daily_pnl, 0)), 2),
            "sector_allocation":   sector_alloc,
            "sector_map":          sector_map,
            "concentration_warnings": conc_warnings,
            "incoming_signal_sector": new_sector,
            "same_sector_already_held": same_sector_held,
            "high_correlation_pairs": [
                {"a": a, "b": b, "corr": c} for a, b, c in corr_pairs
            ],
            "recent_win_rate":    round(recent_win_rate, 2),
            "current_win_streak": win_streak,
            "current_lose_streak": lose_streak,
        }

    def _build_ohlcv_from_ticks(self):
        """Convert accumulated WebSocket ticks into 5-min OHLCV candles for all watchlist symbols."""
        built = 0
        for symbol in self.active_watchlist:
            try:
                instrument = self.data.get_instrument(symbol, "nse_cm")
                exchange = instrument["exchange"] if instrument else "nse_cm"
                self.data.build_candle_from_ticks(symbol, exchange, interval_minutes=5)
                built += 1
            except Exception as e:
                logger.debug(f"Candle build skipped for {symbol}: {e}")
        if built:
            logger.debug(f"Built/refreshed OHLCV candles for {built} symbols from live ticks")

    # ─────────────────────────────────────────────────────────────────────────
    # Core trading cycle
    # ─────────────────────────────────────────────────────────────────────────

    def run_trading_cycle(self):
        """
        Main trading loop — runs every 5 minutes during market hours.
        1. Monitor existing positions (SL / target / trailing stop)
        2. Scan watchlist with BOTH strategies
        3. AI-evaluate top signals
        4. Execute approved trades
        """
        if not self._cycle_lock.acquire(blocking=False):
            logger.debug("Trading cycle already running — skipping overlapping cycle")
            return

        try:
            self._run_trading_cycle_inner()
        finally:
            self._cycle_lock.release()

    def _run_trading_cycle_inner(self):
        if not self.is_market_open():
            return

        if self.trading_paused:
            logger.warning("Trading paused by morning AI brief — skipping cycle")
            return

        logger.info("─── Trading cycle started ───")

        try:
            # Ensure session is still valid — auto-retry auth up to 3 times
            session_ok = False
            for _attempt in range(3):
                try:
                    self.client._ensure_session()
                    session_ok = True
                    break
                except RuntimeError:
                    try:
                        logger.warning(f"Session invalid — re-authenticating (attempt {_attempt+1}/3)")
                        self.client.authenticate()
                        session_ok = True
                        break
                    except Exception as auth_err:
                        logger.error(f"Re-auth attempt {_attempt+1} failed: {auth_err}")
            if not session_ok:
                logger.error("Session could not be recovered after 3 attempts — skipping cycle")
                self.telegram.alert_error("Session recovery failed — skipping trading cycle", "session")
                return

            # Phase 8: refresh calendar context once per cycle (cheap — pure computation)
            self._calendar_ctx = get_calendar_context()
            if self._calendar_ctx.get("trading_notes"):
                for note in self._calendar_ctx["trading_notes"]:
                    logger.info(f"Calendar: {note}")

            # Hard pause on expiry day after 14:00 (gamma risk too high for new entries)
            now = datetime.now()
            if self._calendar_ctx.get("expiry_day") and now.hour >= 14:
                logger.warning("Expiry day after 14:00 — skipping new entries")
                return

            # Build fresh OHLCV candles from accumulated WebSocket ticks
            self._build_ohlcv_from_ticks()
            # Also save REST quotes as candles (fallback when WS is not connected)
            if not self.ws.status().get("connected", False):
                self.data.store_quotes_as_candles(self.active_watchlist)
            import datetime as _dt
            _now = _dt.datetime.now()
            self._last_scan_time  = _now
            self._scan_cycles    += 1
            if self._first_scan_time is None:
                self._first_scan_time = _now

            self._monitor_positions()
            self._monitor_fo_positions()

            # Phase 13: circuit breaker — evaluate intraday drawdown
            portfolio   = self.risk.get_portfolio_summary()
            positions   = portfolio.get("open_positions", {})
            realised_pnl = portfolio.get("daily_pnl", 0)

            # Enrich positions with live prices for unrealised P&L calc
            for sym, pos in positions.items():
                q = self.data.get_live_quote(sym, pos.get("exchange", "nse_cm"))
                if q:
                    pos["current_price"] = q["ltp"]

            cb_status = self.cb.check(realised_pnl, positions)
            prev_cb_state = getattr(self, "_prev_cb_state", CBState.NORMAL)

            if cb_status.state != prev_cb_state:
                self._prev_cb_state = cb_status.state
                emoji = {"caution": "🟡", "halt": "🔴", "close": "🆘", "normal": "🟢"}.get(
                    cb_status.state.value, "⚪"
                )
                self.telegram.send(
                    f"{emoji} <b>Circuit Breaker: {cb_status.state.value.upper()}</b>\n"
                    f"{cb_status.message}"
                )
                self.data.log_event(
                    "CIRCUIT_BREAKER",
                    cb_status.message,
                    metadata={"state": cb_status.state.value,
                              "effective_loss": cb_status.effective_loss},
                )

            if cb_status.force_close:
                logger.warning("Circuit breaker CLOSE — force-closing all positions")
                self.trading_paused = True
                for sym, pos in list(positions.items()):
                    try:
                        q = self.data.get_live_quote(sym, pos.get("exchange", "nse_cm"))
                        price = q["ltp"] if q else pos["entry_price"]
                        self._exit_position(sym, pos, price, "circuit_breaker")
                    except Exception as e:
                        logger.error(f"Force close failed for {sym}: {e}")
                return   # no new signals after force close

            if not cb_status.allow_new_entries:
                logger.info(f"Circuit breaker {cb_status.state.value.upper()} — no new entries")
                return

            # Detect market regime — local technical analysis as base,
            # then upgrade with AI reasoning if ANTHROPIC_API_KEY is set.
            local_regime = self.data.detect_regime("NIFTYBEES-EQ")
            try:
                nifty_quote = self.data.get_live_quote("NIFTYBEES-EQ") or {}
                # Enrich quote with OHLCV-derived context so AI has real data
                _ohlcv = self.data.get_ohlcv("NIFTYBEES-EQ", "nse_cm", limit=20)
                if len(_ohlcv) >= 2:
                    _cur  = float(_ohlcv["close"].iloc[-1])
                    _prev = float(_ohlcv["close"].iloc[0])
                    nifty_quote["change_pct"] = round((_cur - _prev) / _prev * 100, 2)
                    nifty_quote["ltp"]        = _cur
                    nifty_quote["local_regime"] = local_regime
                ai_regime   = self.ai.detect_market_regime(nifty_quote)
                # AI result is authoritative only when confidence > local (0.6+).
                # If AI says ranging at exactly 50% but local says trending, trust local.
                if ai_regime.get("confidence", 0) > 0.6:
                    self._market_regime = ai_regime["regime"]
                    self._regime_data   = ai_regime
                else:
                    self._market_regime = local_regime
                    self._regime_data   = {"regime": local_regime, "confidence": 0.65,
                                           "suggestion": "Local technical regime (AI low-confidence)",
                                           "risk_level": "medium"}
            except Exception as e:
                logger.warning(f"AI regime detection failed, using local: {e}")
                self._market_regime = local_regime
                self._regime_data   = {"regime": local_regime, "confidence": 0.65,
                                       "suggestion": "AI unavailable — local regime used",
                                       "risk_level": "medium"}
            logger.info(
                f"Market regime: {self._market_regime}"
                f" (conf={self._regime_data.get('confidence', 0):.0%},"
                f" risk={self._regime_data.get('risk_level', '?')})"
            )

            # Update bot status in DB (includes regime from above — dashboard sees it immediately)
            portfolio = self.risk.get_portfolio_summary()
            try:
                self.data.update_bot_status(
                    portfolio,
                    paper_trading=self.client.paper_trading,
                    status="running",
                    regime_data=self._regime_data,
                )
            except Exception as e:
                logger.error(f"Failed to update bot status: {e}")
                # Continue anyway, not critical

            # Equity signals via aggregator (regime-weighted)
            try:
                signals = self.aggregator.scan_and_aggregate(
                    self.active_watchlist, self.data, regime=self._market_regime
                )
            except Exception as e:
                logger.error(f"Signal aggregation error: {e}")
                self.telegram.alert_error(str(e), "signal_aggregation")
                signals = []

            # F&O options — enter when regime is trending (local or AI-detected)
            _local_regime = self.data.detect_regime("NIFTYBEES-EQ")
            logger.info(f"Regime check — AI: {self._market_regime} | Local: {_local_regime}")
            _run_options = (self._market_regime in ("trending_up", "trending_down") or
                            _local_regime in ("trending_up", "trending_down"))
            if _run_options:
                for opt_strat in self.options_strategies:
                    try:
                        opt_signal = opt_strat.generate_signal(self.data)
                        if opt_signal:
                            signals.append(opt_signal)
                    except Exception as e:
                        logger.warning(f"Options signal error [{opt_strat.underlying}]: {e}")

            if not signals:
                logger.info("No signals this cycle")
                from datetime import datetime as _dt
                self.data.log_event(
                    "CYCLE",
                    f"Scanned 15 symbols | {self._market_regime} | 0 signals",
                )
                return

            self._signals_today += len(signals[:2])
            for signal in signals[:2]:
                try:
                    self._process_signal(signal)
                except Exception as e:
                    logger.error(f"Error processing signal for {signal.symbol}: {e}", exc_info=True)
                    self.telegram.alert_error(str(e), f"process_signal_{signal.symbol}")

        except Exception as e:
            logger.error(f"Trading cycle error: {e}", exc_info=True)
            self.telegram.alert_error(str(e), "trading_cycle")

    def _process_signal(self, signal: TradeSignal):
        """Full pipeline: risk check → VIX gate → AI eval (with news) → execution."""
        # Phase 10: operator session veto takes priority over everything
        if signal.symbol in self._session_vetoes:
            logger.info(f"Session veto: skipping {signal.symbol}")
            return

        available = self.client.get_available_capital()

        # ── VIX gate ─────────────────────────────────────────────────────────
        vix = self.news.get_india_vix()   # cached, fast after first call
        if vix is not None:
            blocked, reason = self.risk.check_vix_block(vix, signal.strategy)
            if blocked:
                logger.info(f"VIX blocked {signal.symbol}: {reason}")
                self.data.log_event("VETO", f"VIX: {reason}", symbol=signal.symbol)
                return

        # Phase 8: calendar-aware sizing — scale down near expiry / event days
        cal_multiplier = get_calendar_size_multiplier(self._calendar_ctx)
        # Phase 13: circuit breaker size multiplier (0.5 in CAUTION, 0.0 in HALT/CLOSE)
        _portfolio_snap = self.risk.get_portfolio_summary()
        cb_multiplier  = getattr(self.cb.check(
            _portfolio_snap.get("daily_pnl", 0),
            _portfolio_snap.get("open_positions", {}),
        ), "size_multiplier", 1.0)
        cal_available  = available * cal_multiplier * cb_multiplier
        if cb_multiplier < 1.0:
            logger.info(f"CB size reduction: {cb_multiplier:.0%} ({self.cb.state.value})")
        if cal_multiplier < 1.0:
            logger.info(
                f"Calendar size reduction: {cal_multiplier:.0%} "
                f"(risk_level={self._calendar_ctx.get('risk_level', '?')})"
            )

        # Phase 11: Kelly Criterion multiplier — uses ML win_prob if available
        # (ml_pred populated below, but we call Kelly here with None and update after ML)
        kelly = compute_kelly_multiplier(win_prob=None, strategy=signal.strategy)
        kelly_multiplier = kelly["multiplier"]

        if signal.product == "NRML":
            # F&O options: keep the lot_size set by the strategy; do not override with
            # capital-based sizing (options must trade in fixed lot multiples).
            # Quantity was already set to lot_size in OptionsStrategy.generate_signal().
            pass
        else:
            signal.quantity = self.risk.calculate_quantity(
                signal.price, cal_available, signal.confidence,
                vix=vix, kelly_multiplier=kelly_multiplier,
            )

        if not signal.stop_loss:
            signal.stop_loss = self.risk.calculate_stop_loss(signal.price, signal.action)
        if not signal.target:
            signal.target = self.risk.calculate_target(signal.price, signal.action)

        # Log signal generation
        self.data.log_event(
            "SIGNAL_BUY" if signal.action == "BUY" else "SIGNAL_SELL",
            f"{signal.symbol} {signal.strategy} conf={signal.confidence:.0%}",
            symbol=signal.symbol,
            metadata={
                "strategy": signal.strategy,
                "confidence": signal.confidence,
                "price": signal.price,
            }
        )

        # ── Phase 5: Rich portfolio context ─────────────────────────────────
        # Compute sector map before risk approval so the sector check has data.
        rich_ctx = self._build_rich_portfolio_context(signal.symbol)
        sector_map = rich_ctx.get("sector_map", {})

        signal.regime = self._market_regime  # pass regime for adaptive R:R check
        approved, reason = self.risk.approve_trade(signal, available, sector_map=sector_map)
        if not approved:
            logger.info(f"Risk rejected {signal.symbol}: {reason}")
            self.dashboard.log_event(f"Risk REJECT {signal.symbol}: {reason}")
            self.data.log_event("VETO", f"Risk: {reason}", symbol=signal.symbol)
            self.telegram.send(
                f"🛡 <b>Risk Rejected — {signal.symbol.replace('-EQ','')}</b>\n"
                f"{'─'*28}\n"
                f"📊 Strategy:   {signal.strategy}\n"
                f"{'BUY 🟢' if signal.action == 'BUY' else 'SELL 🔴'}  ₹{signal.price:,.2f}  conf={signal.confidence:.0%}\n"
                f"❌ Reason: {reason}\n"
                f"⏰ {__import__('datetime').datetime.now().strftime('%H:%M:%S')}"
            )
            return

        # Log portfolio health at decision time
        if rich_ctx.get("concentration_warnings"):
            for w in rich_ctx["concentration_warnings"]:
                logger.warning(f"Portfolio concentration: {w}")
        if rich_ctx.get("high_correlation_pairs"):
            for p in rich_ctx["high_correlation_pairs"]:
                logger.info(f"Correlated pair: {p['a']} ↔ {p['b']} = {p['corr']:.2f}")

        portfolio = self.risk.get_portfolio_summary()

        # ── Phase 9: Multi-timeframe confirmation ────────────────────────────
        mtf_result = self.mtf.analyze(signal.symbol, signal.action, signal.exchange)
        if mtf_result is not None:
            if mtf_result.veto:
                logger.info(f"MTF vetoed {signal.symbol}: {mtf_result.veto_reason}")
                self.data.log_event(
                    "VETO", f"MTF: {mtf_result.veto_reason[:80]}",
                    symbol=signal.symbol,
                    metadata={"mtf": mtf_result.to_dict()},
                )
                return
            # Apply confidence adjustment from MTF alignment
            signal.confidence = max(0.0, min(1.0,
                signal.confidence + mtf_result.confidence_adjustment
            ))
            logger.info(f"MTF adj {signal.symbol}: {mtf_result.confidence_adjustment:+.2f} → conf={signal.confidence:.0%}")

        # ── News sentiment (Phase 2) ─────────────────────────────────────────
        news_sentiment = None
        try:
            announcements = self.news.get_announcements(signal.symbol)
            if announcements:
                news_sentiment = self.ai.score_news_sentiment(signal.symbol, announcements)
                sentiment_label = news_sentiment.get("label", "neutral")
                sentiment_score = news_sentiment.get("score", 0)
                logger.info(
                    f"News sentiment [{signal.symbol}]: {sentiment_label} "
                    f"({sentiment_score:+.2f}) — {news_sentiment.get('reason', '')}"
                )
                if signal.action == "BUY" and sentiment_score < -0.6:
                    logger.info(f"News sentiment blocked BUY {signal.symbol}: {news_sentiment.get('reason')}")
                    self.data.log_event(
                        "VETO",
                        f"News: {news_sentiment.get('reason', 'Negative sentiment')}",
                        symbol=signal.symbol,
                    )
                    self.telegram.send(
                        f"📰 News blocked BUY {signal.symbol} "
                        f"(score={sentiment_score:+.2f}): {news_sentiment.get('reason', '')[:120]}"
                    )
                    return
        except Exception as e:
            logger.debug(f"News sentiment fetch failed (non-critical): {e}")

        # Market context includes VIX, FII/DII, regime, and full portfolio snapshot
        flows = self.news.get_fii_dii_flows()
        market_context = {
            "regime":          self._market_regime,
            "vix":             vix,
            "fii_net_cr":      flows.get("fii_net"),
            "dii_net_cr":      flows.get("dii_net"),
            "ai_suggestion":   self._regime_data.get("suggestion", ""),
            "portfolio":       rich_ctx,   # full portfolio health for Claude
            "mtf":             mtf_result.to_dict() if mtf_result else None,
            "calendar": {
                "days_to_expiry":      self._calendar_ctx.get("days_to_expiry"),
                "expiry_day":          self._calendar_ctx.get("expiry_day"),
                "expiry_week":         self._calendar_ctx.get("expiry_week"),
                "monthly_expiry_week": self._calendar_ctx.get("monthly_expiry_week"),
                "next_expiry":         self._calendar_ctx.get("next_expiry", {}).get("label"),
                "risk_level":          self._calendar_ctx.get("risk_level"),
                "trading_notes":       self._calendar_ctx.get("trading_notes", []),
                "economic_events_14d": self._calendar_ctx.get("economic_events_14d", []),
                "size_multiplier":     cal_multiplier,
            },
        }

        # Phase 7: ML ensemble — predict win probability from historical patterns
        ml_pred = predict_win_probability(
            signal_conf=signal.confidence,
            action=signal.action,
            strategy=signal.strategy,
            vix=vix,
            regime=self._market_regime,
        )
        if ml_pred is not None:
            market_context["ml_prediction"] = ml_pred
            logger.info(
                f"ML [{signal.symbol}]: win_prob={ml_pred['win_prob']:.0%} "
                f"({ml_pred['ml_confidence']}) → {ml_pred['recommendation']}"
            )
            # Hard skip when ML is highly confident this will lose
            if ml_pred["recommendation"] == "skip" and ml_pred["ml_confidence"] == "high":
                logger.info(
                    f"ML blocked {signal.symbol}: "
                    f"win_prob={ml_pred['win_prob']:.0%} (high confidence loss)"
                )
                self.data.log_event(
                    "VETO",
                    f"ML: win_prob={ml_pred['win_prob']:.0%} — high-confidence loss predicted",
                    symbol=signal.symbol,
                    metadata={"ml_prediction": ml_pred},
                )
                return

            # Phase 11: recompute Kelly with ML win_prob now available → update quantity
            kelly = compute_kelly_multiplier(
                win_prob=ml_pred["win_prob"], strategy=signal.strategy
            )
            market_context["kelly"] = kelly
            if kelly["source"] != "fallback" and signal.product != "NRML":
                signal.quantity = self.risk.calculate_quantity(
                    signal.price, cal_available, signal.confidence,
                    vix=vix, kelly_multiplier=kelly["multiplier"],
                )
                logger.info(
                    f"Kelly sizing [{signal.symbol}]: "
                    f"mult={kelly['multiplier']:.2f} "
                    f"(raw={kelly['kelly_raw']:.2f}, p={kelly['win_prob']:.0%}, "
                    f"b={kelly['payoff']:.2f}, n={kelly['n_trades']}) "
                    f"→ qty={signal.quantity}"
                )

        ai_result = self.ai.evaluate_signal(signal, portfolio, market_context, news_sentiment)

        # Phase 6: record this decision immediately so outcome can be linked later
        decision_id = record_ai_decision(
            signal.symbol, signal.action, signal.strategy,
            signal.confidence, signal.price, ai_result,
        )
        signal._decision_id = decision_id   # carried into _execute_trade → position dict

        tools_used = ai_result.get("tools_used", [])
        if tools_used:
            logger.info(f"AI used tools for {signal.symbol}: {tools_used}")

        if not ai_result.get("approved", True):
            self._ai_rejected += 1
            note     = ai_result.get("reasoning", "")
            concerns = ai_result.get("concerns", [])
            _live_quote = self.data.get_live_quote(signal.symbol,
                                                    getattr(signal, "exchange", "nse_cm"))
            _ltp = _live_quote.get("ltp") if _live_quote else None
            logger.info(f"AI rejected {signal.symbol}: {note}")
            self.telegram.alert_ai_veto(signal, note, concerns, tools_used, _ltp)
            self.dashboard.log_event(f"AI VETO {signal.symbol}: {note[:60]}")
            self.data.log_event("VETO", f"AI: {note[:60]}", symbol=signal.symbol,
                                metadata={"tools_used": tools_used, "reasoning": note[:200]})

            # Structured AI decision record
            self.data.record_ai_decision({
                "symbol":            signal.symbol,
                "strategy":          getattr(signal, "strategy", ""),
                "action":            signal.action,
                "signal_confidence": signal.confidence,
                "final_confidence":  signal.confidence,
                "approved":          False,
                "veto_reason":       note[:500],
                "ai_reasoning":      note,
                "tools_used":        tools_used,
                "signal_price":      signal.price,
                "stop_loss":         signal.stop_loss,
                "target":            signal.target,
                "market_regime":     getattr(self, "_market_regime", "unknown"),
            })

            # Record veto for accuracy tracking — assessed at EOD
            self._veto_outcomes.append({
                "symbol":      signal.symbol,
                "action":      signal.action,
                "entry_price": signal.price,
                "stop_loss":   signal.stop_loss,
                "target":      signal.target,
                "vetoed_at":   _ltp or signal.price,
                "good_veto":   None,
            })
            return

        self._ai_approved += 1

        # Notify operator that AI approved — trade is about to be placed
        _reasoning_short = ai_result.get("reasoning", "")[:120]
        self.telegram.send(
            f"✅ <b>AI Approved — {signal.symbol.replace('-EQ','')}</b>\n"
            f"{'─'*28}\n"
            f"📊 Strategy:   {signal.strategy}\n"
            f"{'BUY 🟢' if signal.action == 'BUY' else 'SELL 🔴'}  ₹{signal.price:,.2f}  conf={signal.confidence:.0%}\n"
            f"🛡 SL: ₹{signal.stop_loss:,.2f}  🎯 Target: ₹{signal.target:,.2f}\n"
            f"💬 {_reasoning_short}\n"
            f"⏰ {__import__('datetime').datetime.now().strftime('%H:%M:%S')}"
        )

        orig_confidence = signal.confidence
        signal.confidence = max(0.0, min(1.0,
            signal.confidence + ai_result.get("confidence_adjustment", 0)
        ))
        if ai_result.get("suggested_stop_loss"):
            signal.stop_loss = ai_result["suggested_stop_loss"]
        if ai_result.get("suggested_target"):
            signal.target = ai_result["suggested_target"]

        reasoning  = ai_result.get("reasoning", "")
        tools_used = ai_result.get("tools_used", [])

        # Structured AI decision record — store row id so we can update outcome at exit
        decision_id = self.data.record_ai_decision({
            "symbol":            signal.symbol,
            "strategy":          getattr(signal, "strategy", ""),
            "action":            signal.action,
            "signal_confidence": orig_confidence,
            "final_confidence":  signal.confidence,
            "approved":          True,
            "veto_reason":       None,
            "ai_reasoning":      reasoning,
            "tools_used":        tools_used,
            "signal_price":      signal.price,
            "stop_loss":         signal.stop_loss,
            "target":            signal.target,
            "market_regime":     getattr(self, "_market_regime", "unknown"),
        })
        # Attach decision_id to signal so _execute_trade can store it in the position
        signal._decision_id = decision_id

        tools_note = f"\n🔧 Verified via: {', '.join(tools_used)}" if tools_used else ""
        self.telegram.alert_trade_signal(signal, reasoning + tools_note)
        self.dashboard.log_signal(signal, reasoning[:60])
        self._execute_trade(signal)

    def _execute_trade(self, signal: TradeSignal):
        """Place the order via Kotak Neo API."""
        try:
            result = self.client.place_order(
                symbol=signal.symbol,
                exchange=signal.exchange,
                transaction_type=signal.action[0],   # "B" or "S"
                quantity=signal.quantity,
                price=signal.price,
                order_type="L",
                product=signal.product,
                trigger_price=0,
            )

            order_no = result.get("nOrdNo", "UNKNOWN")
            stat     = result.get("stat", "Unknown")

            if stat == "Ok" or result.get("paper"):
                logger.info(f"Order placed: {signal.symbol} | Order No: {order_no}")
                self.risk.record_entry(signal, order_no)
                # Attach decision_id to the position so _exit_position can link outcome
                pos = self.risk.open_positions.get(signal.symbol, {})
                if pos and hasattr(signal, "_decision_id"):
                    pos["_decision_id"] = signal._decision_id
                if signal.product == "NRML":
                    # F&O entry — send rich options alert
                    import re as _re
                    _m = _re.match(r'([A-Z]+)(\d{2}[A-Z]{3}\d{2,4})(\d+)(CE|PE)', signal.symbol)
                    _underlying  = _m.group(1) if _m else signal.symbol
                    _strike      = int(_m.group(3)) if _m else 0
                    _option_type = _m.group(4) if _m else ""
                    _days_left   = self._fo_days_to_expiry(signal.symbol)
                    self.telegram.alert_fo_trade(
                        symbol=signal.symbol,
                        action=signal.action,
                        qty=signal.quantity,
                        premium=signal.price,
                        strike=_strike,
                        expiry=None,
                        option_type=_option_type,
                        underlying=_underlying,
                        days_to_expiry=_days_left,
                        paper=self.client.paper_trading,
                    )
                else:
                    self.telegram.alert_order_placed(
                        signal.symbol, signal.action,
                        signal.quantity, signal.price, order_no
                    )
                self.dashboard.log_event(
                    f"ORDER {signal.action} {signal.quantity}×{signal.symbol}"
                    f" @ ₹{signal.price:,.2f}  #{order_no}"
                )
                self.data.log_event(
                    "EXECUTION",
                    f"{signal.action} {signal.quantity}×{signal.symbol} @ ₹{signal.price:.2f}",
                    symbol=signal.symbol,
                    metadata={
                        "action": signal.action,
                        "quantity": signal.quantity,
                        "price": signal.price,
                        "order_no": order_no,
                    }
                )
            else:
                logger.error(f"Order failed: {result}")
                self.telegram.alert_error(str(result), f"place_order_{signal.symbol}")

        except Exception as e:
            logger.error(f"Order execution error: {e}")
            self.telegram.alert_error(str(e), f"execute_{signal.symbol}")

    def _monitor_positions(self):
        """
        Check all open positions for:
        - Stop loss hit
        - Target hit
        - Trailing stop update
        """
        portfolio = self.risk.get_portfolio_summary()
        positions = portfolio.get("open_positions", {})

        for symbol, pos in list(positions.items()):
            quote = self.data.get_live_quote(symbol, pos.get("exchange", "nse_cm"))
            if not quote:
                continue

            current_price = quote["ltp"]

            # Update trailing stop before checking for exit
            if self.risk.update_trailing_stop(symbol, current_price):
                new_sl = self.risk.get_trailing_stop(symbol)
                self.dashboard.log_event(
                    f"Trail SL: {symbol} → ₹{new_sl:,.2f} (price ₹{current_price:,.2f})"
                )
                self.data.log_event(
                    "TRAILING_SL",
                    f"{symbol} SL moved to ₹{new_sl:.2f}",
                    symbol=symbol,
                    metadata={"new_sl": new_sl, "current_price": current_price}
                )

            if self.risk.check_stop_loss_hit(symbol, current_price):
                logger.info(f"STOP LOSS HIT: {symbol} @ ₹{current_price}")
                self._exit_position(symbol, pos, current_price, "stop_loss")

            elif self.risk.check_target_hit(symbol, current_price):
                logger.info(f"TARGET HIT: {symbol} @ ₹{current_price}")
                self._exit_position(symbol, pos, current_price, "target")

    # ─────────────────────────────────────────────────────────────────────────
    # F&O position monitoring
    # ─────────────────────────────────────────────────────────────────────────

    def _monitor_fo_positions(self):
        """
        Check F&O (NRML) positions for:
        - 50% profit exit
        - 30% loss stop
        - 1 day to expiry exit
        """
        fo_pos = {k: v for k, v in self.risk.open_positions.items()
                  if v.get("product") == "NRML"}
        for symbol, pos in list(fo_pos.items()):
            quote = self.data.get_live_quote(symbol, exchange="nse_fo")
            if not quote:
                continue
            current = quote.get("ltp", 0)
            entry   = pos["entry_price"]
            if entry <= 0:
                continue
            pnl_pct = (current - entry) / entry

            if pnl_pct >= 0.50:
                logger.info(f"F&O TARGET 50%: {symbol} @ ₹{current}")
                self._exit_fo_position(symbol, pos, current, "target_50pct")
            elif pnl_pct <= -0.30:
                logger.info(f"F&O STOP 30%: {symbol} @ ₹{current}")
                self._exit_fo_position(symbol, pos, current, "sl_30pct")
            elif self._fo_days_to_expiry(symbol) <= 1:
                logger.info(f"F&O EXPIRY EXIT (1 day left): {symbol}")
                self._exit_fo_position(symbol, pos, current, "expiry_exit")

    def get_session_stats(self) -> dict:
        """
        Full session statistics for /status and EOD report.
        Covers: uptime, WebSocket, scan timing, signal analytics,
        performance metrics, strategy breakdown, veto accuracy.
        """
        import datetime as _dt
        import sqlite3 as _sq

        now          = _dt.datetime.now()
        market_open  = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        # ── Uptime & market coverage ─────────────────────────────────────────
        session_secs      = (market_close - market_open).total_seconds()   # 22500s
        bot_secs          = (now - self._session_start).total_seconds()
        # Did the bot start before market open?
        started_before_open = self._session_start.time() <= market_open.time()
        # Effective coverage = overlap of [bot_start, now] with [market_open, market_close]
        overlap_start = max(self._session_start, market_open)
        overlap_end   = min(now, market_close)
        covered_secs  = max(0, (overlap_end - overlap_start).total_seconds())
        market_uptime_pct = round(min(100.0, covered_secs / session_secs * 100), 1)
        market_coverage   = "YES" if started_before_open and market_uptime_pct >= 95 else "PARTIAL" if market_uptime_pct > 0 else "NO"

        ws_connected  = getattr(self.ws, "connected", False)
        first_scan    = getattr(self, "_first_scan_time", None)
        last_scan_str  = self._last_scan_time.strftime("%H:%M:%S") if self._last_scan_time else "—"
        first_scan_str = first_scan.strftime("%H:%M:%S") if first_scan else "—"
        scan_cycles    = getattr(self, "_scan_cycles", 0)

        # ── Trade metrics from DB ────────────────────────────────────────────
        try:
            db_path = "data/market_data.db"
            conn = _sq.connect(db_path)
            today = _dt.date.today().isoformat()
            rows = conn.execute(
                """SELECT pnl, strategy FROM trades
                   WHERE DATE(entry_time) = ?
                     AND exit_price IS NOT NULL AND exit_price > 0""",
                (today,)
            ).fetchall()
            conn.close()
            trade_rows = [(r[0], r[1]) for r in rows if r[0] is not None]
        except Exception:
            trade_rows = []

        pnls     = [r[0] for r in trade_rows]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p < 0]
        total    = len(pnls)

        win_rate      = round(len(wins) / total * 100, 1) if total > 0 else None
        avg_win       = round(sum(wins) / len(wins), 0)   if wins   else None
        avg_loss      = round(sum(losses) / len(losses), 0) if losses else None
        gross_profit  = sum(wins)
        gross_loss    = abs(sum(losses))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (None if gross_profit == 0 else float("inf"))
        rr_ratio      = round(abs(avg_win / avg_loss), 2) if avg_win and avg_loss else None

        # Max drawdown
        max_drawdown = 0.0
        if pnls:
            cum, peak = 0.0, 0.0
            for p in pnls:
                cum += p
                if cum > peak:
                    peak = cum
                dd = peak - cum
                if dd > max_drawdown:
                    max_drawdown = dd

        # ── Strategy breakdown ────────────────────────────────────────────────
        strat_stats = {}
        for strat in ("momentum", "mean_reversion", "options"):
            s_pnls = [r[0] for r in trade_rows if r[1] == strat]
            s_wins = [p for p in s_pnls if p > 0]
            strat_stats[strat] = {
                "trades":   len(s_pnls),
                "win_rate": round(len(s_wins) / len(s_pnls) * 100, 0) if s_pnls else None,
                "pnl":      round(sum(s_pnls), 0),
            }

        # ── Veto accuracy (outcomes of AI-rejected signals) ───────────────────
        # We track this in-memory via _veto_outcomes: list of (symbol, entry_price, was_good_veto)
        # A good veto = price moved against the signal direction after rejection
        veto_outcomes   = getattr(self, "_veto_outcomes", [])
        good_vetoes     = sum(1 for v in veto_outcomes if v.get("good_veto"))
        missed_opps     = sum(1 for v in veto_outcomes if not v.get("good_veto"))
        total_assessed  = len(veto_outcomes)
        veto_accuracy   = round(good_vetoes / total_assessed * 100, 1) if total_assessed > 0 else None

        # Approval rate
        ai_total      = self._ai_approved + self._ai_rejected
        approval_rate = round(self._ai_approved / ai_total * 100, 1) if ai_total > 0 else None

        return {
            # Operational
            "session_start":      self._session_start.strftime("%H:%M:%S"),
            "market_coverage":    market_coverage,
            "first_scan":         first_scan_str,
            "last_scan":          last_scan_str,
            "scan_cycles":        scan_cycles,
            "ws_connected":       ws_connected,
            "uptime_pct":         market_uptime_pct,
            # Signal analytics
            "signals_today":      self._signals_today,
            "ai_approved":        self._ai_approved,
            "ai_rejected":        self._ai_rejected,
            "approval_rate":      approval_rate,
            # Performance
            "trades_closed":      total,
            "win_rate":           win_rate,
            "profit_factor":      profit_factor,
            "max_drawdown":       round(max_drawdown, 0),
            "avg_win":            avg_win,
            "avg_loss":           avg_loss,
            "rr_ratio":           rr_ratio,
            # Strategy breakdown
            "strategy_stats":     strat_stats,
            # Veto quality
            "good_vetoes":        good_vetoes,
            "missed_opportunities": missed_opps,
            "veto_accuracy":      veto_accuracy,
        }

    def _fo_days_to_expiry(self, symbol: str) -> int:
        """Parse expiry from option symbol like NIFTY29MAY2526000CE → days left."""
        import re
        m = re.search(r'(\d{2})([A-Z]{3})(\d{2,4})', symbol)
        if not m:
            return 99
        day, mon, yr = m.group(1), m.group(2), m.group(3)
        yr = f"20{yr}" if len(yr) == 2 else yr
        try:
            expiry = datetime.strptime(f"{day}{mon}{yr}", "%d%b%Y").date()
            return (expiry - datetime.now().date()).days
        except Exception:
            return 99

    def _exit_fo_position(self, symbol: str, pos: dict, exit_price: float, reason: str):
        """Exit an F&O (NRML) position and send Telegram alert."""
        try:
            self.client.place_order(
                symbol=pos["symbol"],
                exchange="nse_fo",
                transaction_type="S",
                quantity=pos["quantity"],
                price=exit_price,
                order_type="L",
                product="NRML",
            )
            pnl = self.risk.record_exit(symbol, exit_price)

            self.data.record_trade({
                "symbol":      pos["symbol"],
                "exchange":    "nse_fo",
                "action":      pos["action"],
                "strategy":    pos.get("strategy", "options"),
                "entry_price": pos["entry_price"],
                "exit_price":  exit_price,
                "quantity":    pos["quantity"],
                "pnl":         pnl or 0,
                "exit_reason": reason,
                "entry_time":  pos.get("entry_time", ""),
                "order_no":    pos.get("order_no", ""),
            })

            # Parse option details for alert
            import re
            m = re.match(r'([A-Z]+)(\d{2}[A-Z]{3}\d{2,4})(\d+)(CE|PE)', symbol)
            underlying = m.group(1) if m else symbol
            strike     = int(m.group(3)) if m else 0
            option_type = m.group(4) if m else ""
            days_left  = self._fo_days_to_expiry(symbol)

            self.telegram.alert_fo_trade(
                symbol=symbol,
                action="SELL",
                qty=pos["quantity"],
                premium=exit_price,
                strike=strike,
                expiry=None,
                option_type=option_type,
                underlying=underlying,
                days_to_expiry=days_left,
                exit_reason=reason,
                paper=self.client.paper_trading,
            )
            self.data.log_event(
                "FO_EXIT",
                f"{reason.upper()} {symbol} @ ₹{exit_price:.2f} P&L=₹{pnl or 0:,.0f}",
                symbol=symbol,
                metadata={"reason": reason, "exit_price": exit_price, "pnl": pnl or 0}
            )
        except Exception as e:
            logger.error(f"F&O exit failed for {symbol}: {e}")

    def _squareoff_all_fo_positions(self):
        """EOD square-off — close all NRML positions at 3:15 PM — weekdays only."""
        if datetime.now().weekday() >= 5:
            return
        logger.info("EOD F&O square-off triggered")
        fo_pos = {k: v for k, v in self.risk.open_positions.items()
                  if v.get("product") == "NRML"}
        for symbol, pos in list(fo_pos.items()):
            try:
                quote = self.data.get_live_quote(symbol, exchange="nse_fo")
                price = quote.get("ltp", 0) if quote else pos["entry_price"]
                self._exit_fo_position(symbol, pos, price, "eod_squareoff")
            except Exception as e:
                logger.error(f"EOD squareoff failed for {symbol}: {e}")

    def _exit_position(self, symbol: str, pos: dict, exit_price: float, reason: str):
        """Exit a position (stop loss or target)."""
        try:
            exit_action = "S" if pos["action"] == "BUY" else "B"
            self.client.place_order(
                symbol=pos["symbol"],
                exchange=pos["exchange"],
                transaction_type=exit_action,
                quantity=pos["quantity"],
                price=exit_price,
                order_type="L",
                product=pos.get("product", "CNC"),
            )

            pnl = self.risk.record_exit(symbol, exit_price)

            # Link outcome back to the AI decision row
            decision_id = pos.get("_decision_id")
            if decision_id:
                outcome      = "win" if (pnl or 0) > 0 else ("loss" if (pnl or 0) < 0 else "breakeven")
                good_decision = (pnl or 0) > 0   # approved trade → good if it won
                try:
                    self.data.update_ai_outcome(decision_id, outcome, pnl or 0, good_decision)
                except Exception as _e:
                    logger.debug(f"AI outcome update non-critical error: {_e}")

            # Persist trade to SQLite for dashboard / analytics
            self.data.record_trade({
                "symbol":      pos["symbol"],
                "exchange":    pos["exchange"],
                "action":      pos["action"],
                "strategy":    pos.get("strategy", ""),
                "entry_price": pos["entry_price"],
                "exit_price":  exit_price,
                "quantity":    pos["quantity"],
                "pnl":         pnl or 0,
                "exit_reason": reason,
                "entry_time":  pos.get("entry_time", ""),
                "order_no":    pos.get("order_no", ""),
            })

            if reason == "stop_loss":
                self.telegram.alert_stop_loss_hit(
                    symbol, pos["entry_price"], exit_price, pnl or 0, pos["quantity"]
                )
            else:
                self.telegram.alert_target_hit(
                    symbol, pos["entry_price"], exit_price, pnl or 0, pos["quantity"]
                )

            self.dashboard.log_event(
                f"EXIT {reason.upper()}: {symbol} @ ₹{exit_price:,.2f}"
                f"  P&L=₹{pnl or 0:,.0f}"
            )
            self.data.log_event(
                "EXIT",
                f"{reason.upper()} {symbol} @ ₹{exit_price:.2f} P&L=₹{pnl or 0:,.0f}",
                symbol=symbol,
                metadata={
                    "reason": reason,
                    "exit_price": exit_price,
                    "pnl": pnl or 0,
                }
            )

        except Exception as e:
            logger.error(f"Exit position error for {symbol}: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Scheduled jobs
    # ─────────────────────────────────────────────────────────────────────────

    def job_daily_setup(self):
        """Run at 8:45 AM before market opens — weekdays only."""
        if datetime.now().weekday() >= 5:
            return
        logger.info("Running daily setup...")
        self.trading_paused = False  # Reset any previous day's pause
        self.cb.reset_for_new_day()  # Phase 13: reset circuit breaker state

        # Send market-open alert once per day at 8:45 AM
        _today_flag = Path(__file__).parent / f"logs/.market_open_{datetime.now().date()}"
        if not _today_flag.exists():
            self.telegram.alert_market_open()
            _today_flag.touch()
        self._prev_cb_state = CBState.NORMAL
        self.data.download_scrip_master()
        self.client.authenticate()
        # Purge old ticks (keep 2 days) and old events (keep 30 days)
        self.data.cleanup_old_data(tick_days=2, events_days=30)

        # Bootstrap 5m OHLCV for indicator warm-up
        logger.info("Bootstrapping 5m OHLCV from yfinance...")
        candles = self.data.bootstrap_ohlcv(self.active_watchlist, period="60d", interval="5m")
        logger.info(f"5m OHLCV bootstrap: {candles:,} candles loaded")

        # Append yesterday's daily close to the 1-year daily dataset
        logger.info("Refreshing daily OHLCV (appending latest session)...")
        self.data.bootstrap_daily_ohlcv(self.active_watchlist, years=1)

        # Phase 12: run screener to build today's active watchlist
        logger.info("Running daily screener...")
        try:
            screened = self.screener.run(
                top_n=20,
                min_score=0.45,
                always_include=WATCHLIST,   # hardcoded core always in
            )
            if screened:
                self.active_watchlist = screened
                logger.info(f"Screener: active watchlist updated → {len(self.active_watchlist)} symbols")
                self.telegram.send(
                    f"📊 <b>Daily Screener</b>\n"
                    f"{len(self.active_watchlist)} symbols active today.\n"
                    f"Top 5: {', '.join(self.active_watchlist[:5])}"
                )
        except Exception as e:
            logger.warning(f"Screener failed (non-critical) — keeping previous watchlist: {e}")

        # Phase 8: refresh calendar for the day
        self._calendar_ctx = get_calendar_context()
        cal_notes = self._calendar_ctx.get("trading_notes", [])
        if cal_notes:
            logger.info("Calendar notes: " + " | ".join(cal_notes))

        # ── AI Morning Brief (Phase 2: with live VIX + FII/DII) ─────────────
        logger.info("Requesting AI morning brief...")
        try:
            vix      = self.news.get_india_vix()
            flows    = self.news.get_fii_dii_flows()
            headlines = self.news.get_market_headlines()
            brief = self.ai.morning_market_brief(
                headlines=headlines,
                vix=vix,
                fii_net=flows.get("fii_net"),
                dii_net=flows.get("dii_net"),
            )
            caution = brief.get("caution_level", "low")
            bias    = brief.get("bias", "neutral")
            reason  = brief.get("reason", "")
            suggestion = brief.get("suggestion", "")

            logger.info(
                f"Morning brief — bias={bias}, caution={caution}: {reason}"
            )

            # Hard pause on event day (RBI/Budget) — override AI brief
            if any(ev.get("days_away") == 0
                   for ev in self._calendar_ctx.get("economic_events_14d", [])):
                self.trading_paused = True
                event_names = ", ".join(
                    ev["label"] for ev in self._calendar_ctx["economic_events_14d"]
                    if ev.get("days_away") == 0
                )
                self.telegram.send(
                    f"📅 <b>HIGH-IMPACT EVENT TODAY — TRADING PAUSED</b>\n{event_names}"
                )
                logger.warning(f"Trading paused: high-impact event today ({event_names})")

            if brief.get("avoid_trading"):
                self.trading_paused = True
                msg = (
                    f"⛔ <b>AI Morning Brief — TRADING PAUSED</b>\n\n"
                    f"Reason: {reason}\n"
                    f"Suggestion: {suggestion}"
                )
                logger.warning(f"Trading paused by AI: {reason}")
            else:
                caution_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(caution, "⚪")
                msg = (
                    f"{caution_emoji} <b>AI Morning Brief</b>\n"
                    f"Bias: {bias.upper()} | Caution: {caution.upper()}\n"
                    f"{reason}\n"
                    f"💡 {suggestion}"
                )

            # Append calendar footnote
            dte = self._calendar_ctx.get("days_to_expiry")
            if dte is not None:
                exp_label = self._calendar_ctx.get("next_expiry", {}).get("label", "")
                expiry_line = f"\n📅 {exp_label}"
                events_today = [
                    ev["label"] for ev in self._calendar_ctx.get("economic_events_14d", [])
                    if ev.get("days_away", 99) <= 2
                ]
                if events_today:
                    expiry_line += "\n⚡ " + " | ".join(events_today)
                msg += expiry_line

            self.telegram.send(msg)
        except Exception as e:
            logger.warning(f"Morning brief failed (non-critical): {e}")

        self.telegram.send(
            f"📋 Daily setup complete. Scrip master refreshed. OHLCV: {candles:,} candles loaded."
        )

    def _assess_veto_outcomes(self):
        """
        At EOD, check each vetoed signal: did price hit stop_loss (good veto)
        or hit target (missed opportunity)?
        Uses current EOD price as final check.
        """
        for v in self._veto_outcomes:
            if v.get("good_veto") is not None:
                continue  # already assessed
            quote = self.data.get_live_quote(v["symbol"],
                                             "nse_fo" if "CE" in v["symbol"] or "PE" in v["symbol"] else "nse_cm")
            if not quote:
                continue
            eod_price  = quote.get("ltp", v["vetoed_at"])
            entry      = v["entry_price"]
            stop       = v["stop_loss"]
            target     = v["target"]
            action     = v["action"]

            if action == "BUY":
                # Good veto: price fell to or below stop loss level
                # Missed opportunity: price rose to or above target
                if eod_price <= stop:
                    v["good_veto"] = True
                elif eod_price >= target:
                    v["good_veto"] = False
                else:
                    # Price between entry and target — inconclusive, mark as good veto
                    # (avoided uncertainty = conservative win for AI)
                    v["good_veto"] = eod_price < entry
            else:  # SELL
                if eod_price >= stop:
                    v["good_veto"] = True
                elif eod_price <= target:
                    v["good_veto"] = False
                else:
                    v["good_veto"] = eod_price > entry

    def job_market_close(self):
        """Run at 3:35 PM after market closes — weekdays only."""
        if datetime.now().weekday() >= 5:
            return  # no EOD report on weekends
        portfolio = self.risk.get_portfolio_summary()

        # Portfolio analytics report
        trades_df = self.data.get_trades(limit=500)
        if not trades_df.empty:
            trade_list = trades_df.to_dict("records")
            report     = self.analytics.text_report(
                trade_list, portfolio.get("open_positions", {})
            )
            logger.info(f"\n{report}")

        self._assess_veto_outcomes()
        ai_advice     = self.ai.get_portfolio_advice(portfolio)
        session_stats = self.get_session_stats()
        self.telegram.alert_daily_summary(portfolio, ai_advice, session_stats)
        self.data.update_bot_status(portfolio, self.client.paper_trading, status="closed")
        logger.info(f"Market close. Day P&L: ₹{portfolio['daily_pnl']:,.0f}")

        # Write daily journal entry
        self._write_daily_journal(portfolio, session_stats)

    def _write_daily_journal(self, portfolio: dict, session_stats: dict):
        """
        Persist today's session summary to the daily_journal table and
        write a human-readable markdown file to logs/journal/.
        """
        import os
        from datetime import date

        today        = date.today().isoformat()
        cumulative   = sum(
            t.get("pnl", 0)
            for t in self.data.get_trades(limit=10_000).to_dict("records")
        )
        day_pnl      = portfolio.get("daily_pnl", 0)
        open_pos     = len(portfolio.get("open_positions", {}))
        trades_today = portfolio.get("daily_trades", 0)

        # Wins/losses from today's closed trades
        today_trades = self.data.get_trades(limit=200)
        wins = losses = 0
        if not today_trades.empty:
            today_str = today
            day_rows  = today_trades[today_trades["exit_time"].str.startswith(today_str, na=False)]
            wins      = int((day_rows["pnl"] > 0).sum())
            losses    = int((day_rows["pnl"] <= 0).sum())

        stats = {
            "market_regime":    getattr(self, "_market_regime", "unknown"),
            "signals_seen":     session_stats.get("signals_seen", 0),
            "signals_approved": self._ai_approved,
            "signals_vetoed":   self._ai_rejected,
            "trades_opened":    trades_today,
            "trades_closed":    wins + losses,
            "day_pnl":          day_pnl,
            "cumulative_pnl":   cumulative,
            "win_count":        wins,
            "loss_count":       losses,
            "open_positions":   open_pos,
            "notes":            f"Regime={getattr(self, '_market_regime','?')} | Paper={self.client.paper_trading}",
        }
        self.data.write_daily_journal(today, stats)

        # Write markdown file for easy reading
        log_dir = os.path.join(os.path.dirname(__file__), "logs", "journal")
        os.makedirs(log_dir, exist_ok=True)
        md_path = os.path.join(log_dir, f"{today}.md")
        wr       = f"{wins/(wins+losses):.0%}" if (wins + losses) > 0 else "N/A"
        total_t  = self.data.get_trades(limit=10_000)
        all_wins  = int((total_t["pnl"] > 0).sum()) if not total_t.empty else 0
        all_n     = len(total_t)
        all_wr    = f"{all_wins/all_n:.0%}" if all_n else "N/A"

        # AI accuracy from ai_decisions table
        ai_df = self.data.get_ai_decisions(limit=10_000)
        ai_acc = "N/A"
        if not ai_df.empty and "good_decision" in ai_df.columns:
            resolved = ai_df[ai_df["good_decision"].notna()]
            if len(resolved):
                ai_acc = f"{resolved['good_decision'].mean():.0%} ({len(resolved)} decisions)"

        with open(md_path, "w") as f:
            f.write(f"# Daily Journal — {today}\n\n")
            f.write(f"## Session Summary\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| Market Regime | {stats['market_regime']} |\n")
            f.write(f"| Mode | {'PAPER' if self.client.paper_trading else 'LIVE'} |\n")
            f.write(f"| Signals Seen | {stats['signals_seen']} |\n")
            f.write(f"| AI Approved | {stats['signals_approved']} |\n")
            f.write(f"| AI Vetoed | {stats['signals_vetoed']} |\n")
            f.write(f"| Trades Closed | {wins + losses} |\n")
            f.write(f"| Today Wins / Losses | {wins} / {losses} |\n")
            f.write(f"| Today Win Rate | {wr} |\n")
            f.write(f"| Day P&L | ₹{day_pnl:+,.0f} |\n")
            f.write(f"| Open Positions | {open_pos} |\n\n")
            f.write(f"## Cumulative Performance\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| Total Trades | {all_n} |\n")
            f.write(f"| Overall Win Rate | {all_wr} |\n")
            f.write(f"| Cumulative P&L | ₹{cumulative:+,.0f} |\n")
            f.write(f"| AI Decision Accuracy | {ai_acc} |\n\n")
            f.write(f"## Notes\n")
            f.write(f"Paper trading bot running. No indicator optimization until 100+ trades.\n")
            f.write(f"Next evaluation milestone: {max(0, 100 - all_n)} more trades needed.\n\n")
            f.write(f"---\n*Generated by AlgoTrader at market close*\n")

        logger.info(f"Daily journal written → {md_path}")

    def job_weekly_optimisation(self):
        """
        Run every Sunday at 00:30.
        Grid-searches strategy params on the last 500 candles of OHLCV,
        saves best params to DB, re-instantiates strategies with new params,
        and sends a Telegram summary.
        """
        logger.info("=" * 50)
        logger.info("Starting weekly hyperparameter optimisation...")
        logger.info("=" * 50)
        try:
            results = self.optimizer.run_all(self.active_watchlist)
            if not results:
                logger.warning("Optimisation returned no results (insufficient data?)")
                return

            # Re-instantiate strategies with freshly optimised params
            for r in results:
                if r.strategy == "momentum":
                    self.momentum = MomentumStrategy(**r.best_params)
                    self.aggregator.strategies["momentum"] = (
                        self.momentum,
                        self.aggregator.strategies["momentum"][1]
                    )
                    logger.info(f"Momentum strategy updated: {r.best_params}")

                elif r.strategy == "mean_reversion":
                    self.mean_rev = MeanReversionStrategy(**r.best_params)
                    self.aggregator.strategies["mean_reversion"] = (
                        self.mean_rev,
                        self.aggregator.strategies["mean_reversion"][1]
                    )
                    logger.info(f"MeanReversion strategy updated: {r.best_params}")

            # Telegram summary
            lines = ["⚙️ <b>Weekly Optimisation Complete</b>\n"]
            for r in results:
                pf = "∞" if r.best_pnl == float("inf") else f"₹{r.best_pnl:,.0f}"
                lines.append(
                    f"<b>{r.strategy.upper()}</b>\n"
                    f"  Sharpe: {r.best_sharpe:.2f}  Win: {r.best_winrate:.1%}  AvgPnL: {pf}\n"
                    f"  Params: {r.best_params}\n"
                )
                if r.ai_explanation:
                    lines.append(f"  💡 {r.ai_explanation}\n")
            self.telegram.send("\n".join(lines)[:4096])

            # Phase 7: retrain ML ensemble after grid-search (fresh labelled data available)
            try:
                logger.info("Retraining ML ensemble...")
                ml = train_model()
                if ml is not None:
                    reload_model()   # flush in-memory cache so next prediction uses new weights
                    logger.info("ML ensemble retrained successfully")
                    self.telegram.send("🤖 ML ensemble retrained on latest trade history.")
                else:
                    logger.info("ML ensemble skipped — insufficient labelled decisions")
            except Exception as ml_err:
                logger.warning(f"ML retrain failed (non-critical): {ml_err}")

        except Exception as e:
            logger.error(f"Weekly optimisation failed: {e}", exc_info=True)
            self.telegram.alert_error(str(e), "weekly_optimisation")

    def job_weekly_summary(self):
        """
        Run every Friday at 15:40 (5 min after market close job).
        Compiles the week's trading data and sends a structured summary to Telegram.
        """
        import numpy as np
        from datetime import date, timedelta

        logger.info("Generating weekly summary...")
        try:
            today     = date.today()
            # Monday of the current week
            week_start = today - timedelta(days=today.weekday())
            week_end   = today
            week_label = (
                f"{week_start.strftime('%d %b')} – {week_end.strftime('%d %b %Y')}"
            )

            # ── Trades this week ─────────────────────────────────────────────
            all_trades = self.data.get_trades(limit=10_000)
            if not all_trades.empty and "exit_time" in all_trades.columns:
                mask = all_trades["exit_time"].str[:10] >= str(week_start)
                week_trades = all_trades[mask]
            else:
                week_trades = all_trades.iloc[0:0]   # empty

            n        = len(week_trades)
            pnls     = week_trades["pnl"].tolist() if n else []
            wins     = [p for p in pnls if p > 0]
            losses   = [p for p in pnls if p <= 0]
            week_pnl = sum(pnls)
            wr       = len(wins) / n if n else 0
            gw       = sum(wins)
            gl       = abs(sum(losses))
            pf       = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
            aw       = float(np.mean(wins))   if wins   else 0.0
            al       = float(np.mean(losses)) if losses else 0.0

            # ── Cumulative P&L ───────────────────────────────────────────────
            cum_pnl = float(all_trades["pnl"].sum()) if not all_trades.empty else 0.0

            # ── Best / worst symbol ──────────────────────────────────────────
            best_sym = worst_sym = None
            best_pnl = worst_pnl = 0.0
            if n and "symbol" in week_trades.columns:
                sym_pnl = (
                    week_trades.groupby("symbol")["pnl"].sum().sort_values()
                )
                if len(sym_pnl):
                    worst_sym = sym_pnl.index[0]
                    worst_pnl = float(sym_pnl.iloc[0])
                    best_sym  = sym_pnl.index[-1]
                    best_pnl  = float(sym_pnl.iloc[-1])

            # ── Strategy breakdown ───────────────────────────────────────────
            strategy_breakdown = []
            if n and "strategy" in week_trades.columns:
                for strat, grp in week_trades.groupby("strategy"):
                    sp  = grp["pnl"].tolist()
                    sw  = sum(1 for p in sp if p > 0)
                    strategy_breakdown.append({
                        "name":   strat,
                        "trades": len(sp),
                        "wr":     sw / len(sp),
                        "pnl":    sum(sp),
                    })
                strategy_breakdown.sort(key=lambda x: -x["pnl"])

            # ── AI decision accuracy this week ───────────────────────────────
            ai_df = self.data.get_ai_decisions(limit=10_000)
            if not ai_df.empty:
                ai_week = ai_df[ai_df["decided_at"].str[:10] >= str(week_start)]
            else:
                ai_week = ai_df
            ai_total   = len(ai_week)
            ai_approved = int((ai_week["approved"] == 1).sum()) if ai_total else 0
            ai_vetoed   = ai_total - ai_approved
            ai_acc      = None
            if ai_total and "good_decision" in ai_week.columns:
                resolved = ai_week[ai_week["good_decision"].notna()]
                if len(resolved):
                    ai_acc = float(resolved["good_decision"].mean())

            # ── Daily journal strip ──────────────────────────────────────────
            journal_df  = self.data.get_daily_journal(days=7)
            journal_days = []
            if not journal_df.empty:
                for _, row in journal_df.iterrows():
                    if str(row["date"]) >= str(week_start):
                        journal_days.append({
                            "date":    str(row["date"]),
                            "day_pnl": float(row.get("day_pnl", 0) or 0),
                            "regime":  str(row.get("market_regime", "?") or "?"),
                        })
                journal_days.sort(key=lambda x: x["date"])

            # ── Milestone ────────────────────────────────────────────────────
            milestone_trades = len(all_trades)

            report = {
                "week_label":         week_label,
                "week_pnl":           week_pnl,
                "cumulative_pnl":     cum_pnl,
                "total_trades":       n,
                "win_rate":           wr,
                "profit_factor":      pf,
                "avg_winner":         aw,
                "avg_loser":          al,
                "best_symbol":        best_sym,
                "best_symbol_pnl":    best_pnl,
                "worst_symbol":       worst_sym,
                "worst_symbol_pnl":   worst_pnl,
                "strategy_breakdown": strategy_breakdown,
                "ai_decisions_total": ai_total,
                "ai_approved":        ai_approved,
                "ai_vetoed":          ai_vetoed,
                "ai_accuracy":        ai_acc,
                "journal_days":       journal_days,
                "milestone_trades":   milestone_trades,
                "milestone_target":   100,
            }

            self.telegram.alert_weekly_summary(report)
            logger.info(
                f"Weekly summary sent: {n} trades, P&L ₹{week_pnl:+,.0f}, WR {wr:.0%}"
            )

        except Exception as e:
            logger.error(f"Weekly summary failed: {e}", exc_info=True)
            self.telegram.alert_error(str(e), "weekly_summary")

    def job_weekly_review(self):
        """
        Run every Sunday at 01:00 (after optimisation at 00:30).
        Asks Claude to critique its own decisions from the past week and saves the review.
        """
        logger.info("Running weekly AI self-review...")
        try:
            feedback = FeedbackLoop(self.ai)
            review_text = feedback.run_weekly_review()
            if review_text:
                self.telegram.send(
                    f"🧠 <b>Weekly AI Self-Review</b>\n\n{review_text[:3000]}"
                )
                logger.info("Weekly AI self-review saved and sent to Telegram")
        except Exception as e:
            logger.error(f"Weekly AI self-review failed: {e}", exc_info=True)

    def job_refresh_session(self):
        """Re-authenticate every 6 hours."""
        logger.info("Refreshing session...")
        self.client.authenticate()

    # ─────────────────────────────────────────────────────────────────────────
    # Market hours
    # ─────────────────────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """
        NSE trading hours: 9:15 AM to 3:30 PM IST, Mon-Fri,
        excluding NSE_HOLIDAYS.
        """
        now  = datetime.now()
        today = now.date()

        if now.weekday() >= 5:           # Saturday / Sunday
            return False
        if today in NSE_HOLIDAYS:        # Exchange holiday
            return False

        open_time  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
        close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return open_time <= now <= close_time

    # ─────────────────────────────────────────────────────────────────────────
    # Shutdown and cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def shutdown(self):
        """Graceful shutdown sequence."""
        logger.info("=" * 60)
        logger.info("Starting shutdown sequence...")
        logger.info("=" * 60)

        self.running = False

        # Stop Telegram command handler (Phase 10)
        try:
            if self._cmd_handler:
                self._cmd_handler.stop()
        except Exception as e:
            logger.error(f"Error stopping Telegram command handler: {e}")

        # Stop WebSocket
        try:
            if self.ws:
                logger.info("Stopping WebSocket...")
                self.ws.stop()
        except Exception as e:
            logger.error(f"Error stopping WebSocket: {e}")

        # Check for open positions
        try:
            positions = self.risk.open_positions
            if positions:
                logger.warning(f"Found {len(positions)} open positions at shutdown. Review manually.")
                portfolio = self.risk.get_portfolio_summary()
                logger.info(f"Final portfolio: {portfolio}")
        except Exception as e:
            logger.error(f"Error checking positions: {e}")

        # Update final bot status
        try:
            portfolio = self.risk.get_portfolio_summary()
            self.data.update_bot_status(
                portfolio,
                paper_trading=self.client.paper_trading,
                status="stopped",
            )
            logger.info(f"Final portfolio state saved. Day P&L: ₹{portfolio['daily_pnl']:,.0f}")
        except Exception as e:
            logger.error(f"Error saving final status: {e}")

        # Send brief shutdown notification (full EOD report already sent by job_market_close)
        try:
            self.telegram.send("🔴 <b>AlgoTrader stopped.</b>")
        except Exception as e:
            logger.error(f"Error sending shutdown alert: {e}")

        logger.info("=" * 60)
        logger.info("Shutdown sequence complete")
        logger.info("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        """Start the bot with scheduled jobs."""
        if not self.startup():
            return

        import signal
        import sys

        def shutdown_handler(sig, frame):
            logger.info(f"Received signal {sig}. Initiating shutdown...")
            self.shutdown()
            sys.exit(0)

        # Register signal handlers
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        self.running = True
        logger.info("Scheduling jobs...")

        schedule.every().day.at("08:45").do(self.job_daily_setup)
        schedule.every(5).minutes.do(self.run_trading_cycle)
        schedule.every().day.at("15:15").do(self._squareoff_all_fo_positions)
        schedule.every().day.at("15:35").do(self.job_market_close)
        schedule.every(6).hours.do(self.job_refresh_session)
        schedule.every().friday.at("15:40").do(self.job_weekly_summary)
        schedule.every().sunday.at("00:30").do(self.job_weekly_optimisation)
        schedule.every().sunday.at("01:00").do(self.job_weekly_review)

        logger.info("Bot running. Press Ctrl+C to stop.")
        logger.info(f"Paper trading: {self.client.paper_trading}")
        logger.info(f"Watchlist: {self.active_watchlist}")

        _consec_errors = 0
        while self.running:
            try:
                schedule.run_pending()
                _consec_errors = 0
                time.sleep(30)
            except KeyboardInterrupt:
                logger.info("Shutdown requested.")
                self.shutdown()
                break
            except Exception as e:
                _consec_errors += 1
                logger.error(f"Scheduler error: {e}", exc_info=True)
                # Only alert Telegram after 3 consecutive failures to avoid spam
                # (transient DB locks during scrip master download are normal)
                if _consec_errors >= 3:
                    self.telegram.alert_error(str(e), "scheduler")
                    _consec_errors = 0
                time.sleep(60)

        logger.info("AlgoTrader stopped.")

    def run_with_dashboard(self):
        """
        Start the bot AND launch the live dashboard in the foreground.
        The trading loop runs in a background thread; dashboard owns the terminal.
        """
        if not self.startup():
            return

        import threading
        import signal
        import sys

        def shutdown_handler(sig, frame):
            logger.info(f"Received signal {sig}. Initiating shutdown...")
            self.shutdown()
            sys.exit(0)

        # Register signal handlers
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        self.running = True
        schedule.every().day.at("08:45").do(self.job_daily_setup)
        schedule.every(5).minutes.do(self.run_trading_cycle)
        schedule.every().day.at("15:15").do(self._squareoff_all_fo_positions)
        schedule.every().day.at("15:35").do(self.job_market_close)
        schedule.every(6).hours.do(self.job_refresh_session)
        schedule.every().friday.at("15:40").do(self.job_weekly_summary)
        schedule.every().sunday.at("00:30").do(self.job_weekly_optimisation)
        schedule.every().sunday.at("01:00").do(self.job_weekly_review)

        def _scheduler_thread():
            _consec_errors = 0
            while self.running:
                try:
                    schedule.run_pending()
                    _consec_errors = 0
                    time.sleep(30)
                except Exception as e:
                    _consec_errors += 1
                    logger.error(f"Scheduler error: {e}", exc_info=True)
                    if _consec_errors >= 3:
                        self.telegram.alert_error(str(e), "scheduler")
                        _consec_errors = 0
                    time.sleep(60)

        t = threading.Thread(target=_scheduler_thread, daemon=True)
        t.start()

        try:
            self.dashboard.run()   # blocks until Ctrl+C
        except Exception as e:
            logger.error(f"Dashboard error: {e}", exc_info=True)
        finally:
            self.shutdown()
            logger.info("AlgoTrader stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    bot = AlgoTrader()
    if "--dashboard" in sys.argv:
        bot.run_with_dashboard()
    else:
        bot.run()
