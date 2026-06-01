"""
AlgoTrader — Comprehensive Backtesting Engine
==============================================
Fetches 6 or 12 months of 5-minute OHLCV from yfinance, replays each bar
chronologically, and runs Momentum / Mean-Reversion strategies with a full
realistic cost model (Indian market).

Cost model per trade (one side):
  Brokerage:        ₹20 flat (Zerodha-style)
  STT (buy only):   0.1%  of turnover (equity delivery intraday: 0.025%)
                    We use 0.025% (intraday) since positions rarely held >1 day
  GST on brokerage: 18%
  Exchange charge:  0.00322% of turnover (NSE)
  Stamp duty:       0.015% on BUY side only
  Slippage:         0.05% per side (one tick / 2)

Usage:
    python backtest/engine.py                     # 6 months, all watchlist, both strategies
    python backtest/engine.py --months 12          # 12 months
    python backtest/engine.py --strategy momentum  # single strategy
    python backtest/engine.py --capital 500000     # different capital
    python backtest/engine.py --output csv --out-file results.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")

# ── Watchlist (mirrors main.py) ───────────────────────────────────────────────
WATCHLIST = [
    "RELIANCE-EQ", "TCS-EQ",    "INFY-EQ",      "HDFCBANK-EQ",
    "ICICIBANK-EQ","WIPRO-EQ",  "LT-EQ",         "AXISBANK-EQ",
    "ITBEES-EQ",   "NIFTYBEES-EQ",
]

# NSE suffix for yfinance
_YF_SUFFIX = ".NS"
_YF_MAP = {s.replace("-EQ", "") + _YF_SUFFIX: s for s in WATCHLIST}
# e.g. "RELIANCE.NS" → "RELIANCE-EQ"

# ── Cost constants ─────────────────────────────────────────────────────────────
BROKERAGE_FLAT  = 20.0      # ₹ per order (flat)
STT_PCT         = 0.00025   # 0.025% intraday (buy-side for delivery)
GST_PCT         = 0.18      # 18% on brokerage
EXCHANGE_PCT    = 0.0000322 # 0.00322% both sides
STAMP_PCT       = 0.00015   # 0.015% buy side only
SLIPPAGE_PCT    = 0.0005    # 0.05% per side


# ─────────────────────────────────────────────────────────────────────────────
# Data layer
# ─────────────────────────────────────────────────────────────────────────────

def _nse_to_yf(symbol: str) -> str:
    """Convert 'RELIANCE-EQ' → 'RELIANCE.NS'"""
    return symbol.replace("-EQ", "") + _YF_SUFFIX


def fetch_ohlcv(symbol: str, months: int = 6) -> Optional[pd.DataFrame]:
    """
    Download 5-minute OHLCV from yfinance for the requested months.
    yfinance 5m data is limited to the last 60 days — for longer periods
    we stitch multiple 60-day windows.
    Returns DataFrame with columns [timestamp, open, high, low, close, volume]
    filtered to IST market hours (09:15–15:30) only.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    yf_sym = _nse_to_yf(symbol)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=months * 31)

    frames: list[pd.DataFrame] = []

    # yfinance 5m is limited to 60-day windows; fetch in 55-day chunks
    chunk_days = 55
    chunk_end = end_dt
    while chunk_end > start_dt:
        chunk_start = max(chunk_end - timedelta(days=chunk_days), start_dt)
        try:
            df = yf.download(
                yf_sym,
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval="5m",
                progress=False,
                auto_adjust=True,
            )
            if df is not None and len(df) > 0:
                frames.append(df)
        except Exception as e:
            logger.warning(f"  yfinance chunk error for {symbol}: {e}")
        chunk_end = chunk_start - timedelta(days=1)
        time.sleep(0.3)  # polite rate limiting

    if not frames:
        logger.warning(f"  No data fetched for {symbol}")
        return None

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index)

    # Convert to IST (yfinance returns UTC for Indian stocks)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Kolkata")

    # Filter to market hours only
    market_open  = pd.Timestamp("09:15", tz="Asia/Kolkata").time()
    market_close = pd.Timestamp("15:30", tz="Asia/Kolkata").time()
    df = df[
        (df.index.time >= market_open) &
        (df.index.time <= market_close)
    ]

    # Drop NaN and zero-volume rows
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    df = df.reset_index().rename(columns={"Datetime": "timestamp", "index": "timestamp"})
    if "timestamp" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "timestamp"})
    if "timestamp" not in df.columns:
        df.insert(0, "timestamp", df.index)

    # Median anomaly filter (same as dashboard)
    for _ in range(3):
        if len(df) < 2:
            break
        med = df["close"].median()
        bad = (df["close"] - med).abs() / med > 0.05
        if bad.sum() == 0:
            break
        df = df[~bad].reset_index(drop=True)

    logger.info(f"  {symbol}: {len(df)} 5-min candles ({months}m window)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cost model
# ─────────────────────────────────────────────────────────────────────────────

def compute_costs(entry_price: float, exit_price: float,
                  quantity: int, action: str) -> float:
    """
    Return total round-trip transaction costs in ₹.
    action = "BUY"  → buy at entry, sell at exit
    action = "SELL" → sell at entry, buy at exit (short)
    """
    buy_turnover  = entry_price * quantity if action == "BUY" else exit_price * quantity
    sell_turnover = exit_price * quantity  if action == "BUY" else entry_price * quantity

    brokerage_entry = BROKERAGE_FLAT
    brokerage_exit  = BROKERAGE_FLAT
    gst_entry       = brokerage_entry * GST_PCT
    gst_exit        = brokerage_exit  * GST_PCT

    stt             = buy_turnover  * STT_PCT   # intraday STT on buy
    exchange        = (buy_turnover + sell_turnover) * EXCHANGE_PCT
    stamp           = buy_turnover  * STAMP_PCT

    total = (brokerage_entry + brokerage_exit
             + gst_entry + gst_exit
             + stt + exchange + stamp)
    return round(total, 2)


def apply_slippage(price: float, side: str) -> float:
    """Worsen fill price by slippage. side='buy' → higher, 'sell' → lower."""
    factor = 1 + SLIPPAGE_PCT if side == "buy" else 1 - SLIPPAGE_PCT
    return round(price * factor, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Trade & result data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol:      str
    strategy:    str
    action:      str
    entry_price: float
    exit_price:  float
    quantity:    int
    stop_loss:   float
    target:      float
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    exit_reason: str   # "target" | "stop_loss" | "eod" | "end_of_data"
    gross_pnl:   float
    costs:       float
    pnl:         float   # net of costs
    pnl_pct:     float
    confidence:  float


@dataclass
class BacktestResult:
    strategy:        str
    symbols_tested:  int  = 0
    total_trades:    int  = 0
    winning_trades:  int  = 0
    win_rate:        float = 0.0
    total_pnl:       float = 0.0
    total_costs:     float = 0.0
    profit_factor:   float = 0.0
    avg_win:         float = 0.0
    avg_loss:        float = 0.0
    rr_ratio:        float = 0.0
    max_drawdown:    float = 0.0
    max_drawdown_pct:float = 0.0
    sharpe:          float = 0.0
    cagr:            float = 0.0
    initial_capital: float = 100_000.0
    final_capital:   float = 0.0
    months_tested:   float = 0.0
    best_month:      str  = ""
    worst_month:     str  = ""
    best_month_pnl:  float = 0.0
    worst_month_pnl: float = 0.0
    monthly_returns: dict = field(default_factory=dict)  # "YYYY-MM" → ₹ pnl
    trades:          list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Core simulator
# ─────────────────────────────────────────────────────────────────────────────

class StrategyBacktester:
    """
    Walk-forward simulator for a single strategy across multiple symbols.

    Rules:
    - Only BUY signals are simulated (no shorting — Indian equity market constraint)
    - Fixed position size: initial_capital × position_pct (not compounding within a run)
    - Max 1 open position per symbol at a time
    - EOD exit forced at 15:15 IST
    """

    def __init__(self, strategy, strategy_name: str,
                 initial_capital: float = 100_000,
                 position_pct: float = 0.10,
                 min_confidence: float = 0.60):
        self.strategy        = strategy
        self.strategy_name   = strategy_name
        self.initial_capital = initial_capital
        self.position_pct    = position_pct   # fraction of initial_capital per trade
        self.min_confidence  = min_confidence

    def run(self, data_map: dict[str, pd.DataFrame],
            min_candles: int = 50) -> BacktestResult:
        """
        data_map: {symbol → DataFrame with [timestamp, open, high, low, close, volume]}
        Each symbol is simulated independently.
        Returns a BacktestResult with all metrics computed.
        """
        result = BacktestResult(
            strategy=self.strategy_name,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
        )

        all_trades: list[BacktestTrade] = []

        for symbol, df in data_map.items():
            if len(df) < min_candles + 10:
                logger.info(f"  Skipping {symbol}: only {len(df)} candles")
                continue
            result.symbols_tested += 1
            sym_trades = self._run_symbol(symbol, df, min_candles)
            all_trades.extend(sym_trades)

        # Sort all trades by entry time for equity curve
        all_trades.sort(key=lambda t: str(t.entry_time))
        result.trades = all_trades

        self._compute_metrics(result)
        return result

    def _run_symbol(self, symbol: str, df: pd.DataFrame,
                    min_candles: int) -> list[BacktestTrade]:
        trades: list[BacktestTrade] = []
        position: Optional[dict] = None

        # Fixed capital allocation per trade (does NOT compound within run)
        trade_capital = self.initial_capital * self.position_pct

        eod_time = pd.Timestamp("15:15", tz="Asia/Kolkata").time()

        for i in range(min_candles, len(df)):
            window  = df.iloc[:i + 1].copy()
            current = window.iloc[-1]
            ts      = current["timestamp"]
            ts_time = ts.time() if hasattr(ts, "time") else pd.Timestamp(ts).time()

            # ── EOD force-exit at 15:15 ──────────────────────────────────────
            if position and ts_time >= eod_time:
                exit_price = apply_slippage(float(current["close"]), "sell")
                t = self._close(position, symbol, exit_price, ts, "eod")
                trades.append(t)
                position = None
                continue

            # ── Check exit conditions for open position ──────────────────────
            if position:
                reason, price = self._check_exit(position, current)
                if reason:
                    t = self._close(position, symbol, price, ts, reason)
                    trades.append(t)
                    position = None

            # ── Generate new signal (BUY only, no position open) ─────────────
            if not position and ts_time < eod_time:
                try:
                    signal = self.strategy.generate_signal(symbol, window)
                except Exception:
                    signal = None

                # Only take BUY signals — no shorting in Indian equity markets
                if signal and signal.action == "BUY" and signal.confidence >= self.min_confidence:
                    entry_price = apply_slippage(float(current["close"]), "buy")
                    qty = max(1, int(trade_capital / entry_price))
                    logger.info(
                        f"  Signal: {signal.action} {symbol} @ ₹{entry_price:,.2f}"
                        f" | conf={signal.confidence:.0%}"
                    )

                    position = {
                        "action":      "BUY",
                        "entry_price": entry_price,
                        "stop_loss":   signal.stop_loss,
                        "target":      signal.target,
                        "quantity":    qty,
                        "entry_time":  ts,
                        "confidence":  signal.confidence,
                    }

        # Close any open position at end of data
        if position:
            last  = df.iloc[-1]
            price = apply_slippage(float(last["close"]),
                                   "sell" if position["action"] == "BUY" else "buy")
            t = self._close(position, symbol, price, last["timestamp"], "end_of_data")
            trades.append(t)

        return trades

    def _check_exit(self, pos: dict, candle: pd.Series) -> tuple[Optional[str], float]:
        """Only BUY (long) positions — check if low hit SL or high hit target."""
        lo  = float(candle["low"])
        hi  = float(candle["high"])
        sl  = pos["stop_loss"]
        tgt = pos["target"]

        if lo <= sl:
            return "stop_loss", apply_slippage(sl, "sell")
        if hi >= tgt:
            return "target", apply_slippage(tgt, "sell")
        return None, 0.0

    def _close(self, pos: dict, symbol: str, exit_price: float,
               exit_time, reason: str) -> BacktestTrade:
        entry  = pos["entry_price"]
        qty    = pos["quantity"]

        gross  = (exit_price - entry) * qty   # BUY only
        costs  = compute_costs(entry, exit_price, qty, "BUY")
        net    = round(gross - costs, 2)
        pct    = round(net / (entry * qty) * 100, 3) if entry * qty > 0 else 0.0

        return BacktestTrade(
            symbol      = symbol,
            strategy    = self.strategy_name,
            action      = "BUY",
            entry_price = entry,
            exit_price  = round(exit_price, 2),
            quantity    = qty,
            stop_loss   = pos["stop_loss"],
            target      = pos["target"],
            entry_time  = pos["entry_time"],
            exit_time   = exit_time,
            exit_reason = reason,
            gross_pnl   = round(gross, 2),
            costs       = round(costs, 2),
            pnl         = net,
            pnl_pct     = pct,
            confidence  = pos.get("confidence", 0.0),
        )

    def _compute_metrics(self, result: BacktestResult):
        trades = result.trades
        if not trades:
            result.final_capital = result.initial_capital
            return

        pnls   = [t.pnl for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.total_trades   = len(trades)
        result.winning_trades = len(wins)
        result.total_pnl      = round(sum(pnls), 2)
        result.total_costs    = round(sum(t.costs for t in trades), 2)
        result.win_rate       = len(wins) / len(trades) if trades else 0.0
        result.avg_win        = round(float(np.mean(wins)),   2) if wins   else 0.0
        result.avg_loss       = round(float(np.mean(losses)), 2) if losses else 0.0
        # final_capital = initial + net P&L across all symbols combined
        result.final_capital  = round(result.initial_capital + result.total_pnl, 2)
        # Note: total_pnl can exceed initial_capital when multiple symbols
        # trade in parallel (each allocated position_pct of initial_capital)

        gross_wins   = sum(wins)
        gross_losses = abs(sum(losses))
        result.profit_factor = (
            round(gross_wins / gross_losses, 3)
            if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0.0)
        )

        # R:R ratio (avg win / avg loss absolute)
        if result.avg_loss != 0:
            result.rr_ratio = round(result.avg_win / abs(result.avg_loss), 2)

        # Max drawdown (on cumulative P&L series)
        cumulative  = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative + result.initial_capital)
        equity      = cumulative + result.initial_capital
        drawdowns   = equity - running_max
        result.max_drawdown     = round(float(np.min(drawdowns)), 2)
        result.max_drawdown_pct = round(result.max_drawdown / result.initial_capital * 100, 2)

        # Sharpe (annualised, daily P&L bucketed)
        # Group P&L by date
        pnl_by_date: dict[str, float] = {}
        for t in trades:
            d = str(t.exit_time)[:10]
            pnl_by_date[d] = pnl_by_date.get(d, 0) + t.pnl
        daily_pnls = list(pnl_by_date.values())
        if len(daily_pnls) > 1:
            arr = np.array(daily_pnls)
            std = float(np.std(arr))
            mean = float(np.mean(arr))
            result.sharpe = round(mean / std * np.sqrt(252) if std > 0 else 0.0, 3)

        # CAGR
        first_entry = min(str(t.entry_time) for t in trades)
        last_exit   = max(str(t.exit_time)  for t in trades)
        try:
            td = pd.Timestamp(last_exit) - pd.Timestamp(first_entry)
            years = td.days / 365.25
            result.months_tested = round(td.days / 30.44, 1)
            if years > 0 and result.initial_capital > 0:
                ratio = result.final_capital / result.initial_capital
                result.cagr = round((ratio ** (1 / years) - 1) * 100, 2)
        except Exception:
            pass

        # Monthly returns
        monthly: dict[str, float] = {}
        for t in trades:
            month_key = str(t.exit_time)[:7]   # "YYYY-MM"
            monthly[month_key] = monthly.get(month_key, 0) + t.pnl
        result.monthly_returns = dict(sorted(monthly.items()))
        if monthly:
            best_m  = max(monthly, key=monthly.get)
            worst_m = min(monthly, key=monthly.get)
            result.best_month     = best_m
            result.worst_month    = worst_m
            result.best_month_pnl  = round(monthly[best_m],  2)
            result.worst_month_pnl = round(monthly[worst_m], 2)


# ─────────────────────────────────────────────────────────────────────────────
# Combined portfolio result
# ─────────────────────────────────────────────────────────────────────────────

def combine_results(results: list[BacktestResult],
                    initial_capital: float) -> BacktestResult:
    """Merge multiple strategy results into a single combined portfolio view."""
    combined = BacktestResult(
        strategy="Combined",
        initial_capital=initial_capital,
    )
    all_trades = []
    for r in results:
        all_trades.extend(r.trades)
        combined.symbols_tested = max(combined.symbols_tested, r.symbols_tested)

    # Re-run metric computation on pooled trades
    combined.trades = sorted(all_trades, key=lambda t: str(t.entry_time))

    # Create a temporary backtester just for metrics
    _dummy = StrategyBacktester(None, "Combined", initial_capital)
    _dummy._compute_metrics(combined)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _pf_str(pf: float) -> str:
    if pf == float("inf"):
        return "∞"
    if pf == 0.0:
        return "N/A"
    return f"{pf:.3f}"


def _na(val, fmt=".1f", suffix="") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:{fmt}}{suffix}"


def print_result(result: BacktestResult, capital: float):
    W = 54
    sep  = "─" * W
    dsep = "═" * W

    def row(label, val):
        print(f"  {label:<28}{val}")

    print(dsep)
    print(f"  Strategy: {result.strategy:<20}  Capital: ₹{capital:,.0f}")
    print(dsep)

    if result.total_trades == 0:
        print("  ⚠  No trades generated — insufficient candles or signals")
        print(dsep)
        return

    # ── Trade summary ─────────────────────────────────────────────────────────
    print(f"\n{'  Trade Summary':}")
    print(sep)
    row("Symbols tested:",    str(result.symbols_tested))
    row("Total trades:",      str(result.total_trades))
    row("Winning trades:",    f"{result.winning_trades}  ({result.win_rate:.1%})")
    row("Avg win:",           f"₹{result.avg_win:,.0f}")
    row("Avg loss:",          f"₹{result.avg_loss:,.0f}")
    row("Win/Loss R:R:",      f"1:{result.rr_ratio:.2f}" if result.rr_ratio else "N/A")
    row("Profit factor:",     _pf_str(result.profit_factor))

    # ── P&L ──────────────────────────────────────────────────────────────────
    print(f"\n  P&L Summary")
    print(sep)
    row("Total P&L (net):",   f"₹{result.total_pnl:+,.0f}")
    row("Total costs paid:",  f"₹{result.total_costs:,.0f}")
    row("Initial capital:",   f"₹{result.initial_capital:,.0f}")
    row("Final capital:",     f"₹{result.final_capital:,.0f}")
    row("Return:",            f"{(result.final_capital/result.initial_capital - 1)*100:+.2f}%")
    row("CAGR:",              f"{result.cagr:+.2f}%" if result.cagr else "N/A")
    row("Period tested:",     f"{result.months_tested:.1f} months")

    # ── Risk ──────────────────────────────────────────────────────────────────
    print(f"\n  Risk Metrics")
    print(sep)
    row("Sharpe ratio:",      f"{result.sharpe:.3f}")
    row("Max drawdown:",      f"₹{result.max_drawdown:,.0f}  ({result.max_drawdown_pct:.1f}%)")
    row("Best month:",        f"{result.best_month}  ₹{result.best_month_pnl:+,.0f}")
    row("Worst month:",       f"{result.worst_month}  ₹{result.worst_month_pnl:+,.0f}")

    # ── Monthly returns ───────────────────────────────────────────────────────
    if result.monthly_returns:
        print(f"\n  Monthly P&L")
        print(sep)
        for month, pnl in sorted(result.monthly_returns.items()):
            bar_len = int(abs(pnl) / max(abs(v) for v in result.monthly_returns.values()) * 20)
            bar = ("█" * bar_len) if pnl >= 0 else ("░" * bar_len)
            sign = "+" if pnl >= 0 else ""
            print(f"  {month}  {sign}₹{pnl:>9,.0f}  {bar}")

    # ── Exit reason breakdown ─────────────────────────────────────────────────
    if result.trades:
        reasons: dict[str, int] = {}
        for t in result.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print(f"\n  Exit Reasons")
        print(sep)
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / result.total_trades * 100
            print(f"  {reason:<18}  {count:>4}  ({pct:.0f}%)")

    print()
    print(dsep)
    print()


def print_comparison_table(results: list[BacktestResult]):
    """Side-by-side summary for quick comparison."""
    print("\n" + "═" * 80)
    print("  STRATEGY COMPARISON")
    print("═" * 80)
    header = f"  {'Strategy':<16} {'Trades':>6} {'WinRate':>8} {'NetP&L':>12} {'CAGR':>7} {'PF':>6} {'Sharpe':>7} {'MaxDD':>12}"
    print(header)
    print("  " + "─" * 76)
    for r in results:
        pf  = _pf_str(r.profit_factor)
        cagr = f"{r.cagr:+.1f}%" if r.cagr else "N/A"
        print(
            f"  {r.strategy:<16} {r.total_trades:>6} {r.win_rate:>7.1%}"
            f"  ₹{r.total_pnl:>9,.0f}  {cagr:>7} {pf:>6} {r.sharpe:>7.3f}"
            f"  ₹{r.max_drawdown:>9,.0f}"
        )
    print("═" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# CSV / JSON export helpers
# ─────────────────────────────────────────────────────────────────────────────

def results_to_csv(results: list[BacktestResult]) -> str:
    rows = []
    for r in results:
        for t in r.trades:
            rows.append({
                "strategy":    t.strategy,
                "symbol":      t.symbol,
                "action":      t.action,
                "entry_time":  str(t.entry_time)[:19],
                "exit_time":   str(t.exit_time)[:19],
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "quantity":    t.quantity,
                "stop_loss":   t.stop_loss,
                "target":      t.target,
                "exit_reason": t.exit_reason,
                "gross_pnl":   t.gross_pnl,
                "costs":       t.costs,
                "net_pnl":     t.pnl,
                "pnl_pct":     t.pnl_pct,
                "confidence":  t.confidence,
            })
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return buf.getvalue()


def results_to_json(results: list[BacktestResult]) -> str:
    out = []
    for r in results:
        out.append({
            "strategy":        r.strategy,
            "symbols_tested":  r.symbols_tested,
            "total_trades":    r.total_trades,
            "winning_trades":  r.winning_trades,
            "win_rate":        round(r.win_rate, 4),
            "total_pnl":       r.total_pnl,
            "total_costs":     r.total_costs,
            "profit_factor":   r.profit_factor if r.profit_factor != float("inf") else None,
            "avg_win":         r.avg_win,
            "avg_loss":        r.avg_loss,
            "rr_ratio":        r.rr_ratio,
            "sharpe":          r.sharpe,
            "cagr":            r.cagr,
            "max_drawdown":    r.max_drawdown,
            "max_drawdown_pct":r.max_drawdown_pct,
            "months_tested":   r.months_tested,
            "initial_capital": r.initial_capital,
            "final_capital":   r.final_capital,
            "best_month":      r.best_month,
            "best_month_pnl":  r.best_month_pnl,
            "worst_month":     r.worst_month,
            "worst_month_pnl": r.worst_month_pnl,
            "monthly_returns": r.monthly_returns,
        })
    return json.dumps(out, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AlgoTrader Comprehensive Backtest Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--months",    type=int,   default=6,
                   help="Months of history to fetch (6 or 12, default: 6)")
    p.add_argument("--strategy",  choices=["momentum", "mean_reversion", "both"],
                   default="both",
                   help="Strategy to test (default: both)")
    p.add_argument("--capital",   type=float, default=100_000,
                   help="Starting capital ₹ (default: 100000)")
    p.add_argument("--symbols",   nargs="+",  default=None,
                   help="Specific symbols. Default: full watchlist")
    p.add_argument("--min-candles", type=int, default=60,
                   help="Min candles needed (default: 60)")
    p.add_argument("--confidence",  type=float, default=0.60,
                   help="Min signal confidence (default: 0.60)")
    p.add_argument("--output",    choices=["text", "csv", "json"],
                   default="text",
                   help="Output format (default: text)")
    p.add_argument("--out-file",  default=None,
                   help="Save output to file (default: stdout)")
    p.add_argument("--no-fetch",  action="store_true",
                   help="Skip yfinance fetch — use cached parquet files if present")
    p.add_argument("--cache-dir", default="/tmp/bt_cache",
                   help="Directory for cached OHLCV parquet files")
    return p.parse_args()


def load_data(symbols: list[str], months: int,
              no_fetch: bool, cache_dir: str) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for all symbols, using cache if available."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    data_map: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        cache_file = Path(cache_dir) / f"{sym.replace('-EQ','')}_5m_{months}m.parquet"

        def _load_cache(path: Path) -> Optional[pd.DataFrame]:
            csv_path = path.with_suffix(".csv")
            if path.exists():
                try:
                    return pd.read_parquet(path)
                except Exception:
                    pass
            if csv_path.exists():
                return pd.read_csv(csv_path)
            return None

        if no_fetch:
            cached = _load_cache(cache_file)
            if cached is not None:
                logger.info(f"  {sym}: loaded {len(cached)} candles from cache")
                data_map[sym] = cached
                continue

        # Check if cache is recent (< 4 hours old)
        for ext in [".parquet", ".csv"]:
            cpath = cache_file.with_suffix(ext)
            if cpath.exists():
                age_hours = (time.time() - cpath.stat().st_mtime) / 3600
                if age_hours < 4:
                    cached = _load_cache(cache_file)
                    if cached is not None:
                        logger.info(f"  {sym}: {len(cached)} candles from cache ({age_hours:.0f}h old)")
                        data_map[sym] = cached
                        break
            continue
        else:
            pass  # proceed to fetch

        if sym in data_map:
            continue

        # Fetch fresh
        df = fetch_ohlcv(sym, months)
        if df is not None and len(df) > 0:
            try:
                df.to_parquet(cache_file, index=False)
            except ImportError:
                # pyarrow/fastparquet not installed — save as CSV instead
                cache_file = cache_file.with_suffix(".csv")
                df.to_csv(cache_file, index=False)
            data_map[sym] = df
        else:
            logger.warning(f"  {sym}: no data available — skipped")

    return data_map


def main():
    args = parse_args()
    symbols = args.symbols or WATCHLIST

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║     AlgoTrader — Comprehensive Backtest Engine       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Period:    {args.months} months")
    print(f"  Strategy:  {args.strategy}")
    print(f"  Capital:   ₹{args.capital:,.0f}")
    print(f"  Symbols:   {len(symbols)}")
    print(f"  Min conf:  {args.confidence:.0%}")
    print()
    print("  Cost model: ₹20 flat brokerage + 18% GST + STT 0.025%")
    print("              Exchange 0.00322% + Stamp 0.015% + Slippage 0.05%/side")
    print()

    # ── Fetch / load data ──────────────────────────────────────────────────────
    print("  Fetching OHLCV data from yfinance...")
    data_map = load_data(symbols, args.months, args.no_fetch, args.cache_dir)

    if not data_map:
        print("  ERROR: No data fetched. Check internet connection.")
        sys.exit(1)

    total_candles = sum(len(df) for df in data_map.values())
    print(f"  Data ready: {len(data_map)} symbols, {total_candles:,} candles total\n")

    # ── Build strategies ───────────────────────────────────────────────────────
    strategy_pairs: list[tuple[str, object]] = []
    if args.strategy in ("momentum", "both"):
        strategy_pairs.append(("Momentum", MomentumStrategy()))
    if args.strategy in ("mean_reversion", "both"):
        strategy_pairs.append(("Mean Reversion", MeanReversionStrategy()))

    all_results: list[BacktestResult] = []

    for name, strat in strategy_pairs:
        print(f"  Running {name} backtest...")
        bt = StrategyBacktester(
            strategy=strat,
            strategy_name=name,
            initial_capital=args.capital,
            min_confidence=args.confidence,
        )
        result = bt.run(data_map, min_candles=args.min_candles)
        all_results.append(result)
        logger.info(f"  {name} done: {result.total_trades} trades")

    # ── Combined ───────────────────────────────────────────────────────────────
    if len(all_results) > 1:
        combined = combine_results(all_results, args.capital)
        all_results.append(combined)

    # ── Output ─────────────────────────────────────────────────────────────────
    if args.output == "text":
        for r in all_results:
            print_result(r, args.capital)
        if len(all_results) > 1:
            print_comparison_table(all_results)
        output = None  # already printed

    elif args.output == "csv":
        output = results_to_csv([r for r in all_results if r.strategy != "Combined"])

    elif args.output == "json":
        output = results_to_json(all_results)

    if output and args.out_file:
        Path(args.out_file).write_text(output)
        print(f"\n  Saved to: {args.out_file}")
    elif output:
        print(output)


if __name__ == "__main__":
    main()
