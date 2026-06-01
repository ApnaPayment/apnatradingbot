"""
V3 Timeframe Comparison
=======================
Tests Momentum and Mean Reversion across 5m, 15m, and 30m bars.

Key V3 design decisions vs V2:
  - NO time-of-day gate (on higher timeframes the full session is valid)
  - ATR minimum filter kept (economic viability check)
  - Otherwise identical strategy logic + cost model
  - 15m/30m bars built by aggregating the cached 5m OHLCV data

Usage:
    python backtest/timeframe_compare.py
"""

from __future__ import annotations

import os
import sys
import glob
import logging
from pathlib import Path
from typing import Optional
from datetime import time as dtime

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from strategies.momentum      import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from backtest.engine import (
    StrategyBacktester, BacktestResult,
    compute_costs, apply_slippage,
    _pf_str, print_result, print_comparison_table,
)

logging.basicConfig(
    level=logging.WARNING,      # suppress per-signal noise for clean output
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("timeframe_compare")

CACHE_DIR = "/tmp/bt_cache"
SYMBOLS = [
    "RELIANCE-EQ", "TCS-EQ", "INFY-EQ", "HDFCBANK-EQ",
    "ICICIBANK-EQ", "AXISBANK-EQ", "LT-EQ",
]
INITIAL_CAPITAL = 100_000
POSITION_PCT    = 0.10
MIN_CONFIDENCE  = 0.60   # same as V1 baseline — time gate removed so we let ATR do the work
MIN_CANDLES     = 50


# ─────────────────────────────────────────────────────────────────────────────
# 5m → Nm aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Resample 5-minute OHLCV into N-minute bars.
    Input must have [timestamp, open, high, low, close, volume].
    Returns same schema. Market-hours only (09:15–15:30).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Localise to IST if not already
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    df = df.set_index("timestamp").sort_index()

    # Resample
    rule = f"{minutes}min"
    agg = df.resample(rule, closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    # Filter to market hours (first bar of the new resolution may start at 9:15)
    market_open  = dtime(9, 15)
    market_close = dtime(15, 30)
    agg = agg[(agg.index.time >= market_open) & (agg.index.time <= market_close)]

    agg = agg.reset_index().rename(columns={"timestamp": "timestamp"})
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Strategy wrappers that bypass V2 time-of-day gate
# ─────────────────────────────────────────────────────────────────────────────

class MomentumV3(MomentumStrategy):
    """
    V3 Momentum: same indicators, NO time-of-day gate.
    ATR filter and volume_factor=2.0 retained.
    On 15m/30m bars the full session has predictive value.
    """
    def generate_signal(self, symbol: str, df: pd.DataFrame,
                        exchange: str = "nse_cm") -> Optional[object]:
        # Temporarily clear timestamps to bypass the time gate in V2
        df_notimed = df.copy()
        if "timestamp" in df_notimed.columns:
            df_notimed = df_notimed.drop(columns=["timestamp"])
        return super().generate_signal(symbol, df_notimed, exchange)


class MeanReversionV3(MeanReversionStrategy):
    """
    V3 Mean Reversion: same indicators, NO time-of-day gate.
    BB width, ATR, and 2-bar RSI bottoming retained.
    Target = 2×ATR (V2 fix retained).
    """
    def generate_signal(self, symbol: str, df: pd.DataFrame,
                        exchange: str = "nse_cm") -> Optional[object]:
        df_notimed = df.copy()
        if "timestamp" in df_notimed.columns:
            df_notimed = df_notimed.drop(columns=["timestamp"])
        return super().generate_signal(symbol, df_notimed, exchange)


# ─────────────────────────────────────────────────────────────────────────────
# Backtester override: pass timestamps through for EOD logic
# ─────────────────────────────────────────────────────────────────────────────

class TimeframeBacktester(StrategyBacktester):
    """
    Inherits all cost/exit logic from StrategyBacktester.
    Overrides EOD exit time to be appropriate for each timeframe
    (last bar of session at 15:15/15:00/14:30).
    """
    def __init__(self, strategy, strategy_name: str, timeframe_min: int,
                 initial_capital: float = 100_000,
                 position_pct: float = 0.10,
                 min_confidence: float = 0.60):
        super().__init__(strategy, strategy_name, initial_capital,
                         position_pct, min_confidence)
        self.timeframe_min = timeframe_min

    def _run_symbol(self, symbol: str, df: pd.DataFrame,
                    min_candles: int) -> list:
        """
        Identical to parent but EOD exit at bar >= 15:00 IST
        (last complete 15m/30m bar before close).
        """
        from backtest.engine import BacktestTrade
        trades = []
        position = None
        trade_capital = self.initial_capital * self.position_pct

        # EOD cutoff: last 30-min bar we want to exit is 15:00 for 30m, 15:05 for 15m
        eod_cutoff = dtime(15, 0) if self.timeframe_min >= 30 else dtime(15, 10)

        for i in range(min_candles, len(df)):
            window  = df.iloc[:i + 1].copy()
            current = window.iloc[-1]
            ts      = current.get("timestamp", None)

            # Parse bar time
            bar_time = None
            if ts is not None:
                try:
                    t = pd.Timestamp(ts)
                    bar_time = t.time() if t.tzinfo is None else t.tz_convert("Asia/Kolkata").time()
                except Exception:
                    pass

            # EOD force-exit
            if position and bar_time and bar_time >= eod_cutoff:
                exit_price = apply_slippage(float(current["close"]), "sell")
                t = self._close(position, symbol, exit_price, ts, "eod")
                trades.append(t)
                position = None
                continue

            # Check exit conditions for open position
            if position:
                reason, price = self._check_exit(position, current)
                if reason:
                    t = self._close(position, symbol, price, ts, reason)
                    trades.append(t)
                    position = None

            # Generate new signal (no position, before EOD)
            no_eod = (bar_time is None) or (bar_time < eod_cutoff)
            if not position and no_eod:
                try:
                    signal = self.strategy.generate_signal(symbol, window)
                except Exception:
                    signal = None

                if signal and signal.action == "BUY" and signal.confidence >= self.min_confidence:
                    entry_price = apply_slippage(float(current["close"]), "buy")
                    qty = max(1, int(trade_capital / entry_price))
                    position = {
                        "action":      "BUY",
                        "entry_price": entry_price,
                        "stop_loss":   signal.stop_loss,
                        "target":      signal.target,
                        "quantity":    qty,
                        "entry_time":  ts,
                        "confidence":  signal.confidence,
                    }

        # Close open position at end of data
        if position:
            last  = df.iloc[-1]
            price = apply_slippage(float(last["close"]), "sell")
            t = self._close(position, symbol, price, last.get("timestamp"), "end_of_data")
            trades.append(t)

        return trades


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison runner
# ─────────────────────────────────────────────────────────────────────────────

def load_5m(symbol: str) -> Optional[pd.DataFrame]:
    key = symbol.replace("-EQ", "")
    path = Path(CACHE_DIR) / f"{key}_5m_6m.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def run_timeframe(tf_min: int, strategy_name: str, strategy,
                  data_map_5m: dict) -> BacktestResult:
    """
    Aggregate 5m data to tf_min, run strategy, return result.
    """
    data_map = {}
    for sym, df5 in data_map_5m.items():
        if tf_min == 5:
            data_map[sym] = df5.copy()
        else:
            data_map[sym] = aggregate_ohlcv(df5, tf_min)

    bt = TimeframeBacktester(
        strategy       = strategy,
        strategy_name  = strategy_name,
        timeframe_min  = tf_min,
        initial_capital= INITIAL_CAPITAL,
        position_pct   = POSITION_PCT,
        min_confidence = MIN_CONFIDENCE,
    )
    return bt.run(data_map, min_candles=MIN_CANDLES)


def summary_row(tf: str, r: BacktestResult) -> dict:
    pf = r.profit_factor
    return {
        "Timeframe":    tf,
        "Strategy":     r.strategy,
        "Trades":       r.total_trades,
        "Win%":         f"{r.win_rate:.1%}",
        "PF":           _pf_str(pf),
        "Net P&L":      f"₹{r.total_pnl:+,.0f}",
        "Costs":        f"₹{r.total_costs:,.0f}",
        "MaxDD":        f"₹{r.max_drawdown:,.0f}  ({r.max_drawdown_pct:.1f}%)",
        "CAGR":         f"{r.cagr:+.1f}%" if r.cagr else "N/A",
        "Sharpe":       f"{r.sharpe:.3f}",
        "Avg Win":      f"₹{r.avg_win:,.0f}",
        "Avg Loss":     f"₹{r.avg_loss:,.0f}",
    }


def main():
    # ── Load 5m data ─────────────────────────────────────────────────────────
    data_map_5m = {}
    for sym in SYMBOLS:
        df = load_5m(sym)
        if df is not None and len(df) >= MIN_CANDLES + 10:
            data_map_5m[sym] = df
        else:
            print(f"  WARNING: no 5m cache for {sym} — skipped")

    if not data_map_5m:
        print("ERROR: No cached 5m data found. Run engine.py first to populate /tmp/bt_cache/")
        sys.exit(1)

    total_5m = sum(len(df) for df in data_map_5m.values())
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║     AlgoTrader — V3 Timeframe Comparison             ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Symbols:   {', '.join(data_map_5m.keys())}")
    print(f"  5m candles: {total_5m:,}  →  15m: ~{total_5m//3:,}  30m: ~{total_5m//6:,}")
    print(f"  Capital:   ₹{INITIAL_CAPITAL:,.0f}  |  Position: {POSITION_PCT:.0%}  |  MinConf: {MIN_CONFIDENCE:.0%}")
    print(f"  Cost model: ₹20 brokerage + STT/GST/stamp/slippage (same as V2)")
    print()

    timeframes = [5, 15, 30]
    all_results: list[dict] = []        # for summary table
    all_result_objs: list[BacktestResult] = []  # for per-strategy detail

    for tf in timeframes:
        print(f"  ── {tf}-minute bars ──────────────────────────────────────")

        # Show bar count
        if tf > 5:
            sample = aggregate_ohlcv(list(data_map_5m.values())[0], tf)
            bars_per_sym = len(sample)
            print(f"     Example bars ({list(data_map_5m.keys())[0]}): {bars_per_sym}")
        print()

        for strat_name, strat_cls in [("Momentum", MomentumV3), ("Mean Reversion", MeanReversionV3)]:
            label = f"{strat_name} {tf}m"
            strat = strat_cls()
            result = run_timeframe(tf, label, strat, data_map_5m)
            all_result_objs.append(result)
            row = summary_row(f"{tf}m", result)
            all_results.append(row)
            print(f"     {label:<22}  "
                  f"{result.total_trades:>4}T  "
                  f"WR:{result.win_rate:.0%}  "
                  f"PF:{_pf_str(result.profit_factor):>6}  "
                  f"P&L:₹{result.total_pnl:+,.0f}  "
                  f"DD:₹{result.max_drawdown:,.0f}")

        print()

    # ── Detailed report per strategy ─────────────────────────────────────────
    for r in all_result_objs:
        print_result(r, INITIAL_CAPITAL)

    # ── Master comparison table ───────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════════════════════════╗")
    print("║  TIMEFRAME COMPARISON — 5m vs 15m vs 30m                                       ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════╝")
    print()

    # Separate Momentum and MR tables
    for strat_label in ["Momentum", "Mean Reversion"]:
        rows = [r for r in all_results if strat_label in r["Strategy"]]
        print(f"  {strat_label}")
        print(f"  {'TF':>4}  {'Trades':>6}  {'Win%':>6}  {'PF':>6}  {'Net P&L':>12}  {'Costs':>9}  {'AvgWin':>8}  {'AvgLoss':>9}  {'MaxDD':>20}")
        print("  " + "─" * 84)
        for r in rows:
            print(f"  {r['Timeframe']:>4}  {r['Trades']:>6}  {r['Win%']:>6}  {r['PF']:>6}  "
                  f"{r['Net P&L']:>12}  {r['Costs']:>9}  {r['Avg Win']:>8}  {r['Avg Loss']:>9}  "
                  f"{r['MaxDD']:>20}")
        print()

    # ── Expectancy per trade ──────────────────────────────────────────────────
    print("  EXPECTED VALUE PER TRADE (net, after all costs)")
    print("  ─────────────────────────────────────────────────")
    for r_obj in all_result_objs:
        if r_obj.total_trades > 0:
            ev = r_obj.total_pnl / r_obj.total_trades
        else:
            ev = 0
        verdict = "✓ POSITIVE" if ev > 0 else "✗ negative"
        print(f"  {r_obj.strategy:<28}  EV/trade: ₹{ev:+,.1f}  {verdict}")

    print()
    print("  NOTE: Test period = Apr–May 2026 only (~1.6 months).")
    print("  Longer test period would require paid historical 5m data.")
    print("  Directional signal: higher timeframe → fewer trades, lower cost drag.")

if __name__ == "__main__":
    main()
