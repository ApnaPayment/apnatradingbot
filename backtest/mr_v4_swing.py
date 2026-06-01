"""
Mean Reversion V4 — Swing Trading (Multi-Day Hold)
====================================================

Strategy family: Swing Mean Reversion (separate from V1/V2/V3 intraday).

Architecture:
  Tier 1 — Daily trend gate
    daily_close (yesterday) > 10-day SMA of daily closes
    Blocks genuine falling-knife downtrends.

  Tier 2 — 30m oversold setup
    bb_pct  < 0.25   (price near lower Bollinger Band)
    RSI     < 45     (meaningfully oversold on 30m bars)
    BB_width > 1.0%  (bands genuinely wide — not flat market)
    ATR     > 0.25%  (cost viability — ₹52 breakeven check)
    All 4 required.

  Tier 3 — Entry trigger (same 30m bar)
    RSI_now > RSI_prev   (single-bar RSI turn — momentum reversing)
    vol_ratio > 1.2×     (light institutional participation)
    Both required.

  Position management — NO forced EOD exit
    Positions held overnight across sessions.
    Max hold: 5 trading days from entry day.
    Overnight gap handled: if open price gaps below SL → exit at open.

  Exit logic — Trailing ATR stop, no fixed target
    Initial SL  = entry - 1.5 × ATR
    Phase 1:    when high_since_entry ≥ entry + 1×ATR
                → SL moves to entry (breakeven)
    Phase 2:    when high_since_entry ≥ entry + 2×ATR
                → SL trails at (all_time_high - 1×ATR), moves up only
    Day-5 EOD:  force-exit on close of 5th trading day (5-day max hold)

Cost model: identical to V1/V2/V3 — ₹20 brokerage, STT, GST, exchange, stamp, 0.05% slippage.

Usage:
    python backtest/mr_v4_swing.py
"""

from __future__ import annotations

import sys
import logging
import datetime
from pathlib import Path
from typing import Optional
from datetime import time as dtime
from dataclasses import dataclass

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtest.timeframe_compare import aggregate_ohlcv, load_5m, SYMBOLS, INITIAL_CAPITAL
from backtest.engine import compute_costs, apply_slippage, _pf_str

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

POSITION_PCT  = 0.10
MIN_CANDLES   = 30        # 30× 30m bars ≈ 5 days warmup
MAX_HOLD_DAYS = 5         # force-exit after 5 trading days
EOD_TIME      = dtime(15, 15)  # last bar of NSE session

# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Bollinger Bands (20, 2σ)
    df["bb_mid"]   = df["close"].rolling(20).mean()
    bb_std         = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2.0 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2.0 * bb_std
    band_range     = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / band_range
    df["bb_width"] = band_range / df["bb_mid"]

    # RSI (14)
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR (14)
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # Volume MA (20)
    df["vol_ma"]    = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)

    return df


def build_daily_sma(df30: pd.DataFrame, window: int = 10) -> dict:
    """
    From 30m bars, build a dict of date → (daily_close, 10-day SMA).
    Only dates where the SMA is computable (min 5 days of data) are included.
    """
    df = df30.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    df["date"] = df["timestamp"].dt.date

    daily = df.groupby("date")["close"].last().reset_index()
    daily.columns = ["date", "close"]
    daily["sma"] = daily["close"].rolling(window, min_periods=5).mean()

    result = {}
    for _, row in daily.iterrows():
        if not pd.isna(row["sma"]):
            result[row["date"]] = (float(row["close"]), float(row["sma"]))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SwingTrade:
    symbol:         str
    entry_price:    float
    exit_price:     float
    quantity:       int
    entry_time:     object
    exit_time:      object
    exit_reason:    str
    # "sl_initial" | "sl_breakeven" | "sl_trailing" | "gap_down_sl"
    # | "max_hold" | "end_of_data"
    hold_days:      int          # trading days held
    atr_entry:      float
    initial_sl:     float
    final_sl:       float
    peak_price:     float        # highest close seen during hold
    gross_pnl:      float
    costs:          float
    pnl:            float        # net
    pnl_pct:        float


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol walk-forward simulator
# ─────────────────────────────────────────────────────────────────────────────

def run_symbol(symbol: str, df30: pd.DataFrame, daily_sma: dict) -> list[SwingTrade]:
    df = add_indicators(df30.copy())
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    trades:   list[SwingTrade] = []
    position: Optional[dict]   = None
    trade_capital = INITIAL_CAPITAL * POSITION_PCT

    # Track which distinct trading dates we've seen (for max-hold counting)
    seen_dates: list[datetime.date] = []

    for i in range(MIN_CANDLES, len(df)):
        row    = df.iloc[i]
        prev   = df.iloc[i - 1]

        ts       = row["timestamp"]
        bar_time = ts.time()
        bar_date = ts.date()

        # Accumulate unique trading dates (for hold-day counting)
        if not seen_dates or seen_dates[-1] != bar_date:
            seen_dates.append(bar_date)

        # ── Manage open position ──────────────────────────────────────────────
        if position:
            lo    = float(row["low"])
            hi    = float(row["high"])
            op    = float(row["open"])
            cl    = float(row["close"])
            sl    = position["sl"]
            atr   = position["atr"]
            ep    = position["entry_price"]
            phase = position["sl_phase"]

            # Update peak (use both intrabar high and close)
            position["peak"] = max(position["peak"], hi)
            unrealised = position["peak"] - ep

            # ── Overnight gap-down: open gaps below SL ────────────────────
            is_new_day = (position.get("last_bar_date") != bar_date)
            if is_new_day and op < sl:
                exit_px = apply_slippage(op, "sell")
                trades.append(_close(position, symbol, exit_px, ts,
                                     "gap_down_sl", seen_dates))
                position = None
                continue
            position["last_bar_date"] = bar_date

            # ── Advance trailing stop ─────────────────────────────────────
            if unrealised >= 2.0 * atr:
                new_sl = position["peak"] - 1.0 * atr
                if new_sl > sl:
                    position["sl"] = new_sl
                    position["sl_phase"] = "trailing"
            elif unrealised >= 1.0 * atr and phase == "initial":
                new_sl = max(sl, ep)
                if new_sl > sl:
                    position["sl"] = new_sl
                    position["sl_phase"] = "breakeven"

            sl = position["sl"]

            # ── Intrabar SL hit ───────────────────────────────────────────
            if lo <= sl:
                exit_px = apply_slippage(sl, "sell")
                trades.append(_close(position, symbol, exit_px, ts,
                                     f"sl_{position['sl_phase']}", seen_dates))
                position = None
                continue

            # ── Max hold: 5 trading days ──────────────────────────────────
            entry_date = position["entry_date"]
            days_held  = _count_trading_days(entry_date, bar_date, seen_dates)
            if days_held >= MAX_HOLD_DAYS and bar_time >= EOD_TIME:
                exit_px = apply_slippage(cl, "sell")
                trades.append(_close(position, symbol, exit_px, ts,
                                     "max_hold", seen_dates))
                position = None
                continue

        # ── Entry logic ───────────────────────────────────────────────────────
        if position:
            continue

        # Only generate signals on 30m bars (skip final partial bars near close)
        if bar_time >= dtime(14, 45):
            continue

        # ── Tier 1: Daily trend — block only genuine downtrends ────────────
        trend_ok = True
        for lookback in range(1, 8):
            check_date = bar_date - datetime.timedelta(days=lookback)
            if check_date in daily_sma:
                daily_close, sma_val = daily_sma[check_date]
                # Block if daily close is more than 3% below 10-day SMA
                trend_ok = daily_close >= sma_val * 0.97
                break
        if not trend_ok:
            continue

        # ── Tier 2: 30m oversold setup ────────────────────────────────────
        req = ["bb_pct", "bb_width", "rsi", "atr"]
        if any(pd.isna(row[c]) for c in req):
            continue

        bb_pct   = float(row["bb_pct"])
        bb_width = float(row["bb_width"])
        rsi_now  = float(row["rsi"])
        atr      = float(row["atr"])
        price    = float(row["close"])

        setup_ok = (
            bb_pct   < 0.25   and
            rsi_now  < 45     and
            bb_width > 0.010  and
            atr / price > 0.0025
        )
        if not setup_ok:
            continue

        # ── Tier 3: Entry trigger ─────────────────────────────────────────
        rsi_prev  = float(prev["rsi"]) if not pd.isna(prev["rsi"]) else rsi_now
        vol_ratio = float(row["vol_ratio"]) if not pd.isna(row["vol_ratio"]) else 0

        trigger_ok = (rsi_now > rsi_prev and vol_ratio > 1.2)
        if not trigger_ok:
            continue

        # ── Open position ─────────────────────────────────────────────────
        entry_px = apply_slippage(price, "buy")
        qty      = max(1, int(trade_capital / entry_px))
        sl0      = round(entry_px - 1.5 * atr, 2)

        position = {
            "entry_price":    entry_px,
            "initial_sl":     sl0,
            "sl":             sl0,
            "sl_phase":       "initial",
            "atr":            atr,
            "quantity":       qty,
            "entry_time":     ts,
            "entry_date":     bar_date,
            "last_bar_date":  bar_date,
            "peak":           float(row["high"]),
        }

    # ── End of data: close any open position ─────────────────────────────────
    if position:
        last    = df.iloc[-1]
        exit_px = apply_slippage(float(last["close"]), "sell")
        trades.append(_close(position, symbol, exit_px,
                             last["timestamp"], "end_of_data", seen_dates))

    return trades


def _count_trading_days(entry_date: datetime.date, current_date: datetime.date,
                         seen_dates: list) -> int:
    """Count distinct trading days between entry_date and current_date (inclusive)."""
    return sum(1 for d in seen_dates if entry_date <= d <= current_date) - 1


def _close(pos: dict, symbol: str, exit_px: float, exit_time,
           reason: str, seen_dates: list) -> SwingTrade:
    entry = pos["entry_price"]
    qty   = pos["quantity"]
    atr   = pos["atr"]
    gross = (exit_px - entry) * qty
    costs = compute_costs(entry, exit_px, qty, "BUY")
    net   = round(gross - costs, 2)
    pct   = round(net / (entry * qty) * 100, 3)

    exit_date = pd.Timestamp(exit_time).date() if exit_time else pos["entry_date"]
    hold_days = _count_trading_days(pos["entry_date"], exit_date, seen_dates)

    return SwingTrade(
        symbol      = symbol,
        entry_price = entry,
        exit_price  = round(exit_px, 2),
        quantity    = qty,
        entry_time  = pos["entry_time"],
        exit_time   = exit_time,
        exit_reason = reason,
        hold_days   = max(0, hold_days),
        atr_entry   = atr,
        initial_sl  = pos["initial_sl"],
        final_sl    = pos["sl"],
        peak_price  = pos["peak"],
        gross_pnl   = round(gross, 2),
        costs       = round(costs, 2),
        pnl         = net,
        pnl_pct     = pct,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[SwingTrade], label: str = "") -> dict:
    pnls   = [t.pnl for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n      = len(trades)

    wr  = len(wins) / n if n else 0
    aw  = float(np.mean(wins))   if wins   else 0.0
    al  = float(np.mean(losses)) if losses else 0.0
    gw  = sum(wins)
    gl  = abs(sum(losses))
    pf  = round(gw / gl, 3) if gl > 0 else (float("inf") if gw > 0 else 0.0)
    ev  = sum(pnls) / n if n else 0
    tot = round(sum(pnls), 2)
    tc  = round(sum(t.costs for t in trades), 2)

    # Max drawdown on cumulative P&L curve
    cumulative  = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative + INITIAL_CAPITAL)
    equity      = cumulative + INITIAL_CAPITAL
    drawdowns   = equity - running_max
    mdd         = round(float(np.min(drawdowns)), 2) if n else 0.0
    mdd_pct     = round(mdd / INITIAL_CAPITAL * 100, 2)

    # Avg hold duration
    avg_hold = float(np.mean([t.hold_days for t in trades])) if n else 0.0

    # Monthly P&L
    monthly: dict[str, float] = {}
    for t in trades:
        m = str(t.exit_time)[:7]
        monthly[m] = monthly.get(m, 0) + t.pnl
    monthly = dict(sorted(monthly.items()))

    # Exit reason breakdown
    reasons: dict[str, list] = {}
    for t in trades:
        reasons.setdefault(t.exit_reason, []).append(t.pnl)

    return dict(
        label=label, n=n, wr=wr, aw=aw, al=al, pf=pf, ev=ev,
        total_pnl=tot, total_costs=tc, mdd=mdd, mdd_pct=mdd_pct,
        monthly=monthly, wins=len(wins), losses=n - len(wins),
        avg_hold=avg_hold, reasons=reasons,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_report(m: dict, trades: list[SwingTrade]):
    W    = 58
    sep  = "─" * W
    dsep = "═" * W

    pf_verdict  = "✅ PF > 1.0" if m["pf"] > 1.0 else ("🟡 PF > 0.5" if m["pf"] > 0.5 else "❌ PF < 0.5")
    ev_verdict  = "✅ POSITIVE EV" if m["ev"] > 0 else "❌ negative ev"

    print(dsep)
    print(f"  {m['label']}")
    print(dsep)

    print(f"\n  Trade Stats")
    print(sep)
    print(f"  {'Total trades:':<30}{m['n']}")
    print(f"  {'Wins / Losses:':<30}{m['wins']} / {m['losses']}")
    print(f"  {'Win rate:':<30}{m['wr']:.1%}")
    print(f"  {'Avg win (net):':<30}₹{m['aw']:,.0f}")
    print(f"  {'Avg loss (net):':<30}₹{m['al']:,.0f}")
    if m['aw'] and m['al']:
        rr = abs(m['al'] / m['aw'])
        print(f"  {'R:R (risk/reward):':<30}1:{rr:.2f}  (need WR>{1/(1+1/rr):.0%} to break even)")
    print(f"  {'Profit factor:':<30}{_pf_str(m['pf'])}  {pf_verdict}")
    print(f"  {'EV per trade:':<30}₹{m['ev']:+,.0f}  {ev_verdict}")
    print(f"  {'Avg hold (trading days):':<30}{m['avg_hold']:.1f}")

    print(f"\n  P&L")
    print(sep)
    print(f"  {'Total net P&L:':<30}₹{m['total_pnl']:+,.0f}")
    print(f"  {'Total costs paid:':<30}₹{m['total_costs']:,.0f}")
    if m['n']:
        print(f"  {'Cost per trade:':<30}₹{m['total_costs']/m['n']:,.0f}")
    print(f"  {'Return on capital:':<30}{(m['total_pnl']/INITIAL_CAPITAL*100):+.2f}%")

    print(f"\n  Risk")
    print(sep)
    print(f"  {'Max drawdown:':<30}₹{m['mdd']:,.0f}  ({m['mdd_pct']:.1f}%)")

    if m["monthly"]:
        print(f"\n  Monthly P&L")
        print(sep)
        max_abs = max(abs(v) for v in m["monthly"].values()) or 1
        for month, pnl in m["monthly"].items():
            bar_len = int(abs(pnl) / max_abs * 22)
            bar  = "█" * bar_len if pnl >= 0 else "░" * bar_len
            sign = "+" if pnl >= 0 else ""
            print(f"  {month}  {sign}₹{pnl:>9,.0f}  {bar}")

    print(f"\n  Exit Reasons")
    print(sep)
    print(f"  {'Reason':<22} {'Count':>5}  {'WR':>5}  {'AvgNet':>9}  {'Total':>10}")
    for r, plist in sorted(m["reasons"].items(), key=lambda x: -len(x[1])):
        cnt  = len(plist)
        wr_r = sum(1 for p in plist if p > 0) / cnt
        avg  = float(np.mean(plist))
        tot  = sum(plist)
        print(f"  {r:<22} {cnt:>5}  {wr_r:>4.0%}  ₹{avg:>7,.0f}  ₹{tot:>8,.0f}")

    # Hold-day histogram
    if trades:
        print(f"\n  Hold Duration Distribution")
        print(sep)
        hold_counts: dict[int, int] = {}
        for t in trades:
            hold_counts[t.hold_days] = hold_counts.get(t.hold_days, 0) + 1
        for days in sorted(hold_counts):
            bar = "█" * hold_counts[days]
            wr_day = sum(1 for t in trades if t.hold_days == days and t.pnl > 0)
            label = f"Day {days}" if days > 0 else "Same day"
            print(f"  {label:<10}  {hold_counts[days]:>3}  {bar}  "
                  f"WR:{wr_day/hold_counts[days]:.0%}")

    # Trailing stop analysis
    phase2 = [t for t in trades if t.exit_reason == "sl_trailing"]
    if trades:
        avg_peak_gain = float(np.mean([(t.peak_price - t.entry_price) * t.quantity
                                        for t in trades]))
        print(f"\n  Trailing Stop Analysis")
        print(sep)
        print(f"  Trades reaching trailing phase:  {len(phase2)}")
        print(f"  Avg peak unrealised (₹ gross):   ₹{avg_peak_gain:,.0f}")
        if phase2:
            print(f"  Avg pnl when trailing SL hit:    ₹{float(np.mean([t.pnl for t in phase2])):,.0f}")

    print()


def print_comparison(results: list[dict]):
    """Print a summary comparison table for all versions."""
    print("═" * 90)
    print("  MEAN REVERSION — ALL VERSIONS COMPARED")
    print("═" * 90)
    hdr = (f"  {'Version':<28} {'TF':>4} {'Trades':>6} {'WR':>6} "
           f"{'PF':>6} {'EV/T':>8} {'AvgWin':>8} {'AvgLoss':>9} "
           f"{'AvgHold':>8} {'Net P&L':>10} {'MaxDD':>9}")
    print(hdr)
    print("  " + "─" * 88)
    for r in results:
        hold = f"{r.get('avg_hold', 0):.1f}d"
        print(
            f"  {r['label']:<28} {r.get('tf',''):>4} {r['n']:>6} {r['wr']:>5.0%}"
            f"  {_pf_str(r['pf']):>6} {r['ev']:>+7.0f}"
            f"  ₹{r.get('aw',0):>6,.0f}  ₹{r.get('al',0):>7,.0f}"
            f"  {hold:>8}"
            f"  {r.get('total_pnl',0):>+9,.0f}"
            f"  {r.get('mdd',0):>8,.0f}"
        )
    print("═" * 90)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Mean Reversion V4 — Swing Trading (Multi-Day Hold)         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  Strategy:")
    print("    Tier 1 — Daily trend gate:  daily close ≥ 97% of 10-day SMA")
    print("    Tier 2 — 30m setup:         BB%<0.25 + RSI<45 + BBW>1% + ATR>0.25%")
    print("    Tier 3 — Entry trigger:     RSI turning up + vol > 1.2×")
    print("    Exit   — Trailing ATR stop: initial → breakeven → trailing")
    print("    Max hold: 5 trading days.  No forced EOD exit.")
    print()

    # ── Load & aggregate data ─────────────────────────────────────────────────
    all_trades: list[SwingTrade] = []
    skipped = []

    for sym in SYMBOLS:
        df5 = load_5m(sym)
        if df5 is None or len(df5) < 100:
            skipped.append(sym)
            continue

        df30      = aggregate_ohlcv(df5, 30)
        daily_sma = build_daily_sma(df30)

        sym_trades = run_symbol(sym, df30, daily_sma)
        all_trades.extend(sym_trades)

        n   = len(sym_trades)
        w   = sum(1 for t in sym_trades if t.pnl > 0)
        wr  = w / n if n else 0
        pnl = sum(t.pnl for t in sym_trades)
        ah  = float(np.mean([t.hold_days for t in sym_trades])) if sym_trades else 0
        print(f"  {sym:<18}  {n:>3}T  WR:{wr:.0%}  AvgHold:{ah:.1f}d  P&L:₹{pnl:+,.0f}")

    if skipped:
        print(f"\n  Skipped (no data): {', '.join(skipped)}")

    if not all_trades:
        print("\n  ERROR: No trades generated. Check data and filters.")
        return

    print()

    # ── Compute & print detailed report ──────────────────────────────────────
    m = compute_metrics(all_trades, "MR V4 — Swing (5-day max hold)")
    print_report(m, all_trades)

    # ── Comparison table (V2-30m and V3-30m hardcoded from prior runs) ────────
    prior = [
        dict(label="V2 30m (best intraday)",  tf="30m", n=68,  wr=0.250,
             pf=0.144, ev=-54.2, aw=87.0,  al=-118.0, avg_hold=0.0,
             total_pnl=-3687, mdd=-3614),
        dict(label="V3 30m (daily+trail EOD)", tf="30m", n=27,  wr=0.074,
             pf=0.033, ev=-64.4, aw=29.9, al=-72.0,  avg_hold=0.0,
             total_pnl=-1740,  mdd=-1692),
    ]
    m_row = dict(
        label    = "V4 Swing (this run)",
        tf       = "30m+",
        n        = m["n"],
        wr       = m["wr"],
        pf       = m["pf"],
        ev       = m["ev"],
        aw       = m["aw"],
        al       = m["al"],
        avg_hold = m["avg_hold"],
        total_pnl= m["total_pnl"],
        mdd      = m["mdd"],
    )
    print_comparison(prior + [m_row])

    # ── Breakeven analysis ────────────────────────────────────────────────────
    print()
    print("  BREAKEVEN ANALYSIS")
    print("─" * 58)
    print(f"  Current EV/trade:   ₹{m['ev']:+,.0f}")
    if m["aw"] and m["al"]:
        be_wr  = -m["al"] / (m["aw"] - m["al"])
        be_aw  = -m["al"] * (1 - m["wr"]) / m["wr"] if m["wr"] > 0 else 0
        print(f"  Breakeven WR:       {be_wr:.1%}  (current: {m['wr']:.1%})")
        print(f"  OR: need avg win =  ₹{be_aw:,.0f} at current WR {m['wr']:.0%}  (now ₹{m['aw']:,.0f})")
        if m["ev"] < 0:
            gap = be_wr - m["wr"]
            print(f"  Gap to breakeven:   {gap:+.1%} WR  or  ₹{be_aw - m['aw']:+,.0f} avg win")

    # ── Swing-specific analysis ───────────────────────────────────────────────
    if all_trades:
        print()
        print("  SWING-SPECIFIC ANALYSIS")
        print("─" * 58)
        multi_day = [t for t in all_trades if t.hold_days > 0]
        same_day  = [t for t in all_trades if t.hold_days == 0]
        if multi_day:
            md_wr  = sum(1 for t in multi_day if t.pnl > 0) / len(multi_day)
            md_avg = float(np.mean([t.pnl for t in multi_day]))
            print(f"  Multi-day trades:   {len(multi_day)}  WR:{md_wr:.0%}  AvgPnL:₹{md_avg:+,.0f}")
        if same_day:
            sd_wr  = sum(1 for t in same_day if t.pnl > 0) / len(same_day)
            sd_avg = float(np.mean([t.pnl for t in same_day]))
            print(f"  Same-day exits:     {len(same_day)}  WR:{sd_wr:.0%}  AvgPnL:₹{sd_avg:+,.0f}")

        # Gap-down risk
        gap_exits = [t for t in all_trades if t.exit_reason == "gap_down_sl"]
        if gap_exits:
            avg_gap_loss = float(np.mean([t.pnl for t in gap_exits]))
            print(f"  Overnight gap-down exits: {len(gap_exits)}  AvgLoss:₹{avg_gap_loss:,.0f}")
        else:
            print(f"  Overnight gap-down exits: 0  (no overnight gaps triggered SL)")

        # Max-hold exits
        maxh = [t for t in all_trades if t.exit_reason == "max_hold"]
        if maxh:
            mh_wr  = sum(1 for t in maxh if t.pnl > 0) / len(maxh)
            mh_avg = float(np.mean([t.pnl for t in maxh]))
            print(f"  Max-hold (5d) exits:      {len(maxh)}  WR:{mh_wr:.0%}  AvgPnL:₹{mh_avg:+,.0f}")

    print()


if __name__ == "__main__":
    main()
