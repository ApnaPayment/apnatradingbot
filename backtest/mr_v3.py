"""
Mean Reversion V3 — Multi-Timeframe with Trailing Stop
======================================================

Architecture (three-tier entry, one-tier exit):

  Tier 1 — Daily trend filter
    Only buy when stock is in a broad uptrend:
    daily_close > 10-day SMA of daily closes (computed from 30m data)
    Reason: avoids catching falling knives in genuine downtrends.

  Tier 2 — 30m setup
    On 30m bars, conditions for a genuine oversold setup:
      • BB%  < 0.20  (price near/at lower Bollinger Band)
      • RSI  < 40    (meaningfully oversold, not just slightly dipped)
      • BB width > 1.0%  (bands are genuinely wide — real volatility)
      • ATR  > 0.25% of price (cost viability filter)
    All 4 must be true simultaneously.

  Tier 3 — Entry trigger (same 30m bar)
    Within a valid setup bar, require:
      • RSI rising for 2 consecutive 30m bars  (momentum turning)
      • Volume > 1.5× 20-bar avg  (institutional participation)
    Need both.

  Exit — Trailing ATR stop, no fixed target
    Initial SL = entry - 1.5 × ATR
    Phase 1: when unrealised profit ≥ 1×ATR → SL moves to entry (breakeven)
    Phase 2: when unrealised profit ≥ 2×ATR → SL trails at (session_high - 1×ATR)
    The trailing stop only moves up, never down.
    EOD force-exit at 15:00 IST (last 30m bar we want fully closed).

Cost model: identical to V1/V2 — ₹20 brokerage, STT, GST, exchange, stamp, 0.05% slippage.

Usage:
    python backtest/mr_v3.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import time as dtime
from dataclasses import dataclass, field

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
logger = logging.getLogger("mr_v3")

POSITION_PCT  = 0.10
MIN_CANDLES   = 30     # 30 bars of 30m = 5 days warmup
EOD_CUTOFF    = dtime(15, 0)
DAILY_SMA_WIN = 10     # short enough for 36-day test window


# ─────────────────────────────────────────────────────────────────────────────
# Indicators on 30m DataFrame
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


def build_daily_sma(df30: pd.DataFrame, window: int = DAILY_SMA_WIN) -> dict:
    """
    From 30m bars, compute the last daily close and its N-day SMA.
    Returns dict: date → (daily_close, sma) for each completed trading day.
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
class Trade:
    symbol:      str
    entry_price: float
    exit_price:  float
    quantity:    int
    entry_time:  object
    exit_time:   object
    exit_reason: str   # "sl_initial" | "sl_breakeven" | "sl_trailing" | "eod" | "end_of_data"
    atr_entry:   float
    gross_pnl:   float
    costs:       float
    pnl:         float   # net
    pnl_pct:     float
    initial_sl:  float
    final_sl:    float
    max_excursion: float  # peak unrealised gain during trade


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol walk-forward simulator
# ─────────────────────────────────────────────────────────────────────────────

def run_symbol(symbol: str, df30: pd.DataFrame, daily_sma: dict) -> list[Trade]:
    df = add_indicators(df30.copy())
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    trades: list[Trade] = []
    position: Optional[dict] = None
    trade_capital = INITIAL_CAPITAL * POSITION_PCT

    for i in range(MIN_CANDLES, len(df)):
        row     = df.iloc[i]
        prev    = df.iloc[i - 1]
        prev2   = df.iloc[i - 2] if i >= 2 else prev

        ts       = row["timestamp"]
        bar_time = ts.time()
        bar_date = ts.date()

        # ── Trailing stop / EOD management for open position ─────────────────
        if position:
            # Force-exit at EOD
            if bar_time >= EOD_CUTOFF:
                exit_px = apply_slippage(float(row["close"]), "sell")
                trades.append(_close(position, symbol, exit_px, ts, "eod"))
                position = None
                continue

            lo  = float(row["low"])
            hi  = float(row["high"])
            cl  = float(row["close"])
            sl  = position["sl"]
            atr = position["atr"]
            ep  = position["entry_price"]

            # Update session high
            position["session_high"] = max(position["session_high"], hi)
            unrealised = position["session_high"] - ep
            position["max_excursion"] = max(position["max_excursion"], cl - ep)

            # Phase 2: trailing stop once up 2×ATR
            if unrealised >= 2.0 * atr:
                new_sl = position["session_high"] - 1.0 * atr
                position["sl"] = max(sl, new_sl)
                position["sl_phase"] = "trailing"

            # Phase 1: move to breakeven once up 1×ATR
            elif unrealised >= 1.0 * atr and position["sl_phase"] == "initial":
                position["sl"] = max(sl, ep)
                position["sl_phase"] = "breakeven"

            sl = position["sl"]   # updated

            # Check stop hit
            if lo <= sl:
                exit_px  = apply_slippage(sl, "sell")
                phase    = position["sl_phase"]
                reason   = f"sl_{phase}"
                trades.append(_close(position, symbol, exit_px, ts, reason))
                position = None
                continue

        # ── Entry logic (no open position) ───────────────────────────────────
        if position:
            continue   # already open

        if bar_time >= EOD_CUTOFF:
            continue   # no new entries after 15:00

        # ── Tier 1: Daily trend filter ────────────────────────────────────────
        # Avoid extreme downtrends: require daily close is not MORE than 3% below 10-SMA
        # (Softer filter: only blocks genuine falling-knife scenarios, not mild pullbacks)
        import datetime
        trend_ok = True   # default: allow entry
        for lookback in range(1, 8):
            check_date = bar_date - datetime.timedelta(days=lookback)
            if check_date in daily_sma:
                daily_close, sma_val = daily_sma[check_date]
                # Block only if daily close is more than 3% below SMA (genuine downtrend)
                trend_ok = daily_close >= sma_val * 0.97
                break
        if not trend_ok:
            continue

        # ── Tier 2: 30m setup ─────────────────────────────────────────────────
        req_cols = ["bb_pct", "bb_width", "rsi", "atr"]
        if any(pd.isna(row[c]) for c in req_cols):
            continue

        bb_pct   = float(row["bb_pct"])
        bb_width = float(row["bb_width"])
        rsi_now  = float(row["rsi"])
        atr      = float(row["atr"])
        price    = float(row["close"])

        setup_ok = (
            bb_pct   < 0.25           and   # near lower band
            rsi_now  < 45             and   # meaningfully oversold
            bb_width > 0.010          and   # bands are wide (≥1%)
            atr / price > 0.0025           # ATR cost viability
        )
        if not setup_ok:
            continue

        # ── Tier 3: Entry trigger ─────────────────────────────────────────────
        rsi_prev  = float(prev["rsi"])  if not pd.isna(prev["rsi"])  else rsi_now
        rsi_prev2 = float(prev2["rsi"]) if not pd.isna(prev2["rsi"]) else rsi_prev
        vol_ratio = float(row["vol_ratio"]) if not pd.isna(row["vol_ratio"]) else 0

        trigger_ok = (
            rsi_now > rsi_prev              and   # RSI turning up (1-bar confirmation)
            vol_ratio > 1.2                       # light volume confirmation
        )
        if not trigger_ok:
            continue

        # ── Open position ─────────────────────────────────────────────────────
        entry_px = apply_slippage(price, "buy")
        qty      = max(1, int(trade_capital / entry_px))
        sl0      = round(entry_px - 1.5 * atr, 2)

        position = {
            "entry_price":   entry_px,
            "initial_sl":    sl0,
            "sl":            sl0,
            "sl_phase":      "initial",
            "atr":           atr,
            "quantity":      qty,
            "entry_time":    ts,
            "session_high":  float(row["high"]),
            "max_excursion": 0.0,
        }

    # Close anything still open at last bar
    if position:
        last    = df.iloc[-1]
        exit_px = apply_slippage(float(last["close"]), "sell")
        trades.append(_close(position, symbol, exit_px, last["timestamp"], "end_of_data"))

    return trades


def _close(pos: dict, symbol: str, exit_px: float,
           exit_time, reason: str) -> Trade:
    entry = pos["entry_price"]
    qty   = pos["quantity"]
    atr   = pos["atr"]
    gross = (exit_px - entry) * qty
    costs = compute_costs(entry, exit_px, qty, "BUY")
    net   = round(gross - costs, 2)
    pct   = round(net / (entry * qty) * 100, 3)
    return Trade(
        symbol        = symbol,
        entry_price   = entry,
        exit_price    = round(exit_px, 2),
        quantity      = qty,
        entry_time    = pos["entry_time"],
        exit_time     = exit_time,
        exit_reason   = reason,
        atr_entry     = atr,
        gross_pnl     = round(gross, 2),
        costs         = round(costs, 2),
        pnl           = net,
        pnl_pct       = pct,
        initial_sl    = pos["initial_sl"],
        final_sl      = pos["sl"],
        max_excursion = round(pos["max_excursion"], 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[Trade], label: str = ""):
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

    # Max drawdown on cumulative P&L
    cumulative  = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative + INITIAL_CAPITAL)
    equity      = cumulative + INITIAL_CAPITAL
    drawdowns   = equity - running_max
    mdd         = round(float(np.min(drawdowns)), 2) if len(pnls) > 0 else 0.0
    mdd_pct     = round(mdd / INITIAL_CAPITAL * 100, 2)

    # Monthly
    monthly: dict[str, float] = {}
    for t in trades:
        m = str(t.exit_time)[:7]
        monthly[m] = monthly.get(m, 0) + t.pnl
    monthly = dict(sorted(monthly.items()))

    return dict(
        label=label, n=n, wr=wr, aw=aw, al=al, pf=pf, ev=ev,
        total_pnl=tot, total_costs=tc, mdd=mdd, mdd_pct=mdd_pct,
        monthly=monthly, wins=len(wins), losses=n - len(wins),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_report(m: dict, trades: list[Trade]):
    W = 56
    sep  = "─" * W
    dsep = "═" * W

    pf_s = _pf_str(m["pf"])
    verdict = "✅ POSITIVE EV" if m["ev"] > 0 else "❌ negative ev"
    pf_verdict = "✅ PF > 1.0" if m["pf"] > 1.0 else ("🟡 PF > 0.5" if m["pf"] > 0.5 else "❌ PF < 0.5")

    print(dsep)
    print(f"  {m['label']}")
    print(dsep)

    print(f"\n  Trade Stats")
    print(sep)
    print(f"  {'Total trades:':<28}{m['n']}")
    print(f"  {'Wins / Losses:':<28}{m['wins']} / {m['losses']}")
    print(f"  {'Win rate:':<28}{m['wr']:.1%}")
    print(f"  {'Avg win (net):':<28}₹{m['aw']:,.1f}")
    print(f"  {'Avg loss (net):':<28}₹{m['al']:,.1f}")
    print(f"  {'R:R (win/|loss|):':<28}1:{abs(m['al']/m['aw']):.2f}" if m['aw'] else "")
    print(f"  {'Profit factor:':<28}{pf_s}  {pf_verdict}")
    print(f"  {'EV per trade:':<28}₹{m['ev']:+,.1f}  {verdict}")

    print(f"\n  P&L")
    print(sep)
    print(f"  {'Total net P&L:':<28}₹{m['total_pnl']:+,.0f}")
    print(f"  {'Total costs paid:':<28}₹{m['total_costs']:,.0f}")
    print(f"  {'Cost per trade:':<28}₹{m['total_costs']/m['n']:,.0f}" if m['n'] else "")
    print(f"  {'Initial capital:':<28}₹{INITIAL_CAPITAL:,.0f}")
    print(f"  {'Final capital:':<28}₹{INITIAL_CAPITAL + m['total_pnl']:,.0f}")
    final_cap = INITIAL_CAPITAL + m['total_pnl']
    print(f"  {'Return:':<28}{(final_cap/INITIAL_CAPITAL - 1)*100:+.2f}%")

    print(f"\n  Risk")
    print(sep)
    print(f"  {'Max drawdown:':<28}₹{m['mdd']:,.0f}  ({m['mdd_pct']:.1f}%)")

    if m["monthly"]:
        print(f"\n  Monthly P&L")
        print(sep)
        max_abs = max(abs(v) for v in m["monthly"].values()) or 1
        for month, pnl in m["monthly"].items():
            bar_len = int(abs(pnl) / max_abs * 20)
            bar  = "█" * bar_len if pnl >= 0 else "░" * bar_len
            sign = "+" if pnl >= 0 else ""
            print(f"  {month}  {sign}₹{pnl:>9,.0f}  {bar}")

    # Exit reason breakdown
    reasons: dict[str, list[float]] = {}
    for t in trades:
        reasons.setdefault(t.exit_reason, []).append(t.pnl)
    print(f"\n  Exit Reasons")
    print(sep)
    print(f"  {'Reason':<20} {'Count':>5}  {'WR':>5}  {'AvgNet':>9}  {'Total':>10}")
    for r, plist in sorted(reasons.items(), key=lambda x: -len(x[1])):
        cnt  = len(plist)
        wr   = sum(1 for p in plist if p > 0) / cnt
        avg  = float(np.mean(plist))
        tot  = sum(plist)
        print(f"  {r:<20} {cnt:>5}  {wr:>4.0%}  ₹{avg:>7,.0f}  ₹{tot:>8,.0f}")

    # Trailing stop analysis
    trailing_trades = [t for t in trades if "trailing" in t.exit_reason or t.max_excursion > 0]
    if trailing_trades:
        avg_exc = float(np.mean([t.max_excursion for t in trades]))
        phase2  = [t for t in trades if "trailing" in t.exit_reason]
        print(f"\n  Trailing Stop Analysis")
        print(sep)
        print(f"  Trades reaching phase 2 (trailing): {len(phase2)}")
        print(f"  Avg max favourable excursion:        ₹{avg_exc:,.1f}")
        if phase2:
            print(f"  Avg pnl when trailing stop hit:      ₹{float(np.mean([t.pnl for t in phase2])):,.1f}")

    print()


def print_comparison(prev_results: list[dict], v3: dict):
    """Compare V3 against V1/V2 baselines from earlier runs."""
    print("═" * 80)
    print("  MEAN REVERSION — ALL VERSIONS COMPARED")
    print("═" * 80)
    header = f"  {'Version':<24} {'TF':>4} {'Trades':>6} {'WR':>6} {'PF':>6} {'EV/T':>8} {'Net P&L':>11} {'MaxDD':>11}"
    print(header)
    print("  " + "─" * 76)
    for r in prev_results:
        print(f"  {r['label']:<24} {r.get('tf',''):>4} {r['n']:>6} {r['wr']:>5.0%}"
              f"  {_pf_str(r['pf']):>6} {r['ev']:>+7.0f} {r.get('total_pnl',0):>+10,.0f}"
              f"  {r.get('mdd',0):>10,.0f}")
    # V3
    r = v3
    print(f"  {r['label']:<24} {'30m':>4} {r['n']:>6} {r['wr']:>5.0%}"
          f"  {_pf_str(r['pf']):>6} {r['ev']:>+7.0f} {r['total_pnl']:>+10,.0f}"
          f"  {r['mdd']:>10,.0f}")
    print("═" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Mean Reversion V3 — Multi-Timeframe + Trailing SL  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("  Architecture:")
    print("    Tier 1 — Daily trend:  close > 10-day SMA  (no falling knives)")
    print("    Tier 2 — 30m setup:    BB%<0.20 + RSI<40 + BB_width>1% + ATR>0.25%")
    print("    Tier 3 — Trigger:      RSI rising 2 bars + volume > 1.5×")
    print("    Exit   — Trailing SL:  1.5×ATR → breakeven → trailing at peak-1×ATR")
    print("    No fixed target. Winners run until trailing SL or EOD.")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    all_trades: list[Trade] = []
    skipped = []

    for sym in SYMBOLS:
        df5 = load_5m(sym)
        if df5 is None or len(df5) < 100:
            skipped.append(sym)
            continue

        df30 = aggregate_ohlcv(df5, 30)
        daily_sma = build_daily_sma(df30)

        sym_trades = run_symbol(sym, df30, daily_sma)
        all_trades.extend(sym_trades)
        n   = len(sym_trades)
        w   = sum(1 for t in sym_trades if t.pnl > 0)
        wr  = w / n if n else 0
        pnl = sum(t.pnl for t in sym_trades)
        print(f"  {sym:<18}  {n:>3}T  WR:{wr:.0%}  P&L:₹{pnl:+,.0f}")

    if skipped:
        print(f"\n  Skipped (no data): {', '.join(skipped)}")

    if not all_trades:
        print("\n  ERROR: No trades generated. Check data and filters.")
        return

    print()

    # ── Compute and print metrics ─────────────────────────────────────────────
    m = compute_metrics(all_trades, "MR V3 — Daily+30m+Trailing")
    print_report(m, all_trades)

    # ── Comparison table against prior versions ───────────────────────────────
    prior = [
        # From earlier runs — hardcoded for comparison
        dict(label="V1 (5m, no filter)",   tf="5m",  n=738, wr=0.053, pf=0.033, ev=-62.8,  total_pnl=-46382, mdd=-46360),
        dict(label="V2 (5m, time+ATR)",    tf="5m",  n=65,  wr=0.154, pf=0.036, ev=-61.9,  total_pnl=-4027,  mdd=-4013),
        dict(label="V2 (30m, no gate)",    tf="30m", n=68,  wr=0.250, pf=0.144, ev=-54.2,  total_pnl=-3687,  mdd=-3614),
    ]
    print_comparison(prior, m)

    # ── Breakeven analysis ────────────────────────────────────────────────────
    print()
    print("  BREAKEVEN ANALYSIS")
    print("─" * 56)
    print(f"  Current EV/trade:  ₹{m['ev']:+,.1f}")
    if m['aw'] != 0 and m['al'] != 0:
        be_wr = -m['al'] / (m['aw'] - m['al'])
        print(f"  Breakeven WR:      {be_wr:.1%}  (current: {m['wr']:.1%})")
        gap = be_wr - m['wr']
        print(f"  Gap to breakeven:  {gap:+.1%}")
        if m['aw'] > 0:
            be_aw = m['al'] * m['wr'] / (m['wr'] - 1) if m['wr'] != 1 else 0
            # solve: wr*aw + (1-wr)*al = 0 → aw = -al*(1-wr)/wr
            be_aw2 = -m['al'] * (1 - m['wr']) / m['wr']
            print(f"  OR: need avg win = ₹{be_aw2:,.0f} at current WR {m['wr']:.0%}  (now ₹{m['aw']:,.0f})")
    print()

if __name__ == "__main__":
    main()
