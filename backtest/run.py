#!/usr/bin/env python3
"""
Backtest CLI
Run strategies against stored OHLCV history from the command line.

Examples:
    # Backtest momentum on all watchlist symbols
    python backtest/run.py --strategy momentum

    # Backtest mean_reversion on specific symbols, output CSV
    python backtest/run.py --strategy mean_reversion --symbols TCS-EQ INFY-EQ --output csv

    # Compare both strategies, save JSON report
    python backtest/run.py --strategy both --capital 200000 --output json

    # Short backtest with only 100 candles min
    python backtest/run.py --strategy momentum --min-candles 60
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# ─── Path setup ─────────────────────────────────────────────────────────────
# Support running from both the project root and backtest/ subdirectory
_here = Path(__file__).parent
_root = _here.parent
sys.path.insert(0, str(_root))

from backtest.backtester import Backtester
from data.data_manager import DataManager
from data.portfolio_analytics import PortfolioAnalytics
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy

# ─── Default watchlist (mirrors main.py) ─────────────────────────────────────
DEFAULT_WATCHLIST = [
    "RELIANCE-EQ", "TCS-EQ", "INFY-EQ", "HDFCBANK-EQ",
    "ICICIBANK-EQ", "WIPRO-EQ", "LT-EQ", "AXISBANK-EQ",
    "ITBEES-EQ", "NIFTYBEES-EQ",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AlgoTrader — backtesting CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--strategy", choices=["momentum", "mean_reversion", "both"],
        default="both",
        help="Which strategy to backtest (default: both)",
    )
    p.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to test. Omit to use the full default watchlist.",
    )
    p.add_argument(
        "--capital", type=float, default=100_000,
        help="Starting capital in ₹ (default: 100000)",
    )
    p.add_argument(
        "--min-candles", type=int, default=50,
        help="Skip symbols with fewer candles than this (default: 50)",
    )
    p.add_argument(
        "--output", choices=["text", "csv", "json"], default="text",
        help="Output format (default: text)",
    )
    p.add_argument(
        "--out-file", default=None,
        help="Save output to this file path (prints to stdout if omitted)",
    )
    return p.parse_args()


def build_strategies(name: str, capital: float) -> list[tuple[str, Backtester]]:
    pairs = {
        "momentum":      ("Momentum",      MomentumStrategy(),      capital),
        "mean_reversion":("MeanReversion", MeanReversionStrategy(), capital),
    }
    if name == "both":
        return [(n, Backtester(s, c)) for n, (_, s, c) in pairs.items()]
    label, strat, cap = pairs[name]
    return [(label, Backtester(strat, cap))]


def results_to_dicts(results: dict, strategy_label: str) -> list[dict]:
    rows = []
    for symbol, r in results.items():
        pf = r.profit_factor if r.profit_factor != float("inf") else None
        rows.append({
            "strategy":      strategy_label,
            "symbol":        symbol,
            "total_trades":  r.total_trades,
            "win_rate":      round(r.win_rate, 4),
            "total_pnl":     r.total_return,
            "avg_win":       r.avg_win,
            "avg_loss":      r.avg_loss,
            "profit_factor": round(pf, 2) if pf else None,
            "sharpe_ratio":  r.sharpe_ratio,
            "max_drawdown":  r.max_drawdown,
        })
    return rows


def format_text(all_rows: list[dict], backtester_map: dict,
                results_map: dict) -> str:
    lines = []
    for label, bt in backtester_map.items():
        res = results_map[label]
        lines.append(bt.summary_table(res))
        lines.append("")

    # Cross-strategy aggregate
    if len(all_rows) > 0:
        analytics = PortfolioAnalytics()
        trade_records = []
        for label, res in results_map.items():
            for r in res.values():
                for t in r.trades:
                    trade_records.append({
                        "pnl":         t.pnl,
                        "action":      t.action,
                        "strategy":    label,
                        "entry_price": t.entry_price,
                        "exit_price":  t.exit_price,
                        "exit_reason": t.exit_reason,
                    })
        if trade_records:
            lines.append(analytics.text_report(trade_records))
    return "\n".join(lines)


def main():
    args = parse_args()
    symbols = args.symbols or DEFAULT_WATCHLIST

    print(f"Loading data from database...", flush=True)
    data = DataManager()   # no Kotak client needed — reads from SQLite only

    # Check DB has data
    stats = data.get_db_stats()
    if stats["ohlcv_candles"] == 0:
        print("No OHLCV data in database. Run the bot first to accumulate price history.")
        print("(The bot stores ticks → builds candles during trading hours.)")
        sys.exit(1)

    print(f"DB: {stats['ohlcv_candles']} candles across {stats['scrip_instruments']} instruments")
    print(f"Symbols: {symbols}")
    print(f"Strategy: {args.strategy}  |  Capital: ₹{args.capital:,.0f}")
    print()

    pairs       = build_strategies(args.strategy, args.capital)
    bt_map      = {}   # label → Backtester
    results_map = {}   # label → {symbol: BacktestResult}
    all_rows    = []

    for label, bt in pairs:
        print(f"Running {label} backtest...", flush=True)
        results = bt.run_multi(data, symbols, min_candles=args.min_candles)
        bt_map[label]      = bt
        results_map[label] = results
        all_rows.extend(results_to_dicts(results, label))
        print(f"  {label}: {len(results)} symbols tested", flush=True)

    # ─── Format output ───────────────────────────────────────────────────────
    if args.output == "text":
        output = format_text(all_rows, bt_map, results_map)
    elif args.output == "json":
        output = json.dumps(all_rows, indent=2, default=str)
    else:  # csv
        import io
        buf = io.StringIO()
        if all_rows:
            writer = csv.DictWriter(buf, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        output = buf.getvalue()

    if args.out_file:
        Path(args.out_file).write_text(output)
        print(f"\nReport saved to: {args.out_file}")
    else:
        print(output)


if __name__ == "__main__":
    main()
