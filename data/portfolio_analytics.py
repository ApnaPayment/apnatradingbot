"""
Portfolio Analytics
Computes performance metrics from completed trade records.
Works with both live RiskManager data and backtested results.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# Sector map for NSE large/mid caps — extend as your watchlist grows
SECTOR_MAP: dict[str, str] = {
    "RELIANCE":   "Energy/Retail",
    "TCS":        "IT",
    "INFY":       "IT",
    "WIPRO":      "IT",
    "HCLTECH":    "IT",
    "TECHM":      "IT",
    "LTI":        "IT",
    "MPHASIS":    "IT",
    "HDFCBANK":   "Banking",
    "ICICIBANK":  "Banking",
    "AXISBANK":   "Banking",
    "KOTAKBANK":  "Banking",
    "SBIN":       "Banking",
    "BANDHANBNK": "Banking",
    "LT":         "Infrastructure",
    "NTPC":       "Infrastructure",
    "POWERGRID":  "Infrastructure",
    "BHARTIARTL": "Telecom",
    "IDEA":       "Telecom",
    "SUNPHARMA":  "Pharma",
    "DRREDDY":    "Pharma",
    "CIPLA":      "Pharma",
    "DIVISLAB":   "Pharma",
    "MARUTI":     "Auto",
    "TATAMOTORS": "Auto",
    "M&M":        "Auto",
    "BAJAJ-AUTO": "Auto",
    "EICHERMOT":  "Auto",
    "TITAN":      "Consumer",
    "ASIANPAINT": "Consumer",
    "NESTLEIND":  "FMCG",
    "HINDUNILVR": "FMCG",
    "ITC":        "FMCG",
    "TATACONSUM": "FMCG",
    "ITBEES":     "ETF",
    "NIFTYBEES":  "ETF",
    "BANKBEES":   "ETF",
    "GOLDBEES":   "ETF",
}


class PortfolioAnalytics:
    """
    Computes trade performance metrics, sector allocation, and position correlation.

    Expected trade dict schema:
        {
            "symbol":      str,
            "action":      "BUY" | "SELL",
            "strategy":    str,
            "entry_price": float,
            "exit_price":  float,
            "quantity":    int,
            "pnl":         float,
            "exit_reason": str,   # "stop_loss" | "target" | "manual"
            "entry_time":  str,
            "exit_time":   str,
        }
    """

    def __init__(self, data_manager=None):
        self.data = data_manager

    # ─────────────────────────────────────────────────────────────────────────
    # Core metrics
    # ─────────────────────────────────────────────────────────────────────────

    def compute_metrics(self, trades: list[dict]) -> dict:
        """Full performance metric suite from a list of completed trades."""
        if not trades:
            return self._empty_metrics()

        pnls   = [float(t["pnl"]) for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_return  = sum(pnls)
        win_rate      = len(wins) / len(pnls)
        avg_win       = float(np.mean(wins))   if wins   else 0.0
        avg_loss      = float(np.mean(losses)) if losses else 0.0
        profit_factor = (
            sum(wins) / abs(sum(losses))
            if losses and sum(losses) != 0 else None
        )
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Sharpe (trade returns approximated as daily)
        arr = np.array(pnls)
        std = float(np.std(arr))
        sharpe = (float(np.mean(arr)) / std * np.sqrt(252)) if std > 0 and len(pnls) > 1 else 0.0

        # Drawdown
        cumulative  = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns   = cumulative - running_max
        max_dd_idx  = int(np.argmin(drawdowns))
        max_drawdown = float(drawdowns[max_dd_idx])
        peak_at_dd   = float(running_max[max_dd_idx])
        max_dd_pct   = (max_drawdown / peak_at_dd * 100) if peak_at_dd != 0 else 0.0

        calmar = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

        # Exit reason breakdown
        exit_counts: dict[str, int] = {}
        for t in trades:
            reason = t.get("exit_reason", "unknown")
            exit_counts[reason] = exit_counts.get(reason, 0) + 1

        return {
            "total_trades":     len(trades),
            "winning_trades":   len(wins),
            "losing_trades":    len(losses),
            "win_rate":         round(win_rate, 4),
            "total_return":     round(total_return, 2),
            "avg_win":          round(avg_win, 2),
            "avg_loss":         round(avg_loss, 2),
            "profit_factor":    round(profit_factor, 2) if profit_factor else None,
            "expectancy":       round(expectancy, 2),
            "sharpe_ratio":     round(sharpe, 2),
            "max_drawdown":     round(max_drawdown, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "calmar_ratio":     round(calmar, 2),
            "exit_breakdown":   exit_counts,
            "strategy_breakdown": self._strategy_breakdown(trades),
            "equity_curve":     [round(v, 2) for v in cumulative.tolist()],
        }

    def _strategy_breakdown(self, trades: list[dict]) -> dict:
        by_strat: dict[str, list] = {}
        for t in trades:
            strat = t.get("strategy", "unknown")
            by_strat.setdefault(strat, []).append(float(t["pnl"]))

        result = {}
        for strat, pnls in by_strat.items():
            wins = [p for p in pnls if p > 0]
            result[strat] = {
                "trades":    len(pnls),
                "win_rate":  round(len(wins) / len(pnls), 3),
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl":   round(float(np.mean(pnls)), 2),
            }
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Sector allocation
    # ─────────────────────────────────────────────────────────────────────────

    def sector_allocation(self, positions: dict) -> dict:
        """
        Maps open positions to sectors and computes % allocation.
        Returns {sector: {"value": float, "pct": float, "symbols": list}}.
        """
        sector_value:   dict[str, float] = {}
        sector_symbols: dict[str, list]  = {}
        total_value = 0.0

        for sym, pos in positions.items():
            value = float(pos["entry_price"]) * int(pos["quantity"])
            total_value += value
            base   = sym.split("-")[0]
            sector = SECTOR_MAP.get(base, "Other")
            sector_value[sector]   = sector_value.get(sector, 0.0) + value
            sector_symbols.setdefault(sector, []).append(sym)

        if total_value == 0:
            return {}

        return {
            sector: {
                "value":   round(val, 2),
                "pct":     round(val / total_value * 100, 1),
                "symbols": sector_symbols[sector],
            }
            for sector, val in sorted(sector_value.items(), key=lambda x: -x[1])
        }

    def concentration_warnings(self, positions: dict,
                                threshold_pct: float = 40.0) -> list[str]:
        """Warn if any sector exceeds the concentration threshold."""
        alloc = self.sector_allocation(positions)
        return [
            f"Sector concentration risk: {sector} = {d['pct']:.0f}%"
            f" ({', '.join(d['symbols'])})"
            for sector, d in alloc.items()
            if d["pct"] >= threshold_pct
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Correlation
    # ─────────────────────────────────────────────────────────────────────────

    def correlation_matrix(self, symbols: list,
                            exchange: str = "nse_cm") -> Optional[pd.DataFrame]:
        """
        Pearson correlation of log-returns between held symbols using stored OHLCV.
        Returns None if data_manager not set or insufficient data.
        """
        if not self.data or len(symbols) < 2:
            return None

        series: dict[str, np.ndarray] = {}
        for sym in symbols:
            df = self.data.get_ohlcv(sym, exchange, limit=60)
            if len(df) >= 20:
                series[sym] = df["close"].values

        if len(series) < 2:
            return None

        min_len = min(len(v) for v in series.values())
        returns = pd.DataFrame({
            sym: np.diff(np.log(vals[-min_len:]))
            for sym, vals in series.items()
        })
        return returns.corr().round(2)

    def high_correlation_pairs(self, symbols: list, threshold: float = 0.70,
                                exchange: str = "nse_cm") -> list[tuple]:
        """Return (sym_a, sym_b, correlation) tuples above threshold."""
        corr = self.correlation_matrix(symbols, exchange)
        if corr is None:
            return []
        cols = corr.columns.tolist()
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = float(corr.iloc[i, j])
                if abs(val) >= threshold:
                    pairs.append((cols[i], cols[j], round(val, 2)))
        return sorted(pairs, key=lambda x: -abs(x[2]))

    # ─────────────────────────────────────────────────────────────────────────
    # Text report
    # ─────────────────────────────────────────────────────────────────────────

    def text_report(self, trades: list[dict], positions: dict = None) -> str:
        m   = self.compute_metrics(trades)
        pf  = f"{m['profit_factor']:.2f}" if m["profit_factor"] else "∞"
        sep = "─" * 44
        lines = [
            "═" * 44,
            "  Portfolio Analytics",
            "═" * 44,
            f"  Trades:          {m['total_trades']}"
            f"  ({m['winning_trades']}W / {m['losing_trades']}L)",
            f"  Win rate:        {m['win_rate']:.1%}",
            f"  Total P&L:       ₹{m['total_return']:,.0f}",
            f"  Avg win:         ₹{m['avg_win']:,.0f}",
            f"  Avg loss:        ₹{m['avg_loss']:,.0f}",
            f"  Expectancy:      ₹{m['expectancy']:,.0f} / trade",
            f"  Profit factor:   {pf}",
            f"  Sharpe ratio:    {m['sharpe_ratio']:.2f}",
            f"  Max drawdown:    ₹{m['max_drawdown']:,.0f} ({m['max_drawdown_pct']:.1f}%)",
            f"  Calmar ratio:    {m['calmar_ratio']:.2f}",
            sep,
            "  Exit breakdown:",
        ]
        for reason, count in m.get("exit_breakdown", {}).items():
            lines.append(f"    {reason:15s}  {count} trades")

        lines.append(sep)
        lines.append("  Strategy breakdown:")
        for strat, s in m["strategy_breakdown"].items():
            lines.append(
                f"    {strat:25s}  {s['trades']:3d} trades"
                f"  WR={s['win_rate']:.1%}  P&L=₹{s['total_pnl']:,.0f}"
            )

        if positions:
            alloc    = self.sector_allocation(positions)
            warnings = self.concentration_warnings(positions)
            if alloc:
                lines.append(sep)
                lines.append("  Sector allocation:")
                for sector, d in alloc.items():
                    lines.append(
                        f"    {sector:20s}  {d['pct']:5.1f}%  ₹{d['value']:,.0f}"
                    )
            for w in warnings:
                lines.append(f"  ⚠  {w}")

        lines.append("═" * 44)
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _empty_metrics(self) -> dict:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0.0, "total_return": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, "profit_factor": None, "expectancy": 0.0,
            "sharpe_ratio": 0.0, "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "calmar_ratio": 0.0, "exit_breakdown": {}, "strategy_breakdown": {},
            "equity_curve": [],
        }
