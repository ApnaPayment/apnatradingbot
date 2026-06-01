"""
Backtester
Walk-forward simulation of a strategy on stored OHLCV data.
Trades are simulated candle-by-candle — no look-ahead bias.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    symbol: str
    action: str
    entry_price: float
    exit_price: float
    quantity: int
    stop_loss: float
    target: float
    entry_time: str
    exit_time: str
    exit_reason: str        # "stop_loss" | "target" | "end_of_data"
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    trades: list = field(default_factory=list)
    total_return: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0


class Backtester:
    """
    Walk-forward backtester. Gives the strategy only past candles at each step.
    Uses 10% flat position sizing per trade.
    """

    def __init__(self, strategy, initial_capital: float = 100_000):
        self.strategy = strategy
        self.initial_capital = initial_capital

    # ─────────────────────────────────────────────────────────────────────────
    # Core simulation
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, symbol: str, min_candles: int = 50) -> BacktestResult:
        """
        Simulate the strategy on a full OHLCV DataFrame.
        The strategy sees only df.iloc[:i+1] at each step i.
        """
        strategy_name = self.strategy.__class__.__name__
        result = BacktestResult(symbol=symbol, strategy=strategy_name)
        capital = self.initial_capital
        position: Optional[dict] = None

        for i in range(min_candles, len(df)):
            window = df.iloc[: i + 1].copy()
            current = window.iloc[-1]
            ts = str(current["timestamp"])

            if position:
                exit_reason, exit_price = self._check_exit(position, current)
                if exit_reason:
                    trade = self._close_trade(position, symbol, exit_price, ts, exit_reason)
                    result.trades.append(trade)
                    capital += trade.pnl
                    position = None

            if not position:
                signal = self.strategy.generate_signal(symbol, window)
                if signal and signal.confidence >= 0.6:
                    qty = max(1, int((capital * 0.10) / signal.price))
                    position = {
                        "action":      signal.action,
                        "entry_price": signal.price,
                        "stop_loss":   signal.stop_loss,
                        "target":      signal.target,
                        "quantity":    qty,
                        "entry_time":  ts,
                    }

        # Close any still-open position at the last candle
        if position:
            last = df.iloc[-1]
            trade = self._close_trade(
                position, symbol, last["close"],
                str(last["timestamp"]), "end_of_data"
            )
            result.trades.append(trade)

        self._compute_metrics(result)
        return result

    def _check_exit(self, position: dict, candle: pd.Series) -> tuple[Optional[str], float]:
        """Return (exit_reason, exit_price) or (None, 0) if still holding."""
        if position["action"] == "BUY":
            if candle["low"] <= position["stop_loss"]:
                return "stop_loss", position["stop_loss"]
            if candle["high"] >= position["target"]:
                return "target", position["target"]
        else:
            if candle["high"] >= position["stop_loss"]:
                return "stop_loss", position["stop_loss"]
            if candle["low"] <= position["target"]:
                return "target", position["target"]
        return None, 0.0

    def _close_trade(self, pos: dict, symbol: str, exit_price: float,
                     ts: str, reason: str) -> BacktestTrade:
        qty = pos["quantity"]
        entry = pos["entry_price"]
        pnl = (exit_price - entry) * qty if pos["action"] == "BUY" else (entry - exit_price) * qty
        pnl_pct = pnl / (entry * qty) * 100

        return BacktestTrade(
            symbol=symbol,
            action=pos["action"],
            entry_price=entry,
            exit_price=round(exit_price, 2),
            quantity=qty,
            stop_loss=pos["stop_loss"],
            target=pos["target"],
            entry_time=pos["entry_time"],
            exit_time=ts,
            exit_reason=reason,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_metrics(self, result: BacktestResult):
        trades = result.trades
        if not trades:
            return

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.total_trades   = len(trades)
        result.winning_trades = len(wins)
        result.total_return   = round(sum(pnls), 2)
        result.win_rate       = len(wins) / len(trades)
        result.avg_win        = round(float(np.mean(wins)), 2) if wins else 0.0
        result.avg_loss       = round(float(np.mean(losses)), 2) if losses else 0.0
        result.profit_factor  = (
            round(sum(wins) / abs(sum(losses)), 2)
            if losses and sum(losses) != 0 else float("inf")
        )

        if len(pnls) > 1:
            returns = np.array(pnls) / self.initial_capital
            std = float(np.std(returns))
            result.sharpe_ratio = round(
                float(np.mean(returns)) / std * np.sqrt(252) if std > 0 else 0.0, 2
            )

        cumulative  = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns   = cumulative - running_max
        result.max_drawdown = round(float(np.min(drawdowns)), 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────────

    def report(self, result: BacktestResult) -> str:
        sep = "─" * 40
        pf = f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "∞"
        lines = [
            f"{'═' * 40}",
            f"  Backtest: {result.symbol}  [{result.strategy}]",
            f"{'═' * 40}",
            f"  Total trades:    {result.total_trades}",
            f"  Win rate:        {result.win_rate:.1%}",
            f"  Total P&L:       ₹{result.total_return:,.0f}",
            f"  Avg win:         ₹{result.avg_win:,.0f}",
            f"  Avg loss:        ₹{result.avg_loss:,.0f}",
            f"  Profit factor:   {pf}",
            f"  Sharpe ratio:    {result.sharpe_ratio:.2f}",
            f"  Max drawdown:    ₹{result.max_drawdown:,.0f}",
            sep,
        ]
        recent = result.trades[-10:]
        if recent:
            lines.append("  Last 10 trades:")
            for t in recent:
                icon = "✓" if t.pnl > 0 else "✗"
                lines.append(
                    f"  {icon} {t.action:4s} ₹{t.entry_price:,.0f}→₹{t.exit_price:,.0f}"
                    f"  {t.exit_reason:12s}  P&L=₹{t.pnl:,.0f}"
                )
        lines.append(f"{'═' * 40}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-symbol runner
    # ─────────────────────────────────────────────────────────────────────────

    def run_multi(self, data_manager, symbols: list,
                  min_candles: int = 50) -> dict[str, BacktestResult]:
        """Run backtest on multiple symbols from the DataManager."""
        results = {}
        for symbol in symbols:
            df = data_manager.get_ohlcv(symbol, limit=500)
            if len(df) < min_candles + 10:
                logger.warning(f"Skipping {symbol}: only {len(df)} candles in DB")
                continue
            logger.info(f"Backtesting {symbol}...")
            r = self.run(df, symbol, min_candles=min_candles)
            results[symbol] = r
            logger.info(self.report(r))
        return results

    def summary_table(self, results: dict[str, BacktestResult]) -> str:
        """One-line summary row per symbol for quick comparison."""
        header = f"{'Symbol':<15} {'Trades':>6} {'WinRate':>8} {'TotalPnL':>12} {'Sharpe':>8} {'MaxDD':>12}"
        sep = "─" * len(header)
        rows = [header, sep]
        for symbol, r in results.items():
            rows.append(
                f"{symbol:<15} {r.total_trades:>6} {r.win_rate:>7.1%} "
                f"₹{r.total_return:>10,.0f} {r.sharpe_ratio:>8.2f} "
                f"₹{r.max_drawdown:>10,.0f}"
            )
        return "\n".join(rows)
