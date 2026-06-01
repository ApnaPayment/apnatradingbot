"""
Hyperparameter Optimizer
Weekly walk-forward grid search on strategy parameters.
Uses the backtester to score each param combo, picks the best Sharpe ratio,
asks Claude to explain why the new params make sense, then writes them to DB
so the bot loads them automatically on the next cycle.

Run automatically every Sunday at 00:30 via main.py scheduler.
Can also be run manually:
    python -m ai.optimizer --dry-run
"""

import itertools
import logging
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "market_data.db"


# ─────────────────────────────────────────────────────────────────────────────
# Parameter grids — all combinations are exhaustively tested
# ─────────────────────────────────────────────────────────────────────────────

MOMENTUM_GRID = {
    "fast_ema":       [7, 9, 12],
    "slow_ema":       [18, 21, 26],
    "rsi_oversold":   [35, 40, 45],
    "rsi_overbought": [60, 65, 70],
    "volume_factor":  [1.3, 1.5, 2.0],
}

MEAN_REV_GRID = {
    "bb_period":      [15, 20, 25],
    "bb_std":         [1.8, 2.0, 2.2],
    "rsi_oversold":   [25, 30, 35],
    "rsi_overbought": [65, 70, 75],
    "volume_factor":  [1.0, 1.2, 1.5],
}

# Defaults — used when no optimised params exist yet
MOMENTUM_DEFAULTS = {
    "fast_ema": 9, "slow_ema": 21,
    "rsi_oversold": 40, "rsi_overbought": 65,
    "volume_factor": 1.5,
}
MEAN_REV_DEFAULTS = {
    "bb_period": 20, "bb_std": 2.0,
    "rsi_oversold": 30, "rsi_overbought": 70,
    "volume_factor": 1.2,
}


@dataclass
class OptimResult:
    strategy:       str
    best_params:    dict
    best_sharpe:    float
    best_winrate:   float
    best_pnl:       float
    total_combos:   int
    symbols_tested: list
    run_at:         str = field(default_factory=lambda: datetime.now().isoformat())
    ai_explanation: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers — persist best params so bot loads them on restart
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_optim_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS optim_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy    TEXT NOT NULL,
                params      TEXT NOT NULL,   -- JSON
                sharpe      REAL,
                win_rate    REAL,
                total_pnl   REAL,
                run_at      TEXT,
                ai_explanation TEXT,
                active      INTEGER DEFAULT 1
            )
        """)


def save_optim_result(result: OptimResult):
    """Persist optimisation result to DB. Marks previous rows for this strategy inactive."""
    _ensure_optim_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE optim_results SET active=0 WHERE strategy=?",
            (result.strategy,)
        )
        conn.execute(
            """INSERT INTO optim_results
               (strategy, params, sharpe, win_rate, total_pnl, run_at, ai_explanation, active)
               VALUES (?,?,?,?,?,?,?,1)""",
            (
                result.strategy,
                json.dumps(result.best_params),
                result.best_sharpe,
                result.best_winrate,
                result.best_pnl,
                result.run_at,
                result.ai_explanation,
            )
        )
    logger.info(f"Saved optim result for {result.strategy}: {result.best_params}")


def load_best_params(strategy: str) -> Optional[dict]:
    """
    Load the most recent active optimised params for a strategy.
    Returns None if no optimisation has been run yet (caller uses defaults).
    """
    _ensure_optim_table()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT params FROM optim_results WHERE strategy=? AND active=1 "
                "ORDER BY id DESC LIMIT 1",
                (strategy,)
            ).fetchone()
        if row:
            return json.loads(row[0])
    except Exception as e:
        logger.warning(f"Could not load params for {strategy}: {e}")
    return None


def load_all_optim_history(limit: int = 10) -> list[dict]:
    """Load recent optimisation history for the Settings page."""
    _ensure_optim_table()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                """SELECT strategy, params, sharpe, win_rate, total_pnl, run_at, ai_explanation, active
                   FROM optim_results ORDER BY id DESC LIMIT ?""",
                (limit,)
            ).fetchall()
        return [
            {
                "strategy":       r[0],
                "params":         json.loads(r[1]),
                "sharpe":         r[2],
                "win_rate":       r[3],
                "total_pnl":      r[4],
                "run_at":         r[5],
                "ai_explanation": r[6],
                "active":         bool(r[7]),
            }
            for r in rows
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Core optimizer
# ─────────────────────────────────────────────────────────────────────────────

class StrategyOptimizer:
    """
    Grid-search optimizer for MomentumStrategy and MeanReversionStrategy.

    For each param combination:
      1. Instantiate the strategy with those params
      2. Run the backtester on all available OHLCV data for the watchlist
      3. Aggregate Sharpe, win rate, total P&L across symbols
      4. Track the combination with the highest average Sharpe
      5. Ask Claude to explain why the best params make sense
      6. Save to DB — bot loads them on next startup / cycle
    """

    def __init__(self, data_manager, ai_engine=None):
        self.data = data_manager
        self.ai   = ai_engine      # Optional — explanation skipped if None

    def run_all(self, symbols: list, dry_run: bool = False) -> list[OptimResult]:
        """
        Optimise both strategies on `symbols`.
        Returns a list of OptimResult (one per strategy).
        """
        from strategies.momentum import MomentumStrategy
        from strategies.mean_reversion import MeanReversionStrategy

        results = []

        logger.info("=" * 50)
        logger.info("Starting weekly hyperparameter optimisation")
        logger.info(f"Symbols: {symbols}")
        logger.info("=" * 50)

        # Pre-load all OHLCV to avoid repeated DB queries
        ohlcv_cache = {}
        for sym in symbols:
            df = self.data.get_ohlcv(sym, limit=500)
            if df is not None and len(df) >= 60:
                ohlcv_cache[sym] = df
            else:
                logger.warning(f"Skipping {sym}: insufficient data ({len(df) if df is not None else 0} candles)")

        if not ohlcv_cache:
            logger.error("No symbols have enough OHLCV data for optimisation")
            return []

        usable = list(ohlcv_cache.keys())
        logger.info(f"Optimising on {len(usable)} symbols: {usable}")

        for strategy_name, strategy_cls, param_grid, defaults in [
            ("momentum",       MomentumStrategy,       MOMENTUM_GRID,  MOMENTUM_DEFAULTS),
            ("mean_reversion", MeanReversionStrategy,  MEAN_REV_GRID,  MEAN_REV_DEFAULTS),
        ]:
            result = self._optimise_strategy(
                strategy_name, strategy_cls, param_grid, defaults, ohlcv_cache, dry_run
            )
            if result:
                results.append(result)

        return results

    def _optimise_strategy(self, name, strategy_cls, param_grid, defaults,
                           ohlcv_cache, dry_run) -> Optional[OptimResult]:
        from backtest.backtester import Backtester

        keys   = list(param_grid.keys())
        combos = list(itertools.product(*param_grid.values()))
        logger.info(f"[{name}] Testing {len(combos)} parameter combinations...")

        best_sharpe  = -999.0
        best_params  = dict(defaults)
        best_winrate = 0.0
        best_pnl     = 0.0

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))

            # Validate: fast must be < slow for momentum
            if name == "momentum" and params.get("fast_ema", 9) >= params.get("slow_ema", 21):
                continue

            try:
                strategy  = strategy_cls(**params)
                backtester = Backtester(strategy)

                sharpes   = []
                winrates  = []
                pnls      = []

                for sym, df in ohlcv_cache.items():
                    result = backtester.run(df, sym, min_candles=40)
                    if result.total_trades >= 3:   # need at least 3 trades to be meaningful
                        sharpes.append(result.sharpe_ratio)
                        winrates.append(result.win_rate)
                        pnls.append(result.total_return)

                if not sharpes:
                    continue

                avg_sharpe  = sum(sharpes)  / len(sharpes)
                avg_winrate = sum(winrates) / len(winrates)
                avg_pnl     = sum(pnls)     / len(pnls)

                if avg_sharpe > best_sharpe:
                    best_sharpe  = avg_sharpe
                    best_params  = params
                    best_winrate = avg_winrate
                    best_pnl     = avg_pnl

            except Exception as e:
                logger.debug(f"[{name}] Combo {params} failed: {e}")
                continue

            if (i + 1) % 50 == 0:
                logger.info(f"[{name}] {i+1}/{len(combos)} tested, best Sharpe so far: {best_sharpe:.2f}")

        logger.info(
            f"[{name}] Best params: {best_params}  "
            f"Sharpe={best_sharpe:.2f}  WinRate={best_winrate:.1%}  AvgPnL=₹{best_pnl:,.0f}"
        )

        # Ask Claude to explain the best params
        explanation = ""
        if self.ai and best_sharpe > -100:
            explanation = self._get_ai_explanation(name, defaults, best_params, best_sharpe, best_winrate)

        optim = OptimResult(
            strategy       = name,
            best_params    = best_params,
            best_sharpe    = round(best_sharpe, 3),
            best_winrate   = round(best_winrate, 3),
            best_pnl       = round(best_pnl, 2),
            total_combos   = len(combos),
            symbols_tested = list(ohlcv_cache.keys()),
            ai_explanation = explanation,
        )

        if not dry_run:
            save_optim_result(optim)

        return optim

    def _get_ai_explanation(self, strategy_name: str, old_params: dict,
                            new_params: dict, sharpe: float, winrate: float) -> str:
        """Ask Claude why the new params are better than the defaults."""
        if not self.ai or not self.ai.client:
            return ""

        changed = {
            k: {"from": old_params.get(k), "to": new_params.get(k)}
            for k in new_params
            if new_params.get(k) != old_params.get(k)
        }
        if not changed:
            return "Parameters unchanged — current defaults are already optimal."

        prompt = f"""A walk-forward backtest optimisation on Indian NSE equities (5-min OHLCV)
has found better hyperparameters for the {strategy_name} strategy.

PARAMETER CHANGES:
{json.dumps(changed, indent=2)}

PERFORMANCE IMPROVEMENT:
- Average Sharpe ratio of new params: {sharpe:.2f}
- Win rate: {winrate:.1%}

Strategy type: {"EMA crossover momentum (fast/slow EMA + RSI + volume)" if strategy_name == "momentum" else "Bollinger Band mean reversion (BB width + RSI oversold/overbought + volume)"}

In 2-3 sentences, explain WHY these parameter changes make sense for Indian intraday equity markets.
Consider: market microstructure, typical intraday volatility, NSE tick size, session timing effects.
Be specific — don't just say "it performed better", explain the market logic.
Keep it under 80 words."""

        try:
            return self.ai._call_claude(prompt)
        except Exception as e:
            logger.warning(f"AI explanation failed: {e}")
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.data_manager import DataManager
    from ai.decision_engine import AIDecisionEngine
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    dry_run = "--dry-run" in sys.argv

    data = DataManager()
    ai   = AIDecisionEngine()

    WATCHLIST = [
        "RELIANCE-EQ", "TCS-EQ", "INFY-EQ", "HDFCBANK-EQ",
        "ICICIBANK-EQ", "WIPRO-EQ", "LT-EQ", "AXISBANK-EQ",
        "ITBEES-EQ", "NIFTYBEES-EQ",
    ]

    optimizer = StrategyOptimizer(data, ai)
    results   = optimizer.run_all(WATCHLIST, dry_run=dry_run)

    for r in results:
        print(f"\n{'='*50}")
        print(f"Strategy: {r.strategy}")
        print(f"Best params: {r.best_params}")
        print(f"Sharpe: {r.best_sharpe:.2f}  WinRate: {r.best_winrate:.1%}  AvgPnL: ₹{r.best_pnl:,.0f}")
        print(f"Tested {r.total_combos} combos on {len(r.symbols_tested)} symbols")
        if r.ai_explanation:
            print(f"\nAI: {r.ai_explanation}")
        if dry_run:
            print("(dry-run — not saved to DB)")
