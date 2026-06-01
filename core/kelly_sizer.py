"""
Phase 11 — Kelly Criterion Dynamic Position Sizing

Replaces fixed-percentage sizing with a mathematically grounded bet size
based on historical edge, implemented as a multiplier on top of the
existing risk-cap infrastructure in RiskManager.

Kelly formula:
    f* = (p × b − q) / b
where:
    p = win probability (from ML ensemble or historical rate)
    q = 1 − p  (loss probability)
    b = average_win / average_loss  (payoff ratio from feedback loop DB)

Safety constraints:
  1. Fractional Kelly: f_safe = f* × KELLY_FRACTION (default 0.25)
     Full Kelly over-bets badly on noisy estimates; quarter-Kelly is
     the standard practitioner choice.
  2. Hard cap: never exceed max_position_size_pct from RiskConfig.
  3. Floor: never drop below 0.05 (5% of max position) regardless of edge.
  4. Falls back to confidence-based sizing when < MIN_TRADES historical trades.

All sizing is returned as a *multiplier* (0.0–1.0) applied to the
max_position_value computed by RiskManager.calculate_quantity().
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH       = Path(__file__).parent.parent / "data" / "market_data.db"
KELLY_FRACTION = 0.25    # quarter-Kelly for safety
MIN_TRADES     = 15      # minimum completed trades before Kelly is trusted
KELLY_FLOOR    = 0.05    # never size below 5% of max position
KELLY_CAP      = 1.00    # never exceed 100% of RiskManager's max (hard cap is in RiskManager)


# ─────────────────────────────────────────────────────────────────────────────
# Historical edge from feedback loop DB
# ─────────────────────────────────────────────────────────────────────────────

def _load_edge_stats(strategy: Optional[str] = None,
                     days: int = 90) -> Optional[dict]:
    """
    Compute win rate and avg win/loss ratio from ai_decisions table.
    Filters to approved (actually traded) decisions with known outcomes.
    Optionally filter by strategy.

    Returns:
        {
          "win_rate": 0.62,
          "avg_win":  320.0,   # ₹ average winning trade
          "avg_loss": 180.0,   # ₹ average losing trade (absolute)
          "payoff":   1.78,    # avg_win / avg_loss
          "n_trades": 47,
        }
    or None if insufficient data.
    """
    from datetime import datetime, timedelta
    since = (datetime.now() - timedelta(days=days)).isoformat()

    try:
        with sqlite3.connect(DB_PATH) as conn:
            if strategy:
                rows = conn.execute(
                    """SELECT outcome_pnl, outcome_correct
                       FROM ai_decisions
                       WHERE approved=1 AND outcome_correct IS NOT NULL
                         AND decided_at >= ? AND strategy=?""",
                    (since, strategy),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT outcome_pnl, outcome_correct
                       FROM ai_decisions
                       WHERE approved=1 AND outcome_correct IS NOT NULL
                         AND decided_at >= ?""",
                    (since,),
                ).fetchall()
    except Exception as e:
        logger.warning(f"Kelly: DB read failed: {e}")
        return None

    if len(rows) < MIN_TRADES:
        return None

    wins   = [abs(r[0]) for r in rows if r[1] == 1 and r[0] is not None]
    losses = [abs(r[0]) for r in rows if r[1] == 0 and r[0] is not None]

    if not wins or not losses:
        return None

    avg_win  = sum(wins)  / len(wins)
    avg_loss = sum(losses) / len(losses)
    payoff   = avg_win / avg_loss if avg_loss > 0 else 1.0
    win_rate = len(wins) / len(rows)

    return {
        "win_rate": round(win_rate, 4),
        "avg_win":  round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff":   round(payoff, 4),
        "n_trades": len(rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Kelly calculation
# ─────────────────────────────────────────────────────────────────────────────

def kelly_fraction(p: float, b: float) -> float:
    """
    Raw Kelly fraction for win probability `p` and payoff ratio `b`.
    Returns 0.0 for negative-edge situations.
    """
    q = 1.0 - p
    f = (p * b - q) / b
    return max(0.0, f)


def compute_kelly_multiplier(
    win_prob: Optional[float] = None,
    strategy: Optional[str] = None,
    days: int = 90,
) -> dict:
    """
    Compute the position-size multiplier for a single trade.

    Priority:
      1. ML-supplied `win_prob` + historical payoff ratio → full Kelly
      2. Historical win_rate (from DB) + historical payoff ratio → full Kelly
      3. Fallback: 1.0 (let RiskManager confidence scaling handle it)

    Returns:
        {
          "multiplier":   0.38,          # apply to max_position_value
          "kelly_raw":    1.52,          # raw Kelly before fractional / capping
          "kelly_safe":   0.38,          # after KELLY_FRACTION
          "win_prob":     0.68,
          "payoff":       2.1,
          "n_trades":     42,
          "source":       "ml+history",  # "ml+history" | "history" | "fallback"
        }
    """
    edge = _load_edge_stats(strategy=strategy, days=days)

    if edge is None:
        # Not enough history — return neutral multiplier (RiskManager's conf scaling handles it)
        return {
            "multiplier": 1.0, "kelly_raw": None, "kelly_safe": None,
            "win_prob": win_prob, "payoff": None,
            "n_trades": 0, "source": "fallback",
        }

    payoff   = edge["payoff"]
    n_trades = edge["n_trades"]

    if win_prob is not None:
        # Blend ML probability (70%) with historical base rate (30%) for stability
        p      = 0.70 * win_prob + 0.30 * edge["win_rate"]
        source = "ml+history"
    else:
        p      = edge["win_rate"]
        source = "history"

    raw        = kelly_fraction(p, payoff)
    safe       = raw * KELLY_FRACTION
    multiplier = max(KELLY_FLOOR, min(KELLY_CAP, safe))

    logger.info(
        f"Kelly sizing: p={p:.2%} b={payoff:.2f} "
        f"raw={raw:.2f} safe={safe:.2f} → multiplier={multiplier:.2f} "
        f"(n={n_trades}, src={source})"
    )

    return {
        "multiplier": round(multiplier, 4),
        "kelly_raw":  round(raw, 4),
        "kelly_safe": round(safe, 4),
        "win_prob":   round(p, 4),
        "payoff":     round(payoff, 4),
        "n_trades":   n_trades,
        "source":     source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard helper
# ─────────────────────────────────────────────────────────────────────────────

def get_kelly_dashboard_stats(days: int = 90) -> dict:
    """
    Return edge stats + current Kelly fraction for Settings/Risk dashboard display.
    """
    edge = _load_edge_stats(days=days)
    if edge is None:
        from ai.feedback_loop import get_ai_accuracy_stats
        acc = get_ai_accuracy_stats(days=days)
        return {
            "status":    "insufficient_data",
            "n_trades":  acc.get("total_decided", 0),
            "min_trades": MIN_TRADES,
            "message":   f"Need {MIN_TRADES} completed trades; have {acc.get('total_decided', 0)}",
        }

    raw  = kelly_fraction(edge["win_rate"], edge["payoff"])
    safe = raw * KELLY_FRACTION
    mult = max(KELLY_FLOOR, min(KELLY_CAP, safe))
    return {
        "status":         "active",
        "win_rate":       edge["win_rate"],
        "avg_win":        edge["avg_win"],
        "avg_loss":       edge["avg_loss"],
        "payoff":         edge["payoff"],
        "n_trades":       edge["n_trades"],
        "kelly_raw":      round(raw, 4),
        "kelly_safe":     round(safe, 4),
        "kelly_fraction": KELLY_FRACTION,
        "current_multiplier": mult,
        "days":           days,
    }
