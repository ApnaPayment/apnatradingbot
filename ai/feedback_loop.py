"""
AI Feedback Loop
Records every AI trade decision and its eventual outcome.
Runs a weekly self-review: asks Claude to analyse its own decisions,
identify patterns in misses, and suggest how it should reason differently.

The review is stored in DB and shown on the Settings page.
Over time this creates a natural improvement loop where Claude's
context about its own past performance shapes future decisions.

Weekly review run: every Sunday at 01:00 (after optimisation at 00:30).
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "market_data.db"


# ─────────────────────────────────────────────────────────────────────────────
# Schema bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_tables():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                decided_at      TEXT NOT NULL,
                symbol          TEXT NOT NULL,
                action          TEXT,           -- BUY / SELL
                strategy        TEXT,
                signal_conf     REAL,           -- confidence at signal time
                signal_price    REAL,
                approved        INTEGER,        -- 1 approved, 0 vetoed
                ai_confidence_adj REAL DEFAULT 0,
                reasoning       TEXT,
                concerns        TEXT,           -- JSON list
                tools_used      TEXT,           -- JSON list
                -- outcome (filled in when trade closes)
                outcome_pnl     REAL,           -- NULL until closed
                outcome_correct INTEGER,        -- 1 = AI was right, 0 = AI was wrong, NULL = pending
                exit_reason     TEXT,
                closed_at       TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_weekly_reviews (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                week_start  TEXT NOT NULL,
                decisions   INTEGER,
                approvals   INTEGER,
                vetoes      INTEGER,
                accuracy    REAL,
                review_text TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        # Index for fast lookup by symbol+time
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ai_decisions_symbol "
            "ON ai_decisions(symbol, decided_at DESC)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Write helpers
# ─────────────────────────────────────────────────────────────────────────────

def record_ai_decision(symbol: str, action: str, strategy: str,
                       signal_conf: float, signal_price: float,
                       ai_result: dict) -> Optional[int]:
    """
    Persist an AI evaluation result immediately after the decision.
    Returns the row ID so _exit_position() can link the outcome back.
    """
    _ensure_tables()
    approved = 1 if ai_result.get("approved", True) else 0
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                """INSERT INTO ai_decisions
                   (decided_at, symbol, action, strategy, signal_conf, signal_price,
                    approved, ai_confidence_adj, reasoning, concerns, tools_used)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now().isoformat(),
                    symbol, action, strategy,
                    signal_conf, signal_price,
                    approved,
                    ai_result.get("confidence_adjustment", 0.0),
                    ai_result.get("reasoning", ""),
                    json.dumps(ai_result.get("concerns", [])),
                    json.dumps(ai_result.get("tools_used", [])),
                )
            )
            return cur.lastrowid
    except Exception as e:
        logger.warning(f"Failed to record AI decision: {e}")
        return None


def update_ai_decision_outcome(decision_id: int, pnl: float,
                                exit_reason: str, approved_was: bool):
    """
    Fill in the outcome once a trade closes.
    correct = approved=True and pnl>0, OR approved=False (veto) and trade would have lost.
    For vetoes we don't have the actual P&L — we mark outcome_pnl=0 and correct=NULL.
    """
    if decision_id is None:
        return
    try:
        correct: Optional[int] = None
        if approved_was:
            correct = 1 if pnl > 0 else 0   # approved → was it profitable?
        _ensure_tables()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """UPDATE ai_decisions
                   SET outcome_pnl=?, outcome_correct=?, exit_reason=?, closed_at=?
                   WHERE id=?""",
                (
                    round(pnl, 2),
                    correct,
                    exit_reason,
                    datetime.now().isoformat(),
                    decision_id,
                )
            )
    except Exception as e:
        logger.warning(f"Failed to update AI decision outcome: {e}")


def link_trade_to_decision(symbol: str, decided_after: str) -> Optional[int]:
    """
    Find the most recent un-linked AI decision for a symbol made after `decided_after`.
    Used by _exit_position() which may not have the original decision_id in scope.
    """
    _ensure_tables()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                """SELECT id FROM ai_decisions
                   WHERE symbol=? AND approved=1 AND outcome_pnl IS NULL
                     AND decided_at >= ?
                   ORDER BY id DESC LIMIT 1""",
                (symbol, decided_after)
            ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Analytics queries
# ─────────────────────────────────────────────────────────────────────────────

def get_ai_accuracy_stats(days: int = 30) -> dict:
    """
    Compute AI decision accuracy over the last `days` days.
    Returns stats dict for the dashboard.
    """
    _ensure_tables()
    since = (datetime.now() - timedelta(days=days)).isoformat()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                """SELECT approved, outcome_correct, outcome_pnl
                   FROM ai_decisions
                   WHERE decided_at >= ? AND outcome_correct IS NOT NULL""",
                (since,)
            ).fetchall()
    except Exception:
        return {}

    if not rows:
        return {"days": days, "total_decided": 0, "message": "No completed decisions yet"}

    total     = len(rows)
    correct   = sum(1 for r in rows if r[1] == 1)
    approvals = sum(1 for r in rows if r[0] == 1)
    vetoes    = total - approvals

    approved_rows = [r for r in rows if r[0] == 1 and r[2] is not None]
    avg_pnl = sum(r[2] for r in approved_rows) / len(approved_rows) if approved_rows else 0

    return {
        "days":              days,
        "total_decided":     total,
        "approvals":         approvals,
        "vetoes":            vetoes,
        "correct":           correct,
        "accuracy":          round(correct / total, 3) if total > 0 else 0,
        "avg_pnl_on_approved": round(avg_pnl, 2),
    }


def get_recent_decisions(limit: int = 20) -> list[dict]:
    """Fetch recent AI decisions with outcome for the dashboard."""
    _ensure_tables()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                """SELECT id, decided_at, symbol, action, strategy, signal_conf,
                          approved, reasoning, outcome_pnl, outcome_correct, exit_reason
                   FROM ai_decisions
                   ORDER BY id DESC LIMIT ?""",
                (limit,)
            ).fetchall()
        return [
            {
                "id":             r[0],
                "decided_at":     r[1],
                "symbol":         r[2],
                "action":         r[3],
                "strategy":       r[4],
                "signal_conf":    r[5],
                "approved":       bool(r[6]),
                "reasoning":      r[7],
                "outcome_pnl":    r[8],
                "outcome_correct": r[9],
                "exit_reason":    r[10],
            }
            for r in rows
        ]
    except Exception:
        return []


def get_weekly_reviews(limit: int = 4) -> list[dict]:
    """Fetch recent weekly review entries."""
    _ensure_tables()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                """SELECT week_start, decisions, approvals, vetoes, accuracy, review_text, created_at
                   FROM ai_weekly_reviews ORDER BY id DESC LIMIT ?""",
                (limit,)
            ).fetchall()
        return [
            {
                "week_start":  r[0], "decisions": r[1],
                "approvals":   r[2], "vetoes": r[3],
                "accuracy":    r[4], "review_text": r[5],
                "created_at":  r[6],
            }
            for r in rows
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Weekly self-review
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackLoop:
    """
    Runs the weekly self-review: fetches the last 7 days of AI decisions +
    outcomes, asks Claude to analyse its own performance, and saves the review.
    """

    def __init__(self, ai_engine):
        self.ai = ai_engine

    def run_weekly_review(self) -> Optional[str]:
        """
        Pull last 7 days of decisions, ask Claude to self-critique,
        save to DB. Returns the review text.
        """
        _ensure_tables()
        since = (datetime.now() - timedelta(days=7)).isoformat()

        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute(
                    """SELECT symbol, action, strategy, signal_conf, approved,
                              reasoning, concerns, outcome_pnl, outcome_correct, exit_reason
                       FROM ai_decisions
                       WHERE decided_at >= ?
                       ORDER BY id ASC""",
                    (since,)
                ).fetchall()
        except Exception as e:
            logger.error(f"Feedback loop DB read failed: {e}")
            return None

        if not rows:
            logger.info("No AI decisions in last 7 days — skipping review")
            return None

        # Format decisions for the prompt
        decision_lines = []
        total = len(rows)
        correct = sum(1 for r in rows if r[8] == 1)
        approved_count = sum(1 for r in rows if r[4] == 1)

        for r in rows:
            symbol, action, strategy, conf, approved = r[0], r[1], r[2], r[3], r[4]
            reasoning, concerns, pnl, correct_flag, exit_reason = r[5], r[6], r[7], r[8], r[9]
            decision_word = "APPROVED" if approved else "VETOED"
            outcome = f"P&L=₹{pnl:,.0f} ({exit_reason})" if pnl is not None else "pending"
            correct_str = {1: "✓ correct", 0: "✗ wrong", None: "pending"}[correct_flag]
            decision_lines.append(
                f"  [{decision_word}] {symbol} {action} conf={conf:.0%} → {outcome} {correct_str}"
                f"\n    Reasoning: {str(reasoning)[:100]}"
            )

        decisions_text = "\n".join(decision_lines[:30])  # cap at 30 to avoid token overflow
        accuracy = round(correct / max(total, 1) * 100, 1)

        if not self.ai or not self.ai.client:
            logger.info("AI not configured — skipping self-review generation")
            review_text = (
                f"[Week of {datetime.now().strftime('%d %b %Y')}] "
                f"AI not configured. {total} decisions, {accuracy:.0f}% accuracy."
            )
        else:
            prompt = f"""You are reviewing your own trading decisions from the past week.

WEEK SUMMARY:
- Total decisions: {total}
- Approvals: {approved_count}  |  Vetoes: {total - approved_count}
- Accuracy (where outcome known): {accuracy:.1f}%

DECISION LOG (most recent {min(total, 30)} decisions):
{decisions_text}

Analyse your performance critically:
1. Where did you make the most mistakes? (approved losers or vetoed winners?)
2. What patterns do you see in your wrong decisions?
3. What should you weigh differently next week?
4. Any specific setup types you should be more/less aggressive on?

Be specific and self-critical. Reference actual symbols/setups from the log.
Keep under 200 words. This will be shown to the bot operator."""

            try:
                review_text = self.ai._call_claude(prompt)
            except Exception as e:
                logger.error(f"Weekly review Claude call failed: {e}")
                review_text = f"Review generation failed: {e}"

        # Save to DB
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO ai_weekly_reviews
                       (week_start, decisions, approvals, vetoes, accuracy, review_text)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        since[:10],
                        total,
                        approved_count,
                        total - approved_count,
                        round(accuracy / 100, 3),
                        review_text,
                    )
                )
        except Exception as e:
            logger.error(f"Failed to save weekly review: {e}")

        logger.info(
            f"Weekly AI self-review complete: {total} decisions, {accuracy:.1f}% accuracy"
        )
        return review_text
