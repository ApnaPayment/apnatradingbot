"""
Liquidity Engine — Database Layer
Writes to liq_trades, liq_bot_status, liq_ai_decisions tables.
Uses the same SQLite DB as the NSE bot (market_data.db) — WAL mode handles concurrency.
Zero modification to existing tables.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Same DB file as the NSE bot
_DB_PATH = Path(__file__).parent.parent / "data" / "market_data.db"


@contextmanager
def _get_conn():
    conn = sqlite3.connect(str(_DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")   # 30s wait — matches NSE bot's long writes
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_tables():
    """Create liq_* tables if they don't exist. Called once on startup."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS liq_trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                exchange    TEXT NOT NULL,
                action      TEXT,
                strategy    TEXT,
                entry_price REAL,
                exit_price  REAL,
                quantity    INTEGER,
                pnl         REAL,
                exit_reason TEXT,
                entry_time  TEXT,
                exit_time   TEXT,
                order_no    TEXT
            );

            CREATE TABLE IF NOT EXISTS liq_bot_status (
                id              INTEGER PRIMARY KEY DEFAULT 1,
                daily_pnl       REAL    DEFAULT 0,
                daily_trades    INTEGER DEFAULT 0,
                open_positions  TEXT    DEFAULT '{}',
                last_cycle      TEXT,
                status          TEXT    DEFAULT 'stopped',
                market_regime   TEXT    DEFAULT 'unknown',
                segments_active TEXT    DEFAULT '[]'
            );

            INSERT OR IGNORE INTO liq_bot_status (id) VALUES (1);

            CREATE TABLE IF NOT EXISTS liq_ai_decisions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                decided_at   TEXT,
                symbol       TEXT NOT NULL,
                exchange     TEXT,
                action       TEXT,
                strategy     TEXT,
                signal_price REAL,
                signal_conf  REAL,
                approved     INTEGER,
                ai_reasoning TEXT,
                veto_reason  TEXT,
                stop_loss    REAL,
                target       REAL,
                market_regime TEXT,
                outcome      TEXT,
                outcome_pnl  REAL,
                closed_at    TEXT
            );

            CREATE TABLE IF NOT EXISTS session (
                id          INTEGER PRIMARY KEY DEFAULT 1,
                token       TEXT,
                sid         TEXT,
                base_url    TEXT,
                saved_at    TEXT
            );
            INSERT OR IGNORE INTO session (id) VALUES (1);
        """)
    logger.info("LiqDB: tables ensured (liq_trades, liq_bot_status, liq_ai_decisions, session)")


class LiquidityDataManager:
    """
    Database interface for the Liquidity Engine.
    Reads/writes only liq_* tables. The NSE bot's tables are never touched.
    """

    def __init__(self):
        _ensure_tables()

    # ─── Bot status ──────────────────────────────────────────────────────────

    def get_bot_status(self) -> dict:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM liq_bot_status WHERE id=1"
            ).fetchone()
            if not row:
                return {}
            d = dict(row)
            d["open_positions"] = json.loads(d.get("open_positions") or "{}")
            d["segments_active"] = json.loads(d.get("segments_active") or "[]")
            return d

    def update_bot_status(self, *,
                          daily_pnl: float = None,
                          daily_trades: int = None,
                          open_positions: dict = None,
                          status: str = None,
                          market_regime: str = None,
                          segments_active: list = None) -> None:
        fields = []
        vals = []
        if daily_pnl is not None:
            fields.append("daily_pnl=?"); vals.append(daily_pnl)
        if daily_trades is not None:
            fields.append("daily_trades=?"); vals.append(daily_trades)
        if open_positions is not None:
            fields.append("open_positions=?"); vals.append(json.dumps(open_positions))
        if status is not None:
            fields.append("status=?"); vals.append(status)
        if market_regime is not None:
            fields.append("market_regime=?"); vals.append(market_regime)
        if segments_active is not None:
            fields.append("segments_active=?"); vals.append(json.dumps(segments_active))
        fields.append("last_cycle=?"); vals.append(datetime.now().isoformat())
        if not fields:
            return
        sql = f"UPDATE liq_bot_status SET {', '.join(fields)} WHERE id=1"
        with _get_conn() as conn:
            conn.execute(sql, vals)

    def reset_daily(self) -> None:
        """Called at start of each trading day."""
        with _get_conn() as conn:
            conn.execute(
                "UPDATE liq_bot_status SET daily_pnl=0, daily_trades=0, "
                "open_positions='{}' WHERE id=1"
            )
        logger.info("LiqDB: daily state reset")

    # ─── Trades ──────────────────────────────────────────────────────────────

    def record_trade(self, *, symbol: str, exchange: str, action: str,
                     strategy: str, entry_price: float, exit_price: float,
                     quantity: int, pnl: float, exit_reason: str,
                     entry_time: str, exit_time: str, order_no: str) -> int:
        with _get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO liq_trades
                   (symbol, exchange, action, strategy, entry_price, exit_price,
                    quantity, pnl, exit_reason, entry_time, exit_time, order_no)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol, exchange, action, strategy, entry_price, exit_price,
                 quantity, pnl, exit_reason, entry_time, exit_time, order_no)
            )
            return cur.lastrowid

    def get_trades(self, *, since_date: str = None, limit: int = 100) -> list:
        with _get_conn() as conn:
            if since_date:
                rows = conn.execute(
                    "SELECT * FROM liq_trades WHERE entry_time >= ? ORDER BY id DESC LIMIT ?",
                    (since_date, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM liq_trades ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_today_stats(self) -> dict:
        today = date.today().isoformat()
        with _get_conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as trades,
                          COALESCE(SUM(pnl),0) as total_pnl,
                          SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                          SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses
                   FROM liq_trades WHERE entry_time >= ?""",
                (today,)
            ).fetchone()
            return dict(row) if row else {"trades": 0, "total_pnl": 0, "wins": 0, "losses": 0}

    # ─── AI Decisions ────────────────────────────────────────────────────────

    def record_ai_decision(self, *, symbol: str, exchange: str, action: str,
                           strategy: str, signal_price: float, signal_conf: float,
                           approved: bool, ai_reasoning: str, veto_reason: str = "",
                           stop_loss: float = 0, target: float = 0,
                           market_regime: str = "unknown") -> int:
        with _get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO liq_ai_decisions
                   (decided_at, symbol, exchange, action, strategy, signal_price,
                    signal_conf, approved, ai_reasoning, veto_reason,
                    stop_loss, target, market_regime)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (datetime.now().isoformat(), symbol, exchange, action, strategy,
                 signal_price, signal_conf, int(approved), ai_reasoning, veto_reason,
                 stop_loss, target, market_regime)
            )
            return cur.lastrowid

    def update_ai_decision_outcome(self, decision_id: int, outcome: str,
                                   pnl: float) -> None:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE liq_ai_decisions SET outcome=?, outcome_pnl=?, closed_at=? WHERE id=?",
                (outcome, pnl, datetime.now().isoformat(), decision_id)
            )

    # ─── Shared Kotak session token ──────────────────────────────────────────

    def save_session_token(self, token: str, sid: str, base_url: str) -> None:
        """
        Persist the NSE bot's Kotak session credentials to the shared DB.
        Called by main.py after successful authentication so the MCX engine
        can use the same session without triggering re-authentication.
        """
        with _get_conn() as conn:
            conn.execute(
                "UPDATE session SET token=?, sid=?, base_url=?, saved_at=? WHERE id=1",
                (token, sid, base_url, datetime.now().isoformat())
            )
        logger.info("LiqDB: Kotak session token saved to shared DB")

    def get_session_token(self) -> Optional[dict]:
        """
        Read the Kotak session token saved by the NSE bot.
        Returns {token, sid, base_url, saved_at} or None if not set/expired.
        Token is considered stale after 6 hours (Kotak sessions last ~8h).
        """
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT token, sid, base_url, saved_at FROM session WHERE id=1"
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        if not d.get("token") or not d.get("sid"):
            return None
        # Check freshness — reject if older than 6 hours
        saved_at = d.get("saved_at")
        if saved_at:
            try:
                age_hours = (datetime.now() - datetime.fromisoformat(saved_at)).total_seconds() / 3600
                if age_hours > 6:
                    logger.warning(f"LiqDB: Kotak session token is {age_hours:.1f}h old — treating as stale")
                    return None
            except Exception:
                pass
        return d

    # ─── DB stats ────────────────────────────────────────────────────────────

    def get_db_stats(self) -> dict:
        with _get_conn() as conn:
            trades = conn.execute("SELECT COUNT(*) FROM liq_trades").fetchone()[0]
            decisions = conn.execute("SELECT COUNT(*) FROM liq_ai_decisions").fetchone()[0]
            return {"liq_trades": trades, "liq_ai_decisions": decisions}
