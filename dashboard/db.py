"""
Dashboard SQLite helpers — direct database access independent of algo_trader package.
Used by Dash app to read market_data.db without coupling to the bot process.
"""

import sqlite3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "data" / "market_data.db"


def get_conn():
    """Return a SQLite connection with WAL mode and busy timeout for concurrent access."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")   # 30s — matches liquidity_db; bot writes take up to 3-4 min for scrip master but dashboard reads are short
    conn.execute("PRAGMA synchronous=NORMAL")   # faster writes, safe with WAL
    conn.row_factory = sqlite3.Row
    return conn


def get_bot_status() -> dict:
    """Read current bot state (single row)."""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM bot_status WHERE id=1").fetchone()
        if not row:
            return {}
        cols = [d[0] for d in conn.execute(
            "SELECT * FROM bot_status LIMIT 0").description]
        data = dict(zip(cols, row))
        try:
            data["open_positions"] = json.loads(data.get("open_positions", "{}"))
        except Exception:
            data["open_positions"] = {}
        return data


def get_trades(limit: int = 200, symbol: str = None) -> pd.DataFrame:
    """Fetch completed trade history."""
    with get_conn() as conn:
        if symbol:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE symbol=? ORDER BY exit_time DESC LIMIT ?",
                conn, params=(symbol, limit)
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM trades ORDER BY exit_time DESC LIMIT ?",
                conn, params=(limit,)
            )
    return df


def get_events(limit: int = 50, event_type: str = None) -> pd.DataFrame:
    """Fetch recent events for the signal feed."""
    with get_conn() as conn:
        if event_type:
            df = pd.read_sql_query(
                """SELECT id, timestamp, event_type, message, symbol, metadata
                   FROM events WHERE event_type=?
                   ORDER BY timestamp DESC LIMIT ?""",
                conn, params=(event_type, limit)
            )
        else:
            df = pd.read_sql_query(
                """SELECT id, timestamp, event_type, message, symbol, metadata
                   FROM events ORDER BY timestamp DESC LIMIT ?""",
                conn, params=(limit,)
            )
    return df


def get_ohlcv(symbol: str, exchange: str = "nse_cm", limit: int = 200) -> pd.DataFrame:
    """Get OHLCV history for a symbol."""
    with get_conn() as conn:
        df = pd.read_sql_query(
            """SELECT replace(timestamp,'T',' ') as timestamp,
                      open, high, low, close, volume
               FROM ohlcv WHERE symbol=? AND exchange=?
               ORDER BY replace(timestamp,'T',' ') DESC LIMIT ?""",
            conn, params=(symbol, exchange, limit)
        )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df.sort_values("timestamp").reset_index(drop=True)


def get_live_quote(symbol: str, exchange: str = "nse_cm") -> dict:
    """Get the most recent quote for a symbol."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT symbol, exchange, ltp, volume, bid, ask, timestamp
               FROM tick_data WHERE symbol=? AND exchange=?
               ORDER BY timestamp DESC LIMIT 1""",
            (symbol, exchange)
        ).fetchone()
    if row:
        return {
            "symbol": row[0],
            "exchange": row[1],
            "ltp": row[2],
            "volume": row[3],
            "bid": row[4],
            "ask": row[5],
            "timestamp": row[6],
        }
    return {}


def search_symbol(query: str, exchange: str = "nse_cm") -> list:
    """Search scrip master by partial symbol/name."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT symbol, name, token, exchange, series, lot_size
               FROM scrip_master
               WHERE (symbol LIKE ? OR name LIKE ?) AND exchange=?
               LIMIT 20""",
            (f"%{query}%", f"%{query.upper()}%", exchange)
        ).fetchall()
        return [dict(zip(
            ["symbol", "name", "token", "exchange", "series", "lot_size"], r
        )) for r in rows]


def get_symbol_list(exchange: str = "nse_cm") -> list:
    """Get symbols that have OHLCV data, with watchlist pinned to top."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol LIMIT 200",
        ).fetchall()
    symbols = set(r[0] for r in rows)
    watchlist = [
        "NIFTYBEES-EQ", "RELIANCE-EQ", "TCS-EQ", "INFY-EQ", "HDFCBANK-EQ",
        "ICICIBANK-EQ", "WIPRO-EQ", "LT-EQ", "AXISBANK-EQ", "ITBEES-EQ",
    ]
    pinned = [s for s in watchlist if s in symbols]
    rest   = sorted(s for s in symbols if s not in watchlist)
    return pinned + rest if pinned + rest else watchlist


def db_stats() -> dict:
    """Summary of what's in the database."""
    with get_conn() as conn:
        scrip_count = conn.execute("SELECT COUNT(*) FROM scrip_master").fetchone()[0]
        ohlcv_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        tick_count  = conn.execute("SELECT COUNT(*) FROM tick_data").fetchone()[0]
        trade_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        # Liquidity engine tables (may not exist on first run)
        try:
            liq_trade_count = conn.execute("SELECT COUNT(*) FROM liq_trades").fetchone()[0]
        except Exception:
            liq_trade_count = 0
    return {
        "scrip_instruments": scrip_count,
        "ohlcv_candles":     ohlcv_count,
        "ticks_stored":      tick_count,
        "trades_recorded":   trade_count,
        "liq_trades":        liq_trade_count,
        "events_logged":     event_count,
    }


# ─── Liquidity Engine helpers ─────────────────────────────────────────────────

def get_liq_bot_status() -> dict:
    """Read liq_bot_status for the Liquidity Engine dashboard panel."""
    with get_conn() as conn:
        try:
            row = conn.execute("SELECT * FROM liq_bot_status WHERE id=1").fetchone()
        except Exception:
            return {}
        if not row:
            return {}
        cols = [d[0] for d in conn.execute(
            "SELECT * FROM liq_bot_status LIMIT 0").description]
        data = dict(zip(cols, row))
        try:
            data["open_positions"] = json.loads(data.get("open_positions", "{}"))
            data["segments_active"] = json.loads(data.get("segments_active", "[]"))
        except Exception:
            data["open_positions"] = {}
            data["segments_active"] = []
        return data


def get_liq_trades(limit: int = 200, exchange: str = None,
                   since_date: str = None) -> pd.DataFrame:
    """Fetch completed Liquidity Engine trade history (MCX, BSE paper trades)."""
    with get_conn() as conn:
        try:
            conditions = []
            params: list = []
            if exchange:
                conditions.append("exchange=?")
                params.append(exchange)
            if since_date:
                conditions.append("entry_time >= ?")
                params.append(since_date)
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)
            df = pd.read_sql_query(
                f"SELECT * FROM liq_trades {where} ORDER BY id DESC LIMIT ?",
                conn, params=params
            )
        except Exception:
            return pd.DataFrame()
    return df


def get_liq_today_stats() -> dict:
    """Today's paper-trade stats from the Liquidity Engine."""
    from datetime import date
    today = date.today().isoformat()
    with get_conn() as conn:
        try:
            row = conn.execute(
                """SELECT
                     COUNT(*)                                   AS trades,
                     COALESCE(SUM(pnl), 0)                     AS total_pnl,
                     SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)  AS wins,
                     SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END)  AS losses,
                     SUM(CASE WHEN exchange='mcx_fo' THEN 1 ELSE 0 END) AS mcx_trades,
                     SUM(CASE WHEN exchange='bse_cm' THEN 1 ELSE 0 END) AS bse_trades
                   FROM liq_trades WHERE entry_time >= ?""",
                (today,)
            ).fetchone()
            if row:
                return {
                    "trades": row[0], "total_pnl": row[1],
                    "wins": row[2] or 0, "losses": row[3] or 0,
                    "mcx_trades": row[4] or 0, "bse_trades": row[5] or 0,
                }
        except Exception:
            pass
    return {"trades": 0, "total_pnl": 0, "wins": 0, "losses": 0,
            "mcx_trades": 0, "bse_trades": 0}


def get_ai_decisions(limit: int = 200, symbol: str = None) -> pd.DataFrame:
    """Fetch AI decision history (both approved and vetoed) for chart markers."""
    with get_conn() as conn:
        try:
            if symbol:
                df = pd.read_sql_query(
                    "SELECT * FROM ai_decisions WHERE symbol=? ORDER BY decided_at DESC LIMIT ?",
                    conn, params=(symbol, limit)
                )
            else:
                df = pd.read_sql_query(
                    "SELECT * FROM ai_decisions ORDER BY decided_at DESC LIMIT ?",
                    conn, params=(limit,)
                )
            return df
        except Exception:
            return pd.DataFrame()
