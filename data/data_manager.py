"""
Data Manager
Downloads scrip master, fetches live quotes, builds OHLCV history.
Stores everything in SQLite for fast local access.
"""

import os
import time
import sqlite3
import logging
import requests
import pandas as pd
from datetime import datetime, date
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "market_data.db"


class DataManager:
    """
    Central data layer.
    - Scrip master: daily download + SQLite cache
    - Live quotes: real-time polling via Kotak API
    - OHLCV: builds price history from tick data
    """

    def __init__(self, kotak_client=None):
        self.client = kotak_client
        DB_PATH.parent.mkdir(exist_ok=True)

        # Enable WAL mode for better concurrency and configure timeouts
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety and speed
            conn.execute("PRAGMA busy_timeout=300000")  # 5min timeout on lock (scrip master download takes 3-4 min)
            conn.close()
            logger.info("SQLite WAL mode enabled for better concurrency")
        except Exception as e:
            logger.warning(f"Failed to configure SQLite: {e}")

        self._init_db()

    # ─────────────────────────────────────────────────────────────────────────
    # Database setup
    # ─────────────────────────────────────────────────────────────────────────

    def _get_conn(self):
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=300000")  # 5min wait on lock (scrip master download takes 3-4 min)
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scrip_master (
                    token        TEXT,
                    symbol       TEXT,
                    name         TEXT,
                    exchange     TEXT,
                    segment      TEXT,
                    series       TEXT,
                    isin         TEXT,
                    lot_size     INTEGER DEFAULT 1,
                    tick_size    REAL DEFAULT 0.05,
                    last_updated DATE,
                    PRIMARY KEY (token, exchange)
                );

                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol    TEXT,
                    exchange  TEXT,
                    timestamp TEXT,
                    open      REAL,
                    high      REAL,
                    low       REAL,
                    close     REAL,
                    volume    INTEGER,
                    PRIMARY KEY (symbol, exchange, timestamp)
                );

                CREATE TABLE IF NOT EXISTS tick_data (
                    symbol    TEXT,
                    exchange  TEXT,
                    timestamp TEXT,
                    ltp       REAL,
                    volume    INTEGER,
                    bid       REAL,
                    ask       REAL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol       TEXT,
                    exchange     TEXT,
                    action       TEXT,
                    strategy     TEXT,
                    entry_price  REAL,
                    exit_price   REAL,
                    quantity     INTEGER,
                    pnl          REAL,
                    exit_reason  TEXT,
                    entry_time   TEXT,
                    exit_time    TEXT,
                    order_no     TEXT
                );

                CREATE TABLE IF NOT EXISTS bot_status (
                    id               INTEGER PRIMARY KEY CHECK (id = 1),
                    daily_pnl        REAL    DEFAULT 0,
                    daily_trades     INTEGER DEFAULT 0,
                    open_positions   TEXT    DEFAULT '{}',
                    capital_at_risk  REAL    DEFAULT 0,
                    paper_trading         INTEGER DEFAULT 1,
                    last_cycle            TEXT,
                    status                TEXT    DEFAULT 'stopped',
                    market_regime         TEXT    DEFAULT 'unknown',
                    ai_regime_confidence  REAL    DEFAULT 0,
                    ai_regime_suggestion  TEXT    DEFAULT '',
                    ai_regime_risk        TEXT    DEFAULT 'medium'
                );

                CREATE TABLE IF NOT EXISTS events (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event_type TEXT,
                    message   TEXT,
                    symbol    TEXT,
                    metadata  TEXT DEFAULT '{}'
                );

                -- Structured AI decision log (one row per signal evaluated)
                -- NOTE: existing DBs may have this table with different column names;
                -- missing columns are added by the migration block below.
                CREATE TABLE IF NOT EXISTS ai_decisions (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    decided_at        TEXT,
                    symbol            TEXT NOT NULL,
                    action            TEXT,
                    strategy          TEXT,
                    signal_conf       REAL,
                    signal_price      REAL,
                    approved          INTEGER,
                    ai_confidence_adj REAL DEFAULT 0,
                    reasoning         TEXT,
                    concerns          TEXT,
                    tools_used        TEXT,
                    outcome_pnl       REAL,
                    outcome_correct   INTEGER,
                    exit_reason       TEXT,
                    closed_at         TEXT
                );

                -- Daily performance summary (one row per calendar day the bot ran)
                CREATE TABLE IF NOT EXISTS daily_journal (
                    date             TEXT PRIMARY KEY,
                    market_regime    TEXT,
                    signals_seen     INTEGER DEFAULT 0,
                    signals_approved INTEGER DEFAULT 0,
                    signals_vetoed   INTEGER DEFAULT 0,
                    trades_opened    INTEGER DEFAULT 0,
                    trades_closed    INTEGER DEFAULT 0,
                    day_pnl          REAL    DEFAULT 0,
                    cumulative_pnl   REAL    DEFAULT 0,
                    win_count        INTEGER DEFAULT 0,
                    loss_count       INTEGER DEFAULT 0,
                    open_positions   INTEGER DEFAULT 0,
                    notes            TEXT    DEFAULT '',
                    created_at       TEXT
                );

                INSERT OR IGNORE INTO bot_status (id) VALUES (1);

                CREATE INDEX IF NOT EXISTS idx_scrip_symbol    ON scrip_master(symbol);
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol    ON ohlcv(symbol, exchange);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol   ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_time     ON trades(exit_time);
                CREATE INDEX IF NOT EXISTS idx_events_time     ON events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_ai_decisions_ts ON ai_decisions(decided_at);
                CREATE INDEX IF NOT EXISTS idx_ai_decisions_sym ON ai_decisions(symbol);
                CREATE INDEX IF NOT EXISTS idx_journal_date    ON daily_journal(date);
            """)
        # Migrate existing DBs — add columns introduced after initial schema
        # bot_status columns added after initial schema
        _bot_cols = [
            ("bot_status",    "market_regime",        "TEXT",    "'unknown'"),
            ("bot_status",    "ai_regime_confidence", "REAL",    "0"),
            ("bot_status",    "ai_regime_suggestion", "TEXT",    "''"),
            ("bot_status",    "ai_regime_risk",       "TEXT",    "'medium'"),
            # ai_decisions: extended columns for structured decision logging
            ("ai_decisions",  "final_confidence",     "REAL",    "0"),
            ("ai_decisions",  "signal_confidence",    "REAL",    "0"),
            ("ai_decisions",  "veto_reason",          "TEXT",    "''"),
            ("ai_decisions",  "ai_reasoning",         "TEXT",    "''"),
            ("ai_decisions",  "stop_loss",            "REAL",    "0"),
            ("ai_decisions",  "target",               "REAL",    "0"),
            ("ai_decisions",  "market_regime",        "TEXT",    "'unknown'"),
            ("ai_decisions",  "outcome",              "TEXT",    "NULL"),
            ("ai_decisions",  "good_decision",        "INTEGER", "NULL"),
        ]
        with self._get_conn() as conn:
            for table, col, ctype, default in _bot_cols:
                try:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} {ctype} DEFAULT {default}"
                    )
                except Exception as _e:
                    if "duplicate column" not in str(_e).lower():
                        logger.warning(f"Schema migration warning [{table}.{col}]: {_e}")
        logger.info("Database initialized")

    # ─────────────────────────────────────────────────────────────────────────
    # Scrip master
    # ─────────────────────────────────────────────────────────────────────────

    def download_scrip_master(self) -> bool:
        """
        Downloads the latest scrip master CSV from Kotak Neo.
        Should be called once daily before market open.
        """
        if not self.client:
            logger.error("No Kotak client attached")
            return False

        required_columns = ["token", "symbol", "exchange"]

        try:
            logger.info("Fetching scrip master file paths...")
            response = self.client.get_scrip_master_paths()
            data = response.get("data", {})
            # Response: {"data": {"filesPaths": ["url1", "url2", ...]}}
            file_urls = data.get("filesPaths") or data.get("filePaths") or []
            if not file_urls:
                logger.error(f"No scrip master files in response: {response}")
                return False

            # Clear stale scrip master before fresh download
            with self._get_conn() as conn:
                conn.execute("DELETE FROM scrip_master")
                conn.commit()
            logger.info("Cleared old scrip master. Downloading fresh data...")

            total_inserted = 0
            for url in file_urls:
                # Derive segment name from filename (e.g. nse_cm-v1.csv → nse_cm)
                filename = url.split("/")[-1].replace(".csv", "").replace("-v1", "")
                segment = filename

                try:
                    logger.info(f"Downloading scrip master: {segment}")
                    df = pd.read_csv(url)
                    # Strip whitespace from column names (Kotak CSVs have trailing spaces)
                    df.columns = df.columns.str.strip()

                    # Map Kotak column names → our internal names
                    # Kotak "transformed" format: pSymbol=token, pTrdSymbol=symbol, pExchSeg=exchange
                    col_map = {
                        "pSymbol":      "token",
                        "pTrdSymbol":   "symbol",
                        "pExchSeg":     "exchange",
                        "pInstName":    "name",
                        "pISIN":        "isin",
                        "lLotSize":     "lot_size",
                        "dTickSize":    "tick_size",
                        "pSegment":     "series",
                        # fallback for older format
                        "token": "token", "symbol": "symbol",
                        "instrumentname": "name", "lotsize": "lot_size",
                        "ticksize": "tick_size", "isin": "isin",
                    }
                    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

                    # Validate required columns exist after mapping
                    missing = [c for c in required_columns if c not in df.columns]
                    if missing:
                        logger.error(f"Missing required columns after mapping: {missing}. Available: {list(df.columns)}")
                        continue

                    # Validate data quality
                    df = df.dropna(subset=["token", "symbol"])
                    if df.empty:
                        logger.error("No valid rows after removing NULLs")
                        continue

                    df["last_updated"] = str(date.today())
                    df["segment"] = segment

                    keep_cols = ["token", "symbol", "name", "exchange", "segment",
                                 "series", "isin", "lot_size", "tick_size", "last_updated"]
                    df = df[[c for c in keep_cols if c in df.columns]]

                    with self._get_conn() as conn:
                        df.to_sql("scrip_master", conn, if_exists="append",
                                  index=False)
                    total_inserted += len(df)
                    logger.info(f"Inserted {len(df)} instruments for {segment}")

                except Exception as e:
                    logger.error(f"Failed to process {segment}: {e}")
                    continue

            if total_inserted == 0:
                logger.error("No instruments inserted. Scrip master download failed!")
                return False

            logger.info(f"Scrip master updated: {total_inserted} instruments")
            return True

        except Exception as e:
            logger.error(f"Scrip master download failed: {e}")
            return False

    def get_instrument(self, symbol: str, exchange: str = "nse_cm") -> Optional[dict]:
        """Look up instrument details by symbol."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM scrip_master WHERE symbol=? AND exchange=? LIMIT 1",
                (symbol.upper(), exchange)
            ).fetchone()
            if row:
                cols = [d[0] for d in conn.execute(
                    "SELECT * FROM scrip_master LIMIT 0").description]
                return dict(zip(cols, row))
        return None

    def search_symbol(self, query: str, exchange: str = "nse_cm") -> list:
        """Search scrip master by partial name or symbol."""
        with self._get_conn() as conn:
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

    # ─────────────────────────────────────────────────────────────────────────
    # Live quotes
    # ─────────────────────────────────────────────────────────────────────────

    def get_live_quote(self, symbol: str, exchange: str = "nse_cm") -> Optional[dict]:
        """Fetch live quote for a symbol. Returns parsed price data."""
        if not self.client:
            logger.error("No Kotak client available")
            return None

        instrument = self.get_instrument(symbol, exchange)
        if not instrument:
            logger.warning(f"Symbol not found in scrip master: {symbol}")
            return None

        token = instrument["token"]
        try:
            raw = self.client.get_quote(exchange, token)
            quote = self._parse_quote(raw, symbol, exchange)
            if quote:
                try:
                    self._store_tick(quote)
                except Exception as e:
                    logger.error(f"Failed to store tick for {symbol}: {e}")
                    # Return quote anyway, at least in memory
            return quote
        except Exception as e:
            logger.error(f"Quote fetch failed for {symbol}: {e}")
            return None

    def _parse_quote(self, raw, symbol: str, exchange: str) -> Optional[dict]:
        """Parse raw quote API response into clean dict with validation.

        Kotak returns a list: [{"ltp": "270.94", "ohlc": {...}, ...}]
        """
        try:
            # Unwrap list response (Kotak quote API returns a list)
            if isinstance(raw, list):
                if not raw:
                    logger.warning(f"Empty quote list for {symbol}")
                    return None
                data = raw[0]
            elif isinstance(raw, dict):
                data = raw.get("data", raw)
            else:
                logger.error(f"Invalid quote response type: {type(raw)}")
                return None

            if not data:
                logger.warning(f"Empty quote data for {symbol}")
                return None

            # Validate LTP (Kotak returns strings for prices)
            ltp = float(data.get("ltp") or data.get("last_price", 0))
            if ltp <= 0:
                logger.warning(f"Invalid LTP {ltp} for {symbol}")
                return None

            # OHLC is nested under "ohlc" key in Kotak response
            ohlc = data.get("ohlc", {})

            return {
                "symbol":     symbol,
                "exchange":   exchange,
                "ltp":        ltp,
                "open":       float(ohlc.get("open") or data.get("open", 0)),
                "high":       float(ohlc.get("high") or data.get("high", 0)),
                "low":        float(ohlc.get("low") or data.get("low", 0)),
                "close":      float(ohlc.get("close") or data.get("close") or data.get("prev_close", 0)),
                "volume":     int(data.get("last_volume") or data.get("volume", 0)),
                "bid":        float(data.get("total_buy") or data.get("bid", 0)),
                "ask":        float(data.get("total_sell") or data.get("ask", 0)),
                "timestamp":  datetime.now().isoformat(),
                "change_pct": float(data.get("per_change") or data.get("pct_change", 0)),
            }
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Quote parse error for {symbol}: {e} | Raw: {raw}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing quote for {symbol}: {e}")
            return None

    def _store_tick(self, quote: dict):
        """Store a tick in the database with retry on database lock."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._get_conn() as conn:
                    conn.execute(
                        """INSERT INTO tick_data (symbol, exchange, timestamp, ltp, volume, bid, ask)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (quote["symbol"], quote["exchange"], quote["timestamp"],
                         quote["ltp"], quote["volume"], quote["bid"], quote["ask"])
                    )
                    conn.commit()
                return
            except sqlite3.IntegrityError as e:
                logger.debug(f"Duplicate tick ignored: {e}")
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
                logger.error(f"Database error storing tick: {e}")
                raise

    # ─────────────────────────────────────────────────────────────────────────
    # OHLCV history
    # ─────────────────────────────────────────────────────────────────────────

    def store_candle(self, symbol: str, exchange: str, timestamp: str,
                     o: float, h: float, l: float, c: float, v: int):
        """Store a single OHLCV candle."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ohlcv
                   (symbol, exchange, timestamp, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, exchange, timestamp, o, h, l, c, v)
            )

    def store_quotes_as_candles(self, symbols: list, exchange: str = "nse_cm"):
        """
        Fetch live REST quote for each symbol and save as the current 5-min candle.
        Aligns timestamp to the current 5-minute bar (e.g. 09:25:00 for any time 09:25–09:29).
        Called every cycle when WebSocket ticks are unavailable.
        """
        from datetime import datetime as _dt
        now = _dt.now()
        # Floor to current 5-min bar
        bar_minute = (now.minute // 5) * 5
        bar_ts = now.replace(minute=bar_minute, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")

        saved = 0
        for symbol in symbols:
            try:
                time.sleep(0.3)   # 300ms between quotes — avoid Kotak 429 rate limit
                q = self.get_live_quote(symbol, exchange)
                if not q or q.get("ltp", 0) <= 0:
                    continue
                ltp = q["ltp"]
                # Use day OHLC from quote if available, otherwise use LTP for all
                o = q.get("open") or ltp
                h = q.get("high") or ltp
                l = q.get("low")  or ltp
                c = ltp
                v = q.get("volume") or 0
                # Only store if we have meaningful OHLC (open > 0)
                if o > 0:
                    self.store_candle(symbol, exchange, bar_ts, o, h, l, c, v)
                    saved += 1
            except Exception as e:
                logger.debug(f"Quote-to-candle failed for {symbol}: {e}")
        if saved:
            logger.debug(f"Saved {saved} REST quote candles for bar {bar_ts}")
        return saved

    def get_ohlcv(self, symbol: str, exchange: str = "nse_cm",
                  limit: int = 200) -> pd.DataFrame:
        """
        Get OHLCV history as a DataFrame.
        Returns columns: timestamp, open, high, low, close, volume
        """
        with self._get_conn() as conn:
            df = pd.read_sql_query(
                """SELECT timestamp, open, high, low, close, volume
                   FROM ohlcv WHERE symbol=? AND exchange=?
                   ORDER BY timestamp DESC LIMIT ?""",
                conn, params=(symbol, exchange, limit)
            )
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        return df.sort_values("timestamp").reset_index(drop=True)

    def build_candle_from_ticks(self, symbol: str, exchange: str,
                                 interval_minutes: int = 5):
        """Aggregate tick data into OHLCV candles. Only processes ticks newer than latest candle."""
        # Find the most recent candle already stored — only import ticks after it
        with self._get_conn() as conn:
            last_candle_ts = conn.execute(
                "SELECT MAX(timestamp) FROM ohlcv WHERE symbol=? AND exchange=?",
                (symbol, exchange)
            ).fetchone()[0]

        with self._get_conn() as conn:
            if last_candle_ts:
                df = pd.read_sql_query(
                    """SELECT timestamp, ltp, volume FROM tick_data
                       WHERE symbol=? AND exchange=? AND timestamp > ?
                       ORDER BY timestamp""",
                    conn, params=(symbol, exchange, last_candle_ts)
                )
            else:
                df = pd.read_sql_query(
                    """SELECT timestamp, ltp, volume FROM tick_data
                       WHERE symbol=? AND exchange=?
                       ORDER BY timestamp""",
                    conn, params=(symbol, exchange)
                )
        if df.empty:
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        rule = f"{interval_minutes}min"
        ohlcv = df["ltp"].resample(rule).ohlc()
        ohlcv["volume"] = df["volume"].resample(rule).sum()
        ohlcv = ohlcv.dropna()

        with self._get_conn() as conn:
            for ts, row in ohlcv.iterrows():
                conn.execute(
                    """INSERT OR REPLACE INTO ohlcv
                       (symbol, exchange, timestamp, open, high, low, close, volume)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, exchange, str(ts),
                     row["open"], row["high"], row["low"], row["close"], int(row["volume"]))
                )
        logger.info(f"Built {len(ohlcv)} candles for {symbol}")

    def bootstrap_ohlcv(self, symbols: list, period: str = "60d", interval: str = "5m") -> int:
        """
        Download historical OHLCV from Yahoo Finance (yfinance) for the given symbols.
        Called at startup so strategies have warm indicator data immediately.

        Symbol mapping: "RELIANCE-EQ" → "RELIANCE.NS", "NIFTYBEES-EQ" → "NIFTYBEES.NS"
        Returns the total number of candles inserted.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return 0

        # Check if we already have today's candles — skip download if fresh enough
        from datetime import date as _date
        today_str = str(_date.today())
        with self._get_conn() as conn:
            fresh_count = conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM ohlcv WHERE timestamp >= ?",
                (today_str,)
            ).fetchone()[0]
        if fresh_count >= len(symbols) * 0.8:   # 80% of symbols have today's data
            logger.info(f"Bootstrap skipped — {fresh_count}/{len(symbols)} symbols already have today's candles")
            with self._get_conn() as conn:
                return conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]

        total = 0
        for symbol in symbols:
            try:
                # Convert "RELIANCE-EQ" → "RELIANCE.NS"
                base = symbol.split("-")[0]
                yf_ticker = f"{base}.NS"

                df = yf.download(
                    yf_ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )

                if df.empty:
                    logger.warning(f"No yfinance data for {yf_ticker}")
                    continue

                # yfinance 1.4+ returns MultiIndex columns: (field, ticker)
                # Flatten to single level
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                # Make timezone-naive timestamps
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)

                # Keep only NSE market hours: 9:15 AM – 3:30 PM IST
                # yfinance includes pre/post-market candles with stale prices
                df = df[df.index.time >= pd.to_datetime("09:15").time()]
                df = df[df.index.time <= pd.to_datetime("15:30").time()]

                # Remove anomalous candles (>8% move — corporate action artefacts)
                close_col = "Close" if "Close" in df.columns else "close"
                if close_col in df.columns and len(df) > 1:
                    pct_chg = df[close_col].pct_change().abs()
                    bad = pct_chg > 0.08
                    if bad.sum() > 0:
                        logger.warning(f"Removing {bad.sum()} anomalous candles from {yf_ticker}")
                        df = df[~bad]

                # Derive exchange from scrip master (fall back to nse_cm)
                instrument = self.get_instrument(symbol, "nse_cm")
                exchange = instrument["exchange"] if instrument else "nse_cm"

                inserted = 0
                with self._get_conn() as conn:
                    for ts, row in df.iterrows():
                        try:
                            conn.execute(
                                """INSERT OR REPLACE INTO ohlcv
                                   (symbol, exchange, timestamp, open, high, low, close, volume)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                                (
                                    symbol, exchange, str(ts),
                                    float(row["Open"]), float(row["High"]),
                                    float(row["Low"]),  float(row["Close"]),
                                    int(row["Volume"]),
                                )
                            )
                            inserted += 1
                        except Exception:
                            pass
                    conn.commit()

                logger.info(f"Bootstrapped {inserted} candles for {symbol} from yfinance")
                total += inserted

            except Exception as e:
                logger.error(f"Bootstrap failed for {symbol}: {e}")

        logger.info(f"OHLCV bootstrap complete: {total} total candles for {len(symbols)} symbols")
        return total

    def bootstrap_daily_ohlcv(self, symbols: list, years: int = 1) -> int:
        """
        Download DAILY OHLCV from Yahoo Finance for up to `years` of history.
        Daily bars are not subject to yfinance's 60-day intraday limit.

        Stored in the same `ohlcv` table with timestamp = "YYYY-MM-DD" so
        backtest scripts can union intraday + daily data or query separately.
        Exchange stored as "nse_cm_daily" to distinguish from 5m intraday bars.

        Returns total rows inserted.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return 0

        period = f"{years}y"
        total  = 0

        for symbol in symbols:
            try:
                base      = symbol.split("-")[0]
                yf_ticker = f"{base}.NS"

                df = yf.download(
                    yf_ticker,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )

                if df.empty:
                    logger.warning(f"No daily data for {yf_ticker}")
                    continue

                # Flatten MultiIndex columns (yfinance ≥0.2.x)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                inserted = 0
                with self._get_conn() as conn:
                    for ts, row in df.iterrows():
                        date_str = str(ts.date()) if hasattr(ts, "date") else str(ts)[:10]
                        try:
                            conn.execute(
                                """INSERT OR REPLACE INTO ohlcv
                                   (symbol, exchange, timestamp, open, high, low, close, volume)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                                (
                                    symbol, "nse_cm_daily", date_str,
                                    float(row.get("Open",  row.get("open",  0))),
                                    float(row.get("High",  row.get("high",  0))),
                                    float(row.get("Low",   row.get("low",   0))),
                                    float(row.get("Close", row.get("close", 0))),
                                    int(row.get("Volume", row.get("volume", 0))),
                                ),
                            )
                            inserted += 1
                        except Exception:
                            pass
                    conn.commit()

                total += inserted
                logger.info(f"Daily OHLCV {symbol}: {inserted} days inserted ({period})")

            except Exception as e:
                logger.error(f"Daily bootstrap failed for {symbol}: {e}")

        logger.info(f"Daily OHLCV bootstrap complete: {total} rows for {len(symbols)} symbols")
        return total

    # ─────────────────────────────────────────────────────────────────────────
    # Trade history persistence
    # ─────────────────────────────────────────────────────────────────────────

    def record_trade(self, trade: dict):
        """Persist a completed trade to the trades table with validation."""
        try:
            # Validate required fields
            required_keys = ["symbol", "action", "entry_price", "exit_price", "quantity", "pnl"]
            missing = [k for k in required_keys if k not in trade or trade[k] is None]
            if missing:
                raise ValueError(f"Missing required trade fields: {missing}")

            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO trades
                       (symbol, exchange, action, strategy, entry_price, exit_price,
                        quantity, pnl, exit_reason, entry_time, exit_time, order_no)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(trade.get("symbol", "")),
                        str(trade.get("exchange", "nse_cm")),
                        str(trade.get("action", "")),
                        str(trade.get("strategy", "")),
                        float(trade.get("entry_price", 0)),
                        float(trade.get("exit_price", 0)),
                        int(trade.get("quantity", 0)),
                        float(trade.get("pnl", 0)),
                        str(trade.get("exit_reason", "")),
                        str(trade.get("entry_time", "")),
                        str(trade.get("exit_time", str(datetime.now()))),
                        str(trade.get("order_no", "")),
                    ),
                )
                conn.commit()
        except ValueError as e:
            logger.error(f"Trade validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
            raise

    def get_trades(self, limit: int = 200, symbol: str = None) -> pd.DataFrame:
        """Fetch completed trade history as a DataFrame."""
        with self._get_conn() as conn:
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

    # ─────────────────────────────────────────────────────────────────────────
    # Bot status (for dashboard)
    # ─────────────────────────────────────────────────────────────────────────

    def update_bot_status(self, portfolio: dict, paper_trading: bool = True,
                          status: str = "running", regime_data: dict = None):
        """Write current bot state to the bot_status table (single row).

        Args:
            portfolio:    Portfolio summary from RiskManager.
            paper_trading: Whether paper trading is active.
            status:       Bot lifecycle status (running / stopped / closed).
            regime_data:  Optional dict from AIDecisionEngine.detect_market_regime():
                          {"regime": str, "confidence": float,
                           "suggestion": str, "risk_level": str}
        """
        import json
        regime_data = regime_data or {}
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE bot_status SET
                   daily_pnl=?, daily_trades=?, open_positions=?,
                   capital_at_risk=?, paper_trading=?, last_cycle=?, status=?,
                   market_regime=?, ai_regime_confidence=?,
                   ai_regime_suggestion=?, ai_regime_risk=?
                   WHERE id=1""",
                (
                    portfolio.get("daily_pnl", 0),
                    portfolio.get("daily_trades", 0),
                    json.dumps({
                        k: {
                            "action": v["action"],
                            "entry_price": v["entry_price"],
                            "quantity": v["quantity"],
                            "stop_loss": v["stop_loss"],
                            "target": v["target"],
                            "strategy": v.get("strategy", ""),
                        }
                        for k, v in portfolio.get("open_positions", {}).items()
                    }),
                    portfolio.get("capital_at_risk", 0),
                    1 if paper_trading else 0,
                    str(datetime.now()),
                    status,
                    regime_data.get("regime", "unknown"),
                    regime_data.get("confidence", 0.0),
                    regime_data.get("suggestion", ""),
                    regime_data.get("risk_level", "medium"),
                ),
            )

    def get_bot_status(self) -> dict:
        """Read current bot state from the database."""
        import json
        with self._get_conn() as conn:
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

    def log_event(self, event_type: str, message: str, symbol: str = "", metadata: dict = None):
        """Log an event (signal, veto, execution, etc.) for dashboard feed."""
        import json
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO events (timestamp, event_type, message, symbol, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    str(datetime.now()),
                    event_type,
                    message,
                    symbol,
                    json.dumps(metadata or {}),
                ),
            )

    def get_events(self, limit: int = 50, event_type: str = None) -> pd.DataFrame:
        """Fetch recent events for the signal feed."""
        with self._get_conn() as conn:
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

    # ─────────────────────────────────────────────────────────────────────────
    # AI decision logging
    # ─────────────────────────────────────────────────────────────────────────

    def record_ai_decision(self, decision: dict) -> int:
        """
        Persist one AI evaluation of a signal.  Returns the new row id so
        the caller can later call update_ai_outcome() when the trade closes.

        Expected keys in `decision`:
            symbol, strategy, action, signal_confidence, final_confidence,
            approved (bool), veto_reason, ai_reasoning, tools_used (list),
            signal_price, stop_loss, target, market_regime
        """
        import json
        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO ai_decisions
                   (decided_at, symbol, strategy, action,
                    signal_conf, signal_confidence, final_confidence, approved,
                    veto_reason, ai_reasoning, reasoning, tools_used,
                    signal_price, stop_loss, target, market_regime)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(datetime.now()),
                    str(decision.get("symbol", "")),
                    str(decision.get("strategy", "")),
                    str(decision.get("action", "BUY")),
                    float(decision.get("signal_confidence", 0)),  # signal_conf (original col)
                    float(decision.get("signal_confidence", 0)),  # signal_confidence (new col)
                    float(decision.get("final_confidence", 0)),
                    1 if decision.get("approved") else 0,
                    str(decision.get("veto_reason") or ""),
                    str(decision.get("ai_reasoning") or ""),      # ai_reasoning (new col)
                    str(decision.get("ai_reasoning") or ""),      # reasoning (original col)
                    json.dumps(decision.get("tools_used") or []),
                    float(decision.get("signal_price") or 0),
                    float(decision.get("stop_loss") or 0),
                    float(decision.get("target") or 0),
                    str(decision.get("market_regime") or "unknown"),
                ),
            )
            conn.commit()
            return cur.lastrowid

    def update_ai_outcome(self, decision_id: int, outcome: str,
                          pnl: float, good_decision: bool):
        """
        Called at trade exit to record whether the AI decision was correct.
        outcome: "win" | "loss" | "breakeven"
        good_decision: True if approved+win OR vetoed+would-have-lost
        """
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE ai_decisions
                   SET outcome=?, outcome_pnl=?, outcome_correct=?, good_decision=?,
                       exit_reason=?, closed_at=?
                   WHERE id=?""",
                (outcome, float(pnl), 1 if good_decision else 0,
                 1 if good_decision else 0, outcome, str(datetime.now()),
                 decision_id),
            )
            conn.commit()

    def get_ai_decisions(self, limit: int = 200, symbol: str = None,
                         approved_only: bool = False) -> pd.DataFrame:
        """Fetch AI decision log for analysis / dashboard."""
        with self._get_conn() as conn:
            filters = []
            params: list = []
            if symbol:
                filters.append("symbol=?"); params.append(symbol)
            if approved_only:
                filters.append("approved=1")
            where = ("WHERE " + " AND ".join(filters)) if filters else ""
            df = pd.read_sql_query(
                f"""SELECT * FROM ai_decisions {where}
                    ORDER BY decided_at DESC LIMIT ?""",
                conn, params=params + [limit],
            )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Daily performance journal
    # ─────────────────────────────────────────────────────────────────────────

    def write_daily_journal(self, date_str: str, stats: dict):
        """
        Upsert one row into daily_journal.
        stats keys: market_regime, signals_seen, signals_approved, signals_vetoed,
                    trades_opened, trades_closed, day_pnl, cumulative_pnl,
                    win_count, loss_count, open_positions, notes
        """
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO daily_journal
                   (date, market_regime, signals_seen, signals_approved,
                    signals_vetoed, trades_opened, trades_closed,
                    day_pnl, cumulative_pnl, win_count, loss_count,
                    open_positions, notes, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(date) DO UPDATE SET
                     market_regime=excluded.market_regime,
                     signals_seen=excluded.signals_seen,
                     signals_approved=excluded.signals_approved,
                     signals_vetoed=excluded.signals_vetoed,
                     trades_opened=excluded.trades_opened,
                     trades_closed=excluded.trades_closed,
                     day_pnl=excluded.day_pnl,
                     cumulative_pnl=excluded.cumulative_pnl,
                     win_count=excluded.win_count,
                     loss_count=excluded.loss_count,
                     open_positions=excluded.open_positions,
                     notes=excluded.notes""",
                (
                    date_str,
                    str(stats.get("market_regime", "unknown")),
                    int(stats.get("signals_seen", 0)),
                    int(stats.get("signals_approved", 0)),
                    int(stats.get("signals_vetoed", 0)),
                    int(stats.get("trades_opened", 0)),
                    int(stats.get("trades_closed", 0)),
                    float(stats.get("day_pnl", 0)),
                    float(stats.get("cumulative_pnl", 0)),
                    int(stats.get("win_count", 0)),
                    int(stats.get("loss_count", 0)),
                    int(stats.get("open_positions", 0)),
                    str(stats.get("notes", "")),
                    str(datetime.now()),
                ),
            )
            conn.commit()

    def get_daily_journal(self, days: int = 30) -> pd.DataFrame:
        """Fetch recent daily journal entries."""
        with self._get_conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM daily_journal ORDER BY date DESC LIMIT ?",
                conn, params=(days,),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # Market regime detection (local — no API call)
    # ─────────────────────────────────────────────────────────────────────────

    def detect_regime(self, symbol: str = "NIFTYBEES-EQ",
                      exchange: str = "nse_cm") -> str:
        """
        Detect the current market regime from stored OHLCV — no API call needed.
        Uses EMA 9/21 trend direction and ATR-based volatility.

        Returns: "trending_up" | "trending_down" | "high_volatility" | "ranging" | "unknown"
        """
        df = self.get_ohlcv(symbol, exchange, limit=50)
        if len(df) < 25:
            return "unknown"

        import numpy as _np
        close = df["close"]
        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()

        high_low   = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close  = (df["low"]  - df["close"].shift()).abs()
        atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

        current = float(close.iloc[-1])
        fast    = float(ema9.iloc[-1])
        slow    = float(ema21.iloc[-1])
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
        atr_pct = (atr_val / current * 100) if current > 0 else 0.0

        if atr_pct > 2.5:
            regime = "high_volatility"
        elif fast > slow * 1.001:    # EMA9 more than 0.1% above EMA21
            regime = "trending_up"
        elif fast < slow * 0.999:    # EMA9 more than 0.1% below EMA21
            regime = "trending_down"
        else:
            regime = "ranging"

        logger.debug(
            f"Market regime [{symbol}]: {regime}"
            f" (EMA9={fast:.1f} EMA21={slow:.1f} ATR%={atr_pct:.1f}%)"
        )
        return regime

    def get_watchlist_quotes(self, symbols: list, exchange: str = "nse_cm") -> dict:
        """Fetch live quotes for a list of symbols."""
        quotes = {}
        for symbol in symbols:
            q = self.get_live_quote(symbol, exchange)
            if q:
                quotes[symbol] = q
        return quotes

    def is_scrip_master_fresh(self) -> bool:
        """Check if scrip master was downloaded today."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT MAX(last_updated) FROM scrip_master"
            ).fetchone()
            if row and row[0]:
                return row[0] == str(date.today())
        return False

    def cleanup_old_data(self, tick_days: int = 2, events_days: int = 30):
        """
        Delete old tick_data and events rows to prevent unbounded DB growth.
        - tick_data:  keep last `tick_days` days (default 2) — only needed for candle building
        - events:     keep last `events_days` days (default 30) — dashboard shows last 50 rows
        Called daily during job_daily_setup.
        """
        with self._get_conn() as conn:
            tick_deleted = conn.execute(
                "DELETE FROM tick_data WHERE timestamp < datetime('now', ? || ' days')",
                (f"-{tick_days}",)
            ).rowcount
            events_deleted = conn.execute(
                "DELETE FROM events WHERE timestamp < datetime('now', ? || ' days')",
                (f"-{events_days}",)
            ).rowcount
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")  # TRUNCATE needs exclusive lock; PASSIVE is safe with concurrent readers
        logger.info(
            f"DB cleanup: removed {tick_deleted} old ticks, {events_deleted} old events"
        )

    def get_db_stats(self) -> dict:
        """Summary of what's in the database."""
        with self._get_conn() as conn:
            scrip_count = conn.execute("SELECT COUNT(*) FROM scrip_master").fetchone()[0]
            ohlcv_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
            tick_count  = conn.execute("SELECT COUNT(*) FROM tick_data").fetchone()[0]
        return {
            "scrip_instruments": scrip_count,
            "ohlcv_candles": ohlcv_count,
            "ticks_stored": tick_count,
            "scrip_master_fresh": self.is_scrip_master_fresh(),
        }
