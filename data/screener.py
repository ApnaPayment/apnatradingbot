"""
Phase 12 — Dynamic Watchlist Screener

Scans a broad NIFTY-200 universe each morning and ranks symbols by a
composite momentum/breakout score. The top N candidates are promoted into
the active scan list for the trading day.

Scoring factors (equal-weighted, each 0–1 after normalisation):
  1. Price momentum  — 20-day return (higher = better for BUY regime)
  2. Volume surge    — today's volume vs 20-day avg (> 1.5× = strong interest)
  3. RSI position    — sweet spot 50–70 for momentum entries
  4. Trend strength  — EMA9 vs EMA21 gap (positive = uptrend)
  5. ATR breakout    — price distance from 20-day high relative to ATR

Data source: yfinance (same as bootstrap_ohlcv) — no auth required.
Results saved to DB table `screener_runs` for dashboard display.

Typical runtime: ~20 seconds for 200 symbols (yfinance batch download).
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "market_data.db"

# ─────────────────────────────────────────────────────────────────────────────
# NIFTY-200 liquid universe  (NSE symbols with -EQ suffix)
# Update annually; covers most actively traded large/mid-caps
# ─────────────────────────────────────────────────────────────────────────────
NIFTY200_UNIVERSE = [
    # NIFTY 50 core
    "RELIANCE-EQ","TCS-EQ","HDFCBANK-EQ","INFY-EQ","ICICIBANK-EQ",
    "HINDUNILVR-EQ","SBIN-EQ","BHARTIARTL-EQ","ITC-EQ","KOTAKBANK-EQ",
    "LT-EQ","AXISBANK-EQ","MARUTI-EQ","SUNPHARMA-EQ","TITAN-EQ",
    "BAJFINANCE-EQ","NESTLEIND-EQ","WIPRO-EQ","ULTRACEMCO-EQ","ASIANPAINT-EQ",
    "HCLTECH-EQ","POWERGRID-EQ","NTPC-EQ","INDUSINDBK-EQ","ONGC-EQ",
    "JSWSTEEL-EQ","TATAMOTORS-EQ","ADANIENT-EQ","ADANIPORTS-EQ","COALINDIA-EQ",
    "BAJAJFINSV-EQ","DIVISLAB-EQ","DRREDDY-EQ","EICHERMOT-EQ","GRASIM-EQ",
    "HDFCLIFE-EQ","HEROMOTOCO-EQ","HINDALCO-EQ","M&M-EQ","SBILIFE-EQ",
    "SHREECEM-EQ","TATASTEEL-EQ","TECHM-EQ","TATACONSUM-EQ","UPL-EQ",
    "APOLLOHOSP-EQ","BPCL-EQ","BRITANNIA-EQ","CIPLA-EQ","DMART-EQ",
    # NIFTY NEXT 50 / midcap additions
    "AMBUJACEM-EQ","AUROPHARMA-EQ","BANDHANBNK-EQ","BERGEPAINT-EQ","BIOCON-EQ",
    "CHOLAFIN-EQ","COLPAL-EQ","CONCOR-EQ","DABUR-EQ","DLF-EQ",
    "ESCORTS-EQ","FEDERALBNK-EQ","GAIL-EQ","GODREJCP-EQ","GODREJPROP-EQ",
    "HAVELLS-EQ","IDFCFIRSTB-EQ","INDHOTEL-EQ","INDUSTOWER-EQ","IRCTC-EQ",
    "JINDALSTEL-EQ","JUBLFOOD-EQ","LTI-EQ","LUPIN-EQ","MARICO-EQ",
    "MCDOWELL-N-EQ","MFSL-EQ","MINDTREE-EQ","MOTHERSON-EQ","MPHASIS-EQ",
    "MRF-EQ","MUTHOOTFIN-EQ","NAUKRI-EQ","NMDC-EQ","OBEROIRLTY-EQ",
    "OFSS-EQ","PAGEIND-EQ","PEL-EQ","PERSISTENT-EQ","PETRONET-EQ",
    "PIDILITIND-EQ","PIIND-EQ","PNB-EQ","POLYCAB-EQ","PGHH-EQ",
    "RECLTD-EQ","SAIL-EQ","SIEMENS-EQ","SRF-EQ","SUNPHARMA-EQ",
    "SUNTV-EQ","SUPREMEIND-EQ","TATACOMM-EQ","TATAELXSI-EQ","TATAPOWE-EQ",
    "TORNTPHARM-EQ","TRENT-EQ","TRIDENT-EQ","VGUARD-EQ","VOLTAS-EQ",
    "WHIRLPOOL-EQ","WIPRO-EQ","ZEEL-EQ","ZYDUSLIFE-EQ",
    # ETFs (always keep these for regime hedging)
    "NIFTYBEES-EQ","ITBEES-EQ","BANKBEES-EQ","JUNIORBEES-EQ",
]
# Deduplicate while preserving order
_seen: set = set()
NIFTY200_UNIVERSE = [
    s for s in NIFTY200_UNIVERSE if not (s in _seen or _seen.add(s))
]


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_screener_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS screener_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date    TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                score       REAL,
                rank        INTEGER,
                momentum    REAL,
                volume_surge REAL,
                rsi         REAL,
                trend_strength REAL,
                promoted    INTEGER DEFAULT 0,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_screener_date "
            "ON screener_runs(run_date DESC)"
        )


def _save_screener_results(results: list[dict], run_date: str):
    _ensure_screener_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM screener_runs WHERE run_date=?", (run_date,))
        conn.executemany(
            """INSERT INTO screener_runs
               (run_date, symbol, score, rank, momentum, volume_surge, rsi,
                trend_strength, promoted)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [(
                run_date, r["symbol"], r["score"], r["rank"],
                r.get("momentum"), r.get("volume_surge"), r.get("rsi"),
                r.get("trend_strength"), 1 if r.get("promoted") else 0,
            ) for r in results],
        )


def get_latest_screener_results(limit: int = 30) -> list[dict]:
    """Fetch most recent screener run for dashboard display."""
    _ensure_screener_table()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                """SELECT symbol, score, rank, momentum, volume_surge, rsi,
                          trend_strength, promoted, run_date
                   FROM screener_runs
                   WHERE run_date = (SELECT MAX(run_date) FROM screener_runs)
                   ORDER BY rank ASC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {
                "symbol": r[0], "score": r[1], "rank": r[2],
                "momentum": r[3], "volume_surge": r[4], "rsi": r[5],
                "trend_strength": r[6], "promoted": bool(r[7]), "run_date": r[8],
            }
            for r in rows
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScreenerResult:
    symbol:         str
    score:          float = 0.0
    rank:           int   = 0
    momentum:       Optional[float] = None   # 20-day return
    volume_surge:   Optional[float] = None   # today vol / 20d avg vol
    rsi:            Optional[float] = None
    trend_strength: Optional[float] = None   # (EMA9 - EMA21) / EMA21
    promoted:       bool  = False


def _score_symbol(df: pd.DataFrame) -> dict:
    """
    Compute the 5 factor scores for one symbol.
    df must have columns: open, high, low, close, volume (daily bars).
    Returns factor dict with values in [0, 1] each.
    """
    if df is None or len(df) < 22:
        return {}

    close  = df["close"]
    volume = df["volume"]

    # 1. Price momentum — 20-day return, clipped to [-30%, +50%]
    ret20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100
    momentum_score = np.clip((ret20 + 30) / 80, 0, 1)

    # 2. Volume surge — last day vs 20-day avg (capped at 3×)
    avg_vol = volume.iloc[-21:-1].mean()
    surge   = volume.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
    volume_score = np.clip((surge - 0.5) / 2.5, 0, 1)   # 0.5× → 0, 3.0× → 1

    # 3. RSI — sweet spot 50–70 → score 1.0; outside → taper off
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 100
    rsi    = 100 - (100 / (1 + rs))
    if 50 <= rsi <= 70:
        rsi_score = 1.0
    elif rsi < 50:
        rsi_score = max(0.0, rsi / 50)
    else:
        rsi_score = max(0.0, (100 - rsi) / 30)

    # 4. Trend strength — EMA9 vs EMA21 gap as % of price
    ema9  = close.ewm(span=9,  adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    gap_pct = (ema9 - ema21) / ema21 * 100
    trend_score = np.clip((gap_pct + 3) / 8, 0, 1)  # -3% → 0, +5% → 1

    # 5. ATR breakout — price vs 20-day high, normalised by ATR
    high20  = df["high"].iloc[-21:].max()
    atr     = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    dist    = (close.iloc[-1] - high20) / atr if atr > 0 else 0
    breakout_score = np.clip((dist + 1) / 2, 0, 1)   # -1 ATR below = 0, at high = 0.5, above = 1

    # Composite score (equal weights)
    composite = (momentum_score + volume_score + rsi_score + trend_score + breakout_score) / 5.0

    return {
        "momentum":       round(float(ret20), 2),
        "volume_surge":   round(float(surge), 2),
        "rsi":            round(float(rsi), 1),
        "trend_strength": round(float(gap_pct), 3),
        "score":          round(float(composite), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Screener class
# ─────────────────────────────────────────────────────────────────────────────

class Screener:
    """
    Scans the NIFTY-200 universe and returns the top N symbols by composite score.

    Results are saved to DB; promoted symbols are returned as a list for the
    bot to use as its active watchlist for the day.
    """

    def __init__(self, universe: list[str] = None):
        self.universe = universe or NIFTY200_UNIVERSE

    def run(self, top_n: int = 15, min_score: float = 0.45,
            always_include: list[str] = None) -> list[str]:
        """
        Download daily OHLCV, score all symbols, return top N.

        Args:
            top_n:          Maximum number of symbols to promote.
            min_score:      Minimum composite score to be promoted.
            always_include: Symbols to pin regardless of score (e.g., ETFs).

        Returns:
            List of symbol strings (NSE -EQ format), pinned first then ranked.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed — screener unavailable")
            return always_include or []

        run_date = date.today().isoformat()
        pinned   = list(always_include or [])
        logger.info(f"Screener: scanning {len(self.universe)} symbols...")

        # Batch download — yfinance handles up to ~500 tickers at once
        tickers = [s.replace("-EQ", ".NS") for s in self.universe]
        try:
            raw = yf.download(
                tickers, period="30d", interval="1d",
                group_by="ticker", auto_adjust=True, progress=False,
                threads=True,
            )
        except Exception as e:
            logger.error(f"Screener yfinance download failed: {e}")
            return pinned

        results: list[ScreenerResult] = []
        for sym, ticker in zip(self.universe, tickers):
            try:
                # Extract single-symbol slice from multi-ticker download
                if len(self.universe) == 1:
                    df = raw.copy()
                elif ticker in raw.columns.get_level_values(0):
                    df = raw[ticker].dropna()
                else:
                    continue
                df.columns = [c.lower() for c in df.columns]
                factors = _score_symbol(df)
                if not factors:
                    continue
                results.append(ScreenerResult(
                    symbol=sym, score=factors["score"],
                    momentum=factors["momentum"],
                    volume_surge=factors["volume_surge"],
                    rsi=factors["rsi"],
                    trend_strength=factors["trend_strength"],
                ))
            except Exception:
                continue

        if not results:
            logger.warning("Screener: no results — returning pinned only")
            return pinned

        # Rank by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        # Promote top N above min_score, excluding already-pinned
        promoted_syms = []
        for r in results:
            if len(promoted_syms) >= top_n:
                break
            if r.score < min_score:
                break
            if r.symbol not in pinned:
                r.promoted = True
                promoted_syms.append(r.symbol)

        # Save all results to DB
        db_rows = [
            {
                "symbol": r.symbol, "score": r.score, "rank": r.rank,
                "momentum": r.momentum, "volume_surge": r.volume_surge,
                "rsi": r.rsi, "trend_strength": r.trend_strength,
                "promoted": r.promoted,
            }
            for r in results
        ]
        try:
            _save_screener_results(db_rows, run_date)
        except Exception as e:
            logger.warning(f"Screener DB save failed (non-critical): {e}")

        active = pinned + [s for s in promoted_syms if s not in pinned]

        logger.info(
            f"Screener: {len(results)} scored, {len(promoted_syms)} promoted. "
            f"Active watchlist ({len(active)}): {active}"
        )
        return active
