"""
Liquidity Engine — Data Fetcher
Provides OHLCV and live quote data for non-NSE segments via yfinance.
Phase 1: BSE equity (.BO suffix)
Phase 2: MCX commodities (commodity ETF proxies + INR conversion)
Phase 3: NSE Currency Derivatives (Forex pairs)

No Kotak authentication required — this is the key design constraint that
prevents the liquidity engine from invalidating the NSE bot's Kotak session.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ─── BSE equity: yfinance tickers (.BO suffix) ──────────────────────────────
BSE_WATCHLIST = [
    "RELIANCE.BO", "TCS.BO", "INFY.BO", "HDFCBANK.BO", "ICICIBANK.BO",
    "WIPRO.BO", "LT.BO", "AXISBANK.BO", "SBIN.BO", "BHARTIARTL.BO",
]

# ─── MCX: yfinance commodity proxy tickers + INR conversion ─────────────────
# Prices are in USD; multiply by USDINR=X for INR value
MCX_TICKERS = {
    "GOLD":       "GC=F",       # Gold futures (USD/troy oz)
    "SILVER":     "SI=F",       # Silver futures (USD/troy oz)
    "CRUDEOIL":   "CL=F",       # WTI Crude (USD/barrel)
    "NATURALGAS": "NG=F",       # Natural Gas (USD/MMBtu)
    "COPPER":     "HG=F",       # Copper (USD/lb)
}
USDINR_TICKER = "USDINR=X"

# ─── CDE: yfinance Forex pairs ───────────────────────────────────────────────
CDE_TICKERS = {
    "USDINR":  "USDINR=X",
    "EURINR":  "EURINR=X",
    "GBPINR":  "GBPINR=X",
    "JPYINR":  "JPYINR=X",
}

# Cache: symbol → (DataFrame, fetched_at)
_ohlcv_cache: dict = {}
_CACHE_TTL_MINUTES = 4  # refresh before 5-min cycle fires


def _flatten_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Flatten yfinance MultiIndex columns and normalise to lowercase OHLCV
    format expected by all strategies (open, high, low, close, volume).
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # Rename to standard names if needed
    rename_map = {"adj close": "close"}
    df = df.rename(columns=rename_map)

    # Drop timezone from index for compatibility with DataManager
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Keep only required columns
    required = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in required if c in df.columns]].copy()

    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df


class LiquidityDataFetcher:
    """
    All market data for the Liquidity Engine.
    Uses yfinance exclusively — no Kotak auth needed.
    """

    # ─── BSE Equity ─────────────────────────────────────────────────────────

    def get_ohlcv_bse(self, ticker_bo: str, period: str = "60d",
                      interval: str = "5m") -> Optional[pd.DataFrame]:
        """
        Download BSE 5-minute OHLCV for a .BO ticker.
        Returns DataFrame with lowercase columns: open high low close volume.
        Returns None on failure.

        Example: get_ohlcv_bse("RELIANCE.BO")
        """
        cache_key = f"{ticker_bo}_{interval}"
        cached = _ohlcv_cache.get(cache_key)
        if cached:
            df_cached, fetched_at = cached
            age = (datetime.now() - fetched_at).total_seconds() / 60
            if age < _CACHE_TTL_MINUTES:
                return df_cached.copy()

        try:
            raw = yf.download(ticker_bo, period=period, interval=interval,
                              progress=False, auto_adjust=True)
            if raw.empty:
                logger.warning(f"LiqDataFetcher: empty data for {ticker_bo}")
                return None

            df = _flatten_yf(raw, ticker_bo)
            if len(df) < 30:
                logger.warning(f"LiqDataFetcher: only {len(df)} rows for {ticker_bo}")
                return None

            _ohlcv_cache[cache_key] = (df, datetime.now())
            logger.debug(f"LiqDataFetcher: {ticker_bo} → {len(df)} candles")
            return df.copy()

        except Exception as e:
            logger.error(f"LiqDataFetcher: BSE OHLCV failed for {ticker_bo}: {e}")
            return None

    def get_live_quote_bse(self, ticker_bo: str) -> Optional[dict]:
        """
        Return approximate live quote for a BSE stock.
        Uses yfinance fast_info (near-real-time, few minutes delayed).

        Returns: {ltp, volume, timestamp} or None.
        """
        try:
            tkr = yf.Ticker(ticker_bo)
            info = tkr.fast_info
            ltp = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
            vol = getattr(info, "three_month_average_volume", 0)
            if not ltp:
                return None
            return {
                "ltp":        round(float(ltp), 2),
                "last_price": round(float(ltp), 2),
                "volume":     int(vol or 0),
                "timestamp":  datetime.now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"LiqDataFetcher: live quote failed for {ticker_bo}: {e}")
            return None

    def bootstrap_bse(self, tickers: list) -> dict:
        """
        Download 60-day 5m OHLCV for all BSE tickers.
        Returns {ticker: candle_count} summary.
        """
        results = {}
        for ticker in tickers:
            df = self.get_ohlcv_bse(ticker)
            results[ticker] = len(df) if df is not None else 0
            if df is not None:
                logger.info(f"LiqBootstrap: {ticker} → {len(df)} candles")
        return results

    def bootstrap_mcx(self, commodities: list) -> dict:
        """
        Download 60-day 1h OHLCV for all MCX commodities (COMEX proxies in INR).
        Returns {commodity: candle_count} summary.
        """
        results = {}
        for commodity in commodities:
            df = self.get_ohlcv_mcx(commodity)
            results[commodity] = len(df) if df is not None else 0
            if df is not None:
                logger.info(f"LiqBootstrap MCX: {commodity} → {len(df)} candles")
        return results

    # ─── MCX Commodities [Phase 2] ───────────────────────────────────────────

    def get_ohlcv_mcx(self, commodity: str, period: str = "60d",
                      interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        MCX commodity OHLCV in INR.
        Uses USD commodity futures price × USDINR conversion.
        commodity: "GOLD", "SILVER", "CRUDEOIL", "NATURALGAS", "COPPER"
        """
        yf_ticker = MCX_TICKERS.get(commodity.upper())
        if not yf_ticker:
            logger.warning(f"LiqDataFetcher: unknown MCX commodity {commodity}")
            return None

        # ── Cache check (same TTL as BSE) ────────────────────────────────────
        cache_key = f"MCX_{commodity.upper()}_{interval}"
        cached = _ohlcv_cache.get(cache_key)
        if cached:
            df_cached, fetched_at = cached
            age_min = (datetime.now() - fetched_at).total_seconds() / 60
            if age_min < _CACHE_TTL_MINUTES:
                return df_cached.copy()

        try:
            # Get commodity price (USD) and USDINR
            raw_com = yf.download(yf_ticker, period=period, interval=interval,
                                  progress=False, auto_adjust=True)
            raw_fx = yf.download(USDINR_TICKER, period=period, interval="1d",
                                 progress=False, auto_adjust=True)
            if raw_com.empty or raw_fx.empty:
                return None

            df_com = _flatten_yf(raw_com, yf_ticker)
            df_fx = _flatten_yf(raw_fx, USDINR_TICKER)

            # Use latest FX rate for all bars (intraday FX drift < 0.5% on 1h bars — acceptable)
            df_com["date"] = df_com.index.date
            latest_fx = float(df_fx["close"].iloc[-1])
            df_com["fx_rate"] = latest_fx

            # Convert USD prices to INR
            for col in ["open", "high", "low", "close"]:
                if col in df_com.columns:
                    df_com[col] = df_com[col] * df_com["fx_rate"]
            df_com = df_com.drop(columns=["date", "fx_rate"])

            # Store in cache
            _ohlcv_cache[cache_key] = (df_com, datetime.now())
            logger.debug(f"LiqDataFetcher: {commodity} → {len(df_com)} candles (INR @ {latest_fx:.2f})")
            return df_com.copy()

        except Exception as e:
            logger.error(f"LiqDataFetcher: MCX OHLCV failed for {commodity}: {e}")
            return None

    # ─── Currency Derivatives [Phase 2] ─────────────────────────────────────

    def get_ohlcv_cde(self, pair: str, period: str = "60d",
                      interval: str = "5m") -> Optional[pd.DataFrame]:
        """
        Currency pair OHLCV (already in INR for INR pairs).
        pair: "USDINR", "EURINR", "GBPINR", "JPYINR"
        """
        yf_ticker = CDE_TICKERS.get(pair.upper())
        if not yf_ticker:
            return None
        try:
            raw = yf.download(yf_ticker, period=period, interval=interval,
                              progress=False, auto_adjust=True)
            if raw.empty:
                return None
            return _flatten_yf(raw, yf_ticker)
        except Exception as e:
            logger.error(f"LiqDataFetcher: CDE OHLCV failed for {pair}: {e}")
            return None

    def get_live_quote_mcx(self, commodity: str) -> Optional[dict]:
        """
        Return approximate live INR quote for an MCX commodity via yfinance fast_info.
        Price is USD commodity price × latest USDINR rate.
        Returns {ltp, last_price, timestamp} or None.
        """
        yf_ticker = MCX_TICKERS.get(commodity.upper())
        if not yf_ticker:
            logger.warning(f"LiqDataFetcher: unknown MCX commodity {commodity}")
            return None
        try:
            # Get commodity price in USD
            info = yf.Ticker(yf_ticker).fast_info
            ltp_usd = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
            if not ltp_usd:
                return None

            # Get USDINR conversion rate
            fx_info = yf.Ticker(USDINR_TICKER).fast_info
            fx_rate = getattr(fx_info, "last_price", None)
            if not fx_rate:
                logger.warning(f"LiqDataFetcher: USDINR rate unavailable for {commodity} live quote, using fallback 84.0")
                fx_rate = 84.0

            ltp_inr = round(float(ltp_usd) * float(fx_rate), 2)
            return {
                "ltp":        ltp_inr,
                "last_price": ltp_inr,
                "ltp_usd":    round(float(ltp_usd), 4),
                "fx_rate":    round(float(fx_rate), 4),
                "timestamp":  datetime.now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"LiqDataFetcher: MCX live quote failed for {commodity}: {e}")
            return None

    def get_live_quote_mcx_kotak(self, commodity: str,
                                   session_token: str, session_sid: str,
                                   base_url: str) -> Optional[dict]:
        """
        Real MCX live quote via Kotak Neo API using the shared session token.
        Falls back to yfinance (get_live_quote_mcx) if this fails.

        commodity: "GOLD", "SILVER", "CRUDEOIL", "NATURALGAS", "COPPER"
        Returns {ltp, last_price, timestamp} in INR or None.
        """
        import requests

        # MCX commodity → Kotak instrument token (MCX exchange segment: mcx_fo)
        # These are the standard MCX continuous front-month tokens (may need update each rollover)
        MCX_TOKENS = {
            "GOLD":       "238025",   # GOLD near-month
            "SILVER":     "234632",   # SILVER near-month
            "CRUDEOIL":   "237627",   # CRUDEOIL near-month
            "NATURALGAS": "237882",   # NATURALGAS near-month
            "COPPER":     "238337",   # COPPER near-month
        }
        token = MCX_TOKENS.get(commodity.upper())
        if not token:
            logger.warning(f"LiqDataFetcher: no Kotak token mapped for MCX {commodity}")
            return None

        try:
            url = f"{base_url}/1.0/market/quote/ltp"
            headers = {
                "Sid":          session_sid,
                "Auth":         session_token,
                "Content-Type": "application/json",
            }
            payload = {
                "instrument_tokens": [{"exchange": "mcx", "token": token}]
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            # Kotak returns list: [{exchange, token, ltp, ...}]
            items = data if isinstance(data, list) else data.get("data", [])
            if not items:
                return None
            ltp = float(items[0].get("ltp", 0))
            if ltp <= 0:
                return None

            return {
                "ltp":        ltp,
                "last_price": ltp,
                "source":     "kotak_mcx",
                "timestamp":  datetime.now().isoformat(),
            }

        except Exception as e:
            logger.debug(f"LiqDataFetcher: Kotak MCX quote failed for {commodity}: {e} — falling back to yfinance")
            return None   # caller falls back to get_live_quote_mcx()

    def get_live_quote_cde(self, pair: str) -> Optional[dict]:
        """Live FX quote via yfinance fast_info."""
        yf_ticker = CDE_TICKERS.get(pair.upper())
        if not yf_ticker:
            return None
        try:
            info = yf.Ticker(yf_ticker).fast_info
            ltp = getattr(info, "last_price", None)
            if not ltp:
                return None
            return {"ltp": round(float(ltp), 4), "last_price": round(float(ltp), 4),
                    "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.debug(f"LiqDataFetcher: CDE quote failed for {pair}: {e}")
            return None
