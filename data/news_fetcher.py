"""
News & Market Data Fetcher
Pulls India VIX, FII/DII flows, and NSE corporate announcements from NSE public APIs.
No API key required — all endpoints are NSE's public JSON feeds.

Usage:
    fetcher = NewsFetcher()
    vix  = fetcher.get_india_vix()              # float
    flows = fetcher.get_fii_dii_flows()         # dict
    news = fetcher.get_announcements("TCS")     # list[str]
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# NSE public API endpoints — no auth required
_NSE_INDICES_URL      = "https://www.nseindia.com/api/allIndices"
_NSE_FII_DII_URL      = "https://www.nseindia.com/api/fiidiiTradeReact"
_NSE_ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corp-info?symbol={symbol}&market=equities&issuer=&corpType=announcements&fromDate=&toDate="

# Browser-like headers required by NSE (they block plain requests)
_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/",
    "Connection":      "keep-alive",
}

_TIMEOUT = 8  # seconds — NSE can be slow from VPS; 3s caused frequent timeouts


class NewsFetcher:
    """
    Fetches live market intelligence from NSE public endpoints.
    Maintains a session with cookies (NSE requires them after first hit).
    Results are cached with short TTLs to avoid hammering NSE.
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._cookie_loaded = False
        self._cache: dict = {}   # key → (value, expires_at)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get_india_vix(self) -> Optional[float]:
        """
        Return current India VIX value.
        Cached for 5 minutes (VIX doesn't change second-to-second).

        Returns:
            float (e.g. 14.25) or None if unavailable.
        """
        cached = self._get_cache("india_vix")
        if cached is not None:
            return cached

        for attempt in range(3):
            try:
                self._ensure_cookies(force=(attempt > 0))
                resp = self._session.get(_NSE_INDICES_URL, timeout=_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                for index in data.get("data", []):
                    if index.get("index") == "INDIA VIX":
                        vix = float(index.get("last", 0))
                        self._set_cache("india_vix", vix, ttl=300)
                        logger.info(f"India VIX: {vix:.2f}")
                        return vix
                # Data returned but VIX row missing — force cookie refresh and retry
                self._cookie_loaded = False
            except Exception as e:
                if attempt == 2:
                    logger.debug(f"VIX fetch failed after {attempt+1} attempts: {e}")
        return None

    def get_fii_dii_flows(self) -> dict:
        """
        Return FII and DII net buy/sell for the most recent available trading day.
        Cached for 30 minutes (data updates once daily after market close).

        Returns:
            {
                "date":    "28-May-2026",
                "fii_net": 1234.56,    # ₹ crore, positive = net buy, negative = net sell
                "dii_net": -456.78,
                "fii_buy": 12345.0,
                "fii_sell": 11110.44,
                "dii_buy": 8000.0,
                "dii_sell": 8456.78,
            }
            Empty dict if unavailable.
        """
        cached = self._get_cache("fii_dii")
        if cached is not None:
            return cached

        try:
            self._ensure_cookies()
            data = self._session.get(_NSE_FII_DII_URL, timeout=_TIMEOUT).json()
            # NSE returns a list; most recent entry is usually index 0
            rows = data if isinstance(data, list) else data.get("data", [])
            if not rows:
                return {}

            # Parse FII and DII rows (two separate entries per date)
            fii_row = next((r for r in rows if "FII" in r.get("category", "").upper()), None)
            dii_row = next((r for r in rows if "DII" in r.get("category", "").upper()), None)

            result = {}
            if fii_row:
                result["date"]     = fii_row.get("date", "")
                result["fii_buy"]  = float(fii_row.get("buyValue",  0) or 0)
                result["fii_sell"] = float(fii_row.get("sellValue", 0) or 0)
                result["fii_net"]  = result["fii_buy"] - result["fii_sell"]
            if dii_row:
                result["dii_buy"]  = float(dii_row.get("buyValue",  0) or 0)
                result["dii_sell"] = float(dii_row.get("sellValue", 0) or 0)
                result["dii_net"]  = result["dii_buy"] - result["dii_sell"]

            if result:
                self._set_cache("fii_dii", result, ttl=1800)
                logger.info(
                    f"FII/DII [{result.get('date', '?')}]: "
                    f"FII={result.get('fii_net', 0):+,.0f} cr  "
                    f"DII={result.get('dii_net', 0):+,.0f} cr"
                )
            return result

        except Exception as e:
            logger.warning(f"FII/DII fetch failed: {e}")
            return {}

    def get_announcements(self, symbol: str, max_items: int = 5) -> list[str]:
        """
        Return the latest corporate announcements for a symbol as plain-text strings.
        Cached per symbol for 15 minutes.

        Args:
            symbol:    NSE trading symbol e.g. "TCS", "RELIANCE"
                       Strips the "-EQ" suffix automatically.
            max_items: How many recent announcements to return.

        Returns:
            List of strings like ["Board Meeting: Q4 results on May 30",
                                   "Dividend: ₹25 per share declared", ...]
        """
        # Strip -EQ / -BE suffix that the bot uses internally
        clean = symbol.replace("-EQ", "").replace("-BE", "").upper()
        cache_key = f"ann_{clean}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        try:
            self._ensure_cookies()
            url  = _NSE_ANNOUNCEMENTS_URL.format(symbol=clean)
            data = self._session.get(url, timeout=_TIMEOUT).json()
            items = data if isinstance(data, list) else data.get("data", [])

            announcements = []
            for item in items[:max_items]:
                # Different NSE endpoints use different field names
                subject = (item.get("subject") or item.get("desc") or
                           item.get("headline") or "")
                date_str = item.get("bm_date") or item.get("date") or ""
                if subject:
                    announcements.append(f"[{date_str}] {subject}".strip())

            self._set_cache(cache_key, announcements, ttl=900)
            logger.debug(f"Announcements for {clean}: {len(announcements)} items")
            return announcements

        except Exception as e:
            logger.debug(f"Announcements fetch failed for {symbol}: {e}")
            return []

    def get_market_headlines(self, max_items: int = 5) -> list[str]:
        """
        Return recent NSE market-wide news headlines.
        Falls back to an empty list — never raises.
        """
        cache_key = "market_headlines"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        try:
            self._ensure_cookies()
            # NSE equity news feed
            url  = "https://www.nseindia.com/api/home-market-status"
            data = self._session.get(url, timeout=_TIMEOUT).json()
            news = data.get("marketMessages", []) or []
            headlines = [n.get("description", "") for n in news[:max_items] if n.get("description")]
            self._set_cache(cache_key, headlines, ttl=300)
            return headlines
        except Exception as e:
            logger.debug(f"Headlines fetch failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # NSE cookie management (NSE returns 403 without a valid session cookie)
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_cookies(self, force: bool = False):
        """Hit the NSE homepage to get/refresh a session cookie."""
        if self._cookie_loaded and not force:
            return
        try:
            self._session.get("https://www.nseindia.com", timeout=_TIMEOUT)
            self._cookie_loaded = True
        except Exception as e:
            logger.debug(f"NSE cookie load failed (non-fatal): {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Simple TTL cache
    # ─────────────────────────────────────────────────────────────────────────

    def _get_cache(self, key: str):
        entry = self._cache.get(key)
        if entry and datetime.now() < entry[1]:
            return entry[0]
        return None

    def _set_cache(self, key: str, value, ttl: int):
        self._cache[key] = (value, datetime.now() + timedelta(seconds=ttl))
