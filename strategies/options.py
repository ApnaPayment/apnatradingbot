"""
Options Strategy
Directional CE/PE plays on Nifty / BankNifty using the underlying trend.

Approach:
  - Detect underlying trend from OHLCV (momentum of the index ETF)
  - Select ATM or 1-strike OTM call (bullish) or put (bearish)
  - Only trade when trend is clear and 2+ days to expiry remain
  - Risk = premium paid × lot_size (defined, limited)
  - Target = 80–100% profit on premium; stop = 40% loss on premium

NSE F&O symbol format in scrip master:
  Exchange: nse_fo
  Symbol:   NIFTY2451521000CE  (underlying + expiry_date + strike + type)
  The scrip master has all valid symbols — we search it for matching options.
"""

import logging
import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from core.risk_manager import TradeSignal

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

# Default underlying → proxy ETF for OHLCV-based trend reading (NOT for strike calc)
UNDERLYING_ETF = {
    "NIFTY":      "NIFTYBEES-EQ",
    "BANKNIFTY":  "BANKBEES-EQ",
    "SENSEX":     "HDFCSENSEX-EQ",
    "FINNIFTY":   "BANKBEES-EQ",    # closest ETF proxy (financial services)
    "MIDCPNIFTY": "NIFTYBEES-EQ",   # no midcap ETF on watchlist; use Nifty as proxy
}

# Index live-quote symbol + exchange (for ATM strike calculation)
# The ETF trades at ~1/100th of the index; we need the actual index level for F&O strikes
INDEX_QUOTE = {
    "NIFTY":      ("NIFTY",      "nse_cm"),
    "BANKNIFTY":  ("BANKNIFTY",  "nse_cm"),
    "SENSEX":     ("SENSEX",     "bse_cm"),
    "FINNIFTY":   ("FINNIFTY",   "nse_cm"),
    "MIDCPNIFTY": ("MIDCPNIFTY", "nse_cm"),
}

# F&O exchange per underlying
FO_EXCHANGE = {
    "NIFTY":      "nse_fo",
    "BANKNIFTY":  "nse_fo",
    "SENSEX":     "bse_fo",
    "FINNIFTY":   "nse_fo",
    "MIDCPNIFTY": "nse_fo",
}

# Strike step (round to nearest N for ATM)
STRIKE_STEP = {
    "NIFTY":      50,
    "BANKNIFTY":  100,
    "SENSEX":     100,
    "FINNIFTY":   50,
    "MIDCPNIFTY": 25,
}

# Lot sizes (SEBI periodically revises these — check NSE if in doubt)
LOT_SIZES = {
    "NIFTY":     75,
    "BANKNIFTY": 15,
    "SENSEX":    10,
    "FINNIFTY":  40,
    "MIDCPNIFTY":75,
}

# Minimum days to expiry before we refuse to enter new positions
MIN_DAYS_TO_EXPIRY = 2

# Premium thresholds as % of entry premium
STOP_LOSS_PCT  = 0.30   # exit if premium drops 30%
TARGET_PCT     = 0.50   # exit at 50% gain on premium

# Premium bounds per unit (₹)
MIN_PREMIUM = 20        # below this = deep OTM lottery ticket
MAX_PREMIUM_PER_LOT = 10_000  # above this = too expensive per lot


class OptionsStrategy:
    """
    Directional options strategy for index options (Nifty / BankNifty).

    Entry logic:
      1. Read trend of the index ETF from OHLCV (EMA 9/21 + RSI)
      2. Confirm strong trend (not ranging)
      3. Find ATM or 1-OTM strike in scrip master for nearest valid expiry
      4. Generate TradeSignal with SL and target priced in premium terms
    """

    def __init__(self,
                 underlying: str = "NIFTY",
                 strike_offset: int = 0,         # 0 = ATM, 1 = 1 OTM, -1 = 1 ITM
                 min_trend_strength: float = 0.65):
        self.underlying       = underlying.upper()
        self.strike_offset    = strike_offset
        self.min_trend_strength = min_trend_strength
        self.lot_size         = LOT_SIZES.get(self.underlying, 50)
        self.etf_symbol       = UNDERLYING_ETF.get(self.underlying, "NIFTYBEES-EQ")
        self.fo_exchange      = FO_EXCHANGE.get(self.underlying, "nse_fo")

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signal(self, data_manager) -> Optional[TradeSignal]:
        """
        Scan for an options entry opportunity.
        Returns a TradeSignal (product=NRML, exchange=nse_fo) or None.
        """
        # Step 1: Get ETF OHLCV for trend
        df = data_manager.get_ohlcv(self.etf_symbol, "nse_cm", limit=60)
        if len(df) < 25:
            logger.debug(f"Options: not enough ETF data for {self.etf_symbol}")
            return None

        trend, strength, _etf_price = self._detect_trend(df)
        if trend == "neutral" or strength < self.min_trend_strength:
            logger.debug(f"Options: no clear trend (trend={trend} strength={strength:.0%})")
            return None

        # Step 2: Find nearest valid expiry
        expiry = self._nearest_valid_expiry()
        if expiry is None:
            return None

        # Step 3: Get actual index level for ATM strike calculation
        # ETF price (e.g. ₹270 for NIFTYBEES) ≠ index level (e.g. 24,500 for NIFTY 50)
        idx_sym, idx_exc = INDEX_QUOTE.get(self.underlying, (self.etf_symbol, "nse_cm"))
        idx_quote = data_manager.get_live_quote(idx_sym, idx_exc)
        if idx_quote and idx_quote.get("ltp", 0) > 0:
            index_price = idx_quote["ltp"]
        else:
            # Fallback: use last close from ETF OHLCV converted via known ratio
            # (NIFTYBEES ≈ NIFTY/100, BANKBEES ≈ BANKNIFTY/100)
            logger.warning(f"Options: index quote unavailable for {idx_sym}, falling back to ETF×100")
            index_price = _etf_price * 100

        step = STRIKE_STEP.get(self.underlying, 50)
        atm_strike = round(index_price / step) * step
        strike     = atm_strike + (self.strike_offset * step)
        current_price = index_price  # used in reasoning string below

        option_type = "CE" if trend == "bullish" else "PE"

        # Step 4: Find the option in scrip master
        option_symbol = self._find_option_symbol(
            data_manager, expiry, strike, option_type
        )
        if not option_symbol:
            logger.warning(
                f"Options: {self.underlying}{expiry.strftime('%y%b').upper()}"
                f"{strike}{option_type} not found in scrip master"
            )
            return None

        # Step 5: Get live quote for the option
        quote = data_manager.get_live_quote(option_symbol, self.fo_exchange)
        if not quote or quote["ltp"] <= 0:
            logger.debug(f"Options: no live quote for {option_symbol}")
            return None

        premium = quote["ltp"]
        if premium < MIN_PREMIUM:
            logger.info(
                f"Options: {option_symbol} premium ₹{premium:.2f} below"
                f" minimum ₹{MIN_PREMIUM} — skipping (deep OTM)"
            )
            return None
        lot_cost = premium * self.lot_size

        if lot_cost > MAX_PREMIUM_PER_LOT:
            logger.info(
                f"Options: {option_symbol} premium ₹{lot_cost:,.0f} exceeds"
                f" max ₹{MAX_PREMIUM_PER_LOT:,.0f}"
            )
            return None

        # Step 6: Calculate SL and target in premium terms
        stop_loss = round(premium * (1 - STOP_LOSS_PCT), 2)
        target    = round(premium * (1 + TARGET_PCT),    2)
        days_left = (expiry - date.today()).days

        confidence = min(strength + 0.05, 0.90)  # slight boost for options setup
        reasoning  = (
            f"Options {option_type} on {self.underlying}. "
            f"Trend={trend} (strength={strength:.0%}). "
            f"Strike={strike}. Expiry={expiry} ({days_left}d). "
            f"Premium=₹{premium:.2f}/unit. Lot cost=₹{lot_cost:,.0f}."
        )

        logger.info(
            f"Options signal: BUY {option_symbol} @ ₹{premium:.2f}"
            f" lot_cost=₹{lot_cost:,.0f} conf={confidence:.0%}"
        )

        return TradeSignal(
            symbol    = option_symbol,
            exchange  = self.fo_exchange,
            action    = "BUY",
            price     = premium,
            strategy  = "options",
            confidence= confidence,
            stop_loss = stop_loss,
            target    = target,
            quantity  = self.lot_size,
            product   = "NRML",
            reasoning = reasoning,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Trend detection (from ETF OHLCV)
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_trend(self, df: pd.DataFrame) -> tuple[str, float, float]:
        """
        Returns (trend, strength, current_price).
        trend:    "bullish" | "bearish" | "neutral"
        strength: 0.0 to 1.0
        """
        close = df["close"]
        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()

        # RSI
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        latest  = df.iloc[-1]
        current_price = float(latest["close"])
        fast    = float(ema9.iloc[-1])
        slow    = float(ema21.iloc[-1])
        rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        # EMA separation as a % of price
        ema_spread_pct = abs(fast - slow) / slow * 100

        if fast > slow and rsi_val > 55:
            trend    = "bullish"
            strength = min(0.5 + ema_spread_pct * 5 + (rsi_val - 55) / 100, 0.95)
        elif fast < slow and rsi_val < 45:
            trend    = "bearish"
            strength = min(0.5 + ema_spread_pct * 5 + (45 - rsi_val) / 100, 0.95)
        else:
            trend    = "neutral"
            strength = 0.0

        return trend, round(strength, 3), current_price

    # ─────────────────────────────────────────────────────────────────────────
    # Expiry helpers
    # ─────────────────────────────────────────────────────────────────────────

    # NSE/BSE expiry weekday per underlying
    #   NIFTY / FINNIFTY / MIDCPNIFTY → Tuesday  (NSE, changed Jan 2025)
    #   BANKNIFTY                      → Wednesday (NSE, changed 2024)
    #   SENSEX                         → Thursday  (BSE weekly)
    _EXPIRY_WEEKDAY = {
        "NIFTY":      1,   # Tuesday
        "FINNIFTY":   1,
        "MIDCPNIFTY": 1,
        "BANKNIFTY":  2,   # Wednesday
        "SENSEX":     3,   # Thursday
    }

    # Underlyings that only have monthly options in Kotak's scrip master
    _MONTHLY_ONLY = {"BANKNIFTY"}

    def _nearest_valid_expiry(self) -> Optional[date]:
        """
        Return the nearest NSE F&O expiry that is at least MIN_DAYS_TO_EXPIRY away.

        NIFTY/FINNIFTY  : weekly (every Tuesday)
        BANKNIFTY/SENSEX: monthly (last expiry weekday of the month)
        """
        today    = date.today()
        weekday  = self._EXPIRY_WEEKDAY.get(self.underlying, 1)   # default Tuesday

        if self.underlying in self._MONTHLY_ONLY:
            # Find the last <weekday> of each upcoming month
            candidates = []
            for month_offset in range(4):   # check next 4 months
                yr  = today.year + (today.month + month_offset - 1) // 12
                mon = (today.month + month_offset - 1) % 12 + 1
                # Last day of that month
                import calendar
                last_day = calendar.monthrange(yr, mon)[1]
                d = date(yr, mon, last_day)
                # Walk back to the target weekday
                while d.weekday() != weekday:
                    d -= timedelta(days=1)
                if (d - today).days >= MIN_DAYS_TO_EXPIRY:
                    candidates.append(d)
            if not candidates:
                logger.warning(f"Options: no monthly expiry found for {self.underlying}")
                return None
            return candidates[0]
        else:
            # Weekly: find the next <weekday> (e.g. Tuesday for NIFTY)
            days_until = (weekday - today.weekday()) % 7
            candidates = []
            for n in range(8):   # check next 8 weekly expiries
                candidate = today + timedelta(days=days_until + n * 7)
                if (candidate - today).days >= MIN_DAYS_TO_EXPIRY:
                    candidates.append(candidate)
            if not candidates:
                logger.warning(f"Options: no weekly expiry found for {self.underlying}")
                return None
            return candidates[0]

    @staticmethod
    def _date_to_kotak_weekly(d: date) -> str:
        """
        Convert a date to Kotak weekly option format: YY + single-digit/letter month + DD.
        Jan-Sep → 1-9,  Oct→O, Nov→N, Dec→D
        Example: 2026-06-02 → "26602"
        """
        month_code = {10: "O", 11: "N", 12: "D"}
        m = month_code.get(d.month, str(d.month))
        return f"{d.strftime('%y')}{m}{d.strftime('%d')}"

    def _expiry_to_scrip_str(self, expiry: date) -> str:
        """
        Return the Kotak scrip-master expiry substring for LIKE-pattern matching.
        Weekly (NIFTY): YY + single-char-month + DD  e.g. "26602" for Jun 2 2026
        Monthly (BANKNIFTY/SENSEX): YYMMM             e.g. "26JUN"
        """
        if self.underlying in self._MONTHLY_ONLY:
            return expiry.strftime("%y%b").upper()          # e.g. "26JUN"
        return self._date_to_kotak_weekly(expiry)           # e.g. "26602"

    def _find_option_symbol(
        self,
        data_manager,
        expiry: date,
        strike: int,
        option_type: str,
    ) -> Optional[str]:
        """
        Search the scrip master for the matching option symbol.
        Tries multiple common symbol formats used by Kotak Neo F&O.
        """
        expiry_str = self._expiry_to_scrip_str(expiry)
        # Try specific patterns first (exact expiry), then fall back to broader search
        candidates = [
            # Anchor to start — NIFTY% not %NIFTY% to avoid matching BANKNIFTY/FINNIFTY
            f"{self.underlying}{expiry_str}%{strike}{option_type}",
            f"{self.underlying}{expiry.strftime('%y%m%d')}%{strike}{option_type}",
            f"{self.underlying}%{expiry_str}%{strike}%{option_type}",
            # Broad fallback — any expiry, anchored at start
            f"{self.underlying}%{strike}%{option_type}",
        ]

        try:
            with data_manager._get_conn() as conn:
                for like_pattern in candidates:
                    row = conn.execute(
                        """SELECT symbol FROM scrip_master
                           WHERE symbol LIKE ? AND exchange=?
                           ORDER BY symbol LIMIT 1""",
                        (like_pattern, self.fo_exchange)
                    ).fetchone()
                    if row:
                        logger.debug(f"Options symbol found: {row[0]} (pattern={like_pattern!r})")
                        return row[0]
        except Exception as e:
            logger.error(f"Options symbol lookup failed: {e}")
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Greeks (Black-Scholes approximation)
    # ─────────────────────────────────────────────────────────────────────────

    def black_scholes_greeks(
        self,
        spot: float,
        strike: float,
        tte_years: float,   # time to expiry in years
        volatility: float,  # annualised IV as decimal, e.g. 0.15 = 15%
        risk_free: float = 0.065,   # RBI repo rate approx
        option_type: str = "CE",
    ) -> dict:
        """
        Black-Scholes Greeks for a European option.
        Returns: {price, delta, gamma, theta, vega, iv_used}
        """
        from math import log, sqrt, exp
        from statistics import NormalDist

        norm = NormalDist(0, 1)

        if tte_years <= 0 or volatility <= 0:
            return {"price": 0, "delta": 0, "gamma": 0, "theta": 0, "vega": 0}

        d1 = (log(spot / strike) + (risk_free + 0.5 * volatility ** 2) * tte_years) / (
            volatility * sqrt(tte_years)
        )
        d2 = d1 - volatility * sqrt(tte_years)

        nd1  = norm.cdf(d1)
        nd2  = norm.cdf(d2)
        nd1_ = norm.pdf(d1)

        if option_type == "CE":
            price = spot * nd1 - strike * exp(-risk_free * tte_years) * nd2
            delta = nd1
        else:  # PE
            price = strike * exp(-risk_free * tte_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            delta = nd1 - 1

        gamma = nd1_ / (spot * volatility * sqrt(tte_years))
        vega  = spot * nd1_ * sqrt(tte_years) / 100   # per 1% IV change
        theta = (
            -(spot * nd1_ * volatility) / (2 * sqrt(tte_years))
            - risk_free * strike * exp(-risk_free * tte_years) * nd2
        ) / 365   # per day

        return {
            "price":   round(price, 2),
            "delta":   round(delta, 4),
            "gamma":   round(gamma, 6),
            "theta":   round(theta, 2),
            "vega":    round(vega, 2),
            "iv_used": volatility,
        }

    def estimate_iv(
        self,
        market_price: float,
        spot: float,
        strike: float,
        tte_years: float,
        option_type: str = "CE",
        risk_free: float = 0.065,
    ) -> float:
        """
        Estimate implied volatility via bisection (Newton's method alternative).
        Returns IV as a decimal (e.g. 0.18 = 18%).
        """
        lo, hi = 0.01, 5.0
        for _ in range(100):
            mid = (lo + hi) / 2
            price = self.black_scholes_greeks(
                spot, strike, tte_years, mid, risk_free, option_type
            )["price"]
            if abs(price - market_price) < 0.01:
                return mid
            if price < market_price:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
