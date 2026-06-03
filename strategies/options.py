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
from data.calendar import get_fo_expiry

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

# Default underlying → proxy ETF for OHLCV-based trend reading (NOT for strike calc)
# FINNIFTY and MIDCPNIFTY are highly correlated to NIFTY (~0.95+) so NIFTYBEES is a
# valid trend proxy. Strike calculation always uses the actual index live quote.
UNDERLYING_ETF = {
    "NIFTY":      "NIFTYBEES-EQ",
    "BANKNIFTY":  "BANKBEES-EQ",
    "SENSEX":     "HDFCSENSEX-EQ",
    "FINNIFTY":   "NIFTYBEES-EQ",    # NIFTY proxy — highly correlated
    "MIDCPNIFTY": "NIFTYBEES-EQ",    # NIFTY proxy — highly correlated
}

# Index live-quote symbol + exchange (for ATM strike calculation)
# Index live-quote is mandatory for strike calculation — ETF fallback disabled (ratio not exact)
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

# ── SELL mode (option writing) ────────────────────────────────────────────────
# Strategy: SELL CE when bearish (collect premium, profit if market stays flat/falls)
#           SELL PE when bullish (collect premium, profit if market stays flat/rises)
# Exit rules for short options:
#   Target : premium decays to 30% of received → buy back cheap → keep 70% of premium
#   Stop   : premium rises to 200% of received → buy back expensive → cap loss at 100%
# P&L for SELL: positive when premium falls, negative when premium rises
STOP_LOSS_PCT  = 1.00   # buy back if premium doubles (100% loss on received premium)
TARGET_PCT     = 0.70   # buy back when 70% of premium has decayed (keep 70% of credit)

# Premium bounds per unit (₹)
# Too cheap = deep OTM with near-zero delta; too expensive = capital risk per lot is excessive.
# Per-underlying caps based on actual observed premiums (June 2026 audit):
#   NIFTY weekly ATM:    ₹150–₹350  (75 lots × ₹350 = ₹26k lot cost — fits ₹1L budget)
#   BANKNIFTY monthly:  ₹400–₹700  (15 lots × ₹700 = ₹10.5k lot cost — fits ₹1L budget)
#   Generic floor:       ₹80 — below this is deep OTM with near-zero delta
MIN_PREMIUM = 50
MAX_PREMIUM_PER_UNDERLYING = {
    "NIFTY":      700,   # NIFTY ~23k level: ATM premium ~₹200-500; raised from 500
    "BANKNIFTY":  1500,  # BANKNIFTY ~53k level: ATM premium ~₹900-1400; raised from 700
    "SENSEX":     800,   # SENSEX premiums: similar to NIFTY scale
    "FINNIFTY":   400,
    "MIDCPNIFTY": 400,
}
MAX_PREMIUM = 700       # default fallback if underlying not in map above
MAX_PREMIUM_PER_LOT = 20_000  # raised from ₹15k — NIFTY 75 lots × ₹250 = ₹18,750


SIGNAL_COOLDOWN_SECONDS    = 600   # 10 min — suppress duplicate signals on same option symbol
UNDERLYING_COOLDOWN_SECONDS = 1800  # 30 min — one signal per underlying per 30 min


class OptionsStrategy:
    """
    Directional options strategy for index options (Nifty / BankNifty).

    Entry logic:
      1. Read trend of the index ETF from OHLCV (EMA 9/21 + RSI)
      2. Confirm strong trend (not ranging)
      3. Find ATM or 1-OTM strike in scrip master for nearest valid expiry
      4. Generate TradeSignal with SL and target priced in premium terms
    """

    # Class-level cooldown trackers
    _last_signal_time:     dict = {}   # option_symbol → last signal datetime (10 min)
    _last_underlying_time: dict = {}   # underlying   → last signal datetime (30 min)

    def __init__(self,
                 underlying: str = "NIFTY",
                 strike_offset: int = 0,         # 0 = ATM, 1 = 1 OTM, -1 = 1 ITM
                 min_trend_strength: float = 0.65):
        self.underlying       = underlying.upper()
        self.strike_offset    = strike_offset
        self.min_trend_strength = min_trend_strength
        self.lot_size         = LOT_SIZES.get(self.underlying, 50)
        self.etf_symbol       = UNDERLYING_ETF.get(self.underlying)  # None if no ETF proxy
        self.fo_exchange      = FO_EXCHANGE.get(self.underlying, "nse_fo")

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signal(self, data_manager, vix: float = None) -> Optional[TradeSignal]:
        """
        Scan for an options entry opportunity.
        Returns a TradeSignal (product=NRML, exchange=nse_fo) or None.

        Args:
            vix: Current India VIX value. If > 20 (elevated vol), options entry is blocked
                 per SOP — high VIX means gamma risk is unacceptably elevated.
        """
        # SOP: Block new short options when VIX is elevated (>20)
        # Theta income requires a stable vol environment; spikes indicate gap risk
        if vix is not None and vix > 20:
            logger.info(
                f"Options [{self.underlying}]: VIX={vix:.1f} > 20 — "
                f"skipping (elevated vol, gamma risk unacceptable)"
            )
            return None

        # Step 1: Get OHLCV for trend detection.
        # Prefer ETF proxy; fall back to index symbol if no ETF available (FINNIFTY, MIDCPNIFTY).
        ohlcv_symbol  = self.etf_symbol or self.underlying
        ohlcv_exchange = "nse_cm"

        df = data_manager.get_ohlcv(ohlcv_symbol, ohlcv_exchange, limit=60)
        if len(df) < 25:
            logger.debug(f"Options: not enough OHLCV data for {ohlcv_symbol}")
            return None

        trend, strength, _etf_price = self._detect_trend(df)
        if trend == "neutral" or strength < self.min_trend_strength:
            logger.debug(f"Options: no clear trend (trend={trend} strength={strength:.0%})")
            return None

        # Step 2: Find nearest valid expiry
        expiry = self._nearest_valid_expiry()
        if expiry is None:
            return None

        # Step 3: Get actual index level for ATM strike calculation.
        # Tier 1: direct index quote (works for BANKNIFTY, may fail for NIFTY/SENSEX).
        # Tier 2: near-month futures quote (accurate to < 0.5%, always liquid).
        # Tier 3: ETF × 100 from last OHLCV bar (last resort, ±1% error acceptable for ATM ±step).
        idx_sym, idx_exc = INDEX_QUOTE.get(self.underlying, (self.etf_symbol, "nse_cm"))
        idx_quote = data_manager.get_live_quote(idx_sym, idx_exc)
        if idx_quote and idx_quote.get("ltp", 0) > 0:
            index_price = idx_quote["ltp"]
            logger.debug(f"Options: {self.underlying} index level from direct quote: {index_price:.0f}")
        else:
            # Tier 2: near-month futures — futures price ≈ spot + small carry (<0.5%)
            # For ATM strike rounded to step, carry difference of 130pts (0.5% on 26000) is < 3 strikes.
            futures_symbols = {
                "NIFTY":     "NIFTY26JUNFUT",
                "BANKNIFTY": "BANKNIFTY26JUNFUT",
                "SENSEX":    "SENSEX26JUNFUT",
            }
            fut_sym = futures_symbols.get(self.underlying)
            fut_quote = data_manager.get_live_quote(fut_sym, self.fo_exchange) if fut_sym else None
            if fut_quote and fut_quote.get("ltp", 0) > 0:
                index_price = fut_quote["ltp"]
                logger.info(f"Options: {self.underlying} index level from futures {fut_sym}: {index_price:.0f}")
            else:
                # Tier 3: ETF last close × 100 — least accurate but better than skipping entirely.
                # NIFTYBEES × 100 ≈ NIFTY ±1%. At step=50, max strike error = 2 strikes (acceptable for ATM).
                if self.etf_symbol and len(df) > 0:
                    etf_close = float(df["close"].iloc[-1])
                    index_price = etf_close * 100
                    logger.warning(
                        f"Options: {self.underlying} using ETF fallback "
                        f"({self.etf_symbol} ₹{etf_close:.2f} × 100 = {index_price:.0f})"
                    )
                else:
                    logger.warning(f"Options: {self.underlying} index level unavailable — all 3 tiers failed")
                    return None

        step = STRIKE_STEP.get(self.underlying, 50)
        atm_strike = round(index_price / step) * step
        strike     = atm_strike + (self.strike_offset * step)
        current_price = index_price  # used in reasoning string below

        # SELL mode: sell the OPPOSITE option type to collect premium
        # Bearish → SELL CE (call writers profit when market falls/stays flat)
        # Bullish → SELL PE (put writers profit when market rises/stays flat)
        option_type = "PE" if trend == "bullish" else "CE"

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
        max_prem = MAX_PREMIUM_PER_UNDERLYING.get(self.underlying, MAX_PREMIUM)
        if premium < MIN_PREMIUM:
            logger.info(
                f"Options: {option_symbol} premium ₹{premium:.2f} below"
                f" ₹{MIN_PREMIUM} — deep OTM, skipping"
            )
            return None
        if premium > max_prem:
            logger.info(
                f"Options: {option_symbol} premium ₹{premium:.2f} above"
                f" ₹{max_prem} ({self.underlying} cap) — skipping"
            )
            return None
        lot_cost = premium * self.lot_size

        if lot_cost > MAX_PREMIUM_PER_LOT:
            logger.info(
                f"Options: {option_symbol} lot cost ₹{lot_cost:,.0f} exceeds"
                f" max ₹{MAX_PREMIUM_PER_LOT:,.0f}"
            )
            return None

        # Step 6: SL/target for SELL (short option) position
        # We receive premium upfront. Exit by buying back:
        #   Stop loss : premium rises to 2× received → buy back at 2× cost → loss = premium received
        #   Target    : premium falls to 30% of received → buy back cheap → keep 70% of credit
        stop_loss = round(premium * (1 + STOP_LOSS_PCT), 2)   # rises 100% → stop out
        target    = round(premium * (1 - TARGET_PCT),    2)   # falls 70% → take profit
        days_left = (expiry - date.today()).days

        confidence = min(strength + 0.05, 0.90)  # slight boost for options setup
        reasoning  = (
            f"Options {option_type} on {self.underlying}. "
            f"Trend={trend} (strength={strength:.0%}). "
            f"Strike={strike}. Expiry={expiry} ({days_left}d). "
            f"Premium=₹{premium:.2f}/unit. Lot cost=₹{lot_cost:,.0f}."
        )

        from datetime import datetime as _datetime

        # Underlying-level 30-min cooldown — one signal per underlying per 30 min
        # Prevents NIFTY CE at 09:30 and NIFTY PE at 09:35 both firing in quick succession
        last_ul = OptionsStrategy._last_underlying_time.get(self.underlying)
        if last_ul is not None:
            elapsed_ul = (_datetime.now() - last_ul).total_seconds()
            if elapsed_ul < UNDERLYING_COOLDOWN_SECONDS:
                logger.debug(
                    f"Options underlying cooldown for {self.underlying} "
                    f"({elapsed_ul:.0f}s < {UNDERLYING_COOLDOWN_SECONDS}s) — skipping"
                )
                return None

        # Per-symbol 10-min cooldown — suppress duplicate signal on same option contract
        last_sig = OptionsStrategy._last_signal_time.get(option_symbol)
        if last_sig is not None:
            elapsed = (_datetime.now() - last_sig).total_seconds()
            if elapsed < SIGNAL_COOLDOWN_SECONDS:
                logger.debug(
                    f"Options signal cooldown active for {option_symbol} "
                    f"({elapsed:.0f}s < {SIGNAL_COOLDOWN_SECONDS}s) — skipping"
                )
                return None

        OptionsStrategy._last_underlying_time[self.underlying] = _datetime.now()
        OptionsStrategy._last_signal_time[option_symbol]       = _datetime.now()

        logger.info(
            f"Options signal: SELL {option_symbol} @ ₹{premium:.2f}"
            f" credit=₹{lot_cost:,.0f} conf={confidence:.0%}"
        )

        return TradeSignal(
            symbol    = option_symbol,
            exchange  = self.fo_exchange,
            action    = "SELL",
            price     = premium,
            strategy  = "options",
            confidence= confidence,
            stop_loss = stop_loss,   # premium level to buy back at loss
            target    = target,      # premium level to buy back at profit
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
        Return the nearest F&O expiry that is at least MIN_DAYS_TO_EXPIRY away.

        Delegates to calendar.get_fo_expiry — single source of truth for expiry
        dates shared by calendar context, AI prompt, and options strategy.
        This ensures days_to_expiry shown to AI == actual DTE used at entry.
        """
        expiry = get_fo_expiry(
            self.underlying,
            min_days=MIN_DAYS_TO_EXPIRY,
        )
        if expiry is None:
            logger.warning(f"Options: no valid expiry found for {self.underlying} "
                           f"(min_days={MIN_DAYS_TO_EXPIRY})")
        return expiry

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
        # Try specific patterns anchored to the computed expiry date only.
        # The broad fallback (UNDERLYING%STRIKE%TYPE with no expiry) is intentionally
        # removed — it performs substring matching and caused production incident on
        # 2026-06-01 where SENSEX%8500%PE matched SENSEX2660468500PE (strike=68500,
        # 13,500pts deep OTM) because "8500" is a substring of "68500".
        candidates = [
            # Pattern 1: canonical Kotak format — underlying + weekly code + strike + type
            f"{self.underlying}{expiry_str}%{strike}{option_type}",
            # Pattern 2: YYYYMMDD date format variant
            f"{self.underlying}{expiry.strftime('%y%m%d')}%{strike}{option_type}",
            # Pattern 3: expiry anywhere in symbol but strike must be exact suffix
            f"{self.underlying}%{expiry_str}%{strike}{option_type}",
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
                        # ── Exact strike validation ─────────────────────────────────
                        # Kotak symbols are UNDERLYING + EXPIRY_CODE + STRIKE + TYPE
                        # with no separator between expiry digits and strike digits.
                        # endswith() and regex-based approaches fail because "8500"
                        # is a suffix of "68500" — the production bug on 2026-06-01.
                        #
                        # Correct approach: strip the known underlying prefix, then
                        # strip the known expiry_str (primary or alt format), and
                        # verify exactly "{strike}{option_type}" remains.
                        found = row[0]
                        accepted = False

                        if found.startswith(self.underlying):
                            after_ul = found[len(self.underlying):]
                            expected_tail = f"{strike}{option_type}"
                            # Try primary expiry format (weekly code or monthly YYMMM)
                            if after_ul.startswith(expiry_str):
                                remainder = after_ul[len(expiry_str):]
                                if remainder == expected_tail:
                                    accepted = True
                            # Try alternate expiry format (YYMMDD / 6-char)
                            if not accepted:
                                alt_expiry = expiry.strftime('%y%m%d')
                                if after_ul.startswith(alt_expiry):
                                    remainder = after_ul[len(alt_expiry):]
                                    if remainder == expected_tail:
                                        accepted = True

                        if not accepted:
                            logger.warning(
                                f"Options symbol lookup: rejected {found!r} — "
                                f"strike in symbol != expected {strike} "
                                f"(expiry_str={expiry_str!r}, pattern={like_pattern!r})"
                            )
                            continue

                        logger.debug(f"Options symbol found: {found} (pattern={like_pattern!r})")
                        return found
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
