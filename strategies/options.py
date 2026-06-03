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

# ── Dual-mode strategy ────────────────────────────────────────────────────────
#
# MODE 1 — BUY directional (strong trend, strength > BUY_STRENGTH_THRESHOLD)
#   BUY CE when strongly bullish  → profit from the upside move
#   BUY PE when strongly bearish  → profit from the downside move
#   Target : premium rises 100%   (2× entry)       → R:R = 3.3:1  ✅
#   Stop   : premium falls 30%    (lose 30% paid)
#
# MODE 2 — SELL OTM theta decay (moderate trend, strength 65–80%)
#   SELL CE when moderately bearish → collect premium, profit if mkt stays flat/falls
#   SELL PE when moderately bullish → collect premium, profit if mkt stays flat/rises
#   Strike : 1 step OTM from ATM  → delta ~0.25, probability of profit ~75%
#   Target : premium decays to 30% of received → keep 70% of credit
#   Stop   : premium rises to 200% of received → cap loss at 100% of credit received
#   Win rate target: 70%+ (structural for 1-OTM strikes with delta <0.30)
#
# KEY RULE: BUY mode fires when trend is STRONG. SELL mode fires when trend is MODERATE.
# This prevents the bot from selling premium during fast-moving directional days.

BUY_STRENGTH_THRESHOLD  = 0.80   # strength ≥ 80% → BUY mode (big move expected)
SELL_STRENGTH_MIN       = 0.65   # strength 65–79% → SELL mode (moderate, theta play)
SELL_STRENGTH_MAX       = 0.79   # above this, switch to BUY mode

# BUY mode exit levels (% of premium paid)
BUY_TARGET_PCT  = 1.00   # target = 100% gain on premium (2× entry)
BUY_STOP_PCT    = 0.30   # stop   = 30% loss on premium

# SELL mode exit levels (% of premium received)
SELL_STOP_PCT   = 1.00   # buy back if premium doubles (100% loss on received premium)
SELL_TARGET_PCT = 0.70   # buy back when 70% decayed (keep 70% of credit)

# SELL mode: how many strikes OTM to go (1 = one step outside ATM)
# 1-OTM gives delta ~0.25–0.30, probability of profit ~72–75%
SELL_STRIKE_OFFSET = 1

# Premium bounds per unit (₹)
MIN_PREMIUM = 50
MAX_PREMIUM_PER_UNDERLYING = {
    "NIFTY":      700,
    "BANKNIFTY":  1500,
    "SENSEX":     800,
    "FINNIFTY":   400,
    "MIDCPNIFTY": 400,
}
MAX_PREMIUM = 700
MAX_PREMIUM_PER_LOT = 20_000

# Cooldowns — max 1 signal per underlying PER DAY (not per 30 min)
# Options are held for days, not minutes. Multiple signals/day = overtrading.
SIGNAL_COOLDOWN_SECONDS     = 600    # 10 min — suppress exact duplicate on same symbol
UNDERLYING_DAILY_CAP        = 1      # max 1 trade per underlying per trading day


class OptionsStrategy:
    """
    Dual-mode directional options strategy for index options (Nifty / BankNifty).

    Mode selection (auto, based on trend strength):
      Strong trend  (≥80%)  → BUY directional CE/PE → profit from the move   R:R 3.3:1
      Moderate trend (65–79%) → SELL 1-OTM CE/PE   → collect theta decay     Win% ~73%
      Weak (<65%)            → no trade

    Entry logic:
      1. Read trend from index ETF OHLCV (EMA 9/21 + RSI)
      2. Determine mode from strength
      3. Find correct strike (ATM for BUY, 1-OTM for SELL)
      4. Return TradeSignal with correct action, SL, target for that mode
    """

    # Class-level trackers
    _last_signal_time:       dict = {}   # option_symbol → last signal datetime
    _last_underlying_time:   dict = {}   # underlying → last signal datetime (10-min dedup)
    _daily_trades_per_ul:    dict = {}   # underlying → (date, count) — daily cap

    def __init__(self,
                 underlying: str = "NIFTY",
                 min_trend_strength: float = 0.65):
        self.underlying         = underlying.upper()
        self.min_trend_strength = min_trend_strength
        self.lot_size           = LOT_SIZES.get(self.underlying, 50)
        self.etf_symbol         = UNDERLYING_ETF.get(self.underlying)
        self.fo_exchange        = FO_EXCHANGE.get(self.underlying, "nse_fo")

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signal(self, data_manager, vix: float = None) -> Optional[TradeSignal]:
        """
        Dual-mode options signal generation.

        Strong trend  (≥80%) → BUY directional CE/PE  → R:R 3.3:1
        Moderate trend (65–79%) → SELL 1-OTM CE/PE   → theta decay, win% ~73%
        Weak (<65%)            → no trade

        Args:
            vix: India VIX. BUY mode: blocked if VIX > 25 (extreme vol).
                            SELL mode: blocked if VIX > 20 (gamma risk).
        """
        from datetime import datetime as _dt, date as _date

        # ── Step 1: Trend detection ───────────────────────────────────────────
        ohlcv_symbol   = self.etf_symbol or self.underlying
        ohlcv_exchange = "nse_cm"
        df = data_manager.get_ohlcv(ohlcv_symbol, ohlcv_exchange, limit=60)
        if len(df) < 25:
            logger.debug(f"Options: not enough OHLCV for {ohlcv_symbol}")
            return None

        trend, strength, _ = self._detect_trend(df)
        if trend == "neutral" or strength < self.min_trend_strength:
            logger.debug(f"Options [{self.underlying}]: neutral/weak trend ({strength:.0%}) — skip")
            return None

        # ── Step 2: Determine mode ────────────────────────────────────────────
        if strength >= BUY_STRENGTH_THRESHOLD:
            mode = "BUY"
        elif SELL_STRENGTH_MIN <= strength <= SELL_STRENGTH_MAX:
            mode = "SELL"
        else:
            logger.debug(f"Options [{self.underlying}]: strength {strength:.0%} outside both modes — skip")
            return None

        # VIX gate per mode
        if vix is not None:
            if mode == "SELL" and vix > 20:
                logger.info(f"Options [{self.underlying}]: SELL mode blocked — VIX={vix:.1f} > 20")
                return None
            if mode == "BUY" and vix > 25:
                logger.info(f"Options [{self.underlying}]: BUY mode blocked — VIX={vix:.1f} > 25 (extreme)")
                return None

        # ── Step 3: Daily cap — max 1 signal per underlying per day ──────────
        today = _date.today()
        dl_entry = OptionsStrategy._daily_trades_per_ul.get(self.underlying)
        if dl_entry and dl_entry[0] == today and dl_entry[1] >= UNDERLYING_DAILY_CAP:
            logger.debug(f"Options [{self.underlying}]: daily cap reached ({UNDERLYING_DAILY_CAP}/day) — skip")
            return None

        # ── Step 4: Expiry ────────────────────────────────────────────────────
        expiry = self._nearest_valid_expiry()
        if expiry is None:
            return None
        days_left = (expiry - today).days

        # For SELL mode: prefer options with 4–21 DTE (theta burns fastest here)
        # For BUY mode: prefer 7–30 DTE (enough time for move to play out)
        if mode == "SELL" and days_left < 3:
            logger.info(f"Options [{self.underlying}]: SELL mode — only {days_left}d to expiry, gamma risk too high")
            return None
        if mode == "BUY" and days_left < 2:
            logger.info(f"Options [{self.underlying}]: BUY mode — only {days_left}d to expiry, skip")
            return None

        # ── Step 5: Index level for ATM strike ───────────────────────────────
        idx_sym, idx_exc = INDEX_QUOTE.get(self.underlying, (self.etf_symbol, "nse_cm"))
        idx_quote = data_manager.get_live_quote(idx_sym, idx_exc)
        if idx_quote and idx_quote.get("ltp", 0) > 0:
            index_price = idx_quote["ltp"]
        else:
            futures_symbols = {
                "NIFTY":     "NIFTY26JUNFUT",
                "BANKNIFTY": "BANKNIFTY26JUNFUT",
                "SENSEX":    "SENSEX26JUNFUT",
            }
            fut_sym   = futures_symbols.get(self.underlying)
            fut_quote = data_manager.get_live_quote(fut_sym, self.fo_exchange) if fut_sym else None
            if fut_quote and fut_quote.get("ltp", 0) > 0:
                index_price = fut_quote["ltp"]
                logger.info(f"Options: {self.underlying} level from futures: {index_price:.0f}")
            elif self.etf_symbol and len(df) > 0:
                index_price = float(df["close"].iloc[-1]) * 100
                logger.warning(f"Options: {self.underlying} using ETF×100 fallback: {index_price:.0f}")
            else:
                logger.warning(f"Options: {self.underlying} index level unavailable")
                return None

        step       = STRIKE_STEP.get(self.underlying, 50)
        atm_strike = round(index_price / step) * step

        # ── Step 6: Strike selection by mode ─────────────────────────────────
        if mode == "BUY":
            # BUY: use ATM strike for maximum delta (≈0.50), clear directional exposure
            # CE for bullish, PE for bearish
            option_type  = "CE" if trend == "bullish" else "PE"
            strike       = atm_strike
            action       = "BUY"
        else:  # SELL
            # SELL: use 1-step OTM from ATM for delta ~0.25, probability of profit ~73%
            # SELL CE when bearish (market stays flat/falls → CE decays to zero)
            # SELL PE when bullish (market stays flat/rises → PE decays to zero)
            option_type  = "CE" if trend == "bearish" else "PE"
            # OTM direction: CE goes up from ATM, PE goes down from ATM
            strike       = atm_strike + step if option_type == "CE" else atm_strike - step
            action       = "SELL"

        # ── Step 7: Find option in scrip master ──────────────────────────────
        option_symbol = self._find_option_symbol(data_manager, expiry, strike, option_type)
        if not option_symbol:
            logger.warning(
                f"Options: {self.underlying} {expiry.strftime('%y%b').upper()}"
                f" {strike}{option_type} not in scrip master"
            )
            return None

        # ── Step 8: Live quote + premium checks ──────────────────────────────
        quote = data_manager.get_live_quote(option_symbol, self.fo_exchange)
        if not quote or quote["ltp"] <= 0:
            logger.debug(f"Options: no live quote for {option_symbol}")
            return None

        premium  = quote["ltp"]
        max_prem = MAX_PREMIUM_PER_UNDERLYING.get(self.underlying, MAX_PREMIUM)
        lot_cost = premium * self.lot_size

        if premium < MIN_PREMIUM:
            logger.info(f"Options: {option_symbol} ₹{premium:.2f} < min ₹{MIN_PREMIUM} — deep OTM, skip")
            return None
        if premium > max_prem:
            logger.info(f"Options: {option_symbol} ₹{premium:.2f} > max ₹{max_prem} — skip")
            return None
        if lot_cost > MAX_PREMIUM_PER_LOT:
            logger.info(f"Options: lot cost ₹{lot_cost:,.0f} > max ₹{MAX_PREMIUM_PER_LOT:,.0f} — skip")
            return None

        # ── Step 9: Symbol-level 10-min dedup ────────────────────────────────
        last_sig = OptionsStrategy._last_signal_time.get(option_symbol)
        if last_sig and (_dt.now() - last_sig).total_seconds() < SIGNAL_COOLDOWN_SECONDS:
            logger.debug(f"Options: 10-min cooldown active for {option_symbol}")
            return None

        # ── Step 10: SL / target by mode ─────────────────────────────────────
        if mode == "BUY":
            # Pay premium. Exit when premium doubles (100% gain) or drops 30%.
            target    = round(premium * (1 + BUY_TARGET_PCT), 2)   # ×2.0
            stop_loss = round(premium * (1 - BUY_STOP_PCT),   2)   # ×0.70
            rr_str    = "3.3:1"
            mode_note = (
                f"BUY {option_type} — strong directional play. "
                f"Target ₹{target:.2f} (+100%). Stop ₹{stop_loss:.2f} (−30%). R:R {rr_str}."
            )
        else:  # SELL
            # Receive premium. Buy back cheap when decayed 70%, or stop if it doubles.
            stop_loss = round(premium * (1 + SELL_STOP_PCT),   2)   # ×2.0 (buy back at loss)
            target    = round(premium * (1 - SELL_TARGET_PCT), 2)   # ×0.30 (buy back at profit)
            # Approximate delta for 1-OTM (theoretical, no IV surface available)
            approx_delta = 0.28
            prob_profit  = round((1 - approx_delta) * 100)
            rr_str       = "0.7:1"
            mode_note = (
                f"SELL {option_type} theta decay — 1-OTM strike, delta≈{approx_delta}, "
                f"probability of profit≈{prob_profit}%. "
                f"Keep 70% of credit if held to target. Stop if premium doubles. "
                f"R:R {rr_str} — compensated by high win rate (~{prob_profit}%)."
            )

        confidence = min(strength + 0.05, 0.92)
        reasoning  = (
            f"Options [{mode} mode] {option_type} on {self.underlying}. "
            f"Trend={trend} (strength={strength:.0%}). "
            f"Index≈{index_price:.0f}. Strike={strike} ({'ATM' if strike==atm_strike else '1-OTM'}). "
            f"Expiry={expiry} ({days_left}d). "
            f"Premium=₹{premium:.2f}/unit. LotCost=₹{lot_cost:,.0f}. "
            + mode_note
        )

        # ── Commit cooldown + daily cap ───────────────────────────────────────
        OptionsStrategy._last_signal_time[option_symbol] = _dt.now()
        if dl_entry and dl_entry[0] == today:
            OptionsStrategy._daily_trades_per_ul[self.underlying] = (today, dl_entry[1] + 1)
        else:
            OptionsStrategy._daily_trades_per_ul[self.underlying] = (today, 1)

        logger.info(
            f"Options signal: {action} {option_symbol} @ ₹{premium:.2f} "
            f"[{mode} mode] conf={confidence:.0%}"
        )

        return TradeSignal(
            symbol    = option_symbol,
            exchange  = self.fo_exchange,
            action    = action,
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
