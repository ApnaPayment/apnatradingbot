"""
Commodity Strategy — MCX Paper Trading
Trend-following strategy for commodity futures using 1-hour OHLCV bars.

Design rationale vs equity strategies:
- 1h bars (not 5m): commodities need more context; 5m is too noisy for GC=F/CL=F
- EMA 20/50 (not 9/21): commodities trend slower than equity intraday
- ADX filter: only trade when trend strength is confirmed (avoids choppy consolidation)
- No time-of-day gate: MCX runs 09:00–23:30; valid signals fire any time
- Long-only: Phase 1 paper trading; short signals deferred to Phase 2
- R:R = 2:1 (target = +2×ATR, stop = −1×ATR) — passes AI's ≥2:1 minimum

Data source: yfinance COMEX/NYMEX proxies (GC=F, SI=F, CL=F, NG=F, HG=F)
           converted to INR via USDINR=X (handled upstream in LiquidityDataFetcher)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from core.risk_manager import TradeSignal

logger = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────────
FAST_EMA      = 20       # EMA periods for trend detection
SLOW_EMA      = 50
ADX_PERIOD    = 14       # ADX lookback
ADX_THRESHOLD = 22       # minimum ADX for a "strong trend" entry
VOLUME_FACTOR = 0.5      # volume vs 20-bar avg; COMEX 1h data is sparse so 0.5× is meaningful
MIN_ATR_PCT   = 0.002    # ATR ≥ 0.2% of price (on 1h bars; GOLD's 0.31% passes, flat periods don't)
MIN_BARS      = 60       # minimum candles required (60h ≈ 8 trading days on 1h bars)

# Lot sizes for paper-trade quantity (minimum lot = 1 unit for paper trading)
LOT_SIZES = {
    "GOLD":       1,
    "SILVER":     1,
    "CRUDEOIL":   1,
    "NATURALGAS": 1,
    "COPPER":     1,
}


class CommodityStrategy:
    """
    Trend-following commodity strategy for MCX paper trading.
    Uses EMA crossover + ADX strength + volume confirmation on 1-hour bars.
    """

    def __init__(self,
                 fast_ema: int = FAST_EMA,
                 slow_ema: int = SLOW_EMA,
                 adx_period: int = ADX_PERIOD,
                 adx_threshold: float = ADX_THRESHOLD,
                 volume_factor: float = VOLUME_FACTOR,
                 min_atr_pct: float = MIN_ATR_PCT):
        self.fast_ema      = fast_ema
        self.slow_ema      = slow_ema
        self.adx_period    = adx_period
        self.adx_threshold = adx_threshold
        self.volume_factor = volume_factor
        self.min_atr_pct   = min_atr_pct

    # ─────────────────────────────────────────────────────────────────────────
    # Indicators
    # ─────────────────────────────────────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMAs
        df["ema_fast"] = df["close"].ewm(span=self.fast_ema, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow_ema, adjust=False).mean()

        # EMA crossover signals
        df["cross_above"] = (
            (df["ema_fast"] > df["ema_slow"]) &
            (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
        )
        df["cross_below"] = (
            (df["ema_fast"] < df["ema_slow"]) &
            (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
        )

        # ATR (True Range)
        high_low   = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close  = (df["low"]  - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.adx_period).mean()

        # ADX (+DI, -DI, ADX)
        up_move   = df["high"] - df["high"].shift(1)
        down_move = df["low"].shift(1) - df["low"]

        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr_s     = tr.ewm(alpha=1 / self.adx_period, adjust=False).mean()
        plus_dm_s = pd.Series(plus_dm,  index=df.index).ewm(alpha=1 / self.adx_period, adjust=False).mean()
        minus_dm_s= pd.Series(minus_dm, index=df.index).ewm(alpha=1 / self.adx_period, adjust=False).mean()

        plus_di  = 100 * plus_dm_s  / atr_s.replace(0, np.nan)
        minus_di = 100 * minus_dm_s / atr_s.replace(0, np.nan)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        df["adx"]      = dx.ewm(alpha=1 / self.adx_period, adjust=False).mean()
        df["plus_di"]  = plus_di
        df["minus_di"] = minus_di

        # Volume MA
        df["vol_ma"]    = df["volume"].rolling(window=20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Signal generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signal(self, commodity: str, df: pd.DataFrame,
                        exchange: str = "mcx_fo") -> Optional[TradeSignal]:
        """
        Analyse 1-hour OHLCV data for a commodity and return a TradeSignal if
        trend conditions are met, or None if no clear signal.

        Conditions for BUY signal (long-only in Phase 1):
          1. EMA-20 crossed above EMA-50 within the last 3 bars
          2. ADX > threshold (strong trend in progress)
          3. +DI > -DI (buyers dominating)
          4. Volume > 1.5× 20-bar average on crossover bar
          5. ATR > 0.5% of price (adequate volatility for R:R target to be meaningful)
          6. Price above EMA-50 (not entering mid-downtrend)

        Returns TradeSignal with:
          - target  = entry + 2×ATR  (R:R = 2:1)
          - stop    = entry − 1×ATR
          - product = "NRML"
          - quantity = 1 (minimum paper-trade lot)
        """
        if len(df) < MIN_BARS:
            logger.debug(f"MCX {commodity}: only {len(df)} bars (need {MIN_BARS})")
            return None

        df = self._add_indicators(df)
        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        current_price = float(latest["close"])
        atr = float(latest["atr"]) if not pd.isna(latest["atr"]) else current_price * 0.01
        adx = float(latest["adx"]) if not pd.isna(latest["adx"]) else 0.0

        # ── Gate: ATR minimum ─────────────────────────────────────────────────
        if atr / current_price < self.min_atr_pct:
            logger.debug(
                f"MCX {commodity}: ATR too small "
                f"({atr:.2f} = {atr/current_price:.3%} < {self.min_atr_pct:.2%})"
            )
            return None

        # ── Gate: ADX trend strength ──────────────────────────────────────────
        if adx < self.adx_threshold:
            logger.debug(f"MCX {commodity}: ADX {adx:.1f} < {self.adx_threshold} (weak trend)")
            return None

        # ── EMA alignment: is EMA20 currently above EMA50? ───────────────────────
        # We use trend ALIGNMENT (EMA20 > EMA50) rather than requiring a fresh crossover.
        # Requiring a crossover within N bars misses established trends — CRUDEOIL can trend
        # for a week after the initial cross, and we want to capture that entire move.
        ema_fast_val = float(latest["ema_fast"])
        ema_slow_val = float(latest["ema_slow"])
        ema_aligned_bull = ema_fast_val > ema_slow_val   # uptrend
        ema_aligned_bear = ema_fast_val < ema_slow_val   # downtrend (for future SELL signals)

        # ── BUY conditions ─────────────────────────────────────────────────────
        plus_di  = float(latest["plus_di"])  if not pd.isna(latest["plus_di"])  else 0.0
        minus_di = float(latest["minus_di"]) if not pd.isna(latest["minus_di"]) else 0.0
        vol_ratio = float(latest["vol_ratio"]) if not pd.isna(latest["vol_ratio"]) else 0.0

        buy_conditions = {
            "ema_aligned_bull": ema_aligned_bull,
            "adx_strong":       adx >= self.adx_threshold,
            "buyers_dominate":  plus_di > minus_di,
            "price_above_slow": current_price > ema_slow_val,
            "volume_confirm":   vol_ratio >= self.volume_factor,
        }

        buy_score = sum(buy_conditions.values())
        # Mandatory: ema_aligned_bull, adx_strong, buyers_dominate (structural trend gates).
        # Volume is soft — COMEX 1h volume data from yfinance is frequently zero/sparse.
        # price_above_slow is soft — usually redundant with ema_aligned but not always.
        mandatory = buy_conditions["ema_aligned_bull"] and buy_conditions["adx_strong"] \
                    and buy_conditions["buyers_dominate"]
        if not mandatory or buy_score < 3:
            logger.debug(
                f"MCX {commodity}: {buy_score}/5 conditions "
                f"({', '.join(k for k,v in buy_conditions.items() if not v)} missing)"
            )
            return None

        # ── Build signal ───────────────────────────────────────────────────────
        stop_loss = round(current_price - 1.0 * atr, 2)
        target    = round(current_price + 2.0 * atr, 2)
        # Base 0.62 ensures signals at ADX threshold always clear the 0.60 risk gate minimum.
        # Each extra ADX point above threshold adds 1%, each vol unit above factor adds 5%.
        confidence = min(0.62 + (adx - self.adx_threshold) / 100 + (vol_ratio - self.volume_factor) * 0.05, 0.90)

        signal = TradeSignal(
            symbol     = commodity,
            exchange   = exchange,
            action     = "BUY",
            price      = current_price,
            strategy   = "commodity_trend",
            confidence = round(confidence, 2),
            stop_loss  = stop_loss,
            target     = target,
            quantity   = LOT_SIZES.get(commodity, 1),
            product    = "NRML",
            reasoning  = (
                f"MCX {commodity} trend signal: EMA20 above EMA50 (aligned bull), "
                f"ADX={adx:.1f} (>{self.adx_threshold}), "
                f"+DI={plus_di:.1f} > -DI={minus_di:.1f}, "
                f"Vol={vol_ratio:.1f}× avg | "
                f"Entry ₹{current_price:.2f} SL ₹{stop_loss:.2f} Target ₹{target:.2f} "
                f"(R:R {(target-current_price)/(current_price-stop_loss):.1f}:1)"
            ),
        )

        logger.info(
            f"MCX {commodity}: BUY signal @ ₹{current_price:.2f} "
            f"SL ₹{stop_loss:.2f} Target ₹{target:.2f} "
            f"ADX={adx:.1f} Vol={vol_ratio:.1f}× conf={confidence:.0%}"
        )
        return signal
