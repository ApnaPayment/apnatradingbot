"""
Phase 9 — Multi-Timeframe Signal Confirmation
Validates a 5-min signal against 15-min and 30-min timeframes before
allowing it to proceed to the AI evaluation stage.

Logic:
  - Resample existing 5-min OHLCV candles into coarser timeframes (no new API calls)
  - Run a lightweight trend check (EMA cross + RSI direction) on each TF
  - Return an alignment score, verdict, and per-TF detail for AI context

Alignment rules:
  BUY signal → aligned if higher TFs show: EMA9 > EMA21 and RSI > 50
  SELL signal → aligned if higher TFs show: EMA9 < EMA21 and RSI < 50
  "partial" if one TF agrees, the other doesn't
  "against" if both higher TFs oppose

Score: 1.0 = all three aligned, 0.5 = partial, 0.0 = against
Confidence adjustment: +0.10 fully aligned, −0.15 against, 0 partial
Hard veto: only when both higher TFs are against AND signal confidence < 0.65
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Resample a 5-min OHLCV DataFrame into `minutes`-min candles.
    Expects columns: open, high, low, close, volume; index is datetime.
    """
    rule = f"{minutes}min"
    resampled = df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return resampled


def _trend_check(df: pd.DataFrame, action: str) -> dict:
    """
    Run EMA9/21 cross + RSI on a resampled OHLCV frame.
    Returns alignment status for the given signal direction.
    """
    if df is None or len(df) < 30:
        return {"aligned": None, "reason": "insufficient data", "rsi": None,
                "ema_fast": None, "ema_slow": None}

    close     = df["close"]
    ema_fast  = _ema(close, 9).iloc[-1]
    ema_slow  = _ema(close, 21).iloc[-1]
    rsi_val   = _rsi(close, 14).iloc[-1]

    bullish_ema = ema_fast > ema_slow
    bullish_rsi = rsi_val > 50

    if action == "BUY":
        aligned = bullish_ema and bullish_rsi
        partial = bullish_ema or bullish_rsi     # at least one condition met
    else:  # SELL
        aligned = (not bullish_ema) and (not bullish_rsi)
        partial = (not bullish_ema) or (not bullish_rsi)

    if aligned:
        verdict = "aligned"
    elif partial:
        verdict = "partial"
    else:
        verdict = "against"

    return {
        "aligned": verdict,
        "rsi":      round(float(rsi_val), 1) if not np.isnan(rsi_val) else None,
        "ema_fast": round(float(ema_fast), 2),
        "ema_slow": round(float(ema_slow), 2),
        "reason":   (
            f"EMA {'↑' if bullish_ema else '↓'}  RSI {rsi_val:.0f}"
            if not np.isnan(rsi_val) else "RSI unavailable"
        ),
    }


class MTFResult:
    """Container for multi-timeframe analysis results."""
    __slots__ = (
        "symbol", "action", "tf5", "tf15", "tf30",
        "score", "verdict", "confidence_adjustment", "veto", "veto_reason",
    )

    def __init__(self, symbol, action, tf5, tf15, tf30):
        self.symbol = symbol
        self.action = action
        self.tf5    = tf5
        self.tf15   = tf15
        self.tf30   = tf30

        aligned_count = sum(
            1 for tf in [tf15, tf30]
            if tf.get("aligned") == "aligned"
        )
        against_count = sum(
            1 for tf in [tf15, tf30]
            if tf.get("aligned") == "against"
        )
        none_count = sum(
            1 for tf in [tf15, tf30]
            if tf.get("aligned") is None
        )

        # Score (higher = more aligned)
        self.score = aligned_count / max(2 - none_count, 1)

        if aligned_count == 2 or (aligned_count == 1 and none_count == 1):
            self.verdict = "aligned"
            self.confidence_adjustment = +0.10
        elif against_count == 2 or (against_count == 1 and none_count == 1):
            self.verdict = "against"
            self.confidence_adjustment = -0.15
        else:
            self.verdict = "partial"
            self.confidence_adjustment = 0.0

        # Hard veto only when both TFs are clearly against (not just data-missing)
        self.veto = (against_count == 2)
        self.veto_reason = (
            f"Both 15-min and 30-min oppose this {action}: "
            f"15m={tf15.get('reason','?')}  30m={tf30.get('reason','?')}"
            if self.veto else ""
        )

    def to_dict(self) -> dict:
        return {
            "symbol":               self.symbol,
            "action":               self.action,
            "verdict":              self.verdict,
            "score":                round(self.score, 2),
            "confidence_adjustment": self.confidence_adjustment,
            "veto":                 self.veto,
            "tf5":                  self.tf5,
            "tf15":                 self.tf15,
            "tf30":                 self.tf30,
        }

    def __repr__(self):
        return (
            f"MTF({self.symbol} {self.action}: {self.verdict}  "
            f"score={self.score:.2f}  adj={self.confidence_adjustment:+.2f})"
        )


class MTFAnalyzer:
    """
    Resamples 5-min OHLCV from DataManager into 15-min and 30-min frames
    and checks trend alignment for a given signal direction.
    """

    def __init__(self, data_manager):
        self.dm = data_manager

    def analyze(self, symbol: str, action: str,
                exchange: str = "nse_cm") -> Optional[MTFResult]:
        """
        Run multi-timeframe analysis for one signal.
        Returns MTFResult or None if no OHLCV data at all.
        """
        try:
            # Pull enough 5-min candles to resample into 30-min (need ≥ 60 bars)
            df5 = self.dm.get_ohlcv(symbol, exchange=exchange, limit=200)
            if df5 is None or len(df5) < 30:
                logger.debug(f"MTF: insufficient 5-min data for {symbol}")
                return None

            # Ensure DatetimeIndex for resampling
            if not isinstance(df5.index, pd.DatetimeIndex):
                df5 = df5.copy()
                df5.index = pd.to_datetime(df5.index)

            df15 = _resample_ohlcv(df5, 15)
            df30 = _resample_ohlcv(df5, 30)

            tf5  = _trend_check(df5,  action)
            tf15 = _trend_check(df15, action)
            tf30 = _trend_check(df30, action)

            result = MTFResult(symbol, action, tf5, tf15, tf30)
            logger.info(
                f"MTF {symbol}: {result.verdict}  "
                f"15m={tf15.get('aligned','?')}  30m={tf30.get('aligned','?')}  "
                f"adj={result.confidence_adjustment:+.2f}"
            )
            return result

        except Exception as e:
            logger.warning(f"MTF analysis failed for {symbol}: {e}")
            return None
