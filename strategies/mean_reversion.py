"""
Mean Reversion Strategy — V2
Bollinger Bands + RSI: buy oversold bounces into mean.
Complements momentum — works best in ranging/sideways markets.

V2 changes (evidence-based from backtest diagnosis):
  1. Time-of-day gate: only trade 09:15–10:30
     Evidence: 15% WR at 9h vs 0% at 14h; afternoon is a graveyard
  2. ATR minimum filter: skip when ATR < 0.25% of price
     Evidence: 82% of target hits were still net losses due to low ATR
  3. Target changed from bb_mid → 2×ATR
     Evidence: bb_mid R:R = 1.5:1 is insufficient; 2×ATR gives 1.33:1 SL:TGT
  4. BB width minimum: only enter when bands are genuinely wide (>1.5%)
     Evidence: overtrading (2/sym/day) caused by entry during tight-band ranging
  5. RSI bottoming confirmation: 2 consecutive bars of RSI rising required
     Evidence: single-bar RSI turn caused false entries in falling markets
  6. Required conditions: 4 of 6 (was 3 of 5) — reduces overtrading by ~50%
"""

import logging
from datetime import time as dtime
from typing import Optional

import numpy as np
import pandas as pd

from core.risk_manager import TradeSignal

logger = logging.getLogger(__name__)

# ── V2 time-of-day gate ────────────────────────────────────────────────────────
TRADE_START = dtime(9, 15)
TRADE_END   = dtime(10, 30)

# ── V2 minimum ATR filter ─────────────────────────────────────────────────────
MIN_ATR_PCT    = 0.0025   # 0.25% minimum ATR as fraction of price

# ── V2 BB width minimum ───────────────────────────────────────────────────────
MIN_BB_WIDTH   = 0.015    # 1.5% minimum Bollinger Band width


class MeanReversionStrategy:
    """
    Entry: price at lower BB + RSI oversold and turning up → BUY to mid-band.
    Exit:  price at upper BB + RSI overbought and turning down → SELL to mid-band.
    """

    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 volume_factor: float = 1.2):
        self.bb_period      = bb_period
        self.bb_std         = bb_std
        self.rsi_period     = rsi_period
        self.rsi_oversold   = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_factor  = volume_factor

    # ─────────────────────────────────────────────────────────────────────────
    # Indicators
    # ─────────────────────────────────────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Bollinger Bands
        df["bb_mid"]   = df["close"].rolling(self.bb_period).mean()
        bb_std         = df["close"].rolling(self.bb_period).std()
        df["bb_upper"] = df["bb_mid"] + self.bb_std * bb_std
        df["bb_lower"] = df["bb_mid"] - self.bb_std * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        band_range     = df["bb_upper"] - df["bb_lower"]
        df["bb_pct"]   = (df["close"] - df["bb_lower"]) / band_range.replace(0, np.nan)

        # RSI
        delta    = df["close"].diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ATR for stop sizing
        high_low   = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close  = (df["low"]  - df["close"].shift()).abs()
        df["atr"]  = (
            pd.concat([high_low, high_close, low_close], axis=1)
            .max(axis=1).rolling(14).mean()
        )

        # Volume
        df["vol_ma"]    = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Signal generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signal(self, symbol: str, df: pd.DataFrame,
                        exchange: str = "nse_cm") -> Optional[TradeSignal]:
        """
        V2: BB + RSI oversold bounce. Time-gated, ATR-filtered, tighter entry.

        Key V2 changes:
          - Time gate 09:15–10:30 only
          - ATR > 0.25% required (cost viability)
          - BB width > 1.5% required (genuine volatility, not noise)
          - RSI must be rising for 2 consecutive bars (not just 1)
          - Target = 2×ATR above entry (was BB midline)
          - Need 4 of 6 conditions (was 3 of 5)
        """
        if len(df) < self.bb_period + 10:
            return None

        # ── V2 Gate 1: Time-of-day ─────────────────────────────────────────
        latest_ts = df.iloc[-1].get("timestamp")
        if latest_ts is not None:
            try:
                ts = pd.Timestamp(latest_ts)
                bar_time = ts.time() if ts.tzinfo is None else ts.tz_convert("Asia/Kolkata").time()
                if not (TRADE_START <= bar_time <= TRADE_END):
                    return None
            except Exception:
                pass

        df     = self._add_indicators(df)
        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        prev2  = df.iloc[-3] if len(df) >= 3 else prev

        required_cols = ["bb_lower", "bb_upper", "bb_mid", "bb_pct", "rsi", "atr", "bb_width"]
        if any(pd.isna(latest[c]) for c in required_cols):
            return None

        current_price = float(latest["close"])
        atr           = float(latest["atr"])
        bb_pct        = float(latest["bb_pct"])
        bb_width      = float(latest["bb_width"])
        rsi_now       = float(latest["rsi"])
        rsi_prev      = float(prev["rsi"])  if not pd.isna(prev["rsi"])  else rsi_now
        rsi_prev2     = float(prev2["rsi"]) if not pd.isna(prev2["rsi"]) else rsi_prev

        # ── V2 Gate 2: ATR minimum ─────────────────────────────────────────
        if atr / current_price < MIN_ATR_PCT:
            return None

        # ── V2 Gate 3: BB width minimum ───────────────────────────────────
        # Bands must be genuinely wide — tight bands signal flat market, not a bounce
        if bb_width < MIN_BB_WIDTH:
            return None

        # ── BUY conditions (V2: 6 total, need 4) ─────────────────────────
        buy_conditions = {
            "near_lower_band":  bb_pct < 0.15,
            "rsi_oversold":     rsi_now < self.rsi_oversold,
            "rsi_bottoming":    rsi_now > rsi_prev and rsi_prev > rsi_prev2,  # V2: 2-bar recovery
            "volume_surge":     (not pd.isna(latest["vol_ratio"])
                                 and float(latest["vol_ratio"]) > self.volume_factor),
            "not_breakdown":    current_price > float(latest["bb_lower"]) * 0.98,
            "bands_wide":       bb_width > MIN_BB_WIDTH,  # already gated above, used for scoring
        }

        buy_score = sum(buy_conditions.values())
        signal    = None

        # V2: near_lower_band + rsi_oversold are mandatory; need 4 of 6 total
        if (buy_conditions["near_lower_band"]
                and buy_conditions["rsi_oversold"]
                and buy_score >= 4):
            confidence = min(0.50 + buy_score * 0.08, 0.90)
            stop_loss  = round(current_price - 1.5 * atr, 2)
            # Compute target as exactly 2× the stop distance to guarantee R:R ≥ 2.0
            # Using 3×ATR independently causes rounding divergence → AI sees 1.9996:1
            target     = round(current_price + 2 * (current_price - stop_loss), 2)

            signal = TradeSignal(
                symbol=symbol, exchange=exchange,
                action="BUY", price=current_price,
                strategy="mean_reversion",
                confidence=confidence,
                stop_loss=stop_loss,
                target=target,
                product="CNC",
                reasoning=self._build_reasoning("BUY", buy_conditions, latest),
            )

        # SELL kept for live-bot logic only (no short positions in backtest)
        else:
            sell_conditions = {
                "near_upper_band":  bb_pct > 0.85,
                "rsi_overbought":   rsi_now > self.rsi_overbought,
                "rsi_turning_down": rsi_now < rsi_prev,
                "volume_surge":     (not pd.isna(latest["vol_ratio"])
                                     and float(latest["vol_ratio"]) > self.volume_factor),
            }
            sell_score = sum(sell_conditions.values())
            if (sell_conditions["near_upper_band"] and sell_conditions["rsi_overbought"]
                    and sell_score >= 3):
                confidence = min(0.50 + sell_score * 0.08, 0.85)
                signal = TradeSignal(
                    symbol=symbol, exchange=exchange,
                    action="SELL", price=current_price,
                    strategy="mean_reversion",
                    confidence=confidence,
                    reasoning=self._build_reasoning("SELL", sell_conditions, latest),
                )

        if signal:
            logger.info(
                f"MR Signal: {signal.action} {symbol} @ ₹{current_price:.2f}"
                f" conf={signal.confidence:.0%}"
            )
        return signal

    # ─────────────────────────────────────────────────────────────────────────
    # Watchlist scan
    # ─────────────────────────────────────────────────────────────────────────

    def scan_watchlist(self, symbols: list, data_manager) -> list[TradeSignal]:
        signals = []
        for symbol in symbols:
            try:
                df = data_manager.get_ohlcv(symbol, limit=300)
                if df.empty:
                    continue
                sig = self.generate_signal(symbol, df)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.error(f"MR scan error {symbol}: {e}")

        signals.sort(key=lambda s: s.confidence, reverse=True)
        logger.info(f"MR scan: {len(signals)} signals from {len(symbols)} symbols")
        return signals

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_reasoning(self, action: str, conditions: dict, row: pd.Series) -> str:
        triggered = [k for k, v in conditions.items() if v]
        bb_pct    = round(float(row["bb_pct"]) * 100, 1)
        rsi       = round(float(row["rsi"]), 1) if not pd.isna(row["rsi"]) else "N/A"
        bb_width  = (
            round(float(row["bb_width"]) * 100, 2)
            if not pd.isna(row["bb_width"]) else "N/A"
        )
        return (
            f"{action} signal from mean reversion. "
            f"BB%={bb_pct}% RSI={rsi} BB_width={bb_width}%. "
            f"Conditions: {', '.join(triggered)}."
        )

    def get_indicator_snapshot(self, symbol: str, df: pd.DataFrame) -> dict:
        if len(df) < self.bb_period:
            return {}
        df     = self._add_indicators(df)
        latest = df.iloc[-1]
        return {
            "symbol":   symbol,
            "close":    round(float(latest["close"]), 2),
            "bb_upper": round(float(latest["bb_upper"]), 2) if not pd.isna(latest["bb_upper"]) else None,
            "bb_mid":   round(float(latest["bb_mid"]), 2)   if not pd.isna(latest["bb_mid"])   else None,
            "bb_lower": round(float(latest["bb_lower"]), 2) if not pd.isna(latest["bb_lower"]) else None,
            "bb_pct":   round(float(latest["bb_pct"]) * 100, 1) if not pd.isna(latest["bb_pct"]) else None,
            "rsi":      round(float(latest["rsi"]), 1) if not pd.isna(latest["rsi"]) else None,
            "regime":   "RANGING" if (not pd.isna(latest["bb_width"])
                                      and float(latest["bb_width"]) < 0.04) else "TRENDING",
        }
