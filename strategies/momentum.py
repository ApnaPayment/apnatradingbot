"""
Momentum Strategy — V2
Generates buy signals using:
- Moving average crossover (EMA 9/21)
- RSI overbought/oversold
- Volume-confirmed breakouts (2× average, not 1.5×)

V2 changes (evidence-based from backtest diagnosis):
  1. Time-of-day gate: only trade 09:15–10:30 (morning session)
     Evidence: win rate 18% at 9h, drops to 0% by 14h
  2. ATR minimum filter: skip when ATR < 0.25% of price
     Evidence: breakeven requires 2.5×ATR → low-ATR periods unviable
  3. volume_factor raised from 1.5× to 2.0×
     Evidence: genuine breakouts need stronger volume confirmation
  4. above_slow_ema made mandatory (was optional)
     Evidence: buying below EMA-21 during downtrends always lost
"""

import logging
from datetime import time as dtime
from typing import Optional

import numpy as np
import pandas as pd

from core.risk_manager import TradeSignal

logger = logging.getLogger(__name__)

# ── V2 time-of-day gate ────────────────────────────────────────────────────────
# Only trade in the first 75 minutes. Evidence: win rate collapses after 10:30.
TRADE_START = dtime(9, 15)
TRADE_END   = dtime(10, 30)

# ── V2 minimum ATR filter ─────────────────────────────────────────────────────
# Skip if 5m ATR < 0.25% of price. Costs (₹52) need 2.5×ATR to break even;
# below this threshold the trade is structurally unviable regardless of signal.
MIN_ATR_PCT = 0.0025


class MomentumStrategy:
    """
    Classic momentum strategy for Indian equity markets.
    Works best on NSE large/mid cap stocks with good liquidity.
    """

    def __init__(self,
                 fast_ema: int = 9,
                 slow_ema: int = 21,
                 rsi_period: int = 14,
                 rsi_oversold: float = 40,
                 rsi_overbought: float = 65,
                 volume_factor: float = 2.0):   # V2: raised from 1.5 → 2.0
        self.fast_ema       = fast_ema
        self.slow_ema       = slow_ema
        self.rsi_period     = rsi_period
        self.rsi_oversold   = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_factor  = volume_factor

    # ─────────────────────────────────────────────────────────────────────────
    # Indicators
    # ─────────────────────────────────────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the OHLCV dataframe."""
        df = df.copy()

        # EMAs
        df["ema_fast"] = df["close"].ewm(span=self.fast_ema, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow_ema, adjust=False).mean()

        # RSI
        delta = df["close"].diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Volume MA
        df["vol_ma"] = df["volume"].rolling(window=20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma"]

        # ATR (for stop loss sizing)
        high_low   = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close  = (df["low"] - df["close"].shift()).abs()
        df["atr"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

        # Crossover detection
        df["cross_above"] = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift() <= df["ema_slow"].shift())
        df["cross_below"] = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift() >= df["ema_slow"].shift())

        # 52-week high breakout
        df["52w_high"] = df["close"].rolling(window=252, min_periods=50).max()
        df["breakout"]  = df["close"] >= df["52w_high"].shift()

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Signal generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signal(self, symbol: str, df: pd.DataFrame,
                        exchange: str = "nse_cm") -> Optional[TradeSignal]:
        """
        Analyze OHLCV data and return a TradeSignal if conditions are met.
        Returns None if no clear signal.

        V2 gates applied before any indicator computation:
          1. Time-of-day: only 09:15–10:30
          2. ATR minimum: skip low-volatility candles
          3. above_slow_ema mandatory (not optional)
          4. volume_factor = 2.0× (was 1.5×)
        """
        if len(df) < self.slow_ema + 10:
            logger.debug(f"{symbol}: Not enough data ({len(df)} candles)")
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
                pass  # if timestamp parsing fails, don't gate

        df = self._add_indicators(df)
        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        current_price = float(latest["close"])
        atr = float(latest["atr"]) if not pd.isna(latest["atr"]) else current_price * 0.02

        # ── V2 Gate 2: ATR minimum filter ─────────────────────────────────
        if atr / current_price < MIN_ATR_PCT:
            logger.debug(f"{symbol}: ATR too small ({atr:.3f} = {atr/current_price:.3%} < {MIN_ATR_PCT:.2%})")
            return None

        # ── BUY conditions ──────────────────────────────────────────────────
        # Note: rsi_recovering (RSI crosses oversold threshold on same bar as EMA crossover)
        # is intentionally kept as OPTIONAL — it's too coincidental to require.
        # above_slow_ema is MANDATORY — never buy in a downtrend.
        buy_conditions = {
            "ema_crossover":   bool(latest["cross_above"]),
            "rsi_not_hot":     bool(latest["rsi"] < self.rsi_overbought),
            "rsi_recovering":  bool(prev["rsi"] < self.rsi_oversold and latest["rsi"] > self.rsi_oversold),
            "volume_confirm":  bool(latest["vol_ratio"] > self.volume_factor),
            "above_slow_ema":  bool(current_price > latest["ema_slow"]),  # V2: mandatory
        }

        buy_score = sum(buy_conditions.values())
        signal    = None

        # V2: above_slow_ema mandatory + EMA crossover + at least 1 other (score ≥ 3)
        # (Same threshold as V1 but now above_slow_ema is required, not optional)
        if (buy_conditions["ema_crossover"]
                and buy_conditions["above_slow_ema"]
                and buy_score >= 3):
            confidence = min(0.5 + (buy_score * 0.1), 0.95)
            stop_loss  = round(current_price - (2 * atr), 2)
            target     = round(current_price + (3 * atr), 2)

            signal = TradeSignal(
                symbol=symbol, exchange=exchange,
                action="BUY", price=current_price,
                strategy="momentum",
                confidence=confidence,
                stop_loss=stop_loss,
                target=target,
                product="CNC",
                reasoning=self._build_reasoning("BUY", buy_conditions, latest, atr),
            )

        # SELL signals kept for live-bot exit logic (not traded in backtest)
        # No SELL short positions — Indian equity markets constraint
        else:
            sell_conditions = {
                "ema_crossover":  bool(latest["cross_below"]),
                "rsi_overbought": bool(prev["rsi"] > self.rsi_overbought and latest["rsi"] < self.rsi_overbought),
                "below_slow_ema": bool(current_price < latest["ema_slow"]),
            }
            sell_score = sum(sell_conditions.values())
            if sell_conditions["ema_crossover"] and sell_score >= 2:
                confidence = min(0.5 + (sell_score * 0.1), 0.90)
                signal = TradeSignal(
                    symbol=symbol, exchange=exchange,
                    action="SELL", price=current_price,
                    strategy="momentum",
                    confidence=confidence,
                    reasoning=self._build_reasoning("SELL", sell_conditions, latest, atr),
                )

        if signal:
            logger.info(f"Signal: {signal.action} {symbol} @ ₹{current_price:.2f} | conf={signal.confidence:.0%}")

        return signal

    # ─────────────────────────────────────────────────────────────────────────
    # Scan multiple symbols
    # ─────────────────────────────────────────────────────────────────────────

    def scan_watchlist(self, symbols: list, data_manager) -> list[TradeSignal]:
        """
        Scan a list of symbols and return all valid signals.
        Ranked by confidence score.
        """
        signals = []
        for symbol in symbols:
            try:
                df = data_manager.get_ohlcv(symbol, limit=300)
                if df.empty:
                    continue
                signal = self.generate_signal(symbol, df)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        # Sort by confidence descending
        signals.sort(key=lambda s: s.confidence, reverse=True)
        logger.info(f"Scan complete: {len(signals)} signals from {len(symbols)} symbols")
        return signals

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_reasoning(self, action: str, conditions: dict,
                         row: pd.Series, atr: float) -> str:
        triggered = [k for k, v in conditions.items() if v]
        rsi_val   = round(row["rsi"], 1) if not pd.isna(row["rsi"]) else "N/A"
        fast_ema  = round(row["ema_fast"], 2)
        slow_ema  = round(row["ema_slow"], 2)
        vol_ratio = round(row["vol_ratio"], 2) if not pd.isna(row["vol_ratio"]) else "N/A"

        return (
            f"{action} signal from momentum strategy. "
            f"EMA{self.fast_ema}={fast_ema} vs EMA{self.slow_ema}={slow_ema}. "
            f"RSI={rsi_val}. Volume ratio={vol_ratio}x avg. "
            f"ATR={round(atr, 2)}. "
            f"Conditions met: {', '.join(triggered)}."
        )

    def get_indicator_snapshot(self, symbol: str, df: pd.DataFrame) -> dict:
        """Get current indicator values for display/debugging."""
        if len(df) < self.slow_ema:
            return {}
        df = self._add_indicators(df)
        latest = df.iloc[-1]
        return {
            "symbol":    symbol,
            "close":     round(latest["close"], 2),
            "ema_fast":  round(latest["ema_fast"], 2),
            "ema_slow":  round(latest["ema_slow"], 2),
            "rsi":       round(latest["rsi"], 1) if not pd.isna(latest["rsi"]) else None,
            "vol_ratio": round(latest["vol_ratio"], 2) if not pd.isna(latest["vol_ratio"]) else None,
            "atr":       round(latest["atr"], 2) if not pd.isna(latest["atr"]) else None,
            "trend":     "BULLISH" if latest["ema_fast"] > latest["ema_slow"] else "BEARISH",
        }
