"""
Risk Manager
Enforces position sizing, stop loss rules, and capital exposure limits.
All trade decisions must pass through here before execution.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    max_capital: float           = float(os.getenv("MAX_CAPITAL", 100000))
    max_position_size_pct: float = float(os.getenv("MAX_POSITION_SIZE_PCT", 10))
    max_daily_loss_pct: float    = float(os.getenv("MAX_DAILY_LOSS_PCT", 3))
    max_open_positions: int      = int(os.getenv("MAX_OPEN_POSITIONS", 5))
    default_stop_loss_pct: float = 2.0
    default_target_pct: float    = 4.0
    risk_reward_min: float       = 2.0
    trailing_stop_enabled: bool  = True
    trailing_stop_pct: float     = 1.5   # trail 1.5% below current price for longs


@dataclass
class TradeSignal:
    symbol: str
    exchange: str
    action: str             # "BUY" or "SELL"
    price: float
    strategy: str
    confidence: float       # 0.0 to 1.0
    stop_loss: float = 0.0
    target: float = 0.0
    quantity: int = 0
    product: str = "CNC"
    reasoning: str = ""
    regime: str = "unknown"


class RiskManager:
    """
    Central risk gate. Every trade signal must be approved here.
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.open_positions: dict = {}       # symbol -> position dict
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.today: date = date.today()

    # ─────────────────────────────────────────────────────────────────────────
    # Core approval gate
    # ─────────────────────────────────────────────────────────────────────────

    def approve_trade(self, signal: TradeSignal, available_capital: float,
                      sector_map: dict = None) -> tuple[bool, str]:
        """
        Returns (approved: bool, reason: str).
        All checks must pass for the trade to be approved.

        Args:
            sector_map: Optional {symbol: sector_str} for the current portfolio.
                        When provided, enables sector concentration checks.
        """
        self._reset_daily_if_new_day()

        checks = [
            self._check_daily_loss_limit(),
            self._check_max_positions(signal),
            self._check_duplicate_position(signal),
            self._check_confidence(signal),
            self._check_risk_reward(signal),
            self._check_capital(signal, available_capital),
        ]

        if sector_map:
            checks.append(self._check_sector_concentration(signal, sector_map))

        for approved, reason in checks:
            if not approved:
                logger.warning(f"Trade REJECTED [{signal.symbol}]: {reason}")
                return False, reason

        logger.info(f"Trade APPROVED [{signal.symbol}] {signal.action} qty={signal.quantity}")
        return True, "All checks passed"

    # ─────────────────────────────────────────────────────────────────────────
    # Individual checks
    # ─────────────────────────────────────────────────────────────────────────

    def _check_daily_loss_limit(self) -> tuple[bool, str]:
        max_loss = self.config.max_capital * (self.config.max_daily_loss_pct / 100)
        if self.daily_pnl <= -max_loss:
            return False, f"Daily loss limit hit: ₹{self.daily_pnl:.0f} / -₹{max_loss:.0f}"
        return True, ""

    def _check_max_positions(self, signal: TradeSignal) -> tuple[bool, str]:
        if signal.action == "BUY" and len(self.open_positions) >= self.config.max_open_positions:
            return False, f"Max open positions reached ({self.config.max_open_positions})"
        return True, ""

    def _check_duplicate_position(self, signal: TradeSignal) -> tuple[bool, str]:
        if signal.action == "BUY" and signal.symbol in self.open_positions:
            return False, f"Already holding {signal.symbol}"
        return True, ""

    def _check_confidence(self, signal: TradeSignal) -> tuple[bool, str]:
        if signal.confidence < 0.6:
            return False, f"Confidence too low: {signal.confidence:.0%} (min 60%)"
        return True, ""

    def _check_risk_reward(self, signal: TradeSignal) -> tuple[bool, str]:
        if signal.stop_loss <= 0 or signal.target <= 0:
            return True, ""  # Skip if not set
        risk   = abs(signal.price - signal.stop_loss)
        reward = abs(signal.target - signal.price)
        if risk == 0:
            return False, "Stop loss equals entry price"
        rr = round(reward / risk, 2)  # round to avoid float precision issues
        # Options (NRML product): SL=30%, Target=50% gives fixed R:R=1.67 — always pass
        if getattr(signal, "product", "CNC") == "NRML":
            return True, ""
        # Use lower R:R threshold for ranging/mean-reversion regime
        regime = getattr(signal, "regime", "unknown") or "unknown"
        if regime.lower() in ("ranging", "sideways", "mean_reversion"):
            rr_min = 1.5
        else:
            rr_min = self.config.risk_reward_min  # 2.0 for trending
        if rr < rr_min:
            return False, f"Risk:reward {rr:.1f} below minimum {rr_min} ({regime})"
        return True, ""

    def _check_sector_concentration(self, signal: TradeSignal,
                                     sector_map: dict) -> tuple[bool, str]:
        """
        Block a new BUY if it would put 3 or more positions in the same sector.
        sector_map: {symbol: "Banking"} for currently held symbols.
        """
        if signal.action != "BUY":
            return True, ""

        new_symbol = signal.symbol
        new_base   = new_symbol.split("-")[0]

        # Determine the sector of the incoming signal
        from data.portfolio_analytics import SECTOR_MAP
        new_sector = SECTOR_MAP.get(new_base, "Other")

        if new_sector in ("ETF", "Other"):
            return True, ""   # ETFs and unknowns always allowed

        # Count how many open positions are in the same sector
        same_sector = [
            sym for sym, sec in sector_map.items()
            if sec == new_sector
        ]
        if len(same_sector) >= 2:
            return False, (
                f"Sector concentration: already {len(same_sector)} positions in "
                f"{new_sector} ({', '.join(same_sector)}). Max 2 per sector."
            )
        return True, ""

    def _check_capital(self, signal: TradeSignal, available_capital: float) -> tuple[bool, str]:
        max_position = min(
            available_capital,
            self.config.max_capital * (self.config.max_position_size_pct / 100)
        )
        trade_value = signal.price * signal.quantity
        if trade_value > max_position:
            return False, f"Trade value ₹{trade_value:.0f} > max position ₹{max_position:.0f}"
        return True, ""

    # ─────────────────────────────────────────────────────────────────────────
    # Position sizing
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_quantity(self, price: float, available_capital: float,
                           confidence: float = 1.0,
                           vix: float = None,
                           kelly_multiplier: float = 1.0) -> int:
        """
        Position sizing: Kelly multiplier × VIX scaling × confidence floor.

        Sizing layers (applied in order, each reduces the previous):
          1. Hard cap:        min(available × 0.95, max_position_size_pct)
          2. Kelly:           × kelly_multiplier  (from kelly_sizer, default 1.0)
          3. Confidence:      × max(confidence, 0.6)
          4. VIX band:        × 0.25 / 0.50 / 0.75 based on VIX level
        """
        max_position_value = min(
            available_capital * 0.95,
            self.config.max_capital * (self.config.max_position_size_pct / 100)
        )

        # Phase 11: Kelly multiplier — derived from historical win/loss edge
        scaled_value = max_position_value * kelly_multiplier

        # Confidence floor: always size at least 60% of Kelly-adjusted value
        scaled_value *= max(confidence, 0.6)

        # VIX-based size reduction
        if vix is not None:
            if vix > 25:
                scaled_value *= 0.25
                logger.warning(f"VIX={vix:.1f} (extreme) — position size reduced to 25%")
            elif vix > 20:
                scaled_value *= 0.50
                logger.info(f"VIX={vix:.1f} (high) — position size reduced to 50%")
            elif vix > 15:
                scaled_value *= 0.75
                logger.info(f"VIX={vix:.1f} (elevated) — position size reduced to 75%")

        quantity = int(scaled_value / price)
        return max(1, quantity)

    def check_vix_block(self, vix: float, signal_strategy: str) -> tuple[bool, str]:
        """
        Hard block rules based on India VIX:
        - VIX > 25: block ALL new entries
        - VIX > 20: block mean-reversion entries (they need a stable range to work)

        Returns (blocked: bool, reason: str).
        """
        if vix is None:
            return False, ""
        if vix > 25:
            return True, f"India VIX={vix:.1f} — extreme volatility, no new entries"
        if vix > 20 and "mean_reversion" in signal_strategy.lower():
            return True, f"India VIX={vix:.1f} — mean reversion disabled in high-volatility regime"
        return False, ""

    def calculate_stop_loss(self, entry_price: float, action: str,
                            pct: float = None) -> float:
        """Calculate stop loss price."""
        pct = pct or self.config.default_stop_loss_pct
        if action == "BUY":
            return round(entry_price * (1 - pct / 100), 2)
        else:
            return round(entry_price * (1 + pct / 100), 2)

    def calculate_target(self, entry_price: float, action: str,
                         pct: float = None) -> float:
        """Calculate target price."""
        pct = pct or self.config.default_target_pct
        if action == "BUY":
            return round(entry_price * (1 + pct / 100), 2)
        else:
            return round(entry_price * (1 - pct / 100), 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Position tracking
    # ─────────────────────────────────────────────────────────────────────────

    def record_entry(self, signal: TradeSignal, order_no: str):
        """Record a new position after successful order placement."""
        self.open_positions[signal.symbol] = {
            "order_no":   order_no,
            "symbol":     signal.symbol,
            "exchange":   signal.exchange,
            "action":     signal.action,
            "entry_price": signal.price,
            "quantity":   signal.quantity,
            "stop_loss":  signal.stop_loss,
            "target":     signal.target,
            "strategy":   signal.strategy,
            "product":    signal.product,
            "entry_time": str(__import__("datetime").datetime.now()),
        }
        self.daily_trades += 1
        logger.info(f"Position recorded: {signal.symbol} @ ₹{signal.price}")

    def record_exit(self, symbol: str, exit_price: float):
        """Record position close and update daily P&L."""
        if symbol not in self.open_positions:
            logger.warning(f"record_exit: {symbol} not in open positions (already closed?)")
            return 0.0
        pos = self.open_positions.pop(symbol)
        qty = pos["quantity"]
        pnl = (exit_price - pos["entry_price"]) * qty
        if pos["action"] == "SELL":
            pnl = -pnl
        self.daily_pnl += pnl
        logger.info(f"Position closed: {symbol} P&L=₹{pnl:.0f} | Daily P&L=₹{self.daily_pnl:.0f}")
        return pnl

    def check_stop_loss_hit(self, symbol: str, current_price: float) -> bool:
        """Returns True if stop loss has been hit for a position."""
        if symbol not in self.open_positions:
            return False
        pos = self.open_positions[symbol]
        if pos["action"] == "BUY":
            return current_price <= pos["stop_loss"]
        else:
            return current_price >= pos["stop_loss"]

    def check_target_hit(self, symbol: str, current_price: float) -> bool:
        """Returns True if target has been hit for a position."""
        if symbol not in self.open_positions:
            return False
        pos = self.open_positions[symbol]
        if pos["action"] == "BUY":
            return current_price >= pos["target"]
        else:
            return current_price <= pos["target"]

    def get_portfolio_summary(self) -> dict:
        """Current state of all positions and daily P&L."""
        return {
            "open_positions": self.open_positions,
            "position_count": len(self.open_positions),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
            "capital_at_risk": sum(
                p["entry_price"] * p["quantity"]
                for p in self.open_positions.values()
            ),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Trailing stop
    # ─────────────────────────────────────────────────────────────────────────

    def update_trailing_stop(self, symbol: str, current_price: float,
                              trail_pct: float = None) -> bool:
        """
        Ratchet the stop loss up (longs) or down (shorts) to trail current price.
        The stop only ever moves in the profitable direction — never backwards.
        Returns True if the stop was moved.
        """
        if symbol not in self.open_positions:
            return False
        if not self.config.trailing_stop_enabled:
            return False

        pct = trail_pct if trail_pct is not None else self.config.trailing_stop_pct
        pos = self.open_positions[symbol]

        if pos["action"] == "BUY":
            new_stop = round(current_price * (1 - pct / 100), 2)
            if new_stop > pos["stop_loss"]:
                logger.info(
                    f"Trailing stop: {symbol} SL {pos['stop_loss']} → {new_stop}"
                    f" (price ₹{current_price})"
                )
                pos["stop_loss"] = new_stop
                return True

        else:  # SELL / short
            new_stop = round(current_price * (1 + pct / 100), 2)
            if new_stop < pos["stop_loss"]:
                logger.info(
                    f"Trailing stop: {symbol} SL {pos['stop_loss']} → {new_stop}"
                    f" (price ₹{current_price})"
                )
                pos["stop_loss"] = new_stop
                return True

        return False

    def get_trailing_stop(self, symbol: str) -> Optional[float]:
        """Return current stop loss for a position."""
        pos = self.open_positions.get(symbol)
        return pos["stop_loss"] if pos else None

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_daily_if_new_day(self):
        today = date.today()
        if today != self.today:
            logger.info(f"New trading day. Resetting daily counters. Yesterday P&L: ₹{self.daily_pnl:.0f}")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.today = today
