"""
Phase 13 — Intraday Circuit Breaker & Drawdown Protection

Three escalating states triggered by realised + unrealised intraday loss:

  NORMAL   (loss < L1)  — full trading, no restriction
  CAUTION  (loss ≥ L1)  — new position sizes halved; warning sent
  HALT     (loss ≥ L2)  — no new entries; existing positions monitored
  CLOSE    (loss ≥ L3)  — all open positions force-closed, trading stopped

Default thresholds (% of max_capital, configurable via .env):
  L1 = CB_LEVEL1_PCT   default 1.0%  → CAUTION
  L2 = CB_LEVEL2_PCT   default 2.0%  → HALT
  L3 = CB_LEVEL3_PCT   default 2.5%  → CLOSE  (fires before 3% hard limit)

Loss = realised day P&L (negative) + unrealised P&L of open positions (negative).
Unrealised contribution is weighted at 50% so paper losses don't fire prematurely.

State is reset to NORMAL at the start of each trading day.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CBState(Enum):
    NORMAL  = "normal"
    CAUTION = "caution"
    HALT    = "halt"
    CLOSE   = "close"


@dataclass
class CircuitBreakerConfig:
    max_capital:   float = float(os.getenv("MAX_CAPITAL", 100_000))
    level1_pct:    float = float(os.getenv("CB_LEVEL1_PCT", 1.0))   # CAUTION
    level2_pct:    float = float(os.getenv("CB_LEVEL2_PCT", 2.0))   # HALT
    level3_pct:    float = float(os.getenv("CB_LEVEL3_PCT", 2.5))   # CLOSE
    unrealised_weight: float = 0.5   # unrealised loss counts at 50%

    @property
    def l1_amount(self) -> float:
        return self.max_capital * self.level1_pct / 100

    @property
    def l2_amount(self) -> float:
        return self.max_capital * self.level2_pct / 100

    @property
    def l3_amount(self) -> float:
        return self.max_capital * self.level3_pct / 100


@dataclass
class CBStatus:
    state:              CBState
    effective_loss:     float        # realised + weighted unrealised
    realised_loss:      float
    unrealised_loss:    float
    size_multiplier:    float        # 1.0 / 0.5 / 0.0 / 0.0
    allow_new_entries:  bool
    force_close:        bool
    triggered_at:       Optional[str]
    message:            str
    thresholds:         dict = field(default_factory=dict)


class CircuitBreaker:
    """
    Monitors intraday P&L and enforces graduated de-risking.

    Usage in main.py:
        self.cb = CircuitBreaker()
        ...
        # In run_trading_cycle():
        cb_status = self.cb.check(realised_pnl, unrealised_pnl, open_positions)
        if cb_status.force_close:
            self._close_all("circuit_breaker")
        if not cb_status.allow_new_entries:
            return
        # Pass cb_status.size_multiplier to calculate_quantity()
    """

    # SOP: pattern-based loss detection (independent of % loss threshold)
    _LOSS_WINDOW_MINUTES = 30
    _LOSS_WINDOW_HALT    = 5    # 5 losses in 30 min → force HALT

    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self._state          = CBState.NORMAL
        self._triggered_at: Optional[str] = None
        self._last_day: Optional[str] = None   # date string — for daily reset
        self._loss_timestamps: list = []        # recent losing trade timestamps

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def reset_for_new_day(self):
        """Call at start of each trading day (job_daily_setup)."""
        self._state           = CBState.NORMAL
        self._triggered_at    = None
        self._loss_timestamps = []
        logger.info("Circuit breaker reset for new day → NORMAL")

    def record_loss(self, ts: datetime = None) -> None:
        """
        Call after every losing trade closes.
        Maintains a 30-minute sliding window of loss events.
        If 5+ losses occur within 30 min, circuit breaker escalates to HALT.
        """
        ts = ts or datetime.now()
        self._loss_timestamps.append(ts)
        # Trim to sliding window
        cutoff = ts - timedelta(minutes=self._LOSS_WINDOW_MINUTES)
        self._loss_timestamps = [t for t in self._loss_timestamps if t >= cutoff]

        n = len(self._loss_timestamps)
        if n >= self._LOSS_WINDOW_HALT:
            prev = self._state
            state_order = [CBState.NORMAL, CBState.CAUTION, CBState.HALT, CBState.CLOSE]
            halt_idx = state_order.index(CBState.HALT)
            if state_order.index(self._state) < halt_idx:
                self._state        = CBState.HALT
                self._triggered_at = ts.isoformat()
                logger.warning(
                    f"Circuit breaker: {prev.value} → HALT "
                    f"(pattern: {n} losses in {self._LOSS_WINDOW_MINUTES} min)"
                )

    def check(self,
              realised_pnl: float,
              open_positions: dict) -> CBStatus:
        """
        Evaluate current intraday loss and return a CBStatus.

        Args:
            realised_pnl:    Today's booked P&L (negative = loss).
            open_positions:  Dict of {symbol: pos_dict} from RiskManager.
                             Each pos must have entry_price, quantity, and
                             optionally current_price (fetched by caller).

        The caller is responsible for fetching live prices and populating
        pos["current_price"] before calling check().
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_day != today:
            self.reset_for_new_day()
            self._last_day = today

        # Compute unrealised P&L across all open positions
        unrealised = 0.0
        for sym, pos in open_positions.items():
            ltp    = pos.get("current_price") or pos.get("entry_price", 0)
            entry  = pos.get("entry_price", ltp)
            qty    = pos.get("quantity", 0)
            action = pos.get("action", "BUY")
            if action == "BUY":
                unrealised += (ltp - entry) * qty
            else:
                unrealised += (entry - ltp) * qty

        # Effective loss = realised loss + 50% of unrealised loss
        # Only the negative component counts; gains don't reduce the loss counter
        realised_loss   = min(0.0, realised_pnl)
        unrealised_loss = min(0.0, unrealised)
        effective_loss  = abs(realised_loss) + abs(unrealised_loss) * self.config.unrealised_weight

        cfg = self.config
        prev_state = self._state

        # Determine new state (only escalate — never de-escalate within a day)
        if effective_loss >= cfg.l3_amount:
            new_state = CBState.CLOSE
        elif effective_loss >= cfg.l2_amount:
            new_state = CBState.HALT
        elif effective_loss >= cfg.l1_amount:
            new_state = CBState.CAUTION
        else:
            new_state = CBState.NORMAL

        # Only escalate, never downgrade within the same trading day
        state_order = [CBState.NORMAL, CBState.CAUTION, CBState.HALT, CBState.CLOSE]
        if state_order.index(new_state) > state_order.index(self._state):
            self._state        = new_state
            self._triggered_at = datetime.now().isoformat()
            if new_state != prev_state:
                logger.warning(
                    f"Circuit breaker: {prev_state.value} → {new_state.value} "
                    f"(effective_loss=₹{effective_loss:,.0f}, "
                    f"realised=₹{realised_loss:,.0f}, "
                    f"unrealised=₹{unrealised_loss:,.0f})"
                )

        state = self._state
        return CBStatus(
            state             = state,
            effective_loss    = round(effective_loss, 2),
            realised_loss     = round(realised_loss, 2),
            unrealised_loss   = round(unrealised_loss, 2),
            size_multiplier   = {
                CBState.NORMAL:  1.00,
                CBState.CAUTION: 0.50,
                CBState.HALT:    0.00,
                CBState.CLOSE:   0.00,
            }[state],
            allow_new_entries = state in (CBState.NORMAL, CBState.CAUTION),
            force_close       = state == CBState.CLOSE,
            triggered_at      = self._triggered_at,
            message           = _state_message(state, effective_loss, cfg),
            thresholds        = {
                "l1": cfg.l1_amount, "l2": cfg.l2_amount, "l3": cfg.l3_amount,
                "l1_pct": cfg.level1_pct, "l2_pct": cfg.level2_pct,
                "l3_pct": cfg.level3_pct,
                "max_capital": cfg.max_capital,
            },
        )

    @property
    def state(self) -> CBState:
        return self._state

    def to_dict(self) -> dict:
        """Compact representation for bot_status / dashboard."""
        return {
            "state":         self._state.value,
            "triggered_at":  self._triggered_at,
            "l1_pct":        self.config.level1_pct,
            "l2_pct":        self.config.level2_pct,
            "l3_pct":        self.config.level3_pct,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _state_message(state: CBState, loss: float, cfg: CircuitBreakerConfig) -> str:
    if state == CBState.NORMAL:
        return f"Normal — effective loss ₹{loss:,.0f} (below {cfg.level1_pct}% threshold)"
    if state == CBState.CAUTION:
        return (
            f"CAUTION — effective loss ₹{loss:,.0f} ≥ {cfg.level1_pct}% threshold. "
            f"New positions sized at 50%."
        )
    if state == CBState.HALT:
        return (
            f"HALT — effective loss ₹{loss:,.0f} ≥ {cfg.level2_pct}% threshold. "
            f"No new entries. Monitoring exits."
        )
    return (
        f"CLOSE — effective loss ₹{loss:,.0f} ≥ {cfg.level3_pct}% threshold. "
        f"Force-closing all positions."
    )
