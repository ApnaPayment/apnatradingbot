"""
Signal Aggregator
Runs all registered strategies, resolves conflicts, and applies regime-aware
weighting to produce a single confidence-ranked list of TradeSignals.

Consensus rule:
  - Strategies agree on direction → confidence boosted (+5% per extra strategy)
  - Strategies conflict (BUY + SELL on same symbol) → signal dropped
  - Market regime shifts weights between momentum and mean-reversion
"""

import logging
from core.risk_manager import TradeSignal

logger = logging.getLogger(__name__)

# How much each additional agreeing strategy boosts confidence
CONSENSUS_BOOST_PER_STRATEGY = 0.05

# Regime → per-strategy weight overrides
_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "trending_up":    {"momentum": 0.75, "mean_reversion": 0.25},
    "trending_down":  {"momentum": 0.75, "mean_reversion": 0.25},
    "breakout":       {"momentum": 0.85, "mean_reversion": 0.15},
    "ranging":        {"momentum": 0.25, "mean_reversion": 0.75},
    "high_volatility":{"momentum": 0.40, "mean_reversion": 0.40},
}


class SignalAggregator:
    """
    Combines signals from multiple strategies using weighted confidence scoring.

    Usage:
        aggregator = SignalAggregator({
            "momentum":       (MomentumStrategy(),    0.6),
            "mean_reversion": (MeanReversionStrategy(), 0.4),
        })
        signals = aggregator.scan_and_aggregate(WATCHLIST, data_manager, regime="ranging")
    """

    def __init__(self, strategies: dict[str, tuple]):
        """
        Args:
            strategies: {name: (strategy_instance, base_weight)}
                        Weights don't need to sum to 1 — they're used relatively.
        """
        self.strategies = strategies  # name → (instance, base_weight)

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def scan_and_aggregate(
        self,
        symbols: list,
        data_manager,
        regime: str = "unknown",
    ) -> list[TradeSignal]:
        """
        Scan symbols with all strategies, merge results, return ranked signals.

        Args:
            symbols:      Watchlist of trading symbols.
            data_manager: DataManager instance for OHLCV fetching.
            regime:       Market regime string from AIDecisionEngine.detect_market_regime().
                          Controls strategy weight allocation.
        """
        weights = self._effective_weights(regime)

        # symbol → {"signals": [...], "weights": [...], "actions": set()}
        by_symbol: dict[str, dict] = {}

        for name, (strategy, _) in self.strategies.items():
            w = weights.get(name, 0.5)
            try:
                raw_signals = strategy.scan_watchlist(symbols, data_manager)
            except Exception as e:
                logger.error(f"Strategy '{name}' scan failed: {e}")
                continue

            for sig in raw_signals:
                # Guard: skip malformed signals — must be a TradeSignal with string action
                action = getattr(sig, "action", None)
                if not isinstance(action, str):
                    logger.warning(f"Skipping malformed signal from '{name}': type={type(sig).__name__} action={action!r}")
                    continue
                entry = by_symbol.setdefault(sig.symbol, {
                    "signals": [], "weights": [], "actions": set()
                })
                entry["signals"].append(sig)
                entry["weights"].append(w)
                entry["actions"].add(sig.action)

        # Merge each symbol's signals into one
        merged: list[TradeSignal] = []
        raw_count = sum(len(v["signals"]) for v in by_symbol.values())

        for symbol, data in by_symbol.items():
            result = self._merge(symbol, data, regime)
            if result:
                merged.append(result)

        merged.sort(key=lambda s: s.confidence, reverse=True)
        logger.info(
            f"Aggregator [{regime}]: {raw_count} raw → {len(merged)} merged"
            f" from {len(symbols)} symbols"
        )
        return merged

    # ─────────────────────────────────────────────────────────────────────────
    # Merging logic
    # ─────────────────────────────────────────────────────────────────────────

    def _merge(self, symbol: str, data: dict, regime: str) -> TradeSignal | None:
        signals = data["signals"]
        weights = data["weights"]
        actions = data["actions"]

        # Conflicting direction → drop
        if len(actions) > 1:
            logger.debug(f"{symbol}: conflicting signals ({actions}) — skipped")
            return None

        if len(signals) == 1:
            return signals[0]

        # Multiple strategies agree — weighted average + consensus boost
        total_w       = sum(weights)
        weighted_conf = sum(s.confidence * w for s, w in zip(signals, weights)) / total_w
        boost         = CONSENSUS_BOOST_PER_STRATEGY * (len(signals) - 1)
        final_conf    = min(weighted_conf + boost, 0.95)

        # Take first signal's price/SL/target (highest individual confidence)
        base = max(signals, key=lambda s: s.confidence)
        base.confidence = final_conf
        base.strategy   = " + ".join(s.strategy for s in signals)
        base.reasoning += (
            f" [Consensus: {len(signals)} strategies agree"
            f" ({', '.join(s.strategy for s in signals)}). Regime={regime}]"
        )
        return base

    # ─────────────────────────────────────────────────────────────────────────
    # Weight calculation
    # ─────────────────────────────────────────────────────────────────────────

    def _effective_weights(self, regime: str) -> dict[str, float]:
        """
        Merge base weights with regime overrides.
        Only override strategies that are actually registered.
        """
        base = {name: w for name, (_, w) in self.strategies.items()}
        overrides = _REGIME_WEIGHTS.get(regime, {})
        return {
            name: overrides.get(name, base_w)
            for name, base_w in base.items()
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Introspection helpers
    # ─────────────────────────────────────────────────────────────────────────

    def strategy_names(self) -> list[str]:
        return list(self.strategies.keys())

    def describe(self) -> str:
        lines = ["SignalAggregator strategies:"]
        for name, (strat, weight) in self.strategies.items():
            lines.append(f"  {name:20s}  base_weight={weight}")
        lines.append(f"  Registered regimes: {list(_REGIME_WEIGHTS.keys())}")
        return "\n".join(lines)
