"""
Claude Tool Use — Trading Tools
Defines Anthropic tool schemas and their Python executors.

Claude calls these during evaluate_signal() to self-verify before approving a trade.
Example: "Let me check the live spread before approving this BUY" → calls get_live_quote.

Usage (internal to decision_engine.py):
    executor = ToolExecutor(data_manager, risk_manager)
    tools    = executor.tool_schemas()
    result   = executor.run(tool_name, tool_input)
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas (Anthropic format)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "get_live_quote",
        "description": (
            "Get the CURRENT live price for a stock — LTP, bid, ask, volume, today's OHLC. "
            "Call this ONLY to verify that the signal entry price is still valid "
            "(i.e. price hasn't moved significantly since the signal was generated). "
            "Do NOT call this if you just need portfolio/P&L/sector data — that is already "
            "provided in the prompt context above."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE trading symbol e.g. 'RELIANCE-EQ', 'TCS-EQ', 'NIFTY2660923300CE'"
                },
                "exchange": {
                    "type": "string",
                    "description": "Exchange segment. Auto-detected: CE/PE symbols use 'nse_fo', equity uses 'nse_cm'. Only pass this if you need to override.",
                    "default": "nse_cm"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_recent_candles",
        "description": (
            "Get the last N 5-minute OHLCV candles for a symbol. "
            "Call this ONLY when you need to verify trend direction or support/resistance "
            "that is not already clear from the signal reasoning. "
            "Do NOT call this if the signal reasoning already describes the trend clearly. "
            "Do NOT call this for portfolio/P&L/sector data — that is in the prompt context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE trading symbol"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of recent candles to return (default 10, max 50)",
                    "default": 10
                }
            },
            "required": ["symbol"]
        }
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool executor — maps tool names → Python functions
# ─────────────────────────────────────────────────────────────────────────────

# Import the authoritative sector map from portfolio analytics
try:
    from data.portfolio_analytics import SECTOR_MAP as _SECTOR_MAP
except ImportError:
    _SECTOR_MAP = {}  # fallback if circular import


class ToolExecutor:
    """
    Executes Claude's tool calls using live data from DataManager and RiskManager.
    Keeps tool execution fully synchronous — no async needed.
    """

    def __init__(self, data_manager, risk_manager):
        self.data = data_manager
        self.risk = risk_manager

    def tool_schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def run(self, tool_name: str, tool_input: dict) -> Any:
        """
        Dispatch a tool call and return the result as a JSON-serialisable dict.
        Never raises — returns an error dict on failure so Claude can handle it.
        """
        try:
            if tool_name == "get_live_quote":
                return self._get_live_quote(**tool_input)
            elif tool_name == "get_recent_candles":
                return self._get_recent_candles(**tool_input)
            elif tool_name == "get_open_positions":
                return self._get_open_positions()
            elif tool_name == "get_today_pnl":
                return self._get_today_pnl()
            elif tool_name == "get_symbol_sector":
                return self._get_symbol_sector(**tool_input)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.warning(f"Tool '{tool_name}' execution error: {e}")
            return {"error": str(e)}

    # ── Tool implementations ──────────────────────────────────────────────────

    def _get_live_quote(self, symbol: str, exchange: str = None) -> dict:
        # Auto-detect exchange for options: CE/PE symbols are always on nse_fo
        if exchange is None:
            if symbol.endswith("CE") or symbol.endswith("PE"):
                exchange = "nse_fo"
            else:
                exchange = "nse_cm"
        quote = self.data.get_live_quote(symbol, exchange)
        if not quote:
            return {"error": f"No live quote found for {symbol}"}
        return {
            "symbol":     quote.get("symbol"),
            "ltp":        quote.get("ltp"),
            "open":       quote.get("open"),
            "high":       quote.get("high"),
            "low":        quote.get("low"),
            "close":      quote.get("close"),
            "volume":     quote.get("volume"),
            "bid":        quote.get("bid"),
            "ask":        quote.get("ask"),
            "change_pct": quote.get("change_pct"),
            "timestamp":  str(quote.get("timestamp", "")),
            "note": (
                f"Spread: ₹{(quote.get('ask', 0) or 0) - (quote.get('bid', 0) or 0):.2f}"
                if quote.get("ask") and quote.get("bid") else ""
            ),
        }

    def _get_recent_candles(self, symbol: str, limit: int = 10) -> dict:
        limit = min(max(limit, 1), 50)
        df = self.data.get_ohlcv(symbol, limit=limit)
        if df is None or df.empty:
            return {"error": f"No OHLCV data for {symbol}"}

        candles = []
        for _, row in df.tail(limit).iterrows():
            candles.append({
                "time":   str(row.get("timestamp", ""))[:16],
                "open":   round(float(row["open"]),  2),
                "high":   round(float(row["high"]),  2),
                "low":    round(float(row["low"]),   2),
                "close":  round(float(row["close"]), 2),
                "volume": int(row.get("volume", 0)),
            })

        last  = candles[-1] if candles else {}
        first = candles[0]  if candles else {}
        trend = "up" if last.get("close", 0) > first.get("close", 0) else "down"

        return {
            "symbol":       symbol,
            "candle_count": len(candles),
            "trend":        trend,
            "price_range":  f"₹{first.get('close', 0):.2f} → ₹{last.get('close', 0):.2f}",
            "candles":      candles,
        }

    def _get_open_positions(self) -> dict:
        portfolio = self.risk.get_portfolio_summary()
        positions = portfolio.get("open_positions", {})
        if not positions:
            return {"open_positions": [], "count": 0, "note": "No open positions"}

        result = []
        for symbol, pos in positions.items():
            entry = pos.get("entry_price", 0)
            sl    = pos.get("stop_loss", 0)
            tgt   = pos.get("target", 0)
            result.append({
                "symbol":        symbol,
                "action":        pos.get("action"),
                "strategy":      pos.get("strategy", ""),
                "entry_price":   entry,
                "stop_loss":     sl,
                "target":        tgt,
                "quantity":      pos.get("quantity", 0),
                "sector":        _SECTOR_MAP.get(symbol, "Unknown"),
                "sl_distance_pct": (
                    round(abs(entry - sl) / entry * 100, 2) if entry > 0 else None
                ),
            })

        sectors = [p["sector"] for p in result]
        sector_counts = {s: sectors.count(s) for s in set(sectors)}

        return {
            "open_positions":  result,
            "count":           len(result),
            "sectors_held":    sector_counts,
            "warning": (
                "High sector concentration: "
                + ", ".join(f"{s} ({n})" for s, n in sector_counts.items() if n >= 2)
            ) if any(n >= 2 for n in sector_counts.values()) else None,
        }

    def _get_today_pnl(self) -> dict:
        portfolio = self.risk.get_portfolio_summary()
        daily_pnl    = portfolio.get("daily_pnl", 0)
        daily_trades = portfolio.get("daily_trades", 0)
        max_loss     = self.risk.config.max_capital * (self.risk.config.max_daily_loss_pct / 100)
        remaining    = max_loss - abs(min(daily_pnl, 0))

        return {
            "daily_pnl":         round(daily_pnl, 2),
            "daily_trades":      daily_trades,
            "max_daily_loss":    round(max_loss, 2),
            "loss_budget_used":  f"{abs(min(daily_pnl, 0)) / max_loss * 100:.1f}%",
            "loss_budget_remaining": round(remaining, 2),
            "status": "good day" if daily_pnl >= 0 else (
                "caution" if remaining > max_loss * 0.3 else "near limit"
            ),
        }

    def _get_symbol_sector(self, symbol: str) -> dict:
        sector = _SECTOR_MAP.get(symbol.upper(), "Unknown")
        # Count how many open positions are in the same sector
        portfolio = self.risk.get_portfolio_summary()
        same_sector = [
            s for s in portfolio.get("open_positions", {})
            if _SECTOR_MAP.get(s, "?") == sector and s != symbol
        ]
        return {
            "symbol":          symbol,
            "sector":          sector,
            "same_sector_held": same_sector,
            "concentration_warning": len(same_sector) >= 2,
        }
