"""
Stock Options Strategy — Option B (intraday CE/PE instead of equity)

When an equity momentum signal fires for a configured stock, this converts it
into an ATM options trade:
  BUY equity signal  → BUY ATM Call (CE)
  SELL equity signal → BUY ATM Put  (PE)

Product: NRML (exits via 3:15 PM EOD squareoff or 50%/30% premium gates).
Expiry:  Nearest monthly expiry with >= 2 days to go (last Thursday of month).
"""

import logging
import calendar
from datetime import date, timedelta
from typing import Optional

from core.risk_manager import TradeSignal

logger = logging.getLogger(__name__)

# Stock option lot sizes and strike steps (NSE current as of 2026)
STOCK_OPTIONS_CONFIG = {
    "RELIANCE-EQ":  {"root": "RELIANCE",  "lot": 250, "step": 20},
    "HDFCBANK-EQ":  {"root": "HDFCBANK",  "lot": 550, "step": 5},
    "TCS-EQ":       {"root": "TCS",       "lot": 150, "step": 25},
    "INFY-EQ":      {"root": "INFY",      "lot": 300, "step": 20},
    "ICICIBANK-EQ": {"root": "ICICIBANK", "lot": 700, "step": 5},
}

MIN_PREMIUM     = 10    # skip deep-OTM lottery tickets
MAX_LOT_COST    = 15_000  # skip if one lot costs more than this
MIN_DAYS_EXPIRY = 2     # don't enter < 2 days before expiry

# Stock options expire last Thursday of each month on NSE
_EXPIRY_WEEKDAY = 3  # Thursday


class StockOptionsStrategy:
    """
    Converts an equity TradeSignal into an ATM stock option TradeSignal.
    Called from main.py when regime is trending and the stock is in config.
    """

    def convert(
        self,
        equity_signal: TradeSignal,
        data_manager,
    ) -> Optional[TradeSignal]:
        """
        Given an equity BUY/SELL signal, return an ATM CE/PE TradeSignal or None.

        Returns None if:
        - Stock not in config
        - No valid expiry found
        - Option not found in scrip master
        - Premium out of range
        """
        cfg = STOCK_OPTIONS_CONFIG.get(equity_signal.symbol)
        if not cfg:
            return None

        root        = cfg["root"]
        lot_size    = cfg["lot"]
        strike_step = cfg["step"]
        stock_price = equity_signal.price

        option_type = "CE" if equity_signal.action == "BUY" else "PE"

        # ATM strike: round to nearest step
        atm_strike = round(stock_price / strike_step) * strike_step

        # Nearest monthly expiry with enough time
        expiry = self._nearest_expiry()
        if not expiry:
            logger.debug(f"StockOpt: no valid expiry for {root}")
            return None

        # Find option symbol in scrip master
        option_symbol = self._find_symbol(data_manager, root, expiry, atm_strike, option_type)
        if not option_symbol:
            logger.debug(f"StockOpt: {root}{atm_strike}{option_type} not in scrip master")
            return None

        # Live premium
        quote = data_manager.get_live_quote(option_symbol, "nse_fo")
        if not quote:
            logger.debug(f"StockOpt: no live quote for {option_symbol}")
            return None

        premium = quote.get("ltp") or quote.get("last_price", 0)
        if premium < MIN_PREMIUM:
            logger.debug(f"StockOpt: {option_symbol} premium ₹{premium:.2f} too low")
            return None

        lot_cost = premium * lot_size
        if lot_cost > MAX_LOT_COST:
            logger.debug(f"StockOpt: {option_symbol} lot cost ₹{lot_cost:,.0f} > max")
            return None

        stop_loss = round(premium * 0.70, 2)   # exit at 30% loss
        target    = round(premium * 1.50, 2)   # exit at 50% gain

        logger.info(
            f"StockOpt: {option_symbol} @ ₹{premium:.2f} | lot={lot_size} | "
            f"SL=₹{stop_loss:.2f} | TP=₹{target:.2f}"
        )

        return TradeSignal(
            symbol     = option_symbol,
            exchange   = "nse_fo",
            action     = "BUY",
            price      = premium,
            strategy   = "stock_options",
            confidence = equity_signal.confidence,
            stop_loss  = stop_loss,
            target     = target,
            quantity   = lot_size,
            product    = "NRML",
            reasoning  = (
                f"Stock option: BUY {option_type} {root} {atm_strike} expiry {expiry}. "
                f"Triggered by {equity_signal.strategy} signal on {equity_signal.symbol}. "
                f"Premium=₹{premium:.2f}/unit. Lot cost=₹{lot_cost:,.0f}."
            ),
        )

    def _nearest_expiry(self) -> Optional[date]:
        """Last Thursday of the nearest month with >= MIN_DAYS_EXPIRY days left."""
        today = date.today()
        for month_offset in range(4):
            yr  = today.year + (today.month + month_offset - 1) // 12
            mon = (today.month + month_offset - 1) % 12 + 1
            last_day = calendar.monthrange(yr, mon)[1]
            d = date(yr, mon, last_day)
            while d.weekday() != _EXPIRY_WEEKDAY:
                d -= timedelta(days=1)
            if (d - today).days >= MIN_DAYS_EXPIRY:
                return d
        return None

    def _find_symbol(
        self,
        data_manager,
        root: str,
        expiry: date,
        strike: int,
        option_type: str,
    ) -> Optional[str]:
        """Search scrip master for the stock option symbol."""
        expiry_str = expiry.strftime("%y%b").upper()   # e.g. "26JUN"
        patterns = [
            f"{root}{expiry_str}{strike}{option_type}",
            f"{root}{expiry.strftime('%y%m%d')}{strike}{option_type}",
            f"{root}%{expiry_str}%{strike}%{option_type}",
            f"{root}%{strike}%{option_type}",
        ]
        try:
            with data_manager._get_conn() as conn:
                for pat in patterns:
                    row = conn.execute(
                        "SELECT symbol FROM scrip_master WHERE symbol LIKE ? "
                        "AND exchange='nse_fo' LIMIT 1",
                        (pat,),
                    ).fetchone()
                    if row:
                        return row[0]
        except Exception as e:
            logger.debug(f"StockOpt scrip lookup error: {e}")
        return None
