"""
Kotak Neo Trade API Client
Full SDK: authentication, session management, orders, reports, market data.
"""

import os
import time
import logging
import pyotp
import requests
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
logger = logging.getLogger(__name__)


class KotakNeoClient:
    """
    Complete Kotak Neo Trade API client.
    Handles 2-step auth, auto token refresh, and all trading endpoints.
    """

    LOGIN_BASE = "https://mis.kotaksecurities.com"
    NEO_FIN_KEY = "neotradeapi"
    SESSION_VALIDITY_HOURS = 8

    def __init__(self):
        self.consumer_key   = os.getenv("KOTAK_CONSUMER_KEY")
        self.mobile_number  = os.getenv("KOTAK_MOBILE_NUMBER")
        self.ucc            = os.getenv("KOTAK_UCC")
        self.totp_secret    = os.getenv("KOTAK_TOTP_SECRET")
        self.mpin           = os.getenv("KOTAK_MPIN")
        self.paper_trading  = os.getenv("PAPER_TRADING", "true").lower() == "true"

        # Session state
        self.view_token:     Optional[str] = None
        self.view_sid:       Optional[str] = None
        self.session_token:  Optional[str] = None
        self.session_sid:    Optional[str] = None
        self.base_url:       Optional[str] = None
        self.session_time:   Optional[datetime] = None

        self._validate_config()

    # ─────────────────────────────────────────────────────────────────────────
    # Config validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_config(self):
        required = {
            "KOTAK_CONSUMER_KEY": self.consumer_key,
            "KOTAK_MOBILE_NUMBER": self.mobile_number,
            "KOTAK_UCC": self.ucc,
            "KOTAK_TOTP_SECRET": self.totp_secret,
            "KOTAK_MPIN": self.mpin,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing env vars: {', '.join(missing)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_totp(self) -> str:
        totp = pyotp.TOTP(self.totp_secret)
        return totp.now()

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except requests.exceptions.ConnectionError:
            logger.error("Connection error — check your network/static IP")
            raise
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            raise

    def _get(self, url: str, headers: dict, params: dict = None) -> dict:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except requests.exceptions.ConnectionError:
            logger.error("Connection error — check your network/static IP")
            raise
        except requests.exceptions.Timeout:
            logger.error("Request timed out (15s)")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _get: {e}")
            raise

    def _post_form(self, url: str, headers: dict, jdata: dict) -> dict:
        """POST with application/x-www-form-urlencoded + jData payload."""
        import json
        try:
            response = requests.post(
                url,
                headers={**headers, "Content-Type": "application/x-www-form-urlencoded"},
                data={"jData": json.dumps(jdata)},
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise

    def _trading_headers(self) -> dict:
        """Standard headers required for all post-login API calls."""
        self._ensure_session()
        return {
            "Sid":  self.session_sid,
            "Auth": self.session_token,
        }

    def _ensure_session(self):
        """Auto re-authenticate if session has expired."""
        if not self.session_token:
            logger.info("No session found. Authenticating...")
            if not self.authenticate():
                raise RuntimeError("Failed to establish session")
            return

        if self.session_time:
            age = datetime.now() - self.session_time
            if age > timedelta(hours=self.SESSION_VALIDITY_HOURS):
                logger.info("Session expired. Re-authenticating...")
                if not self.authenticate():
                    raise RuntimeError("Session refresh failed")

        # Periodic health check: only every 30 minutes, not on every call
        if self.session_time:
            age_minutes = (datetime.now() - self.session_time).total_seconds() / 60
            if age_minutes > 30 and not self.is_session_healthy():
                logger.warning("Session health check failed. Re-authenticating...")
                if not self.authenticate():
                    raise RuntimeError("Session health check failed")

    # ─────────────────────────────────────────────────────────────────────────
    # Authentication
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize_mobile(self, number: str) -> str:
        """Normalize mobile number to 10-digit format (Kotak expects no country code)."""
        n = number.strip()
        if n.startswith("+91"):
            n = n[3:]
        elif n.startswith("91") and len(n) == 12:
            n = n[2:]
        return n

    def step1_totp_login(self) -> dict:
        """Step 1: Login with TOTP → get view token + view SID."""
        totp = self._generate_totp()
        logger.debug("TOTP generated (value hidden for security)")

        # Kotak expects +91XXXXXXXXXX format
        mobile = self.mobile_number  # Keep as-is from .env (+919509807591)
        logger.info(f"Using mobile: {mobile}, UCC: {self.ucc}")

        url = f"{self.LOGIN_BASE}/login/1.0/tradeApiLogin"
        headers = {
            "Authorization": self.consumer_key,
            "neo-fin-key": self.NEO_FIN_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "mobileNumber": mobile,
            "ucc": self.ucc,
            "totp": totp,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            logger.info(f"Step1 status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Step1 response: {response.text}")
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Step1 HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Step1 error: {e}")
            raise

        if data.get("data", {}).get("status") == "success":
            self.view_token = data["data"]["token"]
            self.view_sid   = data["data"]["sid"]
            logger.info("Step 1 (TOTP) successful")
            return data
        else:
            raise Exception(f"TOTP login failed: {data}")

    def step2_mpin_validate(self) -> dict:
        """Step 2: Validate MPIN → get full trading session."""
        if not self.view_token or not self.view_sid:
            raise Exception("Must complete step1_totp_login first")

        url = f"{self.LOGIN_BASE}/login/1.0/tradeApiValidate"
        headers = {
            "Authorization": self.consumer_key,
            "neo-fin-key": self.NEO_FIN_KEY,
            "sid":  self.view_sid,
            "Auth": self.view_token,
            "Content-Type": "application/json",
        }
        payload = {"mpin": self.mpin}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            logger.info(f"Step2 status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Step2 response: {response.text}")
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Step2 HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Step2 error: {e}")
            raise

        if data.get("data", {}).get("status") == "success":
            self.session_token = data["data"]["token"]
            self.session_sid   = data["data"]["sid"]
            self.base_url      = data["data"]["baseUrl"]
            self.session_time  = datetime.now()
            logger.info(f"Step 2 (MPIN) successful. Base URL: {self.base_url}")
            return data
        else:
            raise Exception(f"MPIN validation failed: {data}")

    def authenticate(self) -> bool:
        """Full 2-step authentication with TOTP refresh retry."""
        logger.info("Starting authentication...")
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                self.step1_totp_login()
                time.sleep(0.5)
                self.step2_mpin_validate()
                logger.info("Authentication complete. Ready to trade.")
                return True
            except Exception as e:
                logger.warning(f"Auth attempt {attempt+1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    logger.info("Refreshing TOTP and retrying...")
                    time.sleep(2)  # Wait before retry
                else:
                    logger.error(f"Authentication failed after {max_attempts} attempts: {e}")
                    return False
        return False

    def is_authenticated(self) -> bool:
        return bool(self.session_token and self.session_sid and self.base_url)

    def is_session_healthy(self) -> bool:
        """Verify session is valid by making a direct API call (no _ensure_session to avoid recursion)."""
        if not self.is_authenticated():
            return False

        try:
            # Call limits endpoint directly — bypass _trading_headers() which calls _ensure_session()
            url = f"{self.base_url}/1.0/user/limits"
            headers = {
                "Sid":  self.session_sid,
                "Auth": self.session_token,
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code in (200, 201):
                return True
            elif response.status_code in (401, 403):
                logger.warning(f"Session health check: unauthorized ({response.status_code})")
                return False
            else:
                # Non-auth errors (500, timeout) — assume session is still OK
                return True
        except requests.exceptions.Timeout:
            logger.warning(f"Session health check timed out — assuming session OK")
            return True  # Timeout ≠ invalid session
        except Exception as e:
            logger.warning(f"Session health check error: {e}")
            return False

    def _post_with_retry(self, url: str, headers: dict, payload: dict, max_retries: int = 3) -> dict:
        """POST with exponential backoff retry on transient failures."""
        import random

        for attempt in range(max_retries):
            try:
                return self._post(url, headers, payload)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = min(30, 2 ** attempt + random.uniform(0, 1))
                logger.warning(f"Retry {attempt+1}/{max_retries} after {wait_time:.1f}s: {e}")
                time.sleep(wait_time)
        raise RuntimeError("Max retries exceeded")

    # ─────────────────────────────────────────────────────────────────────────
    # Order APIs
    # ─────────────────────────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        price: float,
        order_type: str = "L",
        product: str = "CNC",
        trigger_price: float = 0,
        validity: str = "DAY",
        disclosed_qty: int = 0,
        after_market: bool = False,
    ) -> dict:
        """
        Place a new order.

        Args:
            symbol:           Trading symbol e.g. "TCS-EQ"
            exchange:         Exchange segment e.g. "nse_cm", "nse_fo"
            transaction_type: "B" (buy) or "S" (sell)
            quantity:         Number of shares/lots
            price:            Limit price (0 for market orders)
            order_type:       "L" (limit), "MKT" (market), "SL" (stop loss)
            product:          "CNC" (delivery), "NRML" (F&O/intraday), "MIS" (intraday)
            trigger_price:    For SL orders
            validity:         "DAY" or "IOC"
            disclosed_qty:    Disclosed quantity
            after_market:     True for AMO orders
        """
        if self.paper_trading:
            logger.info(f"[PAPER] Would place: {transaction_type} {quantity} {symbol} @ {price}")
            return {
                "nOrdNo": f"PAPER_{int(time.time())}",
                "stat": "Ok",
                "stCode": 200,
                "paper": True
            }

        jdata = {
            "am":  "YES" if after_market else "NO",
            "dq":  str(disclosed_qty),
            "es":  exchange,
            "mp":  "0",
            "pc":  product,
            "pf":  "N",
            "pr":  str(price),
            "pt":  order_type,
            "qt":  str(quantity),
            "rt":  validity,
            "tp":  str(trigger_price),
            "ts":  symbol,
            "tt":  transaction_type,
        }

        url = f"{self.base_url}/quick/order/rule/ms/place"
        return self._post_form(url, self._trading_headers(), jdata)

    def modify_order(
        self,
        order_no: str,
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        price: float,
        order_type: str = "L",
        product: str = "CNC",
        trigger_price: float = 0,
        validity: str = "DAY",
    ) -> dict:
        """Modify an existing pending order."""
        if self.paper_trading:
            logger.info(f"[PAPER] Would modify order {order_no}")
            return {"stat": "Ok", "paper": True}

        jdata = {
            "am":  "NO",
            "dq":  "0",
            "es":  exchange,
            "mp":  "0",
            "no":  order_no,
            "pc":  product,
            "pf":  "N",
            "pr":  str(price),
            "pt":  order_type,
            "qt":  str(quantity),
            "tp":  str(trigger_price),
            "ts":  symbol,
            "tt":  transaction_type,
            "vd":  validity,
        }

        url = f"{self.base_url}/quick/order/vr/modify"
        return self._post_form(url, self._trading_headers(), jdata)

    def cancel_order(self, order_no: str, after_market: bool = False) -> dict:
        """Cancel a pending order."""
        if self.paper_trading:
            logger.info(f"[PAPER] Would cancel order {order_no}")
            return {"stat": "Ok", "paper": True}

        jdata = {"on": order_no, "am": "YES" if after_market else "NO"}
        url = f"{self.base_url}/quick/order/cancel"
        return self._post_form(url, self._trading_headers(), jdata)

    def exit_cover_order(self, order_no: str) -> dict:
        """Exit a cover order."""
        jdata = {"on": order_no, "am": "NO"}
        url = f"{self.base_url}/quick/order/co/exit"
        return self._post_form(url, self._trading_headers(), jdata)

    def exit_bracket_order(self, order_no: str) -> dict:
        """Exit a bracket order."""
        jdata = {"on": order_no, "am": "NO"}
        url = f"{self.base_url}/quick/order/bo/exit"
        return self._post_form(url, self._trading_headers(), jdata)

    # ─────────────────────────────────────────────────────────────────────────
    # Report APIs
    # ─────────────────────────────────────────────────────────────────────────

    def get_order_book(self) -> dict:
        """Get all orders for today."""
        url = f"{self.base_url}/quick/user/orders"
        return self._get(url, self._trading_headers())

    def get_trade_book(self) -> dict:
        """Get all executed trades for today."""
        url = f"{self.base_url}/quick/user/trades"
        return self._get(url, self._trading_headers())

    def get_positions(self) -> dict:
        """Get open positions (intraday and carry-forward)."""
        url = f"{self.base_url}/quick/user/positions"
        return self._get(url, self._trading_headers())

    def get_holdings(self) -> dict:
        """Get long-term portfolio holdings."""
        url = f"{self.base_url}/portfolio/v1/holdings"
        return self._get(url, self._trading_headers())

    def get_order_history(self, order_no: str) -> dict:
        """Get full history of a specific order."""
        url = f"{self.base_url}/quick/order/history"
        return self._post_form(url, self._trading_headers(), {"nOrdNo": order_no})

    # ─────────────────────────────────────────────────────────────────────────
    # Market Data APIs
    # ─────────────────────────────────────────────────────────────────────────

    def get_quote(self, exchange: str, instrument_token: str) -> dict:
        """
        Get live quote for a symbol.
        Args:
            exchange:          e.g. "nse_cm"
            instrument_token:  e.g. "22" (Infosys token)
        """
        self._ensure_session()
        url = f"{self.base_url}/script-details/1.0/quotes/neosymbol/{exchange}|{instrument_token}/all"
        headers = {
            "Authorization": self.consumer_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        return self._get(url, headers)

    def get_scrip_master_paths(self) -> dict:
        """Get download URLs for scrip master CSV files."""
        url = f"{self.LOGIN_BASE}/script-details/1.0/masterscrip/file-paths"
        headers = {
            "Authorization": self.consumer_key,
            "neo-fin-key": self.NEO_FIN_KEY,
            "Sid":  self.session_sid,
            "Auth": self.session_token,
        }
        return self._get(url, headers)

    # ─────────────────────────────────────────────────────────────────────────
    # Utility APIs
    # ─────────────────────────────────────────────────────────────────────────

    def check_margin(
        self,
        exchange: str,
        instrument_token: str,
        price: float,
        quantity: int,
        order_type: str = "L",
        product: str = "CNC",
        transaction_type: str = "B",
    ) -> dict:
        """Check margin required before placing an order."""
        jdata = {
            "brkName":  "KOTAK",
            "brnchId":  "ONLINE",
            "exSeg":    exchange,
            "prc":      str(price),
            "prcTp":    order_type,
            "prod":     product,
            "qty":      str(quantity),
            "tok":      instrument_token,
            "trnsTp":   transaction_type,
        }
        url = f"{self.base_url}/quick/user/check-margin"
        return self._post_form(url, self._trading_headers(), jdata)

    def get_limits(self, exchange: str = "ALL", segment: str = "ALL", product: str = "ALL") -> dict:
        """Get available funds and margin limits."""
        jdata = {"exch": exchange, "seg": segment, "prod": product}
        url = f"{self.base_url}/quick/user/limits"
        return self._post_form(url, self._trading_headers(), jdata)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience helpers
    # ─────────────────────────────────────────────────────────────────────────

    def buy(self, symbol: str, quantity: int, price: float,
            exchange: str = "nse_cm", product: str = "CNC") -> dict:
        """Shorthand for placing a limit buy order."""
        return self.place_order(symbol, exchange, "B", quantity, price, "L", product)

    def sell(self, symbol: str, quantity: int, price: float,
             exchange: str = "nse_cm", product: str = "CNC") -> dict:
        """Shorthand for placing a limit sell order."""
        return self.place_order(symbol, exchange, "S", quantity, price, "L", product)

    def market_buy(self, symbol: str, quantity: int,
                   exchange: str = "nse_cm", product: str = "CNC") -> dict:
        """Market buy — system will convert to limit per SEBI algo rules."""
        return self.place_order(symbol, exchange, "B", quantity, 0, "MKT", product)

    def market_sell(self, symbol: str, quantity: int,
                    exchange: str = "nse_cm", product: str = "CNC") -> dict:
        """Market sell — system will convert to limit per SEBI algo rules."""
        return self.place_order(symbol, exchange, "S", quantity, 0, "MKT", product)

    def get_available_capital(self) -> float:
        """Returns available cash balance.
        In paper trading mode, returns MAX_CAPITAL from env (no real funds needed).
        """
        if self.paper_trading:
            return float(os.getenv("MAX_CAPITAL", 100000))
        try:
            limits = self.get_limits()
            data = limits.get("data", [{}])
            if isinstance(data, list) and data:
                val = float(data[0].get("net", 0))
                if val > 0:
                    return val
            # Fallback if API returns 0 or empty
            logger.warning("Kotak limits API returned 0 — using MAX_CAPITAL as fallback")
            return float(os.getenv("MAX_CAPITAL", 100000))
        except Exception as e:
            logger.error(f"Could not fetch capital: {e}")
            return float(os.getenv("MAX_CAPITAL", 100000))

    def session_info(self) -> dict:
        """Returns current session status."""
        return {
            "authenticated": self.is_authenticated(),
            "base_url": self.base_url,
            "session_age_minutes": (
                int((datetime.now() - self.session_time).total_seconds() / 60)
                if self.session_time else None
            ),
            "paper_trading": self.paper_trading,
        }
