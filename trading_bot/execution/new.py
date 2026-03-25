import logging
import requests
import json
import pandas as pd
import time
import pyotp
import uuid

from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class AngelOneBroker:

    def __init__(self, config: Dict[str, Any]):

        self.api_key = config.get("api_key")
        self.client_code = config.get("client_code")
        self.password = config.get("password")
        self.totp_secret = config.get("totp_secret")

        if not all([self.api_key, self.client_code, self.password]):
            raise ValueError("API key, client code and password required")

        self.base_url = "https://apiconnect.angelbroking.com"
        self.session = requests.Session()

        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None
        self.is_connected = False

        # API rate limit protection
        self.api_delay = 0.4
        self.last_request_time = 0

        # cache historical results
        self.historical_cache = {}

        # generate mac address
        self.mac_address = ":".join(
            ["{:02x}".format((uuid.getnode() >> i) & 0xff) for i in range(0, 8 * 6, 8)][::-1]
        )

    # --------------------------------------------------
    # Rate Limited Request Wrapper
    # --------------------------------------------------

    def _request(self, method: str, endpoint: str, payload=None):

        current = time.time()
        elapsed = current - self.last_request_time

        if elapsed < self.api_delay:
            time.sleep(self.api_delay - elapsed)

        self.last_request_time = time.time()

        url = f"{self.base_url}{endpoint}"

        for attempt in range(3):

            try:

                if method == "GET":
                    r = self.session.get(url)

                else:
                    r = self.session.post(url, data=json.dumps(payload))

                if r.status_code == 200:
                    data = r.json()

                    if data.get("status"):
                        return data

                    if data.get("errorcode") == "AB1019":
                        logger.warning("Rate limit hit. Sleeping...")
                        time.sleep(2)
                        continue

                    return data

                else:
                    logger.warning(f"HTTP error {r.status_code}")
                    time.sleep(1)

            except Exception as e:
                logger.error(e)
                time.sleep(1)

        return None

    # --------------------------------------------------
    # Connect
    # --------------------------------------------------

    def connect(self):

        try:

            totp = None
            if self.totp_secret:
                totp = pyotp.TOTP(self.totp_secret).now()

            login_data = {
                "clientcode": self.client_code,
                "password": self.password,
                "totp": totp,
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": self.mac_address,
                "X-PrivateKey": self.api_key,
            }

            response = requests.post(
                f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword",
                headers=headers,
                data=json.dumps(login_data),
            )

            result = response.json()

            if result.get("status"):

                self.auth_token = result["data"]["jwtToken"]
                self.refresh_token = result["data"]["refreshToken"]
                self.feed_token = result["data"]["feedToken"]

                self.session.headers.update(
                    {
                        "Authorization": f"Bearer {self.auth_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "X-UserType": "USER",
                        "X-SourceID": "WEB",
                        "X-ClientLocalIP": "127.0.0.1",
                        "X-ClientPublicIP": "127.0.0.1",
                        "X-MACAddress": self.mac_address,
                        "X-PrivateKey": self.api_key,
                    }
                )

                self.is_connected = True
                logger.info("Angel One connected")

                return True

            else:
                logger.error(result.get("message"))
                return False

        except Exception as e:
            logger.error(e)
            return False

    # --------------------------------------------------
    # Get LTP
    # --------------------------------------------------

    def get_ltp(self, symbol, exchange, token):

        payload = {
            "exchange": exchange,
            "tradingsymbol": symbol,
            "symboltoken": token,
        }

        data = self._request(
            "POST",
            "/rest/secure/angelbroking/order/v1/getLTP",
            payload,
        )

        if data:
            return float(data["data"]["ltp"])

        return 0

    # --------------------------------------------------
    # Get Quote
    # --------------------------------------------------

    def get_quote(self, symbol, exchange, token):

        payload = {
            "exchange": exchange,
            "tradingsymbol": symbol,
            "symboltoken": token,
        }

        data = self._request(
            "POST",
            "/rest/secure/angelbroking/market/v1/getQuote",
            payload,
        )

        if data:
            return data["data"]

        return {}

    # --------------------------------------------------
    # Historical Data (with caching)
    # --------------------------------------------------

    def get_historical_data(
        self,
        symbol,
        exchange,
        token,
        interval,
        from_date,
        to_date,
    ):

        cache_key = f"{symbol}-{interval}-{from_date}-{to_date}"

        if cache_key in self.historical_cache:
            return self.historical_cache[cache_key]

        payload = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date,
        }

        data = self._request(
            "POST",
            "/rest/secure/angelbroking/historical/v1/getCandleData",
            payload,
        )

        if data and data.get("data"):

            df = pd.DataFrame(
                data["data"],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"])

            self.historical_cache[cache_key] = df

            return df

        return pd.DataFrame()

    # --------------------------------------------------
    # Orders
    # --------------------------------------------------

    def place_order(self, order_data):

        data = self._request(
            "POST",
            "/rest/secure/angelbroking/order/v1/placeOrder",
            order_data,
        )

        if data:
            return data

        return {"status": False}

    # --------------------------------------------------
    # Orders list
    # --------------------------------------------------

    def get_orders(self):

        data = self._request(
            "GET",
            "/rest/secure/angelbroking/order/v1/getOrderBook",
        )

        if data:
            return data.get("data", [])

        return []

    # --------------------------------------------------
    # Positions
    # --------------------------------------------------

    def get_positions(self):

        data = self._request(
            "GET",
            "/rest/secure/angelbroking/order/v1/getPosition",
        )

        if data:
            return data.get("data", [])

        return []

    # --------------------------------------------------
    # Funds
    # --------------------------------------------------

    def get_funds(self):

        data = self._request(
            "GET",
            "/rest/secure/angelbroking/user/v1/getRMS",
        )

        if data:
            return data.get("data", {})

        return {}
