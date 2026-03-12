"""
Angel One SmartAPI Integration.
Implementation of BrokerInterface for Angel One SmartAPI.
"""

import logging
from typing import Dict, List, Optional, Any
import requests
import json
import pandas as pd
from datetime import datetime
import pyotp

from .broker_interface import BrokerInterface, BrokerOrder, BrokerPosition, OrderType, OrderStatus

logger = logging.getLogger(__name__)


class AngelOneBroker(BrokerInterface):
    """
    Angel One SmartAPI implementation.
    
    Required config:
    - api_key: Angel One API key
    - client_code: Angel One client code
    - password: Angel One password
    - totp_secret: TOTP secret for 2FA (optional)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.client_code = config.get('client_code')
        self.password = config.get('password')
        self.totp_secret = config.get('totp_secret')
        
        # Angel One API endpoints
        self.base_url = "https://apiconnect.angelbroking.com"
        self.session = requests.Session()
        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None
        
        if not all([self.api_key, self.client_code, self.password]):
            raise ValueError("Angel One API key, client code, and password are required")
    
    def connect(self) -> bool:
        """Establish connection to Angel One SmartAPI."""
        try:
            # Generate TOTP if secret is provided
            totp = None
            if self.totp_secret:
                totp = pyotp.TOTP(self.totp_secret).now()
            
            # Login payload
            login_data = {
                "clientcode": self.client_code,
                "password": self.password,
                "totp": totp
            }
            
            # Set headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "1.1.1.1",
                "X-MACAddress": "00:00:00:00:00:00",
                "X-PrivateKey": self.api_key
            }
            
            # Make login request
            response = self.session.post(
                f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword",
                headers=headers,
                data=json.dumps(login_data)
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status'):
                    self.auth_token = result['data']['jwtToken']
                    self.refresh_token = result['data']['refreshToken']
                    self.feed_token = result['data']['feedToken']
                    
                    # Update session headers
                    self.session.headers.update({
                        'Authorization': f'Bearer {self.auth_token}',
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                        'X-UserType': 'USER',
                        'X-SourceID': 'WEB',
                        'X-ClientLocalIP': '127.0.0.1',
                        'X-ClientPublicIP': '1.1.1.1',
                        'X-MACAddress': '00:00:00:00:00:00',
                        'X-PrivateKey': self.api_key
                    })
                    
                    # Get user profile to verify connection
                    profile = self.get_profile()
                    if profile:
                        logger.info(f"Connected to Angel One for client: {self.client_code}")
                        self.is_connected = True
                        return True
                else:
                    logger.error(f"Angel One login failed: {result.get('message')}")
                    return False
            else:
                logger.error(f"Angel One connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Angel One: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Angel One API."""
        try:
            if self.auth_token:
                # Logout request
                logout_data = {"clientcode": self.client_code}
                self.session.post(
                    f"{self.base_url}/rest/secure/angelbroking/user/v1/logout",
                    data=json.dumps(logout_data)
                )
            
            self.auth_token = None
            self.refresh_token = None
            self.feed_token = None
            self.is_connected = False
            logger.info("Disconnected from Angel One")
            
        except Exception as e:
            logger.error(f"Error during Angel One disconnect: {e}")
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile."""
        try:
            response = self.session.get(
                f"{self.base_url}/rest/secure/angelbroking/user/v1/getProfile"
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status'):
                    return result['data']
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get profile: {e}")
            return {}
    
    def place_order(self, order: BrokerOrder) -> Dict[str, Any]:
        """Place order with Angel One."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Angel One")
        
        if not self.validate_order(order):
            raise ValueError("Invalid order parameters")
        
        try:
            # Convert order to Angel One format
            angel_order = {
                "variety": order.variety or "NORMAL",
                "tradingsymbol": order.symbol,
                "symboltoken": self._get_symbol_token(order.symbol, order.exchange),
                "transactiontype": order.transaction_type,
                "exchange": order.exchange,
                "ordertype": order.order_type.value,
                "producttype": order.product_type or "INTRADAY",
                "duration": order.validity or "DAY",
                "price": str(order.price) if order.price else "0",
                "squareoff": "0",
                "stoploss": str(order.trigger_price) if order.trigger_price else "0",
                "quantity": str(order.quantity)
            }
            
            # Place order
            response = self.session.post(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/placeOrder",
                data=json.dumps(angel_order)
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status'):
                    logger.info(f"Order placed successfully: {result}")
                    return {
                        'order_id': result['data']['orderid'],
                        'status': 'SUCCESS',
                        'message': 'Order placed successfully'
                    }
                else:
                    logger.error(f"Order placement failed: {result.get('message')}")
                    return {
                        'order_id': None,
                        'status': 'ERROR',
                        'message': result.get('message', 'Unknown error')
                    }
            else:
                logger.error(f"Order placement failed: {response.status_code}")
                return {
                    'order_id': None,
                    'status': 'ERROR',
                    'message': f'HTTP {response.status_code}'
                }
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {
                'order_id': None,
                'status': 'ERROR',
                'message': str(e)
            }
    
    def modify_order(self, order_id: str, **kwargs) -> Dict[str, Any]:
        """Modify existing order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Angel One")
        
        try:
            modify_data = {
                "variety": kwargs.get('variety', 'NORMAL'),
                "orderid": order_id,
                "ordertype": kwargs.get('order_type', 'LIMIT'),
                "producttype": kwargs.get('product_type', 'INTRADAY'),
                "duration": kwargs.get('validity', 'DAY'),
                "price": str(kwargs.get('price', 0)),
                "quantity": str(kwargs.get('quantity', 0)),
                "tradingsymbol": kwargs.get('symbol', ''),
                "symboltoken": kwargs.get('symbol_token', ''),
                "exchange": kwargs.get('exchange', 'NSE')
            }
            
            response = self.session.post(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/modifyOrder",
                data=json.dumps(modify_data)
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status'):
                    logger.info(f"Order modified successfully: {result}")
                    return {
                        'order_id': result['data']['orderid'],
                        'status': 'SUCCESS',
                        'message': 'Order modified successfully'
                    }
            
            return {
                'order_id': order_id,
                'status': 'ERROR',
                'message': 'Order modification failed'
            }
            
        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return {
                'order_id': order_id,
                'status': 'ERROR',
                'message': str(e)
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel existing order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Angel One")
        
        try:
            cancel_data = {
                "variety": "NORMAL",
                "orderid": order_id
            }
            
            response = self.session.post(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/cancelOrder",
                data=json.dumps(cancel_data)
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status'):
                    logger.info(f"Order cancelled successfully: {result}")
                    return {
                        'order_id': result['data']['orderid'],
                        'status': 'SUCCESS',
                        'message': 'Order cancelled successfully'
                    }
            
            return {
                'order_id': order_id,
                'status': 'ERROR',
                'message': 'Order cancellation failed'
            }
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return {
                'order_id': order_id,
                'status': 'ERROR',
                'message': str(e)
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of specific order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Angel One")
        
        try:
            orders = self.get_orders()
            for order in orders:
                if order.get('orderid') == order_id:
                    return order
            
            return {'error': f'Order {order_id} not found'}
            
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return {'error': str(e)}
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders for the day."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Angel One")
        
        try:
            response = self.session.get(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/getOrderBook"
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') and result.get('data'):
                    return result['data']
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def get_positions(self) -> List[BrokerPosition]:
        """Get current positions."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Angel One")
        
        try:
            response = self.session.get(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/getPosition"
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') and result.get('data'):
                    positions = []
                    
                    for pos in result['data']:
                        if float(pos.get('netqty', 0)) != 0:  # Only non-zero positions
                            position = BrokerPosition(
                                symbol=pos.get('tradingsymbol', ''),
                                exchange=pos.get('exchange', ''),
                                quantity=int(float(pos.get('netqty', 0))),
                                average_price=float(pos.get('avgnetprice', 0)),
                                last_price=float(pos.get('ltp', 0)),
                                pnl=float(pos.get('pnl', 0)),
                                unrealized_pnl=float(pos.get('unrealised', 0)),
                                realized_pnl=float(pos.get('realised', 0))
                            )
                            positions.append(position)
                    
                    return positions
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        """Get current holdings."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Angel One")
        
        try:
            response = self.session.get(
                f"{self.base_url}/rest/secure/angelbroking/portfolio/v1/getHolding"
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') and result.get('data'):
                    return result['data']
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get holdings: {e}")
            return []
    
    def get_funds(self) -> Dict[str, float]:
        """Get account funds information."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Angel One")
        
        try:
            response = self.session.get(
                f"{self.base_url}/rest/secure/angelbroking/user/v1/getRMS"
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') and result.get('data'):
                    data = result['data']
                    return {
                        'available_cash': float(data.get('availablecash', 0)),
                        'available_margin': float(data.get('availablemargin', 0)),
                        'used_margin': float(data.get('utilisedmargin', 0)),
                        'total_margin': float(data.get('net', 0))
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get funds: {e}")
            return {}
    
    def get_ltp(self, symbol: str, exchange: str) -> float:
        """Get last traded price."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Angel One")
        
        try:
            symbol_token = self._get_symbol_token(symbol, exchange)
            if not symbol_token:
                return 0.0
            
            ltp_data = {
                "exchange": exchange,
                "tradingsymbol": symbol,
                "symboltoken": symbol_token
            }
            
            response = self.session.post(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/getLTP",
                data=json.dumps(ltp_data)
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') and result.get('data'):
                    return float(result['data'].get('ltp', 0))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to get LTP for {symbol}: {e}")
            return 0.0
    
    def get_quote(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Get detailed quote."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Angel One")
        
        try:
            symbol_token = self._get_symbol_token(symbol, exchange)
            if not symbol_token:
                return {}
            
            quote_data = {
                "exchange": exchange,
                "tradingsymbol": symbol,
                "symboltoken": symbol_token
            }
            
            response = self.session.post(
                f"{self.base_url}/rest/secure/angelbroking/market/v1/getQuote",
                data=json.dumps(quote_data)
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') and result.get('data'):
                    return result['data']
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, exchange: str, interval: str, 
                          from_date: str, to_date: str) -> pd.DataFrame:
        """
        Get historical data from Angel One.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NSE, BSE)
            interval: Data interval (ONE_MINUTE, THREE_MINUTE, FIVE_MINUTE, etc.)
            from_date: Start date (YYYY-MM-DD HH:MM)
            to_date: End date (YYYY-MM-DD HH:MM)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Angel One")
        
        try:
            symbol_token = self._get_symbol_token(symbol, exchange)
            if not symbol_token:
                return pd.DataFrame()
            
            hist_data = {
                "exchange": exchange,
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            response = self.session.post(
                f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData",
                data=json.dumps(hist_data)
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') and result.get('data'):
                    data = result['data']
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def _get_symbol_token(self, symbol: str, exchange: str) -> str:
        """
        Get symbol token for a trading symbol.
        This is a simplified implementation - in production, you should
        maintain a proper symbol master database.
        """
        # Common symbol tokens (you should expand this or use a proper master file)
        symbol_tokens = {
            'NSE': {
                'RELIANCE': '2885',
                'TCS': '11536',
                'HDFC': '1333',
                'INFY': '1594',
                'ICICIBANK': '4963',
                'SBIN': '3045',
                'BHARTIARTL': '10604',
                'ITC': '1660',
                'HDFCBANK': '1333',
                'LT': '11483'
            },
            'BSE': {
                'RELIANCE': '500325',
                'TCS': '532540',
                'HDFC': '500180'
            }
        }
        
        return symbol_tokens.get(exchange, {}).get(symbol, '')
    
    def subscribe_live_data(self, symbols: List[str], callback) -> None:
        """
        Subscribe to live market data.
        Note: This is handled by the separate AngelOneDataFeed class.
        """
        # This method is implemented in AngelOneDataFeed
        pass
    
    def unsubscribe_live_data(self, symbols: List[str]) -> None:
        """
        Unsubscribe from live market data.
        Note: This is handled by the separate AngelOneDataFeed class.
        """
        # This method is implemented in AngelOneDataFeed
        pass
    
    def refresh_access_token(self) -> bool:
        """Refresh access token using refresh token."""
        try:
            if not self.refresh_token:
                return False
            
            refresh_data = {
                "refreshToken": self.refresh_token
            }
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "1.1.1.1",
                "X-MACAddress": "00:00:00:00:00:00",
                "X-PrivateKey": self.api_key
            }
            
            response = self.session.post(
                f"{self.base_url}/rest/auth/angelbroking/jwt/v1/generateTokens",
                headers=headers,
                data=json.dumps(refresh_data)
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status'):
                    self.auth_token = result['data']['jwtToken']
                    self.refresh_token = result['data']['refreshToken']
                    
                    # Update session headers
                    self.session.headers.update({
                        'Authorization': f'Bearer {self.auth_token}'
                    })
                    
                    logger.info("Access token refreshed successfully")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            return False