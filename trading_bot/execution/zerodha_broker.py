"""
Zerodha Kite API Integration.
Implementation of BrokerInterface for Zerodha Kite.
"""

import logging
from typing import Dict, List, Optional, Any
from kiteconnect import KiteConnect
import pandas as pd

from .broker_interface import BrokerInterface, BrokerOrder, BrokerPosition, OrderType, OrderStatus

logger = logging.getLogger(__name__)


class ZerodhaBroker(BrokerInterface):
    """
    Zerodha Kite API implementation.
    
    Required config:
    - api_key: Kite API key
    - api_secret: Kite API secret  
    - access_token: Access token (obtained after login)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.access_token = config.get('access_token')
        self.kite = None
        
        if not all([self.api_key, self.api_secret]):
            raise ValueError("Zerodha API key and secret are required")
    
    def connect(self) -> bool:
        """Establish connection to Zerodha Kite API."""
        try:
            self.kite = KiteConnect(api_key=self.api_key)
            
            if self.access_token:
                self.kite.set_access_token(self.access_token)
                # Verify connection
                profile = self.kite.profile()
                logger.info(f"Connected to Zerodha for user: {profile.get('user_name')}")
                self.is_connected = True
                return True
            else:
                logger.error("Access token required for Zerodha connection")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Zerodha: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Zerodha API."""
        self.kite = None
        self.is_connected = False
        logger.info("Disconnected from Zerodha")
    
    def place_order(self, order: BrokerOrder) -> Dict[str, Any]:
        """Place order with Zerodha."""
        if not self.is_connected or not self.kite:
            raise ConnectionError("Not connected to Zerodha")
        
        if not self.validate_order(order):
            raise ValueError("Invalid order parameters")
        
        try:
            # Convert order to Kite format
            kite_order = {
                'exchange': order.exchange,
                'tradingsymbol': order.symbol,
                'transaction_type': order.transaction_type,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'variety': order.variety,
                'validity': order.validity
            }
            
            # Add price for limit orders
            if order.order_type != OrderType.MARKET:
                kite_order['price'] = order.price
            
            # Add trigger price for stop loss orders
            if order.trigger_price:
                kite_order['trigger_price'] = order.trigger_price
            
            # Add disclosed quantity if specified
            if order.disclosed_quantity > 0:
                kite_order['disclosed_quantity'] = order.disclosed_quantity
            
            # Add tag if specified
            if order.tag:
                kite_order['tag'] = order.tag
            
            # Place order
            response = self.kite.place_order(**kite_order)
            
            logger.info(f"Order placed successfully: {response}")
            return {
                'order_id': response['order_id'],
                'status': 'SUCCESS',
                'message': 'Order placed successfully'
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
        if not self.is_connected or not self.kite:
            raise ConnectionError("Not connected to Zerodha")
        
        try:
            response = self.kite.modify_order(
                order_id=order_id,
                variety=kwargs.get('variety', 'regular'),
                **kwargs
            )
            
            logger.info(f"Order modified successfully: {response}")
            return {
                'order_id': response['order_id'],
                'status': 'SUCCESS',
                'message': 'Order modified successfully'
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
        if not self.is_connected or not self.kite:
            raise ConnectionError("Not connected to Zerodha")
        
        try:
            response = self.kite.cancel_order(
                order_id=order_id,
                variety='regular'  # Default variety
            )
            
            logger.info(f"Order cancelled successfully: {response}")
            return {
                'order_id': response['order_id'],
                'status': 'SUCCESS',
                'message': 'Order cancelled successfully'
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
        if not self.is_connected or not self.kite:
            raise ConnectionError("Not connected to Zerodha")
        
        try:
            orders = self.kite.orders()
            for order in orders:
                if order['order_id'] == order_id:
                    return order
            
            return {'error': f'Order {order_id} not found'}
            
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return {'error': str(e)}
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders for the day."""
        if not self.is_connected or not self.kite:
            raise ConnectionError("Not connected to Zerodha")
        
        try:
            return self.kite.orders()
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def get_positions(self) -> List[BrokerPosition]:
        """Get current positions."""
        if not self.is_connected or not self.kite:
            raise ConnectionError("Not connected to Zerodha")
        
        try:
            positions_data = self.kite.positions()
            positions = []
            
            # Process both day and net positions
            for pos_type in ['day', 'net']:
                for pos in positions_data.get(pos_type, []):
                    if pos['quantity'] != 0:  # Only include non-zero positions
                        position = BrokerPosition(
                            symbol=pos['tradingsymbol'],
                            exchange=pos['exchange'],
                            quantity=pos['quantity'],
                            average_price=pos['average_price'],
                            last_price=pos['last_price'],
                            pnl=pos['pnl'],
                            unrealized_pnl=pos['unrealised'],
                            realized_pnl=pos['realised']
                        )
                        positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        """Get current holdings."""
        if not self.is_connected or not self.kite:
            raise ConnectionError("Not connected to Zerodha")
        
        try:
            return self.kite.holdings()
        except Exception as e:
            logger.error(f"Failed to get holdings: {e}")
            return []
    
    def get_funds(self) -> Dict[str, float]:
        """Get account funds information."""
        if not self.is_connected or not self.kite:
            raise ConnectionError("Not connected to Zerodha")
        
        try:
            margins = self.kite.margins()
            equity_margin = margins.get('equity', {})
            
            return {
                'available_cash': equity_margin.get('available', {}).get('cash', 0.0),
                'available_margin': equity_margin.get('available', {}).get('adhoc_margin', 0.0),
                'used_margin': equity_margin.get('utilised', {}).get('debits', 0.0),
                'total_margin': equity_margin.get('net', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get funds: {e}")
            return {}
    
    def get_ltp(self, symbol: str, exchange: str) -> float:
        """Get last traded price."""
        if not self.is_connected or not self.kite:
            raise ConnectionError("Not connected to Zerodha")
        
        try:
            instrument_key = f"{exchange}:{symbol}"
            ltp_data = self.kite.ltp([instrument_key])
            return ltp_data[instrument_key]['last_price']
            
        except Exception as e:
            logger.error(f"Failed to get LTP for {symbol}: {e}")
            return 0.0
    
    def get_quote(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Get detailed quote."""
        if not self.is_connected or not self.kite:
            raise ConnectionError("Not connected to Zerodha")
        
        try:
            instrument_key = f"{exchange}:{symbol}"
            quote_data = self.kite.quote([instrument_key])
            return quote_data[instrument_key]
            
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return {}
    
    def subscribe_live_data(self, symbols: List[str], callback) -> None:
        """Subscribe to live market data via WebSocket."""
        # TODO: Implement KiteTicker WebSocket integration
        logger.warning("Live data subscription not implemented yet")
        pass
    
    def unsubscribe_live_data(self, symbols: List[str]) -> None:
        """Unsubscribe from live market data."""
        # TODO: Implement KiteTicker WebSocket integration
        logger.warning("Live data unsubscription not implemented yet")
        pass
    
    def get_instruments(self, exchange: str = None) -> pd.DataFrame:
        """
        Get instruments list from Zerodha.
        
        Args:
            exchange: Exchange name (NSE, BSE, etc.)
            
        Returns:
            DataFrame with instrument details
        """
        if not self.is_connected or not self.kite:
            raise ConnectionError("Not connected to Zerodha")
        
        try:
            if exchange:
                instruments = self.kite.instruments(exchange)
            else:
                instruments = self.kite.instruments()
            
            return pd.DataFrame(instruments)
            
        except Exception as e:
            logger.error(f"Failed to get instruments: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, instrument_token: str, from_date: str, 
                          to_date: str, interval: str) -> pd.DataFrame:
        """
        Get historical data from Zerodha.
        
        Args:
            instrument_token: Instrument token
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            interval: Data interval (minute, 3minute, 5minute, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected or not self.kite:
            raise ConnectionError("Not connected to Zerodha")
        
        try:
            historical_data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            return pd.DataFrame(historical_data)
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()