"""
Broker Interface - Abstract interface for broker integrations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from ..core.models import Order, Position, Trade


class OrderType(Enum):
    """Order types supported by brokers."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "SL"
    STOP_LOSS_MARKET = "SL-M"
    BRACKET_ORDER = "BO"
    COVER_ORDER = "CO"


class OrderStatus(Enum):
    """Order status types."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"


@dataclass
class BrokerOrder:
    """Broker order representation."""
    order_id: str
    symbol: str
    exchange: str
    transaction_type: str  # BUY/SELL
    order_type: OrderType
    quantity: int
    price: float
    trigger_price: Optional[float] = None
    disclosed_quantity: int = 0
    validity: str = "DAY"
    variety: str = "regular"
    tag: Optional[str] = None


@dataclass
class BrokerPosition:
    """Broker position representation."""
    symbol: str
    exchange: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    unrealized_pnl: float
    realized_pnl: float


class BrokerInterface(ABC):
    """
    Abstract interface for broker integrations.
    All broker implementations must inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_connected = False
        self.session_token = None
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to broker API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker API."""
        pass
    
    @abstractmethod
    def place_order(self, order: BrokerOrder) -> Dict[str, Any]:
        """
        Place an order with the broker.
        
        Args:
            order: BrokerOrder object with order details
            
        Returns:
            Dict containing order_id and status
        """
        pass
    
    @abstractmethod
    def modify_order(self, order_id: str, **kwargs) -> Dict[str, Any]:
        """
        Modify an existing order.
        
        Args:
            order_id: Order ID to modify
            **kwargs: Order parameters to modify
            
        Returns:
            Dict containing modified order details
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dict containing cancellation status
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of a specific order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Dict containing order status and details
        """
        pass
    
    @abstractmethod
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get all orders for the day.
        
        Returns:
            List of order dictionaries
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """
        Get current positions.
        
        Returns:
            List of BrokerPosition objects
        """
        pass
    
    @abstractmethod
    def get_holdings(self) -> List[Dict[str, Any]]:
        """
        Get current holdings.
        
        Returns:
            List of holding dictionaries
        """
        pass
    
    @abstractmethod
    def get_funds(self) -> Dict[str, float]:
        """
        Get account funds information.
        
        Returns:
            Dict containing available margin, used margin, etc.
        """
        pass
    
    @abstractmethod
    def get_ltp(self, symbol: str, exchange: str) -> float:
        """
        Get last traded price for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Last traded price
        """
        pass
    
    @abstractmethod
    def get_quote(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get detailed quote for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Dict containing OHLC, volume, etc.
        """
        pass
    
    @abstractmethod
    def subscribe_live_data(self, symbols: List[str], callback) -> None:
        """
        Subscribe to live market data.
        
        Args:
            symbols: List of symbols to subscribe
            callback: Callback function for data updates
        """
        pass
    
    @abstractmethod
    def unsubscribe_live_data(self, symbols: List[str]) -> None:
        """
        Unsubscribe from live market data.
        
        Args:
            symbols: List of symbols to unsubscribe
        """
        pass
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            bool: True if market is open
        """
        # TODO: Implement market hours check
        # This should check NSE/BSE market timings
        from datetime import datetime, time
        
        now = datetime.now().time()
        market_open = time(9, 15)  # 9:15 AM
        market_close = time(15, 30)  # 3:30 PM
        
        return market_open <= now <= market_close
    
    def validate_order(self, order: BrokerOrder) -> bool:
        """
        Validate order parameters before placing.
        
        Args:
            order: BrokerOrder to validate
            
        Returns:
            bool: True if order is valid
        """
        # Basic validation
        if order.quantity <= 0:
            return False
        
        if order.price <= 0 and order.order_type != OrderType.MARKET:
            return False
        
        if not order.symbol or not order.exchange:
            return False
        
        return True
    
    def calculate_margin_required(self, symbol: str, quantity: int, price: float) -> float:
        """
        Calculate margin required for a trade.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to trade
            price: Price per share
            
        Returns:
            Margin required
        """
        # TODO: Implement proper margin calculation
        # This should use broker's margin calculator
        return quantity * price * 0.2  # Assuming 20% margin