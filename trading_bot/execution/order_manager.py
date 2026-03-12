"""
Order Manager - Handles order lifecycle and management.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

from .broker_interface import BrokerInterface, BrokerOrder, OrderStatus

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """Order states."""
    PENDING = "PENDING"
    PLACED = "PLACED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderManager:
    """
    Manages order lifecycle and tracking.
    """
    
    def __init__(self, broker: BrokerInterface):
        self.broker = broker
        self.orders = {}  # order_id -> order_info
        self.pending_orders = {}  # strategy orders waiting to be placed
        
        logger.info("Order manager initialized")
    
    def place_order(self, order: BrokerOrder) -> Dict[str, Any]:
        """
        Place order through broker.
        
        Args:
            order: BrokerOrder to place
            
        Returns:
            Dict with order result
        """
        try:
            # Place order with broker
            result = self.broker.place_order(order)
            
            if result.get('status') == 'SUCCESS':
                order_id = result.get('order_id')
                
                # Track order
                self.orders[order_id] = {
                    'order': order,
                    'state': OrderState.PLACED,
                    'placed_time': datetime.now(),
                    'filled_quantity': 0,
                    'average_price': 0.0,
                    'broker_result': result
                }
                
                logger.info(f"Order placed: {order_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {
                'status': 'ERROR',
                'message': str(e)
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order."""
        try:
            result = self.broker.cancel_order(order_id)
            
            if result.get('status') == 'SUCCESS' and order_id in self.orders:
                self.orders[order_id]['state'] = OrderState.CANCELLED
                self.orders[order_id]['cancelled_time'] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return {
                'status': 'ERROR',
                'message': str(e)
            }
    
    def modify_order(self, order_id: str, **kwargs) -> Dict[str, Any]:
        """Modify order."""
        try:
            result = self.broker.modify_order(order_id, **kwargs)
            
            if result.get('status') == 'SUCCESS' and order_id in self.orders:
                self.orders[order_id]['modified_time'] = datetime.now()
                self.orders[order_id]['modifications'] = kwargs
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return {
                'status': 'ERROR',
                'message': str(e)
            }
    
    def update_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Update order status from broker."""
        try:
            status = self.broker.get_order_status(order_id)
            
            if order_id in self.orders and status:
                # Update order state based on broker status
                broker_status = status.get('status', '').upper()
                
                if broker_status in ['COMPLETE', 'FILLED']:
                    self.orders[order_id]['state'] = OrderState.FILLED
                    self.orders[order_id]['filled_time'] = datetime.now()
                    self.orders[order_id]['filled_quantity'] = status.get('quantity', 0)
                    self.orders[order_id]['average_price'] = status.get('average_price', 0)
                elif broker_status == 'CANCELLED':
                    self.orders[order_id]['state'] = OrderState.CANCELLED
                elif broker_status == 'REJECTED':
                    self.orders[order_id]['state'] = OrderState.REJECTED
                elif broker_status in ['PARTIAL', 'PARTIALLY_FILLED']:
                    self.orders[order_id]['state'] = OrderState.PARTIALLY_FILLED
                    self.orders[order_id]['filled_quantity'] = status.get('filled_quantity', 0)
                
                self.orders[order_id]['last_update'] = datetime.now()
                self.orders[order_id]['broker_status'] = status
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to update order status for {order_id}: {e}")
            return None
    
    def get_order_info(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order information."""
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders."""
        active_states = [OrderState.PENDING, OrderState.PLACED, OrderState.PARTIALLY_FILLED]
        return [
            order_info for order_info in self.orders.values()
            if order_info['state'] in active_states
        ]
    
    def get_filled_orders(self) -> List[Dict[str, Any]]:
        """Get all filled orders."""
        return [
            order_info for order_info in self.orders.values()
            if order_info['state'] == OrderState.FILLED
        ]
    
    def update_all_orders(self):
        """Update status of all active orders."""
        active_orders = self.get_active_orders()
        
        for order_info in active_orders:
            order_id = order_info.get('broker_result', {}).get('order_id')
            if order_id:
                self.update_order_status(order_id)
    
    def cleanup_old_orders(self, days: int = 7):
        """Remove old completed orders."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        orders_to_remove = []
        for order_id, order_info in self.orders.items():
            if (order_info['state'] in [OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED] and
                order_info.get('placed_time', datetime.now()) < cutoff_time):
                orders_to_remove.append(order_id)
        
        for order_id in orders_to_remove:
            del self.orders[order_id]
        
        if orders_to_remove:
            logger.info(f"Cleaned up {len(orders_to_remove)} old orders")