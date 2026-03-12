"""
Position Manager - Handles position tracking and management.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from .broker_interface import BrokerPosition

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Position data class."""
    symbol: str
    exchange: str
    quantity: int
    average_price: float
    current_price: float
    pnl: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: datetime
    strategy_name: str = ""
    stop_loss: float = 0.0
    target: float = 0.0


class PositionManager:
    """
    Manages trading positions and P&L tracking.
    """
    
    def __init__(self, broker):
        self.broker = broker
        self.positions = {}  # symbol -> Position
        self.closed_positions = []  # Historical positions
        
        logger.info("Position manager initialized")
    
    def update_positions(self):
        """Update positions from broker."""
        try:
            broker_positions = self.broker.get_positions()
            
            # Clear existing positions
            self.positions.clear()
            
            # Add current positions
            for broker_pos in broker_positions:
                if broker_pos.quantity != 0:  # Only non-zero positions
                    position = Position(
                        symbol=broker_pos.symbol,
                        exchange=broker_pos.exchange,
                        quantity=broker_pos.quantity,
                        average_price=broker_pos.average_price,
                        current_price=broker_pos.last_price,
                        pnl=broker_pos.pnl,
                        unrealized_pnl=broker_pos.unrealized_pnl,
                        realized_pnl=broker_pos.realized_pnl,
                        entry_time=datetime.now(),  # Approximate
                        strategy_name=getattr(broker_pos, 'strategy_name', ''),
                        stop_loss=getattr(broker_pos, 'stop_loss', 0.0),
                        target=getattr(broker_pos, 'target', 0.0)
                    )
                    
                    self.positions[broker_pos.symbol] = position
            
            logger.debug(f"Updated {len(self.positions)} positions")
            
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all current positions."""
        return list(self.positions.values())
    
    def get_long_positions(self) -> List[Position]:
        """Get all long positions."""
        return [pos for pos in self.positions.values() if pos.quantity > 0]
    
    def get_short_positions(self) -> List[Position]:
        """Get all short positions."""
        return [pos for pos in self.positions.values() if pos.quantity < 0]
    
    def get_positions_by_strategy(self, strategy_name: str) -> List[Position]:
        """Get positions for specific strategy."""
        return [
            pos for pos in self.positions.values() 
            if pos.strategy_name == strategy_name
        ]
    
    def calculate_total_pnl(self) -> Dict[str, float]:
        """Calculate total P&L across all positions."""
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        total_pnl = total_unrealized + total_realized
        
        return {
            'total_pnl': total_pnl,
            'unrealized_pnl': total_unrealized,
            'realized_pnl': total_realized
        }
    
    def calculate_strategy_pnl(self, strategy_name: str) -> Dict[str, float]:
        """Calculate P&L for specific strategy."""
        strategy_positions = self.get_positions_by_strategy(strategy_name)
        
        unrealized = sum(pos.unrealized_pnl for pos in strategy_positions)
        realized = sum(pos.realized_pnl for pos in strategy_positions)
        
        return {
            'total_pnl': unrealized + realized,
            'unrealized_pnl': unrealized,
            'realized_pnl': realized
        }
    
    def update_position_prices(self):
        """Update current prices for all positions."""
        for symbol, position in self.positions.items():
            try:
                current_price = self.broker.get_ltp(symbol, position.exchange)
                if current_price > 0:
                    position.current_price = current_price
                    
                    # Recalculate unrealized P&L
                    if position.quantity > 0:  # Long position
                        position.unrealized_pnl = (current_price - position.average_price) * position.quantity
                    else:  # Short position
                        position.unrealized_pnl = (position.average_price - current_price) * abs(position.quantity)
                    
            except Exception as e:
                logger.error(f"Failed to update price for {symbol}: {e}")
    
    def check_stop_loss_targets(self) -> List[Dict[str, Any]]:
        """Check if any positions hit stop loss or target."""
        alerts = []
        
        for symbol, position in self.positions.items():
            if position.stop_loss > 0 or position.target > 0:
                current_price = position.current_price
                
                # Check stop loss
                if position.stop_loss > 0:
                    if (position.quantity > 0 and current_price <= position.stop_loss) or \
                       (position.quantity < 0 and current_price >= position.stop_loss):
                        alerts.append({
                            'type': 'stop_loss',
                            'symbol': symbol,
                            'position': position,
                            'current_price': current_price,
                            'trigger_price': position.stop_loss
                        })
                
                # Check target
                if position.target > 0:
                    if (position.quantity > 0 and current_price >= position.target) or \
                       (position.quantity < 0 and current_price <= position.target):
                        alerts.append({
                            'type': 'target',
                            'symbol': symbol,
                            'position': position,
                            'current_price': current_price,
                            'trigger_price': position.target
                        })
        
        return alerts
    
    def close_position(self, symbol: str, reason: str = "Manual close"):
        """Mark position as closed and move to history."""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.exit_time = datetime.now()
            position.exit_reason = reason
            
            # Move to closed positions
            self.closed_positions.append(position)
            
            # Remove from active positions
            del self.positions[symbol]
            
            logger.info(f"Position closed: {symbol} - {reason}")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions."""
        pnl_data = self.calculate_total_pnl()
        
        return {
            'total_positions': len(self.positions),
            'long_positions': len(self.get_long_positions()),
            'short_positions': len(self.get_short_positions()),
            'total_pnl': pnl_data['total_pnl'],
            'unrealized_pnl': pnl_data['unrealized_pnl'],
            'realized_pnl': pnl_data['realized_pnl'],
            'positions': [
                {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'average_price': pos.average_price,
                    'current_price': pos.current_price,
                    'pnl': pos.unrealized_pnl,
                    'strategy': pos.strategy_name
                }
                for pos in self.positions.values()
            ]
        }
    
    def cleanup_old_closed_positions(self, days: int = 30):
        """Remove old closed positions from history."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        self.closed_positions = [
            pos for pos in self.closed_positions
            if getattr(pos, 'exit_time', datetime.now()) >= cutoff_time
        ]