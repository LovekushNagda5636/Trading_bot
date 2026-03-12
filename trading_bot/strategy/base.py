"""
Base strategy framework for the trading bot.
Provides abstract base classes and common functionality for trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import structlog

from ..core.models import TradingSignal, TradeDecision, MarketData, Position, OrderSide, OrderType
from ..core.config import StrategyConfig

logger = structlog.get_logger(__name__)


class StrategyState(Enum):
    """Strategy execution states."""
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


@dataclass
class MarketContext:
    """Market context information for strategy decisions."""
    symbol: str
    current_price: Decimal
    market_data: Optional[MarketData] = None
    current_position: Optional[Position] = None
    available_capital: Decimal = Decimal('0')
    portfolio_exposure: Decimal = Decimal('0')
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExitCondition:
    """Exit condition for a trade."""
    condition_type: str  # "stop_loss", "take_profit", "time_based", "indicator"
    trigger_value: Optional[Decimal] = None
    trigger_indicator: Optional[str] = None
    enabled: bool = True


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal('0')
    max_drawdown: Decimal = Decimal('0')
    win_rate: float = 0.0
    avg_win: Decimal = Decimal('0')
    avg_loss: Decimal = Decimal('0')
    sharpe_ratio: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    Defines the interface that all strategies must implement.
    """
    
    def __init__(self, strategy_config: StrategyConfig):
        self.config = strategy_config
        self.strategy_id = strategy_config.strategy_id
        self.strategy_type = strategy_config.strategy_type
        self.enabled = strategy_config.enabled
        self.allocation = strategy_config.allocation
        self.parameters = strategy_config.parameters
        
        self.state = StrategyState.INACTIVE
        self.performance = StrategyPerformance()
        self.active_positions: Dict[str, Position] = {}
        self.signal_history: List[TradingSignal] = []
        self.decision_history: List[TradeDecision] = []
        
        # Strategy-specific settings
        self.max_positions = self.parameters.get('max_positions', 5)
        self.position_size_pct = self.parameters.get('position_size_pct', 0.1)  # 10% of allocation
        self.stop_loss_pct = self.parameters.get('stop_loss_pct', 0.02)  # 2% stop loss
        self.take_profit_pct = self.parameters.get('take_profit_pct', 0.04)  # 4% take profit
        
        logger.info(f"Initialized strategy: {self.strategy_id} ({self.strategy_type})")
    
    @abstractmethod
    def should_enter_trade(self, signal: TradingSignal, context: MarketContext) -> bool:
        """
        Determine if the strategy should enter a trade based on the signal and market context.
        
        Args:
            signal: Trading signal from technical analysis
            context: Current market context
            
        Returns:
            True if should enter trade, False otherwise
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, context: MarketContext) -> int:
        """
        Calculate the position size for a trade.
        
        Args:
            context: Current market context
            
        Returns:
            Number of shares/units to trade
        """
        pass
    
    @abstractmethod
    def get_exit_conditions(self, context: MarketContext) -> List[ExitCondition]:
        """
        Get exit conditions for the current position.
        
        Args:
            context: Current market context
            
        Returns:
            List of exit conditions
        """
        pass
    
    def should_exit_trade(self, position: Position, context: MarketContext) -> bool:
        """
        Determine if should exit an existing position.
        
        Args:
            position: Current position
            context: Market context
            
        Returns:
            True if should exit, False otherwise
        """
        exit_conditions = self.get_exit_conditions(context)
        
        for condition in exit_conditions:
            if not condition.enabled:
                continue
                
            if self._check_exit_condition(condition, position, context):
                logger.info(f"Exit condition triggered: {condition.condition_type} for {position.symbol}")
                return True
        
        return False
    
    def _check_exit_condition(self, condition: ExitCondition, position: Position, context: MarketContext) -> bool:
        """Check if an exit condition is met."""
        try:
            if condition.condition_type == "stop_loss":
                if position.is_long:
                    return context.current_price <= (position.average_price * (1 - self.stop_loss_pct))
                else:
                    return context.current_price >= (position.average_price * (1 + self.stop_loss_pct))
            
            elif condition.condition_type == "take_profit":
                if position.is_long:
                    return context.current_price >= (position.average_price * (1 + self.take_profit_pct))
                else:
                    return context.current_price <= (position.average_price * (1 - self.take_profit_pct))
            
            elif condition.condition_type == "time_based":
                # Example: Exit after holding for certain time
                hold_time = datetime.now() - position.timestamp
                max_hold_hours = self.parameters.get('max_hold_hours', 24)
                return hold_time.total_seconds() > (max_hold_hours * 3600)
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking exit condition {condition.condition_type}: {e}")
            return False
    
    def evaluate_signal(self, signal: TradingSignal, context: MarketContext) -> Optional[TradeDecision]:
        """
        Evaluate a trading signal and generate a trade decision.
        
        Args:
            signal: Trading signal to evaluate
            context: Market context
            
        Returns:
            TradeDecision if trade should be made, None otherwise
        """
        try:
            if not self.enabled or self.state != StrategyState.ACTIVE:
                return None
            
            # Check if we should enter the trade
            if not self.should_enter_trade(signal, context):
                return None
            
            # Check position limits
            if len(self.active_positions) >= self.max_positions:
                logger.debug(f"Max positions reached for strategy {self.strategy_id}")
                return None
            
            # Check if we already have a position in this symbol
            if signal.symbol in self.active_positions:
                logger.debug(f"Already have position in {signal.symbol}")
                return None
            
            # Calculate position size
            quantity = self.calculate_position_size(context)
            if quantity <= 0:
                logger.debug(f"Invalid position size calculated: {quantity}")
                return None
            
            # Determine order side based on signal
            order_side = OrderSide.BUY if signal.signal_type.value == "BUY" else OrderSide.SELL
            
            # Create trade decision
            decision = TradeDecision(
                symbol=signal.symbol,
                action=order_side,
                quantity=quantity,
                order_type=OrderType.MARKET,  # Default to market orders
                strategy_id=self.strategy_id,
                stop_loss=self._calculate_stop_loss(context.current_price, order_side),
                take_profit=self._calculate_take_profit(context.current_price, order_side)
            )
            
            # Store signal and decision history
            self.signal_history.append(signal)
            self.decision_history.append(decision)
            
            # Keep history limited
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-1000:]
            
            logger.info(f"Strategy {self.strategy_id} generated trade decision: "
                       f"{decision.action.value} {decision.quantity} {decision.symbol}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error evaluating signal in strategy {self.strategy_id}: {e}")
            return None
    
    def _calculate_stop_loss(self, current_price: Decimal, order_side: OrderSide) -> Optional[Decimal]:
        """Calculate stop loss price."""
        if order_side == OrderSide.BUY:
            return current_price * (1 - Decimal(str(self.stop_loss_pct)))
        else:
            return current_price * (1 + Decimal(str(self.stop_loss_pct)))
    
    def _calculate_take_profit(self, current_price: Decimal, order_side: OrderSide) -> Optional[Decimal]:
        """Calculate take profit price."""
        if order_side == OrderSide.BUY:
            return current_price * (1 + Decimal(str(self.take_profit_pct)))
        else:
            return current_price * (1 - Decimal(str(self.take_profit_pct)))
    
    def update_position(self, symbol: str, position: Optional[Position]) -> None:
        """Update position information."""
        if position is None or position.quantity == 0:
            self.active_positions.pop(symbol, None)
        else:
            self.active_positions[symbol] = position
    
    def update_performance(self, trade_pnl: Decimal, is_winning_trade: bool) -> None:
        """Update strategy performance metrics."""
        self.performance.total_trades += 1
        self.performance.total_pnl += trade_pnl
        
        if is_winning_trade:
            self.performance.winning_trades += 1
            if self.performance.winning_trades > 0:
                self.performance.avg_win = (
                    (self.performance.avg_win * (self.performance.winning_trades - 1) + trade_pnl) /
                    self.performance.winning_trades
                )
        else:
            self.performance.losing_trades += 1
            if self.performance.losing_trades > 0:
                self.performance.avg_loss = (
                    (self.performance.avg_loss * (self.performance.losing_trades - 1) + abs(trade_pnl)) /
                    self.performance.losing_trades
                )
        
        # Update win rate
        if self.performance.total_trades > 0:
            self.performance.win_rate = self.performance.winning_trades / self.performance.total_trades
        
        self.performance.last_updated = datetime.now()
    
    def start(self) -> bool:
        """Start the strategy."""
        try:
            self.state = StrategyState.ACTIVE
            logger.info(f"Started strategy: {self.strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Error starting strategy {self.strategy_id}: {e}")
            self.state = StrategyState.ERROR
            return False
    
    def stop(self) -> bool:
        """Stop the strategy."""
        try:
            self.state = StrategyState.INACTIVE
            logger.info(f"Stopped strategy: {self.strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Error stopping strategy {self.strategy_id}: {e}")
            return False
    
    def pause(self) -> bool:
        """Pause the strategy."""
        try:
            self.state = StrategyState.PAUSED
            logger.info(f"Paused strategy: {self.strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Error pausing strategy {self.strategy_id}: {e}")
            return False
    
    def resume(self) -> bool:
        """Resume the strategy."""
        try:
            self.state = StrategyState.ACTIVE
            logger.info(f"Resumed strategy: {self.strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Error resuming strategy {self.strategy_id}: {e}")
            return False
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> bool:
        """Update strategy parameters."""
        try:
            self.parameters.update(new_parameters)
            
            # Update derived parameters
            self.max_positions = self.parameters.get('max_positions', self.max_positions)
            self.position_size_pct = self.parameters.get('position_size_pct', self.position_size_pct)
            self.stop_loss_pct = self.parameters.get('stop_loss_pct', self.stop_loss_pct)
            self.take_profit_pct = self.parameters.get('take_profit_pct', self.take_profit_pct)
            
            logger.info(f"Updated parameters for strategy {self.strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating parameters for strategy {self.strategy_id}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status and performance."""
        return {
            'strategy_id': self.strategy_id,
            'strategy_type': self.strategy_type,
            'state': self.state.value,
            'enabled': self.enabled,
            'allocation': float(self.allocation),
            'active_positions': len(self.active_positions),
            'max_positions': self.max_positions,
            'performance': {
                'total_trades': self.performance.total_trades,
                'winning_trades': self.performance.winning_trades,
                'losing_trades': self.performance.losing_trades,
                'win_rate': self.performance.win_rate,
                'total_pnl': float(self.performance.total_pnl),
                'avg_win': float(self.performance.avg_win),
                'avg_loss': float(self.performance.avg_loss)
            },
            'parameters': self.parameters
        }