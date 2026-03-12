"""
Strategy engine for managing and executing trading strategies.
Coordinates between signals, strategies, and trade decisions.
"""

import asyncio
from typing import Dict, List, Optional, Set
from datetime import datetime
import structlog

from ..core.events import EventHandler, EventBus, TradingSignalEvent, TradeDecisionEvent, EventType
from ..core.models import TradingSignal, TradeDecision, MarketData, Position
from ..core.config import config_manager, StrategyConfig
from .base import Strategy, MarketContext, StrategyState
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy

logger = structlog.get_logger(__name__)


class StrategyEngine(EventHandler):
    """
    Strategy engine that manages multiple trading strategies.
    Processes signals and generates trade decisions based on active strategies.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.strategies: Dict[str, Strategy] = {}
        self.strategy_factories = {
            'momentum': MomentumStrategy,
            'mean_reversion': MeanReversionStrategy
        }
        
        # Market data cache for strategy context
        self.market_data_cache: Dict[str, MarketData] = {}
        self.position_cache: Dict[str, Position] = {}
        self.available_capital = 100000  # Default capital
        
        # Strategy conflict resolution
        self.conflict_resolution = 'priority'  # 'priority', 'allocation', 'first'
        self.max_concurrent_signals = 5
        
        # Subscribe to events
        self.event_bus.subscribe(EventType.TRADING_SIGNAL, self)
        
        # Load strategies from configuration
        self._load_strategies()
    
    def _load_strategies(self) -> None:
        """Load strategies from configuration."""
        try:
            strategy_configs = config_manager.trading_config.get_active_strategies()
            
            for config in strategy_configs:
                self.add_strategy(config)
            
            logger.info(f"Loaded {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
    
    def add_strategy(self, strategy_config: StrategyConfig) -> bool:
        """Add a new strategy to the engine."""
        try:
            strategy_type = strategy_config.strategy_type
            
            if strategy_type not in self.strategy_factories:
                logger.error(f"Unknown strategy type: {strategy_type}")
                return False
            
            # Create strategy instance
            strategy_class = self.strategy_factories[strategy_type]
            strategy = strategy_class(strategy_config)
            
            # Add to strategies dict
            self.strategies[strategy_config.strategy_id] = strategy
            
            # Start strategy if enabled
            if strategy_config.enabled:
                strategy.start()
            
            logger.info(f"Added strategy: {strategy_config.strategy_id} ({strategy_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding strategy {strategy_config.strategy_id}: {e}")
            return False
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy from the engine."""
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"Strategy not found: {strategy_id}")
                return False
            
            strategy = self.strategies[strategy_id]
            strategy.stop()
            
            del self.strategies[strategy_id]
            
            logger.info(f"Removed strategy: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing strategy {strategy_id}: {e}")
            return False
    
    async def handle_event(self, event) -> None:
        """Handle incoming events."""
        if isinstance(event, TradingSignalEvent):
            await self._process_trading_signal(event)
    
    async def _process_trading_signal(self, event: TradingSignalEvent) -> None:
        """Process trading signal and generate trade decisions."""
        try:
            # Convert event to TradingSignal object
            signal = TradingSignal(
                symbol=event.symbol,
                signal_type=event.signal_type,
                strength=event.strength,
                indicators_used=event.indicators_used,
                price_target=event.price_target
            )
            
            logger.debug(f"Processing signal: {signal.signal_type} for {signal.symbol}")
            
            # Get market context
            context = self._build_market_context(signal.symbol)
            if not context:
                logger.warning(f"Could not build market context for {signal.symbol}")
                return
            
            # Evaluate signal with all active strategies
            decisions = []
            for strategy_id, strategy in self.strategies.items():
                if strategy.state != StrategyState.ACTIVE:
                    continue
                
                decision = strategy.evaluate_signal(signal, context)
                if decision:
                    decisions.append(decision)
            
            # Resolve conflicts if multiple strategies want to trade
            if len(decisions) > 1:
                decisions = self._resolve_strategy_conflicts(decisions)
            
            # Publish trade decisions
            for decision in decisions:
                await self._publish_trade_decision(decision)
            
        except Exception as e:
            logger.error(f"Error processing trading signal: {e}")
    
    def _build_market_context(self, symbol: str) -> Optional[MarketContext]:
        """Build market context for strategy evaluation."""
        try:
            # Get current market data
            market_data = self.market_data_cache.get(symbol)
            if not market_data or not market_data.current_tick:
                logger.debug(f"No market data available for {symbol}")
                return None
            
            current_price = market_data.current_tick.price
            
            # Get current position
            current_position = self.position_cache.get(symbol)
            
            # Calculate portfolio exposure
            portfolio_exposure = sum(
                abs(pos.quantity) * pos.current_price 
                for pos in self.position_cache.values()
            )
            
            context = MarketContext(
                symbol=symbol,
                current_price=current_price,
                market_data=market_data,
                current_position=current_position,
                available_capital=self.available_capital,
                portfolio_exposure=portfolio_exposure
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error building market context for {symbol}: {e}")
            return None
    
    def _resolve_strategy_conflicts(self, decisions: List[TradeDecision]) -> List[TradeDecision]:
        """Resolve conflicts when multiple strategies want to trade the same symbol."""
        try:
            if len(decisions) <= 1:
                return decisions
            
            logger.info(f"Resolving conflicts between {len(decisions)} strategy decisions")
            
            if self.conflict_resolution == 'priority':
                # Use strategy priority (could be based on performance, allocation, etc.)
                return self._resolve_by_priority(decisions)
            
            elif self.conflict_resolution == 'allocation':
                # Combine decisions based on strategy allocations
                return self._resolve_by_allocation(decisions)
            
            elif self.conflict_resolution == 'first':
                # Use first decision (first come, first served)
                return [decisions[0]]
            
            else:
                logger.warning(f"Unknown conflict resolution method: {self.conflict_resolution}")
                return [decisions[0]]
            
        except Exception as e:
            logger.error(f"Error resolving strategy conflicts: {e}")
            return decisions[:1]  # Return first decision as fallback
    
    def _resolve_by_priority(self, decisions: List[TradeDecision]) -> List[TradeDecision]:
        """Resolve conflicts by strategy priority."""
        # For now, use strategy performance as priority
        # In practice, you might have explicit priority settings
        
        strategy_priorities = {}
        for decision in decisions:
            strategy = self.strategies.get(decision.strategy_id)
            if strategy:
                # Use win rate as priority metric
                priority = strategy.performance.win_rate
                strategy_priorities[decision.strategy_id] = priority
        
        # Sort decisions by priority (highest first)
        sorted_decisions = sorted(
            decisions,
            key=lambda d: strategy_priorities.get(d.strategy_id, 0),
            reverse=True
        )
        
        return [sorted_decisions[0]]  # Return highest priority decision
    
    def _resolve_by_allocation(self, decisions: List[TradeDecision]) -> List[TradeDecision]:
        """Resolve conflicts by combining decisions based on allocations."""
        # This is more complex - would need to adjust quantities based on allocations
        # For now, return the decision from strategy with highest allocation
        
        max_allocation = 0
        best_decision = decisions[0]
        
        for decision in decisions:
            strategy = self.strategies.get(decision.strategy_id)
            if strategy and strategy.allocation > max_allocation:
                max_allocation = strategy.allocation
                best_decision = decision
        
        return [best_decision]
    
    async def _publish_trade_decision(self, decision: TradeDecision) -> None:
        """Publish trade decision to event bus."""
        try:
            decision_event = TradeDecisionEvent(
                symbol=decision.symbol,
                action=decision.action.value,
                quantity=decision.quantity,
                order_type=decision.order_type.value,
                price=decision.price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                strategy_id=decision.strategy_id,
                source_component="StrategyEngine"
            )
            
            await self.event_bus.publish(decision_event)
            
            logger.info(f"Published trade decision: {decision.action.value} {decision.quantity} "
                       f"{decision.symbol} from strategy {decision.strategy_id}")
            
        except Exception as e:
            logger.error(f"Error publishing trade decision: {e}")
    
    def update_market_data(self, symbol: str, market_data: MarketData) -> None:
        """Update market data cache."""
        self.market_data_cache[symbol] = market_data
    
    def update_position(self, symbol: str, position: Optional[Position]) -> None:
        """Update position cache."""
        if position is None or position.quantity == 0:
            self.position_cache.pop(symbol, None)
        else:
            self.position_cache[symbol] = position
        
        # Update position in relevant strategies
        for strategy in self.strategies.values():
            strategy.update_position(symbol, position)
    
    def update_available_capital(self, capital: float) -> None:
        """Update available capital."""
        self.available_capital = capital
    
    def start_strategy(self, strategy_id: str) -> bool:
        """Start a specific strategy."""
        if strategy_id not in self.strategies:
            logger.error(f"Strategy not found: {strategy_id}")
            return False
        
        return self.strategies[strategy_id].start()
    
    def stop_strategy(self, strategy_id: str) -> bool:
        """Stop a specific strategy."""
        if strategy_id not in self.strategies:
            logger.error(f"Strategy not found: {strategy_id}")
            return False
        
        return self.strategies[strategy_id].stop()
    
    def pause_strategy(self, strategy_id: str) -> bool:
        """Pause a specific strategy."""
        if strategy_id not in self.strategies:
            logger.error(f"Strategy not found: {strategy_id}")
            return False
        
        return self.strategies[strategy_id].pause()
    
    def resume_strategy(self, strategy_id: str) -> bool:
        """Resume a specific strategy."""
        if strategy_id not in self.strategies:
            logger.error(f"Strategy not found: {strategy_id}")
            return False
        
        return self.strategies[strategy_id].resume()
    
    def update_strategy_parameters(self, strategy_id: str, parameters: Dict) -> bool:
        """Update strategy parameters."""
        if strategy_id not in self.strategies:
            logger.error(f"Strategy not found: {strategy_id}")
            return False
        
        return self.strategies[strategy_id].update_parameters(parameters)
    
    def get_strategy_status(self, strategy_id: Optional[str] = None) -> Dict:
        """Get status of strategies."""
        if strategy_id:
            if strategy_id not in self.strategies:
                return {}
            return self.strategies[strategy_id].get_status()
        
        # Return status of all strategies
        return {
            strategy_id: strategy.get_status()
            for strategy_id, strategy in self.strategies.items()
        }
    
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategy IDs."""
        return [
            strategy_id for strategy_id, strategy in self.strategies.items()
            if strategy.state == StrategyState.ACTIVE
        ]
    
    def get_strategy_performance(self) -> Dict:
        """Get performance summary of all strategies."""
        performance = {}
        
        for strategy_id, strategy in self.strategies.items():
            performance[strategy_id] = {
                'total_trades': strategy.performance.total_trades,
                'win_rate': strategy.performance.win_rate,
                'total_pnl': float(strategy.performance.total_pnl),
                'active_positions': len(strategy.active_positions)
            }
        
        return performance


# Create global strategy engine instance (will be initialized with event_bus in main)
strategy_engine = None