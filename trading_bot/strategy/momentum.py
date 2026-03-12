"""
Momentum trading strategy implementation.
Trades based on price momentum and technical indicator signals.
"""

from decimal import Decimal
from typing import List, Optional
import structlog

from .base import Strategy, MarketContext, ExitCondition
from ..core.models import TradingSignal, SignalType, OrderSide
from ..core.config import StrategyConfig

logger = structlog.get_logger(__name__)


class MomentumStrategy(Strategy):
    """
    Momentum trading strategy that follows price trends.
    
    Strategy Logic:
    - Enter long positions when price shows strong upward momentum
    - Enter short positions when price shows strong downward momentum
    - Use multiple indicators to confirm momentum
    - Exit when momentum reverses or risk limits are hit
    """
    
    def __init__(self, strategy_config: StrategyConfig):
        super().__init__(strategy_config)
        
        # Momentum-specific parameters
        self.min_signal_strength = self.parameters.get('min_signal_strength', 0.6)
        self.momentum_lookback = self.parameters.get('momentum_lookback', 20)
        self.volume_threshold = self.parameters.get('volume_threshold', 1.5)  # 1.5x avg volume
        self.rsi_entry_threshold = self.parameters.get('rsi_entry_threshold', 50)
        self.macd_confirmation = self.parameters.get('macd_confirmation', True)
        
        logger.info(f"Initialized momentum strategy with parameters: {self.parameters}")
    
    def should_enter_trade(self, signal: TradingSignal, context: MarketContext) -> bool:
        """
        Determine if should enter trade based on momentum criteria.
        
        Entry Conditions:
        1. Signal strength above minimum threshold
        2. Price momentum in same direction as signal
        3. Volume confirmation (if available)
        4. RSI not in extreme overbought/oversold territory
        5. MACD confirmation (optional)
        """
        try:
            # Check signal strength
            if signal.strength < self.min_signal_strength:
                logger.debug(f"Signal strength {signal.strength} below threshold {self.min_signal_strength}")
                return False
            
            # Check signal confidence
            if signal.confidence < 0.5:
                logger.debug(f"Signal confidence {signal.confidence} too low")
                return False
            
            # Check if we have market data for additional analysis
            if not context.market_data:
                logger.debug("No market data available for momentum analysis")
                return signal.strength > 0.8  # Only very strong signals without market data
            
            # Check price momentum
            if not self._check_price_momentum(signal, context):
                logger.debug("Price momentum check failed")
                return False
            
            # Check RSI levels to avoid extreme conditions
            if not self._check_rsi_levels(signal, context):
                logger.debug("RSI levels check failed")
                return False
            
            # Optional MACD confirmation
            if self.macd_confirmation and not self._check_macd_confirmation(signal, context):
                logger.debug("MACD confirmation failed")
                return False
            
            logger.info(f"Momentum strategy entry conditions met for {signal.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error in momentum strategy entry check: {e}")
            return False
    
    def _check_price_momentum(self, signal: TradingSignal, context: MarketContext) -> bool:
        """Check if price momentum aligns with signal direction."""
        try:
            market_data = context.market_data
            if not market_data or not market_data.ohlc_1min:
                return True  # Skip check if no data
            
            # Get recent price data
            recent_ohlc = market_data.ohlc_1min[-self.momentum_lookback:]
            if len(recent_ohlc) < 5:
                return True  # Not enough data
            
            # Calculate price momentum (rate of change)
            start_price = recent_ohlc[0].close_price
            end_price = recent_ohlc[-1].close_price
            momentum = (end_price - start_price) / start_price
            
            # Check if momentum aligns with signal
            if signal.signal_type == SignalType.BUY:
                return momentum > 0.01  # At least 1% positive momentum
            elif signal.signal_type == SignalType.SELL:
                return momentum < -0.01  # At least 1% negative momentum
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking price momentum: {e}")
            return True  # Default to allow trade if check fails
    
    def _check_rsi_levels(self, signal: TradingSignal, context: MarketContext) -> bool:
        """Check RSI levels to avoid extreme overbought/oversold conditions."""
        try:
            # This would require access to current indicators
            # For now, we'll use a simplified check based on signal indicators
            if 'rsi' in signal.indicators_used:
                # If RSI was used in signal generation, trust the signal
                return True
            
            # Default check - avoid entering when RSI might be extreme
            # This is a placeholder - in practice, you'd get current RSI value
            return True
            
        except Exception as e:
            logger.error(f"Error checking RSI levels: {e}")
            return True
    
    def _check_macd_confirmation(self, signal: TradingSignal, context: MarketContext) -> bool:
        """Check MACD for momentum confirmation."""
        try:
            # Check if MACD was used in signal generation
            macd_indicators = ['macd_line', 'macd_signal', 'macd_histogram']
            if any(indicator in signal.indicators_used for indicator in macd_indicators):
                return True  # MACD already confirmed the signal
            
            # If MACD confirmation is required but not available, be more conservative
            return signal.strength > 0.8
            
        except Exception as e:
            logger.error(f"Error checking MACD confirmation: {e}")
            return True
    
    def calculate_position_size(self, context: MarketContext) -> int:
        """
        Calculate position size based on momentum strategy rules.
        
        Position Sizing Logic:
        - Base size on allocation percentage
        - Adjust based on signal strength
        - Consider available capital and current exposure
        - Apply volatility adjustment
        """
        try:
            # Base position size from allocation
            base_capital = context.available_capital * self.allocation
            position_capital = base_capital * Decimal(str(self.position_size_pct))
            
            # Calculate base quantity
            base_quantity = int(position_capital / context.current_price)
            
            if base_quantity <= 0:
                return 0
            
            # Adjust for signal strength (stronger signals get larger positions)
            # This would be based on the current signal being processed
            strength_multiplier = 1.0  # Default multiplier
            
            # Adjust for volatility (lower volatility allows larger positions)
            volatility_multiplier = self._calculate_volatility_adjustment(context)
            
            # Calculate final quantity
            final_quantity = int(base_quantity * strength_multiplier * volatility_multiplier)
            
            # Ensure minimum position size
            min_quantity = self.parameters.get('min_position_size', 1)
            final_quantity = max(final_quantity, min_quantity)
            
            # Ensure maximum position size
            max_quantity = self.parameters.get('max_position_size', base_quantity * 2)
            final_quantity = min(final_quantity, max_quantity)
            
            logger.debug(f"Calculated position size: {final_quantity} for {context.symbol}")
            return final_quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def _calculate_volatility_adjustment(self, context: MarketContext) -> float:
        """Calculate volatility adjustment factor for position sizing."""
        try:
            if not context.market_data or not context.market_data.ohlc_1min:
                return 1.0  # Default multiplier
            
            # Calculate recent volatility (simplified)
            recent_ohlc = context.market_data.ohlc_1min[-20:]  # Last 20 periods
            if len(recent_ohlc) < 10:
                return 1.0
            
            # Calculate price changes
            price_changes = []
            for i in range(1, len(recent_ohlc)):
                change = (recent_ohlc[i].close_price - recent_ohlc[i-1].close_price) / recent_ohlc[i-1].close_price
                price_changes.append(float(change))
            
            # Calculate standard deviation (volatility)
            if not price_changes:
                return 1.0
            
            mean_change = sum(price_changes) / len(price_changes)
            variance = sum((change - mean_change) ** 2 for change in price_changes) / len(price_changes)
            volatility = variance ** 0.5
            
            # Adjust position size inversely to volatility
            # Higher volatility = smaller positions
            if volatility > 0.02:  # High volatility (>2%)
                return 0.7
            elif volatility > 0.01:  # Medium volatility (1-2%)
                return 0.85
            else:  # Low volatility (<1%)
                return 1.2
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0
    
    def get_exit_conditions(self, context: MarketContext) -> List[ExitCondition]:
        """
        Get exit conditions for momentum strategy.
        
        Exit Conditions:
        1. Stop loss (risk management)
        2. Take profit (profit taking)
        3. Momentum reversal
        4. Time-based exit
        """
        conditions = []
        
        # Standard stop loss
        conditions.append(ExitCondition(
            condition_type="stop_loss",
            enabled=True
        ))
        
        # Take profit
        conditions.append(ExitCondition(
            condition_type="take_profit",
            enabled=True
        ))
        
        # Time-based exit (momentum strategies shouldn't hold too long)
        max_hold_hours = self.parameters.get('max_hold_hours', 8)  # 8 hours max
        conditions.append(ExitCondition(
            condition_type="time_based",
            enabled=True
        ))
        
        # Momentum reversal exit (would need additional implementation)
        conditions.append(ExitCondition(
            condition_type="momentum_reversal",
            enabled=self.parameters.get('momentum_reversal_exit', True)
        ))
        
        return conditions
    
    def should_exit_trade(self, position, context: MarketContext) -> bool:
        """
        Enhanced exit logic for momentum strategy.
        Includes momentum reversal detection.
        """
        # Check standard exit conditions first
        if super().should_exit_trade(position, context):
            return True
        
        # Check for momentum reversal
        if self._check_momentum_reversal(position, context):
            logger.info(f"Momentum reversal detected for {position.symbol}")
            return True
        
        return False
    
    def _check_momentum_reversal(self, position, context: MarketContext) -> bool:
        """Check if momentum has reversed."""
        try:
            if not context.market_data or not context.market_data.ohlc_1min:
                return False
            
            recent_ohlc = context.market_data.ohlc_1min[-10:]  # Last 10 periods
            if len(recent_ohlc) < 5:
                return False
            
            # Calculate recent momentum
            start_price = recent_ohlc[0].close_price
            end_price = recent_ohlc[-1].close_price
            recent_momentum = (end_price - start_price) / start_price
            
            # Check if momentum has reversed relative to position direction
            if position.is_long and recent_momentum < -0.005:  # 0.5% negative momentum
                return True
            elif position.is_short and recent_momentum > 0.005:  # 0.5% positive momentum
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking momentum reversal: {e}")
            return False