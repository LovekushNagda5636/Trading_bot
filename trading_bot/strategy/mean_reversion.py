"""
Mean reversion trading strategy implementation.
Trades based on the assumption that prices will revert to their mean.
"""

from decimal import Decimal
from typing import List, Optional
import structlog

from .base import Strategy, MarketContext, ExitCondition
from ..core.models import TradingSignal, SignalType, OrderSide
from ..core.config import StrategyConfig

logger = structlog.get_logger(__name__)


class MeanReversionStrategy(Strategy):
    """
    Mean reversion trading strategy.
    
    Strategy Logic:
    - Enter positions when price deviates significantly from mean
    - Expect price to revert back to the mean
    - Use Bollinger Bands, RSI, and statistical measures
    - Exit when price approaches mean or risk limits are hit
    """
    
    def __init__(self, strategy_config: StrategyConfig):
        super().__init__(strategy_config)
        
        # Mean reversion specific parameters
        self.deviation_threshold = self.parameters.get('deviation_threshold', 2.0)  # Standard deviations
        self.rsi_oversold = self.parameters.get('rsi_oversold', 30)
        self.rsi_overbought = self.parameters.get('rsi_overbought', 70)
        self.mean_lookback = self.parameters.get('mean_lookback', 20)
        self.min_reversion_target = self.parameters.get('min_reversion_target', 0.01)  # 1% minimum target
        self.max_deviation_entry = self.parameters.get('max_deviation_entry', 3.0)  # Don't enter if too extreme
        
        logger.info(f"Initialized mean reversion strategy with parameters: {self.parameters}")
    
    def should_enter_trade(self, signal: TradingSignal, context: MarketContext) -> bool:
        """
        Determine if should enter trade based on mean reversion criteria.
        
        Entry Conditions:
        1. Price is significantly deviated from mean (but not too extreme)
        2. RSI indicates oversold (for buy) or overbought (for sell)
        3. Bollinger Bands confirm deviation
        4. Signal strength meets minimum threshold
        """
        try:
            # Check signal strength
            if signal.strength < 0.5:  # Lower threshold for mean reversion
                logger.debug(f"Signal strength {signal.strength} below threshold")
                return False
            
            # Mean reversion works best with RSI and Bollinger Bands signals
            if not self._has_mean_reversion_indicators(signal):
                logger.debug("Signal doesn't have mean reversion indicators")
                return False
            
            # Check if price deviation is in acceptable range
            if not self._check_price_deviation(signal, context):
                logger.debug("Price deviation check failed")
                return False
            
            # Check RSI levels for mean reversion opportunities
            if not self._check_rsi_mean_reversion(signal, context):
                logger.debug("RSI mean reversion check failed")
                return False
            
            # Check Bollinger Bands for deviation confirmation
            if not self._check_bollinger_deviation(signal, context):
                logger.debug("Bollinger Bands deviation check failed")
                return False
            
            logger.info(f"Mean reversion strategy entry conditions met for {signal.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error in mean reversion strategy entry check: {e}")
            return False
    
    def _has_mean_reversion_indicators(self, signal: TradingSignal) -> bool:
        """Check if signal has indicators suitable for mean reversion."""
        mean_reversion_indicators = ['rsi', 'bb_upper', 'bb_lower', 'bb_middle']
        return any(indicator in signal.indicators_used for indicator in mean_reversion_indicators)
    
    def _check_price_deviation(self, signal: TradingSignal, context: MarketContext) -> bool:
        """Check if price deviation is suitable for mean reversion."""
        try:
            if not context.market_data or not context.market_data.ohlc_1min:
                return True  # Skip check if no data
            
            # Calculate mean and standard deviation
            recent_ohlc = context.market_data.ohlc_1min[-self.mean_lookback:]
            if len(recent_ohlc) < 10:
                return True  # Not enough data
            
            prices = [float(ohlc.close_price) for ohlc in recent_ohlc]
            mean_price = sum(prices) / len(prices)
            variance = sum((price - mean_price) ** 2 for price in prices) / len(prices)
            std_dev = variance ** 0.5
            
            current_price = float(context.current_price)
            deviation = abs(current_price - mean_price) / std_dev
            
            # Check if deviation is in acceptable range
            if deviation < 1.0:  # Too close to mean
                return False
            if deviation > self.max_deviation_entry:  # Too extreme
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking price deviation: {e}")
            return True
    
    def _check_rsi_mean_reversion(self, signal: TradingSignal, context: MarketContext) -> bool:
        """Check RSI for mean reversion opportunities."""
        try:
            # If RSI was used in signal generation, check the signal type
            if 'rsi' in signal.indicators_used:
                if signal.signal_type == SignalType.BUY:
                    # For buy signals, we want RSI to be oversold
                    return True  # Trust the signal generation logic
                elif signal.signal_type == SignalType.SELL:
                    # For sell signals, we want RSI to be overbought
                    return True  # Trust the signal generation logic
            
            # If no RSI in signal, be more conservative
            return signal.strength > 0.7
            
        except Exception as e:
            logger.error(f"Error checking RSI mean reversion: {e}")
            return True
    
    def _check_bollinger_deviation(self, signal: TradingSignal, context: MarketContext) -> bool:
        """Check Bollinger Bands for price deviation."""
        try:
            # If Bollinger Bands were used in signal generation
            bb_indicators = ['bb_upper', 'bb_lower', 'bb_middle']
            if any(indicator in signal.indicators_used for indicator in bb_indicators):
                # Trust the signal generation logic
                return True
            
            # If no Bollinger Bands data, use other criteria
            return signal.strength > 0.6
            
        except Exception as e:
            logger.error(f"Error checking Bollinger deviation: {e}")
            return True
    
    def calculate_position_size(self, context: MarketContext) -> int:
        """
        Calculate position size for mean reversion strategy.
        
        Position Sizing Logic:
        - Smaller positions due to higher risk of mean reversion strategies
        - Adjust based on deviation magnitude
        - Consider volatility and available capital
        """
        try:
            # Base position size (smaller than momentum strategy)
            base_capital = context.available_capital * self.allocation
            position_capital = base_capital * Decimal(str(self.position_size_pct * 0.8))  # 80% of normal size
            
            # Calculate base quantity
            base_quantity = int(position_capital / context.current_price)
            
            if base_quantity <= 0:
                return 0
            
            # Adjust for deviation magnitude
            deviation_multiplier = self._calculate_deviation_adjustment(context)
            
            # Adjust for volatility (mean reversion works better in lower volatility)
            volatility_multiplier = self._calculate_volatility_adjustment(context)
            
            # Calculate final quantity
            final_quantity = int(base_quantity * deviation_multiplier * volatility_multiplier)
            
            # Ensure minimum position size
            min_quantity = self.parameters.get('min_position_size', 1)
            final_quantity = max(final_quantity, min_quantity)
            
            # Ensure maximum position size (conservative for mean reversion)
            max_quantity = self.parameters.get('max_position_size', base_quantity)
            final_quantity = min(final_quantity, max_quantity)
            
            logger.debug(f"Calculated mean reversion position size: {final_quantity} for {context.symbol}")
            return final_quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def _calculate_deviation_adjustment(self, context: MarketContext) -> float:
        """Calculate adjustment based on price deviation from mean."""
        try:
            if not context.market_data or not context.market_data.ohlc_1min:
                return 1.0
            
            recent_ohlc = context.market_data.ohlc_1min[-self.mean_lookback:]
            if len(recent_ohlc) < 10:
                return 1.0
            
            prices = [float(ohlc.close_price) for ohlc in recent_ohlc]
            mean_price = sum(prices) / len(prices)
            variance = sum((price - mean_price) ** 2 for price in prices) / len(prices)
            std_dev = variance ** 0.5
            
            current_price = float(context.current_price)
            deviation = abs(current_price - mean_price) / std_dev
            
            # Larger positions for larger deviations (up to a point)
            if deviation > 2.5:
                return 1.3  # 30% larger position
            elif deviation > 2.0:
                return 1.2  # 20% larger position
            elif deviation > 1.5:
                return 1.1  # 10% larger position
            else:
                return 1.0  # Normal position
            
        except Exception as e:
            logger.error(f"Error calculating deviation adjustment: {e}")
            return 1.0
    
    def _calculate_volatility_adjustment(self, context: MarketContext) -> float:
        """Calculate volatility adjustment for mean reversion."""
        try:
            if not context.market_data or not context.market_data.ohlc_1min:
                return 1.0
            
            recent_ohlc = context.market_data.ohlc_1min[-20:]
            if len(recent_ohlc) < 10:
                return 1.0
            
            # Calculate volatility
            price_changes = []
            for i in range(1, len(recent_ohlc)):
                change = (recent_ohlc[i].close_price - recent_ohlc[i-1].close_price) / recent_ohlc[i-1].close_price
                price_changes.append(float(change))
            
            if not price_changes:
                return 1.0
            
            mean_change = sum(price_changes) / len(price_changes)
            variance = sum((change - mean_change) ** 2 for change in price_changes) / len(price_changes)
            volatility = variance ** 0.5
            
            # Mean reversion works better in lower volatility environments
            if volatility > 0.03:  # Very high volatility (>3%)
                return 0.5  # Much smaller positions
            elif volatility > 0.02:  # High volatility (2-3%)
                return 0.7  # Smaller positions
            elif volatility > 0.01:  # Medium volatility (1-2%)
                return 1.0  # Normal positions
            else:  # Low volatility (<1%)
                return 1.2  # Slightly larger positions
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0
    
    def get_exit_conditions(self, context: MarketContext) -> List[ExitCondition]:
        """
        Get exit conditions for mean reversion strategy.
        
        Exit Conditions:
        1. Price reverts to mean (profit target)
        2. Stop loss (risk management)
        3. Time-based exit (mean reversion should happen quickly)
        4. Trend continuation (stop loss if trend continues against us)
        """
        conditions = []
        
        # Mean reversion target (custom exit condition)
        conditions.append(ExitCondition(
            condition_type="mean_reversion",
            enabled=True
        ))
        
        # Tighter stop loss for mean reversion
        conditions.append(ExitCondition(
            condition_type="stop_loss",
            enabled=True
        ))
        
        # Shorter time-based exit (mean reversion should happen quickly)
        conditions.append(ExitCondition(
            condition_type="time_based",
            enabled=True
        ))
        
        # Trend continuation exit
        conditions.append(ExitCondition(
            condition_type="trend_continuation",
            enabled=self.parameters.get('trend_continuation_exit', True)
        ))
        
        return conditions
    
    def should_exit_trade(self, position, context: MarketContext) -> bool:
        """
        Enhanced exit logic for mean reversion strategy.
        """
        # Check standard exit conditions first
        if super().should_exit_trade(position, context):
            return True
        
        # Check for mean reversion (price returning to mean)
        if self._check_mean_reversion_exit(position, context):
            logger.info(f"Mean reversion target reached for {position.symbol}")
            return True
        
        # Check for trend continuation (against our position)
        if self._check_trend_continuation(position, context):
            logger.info(f"Trend continuation detected for {position.symbol}")
            return True
        
        return False
    
    def _check_mean_reversion_exit(self, position, context: MarketContext) -> bool:
        """Check if price has reverted to mean."""
        try:
            if not context.market_data or not context.market_data.ohlc_1min:
                return False
            
            recent_ohlc = context.market_data.ohlc_1min[-self.mean_lookback:]
            if len(recent_ohlc) < 10:
                return False
            
            # Calculate mean price
            prices = [float(ohlc.close_price) for ohlc in recent_ohlc]
            mean_price = sum(prices) / len(prices)
            current_price = float(context.current_price)
            
            # Check if we're close to mean and have some profit
            distance_to_mean = abs(current_price - mean_price) / mean_price
            
            if distance_to_mean < 0.005:  # Within 0.5% of mean
                # Check if we have minimum profit
                if position.is_long:
                    profit_pct = (current_price - float(position.average_price)) / float(position.average_price)
                else:
                    profit_pct = (float(position.average_price) - current_price) / float(position.average_price)
                
                return profit_pct > self.min_reversion_target
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking mean reversion exit: {e}")
            return False
    
    def _check_trend_continuation(self, position, context: MarketContext) -> bool:
        """Check if trend is continuing against our position."""
        try:
            if not context.market_data or not context.market_data.ohlc_1min:
                return False
            
            recent_ohlc = context.market_data.ohlc_1min[-10:]  # Last 10 periods
            if len(recent_ohlc) < 5:
                return False
            
            # Calculate trend strength
            start_price = recent_ohlc[0].close_price
            end_price = recent_ohlc[-1].close_price
            trend = (end_price - start_price) / start_price
            
            # Check if trend is strongly against our position
            trend_threshold = 0.01  # 1% trend
            
            if position.is_long and trend < -trend_threshold:
                return True  # Strong downtrend against long position
            elif position.is_short and trend > trend_threshold:
                return True  # Strong uptrend against short position
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking trend continuation: {e}")
            return False