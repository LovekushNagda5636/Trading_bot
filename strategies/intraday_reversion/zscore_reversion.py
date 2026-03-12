"""
Z-Score Mean Reversion Strategy
Statistical mean reversion strategy using Z-Score for entry signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class ZScoreReversionStrategy(BaseStrategy):
    """
    Z-Score Mean Reversion Strategy.
    
    Entry: When price Z-Score reaches extreme levels indicating mean reversion opportunity
    Exit: When Z-Score returns to neutral or continues extreme
    """
    
    @property
    def name(self) -> str:
        return "Z-Score Reversion"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'lookback_period': 20,
            'zscore_entry_threshold': 2.0,
            'zscore_extreme_threshold': 2.5,
            'zscore_exit_threshold': 0.5,
            'volume_threshold': 1.3,
            'min_volume': 6000,
            'stop_loss_pct': 1.5,
            'target_zscore': 0.0,  # Target return to mean (Z-Score = 0)
            'bollinger_confirmation': True,
            'time_filter_start': "10:00",
            'time_filter_end': "14:30"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Z-Score reversion indicators."""
        df = df.copy()
        
        lookback_period = self.get_param('lookback_period', 20)
        
        # Calculate rolling mean and standard deviation
        df['price_mean'] = df['close'].rolling(window=lookback_period).mean()
        df['price_std'] = df['close'].rolling(window=lookback_period).std()
        
        # Calculate Z-Score
        df['zscore'] = (df['close'] - df['price_mean']) / df['price_std']
        
        # Z-Score thresholds
        entry_threshold = self.get_param('zscore_entry_threshold', 2.0)
        extreme_threshold = self.get_param('zscore_extreme_threshold', 2.5)
        exit_threshold = self.get_param('zscore_exit_threshold', 0.5)
        
        df['zscore_oversold'] = df['zscore'] < -entry_threshold
        df['zscore_overbought'] = df['zscore'] > entry_threshold
        df['zscore_extreme_oversold'] = df['zscore'] < -extreme_threshold
        df['zscore_extreme_overbought'] = df['zscore'] > extreme_threshold
        df['zscore_neutral'] = abs(df['zscore']) < exit_threshold
        
        # Z-Score momentum
        df['zscore_momentum'] = df['zscore'] - df['zscore'].shift(1)
        df['zscore_reversing_up'] = (df['zscore_momentum'] > 0) & (df['zscore_momentum'].shift(1) <= 0)
        df['zscore_reversing_down'] = (df['zscore_momentum'] < 0) & (df['zscore_momentum'].shift(1) >= 0)
        
        # Mean reversion signals
        df['zscore_bullish_reversion'] = (
            df['zscore_oversold'] & 
            df['zscore_reversing_up'] &
            (df['close'] > df['close'].shift(1))
        )
        
        df['zscore_bearish_reversion'] = (
            df['zscore_overbought'] & 
            df['zscore_reversing_down'] &
            (df['close'] < df['close'].shift(1))
        )
        
        # Bollinger Bands confirmation (optional)
        if self.get_param('bollinger_confirmation', True):
            bb_period = lookback_period
            bb_std = 2.0
            df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            df['bb_std_dev'] = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * df['bb_std_dev'])
            df['bb_lower'] = df['bb_middle'] - (bb_std * df['bb_std_dev'])
            
            df['bb_oversold'] = df['close'] < df['bb_lower']
            df['bb_overbought'] = df['close'] > df['bb_upper']
            
            # Combine Z-Score with Bollinger confirmation
            df['zscore_bullish_reversion'] &= df['bb_oversold']
            df['zscore_bearish_reversion'] &= df['bb_overbought']
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.3))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "10:00")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "14:30")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Z-Score strength (absolute value)
        df['zscore_strength'] = abs(df['zscore'])
        
        # Distance to mean (for target calculation)
        df['distance_to_mean'] = abs(df['close'] - df['price_mean'])
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate Z-Score reversion signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Time filter
            if not row['time_filter']:
                continue
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 6000):
                continue
            
            # Skip if insufficient data
            if pd.isna(row['zscore']) or pd.isna(row['price_std']) or row['price_std'] == 0:
                continue
            
            # Bullish Z-Score reversion signal
            if (row['zscore_bullish_reversion'] and self.is_market_open(row.name)):
                
                # Higher confidence for extreme Z-Scores
                confidence = 0.85 if row['zscore_extreme_oversold'] else 0.75
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Z-Score bullish reversion (Z: {row['zscore']:.2f}, Mean: {row['price_mean']:.2f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish Z-Score reversion signal
            elif (row['zscore_bearish_reversion'] and self.is_market_open(row.name)):
                
                # Higher confidence for extreme Z-Scores
                confidence = 0.85 if row['zscore_extreme_overbought'] else 0.75
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Z-Score bearish reversion (Z: {row['zscore']:.2f}, Mean: {row['price_mean']:.2f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check Z-Score reversion entry conditions."""
        if current_idx < 25:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Time filter
        if not current['time_filter']:
            return False, "Outside trading hours"
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if current['volume'] < self.get_param('min_volume', 6000):
            return False, "Low volume"
        
        # Data quality check
        if pd.isna(current['zscore']) or pd.isna(current['price_std']) or current['price_std'] == 0:
            return False, "Invalid Z-Score data"
        
        # Bullish reversion
        if current['zscore_bullish_reversion']:
            return True, f"Z-Score bullish reversion (Z: {current['zscore']:.2f})"
        
        # Bearish reversion
        if current['zscore_bearish_reversion']:
            return True, f"Z-Score bearish reversion (Z: {current['zscore']:.2f})"
        
        return False, "No Z-Score reversion signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check Z-Score reversion exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Z-Score returned to neutral zone
        if current['zscore_neutral']:
            return True, f"Z-Score returned to neutral (Z: {current['zscore']:.2f})"
        
        # Z-Score moved to opposite extreme (failed reversion)
        if (entry_price < current['close'] and current['zscore_overbought']):
            return True, f"Z-Score moved to overbought (Z: {current['zscore']:.2f})"
        
        if (entry_price > current['close'] and current['zscore_oversold']):
            return True, f"Z-Score moved to oversold (Z: {current['zscore']:.2f})"
        
        # Target reached (close to mean)
        if not pd.isna(current['price_mean']):
            distance_to_mean = abs(current['close'] - current['price_mean'])
            entry_distance = abs(entry_price - current['price_mean'])
            
            # Exit when 80% of the way back to mean
            if distance_to_mean <= entry_distance * 0.2:
                return True, "Target reached (80% return to mean)"
        
        # Time-based exit (after 1 hour)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 1:
            return True, "Time-based exit (1 hour)"
        
        return False, "Hold Z-Score reversion position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate stop loss based on statistical levels."""
        current = df.iloc[current_idx]
        
        # Use 3 standard deviations as stop loss
        if not pd.isna(current['price_mean']) and not pd.isna(current['price_std']) and current['price_std'] > 0:
            if current['zscore'] < 0:  # Long position (price below mean)
                # Stop at 3 standard deviations below mean
                return current['price_mean'] - (3 * current['price_std'])
            else:  # Short position (price above mean)
                # Stop at 3 standard deviations above mean
                return current['price_mean'] + (3 * current['price_std'])
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.5) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target as return toward mean."""
        current = df.iloc[current_idx]
        
        # Target is partial return to mean
        if not pd.isna(current['price_mean']):
            target_zscore = self.get_param('target_zscore', 0.0)
            
            if not pd.isna(current['price_std']) and current['price_std'] > 0:
                # Target price based on target Z-Score
                target_price = current['price_mean'] + (target_zscore * current['price_std'])
                
                # Ensure target is in the right direction
                if current['zscore'] < 0:  # Long position
                    return max(target_price, entry_price * 1.01)  # At least 1% profit
                else:  # Short position
                    return min(target_price, entry_price * 0.99)  # At least 1% profit
            else:
                # Simple return to mean
                return current['price_mean']
        
        # Fallback target
        return entry_price * 1.015