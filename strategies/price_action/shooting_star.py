"""
Shooting Star Strategy
Price action strategy based on shooting star candlestick patterns indicating bearish reversal.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class ShootingStarStrategy(BaseStrategy):
    """
    Shooting Star Strategy.
    
    Entry: When shooting star pattern forms at resistance levels with volume confirmation
    Exit: When pattern fails or target reached
    """
    
    @property
    def name(self) -> str:
        return "Shooting Star"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'min_upper_wick_ratio': 2.0,  # Minimum upper wick to body ratio
            'max_lower_wick_ratio': 0.5,  # Maximum lower wick to upper wick ratio
            'max_body_pct': 25,           # Maximum body as % of total range
            'min_range_atr': 0.7,         # Minimum range as multiple of ATR
            'volume_threshold': 1.4,
            'min_volume': 5000,
            'atr_period': 14,
            'resistance_period': 20,
            'level_tolerance': 0.3,       # Tolerance for resistance level proximity
            'stop_loss_pct': 1.0,
            'target_pct': 2.5,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate shooting star indicators."""
        df = df.copy()
        
        # Calculate ATR
        atr_period = self.get_param('atr_period', 14)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
        
        # Candlestick components
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Body percentage of total range
        df['body_pct'] = (df['body_size'] / df['total_range'] * 100).fillna(0)
        
        # Wick ratios
        df['upper_wick_ratio'] = (df['upper_wick'] / df['body_size']).replace([np.inf, -np.inf], 0)
        df['lower_to_upper_ratio'] = (df['lower_wick'] / df['upper_wick']).replace([np.inf, -np.inf], 0)
        
        # Shooting star detection
        min_upper_wick_ratio = self.get_param('min_upper_wick_ratio', 2.0)
        max_lower_wick_ratio = self.get_param('max_lower_wick_ratio', 0.5)
        max_body_pct = self.get_param('max_body_pct', 25)
        min_range_atr = self.get_param('min_range_atr', 0.7)
        
        df['is_shooting_star'] = (
            (df['upper_wick_ratio'] >= min_upper_wick_ratio) &
            (df['lower_to_upper_ratio'] <= max_lower_wick_ratio) &
            (df['body_pct'] <= max_body_pct) &
            (df['total_range'] >= df['atr'] * min_range_atr) &
            (df['upper_wick'] > 0)  # Must have an upper wick
        )
        
        # Shooting star types
        df['bearish_shooting_star'] = (
            df['is_shooting_star'] &
            (df['close'] <= df['open'])  # Bearish or neutral body
        )
        
        df['inverted_hammer'] = (
            df['is_shooting_star'] &
            (df['close'] > df['open'])  # Bullish body (inverted hammer in downtrend)
        )
        
        # Resistance levels
        resistance_period = self.get_param('resistance_period', 20)
        df['resistance'] = df['high'].rolling(window=resistance_period).max()
        df['resistance_ma'] = df['high'].rolling(window=resistance_period).mean()
        
        # Resistance proximity
        level_tolerance = self.get_param('level_tolerance', 0.3) / 100
        
        df['near_resistance'] = (
            (abs(df['high'] - df['resistance']) / df['resistance'] <= level_tolerance) |
            (abs(df['high'] - df['resistance_ma']) / df['resistance_ma'] <= level_tolerance)
        )
        
        # Uptrend context (shooting star should appear after advance)
        df['ema_short'] = df['close'].ewm(span=10).mean()
        df['ema_long'] = df['close'].ewm(span=20).mean()
        df['in_uptrend'] = df['ema_short'] > df['ema_long']
        
        # Recent advance (price should be rising before shooting star)
        df['recent_advance'] = (
            (df['close'] > df['close'].shift(3)) &
            (df['close'].shift(1) > df['close'].shift(4))
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.4))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Valid shooting star signals
        df['valid_shooting_star'] = (
            df['bearish_shooting_star'] &
            df['near_resistance'] &
            (df['in_uptrend'] | df['recent_advance']) &
            df['volume_spike']
        )
        
        # Shooting star strength
        df['shooting_star_strength'] = 0.0
        
        for i in range(len(df)):
            if df['valid_shooting_star'].iloc[i]:
                # Base strength from upper wick ratio
                wick_ratio = df['upper_wick_ratio'].iloc[i]
                strength = min(wick_ratio / 4, 1.0)  # Normalize to 0-1
                
                # Boost for smaller body
                body_factor = 1 - (df['body_pct'].iloc[i] / 100)
                strength *= (1 + body_factor * 0.4)
                
                # Boost for bearish close
                if df['close'].iloc[i] <= df['open'].iloc[i]:
                    strength *= 1.2
                
                # Boost for volume
                if df['volume_spike'].iloc[i]:
                    strength *= 1.3
                
                # Boost for range size
                if not np.isnan(df['atr'].iloc[i]):
                    range_factor = df['total_range'].iloc[i] / df['atr'].iloc[i]
                    if range_factor > 1.5:
                        strength *= 1.1
                
                # Boost for stronger uptrend context
                if df['in_uptrend'].iloc[i] and df['recent_advance'].iloc[i]:
                    strength *= 1.2
                
                df.loc[df.index[i], 'shooting_star_strength'] = min(strength, 1.0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate shooting star signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Time filter
            if not row['time_filter']:
                continue
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 5000):
                continue
            
            # Valid shooting star signal
            if (row['valid_shooting_star'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Shooting star at resistance (Wick ratio: {row['upper_wick_ratio']:.1f}, Strength: {row['shooting_star_strength']:.2f})",
                    confidence=0.7 + (row['shooting_star_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check shooting star entry conditions."""
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
        if not current['volume_spike']:
            return False, "No volume confirmation"
        
        # Valid shooting star
        if current['valid_shooting_star']:
            return True, f"Shooting star at resistance (wick ratio: {current['upper_wick_ratio']:.1f})"
        
        return False, "No valid shooting star signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check shooting star exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Shooting star invalidation - price broke the shooting star high
        if current_idx > 0:
            entry_bar = df_with_indicators.iloc[current_idx - 1] if current_idx > 0 else current
            
            if current['close'] > entry_bar['high']:
                return True, "Shooting star invalidated - broke shooting star high"
        
        # Inverted hammer pattern (bullish reversal after downtrend)
        if (current['inverted_hammer'] and 
            current['close'] < entry_price and
            current['ema_short'] < current['ema_long']):
            return True, "Inverted hammer pattern - potential reversal"
        
        # Time-based exit (after 2 hours)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 2:
            return True, "Time-based exit (2 hours)"
        
        return False, "Hold shooting star position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use shooting star high as stop loss."""
        current = df.iloc[current_idx]
        
        # Use the shooting star's high as stop loss
        if current['valid_shooting_star']:
            return current['high']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 + stop_pct)  # Stop above entry for short
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on shooting star size and support levels."""
        current = df.iloc[current_idx]
        
        # Target based on shooting star range
        star_range = current['total_range']
        
        # Primary target: 2x shooting star range below entry
        primary_target = entry_price - (star_range * 2)
        
        # Check for nearby support
        if current_idx >= 20:
            recent_data = df.iloc[current_idx-20:current_idx]
            support = recent_data['low'].min()
            
            # Use closer target if support is nearby
            if support > primary_target and support < entry_price * 0.99:
                return support * 1.01  # Just above support
        
        return primary_target