"""
Hammer Strategy
Price action strategy based on hammer candlestick patterns indicating bullish reversal.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class HammerStrategy(BaseStrategy):
    """
    Hammer Strategy.
    
    Entry: When hammer pattern forms at support levels with volume confirmation
    Exit: When pattern fails or target reached
    """
    
    @property
    def name(self) -> str:
        return "Hammer"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'min_lower_wick_ratio': 2.0,  # Minimum lower wick to body ratio
            'max_upper_wick_ratio': 0.5,  # Maximum upper wick to lower wick ratio
            'max_body_pct': 25,           # Maximum body as % of total range
            'min_range_atr': 0.7,         # Minimum range as multiple of ATR
            'volume_threshold': 1.4,
            'min_volume': 5000,
            'atr_period': 14,
            'support_period': 20,
            'level_tolerance': 0.3,       # Tolerance for support level proximity
            'stop_loss_pct': 1.0,
            'target_pct': 2.5,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate hammer indicators."""
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
        df['lower_wick_ratio'] = (df['lower_wick'] / df['body_size']).replace([np.inf, -np.inf], 0)
        df['upper_to_lower_ratio'] = (df['upper_wick'] / df['lower_wick']).replace([np.inf, -np.inf], 0)
        
        # Hammer detection
        min_lower_wick_ratio = self.get_param('min_lower_wick_ratio', 2.0)
        max_upper_wick_ratio = self.get_param('max_upper_wick_ratio', 0.5)
        max_body_pct = self.get_param('max_body_pct', 25)
        min_range_atr = self.get_param('min_range_atr', 0.7)
        
        df['is_hammer'] = (
            (df['lower_wick_ratio'] >= min_lower_wick_ratio) &
            (df['upper_to_lower_ratio'] <= max_upper_wick_ratio) &
            (df['body_pct'] <= max_body_pct) &
            (df['total_range'] >= df['atr'] * min_range_atr) &
            (df['lower_wick'] > 0)  # Must have a lower wick
        )
        
        # Hammer types
        df['bullish_hammer'] = (
            df['is_hammer'] &
            (df['close'] >= df['open'])  # Bullish or neutral body
        )
        
        df['hanging_man'] = (
            df['is_hammer'] &
            (df['close'] < df['open'])  # Bearish body (hanging man in uptrend)
        )
        
        # Support levels
        support_period = self.get_param('support_period', 20)
        df['support'] = df['low'].rolling(window=support_period).min()
        df['support_ma'] = df['low'].rolling(window=support_period).mean()
        
        # Support proximity
        level_tolerance = self.get_param('level_tolerance', 0.3) / 100
        
        df['near_support'] = (
            (abs(df['low'] - df['support']) / df['support'] <= level_tolerance) |
            (abs(df['low'] - df['support_ma']) / df['support_ma'] <= level_tolerance)
        )
        
        # Downtrend context (hammer should appear after decline)
        df['ema_short'] = df['close'].ewm(span=10).mean()
        df['ema_long'] = df['close'].ewm(span=20).mean()
        df['in_downtrend'] = df['ema_short'] < df['ema_long']
        
        # Recent decline (price should be declining before hammer)
        df['recent_decline'] = (
            (df['close'] < df['close'].shift(3)) &
            (df['close'].shift(1) < df['close'].shift(4))
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.4))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Valid hammer signals
        df['valid_hammer'] = (
            df['bullish_hammer'] &
            df['near_support'] &
            (df['in_downtrend'] | df['recent_decline']) &
            df['volume_spike']
        )
        
        # Hammer strength
        df['hammer_strength'] = 0.0
        
        for i in range(len(df)):
            if df['valid_hammer'].iloc[i]:
                # Base strength from lower wick ratio
                wick_ratio = df['lower_wick_ratio'].iloc[i]
                strength = min(wick_ratio / 4, 1.0)  # Normalize to 0-1
                
                # Boost for smaller body
                body_factor = 1 - (df['body_pct'].iloc[i] / 100)
                strength *= (1 + body_factor * 0.4)
                
                # Boost for bullish close
                if df['close'].iloc[i] >= df['open'].iloc[i]:
                    strength *= 1.2
                
                # Boost for volume
                if df['volume_spike'].iloc[i]:
                    strength *= 1.3
                
                # Boost for range size
                if not np.isnan(df['atr'].iloc[i]):
                    range_factor = df['total_range'].iloc[i] / df['atr'].iloc[i]
                    if range_factor > 1.5:
                        strength *= 1.1
                
                # Boost for stronger downtrend context
                if df['in_downtrend'].iloc[i] and df['recent_decline'].iloc[i]:
                    strength *= 1.2
                
                df.loc[df.index[i], 'hammer_strength'] = min(strength, 1.0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate hammer signals."""
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
            
            # Valid hammer signal
            if (row['valid_hammer'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Hammer at support (Wick ratio: {row['lower_wick_ratio']:.1f}, Strength: {row['hammer_strength']:.2f})",
                    confidence=0.7 + (row['hammer_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check hammer entry conditions."""
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
        
        # Valid hammer
        if current['valid_hammer']:
            return True, f"Hammer at support (wick ratio: {current['lower_wick_ratio']:.1f})"
        
        return False, "No valid hammer signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check hammer exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Hammer invalidation - price broke the hammer low
        if current_idx > 0:
            entry_bar = df_with_indicators.iloc[current_idx - 1] if current_idx > 0 else current
            
            if current['close'] < entry_bar['low']:
                return True, "Hammer invalidated - broke hammer low"
        
        # Hanging man pattern (bearish reversal after uptrend)
        if (current['hanging_man'] and 
            current['close'] > entry_price and
            current['ema_short'] > current['ema_long']):
            return True, "Hanging man pattern - potential reversal"
        
        # Time-based exit (after 2 hours)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 2:
            return True, "Time-based exit (2 hours)"
        
        return False, "Hold hammer position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use hammer low as stop loss."""
        current = df.iloc[current_idx]
        
        # Use the hammer's low as stop loss
        if current['valid_hammer']:
            return current['low']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on hammer size and resistance levels."""
        current = df.iloc[current_idx]
        
        # Target based on hammer range
        hammer_range = current['total_range']
        
        # Primary target: 2x hammer range above entry
        primary_target = entry_price + (hammer_range * 2)
        
        # Check for nearby resistance
        if current_idx >= 20:
            recent_data = df.iloc[current_idx-20:current_idx]
            resistance = recent_data['high'].max()
            
            # Use closer target if resistance is nearby
            if resistance < primary_target and resistance > entry_price * 1.01:
                return resistance * 0.99  # Just below resistance
        
        return primary_target