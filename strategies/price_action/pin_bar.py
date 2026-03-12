"""
Pin Bar Strategy
Price action strategy based on pin bar (hammer/shooting star) candlestick patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class PinBarStrategy(BaseStrategy):
    """
    Pin Bar Strategy.
    
    Entry: When pin bar pattern forms at key levels with volume confirmation
    Exit: When pattern fails or target reached
    """
    
    @property
    def name(self) -> str:
        return "Pin Bar"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'min_wick_ratio': 2.0,  # Minimum wick to body ratio
            'max_body_pct': 30,     # Maximum body as % of total range
            'min_range_atr': 0.8,   # Minimum range as multiple of ATR
            'volume_threshold': 1.3,
            'min_volume': 5000,
            'atr_period': 14,
            'support_resistance_period': 20,
            'level_tolerance': 0.2,  # Tolerance for key level proximity
            'stop_loss_pct': 1.0,
            'target_pct': 2.5,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pin bar indicators."""
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
        df['lower_wick_ratio'] = (df['lower_wick'] / df['body_size']).replace([np.inf, -np.inf], 0)
        
        # Pin bar detection
        min_wick_ratio = self.get_param('min_wick_ratio', 2.0)
        max_body_pct = self.get_param('max_body_pct', 30)
        min_range_atr = self.get_param('min_range_atr', 0.8)
        
        # Bullish pin bar (hammer) - long lower wick, small body, small upper wick
        df['bullish_pin_bar'] = (
            (df['lower_wick_ratio'] >= min_wick_ratio) &
            (df['body_pct'] <= max_body_pct) &
            (df['upper_wick'] <= df['lower_wick'] * 0.5) &  # Upper wick should be small
            (df['total_range'] >= df['atr'] * min_range_atr)
        )
        
        # Bearish pin bar (shooting star) - long upper wick, small body, small lower wick
        df['bearish_pin_bar'] = (
            (df['upper_wick_ratio'] >= min_wick_ratio) &
            (df['body_pct'] <= max_body_pct) &
            (df['lower_wick'] <= df['upper_wick'] * 0.5) &  # Lower wick should be small
            (df['total_range'] >= df['atr'] * min_range_atr)
        )
        
        # Support and resistance levels
        sr_period = self.get_param('support_resistance_period', 20)
        df['resistance'] = df['high'].rolling(window=sr_period).max()
        df['support'] = df['low'].rolling(window=sr_period).min()
        
        # Key level proximity
        level_tolerance = self.get_param('level_tolerance', 0.2) / 100
        
        df['near_support'] = (
            abs(df['low'] - df['support']) / df['support'] <= level_tolerance
        )
        
        df['near_resistance'] = (
            abs(df['high'] - df['resistance']) / df['resistance'] <= level_tolerance
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.3))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Valid pin bar signals (at key levels)
        df['valid_bullish_pin'] = (
            df['bullish_pin_bar'] &
            df['near_support'] &
            df['volume_spike']
        )
        
        df['valid_bearish_pin'] = (
            df['bearish_pin_bar'] &
            df['near_resistance'] &
            df['volume_spike']
        )
        
        # Pin bar strength
        df['pin_strength'] = 0.0
        
        for i in range(len(df)):
            if df['valid_bullish_pin'].iloc[i] or df['valid_bearish_pin'].iloc[i]:
                # Base strength from wick ratio
                if df['valid_bullish_pin'].iloc[i]:
                    wick_ratio = df['lower_wick_ratio'].iloc[i]
                else:
                    wick_ratio = df['upper_wick_ratio'].iloc[i]
                
                strength = min(wick_ratio / 5, 1.0)  # Normalize to 0-1
                
                # Boost for smaller body
                body_factor = 1 - (df['body_pct'].iloc[i] / 100)
                strength *= (1 + body_factor * 0.3)
                
                # Boost for volume
                if df['volume_spike'].iloc[i]:
                    strength *= 1.2
                
                # Boost for range size
                if not np.isnan(df['atr'].iloc[i]):
                    range_factor = df['total_range'].iloc[i] / df['atr'].iloc[i]
                    if range_factor > 1.5:
                        strength *= 1.1
                
                df.loc[df.index[i], 'pin_strength'] = min(strength, 1.0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate pin bar signals."""
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
            
            # Bullish pin bar signal
            if (row['valid_bullish_pin'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bullish pin bar at support (Wick ratio: {row['lower_wick_ratio']:.1f}, Strength: {row['pin_strength']:.2f})",
                    confidence=0.7 + (row['pin_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish pin bar signal
            elif (row['valid_bearish_pin'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bearish pin bar at resistance (Wick ratio: {row['upper_wick_ratio']:.1f}, Strength: {row['pin_strength']:.2f})",
                    confidence=0.7 + (row['pin_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check pin bar entry conditions."""
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
        
        # Bullish pin bar
        if current['valid_bullish_pin']:
            return True, f"Bullish pin bar at support (wick ratio: {current['lower_wick_ratio']:.1f})"
        
        # Bearish pin bar
        if current['valid_bearish_pin']:
            return True, f"Bearish pin bar at resistance (wick ratio: {current['upper_wick_ratio']:.1f})"
        
        return False, "No valid pin bar signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check pin bar exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Pin bar invalidation - price broke the pin bar low/high
        if current_idx > 0:
            entry_bar = df_with_indicators.iloc[current_idx - 1] if current_idx > 0 else current
            
            if (entry_price > current['close'] and 
                current['close'] < entry_bar['low']):
                return True, "Bullish pin bar invalidated - broke pin bar low"
            
            if (entry_price < current['close'] and 
                current['close'] > entry_bar['high']):
                return True, "Bearish pin bar invalidated - broke pin bar high"
        
        # Opposite pin bar signal
        if (entry_price > current['close'] and current['valid_bearish_pin']):
            return True, "Opposite bearish pin bar formed"
        
        if (entry_price < current['close'] and current['valid_bullish_pin']):
            return True, "Opposite bullish pin bar formed"
        
        # Time-based exit (after 2 hours)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 2:
            return True, "Time-based exit (2 hours)"
        
        return False, "Hold pin bar position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use pin bar extreme as stop loss."""
        current = df.iloc[current_idx]
        
        # Use the pin bar's extreme point as stop
        if current['valid_bullish_pin']:
            # Stop below the pin bar low
            return current['low']
        elif current['valid_bearish_pin']:
            # Stop above the pin bar high
            return current['high']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on pin bar size and key levels."""
        current = df.iloc[current_idx]
        
        # Target based on pin bar range
        pin_range = current['total_range']
        
        if current['valid_bullish_pin']:
            # Target above entry by 2x pin bar range
            return entry_price + (pin_range * 2)
        elif current['valid_bearish_pin']:
            # Target below entry by 2x pin bar range
            return entry_price - (pin_range * 2)
        
        # Fallback target
        target_pct = self.get_param('target_pct', 2.5) / 100
        return entry_price * (1 + target_pct)