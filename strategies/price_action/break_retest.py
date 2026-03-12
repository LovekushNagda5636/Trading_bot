"""
Break and Retest Strategy
Price action strategy based on breakout followed by retest of the broken level.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class BreakRetestStrategy(BaseStrategy):
    """
    Break and Retest Strategy.
    
    Entry: When price breaks a key level and retests it successfully
    Exit: When retest fails or target reached
    """
    
    @property
    def name(self) -> str:
        return "Break and Retest"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'lookback_period': 20,
            'min_break_pct': 0.2,  # Minimum break percentage
            'retest_tolerance': 0.1,  # Retest tolerance percentage
            'volume_threshold': 1.3,
            'min_volume': 5000,
            'max_retest_bars': 10,  # Max bars for retest to occur
            'stop_loss_pct': 1.0,
            'target_pct': 2.5,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate break and retest indicators."""
        df = df.copy()
        
        lookback = self.get_param('lookback_period', 20)
        min_break_pct = self.get_param('min_break_pct', 0.2) / 100
        retest_tolerance = self.get_param('retest_tolerance', 0.1) / 100
        max_retest_bars = self.get_param('max_retest_bars', 10)
        
        # Calculate support and resistance levels
        df['resistance'] = df['high'].rolling(window=lookback).max()
        df['support'] = df['low'].rolling(window=lookback).min()
        
        # Detect breakouts
        df['resistance_break'] = (
            (df['close'] > df['resistance'].shift(1)) &
            (df['close'].shift(1) <= df['resistance'].shift(1)) &
            ((df['close'] - df['resistance'].shift(1)) / df['resistance'].shift(1) > min_break_pct)
        )
        
        df['support_break'] = (
            (df['close'] < df['support'].shift(1)) &
            (df['close'].shift(1) >= df['support'].shift(1)) &
            ((df['support'].shift(1) - df['close']) / df['support'].shift(1) > min_break_pct)
        )
        
        # Track breakout levels and bars since break
        df['breakout_level'] = np.nan
        df['bars_since_break'] = 0
        df['break_direction'] = 0  # 1 for upward break, -1 for downward break
        
        current_level = np.nan
        current_direction = 0
        bars_count = 0
        
        for i in range(len(df)):
            if df['resistance_break'].iloc[i]:
                current_level = df['resistance'].iloc[i-1] if i > 0 else df['high'].iloc[i]
                current_direction = 1
                bars_count = 0
            elif df['support_break'].iloc[i]:
                current_level = df['support'].iloc[i-1] if i > 0 else df['low'].iloc[i]
                current_direction = -1
                bars_count = 0
            
            if not np.isnan(current_level):
                bars_count += 1
                if bars_count <= max_retest_bars:
                    df.loc[df.index[i], 'breakout_level'] = current_level
                    df.loc[df.index[i], 'break_direction'] = current_direction
                    df.loc[df.index[i], 'bars_since_break'] = bars_count
                else:
                    # Reset if too many bars have passed
                    current_level = np.nan
                    current_direction = 0
                    bars_count = 0
        
        # Detect retests
        df['bullish_retest'] = False
        df['bearish_retest'] = False
        
        for i in range(len(df)):
            if (df['break_direction'].iloc[i] == 1 and 
                not np.isnan(df['breakout_level'].iloc[i])):
                
                level = df['breakout_level'].iloc[i]
                tolerance = level * retest_tolerance
                
                # Bullish retest: price comes back to test resistance (now support)
                if (df['low'].iloc[i] <= level + tolerance and 
                    df['low'].iloc[i] >= level - tolerance and
                    df['close'].iloc[i] > level):
                    df.loc[df.index[i], 'bullish_retest'] = True
            
            elif (df['break_direction'].iloc[i] == -1 and 
                  not np.isnan(df['breakout_level'].iloc[i])):
                
                level = df['breakout_level'].iloc[i]
                tolerance = level * retest_tolerance
                
                # Bearish retest: price comes back to test support (now resistance)
                if (df['high'].iloc[i] >= level - tolerance and 
                    df['high'].iloc[i] <= level + tolerance and
                    df['close'].iloc[i] < level):
                    df.loc[df.index[i], 'bearish_retest'] = True
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.3))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Retest strength
        df['retest_strength'] = 0.0
        for i in range(len(df)):
            if df['bullish_retest'].iloc[i] or df['bearish_retest'].iloc[i]:
                # Base strength from how close to the level
                level = df['breakout_level'].iloc[i]
                if not np.isnan(level):
                    distance = abs(df['close'].iloc[i] - level) / level
                    strength = max(0, 1 - (distance / retest_tolerance))
                    
                    # Boost for volume
                    if df['volume_spike'].iloc[i]:
                        strength *= 1.2
                    
                    # Boost for quick retest (within 5 bars)
                    if df['bars_since_break'].iloc[i] <= 5:
                        strength *= 1.1
                    
                    df.loc[df.index[i], 'retest_strength'] = min(strength, 1.0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate break and retest signals."""
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
            
            # Bullish retest signal
            if (row['bullish_retest'] and row['volume_spike'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bullish retest at {row['breakout_level']:.2f} (Strength: {row['retest_strength']:.2f})",
                    confidence=0.7 + (row['retest_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish retest signal
            elif (row['bearish_retest'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bearish retest at {row['breakout_level']:.2f} (Strength: {row['retest_strength']:.2f})",
                    confidence=0.7 + (row['retest_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check break and retest entry conditions."""
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
        
        # Bullish retest
        if current['bullish_retest']:
            return True, f"Bullish retest at {current['breakout_level']:.2f}"
        
        # Bearish retest
        if current['bearish_retest']:
            return True, f"Bearish retest at {current['breakout_level']:.2f}"
        
        return False, "No retest signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check break and retest exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Retest failed - price broke back through the level
        if not np.isnan(current['breakout_level']):
            level = current['breakout_level']
            
            if (entry_price > level and current['close'] < level):
                return True, "Bullish retest failed - broke back below level"
            
            if (entry_price < level and current['close'] > level):
                return True, "Bearish retest failed - broke back above level"
        
        # Time-based exit (after 2 hours)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 2:
            return True, "Time-based exit (2 hours)"
        
        return False, "Hold retest position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use the breakout level as stop loss."""
        current = df.iloc[current_idx]
        
        # Use the breakout level as stop
        if not np.isnan(current['breakout_level']):
            level = current['breakout_level']
            buffer = level * 0.001  # 0.1% buffer
            
            if current['break_direction'] == 1:  # Bullish retest
                return level - buffer
            elif current['break_direction'] == -1:  # Bearish retest
                return level + buffer
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on breakout range."""
        current = df.iloc[current_idx]
        
        # Target based on the range that was broken
        if (current_idx >= 20 and not np.isnan(current['breakout_level'])):
            lookback = self.get_param('lookback_period', 20)
            recent_data = df.iloc[current_idx-lookback:current_idx]
            
            if current['break_direction'] == 1:  # Bullish
                range_size = recent_data['high'].max() - recent_data['low'].min()
                return entry_price + range_size
            elif current['break_direction'] == -1:  # Bearish
                range_size = recent_data['high'].max() - recent_data['low'].min()
                return entry_price - range_size
        
        # Fallback target
        target_pct = self.get_param('target_pct', 2.5) / 100
        return entry_price * (1 + target_pct)