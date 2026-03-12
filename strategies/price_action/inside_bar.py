"""
Inside Bar Strategy
Price action strategy based on inside bar patterns indicating consolidation and potential breakout.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class InsideBarStrategy(BaseStrategy):
    """
    Inside Bar Strategy.
    
    Entry: When inside bar pattern forms and breaks out with volume
    Exit: When breakout fails or target reached
    """
    
    @property
    def name(self) -> str:
        return "Inside Bar"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'min_mother_bar_size': 0.5,  # Minimum mother bar size as % of ATR
            'max_inside_bars': 5,  # Maximum consecutive inside bars
            'breakout_threshold': 0.1,  # Breakout threshold as % of mother bar range
            'volume_threshold': 1.4,
            'min_volume': 5000,
            'atr_period': 14,
            'stop_loss_pct': 0.8,
            'target_pct': 2.0,
            'time_filter_start': "09:45",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate inside bar indicators."""
        df = df.copy()
        
        # Calculate ATR for context
        atr_period = self.get_param('atr_period', 14)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
        
        # Bar range
        df['bar_range'] = df['high'] - df['low']
        
        # Inside bar detection
        df['inside_bar'] = (
            (df['high'] < df['high'].shift(1)) &
            (df['low'] > df['low'].shift(1))
        )
        
        # Mother bar identification
        df['mother_bar'] = False
        df['mother_bar_high'] = np.nan
        df['mother_bar_low'] = np.nan
        df['inside_bar_count'] = 0
        df['mother_bar_range'] = np.nan
        
        min_mother_size = self.get_param('min_mother_bar_size', 0.5)
        max_inside_bars = self.get_param('max_inside_bars', 5)
        
        current_mother_high = np.nan
        current_mother_low = np.nan
        inside_count = 0
        
        for i in range(1, len(df)):
            if df['inside_bar'].iloc[i]:
                if inside_count == 0:
                    # First inside bar - previous bar becomes mother bar
                    prev_range = df['bar_range'].iloc[i-1]
                    prev_atr = df['atr'].iloc[i-1]
                    
                    if not np.isnan(prev_atr) and prev_range >= (prev_atr * min_mother_size):
                        current_mother_high = df['high'].iloc[i-1]
                        current_mother_low = df['low'].iloc[i-1]
                        df.loc[df.index[i-1], 'mother_bar'] = True
                        inside_count = 1
                    else:
                        inside_count = 0
                        current_mother_high = np.nan
                        current_mother_low = np.nan
                else:
                    inside_count += 1
                    if inside_count > max_inside_bars:
                        # Reset if too many inside bars
                        inside_count = 0
                        current_mother_high = np.nan
                        current_mother_low = np.nan
            else:
                inside_count = 0
                current_mother_high = np.nan
                current_mother_low = np.nan
            
            # Update current bar info
            if not np.isnan(current_mother_high):
                df.loc[df.index[i], 'mother_bar_high'] = current_mother_high
                df.loc[df.index[i], 'mother_bar_low'] = current_mother_low
                df.loc[df.index[i], 'inside_bar_count'] = inside_count
                df.loc[df.index[i], 'mother_bar_range'] = current_mother_high - current_mother_low
        
        # Breakout detection
        breakout_threshold = self.get_param('breakout_threshold', 0.1) / 100
        
        df['bullish_breakout'] = False
        df['bearish_breakout'] = False
        
        for i in range(len(df)):
            if (not np.isnan(df['mother_bar_high'].iloc[i]) and 
                df['inside_bar_count'].iloc[i] > 0):
                
                mother_high = df['mother_bar_high'].iloc[i]
                mother_low = df['mother_bar_low'].iloc[i]
                mother_range = df['mother_bar_range'].iloc[i]
                
                threshold = mother_range * breakout_threshold
                
                # Bullish breakout
                if df['close'].iloc[i] > mother_high + threshold:
                    df.loc[df.index[i], 'bullish_breakout'] = True
                
                # Bearish breakout
                elif df['close'].iloc[i] < mother_low - threshold:
                    df.loc[df.index[i], 'bearish_breakout'] = True
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.4))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:45")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Pattern strength
        df['pattern_strength'] = 0.0
        
        for i in range(len(df)):
            if df['bullish_breakout'].iloc[i] or df['bearish_breakout'].iloc[i]:
                # Base strength from inside bar count (more inside bars = stronger pattern)
                inside_count = df['inside_bar_count'].iloc[i]
                strength = min(inside_count / 3, 1.0)  # Normalize to 0-1
                
                # Boost for volume
                if df['volume_spike'].iloc[i]:
                    strength *= 1.3
                
                # Boost for larger mother bar
                if not np.isnan(df['mother_bar_range'].iloc[i]) and not np.isnan(df['atr'].iloc[i]):
                    range_ratio = df['mother_bar_range'].iloc[i] / df['atr'].iloc[i]
                    if range_ratio > 1.5:
                        strength *= 1.2
                
                df.loc[df.index[i], 'pattern_strength'] = min(strength, 1.0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate inside bar breakout signals."""
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
            
            # Bullish breakout signal
            if (row['bullish_breakout'] and row['volume_spike'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Inside bar bullish breakout (Count: {row['inside_bar_count']}, Strength: {row['pattern_strength']:.2f})",
                    confidence=0.7 + (row['pattern_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish breakout signal
            elif (row['bearish_breakout'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Inside bar bearish breakout (Count: {row['inside_bar_count']}, Strength: {row['pattern_strength']:.2f})",
                    confidence=0.7 + (row['pattern_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check inside bar breakout entry conditions."""
        if current_idx < 20:
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
        
        # Bullish breakout
        if current['bullish_breakout']:
            return True, f"Inside bar bullish breakout (count: {current['inside_bar_count']})"
        
        # Bearish breakout
        if current['bearish_breakout']:
            return True, f"Inside bar bearish breakout (count: {current['inside_bar_count']})"
        
        return False, "No inside bar breakout"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check inside bar exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Breakout failure - price returned to mother bar range
        if not np.isnan(current['mother_bar_high']) and not np.isnan(current['mother_bar_low']):
            mother_high = current['mother_bar_high']
            mother_low = current['mother_bar_low']
            
            if (entry_price > mother_high and current['close'] < mother_high):
                return True, "Bullish breakout failed - returned to mother bar"
            
            if (entry_price < mother_low and current['close'] > mother_low):
                return True, "Bearish breakout failed - returned to mother bar"
        
        # Time-based exit (after 1.5 hours)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 1.5:
            return True, "Time-based exit (1.5 hours)"
        
        return False, "Hold inside bar position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use mother bar boundary as stop loss."""
        current = df.iloc[current_idx]
        
        # Use mother bar levels as stops
        if not np.isnan(current['mother_bar_high']) and not np.isnan(current['mother_bar_low']):
            if current['bullish_breakout']:
                return current['mother_bar_low']
            elif current['bearish_breakout']:
                return current['mother_bar_high']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 0.8) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on mother bar range."""
        current = df.iloc[current_idx]
        
        # Target based on mother bar range
        if not np.isnan(current['mother_bar_range']):
            range_size = current['mother_bar_range']
            
            if current['bullish_breakout']:
                return entry_price + range_size
            elif current['bearish_breakout']:
                return entry_price - range_size
        
        # Fallback target
        target_pct = self.get_param('target_pct', 2.0) / 100
        return entry_price * (1 + target_pct)