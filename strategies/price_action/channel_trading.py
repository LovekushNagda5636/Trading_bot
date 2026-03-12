"""
Channel Trading Strategy
Price action strategy based on trading within price channels (parallel support and resistance).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class ChannelTradingStrategy(BaseStrategy):
    """
    Channel Trading Strategy.
    
    Entry: When price bounces off channel boundaries or breaks out
    Exit: When reaching opposite boundary or channel breaks
    """
    
    @property
    def name(self) -> str:
        return "Channel Trading"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'lookback_period': 40,
            'min_channel_width': 1.0,   # Minimum channel width as % of price
            'max_channel_width': 8.0,   # Maximum channel width as % of price
            'min_touches': 4,           # Minimum touches to validate channel
            'touch_tolerance': 0.3,     # Tolerance for channel touches (%)
            'min_r_squared': 0.6,       # Minimum R-squared for channel validity
            'breakout_threshold': 0.2,  # Breakout threshold (%)
            'volume_threshold': 1.3,
            'min_volume': 5000,
            'stop_loss_pct': 1.0,
            'target_pct': 2.0,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate channel trading indicators."""
        df = df.copy()
        
        lookback = self.get_param('lookback_period', 40)
        min_channel_width = self.get_param('min_channel_width', 1.0) / 100
        max_channel_width = self.get_param('max_channel_width', 8.0) / 100
        min_touches = self.get_param('min_touches', 4)
        touch_tolerance = self.get_param('touch_tolerance', 0.3) / 100
        min_r_squared = self.get_param('min_r_squared', 0.6)
        breakout_threshold = self.get_param('breakout_threshold', 0.2) / 100
        
        # Initialize channel columns
        df['channel_upper'] = np.nan
        df['channel_lower'] = np.nan
        df['channel_slope'] = np.nan
        df['channel_width'] = np.nan
        df['channel_r_squared'] = np.nan
        df['channel_touches'] = 0
        df['channel_valid'] = False
        
        # Calculate channels for each bar
        for i in range(lookback, len(df)):
            window_data = df.iloc[i-lookback:i+1]
            
            # Find channel
            channel_info = self._find_channel(
                window_data, min_touches, touch_tolerance, min_r_squared,
                min_channel_width, max_channel_width
            )
            
            if channel_info is not None:
                upper_line, lower_line, slope, width, r_squared, touches = channel_info
                
                # Calculate current channel levels
                current_upper = upper_line[0] * i + upper_line[1]
                current_lower = lower_line[0] * i + lower_line[1]
                
                df.loc[df.index[i], 'channel_upper'] = current_upper
                df.loc[df.index[i], 'channel_lower'] = current_lower
                df.loc[df.index[i], 'channel_slope'] = slope
                df.loc[df.index[i], 'channel_width'] = width
                df.loc[df.index[i], 'channel_r_squared'] = r_squared
                df.loc[df.index[i], 'channel_touches'] = touches
                df.loc[df.index[i], 'channel_valid'] = True
        
        # Channel position
        df['channel_position'] = np.nan  # 0 = bottom, 0.5 = middle, 1 = top
        
        for i in range(len(df)):
            if df['channel_valid'].iloc[i]:
                upper = df['channel_upper'].iloc[i]
                lower = df['channel_lower'].iloc[i]
                close = df['close'].iloc[i]
                
                if upper > lower:
                    position = (close - lower) / (upper - lower)
                    df.loc[df.index[i], 'channel_position'] = np.clip(position, 0, 1)
        
        # Channel signals
        df['channel_support_bounce'] = (
            df['channel_valid'] &
            (df['channel_position'] <= 0.1) &  # Near bottom
            (df['close'] > df['open']) &       # Bullish candle
            (df['low'] <= df['channel_lower'] * (1 + touch_tolerance))
        )
        
        df['channel_resistance_bounce'] = (
            df['channel_valid'] &
            (df['channel_position'] >= 0.9) &  # Near top
            (df['close'] < df['open']) &       # Bearish candle
            (df['high'] >= df['channel_upper'] * (1 - touch_tolerance))
        )
        
        # Channel breakouts
        df['channel_upper_breakout'] = (
            df['channel_valid'] &
            (df['close'] > df['channel_upper'] * (1 + breakout_threshold)) &
            (df['close'].shift(1) <= df['channel_upper'].shift(1))
        )
        
        df['channel_lower_breakout'] = (
            df['channel_valid'] &
            (df['close'] < df['channel_lower'] * (1 - breakout_threshold)) &
            (df['close'].shift(1) >= df['channel_lower'].shift(1))
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.3))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Signal strength
        df['channel_strength'] = 0.0
        
        for i in range(len(df)):
            if (df['channel_support_bounce'].iloc[i] or 
                df['channel_resistance_bounce'].iloc[i] or
                df['channel_upper_breakout'].iloc[i] or 
                df['channel_lower_breakout'].iloc[i]):
                
                # Base strength from channel quality
                r_squared = df['channel_r_squared'].iloc[i]
                touches = df['channel_touches'].iloc[i]
                
                if not np.isnan(r_squared):
                    strength = r_squared * min(touches / 6, 1.0)
                    
                    # Boost for volume
                    if df['volume_spike'].iloc[i]:
                        strength *= 1.3
                    
                    # Boost for channel width (wider channels are more significant)
                    width = df['channel_width'].iloc[i]
                    if not np.isnan(width) and width > 0.02:  # > 2%
                        strength *= 1.2
                    
                    # Boost for breakouts
                    if (df['channel_upper_breakout'].iloc[i] or 
                        df['channel_lower_breakout'].iloc[i]):
                        strength *= 1.4
                    
                    df.loc[df.index[i], 'channel_strength'] = min(strength, 1.0)
        
        return df
    
    def _find_channel(self, data, min_touches, touch_tolerance, min_r_squared, 
                     min_width, max_width):
        """Find parallel channel lines."""
        try:
            # Find pivot points
            pivot_highs = self._find_pivot_points(data, 'high')
            pivot_lows = self._find_pivot_points(data, 'low')
            
            if len(pivot_highs) < 2 or len(pivot_lows) < 2:
                return None
            
            best_channel = None
            best_score = 0
            
            # Try different combinations of pivot points for upper line
            for i in range(len(pivot_highs)):
                for j in range(i + 1, len(pivot_highs)):
                    # Calculate upper trendline
                    x_vals = [pivot_highs[i][0], pivot_highs[j][0]]
                    y_vals = [pivot_highs[i][1], pivot_highs[j][1]]
                    
                    if len(x_vals) >= 2:
                        slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)
                        upper_r_squared = r_value ** 2
                        
                        # Find parallel lower line
                        lower_line = self._find_parallel_line(
                            data, slope, pivot_lows, touch_tolerance
                        )
                        
                        if lower_line is not None:
                            lower_slope, lower_intercept, lower_r_squared = lower_line
                            
                            # Calculate channel width
                            mid_point = len(data) // 2
                            upper_val = slope * mid_point + intercept
                            lower_val = lower_slope * mid_point + lower_intercept
                            
                            if upper_val > 0 and lower_val > 0:
                                width = abs(upper_val - lower_val) / ((upper_val + lower_val) / 2)
                                
                                # Check channel validity
                                if (min_width <= width <= max_width and
                                    upper_r_squared >= min_r_squared and
                                    lower_r_squared >= min_r_squared):
                                    
                                    # Count total touches
                                    upper_touches = self._count_line_touches(
                                        data, slope, intercept, 'high', touch_tolerance
                                    )
                                    lower_touches = self._count_line_touches(
                                        data, lower_slope, lower_intercept, 'low', touch_tolerance
                                    )
                                    
                                    total_touches = upper_touches + lower_touches
                                    
                                    if total_touches >= min_touches:
                                        # Score this channel
                                        avg_r_squared = (upper_r_squared + lower_r_squared) / 2
                                        score = avg_r_squared * total_touches * (1 / width)
                                        
                                        if score > best_score:
                                            best_channel = (
                                                (slope, intercept),           # upper line
                                                (lower_slope, lower_intercept), # lower line
                                                slope,                        # slope
                                                width,                        # width
                                                avg_r_squared,               # r_squared
                                                total_touches                # touches
                                            )
                                            best_score = score
            
            return best_channel
            
        except Exception:
            return None
    
    def _find_pivot_points(self, data, price_type, window=3):
        """Find pivot points."""
        pivots = []
        
        for i in range(window, len(data) - window):
            current_price = data[price_type].iloc[i]
            is_pivot = True
            
            if price_type == 'high':
                # Check if current point is higher than surrounding points
                for j in range(i - window, i + window + 1):
                    if j != i and data[price_type].iloc[j] > current_price:
                        is_pivot = False
                        break
            else:  # low
                # Check if current point is lower than surrounding points
                for j in range(i - window, i + window + 1):
                    if j != i and data[price_type].iloc[j] < current_price:
                        is_pivot = False
                        break
            
            if is_pivot:
                pivots.append((i, current_price))
        
        return pivots
    
    def _find_parallel_line(self, data, slope, pivot_points, tolerance):
        """Find parallel line to given slope using pivot points."""
        best_line = None
        best_r_squared = 0
        
        for i in range(len(pivot_points)):
            for j in range(i + 1, len(pivot_points)):
                x_vals = [pivot_points[i][0], pivot_points[j][0]]
                y_vals = [pivot_points[i][1], pivot_points[j][1]]
                
                if len(x_vals) >= 2:
                    line_slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)
                    r_squared = r_value ** 2
                    
                    # Check if slopes are similar (parallel)
                    slope_diff = abs(line_slope - slope) / abs(slope) if slope != 0 else abs(line_slope)
                    
                    if slope_diff <= 0.1 and r_squared > best_r_squared:  # 10% tolerance
                        best_line = (line_slope, intercept, r_squared)
                        best_r_squared = r_squared
        
        return best_line
    
    def _count_line_touches(self, data, slope, intercept, price_type, tolerance):
        """Count touches to a line."""
        touches = 0
        
        for i in range(len(data)):
            line_value = slope * i + intercept
            actual_value = data[price_type].iloc[i]
            
            if line_value > 0:
                diff_pct = abs(actual_value - line_value) / line_value
                if diff_pct <= tolerance:
                    touches += 1
        
        return touches
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate channel trading signals."""
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
            
            # Channel support bounce (buy)
            if (row['channel_support_bounce'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Channel support bounce (Position: {row['channel_position']:.2f}, Strength: {row['channel_strength']:.2f})",
                    confidence=0.7 + (row['channel_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Channel resistance bounce (sell)
            elif (row['channel_resistance_bounce'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Channel resistance bounce (Position: {row['channel_position']:.2f}, Strength: {row['channel_strength']:.2f})",
                    confidence=0.7 + (row['channel_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Channel breakouts
            elif (row['channel_upper_breakout'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Channel upper breakout (Strength: {row['channel_strength']:.2f})",
                    confidence=0.8 + (row['channel_strength'] * 0.15),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            elif (row['channel_lower_breakout'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Channel lower breakout (Strength: {row['channel_strength']:.2f})",
                    confidence=0.8 + (row['channel_strength'] * 0.15),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check channel trading entry conditions."""
        if current_idx < 45:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Time filter
        if not current['time_filter']:
            return False, "Outside trading hours"
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Channel support bounce
        if current['channel_support_bounce']:
            return True, f"Channel support bounce (position: {current['channel_position']:.2f})"
        
        # Channel resistance bounce
        if current['channel_resistance_bounce']:
            return True, f"Channel resistance bounce (position: {current['channel_position']:.2f})"
        
        # Channel breakouts
        if current['channel_upper_breakout'] and current['volume_spike']:
            return True, "Channel upper breakout"
        
        if current['channel_lower_breakout'] and current['volume_spike']:
            return True, "Channel lower breakout"
        
        return False, "No channel signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check channel trading exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Channel invalidation
        if not current['channel_valid']:
            return True, "Channel no longer valid"
        
        # Reached opposite channel boundary
        if not np.isnan(current['channel_position']):
            if (entry_price < current['close'] and current['channel_position'] >= 0.9):
                return True, "Reached channel resistance"
            
            if (entry_price > current['close'] and current['channel_position'] <= 0.1):
                return True, "Reached channel support"
        
        # Time-based exit (after 4 hours)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 4:
            return True, "Time-based exit (4 hours)"
        
        return False, "Hold channel position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use channel boundaries as stop loss."""
        current = df.iloc[current_idx]
        
        # Use channel boundaries as stops
        if (current['channel_support_bounce'] or current['channel_upper_breakout']):
            if not np.isnan(current['channel_lower']):
                return current['channel_lower'] * 0.998  # Below support
        
        elif (current['channel_resistance_bounce'] or current['channel_lower_breakout']):
            if not np.isnan(current['channel_upper']):
                return current['channel_upper'] * 1.002  # Above resistance
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on channel boundaries."""
        current = df.iloc[current_idx]
        
        # For bounces, target opposite boundary
        if current['channel_support_bounce'] and not np.isnan(current['channel_upper']):
            return current['channel_upper'] * 0.99  # Just below resistance
        
        elif current['channel_resistance_bounce'] and not np.isnan(current['channel_lower']):
            return current['channel_lower'] * 1.01  # Just above support
        
        # For breakouts, target channel width beyond breakout
        elif not np.isnan(current['channel_width']):
            width_points = entry_price * current['channel_width']
            
            if current['channel_upper_breakout']:
                return entry_price + width_points
            elif current['channel_lower_breakout']:
                return entry_price - width_points
        
        # Fallback target
        target_pct = self.get_param('target_pct', 2.0) / 100
        return entry_price * (1 + target_pct)