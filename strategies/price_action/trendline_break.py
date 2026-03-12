"""
Trendline Break Strategy
Price action strategy based on trendline breakouts with volume confirmation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class TrendlineBreakStrategy(BaseStrategy):
    """
    Trendline Break Strategy.
    
    Entry: When price breaks a significant trendline with volume confirmation
    Exit: When breakout fails or target reached
    """
    
    @property
    def name(self) -> str:
        return "Trendline Break"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'lookback_period': 30,
            'min_touches': 3,           # Minimum touches to validate trendline
            'touch_tolerance': 0.2,     # Tolerance for trendline touches (%)
            'min_break_pct': 0.15,      # Minimum break percentage
            'min_r_squared': 0.7,       # Minimum R-squared for trendline validity
            'volume_threshold': 1.5,
            'min_volume': 6000,
            'stop_loss_pct': 1.2,
            'target_pct': 3.0,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trendline break indicators."""
        df = df.copy()
        
        lookback = self.get_param('lookback_period', 30)
        min_touches = self.get_param('min_touches', 3)
        touch_tolerance = self.get_param('touch_tolerance', 0.2) / 100
        min_break_pct = self.get_param('min_break_pct', 0.15) / 100
        min_r_squared = self.get_param('min_r_squared', 0.7)
        
        # Initialize trendline columns
        df['support_trendline'] = np.nan
        df['resistance_trendline'] = np.nan
        df['support_slope'] = np.nan
        df['resistance_slope'] = np.nan
        df['support_r_squared'] = np.nan
        df['resistance_r_squared'] = np.nan
        df['support_touches'] = 0
        df['resistance_touches'] = 0
        
        # Calculate trendlines for each bar
        for i in range(lookback, len(df)):
            window_data = df.iloc[i-lookback:i+1]
            
            # Find support trendline (connecting lows)
            support_line, support_stats = self._calculate_trendline(
                window_data, 'low', min_touches, touch_tolerance, min_r_squared
            )
            
            if support_line is not None:
                slope, intercept, r_squared, touches = support_stats
                current_support = slope * i + intercept
                
                df.loc[df.index[i], 'support_trendline'] = current_support
                df.loc[df.index[i], 'support_slope'] = slope
                df.loc[df.index[i], 'support_r_squared'] = r_squared
                df.loc[df.index[i], 'support_touches'] = touches
            
            # Find resistance trendline (connecting highs)
            resistance_line, resistance_stats = self._calculate_trendline(
                window_data, 'high', min_touches, touch_tolerance, min_r_squared
            )
            
            if resistance_line is not None:
                slope, intercept, r_squared, touches = resistance_stats
                current_resistance = slope * i + intercept
                
                df.loc[df.index[i], 'resistance_trendline'] = current_resistance
                df.loc[df.index[i], 'resistance_slope'] = slope
                df.loc[df.index[i], 'resistance_r_squared'] = r_squared
                df.loc[df.index[i], 'resistance_touches'] = touches
        
        # Detect trendline breaks
        df['support_break'] = (
            ~df['support_trendline'].isna() &
            (df['close'] > df['support_trendline']) &
            (df['close'].shift(1) <= df['support_trendline'].shift(1)) &
            ((df['close'] - df['support_trendline']) / df['support_trendline'] > min_break_pct)
        )
        
        df['resistance_break'] = (
            ~df['resistance_trendline'].isna() &
            (df['close'] < df['resistance_trendline']) &
            (df['close'].shift(1) >= df['resistance_trendline'].shift(1)) &
            ((df['resistance_trendline'] - df['close']) / df['resistance_trendline'] > min_break_pct)
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.5))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Break strength
        df['break_strength'] = 0.0
        
        for i in range(len(df)):
            if df['support_break'].iloc[i] or df['resistance_break'].iloc[i]:
                # Base strength from R-squared and touches
                if df['support_break'].iloc[i]:
                    r_squared = df['support_r_squared'].iloc[i]
                    touches = df['support_touches'].iloc[i]
                    trendline_value = df['support_trendline'].iloc[i]
                else:
                    r_squared = df['resistance_r_squared'].iloc[i]
                    touches = df['resistance_touches'].iloc[i]
                    trendline_value = df['resistance_trendline'].iloc[i]
                
                if not np.isnan(r_squared) and not np.isnan(trendline_value):
                    # Strength from trendline quality
                    strength = r_squared * min(touches / 5, 1.0)
                    
                    # Boost for volume
                    if df['volume_spike'].iloc[i]:
                        strength *= 1.3
                    
                    # Boost for break magnitude
                    break_pct = abs(df['close'].iloc[i] - trendline_value) / trendline_value
                    if break_pct > 0.005:  # > 0.5%
                        strength *= 1.2
                    
                    df.loc[df.index[i], 'break_strength'] = min(strength, 1.0)
        
        return df
    
    def _calculate_trendline(self, window_data, price_type, min_touches, touch_tolerance, min_r_squared):
        """Calculate trendline for given window data."""
        try:
            # Find pivot points
            if price_type == 'low':
                pivots = self._find_pivot_lows(window_data)
            else:
                pivots = self._find_pivot_highs(window_data)
            
            if len(pivots) < min_touches:
                return None, None
            
            # Try different combinations of pivot points
            best_line = None
            best_stats = None
            best_score = 0
            
            for i in range(len(pivots)):
                for j in range(i + 1, len(pivots)):
                    x_vals = [pivots[i][0], pivots[j][0]]
                    y_vals = [pivots[i][1], pivots[j][1]]
                    
                    # Calculate line equation
                    if len(x_vals) >= 2:
                        slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)
                        r_squared = r_value ** 2
                        
                        # Count touches within tolerance
                        touches = self._count_touches(
                            window_data, slope, intercept, price_type, touch_tolerance
                        )
                        
                        # Score this trendline
                        score = r_squared * touches
                        
                        if (touches >= min_touches and 
                            r_squared >= min_r_squared and 
                            score > best_score):
                            
                            best_line = (slope, intercept)
                            best_stats = (slope, intercept, r_squared, touches)
                            best_score = score
            
            return best_line, best_stats
            
        except Exception:
            return None, None
    
    def _find_pivot_lows(self, data, window=3):
        """Find pivot low points."""
        pivots = []
        for i in range(window, len(data) - window):
            current_low = data['low'].iloc[i]
            is_pivot = True
            
            # Check if current point is lower than surrounding points
            for j in range(i - window, i + window + 1):
                if j != i and data['low'].iloc[j] < current_low:
                    is_pivot = False
                    break
            
            if is_pivot:
                pivots.append((i, current_low))
        
        return pivots
    
    def _find_pivot_highs(self, data, window=3):
        """Find pivot high points."""
        pivots = []
        for i in range(window, len(data) - window):
            current_high = data['high'].iloc[i]
            is_pivot = True
            
            # Check if current point is higher than surrounding points
            for j in range(i - window, i + window + 1):
                if j != i and data['high'].iloc[j] > current_high:
                    is_pivot = False
                    break
            
            if is_pivot:
                pivots.append((i, current_high))
        
        return pivots
    
    def _count_touches(self, data, slope, intercept, price_type, tolerance):
        """Count how many points touch the trendline within tolerance."""
        touches = 0
        
        for i in range(len(data)):
            trendline_value = slope * i + intercept
            actual_value = data[price_type].iloc[i]
            
            # Calculate percentage difference
            if trendline_value > 0:
                diff_pct = abs(actual_value - trendline_value) / trendline_value
                if diff_pct <= tolerance:
                    touches += 1
        
        return touches
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate trendline break signals."""
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
            
            # Support trendline break (bullish)
            if (row['support_break'] and row['volume_spike'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Support trendline break (Touches: {row['support_touches']}, R²: {row['support_r_squared']:.2f})",
                    confidence=0.7 + (row['break_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Resistance trendline break (bearish)
            elif (row['resistance_break'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Resistance trendline break (Touches: {row['resistance_touches']}, R²: {row['resistance_r_squared']:.2f})",
                    confidence=0.7 + (row['break_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check trendline break entry conditions."""
        if current_idx < 35:
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
        
        # Support break
        if current['support_break']:
            return True, f"Support trendline break (touches: {current['support_touches']})"
        
        # Resistance break
        if current['resistance_break']:
            return True, f"Resistance trendline break (touches: {current['resistance_touches']})"
        
        return False, "No trendline break"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check trendline break exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # False breakout - price returned to trendline
        if not np.isnan(current['support_trendline']) and entry_price > current['support_trendline']:
            if current['close'] < current['support_trendline']:
                return True, "False breakout - returned below support trendline"
        
        if not np.isnan(current['resistance_trendline']) and entry_price < current['resistance_trendline']:
            if current['close'] > current['resistance_trendline']:
                return True, "False breakout - returned above resistance trendline"
        
        # Time-based exit (after 3 hours)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 3:
            return True, "Time-based exit (3 hours)"
        
        return False, "Hold trendline break position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use trendline as stop loss."""
        current = df.iloc[current_idx]
        
        # Use the broken trendline as stop
        if current['support_break'] and not np.isnan(current['support_trendline']):
            return current['support_trendline'] * 0.998  # Small buffer below
        elif current['resistance_break'] and not np.isnan(current['resistance_trendline']):
            return current['resistance_trendline'] * 1.002  # Small buffer above
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.2) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on trendline angle and recent range."""
        current = df.iloc[current_idx]
        
        # Target based on recent range
        if current_idx >= 20:
            recent_data = df.iloc[current_idx-20:current_idx]
            recent_range = recent_data['high'].max() - recent_data['low'].min()
            
            if current['support_break']:
                return entry_price + recent_range
            elif current['resistance_break']:
                return entry_price - recent_range
        
        # Fallback target
        target_pct = self.get_param('target_pct', 3.0) / 100
        return entry_price * (1 + target_pct)