"""
Pivot Point Breakout Strategy
Breakout strategy using pivot points and support/resistance levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class PivotBreakoutStrategy(BaseStrategy):
    """
    Pivot Point Breakout Strategy.
    
    Entry: When price breaks above/below pivot levels with volume
    Exit: When price returns to pivot or reaches next level
    """
    
    @property
    def name(self) -> str:
        return "Pivot Breakout"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'pivot_type': 'standard',  # 'standard', 'fibonacci', 'camarilla'
            'breakout_confirmation': 0.1,  # % beyond pivot for confirmation
            'volume_threshold': 1.5,
            'min_volume': 7000,
            'stop_loss_pct': 1.0,
            'use_next_level_target': True,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pivot point indicators."""
        df = df.copy()
        
        # Add date column for daily pivot calculation
        df['date'] = df.index.date
        
        # Calculate pivot points for each day
        df['pivot'] = np.nan
        df['r1'] = np.nan
        df['r2'] = np.nan
        df['r3'] = np.nan
        df['s1'] = np.nan
        df['s2'] = np.nan
        df['s3'] = np.nan
        
        pivot_type = self.get_param('pivot_type', 'standard')
        
        unique_dates = sorted(df['date'].unique())
        
        for i, date in enumerate(unique_dates):
            if i == 0:
                continue  # Skip first day (no previous day)
            
            # Get previous day's data
            prev_date = unique_dates[i-1]
            prev_day_data = df[df['date'] == prev_date]
            
            if len(prev_day_data) > 0:
                prev_high = prev_day_data['high'].max()
                prev_low = prev_day_data['low'].min()
                prev_close = prev_day_data['close'].iloc[-1]
                
                # Calculate pivot points based on type
                if pivot_type == 'standard':
                    pivot = (prev_high + prev_low + prev_close) / 3
                    r1 = 2 * pivot - prev_low
                    r2 = pivot + (prev_high - prev_low)
                    r3 = prev_high + 2 * (pivot - prev_low)
                    s1 = 2 * pivot - prev_high
                    s2 = pivot - (prev_high - prev_low)
                    s3 = prev_low - 2 * (prev_high - pivot)
                
                elif pivot_type == 'fibonacci':
                    pivot = (prev_high + prev_low + prev_close) / 3
                    r1 = pivot + 0.382 * (prev_high - prev_low)
                    r2 = pivot + 0.618 * (prev_high - prev_low)
                    r3 = pivot + (prev_high - prev_low)
                    s1 = pivot - 0.382 * (prev_high - prev_low)
                    s2 = pivot - 0.618 * (prev_high - prev_low)
                    s3 = pivot - (prev_high - prev_low)
                
                elif pivot_type == 'camarilla':
                    pivot = (prev_high + prev_low + prev_close) / 3
                    r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
                    r2 = prev_close + 1.1 * (prev_high - prev_low) / 6
                    r3 = prev_close + 1.1 * (prev_high - prev_low) / 4
                    s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
                    s2 = prev_close - 1.1 * (prev_high - prev_low) / 6
                    s3 = prev_close - 1.1 * (prev_high - prev_low) / 4
                
                # Apply to current day
                current_day_mask = df['date'] == date
                df.loc[current_day_mask, 'pivot'] = pivot
                df.loc[current_day_mask, 'r1'] = r1
                df.loc[current_day_mask, 'r2'] = r2
                df.loc[current_day_mask, 'r3'] = r3
                df.loc[current_day_mask, 's1'] = s1
                df.loc[current_day_mask, 's2'] = s2
                df.loc[current_day_mask, 's3'] = s3
        
        # Breakout detection
        breakout_conf = self.get_param('breakout_confirmation', 0.1) / 100
        
        # Resistance breakouts
        df['r1_breakout'] = df['close'] > df['r1'] * (1 + breakout_conf)
        df['r2_breakout'] = df['close'] > df['r2'] * (1 + breakout_conf)
        df['r3_breakout'] = df['close'] > df['r3'] * (1 + breakout_conf)
        
        # Support breakouts
        df['s1_breakout'] = df['close'] < df['s1'] * (1 - breakout_conf)
        df['s2_breakout'] = df['close'] < df['s2'] * (1 - breakout_conf)
        df['s3_breakout'] = df['close'] < df['s3'] * (1 - breakout_conf)
        
        # Pivot breakouts
        df['pivot_breakout_up'] = df['close'] > df['pivot'] * (1 + breakout_conf)
        df['pivot_breakout_down'] = df['close'] < df['pivot'] * (1 - breakout_conf)
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.5))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Identify which level was broken
        df['broken_level'] = 'none'
        df['broken_level_value'] = np.nan
        df['next_target'] = np.nan
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            if row['r3_breakout']:
                df.loc[df.index[i], 'broken_level'] = 'R3'
                df.loc[df.index[i], 'broken_level_value'] = row['r3']
            elif row['r2_breakout']:
                df.loc[df.index[i], 'broken_level'] = 'R2'
                df.loc[df.index[i], 'broken_level_value'] = row['r2']
                df.loc[df.index[i], 'next_target'] = row['r3']
            elif row['r1_breakout']:
                df.loc[df.index[i], 'broken_level'] = 'R1'
                df.loc[df.index[i], 'broken_level_value'] = row['r1']
                df.loc[df.index[i], 'next_target'] = row['r2']
            elif row['pivot_breakout_up']:
                df.loc[df.index[i], 'broken_level'] = 'Pivot_Up'
                df.loc[df.index[i], 'broken_level_value'] = row['pivot']
                df.loc[df.index[i], 'next_target'] = row['r1']
            elif row['pivot_breakout_down']:
                df.loc[df.index[i], 'broken_level'] = 'Pivot_Down'
                df.loc[df.index[i], 'broken_level_value'] = row['pivot']
                df.loc[df.index[i], 'next_target'] = row['s1']
            elif row['s1_breakout']:
                df.loc[df.index[i], 'broken_level'] = 'S1'
                df.loc[df.index[i], 'broken_level_value'] = row['s1']
                df.loc[df.index[i], 'next_target'] = row['s2']
            elif row['s2_breakout']:
                df.loc[df.index[i], 'broken_level'] = 'S2'
                df.loc[df.index[i], 'broken_level_value'] = row['s2']
                df.loc[df.index[i], 'next_target'] = row['s3']
            elif row['s3_breakout']:
                df.loc[df.index[i], 'broken_level'] = 'S3'
                df.loc[df.index[i], 'broken_level_value'] = row['s3']
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate pivot breakout signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Skip if no pivot data
            if pd.isna(row['pivot']):
                continue
            
            # Time filter
            if not row['time_filter']:
                continue
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 7000):
                continue
            
            # Check for breakouts
            if (row['broken_level'] != 'none' and row['volume_spike'] and 
                self.is_market_open(row.name)):
                
                # Determine action based on broken level
                if row['broken_level'] in ['R1', 'R2', 'R3', 'Pivot_Up']:
                    action = "BUY"
                    reason = f"Pivot {row['broken_level']} breakout"
                else:
                    action = "SELL"
                    reason = f"Pivot {row['broken_level']} breakdown"
                
                signal = Signal(
                    action=action,
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"{reason} (Level: {row['broken_level_value']:.2f})",
                    confidence=0.75,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check pivot breakout entry conditions."""
        if current_idx < 10:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Check pivot availability
        if pd.isna(current['pivot']):
            return False, "No pivot data"
        
        # Time filter
        if not current['time_filter']:
            return False, "Outside trading hours"
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if not current['volume_spike']:
            return False, "No volume confirmation"
        
        # Check for breakout
        if current['broken_level'] != 'none':
            return True, f"Pivot {current['broken_level']} breakout"
        
        return False, "No pivot breakout"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check pivot breakout exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Return to broken level (failed breakout)
        if not pd.isna(current['broken_level_value']):
            if (entry_price > current['broken_level_value'] and 
                current['close'] < current['broken_level_value']):
                return True, "Failed breakout - returned to pivot level"
            
            if (entry_price < current['broken_level_value'] and 
                current['close'] > current['broken_level_value']):
                return True, "Failed breakdown - returned to pivot level"
        
        # Reached next pivot level
        if (self.get_param('use_next_level_target', True) and 
            not pd.isna(current['next_target'])):
            
            if (entry_price < current['next_target'] and 
                current['close'] >= current['next_target']):
                return True, "Reached next pivot level"
            
            if (entry_price > current['next_target'] and 
                current['close'] <= current['next_target']):
                return True, "Reached next pivot level"
        
        return False, "Hold pivot breakout position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use broken pivot level as stop loss."""
        current = df.iloc[current_idx]
        
        # Use the broken level as stop
        if not pd.isna(current['broken_level_value']):
            return current['broken_level_value']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use next pivot level as target."""
        current = df.iloc[current_idx]
        
        # Use next level as target if available
        if (self.get_param('use_next_level_target', True) and 
            not pd.isna(current['next_target'])):
            return current['next_target']
        
        # Fallback to percentage target
        return entry_price * 1.02