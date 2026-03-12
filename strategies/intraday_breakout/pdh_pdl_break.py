"""
Previous Day High/Low Breakout Strategy
Breakout strategy using previous day's high and low levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class PDHPDLBreakoutStrategy(BaseStrategy):
    """
    Previous Day High/Low Breakout Strategy.
    
    Entry: When price breaks above PDH or below PDL with volume confirmation
    Exit: End of day or when price returns to previous day's range
    """
    
    @property
    def name(self) -> str:
        return "PDH PDL Breakout"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'breakout_confirmation': 0.1,  # % above/below PDH/PDL for confirmation
            'volume_threshold': 1.5,
            'min_volume': 8000,
            'stop_loss_pct': 1.0,
            'target_multiplier': 2.0,
            'max_trades_per_day': 2,
            'time_filter_start': "10:00",  # Avoid early morning volatility
            'time_filter_end': "14:30"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate PDH/PDL indicators."""
        df = df.copy()
        
        # Add date column
        df['date'] = df.index.date
        
        # Calculate previous day high/low for each day
        df['pdh'] = np.nan
        df['pdl'] = np.nan
        df['pd_range'] = np.nan
        
        unique_dates = sorted(df['date'].unique())
        
        for i, date in enumerate(unique_dates):
            if i == 0:
                continue  # Skip first day (no previous day)
            
            # Get previous day's data
            prev_date = unique_dates[i-1]
            prev_day_data = df[df['date'] == prev_date]
            
            if len(prev_day_data) > 0:
                pdh = prev_day_data['high'].max()
                pdl = prev_day_data['low'].min()
                pd_range = pdh - pdl
                
                # Apply to current day
                current_day_mask = df['date'] == date
                df.loc[current_day_mask, 'pdh'] = pdh
                df.loc[current_day_mask, 'pdl'] = pdl
                df.loc[current_day_mask, 'pd_range'] = pd_range
        
        # Breakout levels with confirmation
        breakout_conf = self.get_param('breakout_confirmation', 0.1) / 100
        df['pdh_breakout_level'] = df['pdh'] * (1 + breakout_conf)
        df['pdl_breakout_level'] = df['pdl'] * (1 - breakout_conf)
        
        # Breakout detection
        df['pdh_breakout'] = (df['close'] > df['pdh_breakout_level']) & (~pd.isna(df['pdh']))
        df['pdl_breakout'] = (df['close'] < df['pdl_breakout_level']) & (~pd.isna(df['pdl']))
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.5))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "10:00")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "14:30")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate PDH/PDL breakout signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        daily_trades = {}  # Track trades per day
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Skip if no PDH/PDL data
            if pd.isna(row['pdh']) or pd.isna(row['pdl']):
                continue
            
            # Time filter
            if not row['time_filter']:
                continue
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 8000):
                continue
            
            # Check daily trade limit
            date = row['date']
            if daily_trades.get(date, 0) >= self.get_param('max_trades_per_day', 2):
                continue
            
            # PDH breakout signal
            if (row['pdh_breakout'] and row['volume_spike'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"PDH breakout (PDH: {row['pdh']:.2f})",
                    confidence=0.8,
                    stop_loss=row['pdh'],  # Use PDH as stop
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
                daily_trades[date] = daily_trades.get(date, 0) + 1
            
            # PDL breakout signal
            elif (row['pdl_breakout'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"PDL breakout (PDL: {row['pdl']:.2f})",
                    confidence=0.8,
                    stop_loss=row['pdl'],  # Use PDL as stop
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
                daily_trades[date] = daily_trades.get(date, 0) + 1
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check PDH/PDL entry conditions."""
        if current_idx < 20:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Check if PDH/PDL data available
        if pd.isna(current['pdh']) or pd.isna(current['pdl']):
            return False, "No previous day data"
        
        # Market session check
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Time filter
        if not current['time_filter']:
            return False, "Outside trading time window"
        
        # Volume check
        if not current['volume_spike']:
            return False, "No volume confirmation"
        
        # PDH breakout
        if current['pdh_breakout']:
            return True, f"PDH breakout (PDH: {current['pdh']:.2f})"
        
        # PDL breakout
        if current['pdl_breakout']:
            return True, f"PDL breakout (PDL: {current['pdl']:.2f})"
        
        return False, "No PDH/PDL breakout"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check PDH/PDL exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # End of day exit
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "End of day"
        
        # Return to previous day's range (failed breakout)
        if not pd.isna(current['pdh']) and not pd.isna(current['pdl']):
            if (entry_price > current['pdh'] and 
                current['close'] < current['pdh']):
                return True, "Failed PDH breakout"
            
            if (entry_price < current['pdl'] and 
                current['close'] > current['pdl']):
                return True, "Failed PDL breakout"
        
        return False, "Hold PDH/PDL position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use PDH/PDL as stop loss."""
        current = df.iloc[current_idx]
        
        if not pd.isna(current['pdh']) and not pd.isna(current['pdl']):
            # For long positions, stop at PDH
            if entry_price > current['pdh']:
                return current['pdh']
            # For short positions, stop at PDL
            elif entry_price < current['pdl']:
                return current['pdl']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on previous day's range."""
        current = df.iloc[current_idx]
        
        if ('pd_range' in current and not pd.isna(current['pd_range']) and 
            current['pd_range'] > 0):
            
            range_size = current['pd_range']
            multiplier = self.get_param('target_multiplier', 2.0)
            
            # Target is range_size * multiplier away from entry
            if entry_price > current['pdh']:  # Long breakout
                return entry_price + (range_size * multiplier)
            elif entry_price < current['pdl']:  # Short breakout
                return entry_price - (range_size * multiplier)
        
        # Fallback target
        return entry_price * 1.025