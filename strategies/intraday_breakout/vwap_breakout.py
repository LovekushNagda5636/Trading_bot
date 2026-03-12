"""
VWAP Breakout Strategy
Breakout strategy using Volume Weighted Average Price as key level.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class VWAPBreakoutStrategy(BaseStrategy):
    """
    VWAP Breakout Strategy.
    
    Entry: When price breaks above/below VWAP with volume confirmation
    Exit: When price returns to VWAP or end of day
    """
    
    @property
    def name(self) -> str:
        return "VWAP Breakout"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'vwap_deviation_bands': True,
            'deviation_multiplier': 1.0,
            'volume_threshold': 1.5,
            'min_volume': 8000,
            'breakout_confirmation': 0.05,  # % beyond VWAP for confirmation
            'stop_loss_pct': 0.8,
            'target_deviation': 2.0,  # Target at 2x deviation band
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP and related indicators."""
        df = df.copy()
        
        # Add date column for daily VWAP calculation
        df['date'] = df.index.date
        
        # Calculate VWAP for each day
        df['vwap'] = np.nan
        df['vwap_volume'] = np.nan
        df['vwap_deviation'] = np.nan
        
        for date in df['date'].unique():
            day_mask = df['date'] == date
            day_data = df[day_mask].copy()
            
            if len(day_data) > 0:
                # Calculate typical price
                day_data['typical_price'] = (day_data['high'] + day_data['low'] + day_data['close']) / 3
                
                # Calculate cumulative VWAP
                day_data['cum_volume'] = day_data['volume'].cumsum()
                day_data['cum_tp_volume'] = (day_data['typical_price'] * day_data['volume']).cumsum()
                day_data['vwap'] = day_data['cum_tp_volume'] / day_data['cum_volume']
                
                # Calculate VWAP deviation (standard deviation of price from VWAP)
                if self.get_param('vwap_deviation_bands', True):
                    day_data['price_vwap_diff'] = day_data['typical_price'] - day_data['vwap']
                    day_data['cum_squared_diff'] = (day_data['price_vwap_diff'] ** 2 * day_data['volume']).cumsum()
                    day_data['vwap_variance'] = day_data['cum_squared_diff'] / day_data['cum_volume']
                    day_data['vwap_deviation'] = np.sqrt(day_data['vwap_variance'])
                
                # Update main dataframe
                df.loc[day_mask, 'vwap'] = day_data['vwap']
                df.loc[day_mask, 'vwap_volume'] = day_data['cum_volume']
                if self.get_param('vwap_deviation_bands', True):
                    df.loc[day_mask, 'vwap_deviation'] = day_data['vwap_deviation']
        
        # VWAP bands
        if self.get_param('vwap_deviation_bands', True):
            deviation_mult = self.get_param('deviation_multiplier', 1.0)
            df['vwap_upper'] = df['vwap'] + (deviation_mult * df['vwap_deviation'])
            df['vwap_lower'] = df['vwap'] - (deviation_mult * df['vwap_deviation'])
            df['vwap_upper2'] = df['vwap'] + (2 * deviation_mult * df['vwap_deviation'])
            df['vwap_lower2'] = df['vwap'] - (2 * deviation_mult * df['vwap_deviation'])
        
        # Price position relative to VWAP
        df['above_vwap'] = df['close'] > df['vwap']
        df['below_vwap'] = df['close'] < df['vwap']
        df['vwap_distance_pct'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # Breakout detection
        breakout_conf = self.get_param('breakout_confirmation', 0.05) / 100
        df['vwap_breakout_level_up'] = df['vwap'] * (1 + breakout_conf)
        df['vwap_breakout_level_down'] = df['vwap'] * (1 - breakout_conf)
        
        df['vwap_breakout_up'] = df['close'] > df['vwap_breakout_level_up']
        df['vwap_breakout_down'] = df['close'] < df['vwap_breakout_level_down']
        
        # First breakout detection
        df['first_vwap_break_up'] = df['vwap_breakout_up'] & ~df['vwap_breakout_up'].shift(1)
        df['first_vwap_break_down'] = df['vwap_breakout_down'] & ~df['vwap_breakout_down'].shift(1)
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.5))
        df['relative_volume'] = df['volume'] / df['volume_ma']
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # VWAP slope (momentum)
        df['vwap_slope'] = df['vwap'] - df['vwap'].shift(5)
        df['vwap_rising'] = df['vwap_slope'] > 0
        df['vwap_falling'] = df['vwap_slope'] < 0
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate VWAP breakout signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Skip if no VWAP data
            if pd.isna(row['vwap']):
                continue
            
            # Time filter
            if not row['time_filter']:
                continue
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 8000):
                continue
            
            # VWAP breakout up
            if (row['first_vwap_break_up'] and row['volume_spike'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"VWAP breakout up (VWAP: {row['vwap']:.2f}, Distance: {row['vwap_distance_pct']:.2f}%)",
                    confidence=0.75 + min(row['relative_volume'] * 0.1, 0.15),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # VWAP breakout down
            elif (row['first_vwap_break_down'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"VWAP breakout down (VWAP: {row['vwap']:.2f}, Distance: {row['vwap_distance_pct']:.2f}%)",
                    confidence=0.75 + min(row['relative_volume'] * 0.1, 0.15),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check VWAP breakout entry conditions."""
        if current_idx < 10:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Check VWAP availability
        if pd.isna(current['vwap']):
            return False, "No VWAP data"
        
        # Time filter
        if not current['time_filter']:
            return False, "Outside trading hours"
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if not current['volume_spike']:
            return False, "No volume confirmation"
        
        # VWAP breakout up
        if current['first_vwap_break_up']:
            return True, f"VWAP breakout up (distance: {current['vwap_distance_pct']:.2f}%)"
        
        # VWAP breakout down
        if current['first_vwap_break_down']:
            return True, f"VWAP breakout down (distance: {current['vwap_distance_pct']:.2f}%)"
        
        return False, "No VWAP breakout"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check VWAP breakout exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Return to VWAP (failed breakout)
        if (entry_price > current['vwap'] and 
            current['close'] < current['vwap']):
            return True, "Price returned to VWAP"
        
        if (entry_price < current['vwap'] and 
            current['close'] > current['vwap']):
            return True, "Price returned to VWAP"
        
        # Reached deviation bands (profit taking)
        if (self.get_param('vwap_deviation_bands', True) and 
            not pd.isna(current['vwap_upper2']) and not pd.isna(current['vwap_lower2'])):
            
            if (entry_price > current['vwap'] and 
                current['close'] > current['vwap_upper2']):
                return True, "Reached upper deviation band"
            
            if (entry_price < current['vwap'] and 
                current['close'] < current['vwap_lower2']):
                return True, "Reached lower deviation band"
        
        return False, "Hold VWAP breakout position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use VWAP as stop loss."""
        current = df.iloc[current_idx]
        
        # Use VWAP as dynamic stop
        if not pd.isna(current['vwap']):
            return current['vwap']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 0.8) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use VWAP deviation bands for target."""
        current = df.iloc[current_idx]
        
        # Use deviation bands if available
        if (self.get_param('vwap_deviation_bands', True) and 
            not pd.isna(current['vwap_deviation']) and current['vwap_deviation'] > 0):
            
            target_mult = self.get_param('target_deviation', 2.0)
            deviation = current['vwap_deviation'] * target_mult
            
            if entry_price > current['vwap']:  # Long
                return current['vwap'] + deviation
            else:  # Short
                return current['vwap'] - deviation
        
        # Fallback target
        return entry_price * 1.025