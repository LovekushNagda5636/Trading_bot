"""
VWAP Reclaim Strategy
Strategy based on price reclaiming VWAP after being below/above it.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class VWAPReclaimStrategy(BaseStrategy):
    """
    VWAP Reclaim Strategy.
    
    Entry: When price reclaims VWAP after being away from it
    Exit: When price fails to hold VWAP or reaches target
    """
    
    @property
    def name(self) -> str:
        return "VWAP Reclaim"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'min_distance_pct': 0.3,  # Minimum distance from VWAP before reclaim
            'reclaim_confirmation_bars': 2,
            'volume_threshold': 1.3,
            'min_volume': 6000,
            'stop_loss_pct': 0.8,
            'target_pct': 1.8,
            'time_filter_start': "10:00",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP reclaim indicators."""
        df = df.copy()
        
        # Add date column for daily VWAP calculation
        df['date'] = df.index.date
        
        # Calculate VWAP for each day
        df['vwap'] = np.nan
        
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
                
                # Update main dataframe
                df.loc[day_mask, 'vwap'] = day_data['vwap']
        
        # Price position relative to VWAP
        df['vwap_distance_pct'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        df['above_vwap'] = df['close'] > df['vwap']
        df['below_vwap'] = df['close'] < df['vwap']
        
        # Track time away from VWAP
        min_distance = self.get_param('min_distance_pct', 0.3)
        df['far_above_vwap'] = df['vwap_distance_pct'] > min_distance
        df['far_below_vwap'] = df['vwap_distance_pct'] < -min_distance
        
        # Reclaim detection
        confirmation_bars = self.get_param('reclaim_confirmation_bars', 2)
        df['vwap_reclaim_up'] = False
        df['vwap_reclaim_down'] = False
        
        for i in range(confirmation_bars, len(df)):
            # Check if price was far below VWAP and now reclaimed
            if (df['above_vwap'].iloc[i] and 
                any(df['far_below_vwap'].iloc[i-confirmation_bars:i])):
                df.loc[df.index[i], 'vwap_reclaim_up'] = True
            
            # Check if price was far above VWAP and now reclaimed below
            if (df['below_vwap'].iloc[i] and 
                any(df['far_above_vwap'].iloc[i-confirmation_bars:i])):
                df.loc[df.index[i], 'vwap_reclaim_down'] = True
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.3))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "10:00")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # VWAP slope (momentum)
        df['vwap_slope'] = df['vwap'] - df['vwap'].shift(3)
        df['vwap_rising'] = df['vwap_slope'] > 0
        df['vwap_falling'] = df['vwap_slope'] < 0
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate VWAP reclaim signals."""
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
            if row['volume'] < self.get_param('min_volume', 6000):
                continue
            
            # VWAP reclaim up (bullish)
            if (row['vwap_reclaim_up'] and row['volume_spike'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"VWAP reclaim up (Distance: {row['vwap_distance_pct']:.2f}%)",
                    confidence=0.75,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # VWAP reclaim down (bearish)
            elif (row['vwap_reclaim_down'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"VWAP reclaim down (Distance: {row['vwap_distance_pct']:.2f}%)",
                    confidence=0.75,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check VWAP reclaim entry conditions."""
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
        
        # VWAP reclaim up
        if current['vwap_reclaim_up']:
            return True, f"VWAP reclaim up (distance: {current['vwap_distance_pct']:.2f}%)"
        
        # VWAP reclaim down
        if current['vwap_reclaim_down']:
            return True, f"VWAP reclaim down (distance: {current['vwap_distance_pct']:.2f}%)"
        
        return False, "No VWAP reclaim"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check VWAP reclaim exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Failed to hold VWAP
        if (entry_price > current['vwap'] and current['below_vwap']):
            return True, "Failed to hold above VWAP"
        
        if (entry_price < current['vwap'] and current['above_vwap']):
            return True, "Failed to hold below VWAP"
        
        # Time-based exit (after 1 hour)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 1:
            return True, "Time-based exit (1 hour)"
        
        return False, "Hold VWAP reclaim position"
    
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
        """Calculate target based on distance from VWAP."""
        current = df.iloc[current_idx]
        
        # Base target
        target_pct = self.get_param('target_pct', 1.8) / 100
        
        # Adjust based on initial distance from VWAP
        if not pd.isna(current['vwap_distance_pct']):
            distance_adjustment = min(abs(current['vwap_distance_pct']) * 0.1, 0.3)
            target_pct *= (1 + distance_adjustment)
        
        return entry_price * (1 + target_pct)