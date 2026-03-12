"""
Opening Range Breakout (ORB) Strategy
Classic intraday strategy trading breakouts from opening range.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class ORBStrategy(BaseStrategy):
    """
    Opening Range Breakout Strategy.
    
    Entry: When price breaks above/below the opening range
    Exit: End of day or when price returns to opening range
    """
    
    @property
    def name(self) -> str:
        return "Opening Range Breakout"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'orb_minutes': 15,  # First 15 minutes for range
            'min_range_pct': 0.5,  # Minimum range as % of price
            'max_range_pct': 3.0,  # Maximum range as % of price
            'breakout_confirmation': 0.1,  # % above/below range for confirmation
            'stop_loss_pct': 1.0,
            'target_multiplier': 2.0,  # Target = range_size * multiplier
            'volume_threshold': 1.5,
            'max_trades_per_day': 2
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ORB indicators."""
        df = df.copy()
        
        orb_minutes = self.get_param('orb_minutes', 15)
        
        # Identify market open (9:15 AM)
        df['market_open'] = df.index.time == pd.Timestamp("09:15").time()
        df['orb_period'] = False
        
        # TODO: Mark ORB period (first 15 minutes after market open)
        for i in range(len(df)):
            time_now = df.index[i].time()
            if pd.Timestamp("09:15").time() <= time_now <= pd.Timestamp("09:30").time():
                df.iloc[i, df.columns.get_loc('orb_period')] = True
        
        # Calculate opening range for each day
        df['date'] = df.index.date
        df['orb_high'] = np.nan
        df['orb_low'] = np.nan
        df['orb_range'] = np.nan
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            orb_data = day_data[day_data['orb_period']]
            
            if len(orb_data) > 0:
                orb_high = orb_data['high'].max()
                orb_low = orb_data['low'].min()
                orb_range = orb_high - orb_low
                
                # Apply range to entire day
                day_mask = df['date'] == date
                df.loc[day_mask, 'orb_high'] = orb_high
                df.loc[day_mask, 'orb_low'] = orb_low
                df.loc[day_mask, 'orb_range'] = orb_range
        
        # Range validation
        min_range = self.get_param('min_range_pct', 0.5) / 100
        max_range = self.get_param('max_range_pct', 3.0) / 100
        
        df['range_pct'] = df['orb_range'] / df['close']
        df['valid_range'] = (df['range_pct'] >= min_range) & (df['range_pct'] <= max_range)
        
        # Breakout detection
        breakout_conf = self.get_param('breakout_confirmation', 0.1) / 100
        df['breakout_buffer_high'] = df['orb_high'] * (1 + breakout_conf)
        df['breakout_buffer_low'] = df['orb_low'] * (1 - breakout_conf)
        
        df['breakout_high'] = (df['close'] > df['breakout_buffer_high']) & df['valid_range']
        df['breakout_low'] = (df['close'] < df['breakout_buffer_low']) & df['valid_range']
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.5))
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate ORB signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        daily_trades = {}  # Track trades per day
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Skip ORB period
            if row['orb_period']:
                continue
            
            # Check daily trade limit
            date = row['date']
            if daily_trades.get(date, 0) >= self.get_param('max_trades_per_day', 2):
                continue
            
            # TODO: Add market session validation
            if not self.is_market_open(row.name):
                continue
            
            # High breakout signal
            if (row['breakout_high'] and row['volume_spike'] and 
                not row['orb_period']):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"ORB high breakout (Range: {row['orb_range']:.2f})",
                    confidence=0.75,
                    stop_loss=row['orb_low'],
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
                daily_trades[date] = daily_trades.get(date, 0) + 1
            
            # Low breakout signal
            elif (row['breakout_low'] and row['volume_spike'] and 
                  not row['orb_period']):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"ORB low breakout (Range: {row['orb_range']:.2f})",
                    confidence=0.75,
                    stop_loss=row['orb_high'],
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
                daily_trades[date] = daily_trades.get(date, 0) + 1
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check ORB entry conditions."""
        if current_idx < 20:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Skip during ORB period
        if current['orb_period']:
            return False, "During ORB period"
        
        # Market session check
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Range validation
        if not current['valid_range']:
            return False, "Invalid opening range"
        
        # Volume confirmation
        if not current['volume_spike']:
            return False, "No volume confirmation"
        
        # High breakout
        if current['breakout_high']:
            return True, "ORB high breakout"
        
        # Low breakout
        if current['breakout_low']:
            return True, "ORB low breakout"
        
        return False, "No ORB breakout"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check ORB exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # End of day exit
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "End of day"
        
        # Return to opening range (failed breakout)
        if (entry_price > current['orb_high'] and 
            current['close'] < current['orb_high']):
            return True, "Failed breakout - returned to range"
        
        if (entry_price < current['orb_low'] and 
            current['close'] > current['orb_low']):
            return True, "Failed breakout - returned to range"
        
        # TODO: Add time-based exit (e.g., exit after 2 hours)
        
        return False, "Hold ORB position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use opposite side of opening range as stop loss."""
        current = df.iloc[current_idx]
        
        if 'orb_high' in current and 'orb_low' in current:
            # For long positions, stop at ORB low
            if entry_price > current['orb_high']:
                return current['orb_low']
            # For short positions, stop at ORB high
            elif entry_price < current['orb_low']:
                return current['orb_high']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on opening range size."""
        current = df.iloc[current_idx]
        
        if 'orb_range' in current and current['orb_range'] > 0:
            range_size = current['orb_range']
            multiplier = self.get_param('target_multiplier', 2.0)
            
            # Target is range_size * multiplier away from entry
            if entry_price > current['orb_high']:  # Long breakout
                return entry_price + (range_size * multiplier)
            elif entry_price < current['orb_low']:  # Short breakout
                return entry_price - (range_size * multiplier)
        
        # Fallback target
        return entry_price * 1.02