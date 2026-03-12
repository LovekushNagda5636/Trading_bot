"""
Donchian Channel Breakout Strategy
Trend following strategy using Donchian channels for breakout signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class DonchianBreakoutStrategy(BaseStrategy):
    """
    Donchian Channel Breakout Strategy for trend following.
    
    Entry: When price breaks above/below Donchian channel
    Exit: When price returns to middle line or opposite breakout
    """
    
    @property
    def name(self) -> str:
        return "Donchian Breakout"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_15
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'period': 20,
            'exit_period': 10,
            'min_volume': 8000,
            'atr_period': 14,
            'stop_loss_atr': 2.0,
            'target_atr': 3.0,
            'breakout_confirmation': True
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Donchian Channel indicators."""
        df = df.copy()
        
        period = self.get_param('period', 20)
        exit_period = self.get_param('exit_period', 10)
        atr_period = self.get_param('atr_period', 14)
        
        # Calculate Donchian channels
        df['donchian_high'] = df['high'].rolling(window=period).max()
        df['donchian_low'] = df['low'].rolling(window=period).min()
        df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2
        
        # Exit channels (shorter period)
        df['exit_high'] = df['high'].rolling(window=exit_period).max()
        df['exit_low'] = df['low'].rolling(window=exit_period).min()
        
        # Calculate ATR for stop loss and targets
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
        
        # Generate breakout signals
        df['upper_breakout'] = df['close'] > df['donchian_high'].shift(1)
        df['lower_breakout'] = df['close'] < df['donchian_low'].shift(1)
        
        # Confirmation filters
        if self.get_param('breakout_confirmation', True):
            df['volume_avg'] = df['volume'].rolling(window=20).mean()
            df['volume_confirm'] = df['volume'] > df['volume_avg'] * 1.2
        else:
            df['volume_confirm'] = True
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate Donchian breakout signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 8000):
                continue
            
            # Upper breakout signal
            if (row['upper_breakout'] and row['volume_confirm'] and 
                not pd.isna(row['donchian_high'])):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Donchian upper channel breakout",
                    confidence=0.75,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Lower breakout signal
            elif (row['lower_breakout'] and row['volume_confirm'] and 
                  not pd.isna(row['donchian_low'])):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Donchian lower channel breakout",
                    confidence=0.75,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check Donchian entry conditions."""
        if current_idx < 25:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if current['volume'] < self.get_param('min_volume', 8000):
            return False, "Low volume"
        
        # Upper breakout
        if current['upper_breakout'] and current['volume_confirm']:
            return True, "Donchian upper breakout"
        
        # Lower breakout
        if current['lower_breakout'] and current['volume_confirm']:
            return True, "Donchian lower breakout"
        
        return False, "No Donchian breakout"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check Donchian exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # Exit on opposite breakout
        if current['upper_breakout'] or current['lower_breakout']:
            return True, "Opposite Donchian breakout"
        
        # Exit on return to middle line
        if (entry_price > current['donchian_mid'] and 
            current['close'] < current['donchian_mid']):
            return True, "Returned to Donchian middle"
        
        if (entry_price < current['donchian_mid'] and 
            current['close'] > current['donchian_mid']):
            return True, "Returned to Donchian middle"
        
        return False, "Hold Donchian position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate ATR-based stop loss."""
        current = df.iloc[current_idx]
        
        if 'atr' in current and not pd.isna(current['atr']):
            atr_multiplier = self.get_param('stop_loss_atr', 2.0)
            return entry_price - (current['atr'] * atr_multiplier)
        
        # Fallback
        return entry_price * 0.98
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate ATR-based target."""
        current = df.iloc[current_idx]
        
        if 'atr' in current and not pd.isna(current['atr']):
            atr_multiplier = self.get_param('target_atr', 3.0)
            return entry_price + (current['atr'] * atr_multiplier)
        
        # Fallback
        return entry_price * 1.03