"""
ATR Breakout Strategy
Breakout strategy using Average True Range for dynamic breakout levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class ATRBreakoutStrategy(BaseStrategy):
    """
    ATR Breakout Strategy using dynamic volatility-based levels.
    
    Entry: When price moves beyond ATR-based breakout levels
    Exit: When price returns to base or opposite breakout occurs
    """
    
    @property
    def name(self) -> str:
        return "ATR Breakout"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'breakout_multiplier': 1.5,
            'base_period': 20,
            'volume_threshold': 1.6,
            'min_volume': 6000,
            'stop_loss_atr': 1.0,
            'target_atr': 2.5,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR breakout indicators."""
        df = df.copy()
        
        atr_period = self.get_param('atr_period', 14)
        base_period = self.get_param('base_period', 20)
        breakout_multiplier = self.get_param('breakout_multiplier', 1.5)
        
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
        
        # Calculate base levels (using EMA for smoother base)
        df['base_high'] = df['high'].rolling(window=base_period).max()
        df['base_low'] = df['low'].rolling(window=base_period).min()
        df['base_middle'] = (df['base_high'] + df['base_low']) / 2
        
        # Alternative: Use EMA as base
        df['ema_base'] = df['close'].ewm(span=base_period).mean()
        
        # ATR-based breakout levels from base
        df['atr_breakout_high'] = df['ema_base'] + (breakout_multiplier * df['atr'])
        df['atr_breakout_low'] = df['ema_base'] - (breakout_multiplier * df['atr'])
        
        # Breakout detection
        df['atr_break_up'] = df['close'] > df['atr_breakout_high']
        df['atr_break_down'] = df['close'] < df['atr_breakout_low']
        
        # First breakout detection (avoid multiple signals)
        df['first_break_up'] = df['atr_break_up'] & ~df['atr_break_up'].shift(1)
        df['first_break_down'] = df['atr_break_down'] & ~df['atr_break_down'].shift(1)
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.6))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Volatility state
        df['atr_percentile'] = df['atr'].rolling(window=50).rank(pct=True)
        df['high_volatility'] = df['atr_percentile'] > 0.7
        df['low_volatility'] = df['atr_percentile'] < 0.3
        
        # Trend context
        df['trend_ema'] = df['close'].ewm(span=50).mean()
        df['uptrend'] = df['close'] > df['trend_ema']
        df['downtrend'] = df['close'] < df['trend_ema']
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate ATR breakout signals."""
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
            
            # ATR breakout up
            if (row['first_break_up'] and row['volume_spike'] and 
                self.is_market_open(row.name)):
                
                # Higher confidence in trending markets
                confidence = 0.8 if row['uptrend'] else 0.7
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"ATR breakout up (ATR: {row['atr']:.2f}, Level: {row['atr_breakout_high']:.2f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # ATR breakout down
            elif (row['first_break_down'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                # Higher confidence in trending markets
                confidence = 0.8 if row['downtrend'] else 0.7
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"ATR breakout down (ATR: {row['atr']:.2f}, Level: {row['atr_breakout_low']:.2f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check ATR breakout entry conditions."""
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
        
        # ATR breakout up
        if current['first_break_up']:
            trend_context = "with trend" if current['uptrend'] else "against trend"
            return True, f"ATR breakout up ({trend_context}, ATR: {current['atr']:.2f})"
        
        # ATR breakout down
        if current['first_break_down']:
            trend_context = "with trend" if current['downtrend'] else "against trend"
            return True, f"ATR breakout down ({trend_context}, ATR: {current['atr']:.2f})"
        
        return False, "No ATR breakout"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check ATR breakout exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Return to base (failed breakout)
        if (entry_price > current['ema_base'] and 
            current['close'] < current['ema_base']):
            return True, "Price returned to base level"
        
        if (entry_price < current['ema_base'] and 
            current['close'] > current['ema_base']):
            return True, "Price returned to base level"
        
        # Opposite ATR breakout
        if (entry_price > current['ema_base'] and 
            current['first_break_down']):
            return True, "Opposite ATR breakout"
        
        if (entry_price < current['ema_base'] and 
            current['first_break_up']):
            return True, "Opposite ATR breakout"
        
        # Time-based exit (after 2 hours)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 2:
            return True, "Time-based exit (2 hours)"
        
        return False, "Hold ATR breakout position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use ATR-based stop loss."""
        current = df.iloc[current_idx]
        
        if 'atr' in current and not pd.isna(current['atr']):
            atr_multiplier = self.get_param('stop_loss_atr', 1.0)
            return entry_price - (current['atr'] * atr_multiplier)
        
        # Fallback to base level
        if 'ema_base' in current:
            return current['ema_base']
        
        return entry_price * 0.985
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use ATR-based target."""
        current = df.iloc[current_idx]
        
        if 'atr' in current and not pd.isna(current['atr']):
            atr_multiplier = self.get_param('target_atr', 2.5)
            return entry_price + (current['atr'] * atr_multiplier)
        
        return entry_price * 1.025