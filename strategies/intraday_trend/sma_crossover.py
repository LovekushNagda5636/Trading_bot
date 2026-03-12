"""
SMA Crossover Strategy
Simple Moving Average crossover strategy for trend following.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class SMACrossoverStrategy(BaseStrategy):
    """
    SMA Crossover Strategy for trend following.
    
    Entry: When fast SMA crosses above/below slow SMA
    Exit: When SMA crosses back or stop loss/target hit
    """
    
    @property
    def name(self) -> str:
        return "SMA Crossover"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'fast_sma': 10,
            'slow_sma': 20,
            'stop_loss_pct': 1.0,
            'target_pct': 2.0,
            'min_volume': 8000,
            'trend_filter': True,
            'trend_sma': 50
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA indicators."""
        df = df.copy()
        
        fast_period = self.get_param('fast_sma', 10)
        slow_period = self.get_param('slow_sma', 20)
        trend_period = self.get_param('trend_sma', 50)
        
        # Calculate SMAs
        df['sma_fast'] = df['close'].rolling(window=fast_period).mean()
        df['sma_slow'] = df['close'].rolling(window=slow_period).mean()
        
        if self.get_param('trend_filter', True):
            df['sma_trend'] = df['close'].rolling(window=trend_period).mean()
        
        # Crossover signals
        df['sma_cross_up'] = (df['sma_fast'] > df['sma_slow']) & (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1))
        df['sma_cross_down'] = (df['sma_fast'] < df['sma_slow']) & (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1))
        
        # Trend direction
        if self.get_param('trend_filter', True):
            df['uptrend'] = df['close'] > df['sma_trend']
            df['downtrend'] = df['close'] < df['sma_trend']
        else:
            df['uptrend'] = True
            df['downtrend'] = True
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate SMA crossover signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            if row['volume'] < self.get_param('min_volume', 8000):
                continue
            
            # Bullish crossover
            if row['sma_cross_up'] and row['uptrend']:
                signal = Signal(
                    action="BUY",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Fast SMA crossed above Slow SMA in uptrend",
                    confidence=0.7,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish crossover
            elif row['sma_cross_down'] and row['downtrend']:
                signal = Signal(
                    action="SELL",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Fast SMA crossed below Slow SMA in downtrend",
                    confidence=0.7,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check SMA crossover entry conditions."""
        if current_idx < 1:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        if current['volume'] < self.get_param('min_volume', 8000):
            return False, "Volume too low"
        
        # Bullish entry
        if current['sma_cross_up'] and current['uptrend']:
            return True, "Bullish SMA crossover in uptrend"
        
        # Bearish entry  
        if current['sma_cross_down'] and current['downtrend']:
            return True, "Bearish SMA crossover in downtrend"
        
        return False, "No crossover signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check SMA crossover exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing soon"
        
        # Opposite crossover
        if current['sma_cross_up'] or current['sma_cross_down']:
            return True, "Opposite SMA crossover"
        
        return False, "No exit signal"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate stop loss price."""
        stop_loss_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_loss_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target price."""
        target_pct = self.get_param('target_pct', 2.0) / 100
        return entry_price * (1 + target_pct)