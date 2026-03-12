"""
EMA Crossover Strategy
Classic trend following strategy using exponential moving average crossovers.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy for intraday trend following.
    
    Entry: When fast EMA crosses above slow EMA (bullish) or below (bearish)
    Exit: When EMA crosses back or stop loss/target hit
    """
    
    @property
    def name(self) -> str:
        return "EMA Crossover"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'fast_ema': 9,
            'slow_ema': 21,
            'stop_loss_pct': 1.0,  # 1% stop loss
            'target_pct': 2.0,     # 2% target
            'min_volume': 10000,   # Minimum volume filter
            'trend_filter': True,  # Use 200 EMA as trend filter
            'trend_ema': 200
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA indicators."""
        df = df.copy()
        
        fast_period = self.get_param('fast_ema', 9)
        slow_period = self.get_param('slow_ema', 21)
        trend_period = self.get_param('trend_ema', 200)
        
        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=fast_period).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_period).mean()
        
        if self.get_param('trend_filter', True):
            df['ema_trend'] = df['close'].ewm(span=trend_period).mean()
        
        # Crossover signals
        df['ema_cross_up'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['ema_cross_down'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
        
        # Trend direction
        if self.get_param('trend_filter', True):
            df['uptrend'] = df['close'] > df['ema_trend']
            df['downtrend'] = df['close'] < df['ema_trend']
        else:
            df['uptrend'] = True
            df['downtrend'] = True
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate EMA crossover signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # TODO: Add volume filter and market session checks
            if row['volume'] < self.get_param('min_volume', 10000):
                continue
            
            # Bullish crossover
            if row['ema_cross_up'] and row['uptrend']:
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Fast EMA crossed above Slow EMA in uptrend",
                    confidence=0.7,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish crossover
            elif row['ema_cross_down'] and row['downtrend']:
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Fast EMA crossed below Slow EMA in downtrend",
                    confidence=0.7,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check if we should enter a trade."""
        if current_idx < 1:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # TODO: Add market session validation
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume filter
        if current['volume'] < self.get_param('min_volume', 10000):
            return False, "Volume too low"
        
        # Bullish entry
        if current['ema_cross_up'] and current['uptrend']:
            return True, "Bullish EMA crossover in uptrend"
        
        # Bearish entry  
        if current['ema_cross_down'] and current['downtrend']:
            return True, "Bearish EMA crossover in downtrend"
        
        return False, "No crossover signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check if we should exit a trade."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # TODO: Add time-based exit for intraday
        # Exit before market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing soon"
        
        # Opposite crossover
        if current['ema_cross_up'] or current['ema_cross_down']:
            return True, "Opposite EMA crossover"
        
        # Stop loss / Target hit (handled by risk management)
        return False, "No exit signal"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate stop loss price."""
        stop_loss_pct = self.get_param('stop_loss_pct', 1.0) / 100
        
        # TODO: Use ATR-based stop loss for better risk management
        return entry_price * (1 - stop_loss_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target price."""
        target_pct = self.get_param('target_pct', 2.0) / 100
        
        # TODO: Use support/resistance levels for dynamic targets
        return entry_price * (1 + target_pct)