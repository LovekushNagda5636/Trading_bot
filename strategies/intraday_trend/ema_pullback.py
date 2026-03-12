"""
EMA Pullback Strategy
Trend following strategy that enters on pullbacks to EMA in trending markets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class EMAPullbackStrategy(BaseStrategy):
    """
    EMA Pullback Strategy for intraday trend following.
    
    Entry: When price pulls back to EMA in a trending market and bounces
    Exit: When trend reverses or stop loss/target hit
    """
    
    @property
    def name(self) -> str:
        return "EMA Pullback"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_15
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'ema_period': 20,
            'trend_ema': 50,
            'pullback_threshold': 0.5,  # % pullback to EMA
            'bounce_confirmation': 2,   # Bars to confirm bounce
            'stop_loss_pct': 1.5,
            'target_pct': 3.0,
            'min_trend_strength': 0.02,  # Minimum trend strength
            'volume_multiplier': 1.2     # Volume confirmation
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pullback indicators."""
        df = df.copy()
        
        ema_period = self.get_param('ema_period', 20)
        trend_period = self.get_param('trend_ema', 50)
        
        # EMAs
        df['ema'] = df['close'].ewm(span=ema_period).mean()
        df['trend_ema'] = df['close'].ewm(span=trend_period).mean()
        
        # Trend direction and strength
        df['trend_up'] = df['close'] > df['trend_ema']
        df['trend_down'] = df['close'] < df['trend_ema']
        df['trend_strength'] = abs(df['close'] - df['trend_ema']) / df['trend_ema']
        
        # Pullback detection
        df['distance_to_ema'] = (df['close'] - df['ema']) / df['ema']
        df['near_ema'] = abs(df['distance_to_ema']) < (self.get_param('pullback_threshold', 0.5) / 100)
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_multiplier', 1.2))
        
        # Bounce confirmation
        df['bounce_up'] = (df['close'] > df['ema']) & (df['close'].shift(1) <= df['ema'].shift(1))
        df['bounce_down'] = (df['close'] < df['ema']) & (df['close'].shift(1) >= df['ema'].shift(1))
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate pullback signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # TODO: Add market session and liquidity filters
            if row['trend_strength'] < self.get_param('min_trend_strength', 0.02):
                continue
            
            # Bullish pullback
            if (row['trend_up'] and row['bounce_up'] and 
                row['volume_spike'] and i > 0):
                
                signal = Signal(
                    action="BUY",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Bullish bounce from EMA in uptrend",
                    confidence=0.7,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish pullback
            elif (row['trend_down'] and row['bounce_down'] and 
                  row['volume_spike'] and i > 0):
                
                signal = Signal(
                    action="SELL",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Bearish bounce from EMA in downtrend",
                    confidence=0.7,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check if we should enter a pullback trade."""
        if current_idx < 20:
            return False, "Insufficient data for trend analysis"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market session check
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Trend strength filter
        if current['trend_strength'] < self.get_param('min_trend_strength', 0.02):
            return False, "Trend too weak"
        
        # Bullish pullback entry
        if (current['trend_up'] and current['bounce_up'] and current['volume_spike']):
            return True, "Bullish pullback entry"
        
        # Bearish pullback entry
        if (current['trend_down'] and current['bounce_down'] and current['volume_spike']):
            return True, "Bearish pullback entry"
        
        return False, "No pullback setup"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check if we should exit pullback trade."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # TODO: Add trailing stop based on EMA
        # Market close exit
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Trend reversal
        if current['trend_strength'] < (self.get_param('min_trend_strength', 0.02) / 2):
            return True, "Trend weakening"
        
        return False, "Hold position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate stop loss below/above EMA."""
        current = df.iloc[current_idx]
        ema_value = current['ema']
        
        # TODO: Use ATR for dynamic stop loss
        stop_loss_pct = self.get_param('stop_loss_pct', 1.5) / 100
        
        # Place stop below EMA for long, above EMA for short
        if entry_price > ema_value:  # Long position
            return min(ema_value * 0.995, entry_price * (1 - stop_loss_pct))
        else:  # Short position
            return max(ema_value * 1.005, entry_price * (1 + stop_loss_pct))
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on trend strength."""
        current = df.iloc[current_idx]
        target_pct = self.get_param('target_pct', 3.0) / 100
        
        # TODO: Use previous swing highs/lows as targets
        trend_multiplier = min(current['trend_strength'] * 10, 2.0)
        adjusted_target = target_pct * trend_multiplier
        
        return entry_price * (1 + adjusted_target)