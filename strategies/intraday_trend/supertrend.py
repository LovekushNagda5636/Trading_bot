"""
SuperTrend Strategy
Trend following strategy using SuperTrend indicator for entry and exit signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class SuperTrendStrategy(BaseStrategy):
    """
    SuperTrend Strategy for intraday trend following.
    
    Entry: When price crosses above/below SuperTrend line
    Exit: When SuperTrend changes direction or market close
    """
    
    @property
    def name(self) -> str:
        return "SuperTrend"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 10,
            'multiplier': 3.0,
            'min_volume': 5000,
            'trend_filter': True,
            'trend_ema': 200,
            'stop_loss_pct': 0.5,  # Tight stop as SuperTrend acts as trailing stop
            'target_rr': 2.0       # Risk-reward ratio
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SuperTrend indicator."""
        df = df.copy()
        
        atr_period = self.get_param('atr_period', 10)
        multiplier = self.get_param('multiplier', 3.0)
        
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
        
        # Calculate SuperTrend
        df['hl2'] = (df['high'] + df['low']) / 2
        df['upper_band'] = df['hl2'] + (multiplier * df['atr'])
        df['lower_band'] = df['hl2'] - (multiplier * df['atr'])
        
        # SuperTrend calculation
        df['supertrend'] = np.nan
        df['supertrend_direction'] = np.nan
        
        for i in range(1, len(df)):
            # Upper band
            if df['upper_band'].iloc[i] < df['upper_band'].iloc[i-1] or df['close'].iloc[i-1] > df['upper_band'].iloc[i-1]:
                df.loc[df.index[i], 'upper_band'] = df['upper_band'].iloc[i]
            else:
                df.loc[df.index[i], 'upper_band'] = df['upper_band'].iloc[i-1]
            
            # Lower band
            if df['lower_band'].iloc[i] > df['lower_band'].iloc[i-1] or df['close'].iloc[i-1] < df['lower_band'].iloc[i-1]:
                df.loc[df.index[i], 'lower_band'] = df['lower_band'].iloc[i]
            else:
                df.loc[df.index[i], 'lower_band'] = df['lower_band'].iloc[i-1]
            
            # SuperTrend
            if pd.isna(df['supertrend'].iloc[i-1]):
                df.loc[df.index[i], 'supertrend'] = df['upper_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1
            elif df['supertrend'].iloc[i-1] == df['upper_band'].iloc[i-1] and df['close'].iloc[i] <= df['upper_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['upper_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1
            elif df['supertrend'].iloc[i-1] == df['upper_band'].iloc[i-1] and df['close'].iloc[i] > df['upper_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['lower_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1
            elif df['supertrend'].iloc[i-1] == df['lower_band'].iloc[i-1] and df['close'].iloc[i] >= df['lower_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['lower_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1
            elif df['supertrend'].iloc[i-1] == df['lower_band'].iloc[i-1] and df['close'].iloc[i] < df['lower_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['upper_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1
        
        # Signal generation
        df['st_bullish'] = (df['close'] > df['supertrend']) & (df['close'].shift(1) <= df['supertrend'].shift(1))
        df['st_bearish'] = (df['close'] < df['supertrend']) & (df['close'].shift(1) >= df['supertrend'].shift(1))
        
        # Trend filter
        if self.get_param('trend_filter', True):
            trend_period = self.get_param('trend_ema', 200)
            df['trend_ema'] = df['close'].ewm(span=trend_period).mean()
            df['uptrend'] = df['close'] > df['trend_ema']
            df['downtrend'] = df['close'] < df['trend_ema']
        else:
            df['uptrend'] = True
            df['downtrend'] = True
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate SuperTrend signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 5000):
                continue
            
            # TODO: Add market session and volatility filters
            
            # Bullish SuperTrend signal
            if row['st_bullish'] and row['uptrend']:
                signal = Signal(
                    action="BUY",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Price crossed above SuperTrend in uptrend",
                    confidence=0.8,
                    stop_loss=row['supertrend'],  # Use SuperTrend as stop
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish SuperTrend signal
            elif row['st_bearish'] and row['downtrend']:
                signal = Signal(
                    action="SELL",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Price crossed below SuperTrend in downtrend",
                    confidence=0.8,
                    stop_loss=row['supertrend'],  # Use SuperTrend as stop
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check SuperTrend entry conditions."""
        if current_idx < 20:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if current['volume'] < self.get_param('min_volume', 5000):
            return False, "Low volume"
        
        # Bullish entry
        if current['st_bullish'] and current['uptrend']:
            return True, "SuperTrend bullish crossover"
        
        # Bearish entry
        if current['st_bearish'] and current['downtrend']:
            return True, "SuperTrend bearish crossover"
        
        return False, "No SuperTrend signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check SuperTrend exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # SuperTrend direction change
        if current['st_bullish'] or current['st_bearish']:
            return True, "SuperTrend direction change"
        
        # TODO: Add time-based exit and profit booking rules
        
        return False, "Hold with SuperTrend"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use SuperTrend as dynamic stop loss."""
        current = df.iloc[current_idx]
        
        # SuperTrend acts as trailing stop
        if 'supertrend' in current:
            return current['supertrend']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 0.5) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on ATR and risk-reward ratio."""
        current = df.iloc[current_idx]
        
        if 'atr' in current and 'supertrend' in current:
            risk = abs(entry_price - current['supertrend'])
            reward = risk * self.get_param('target_rr', 2.0)
            
            if entry_price > current['supertrend']:  # Long
                return entry_price + reward
            else:  # Short
                return entry_price - reward
        
        # Fallback target
        return entry_price * 1.02  # 2% target