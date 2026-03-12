"""
Moving Average Ribbon Strategy
Trend following strategy using multiple moving averages to identify strong trends.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class MARibbonStrategy(BaseStrategy):
    """
    Moving Average Ribbon Strategy for trend identification.
    
    Entry: When price is above/below all MAs and MAs are properly aligned
    Exit: When MA alignment breaks or price crosses back through ribbon
    """
    
    @property
    def name(self) -> str:
        return "MA Ribbon"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'ma_periods': [5, 8, 13, 21, 34, 55],  # Fibonacci-based periods
            'ma_type': 'EMA',  # 'SMA' or 'EMA'
            'min_volume': 8000,
            'ribbon_separation_min': 0.2,  # Minimum % separation between MAs
            'stop_loss_pct': 0.8,
            'target_pct': 2.0,
            'trend_strength_threshold': 0.8
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Moving Average Ribbon indicators."""
        df = df.copy()
        
        ma_periods = self.get_param('ma_periods', [5, 8, 13, 21, 34, 55])
        ma_type = self.get_param('ma_type', 'EMA')
        
        # Calculate multiple moving averages
        ma_columns = []
        for period in ma_periods:
            col_name = f'ma_{period}'
            ma_columns.append(col_name)
            
            if ma_type == 'EMA':
                df[col_name] = df['close'].ewm(span=period).mean()
            else:  # SMA
                df[col_name] = df['close'].rolling(window=period).mean()
        
        # Calculate ribbon properties
        df['ma_min'] = df[ma_columns].min(axis=1)
        df['ma_max'] = df[ma_columns].max(axis=1)
        df['ribbon_width'] = (df['ma_max'] - df['ma_min']) / df['close'] * 100
        
        # Check MA alignment (bullish: fast > slow, bearish: fast < slow)
        df['bullish_alignment'] = True
        df['bearish_alignment'] = True
        
        for i in range(len(ma_periods) - 1):
            fast_ma = f'ma_{ma_periods[i]}'
            slow_ma = f'ma_{ma_periods[i + 1]}'
            
            df['bullish_alignment'] &= (df[fast_ma] > df[slow_ma])
            df['bearish_alignment'] &= (df[fast_ma] < df[slow_ma])
        
        # Price position relative to ribbon
        df['above_ribbon'] = df['close'] > df['ma_max']
        df['below_ribbon'] = df['close'] < df['ma_min']
        df['in_ribbon'] = ~(df['above_ribbon'] | df['below_ribbon'])
        
        # Ribbon separation (trend strength)
        min_separation = self.get_param('ribbon_separation_min', 0.2)
        df['ribbon_separated'] = df['ribbon_width'] >= min_separation
        
        # Trend strength calculation
        df['trend_strength'] = 0.0
        
        for i in range(len(df)):
            if df['bullish_alignment'].iloc[i] and df['above_ribbon'].iloc[i]:
                # Calculate how far price is above ribbon
                distance = (df['close'].iloc[i] - df['ma_max'].iloc[i]) / df['close'].iloc[i]
                df.loc[df.index[i], 'trend_strength'] = min(distance * 10, 1.0)  # Normalize to 0-1
            elif df['bearish_alignment'].iloc[i] and df['below_ribbon'].iloc[i]:
                # Calculate how far price is below ribbon
                distance = (df['ma_min'].iloc[i] - df['close'].iloc[i]) / df['close'].iloc[i]
                df.loc[df.index[i], 'trend_strength'] = min(distance * 10, 1.0)  # Normalize to 0-1
        
        # Entry signals
        strength_threshold = self.get_param('trend_strength_threshold', 0.8)
        
        df['ribbon_bullish'] = (
            df['above_ribbon'] &
            df['bullish_alignment'] &
            df['ribbon_separated'] &
            (df['trend_strength'] >= strength_threshold)
        )
        
        df['ribbon_bearish'] = (
            df['below_ribbon'] &
            df['bearish_alignment'] &
            df['ribbon_separated'] &
            (df['trend_strength'] >= strength_threshold)
        )
        
        # Breakout signals (price crossing ribbon)
        df['ribbon_breakout_up'] = (
            df['above_ribbon'] & 
            df['in_ribbon'].shift(1) &
            df['bullish_alignment']
        )
        
        df['ribbon_breakout_down'] = (
            df['below_ribbon'] & 
            df['in_ribbon'].shift(1) &
            df['bearish_alignment']
        )
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate MA Ribbon signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 8000):
                continue
            
            # Bullish ribbon signal
            if row['ribbon_bullish'] or row['ribbon_breakout_up']:
                reason = "Strong bullish ribbon alignment" if row['ribbon_bullish'] else "Ribbon breakout upward"
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=reason,
                    confidence=min(0.6 + row['trend_strength'] * 0.3, 0.9),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish ribbon signal
            elif row['ribbon_bearish'] or row['ribbon_breakout_down']:
                reason = "Strong bearish ribbon alignment" if row['ribbon_bearish'] else "Ribbon breakout downward"
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=reason,
                    confidence=min(0.6 + row['trend_strength'] * 0.3, 0.9),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check MA Ribbon entry conditions."""
        if current_idx < 60:  # Need enough data for longest MA
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if current['volume'] < self.get_param('min_volume', 8000):
            return False, "Low volume"
        
        # Bullish ribbon entry
        if current['ribbon_bullish'] or current['ribbon_breakout_up']:
            return True, f"MA Ribbon bullish (strength: {current['trend_strength']:.2f})"
        
        # Bearish ribbon entry
        if current['ribbon_bearish'] or current['ribbon_breakout_down']:
            return True, f"MA Ribbon bearish (strength: {current['trend_strength']:.2f})"
        
        return False, "No MA Ribbon signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check MA Ribbon exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # Price returned to ribbon (trend weakening)
        if current['in_ribbon']:
            return True, "Price returned to MA ribbon"
        
        # MA alignment broken
        if not current['bullish_alignment'] and not current['bearish_alignment']:
            return True, "MA alignment broken"
        
        # Trend strength weakened significantly
        if current['trend_strength'] < 0.3:
            return True, "Trend strength weakened"
        
        return False, "Hold MA Ribbon position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use ribbon boundary as stop loss."""
        current = df.iloc[current_idx]
        
        # Use ribbon boundary as dynamic stop
        if 'ma_max' in current and 'ma_min' in current:
            if entry_price > current['ma_max']:  # Long position
                return current['ma_max']
            elif entry_price < current['ma_min']:  # Short position
                return current['ma_min']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 0.8) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on ribbon width."""
        current = df.iloc[current_idx]
        
        if 'ribbon_width' in current and current['ribbon_width'] > 0:
            # Target based on ribbon width (trend momentum)
            ribbon_size = (current['ma_max'] - current['ma_min'])
            target_multiplier = 1.5
            
            if entry_price > current['ma_max']:  # Long
                return entry_price + (ribbon_size * target_multiplier)
            elif entry_price < current['ma_min']:  # Short
                return entry_price - (ribbon_size * target_multiplier)
        
        # Fallback target
        target_pct = self.get_param('target_pct', 2.0) / 100
        return entry_price * (1 + target_pct)