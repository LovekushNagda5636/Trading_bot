"""
Range Breakout Strategy
Generic range breakout strategy for consolidation patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class RangeBreakoutStrategy(BaseStrategy):
    """
    Range Breakout Strategy for consolidation breakouts.
    
    Entry: When price breaks out of defined range with volume
    Exit: When price returns to range or end of day
    """
    
    @property
    def name(self) -> str:
        return "Range Breakout"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'range_period': 20,
            'min_range_pct': 0.5,
            'max_range_pct': 2.5,
            'breakout_confirmation': 0.1,
            'volume_threshold': 1.8,
            'min_volume': 6000,
            'consolidation_bars': 10,
            'stop_loss_pct': 1.0,
            'target_multiplier': 2.5
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate range breakout indicators."""
        df = df.copy()
        
        range_period = self.get_param('range_period', 20)
        consolidation_bars = self.get_param('consolidation_bars', 10)
        
        # Calculate rolling range
        df['range_high'] = df['high'].rolling(window=range_period).max()
        df['range_low'] = df['low'].rolling(window=range_period).min()
        df['range_size'] = df['range_high'] - df['range_low']
        df['range_pct'] = df['range_size'] / df['close'] * 100
        
        # Range validation
        min_range = self.get_param('min_range_pct', 0.5)
        max_range = self.get_param('max_range_pct', 2.5)
        df['valid_range'] = (df['range_pct'] >= min_range) & (df['range_pct'] <= max_range)
        
        # Consolidation detection (price staying within range)
        df['in_range'] = (df['close'] >= df['range_low']) & (df['close'] <= df['range_high'])
        df['consolidation_count'] = 0
        
        for i in range(consolidation_bars, len(df)):
            recent_bars = df.iloc[i-consolidation_bars:i]
            if recent_bars['in_range'].sum() >= consolidation_bars * 0.8:  # 80% of bars in range
                df.loc[df.index[i], 'consolidation_count'] = recent_bars['in_range'].sum()
        
        df['consolidating'] = df['consolidation_count'] >= consolidation_bars * 0.8
        
        # Breakout detection
        breakout_conf = self.get_param('breakout_confirmation', 0.1) / 100
        df['breakout_high_level'] = df['range_high'] * (1 + breakout_conf)
        df['breakout_low_level'] = df['range_low'] * (1 - breakout_conf)
        
        df['range_breakout_high'] = (
            (df['close'] > df['breakout_high_level']) & 
            df['valid_range'] & 
            df['consolidating']
        )
        
        df['range_breakout_low'] = (
            (df['close'] < df['breakout_low_level']) & 
            df['valid_range'] & 
            df['consolidating']
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.8))
        
        # Range strength (how well defined the range is)
        df['range_touches_high'] = (df['high'].rolling(window=range_period) >= df['range_high'] * 0.99).sum()
        df['range_touches_low'] = (df['low'].rolling(window=range_period) <= df['range_low'] * 1.01).sum()
        df['range_strength'] = (df['range_touches_high'] + df['range_touches_low']) / range_period
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate range breakout signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 6000):
                continue
            
            # Range strength filter
            if row['range_strength'] < 0.3:  # Weak range definition
                continue
            
            # High breakout signal
            if (row['range_breakout_high'] and row['volume_spike'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Range high breakout (Range: {row['range_size']:.2f}, Strength: {row['range_strength']:.2f})",
                    confidence=0.7 + (row['range_strength'] * 0.2),
                    stop_loss=row['range_low'],
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Low breakout signal
            elif (row['range_breakout_low'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Range low breakout (Range: {row['range_size']:.2f}, Strength: {row['range_strength']:.2f})",
                    confidence=0.7 + (row['range_strength'] * 0.2),
                    stop_loss=row['range_high'],
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check range breakout entry conditions."""
        if current_idx < 25:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Range validation
        if not current['valid_range']:
            return False, "Invalid range size"
        
        # Consolidation check
        if not current['consolidating']:
            return False, "No consolidation pattern"
        
        # Volume confirmation
        if not current['volume_spike']:
            return False, "No volume confirmation"
        
        # Range strength check
        if current['range_strength'] < 0.3:
            return False, "Weak range definition"
        
        # High breakout
        if current['range_breakout_high']:
            return True, f"Range high breakout (strength: {current['range_strength']:.2f})"
        
        # Low breakout
        if current['range_breakout_low']:
            return True, f"Range low breakout (strength: {current['range_strength']:.2f})"
        
        return False, "No range breakout"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check range breakout exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # Return to range (failed breakout)
        if current['in_range']:
            return True, "Price returned to range"
        
        # Range redefinition (new consolidation forming)
        if current['consolidating'] and current_idx > 0:
            previous = df_with_indicators.iloc[current_idx - 1]
            if (abs(current['range_high'] - previous['range_high']) > current['range_size'] * 0.1 or
                abs(current['range_low'] - previous['range_low']) > current['range_size'] * 0.1):
                return True, "Range redefined"
        
        return False, "Hold range breakout position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use range boundary as stop loss."""
        current = df.iloc[current_idx]
        
        if 'range_high' in current and 'range_low' in current:
            # For long positions, stop at range low
            if entry_price > current['range_high']:
                return current['range_low']
            # For short positions, stop at range high
            elif entry_price < current['range_low']:
                return current['range_high']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on range size."""
        current = df.iloc[current_idx]
        
        if 'range_size' in current and current['range_size'] > 0:
            range_size = current['range_size']
            multiplier = self.get_param('target_multiplier', 2.5)
            
            # Target is range_size * multiplier away from entry
            if entry_price > current['range_high']:  # Long breakout
                return entry_price + (range_size * multiplier)
            elif entry_price < current['range_low']:  # Short breakout
                return entry_price - (range_size * multiplier)
        
        # Fallback target
        return entry_price * 1.025