"""
MACD Strategy
Moving Average Convergence Divergence strategy for trend following.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class MACDStrategy(BaseStrategy):
    """
    MACD Strategy for trend following and momentum.
    
    Entry: When MACD line crosses above/below signal line
    Exit: When MACD crosses back or divergence occurs
    """
    
    @property
    def name(self) -> str:
        return "MACD"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'fast_ema': 12,
            'slow_ema': 26,
            'signal_ema': 9,
            'min_volume': 7000,
            'histogram_threshold': 0.1,
            'stop_loss_pct': 1.0,
            'target_pct': 2.0,
            'zero_line_filter': True
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators."""
        df = df.copy()
        
        fast_period = self.get_param('fast_ema', 12)
        slow_period = self.get_param('slow_ema', 26)
        signal_period = self.get_param('signal_ema', 9)
        
        # Calculate MACD components
        df['ema_fast'] = df['close'].ewm(span=fast_period).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_period).mean()
        df['macd_line'] = df['ema_fast'] - df['ema_slow']
        df['signal_line'] = df['macd_line'].ewm(span=signal_period).mean()
        df['histogram'] = df['macd_line'] - df['signal_line']
        
        # MACD crossover signals
        df['macd_bullish_cross'] = (df['macd_line'] > df['signal_line']) & (df['macd_line'].shift(1) <= df['signal_line'].shift(1))
        df['macd_bearish_cross'] = (df['macd_line'] < df['signal_line']) & (df['macd_line'].shift(1) >= df['signal_line'].shift(1))
        
        # Zero line analysis
        df['macd_above_zero'] = df['macd_line'] > 0
        df['macd_below_zero'] = df['macd_line'] < 0
        
        # Histogram analysis
        histogram_threshold = self.get_param('histogram_threshold', 0.1)
        df['histogram_increasing'] = df['histogram'] > df['histogram'].shift(1)
        df['histogram_decreasing'] = df['histogram'] < df['histogram'].shift(1)
        df['histogram_strong'] = abs(df['histogram']) > histogram_threshold
        
        # Divergence detection (simplified)
        df['price_higher_high'] = (df['close'] > df['close'].shift(5)) & (df['close'].shift(5) > df['close'].shift(10))
        df['price_lower_low'] = (df['close'] < df['close'].shift(5)) & (df['close'].shift(5) < df['close'].shift(10))
        df['macd_higher_high'] = (df['macd_line'] > df['macd_line'].shift(5)) & (df['macd_line'].shift(5) > df['macd_line'].shift(10))
        df['macd_lower_low'] = (df['macd_line'] < df['macd_line'].shift(5)) & (df['macd_line'].shift(5) < df['macd_line'].shift(10))
        
        df['bullish_divergence'] = df['price_lower_low'] & ~df['macd_lower_low']
        df['bearish_divergence'] = df['price_higher_high'] & ~df['macd_higher_high']
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate MACD signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            if row['volume'] < self.get_param('min_volume', 7000):
                continue
            
            # Bullish MACD crossover
            if row['macd_bullish_cross']:
                # Higher confidence if above zero line
                confidence = 0.8 if row['macd_above_zero'] else 0.7
                
                # Check zero line filter
                if self.get_param('zero_line_filter', True) and row['macd_below_zero']:
                    confidence *= 0.8
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"MACD bullish crossover (MACD: {row['macd_line']:.4f}, Signal: {row['signal_line']:.4f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish MACD crossover
            elif row['macd_bearish_cross']:
                # Higher confidence if below zero line
                confidence = 0.8 if row['macd_below_zero'] else 0.7
                
                # Check zero line filter
                if self.get_param('zero_line_filter', True) and row['macd_above_zero']:
                    confidence *= 0.8
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"MACD bearish crossover (MACD: {row['macd_line']:.4f}, Signal: {row['signal_line']:.4f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check MACD entry conditions."""
        if current_idx < 30:  # Need enough data for MACD
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        if current['volume'] < self.get_param('min_volume', 7000):
            return False, "Low volume"
        
        # Bullish MACD entry
        if current['macd_bullish_cross']:
            context = "above zero" if current['macd_above_zero'] else "below zero"
            return True, f"MACD bullish crossover ({context})"
        
        # Bearish MACD entry
        if current['macd_bearish_cross']:
            context = "below zero" if current['macd_below_zero'] else "above zero"
            return True, f"MACD bearish crossover ({context})"
        
        return False, "No MACD signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check MACD exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # Opposite MACD crossover
        if current['macd_bullish_cross'] or current['macd_bearish_cross']:
            return True, "MACD crossover reversal"
        
        # Histogram divergence (momentum weakening)
        if (entry_price > current['close'] and 
            current['histogram_decreasing'] and current['histogram_strong']):
            return True, "MACD histogram weakening"
        
        if (entry_price < current['close'] and 
            current['histogram_increasing'] and current['histogram_strong']):
            return True, "MACD histogram strengthening against position"
        
        return False, "Hold MACD position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate stop loss based on recent swing."""
        current = df.iloc[current_idx]
        
        # Use recent swing low/high
        if current_idx >= 10:
            recent_data = df.iloc[current_idx-10:current_idx]
            
            if entry_price > current['close']:  # Long position
                return recent_data['low'].min()
            else:  # Short position
                return recent_data['high'].max()
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on MACD momentum."""
        current = df.iloc[current_idx]
        
        # Base target
        target_pct = self.get_param('target_pct', 2.0) / 100
        
        # Adjust based on histogram strength
        if 'histogram' in current and abs(current['histogram']) > 0.1:
            # Stronger histogram = higher target
            histogram_adjustment = min(abs(current['histogram']) * 5, 0.5)  # Max 50% adjustment
            target_pct *= (1 + histogram_adjustment)
        
        return entry_price * (1 + target_pct)