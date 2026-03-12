"""
Engulfing Pattern Strategy
Price action strategy based on bullish and bearish engulfing candlestick patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class EngulfingStrategy(BaseStrategy):
    """
    Engulfing Pattern Strategy.
    
    Entry: When bullish/bearish engulfing pattern forms with volume confirmation
    Exit: When pattern fails or target reached
    """
    
    @property
    def name(self) -> str:
        return "Engulfing Pattern"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'min_body_size_pct': 0.3,  # Minimum body size as % of range
            'engulfing_ratio': 1.1,    # Engulfing body must be 110% of previous
            'volume_threshold': 1.5,
            'min_volume': 6000,
            'trend_filter': True,
            'trend_ema': 50,
            'stop_loss_pct': 1.0,
            'target_pct': 2.0,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate engulfing pattern indicators."""
        df = df.copy()
        
        # Candlestick components
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Body size as percentage of total range
        df['body_pct'] = df['body_size'] / df['total_range'] * 100
        
        # Candle colors
        df['bullish_candle'] = df['close'] > df['open']
        df['bearish_candle'] = df['close'] < df['open']
        
        # Minimum body size filter
        min_body_pct = self.get_param('min_body_size_pct', 0.3)
        df['significant_body'] = df['body_pct'] > min_body_pct
        
        # Engulfing pattern detection
        engulfing_ratio = self.get_param('engulfing_ratio', 1.1)
        
        df['bullish_engulfing'] = (
            df['bullish_candle'] &  # Current candle is bullish
            df['bearish_candle'].shift(1) &  # Previous candle is bearish
            df['significant_body'] &  # Current body is significant
            df['significant_body'].shift(1) &  # Previous body is significant
            (df['open'] < df['close'].shift(1)) &  # Opens below previous close
            (df['close'] > df['open'].shift(1)) &  # Closes above previous open
            (df['body_size'] >= df['body_size'].shift(1) * engulfing_ratio)  # Body engulfs previous
        )
        
        df['bearish_engulfing'] = (
            df['bearish_candle'] &  # Current candle is bearish
            df['bullish_candle'].shift(1) &  # Previous candle is bullish
            df['significant_body'] &  # Current body is significant
            df['significant_body'].shift(1) &  # Previous body is significant
            (df['open'] > df['close'].shift(1)) &  # Opens above previous close
            (df['close'] < df['open'].shift(1)) &  # Closes below previous open
            (df['body_size'] >= df['body_size'].shift(1) * engulfing_ratio)  # Body engulfs previous
        )
        
        # Trend filter
        if self.get_param('trend_filter', True):
            trend_period = self.get_param('trend_ema', 50)
            df['trend_ema'] = df['close'].ewm(span=trend_period).mean()
            df['uptrend'] = df['close'] > df['trend_ema']
            df['downtrend'] = df['close'] < df['trend_ema']
        else:
            df['uptrend'] = True
            df['downtrend'] = True
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.5))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Pattern strength (based on size and volume)
        df['pattern_strength'] = 0.0
        
        for i in range(len(df)):
            if df['bullish_engulfing'].iloc[i] or df['bearish_engulfing'].iloc[i]:
                # Base strength from body size ratio
                body_ratio = df['body_size'].iloc[i] / df['body_size'].iloc[i-1] if i > 0 else 1
                strength = min(body_ratio / 2, 1.0)  # Normalize to 0-1
                
                # Boost for volume
                if df['volume_spike'].iloc[i]:
                    strength *= 1.2
                
                # Boost for trend alignment
                if ((df['bullish_engulfing'].iloc[i] and df['uptrend'].iloc[i]) or
                    (df['bearish_engulfing'].iloc[i] and df['downtrend'].iloc[i])):
                    strength *= 1.1
                
                df.loc[df.index[i], 'pattern_strength'] = min(strength, 1.0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate engulfing pattern signals."""
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
            
            # Bullish engulfing signal
            if (row['bullish_engulfing'] and row['volume_spike'] and 
                row['uptrend'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bullish engulfing pattern (Strength: {row['pattern_strength']:.2f})",
                    confidence=0.7 + (row['pattern_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish engulfing signal
            elif (row['bearish_engulfing'] and row['volume_spike'] and 
                  row['downtrend'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bearish engulfing pattern (Strength: {row['pattern_strength']:.2f})",
                    confidence=0.7 + (row['pattern_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check engulfing pattern entry conditions."""
        if current_idx < 2:
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
        
        # Bullish engulfing
        if current['bullish_engulfing'] and current['uptrend']:
            return True, f"Bullish engulfing (strength: {current['pattern_strength']:.2f})"
        
        # Bearish engulfing
        if current['bearish_engulfing'] and current['downtrend']:
            return True, f"Bearish engulfing (strength: {current['pattern_strength']:.2f})"
        
        return False, "No engulfing pattern"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check engulfing pattern exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Opposite engulfing pattern
        if (entry_price > current['close'] and current['bearish_engulfing']):
            return True, "Opposite bearish engulfing"
        
        if (entry_price < current['close'] and current['bullish_engulfing']):
            return True, "Opposite bullish engulfing"
        
        # Pattern failure (price returned to pattern range)
        if current_idx > 0:
            entry_bar = df_with_indicators.iloc[current_idx - 1] if current_idx > 0 else current
            
            if (entry_price > entry_bar['open'] and 
                current['close'] < entry_bar['open']):
                return True, "Pattern failed - price below entry open"
            
            if (entry_price < entry_bar['open'] and 
                current['close'] > entry_bar['open']):
                return True, "Pattern failed - price above entry open"
        
        # Time-based exit (after 1 hour)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 1:
            return True, "Time-based exit (1 hour)"
        
        return False, "Hold engulfing position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use pattern low/high as stop loss."""
        current = df.iloc[current_idx]
        
        # Use the low/high of the engulfing pattern
        if current_idx > 0:
            previous = df.iloc[current_idx - 1]
            
            if entry_price > current['open']:  # Long position
                # Stop below the low of the pattern
                return min(current['low'], previous['low'])
            else:  # Short position
                # Stop above the high of the pattern
                return max(current['high'], previous['high'])
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on pattern size."""
        current = df.iloc[current_idx]
        
        # Target based on pattern body size
        if 'body_size' in current and current['body_size'] > 0:
            pattern_size = current['body_size']
            target_multiplier = 1.5  # 1.5x pattern size
            
            if entry_price > current['open']:  # Long
                return entry_price + (pattern_size * target_multiplier)
            else:  # Short
                return entry_price - (pattern_size * target_multiplier)
        
        # Fallback target
        target_pct = self.get_param('target_pct', 2.0) / 100
        return entry_price * (1 + target_pct)