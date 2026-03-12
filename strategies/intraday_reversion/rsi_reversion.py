"""
RSI Mean Reversion Strategy
Mean reversion strategy using RSI oversold/overbought levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class RSIReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy.
    
    Entry: When RSI reaches extreme levels and shows reversal signs
    Exit: When RSI returns to neutral zone or continues extreme
    """
    
    @property
    def name(self) -> str:
        return "RSI Reversion"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'rsi_period': 14,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'rsi_extreme_oversold': 15,
            'rsi_extreme_overbought': 85,
            'rsi_exit_neutral': 50,
            'volume_threshold': 1.2,
            'min_volume': 5000,
            'stop_loss_pct': 1.5,
            'target_pct': 2.0,
            'divergence_lookback': 10,
            'time_filter_start': "10:00",
            'time_filter_end': "14:30"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI reversion indicators."""
        df = df.copy()
        
        rsi_period = self.get_param('rsi_period', 14)
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI levels
        rsi_oversold = self.get_param('rsi_oversold', 25)
        rsi_overbought = self.get_param('rsi_overbought', 75)
        rsi_extreme_oversold = self.get_param('rsi_extreme_oversold', 15)
        rsi_extreme_overbought = self.get_param('rsi_extreme_overbought', 85)
        
        df['rsi_oversold'] = df['rsi'] < rsi_oversold
        df['rsi_overbought'] = df['rsi'] > rsi_overbought
        df['rsi_extreme_oversold'] = df['rsi'] < rsi_extreme_oversold
        df['rsi_extreme_overbought'] = df['rsi'] > rsi_extreme_overbought
        
        # RSI momentum and reversal signals
        df['rsi_rising'] = df['rsi'] > df['rsi'].shift(1)
        df['rsi_falling'] = df['rsi'] < df['rsi'].shift(1)
        
        # Reversal detection
        df['rsi_bullish_reversal'] = (
            df['rsi_oversold'] & 
            df['rsi_rising'] & 
            (df['close'] > df['close'].shift(1))
        )
        
        df['rsi_bearish_reversal'] = (
            df['rsi_overbought'] & 
            df['rsi_falling'] & 
            (df['close'] < df['close'].shift(1))
        )
        
        # RSI divergence detection (simplified)
        divergence_lookback = self.get_param('divergence_lookback', 10)
        df['price_higher_high'] = df['close'] > df['close'].shift(divergence_lookback)
        df['price_lower_low'] = df['close'] < df['close'].shift(divergence_lookback)
        df['rsi_higher_high'] = df['rsi'] > df['rsi'].shift(divergence_lookback)
        df['rsi_lower_low'] = df['rsi'] < df['rsi'].shift(divergence_lookback)
        
        df['bullish_divergence'] = (
            df['price_lower_low'] & 
            ~df['rsi_lower_low'] & 
            df['rsi_oversold']
        )
        
        df['bearish_divergence'] = (
            df['price_higher_high'] & 
            ~df['rsi_higher_high'] & 
            df['rsi_overbought']
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.2))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "10:00")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "14:30")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # RSI strength (distance from 50)
        df['rsi_strength'] = abs(df['rsi'] - 50) / 50
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate RSI reversion signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Time filter
            if not row['time_filter']:
                continue
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 5000):
                continue
            
            # Bullish RSI reversion signal
            if ((row['rsi_bullish_reversal'] or row['bullish_divergence']) and 
                self.is_market_open(row.name)):
                
                # Higher confidence for extreme levels or divergence
                confidence = 0.8 if (row['rsi_extreme_oversold'] or row['bullish_divergence']) else 0.7
                
                signal = Signal(
                    action="BUY",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"RSI bullish reversion (RSI: {row['rsi']:.1f}, Strength: {row['rsi_strength']:.2f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish RSI reversion signal
            elif ((row['rsi_bearish_reversal'] or row['bearish_divergence']) and 
                  self.is_market_open(row.name)):
                
                # Higher confidence for extreme levels or divergence
                confidence = 0.8 if (row['rsi_extreme_overbought'] or row['bearish_divergence']) else 0.7
                
                signal = Signal(
                    action="SELL",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"RSI bearish reversion (RSI: {row['rsi']:.1f}, Strength: {row['rsi_strength']:.2f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check RSI reversion entry conditions."""
        if current_idx < 20:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Time filter
        if not current['time_filter']:
            return False, "Outside trading hours"
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if current['volume'] < self.get_param('min_volume', 5000):
            return False, "Low volume"
        
        # Bullish reversion
        if current['rsi_bullish_reversal'] or current['bullish_divergence']:
            signal_type = "divergence" if current['bullish_divergence'] else "reversal"
            return True, f"RSI bullish {signal_type} (RSI: {current['rsi']:.1f})"
        
        # Bearish reversion
        if current['rsi_bearish_reversal'] or current['bearish_divergence']:
            signal_type = "divergence" if current['bearish_divergence'] else "reversal"
            return True, f"RSI bearish {signal_type} (RSI: {current['rsi']:.1f})"
        
        return False, "No RSI reversion signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check RSI reversion exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # RSI returned to neutral zone
        rsi_exit_neutral = self.get_param('rsi_exit_neutral', 50)
        
        if (entry_price < current['close'] and current['rsi'] > rsi_exit_neutral):
            return True, f"RSI returned to neutral (RSI: {current['rsi']:.1f})"
        
        if (entry_price > current['close'] and current['rsi'] < rsi_exit_neutral):
            return True, f"RSI returned to neutral (RSI: {current['rsi']:.1f})"
        
        # RSI moved to opposite extreme (failed reversion)
        if (entry_price < current['close'] and current['rsi_overbought']):
            return True, "RSI moved to overbought"
        
        if (entry_price > current['close'] and current['rsi_oversold']):
            return True, "RSI moved to oversold"
        
        # Time-based exit (after 45 minutes)
        time_in_trade = (current.name - entry_time).total_seconds() / 60  # minutes
        if time_in_trade > 45:
            return True, "Time-based exit (45 minutes)"
        
        return False, "Hold RSI reversion position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate stop loss for RSI reversion."""
        current = df.iloc[current_idx]
        
        # Use recent swing low/high
        if current_idx >= 10:
            recent_data = df.iloc[current_idx-10:current_idx]
            
            if current['rsi_oversold']:  # Long position
                return recent_data['low'].min()
            elif current['rsi_overbought']:  # Short position
                return recent_data['high'].max()
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.5) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on RSI strength."""
        current = df.iloc[current_idx]
        
        # Base target
        target_pct = self.get_param('target_pct', 2.0) / 100
        
        # Adjust based on RSI strength
        if 'rsi_strength' in current:
            # Higher RSI strength = higher target potential
            strength_adjustment = current['rsi_strength'] * 0.5  # Max 50% adjustment
            target_pct *= (1 + strength_adjustment)
        
        return entry_price * (1 + target_pct)