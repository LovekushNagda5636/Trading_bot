"""
Stochastic Mean Reversion Strategy
Mean reversion strategy using Stochastic oscillator oversold/overbought levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class StochasticReversionStrategy(BaseStrategy):
    """
    Stochastic Mean Reversion Strategy.
    
    Entry: When Stochastic reaches extreme levels and shows reversal signs
    Exit: When Stochastic returns to neutral zone or continues extreme
    """
    
    @property
    def name(self) -> str:
        return "Stochastic Reversion"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'k_period': 14,
            'd_period': 3,
            'smooth_k': 3,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            'stoch_extreme_oversold': 10,
            'stoch_extreme_overbought': 90,
            'volume_threshold': 1.2,
            'min_volume': 5000,
            'stop_loss_pct': 1.2,
            'target_pct': 2.0,
            'cross_confirmation': True,
            'time_filter_start': "10:00",
            'time_filter_end': "14:30"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic reversion indicators."""
        df = df.copy()
        
        k_period = self.get_param('k_period', 14)
        d_period = self.get_param('d_period', 3)
        smooth_k = self.get_param('smooth_k', 3)
        
        # Calculate Stochastic oscillator
        df['lowest_low'] = df['low'].rolling(window=k_period).min()
        df['highest_high'] = df['high'].rolling(window=k_period).max()
        
        # %K calculation
        df['k_raw'] = ((df['close'] - df['lowest_low']) / 
                       (df['highest_high'] - df['lowest_low'])) * 100
        
        # Smooth %K
        df['k_percent'] = df['k_raw'].rolling(window=smooth_k).mean()
        
        # %D calculation (signal line)
        df['d_percent'] = df['k_percent'].rolling(window=d_period).mean()
        
        # Stochastic levels
        stoch_oversold = self.get_param('stoch_oversold', 20)
        stoch_overbought = self.get_param('stoch_overbought', 80)
        stoch_extreme_oversold = self.get_param('stoch_extreme_oversold', 10)
        stoch_extreme_overbought = self.get_param('stoch_extreme_overbought', 90)
        
        df['stoch_oversold'] = df['k_percent'] < stoch_oversold
        df['stoch_overbought'] = df['k_percent'] > stoch_overbought
        df['stoch_extreme_oversold'] = df['k_percent'] < stoch_extreme_oversold
        df['stoch_extreme_overbought'] = df['k_percent'] > stoch_extreme_overbought
        
        # Stochastic crossovers
        df['stoch_k_above_d'] = df['k_percent'] > df['d_percent']
        df['stoch_k_below_d'] = df['k_percent'] < df['d_percent']
        
        df['stoch_bullish_cross'] = (
            df['stoch_k_above_d'] & 
            df['stoch_k_below_d'].shift(1)
        )
        
        df['stoch_bearish_cross'] = (
            df['stoch_k_below_d'] & 
            df['stoch_k_above_d'].shift(1)
        )
        
        # Reversal signals
        if self.get_param('cross_confirmation', True):
            df['stoch_bullish_reversal'] = (
                df['stoch_oversold'] & 
                df['stoch_bullish_cross'] &
                (df['close'] > df['close'].shift(1))
            )
            
            df['stoch_bearish_reversal'] = (
                df['stoch_overbought'] & 
                df['stoch_bearish_cross'] &
                (df['close'] < df['close'].shift(1))
            )
        else:
            df['stoch_bullish_reversal'] = (
                df['stoch_oversold'] & 
                (df['k_percent'] > df['k_percent'].shift(1)) &
                (df['close'] > df['close'].shift(1))
            )
            
            df['stoch_bearish_reversal'] = (
                df['stoch_overbought'] & 
                (df['k_percent'] < df['k_percent'].shift(1)) &
                (df['close'] < df['close'].shift(1))
            )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.2))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "10:00")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "14:30")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Stochastic momentum
        df['stoch_momentum'] = df['k_percent'] - df['k_percent'].shift(3)
        df['stoch_strength'] = abs(df['k_percent'] - 50) / 50
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate Stochastic reversion signals."""
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
            
            # Bullish Stochastic reversion signal
            if (row['stoch_bullish_reversal'] and self.is_market_open(row.name)):
                
                # Higher confidence for extreme levels
                confidence = 0.8 if row['stoch_extreme_oversold'] else 0.7
                
                signal = Signal(
                    action="BUY",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Stochastic bullish reversion (%K: {row['k_percent']:.1f}, %D: {row['d_percent']:.1f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish Stochastic reversion signal
            elif (row['stoch_bearish_reversal'] and self.is_market_open(row.name)):
                
                # Higher confidence for extreme levels
                confidence = 0.8 if row['stoch_extreme_overbought'] else 0.7
                
                signal = Signal(
                    action="SELL",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Stochastic bearish reversion (%K: {row['k_percent']:.1f}, %D: {row['d_percent']:.1f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check Stochastic reversion entry conditions."""
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
        if current['stoch_bullish_reversal']:
            return True, f"Stochastic bullish reversion (%K: {current['k_percent']:.1f})"
        
        # Bearish reversion
        if current['stoch_bearish_reversal']:
            return True, f"Stochastic bearish reversion (%K: {current['k_percent']:.1f})"
        
        return False, "No Stochastic reversion signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check Stochastic reversion exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Stochastic returned to neutral zone (30-70)
        if (entry_price < current['close'] and current['k_percent'] > 70):
            return True, f"Stochastic reached overbought (%K: {current['k_percent']:.1f})"
        
        if (entry_price > current['close'] and current['k_percent'] < 30):
            return True, f"Stochastic reached oversold (%K: {current['k_percent']:.1f})"
        
        # Opposite crossover
        if (entry_price < current['close'] and current['stoch_bearish_cross']):
            return True, "Stochastic bearish crossover"
        
        if (entry_price > current['close'] and current['stoch_bullish_cross']):
            return True, "Stochastic bullish crossover"
        
        # Time-based exit (after 30 minutes)
        time_in_trade = (current.name - entry_time).total_seconds() / 60  # minutes
        if time_in_trade > 30:
            return True, "Time-based exit (30 minutes)"
        
        return False, "Hold Stochastic reversion position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate stop loss for Stochastic reversion."""
        current = df.iloc[current_idx]
        
        # Use recent swing low/high
        if current_idx >= 8:
            recent_data = df.iloc[current_idx-8:current_idx]
            
            if current['stoch_oversold']:  # Long position
                return recent_data['low'].min()
            elif current['stoch_overbought']:  # Short position
                return recent_data['high'].max()
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.2) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on Stochastic strength."""
        current = df.iloc[current_idx]
        
        # Base target
        target_pct = self.get_param('target_pct', 2.0) / 100
        
        # Adjust based on Stochastic strength
        if 'stoch_strength' in current:
            # Higher Stochastic strength = higher target potential
            strength_adjustment = current['stoch_strength'] * 0.3  # Max 30% adjustment
            target_pct *= (1 + strength_adjustment)
        
        return entry_price * (1 + target_pct)