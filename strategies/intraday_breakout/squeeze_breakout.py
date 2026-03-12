"""
Squeeze Breakout Strategy
Combined Bollinger Bands and Keltner Channel squeeze breakout strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class SqueezeBreakoutStrategy(BaseStrategy):
    """
    Squeeze Breakout Strategy combining BB and KC.
    
    Entry: When price breaks out after BB/KC squeeze with momentum
    Exit: When squeeze forms again or momentum reverses
    """
    
    @property
    def name(self) -> str:
        return "Squeeze Breakout"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'bb_period': 20,
            'bb_std': 2.0,
            'kc_period': 20,
            'atr_period': 14,
            'atr_multiplier': 1.5,
            'momentum_period': 12,
            'volume_threshold': 1.8,
            'min_volume': 8000,
            'min_squeeze_bars': 6,
            'stop_loss_atr': 2.0,
            'target_atr': 4.0
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate squeeze breakout indicators."""
        df = df.copy()
        
        bb_period = self.get_param('bb_period', 20)
        bb_std = self.get_param('bb_std', 2.0)
        kc_period = self.get_param('kc_period', 20)
        atr_period = self.get_param('atr_period', 14)
        atr_multiplier = self.get_param('atr_multiplier', 1.5)
        momentum_period = self.get_param('momentum_period', 12)
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        df['bb_std_dev'] = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * df['bb_std_dev'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * df['bb_std_dev'])
        
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
        
        # Calculate Keltner Channels
        df['kc_middle'] = df['close'].ewm(span=kc_period).mean()
        df['kc_upper'] = df['kc_middle'] + (atr_multiplier * df['atr'])
        df['kc_lower'] = df['kc_middle'] - (atr_multiplier * df['atr'])
        
        # Squeeze detection (BB inside KC)
        df['squeeze_on'] = (df['bb_upper'] <= df['kc_upper']) & (df['bb_lower'] >= df['kc_lower'])
        df['squeeze_off'] = ~df['squeeze_on']
        
        # Count consecutive squeeze bars
        df['squeeze_count'] = 0
        min_squeeze_bars = self.get_param('min_squeeze_bars', 6)
        
        for i in range(min_squeeze_bars, len(df)):
            if df['squeeze_on'].iloc[i]:
                count = 1
                for j in range(i-1, max(i-min_squeeze_bars*2, 0), -1):
                    if df['squeeze_on'].iloc[j]:
                        count += 1
                    else:
                        break
                df.loc[df.index[i], 'squeeze_count'] = count
        
        # Squeeze release detection
        df['squeeze_release'] = (
            df['squeeze_off'] & 
            df['squeeze_on'].shift(1) & 
            (df['squeeze_count'].shift(1) >= min_squeeze_bars)
        )
        
        # Momentum oscillator (linear regression of close prices)
        df['momentum'] = 0.0
        for i in range(momentum_period, len(df)):
            y = df['close'].iloc[i-momentum_period:i].values
            x = np.arange(len(y))
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                df.loc[df.index[i], 'momentum'] = slope
        
        # Momentum direction
        df['momentum_up'] = df['momentum'] > 0
        df['momentum_down'] = df['momentum'] < 0
        df['momentum_strength'] = abs(df['momentum'])
        
        # Breakout signals
        df['squeeze_breakout_up'] = (
            df['squeeze_release'] & 
            df['momentum_up'] & 
            (df['close'] > df['bb_middle'])
        )
        
        df['squeeze_breakout_down'] = (
            df['squeeze_release'] & 
            df['momentum_down'] & 
            (df['close'] < df['bb_middle'])
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.8))
        
        # Volatility expansion
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['volatility_expanding'] = df['bb_width'] > df['bb_width'].shift(1)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate squeeze breakout signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 8000):
                continue
            
            # Bullish squeeze breakout
            if (row['squeeze_breakout_up'] and row['volume_spike'] and 
                row['volatility_expanding'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.9,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bullish squeeze breakout (Squeeze: {int(row['squeeze_count'])} bars, Momentum: {row['momentum']:.4f})",
                    confidence=0.85,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish squeeze breakout
            elif (row['squeeze_breakout_down'] and row['volume_spike'] and 
                  row['volatility_expanding'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.9,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bearish squeeze breakout (Squeeze: {int(row['squeeze_count'])} bars, Momentum: {row['momentum']:.4f})",
                    confidence=0.85,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check squeeze breakout entry conditions."""
        if current_idx < 30:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if not current['volume_spike']:
            return False, "No volume confirmation"
        
        # Volatility expansion check
        if not current['volatility_expanding']:
            return False, "No volatility expansion"
        
        # Bullish squeeze breakout
        if current['squeeze_breakout_up']:
            return True, f"Bullish squeeze breakout (momentum: {current['momentum']:.4f})"
        
        # Bearish squeeze breakout
        if current['squeeze_breakout_down']:
            return True, f"Bearish squeeze breakout (momentum: {current['momentum']:.4f})"
        
        return False, "No squeeze breakout"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check squeeze breakout exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # New squeeze forming
        if current['squeeze_on']:
            return True, "New squeeze forming"
        
        # Momentum reversal
        if current_idx > 0:
            previous = df_with_indicators.iloc[current_idx - 1]
            
            # Exit long on momentum turning negative
            if (entry_price > previous['close'] and 
                current['momentum_down'] and previous['momentum_up']):
                return True, "Momentum turned bearish"
            
            # Exit short on momentum turning positive
            if (entry_price < previous['close'] and 
                current['momentum_up'] and previous['momentum_down']):
                return True, "Momentum turned bullish"
        
        # Volatility contraction
        if not current['volatility_expanding']:
            time_in_trade = (current.name - entry_time).total_seconds() / 60
            if time_in_trade > 30:  # After 30 minutes
                return True, "Volatility contracting after 30 min"
        
        return False, "Hold squeeze breakout position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use ATR-based stop loss."""
        current = df.iloc[current_idx]
        
        if 'atr' in current and not pd.isna(current['atr']):
            atr_multiplier = self.get_param('stop_loss_atr', 2.0)
            return entry_price - (current['atr'] * atr_multiplier)
        
        # Fallback to middle band
        if 'bb_middle' in current:
            return current['bb_middle']
        
        return entry_price * 0.98
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use ATR-based target with momentum consideration."""
        current = df.iloc[current_idx]
        
        if 'atr' in current and not pd.isna(current['atr']):
            base_multiplier = self.get_param('target_atr', 4.0)
            
            # Adjust target based on momentum strength
            momentum_adjustment = min(current['momentum_strength'] * 2, 1.0)
            final_multiplier = base_multiplier * (1 + momentum_adjustment)
            
            return entry_price + (current['atr'] * final_multiplier)
        
        return entry_price * 1.04