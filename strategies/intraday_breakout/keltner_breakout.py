"""
Keltner Channel Breakout Strategy
Breakout strategy using Keltner Channels based on ATR.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class KeltnerBreakoutStrategy(BaseStrategy):
    """
    Keltner Channel Breakout Strategy.
    
    Entry: When price breaks above/below Keltner channels
    Exit: When price returns to middle line or opposite breakout
    """
    
    @property
    def name(self) -> str:
        return "Keltner Breakout"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'kc_period': 20,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'volume_threshold': 1.6,
            'min_volume': 7000,
            'breakout_confirmation': 0.1,
            'stop_loss_atr': 1.5,
            'target_atr': 3.0
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channel indicators."""
        df = df.copy()
        
        kc_period = self.get_param('kc_period', 20)
        atr_period = self.get_param('atr_period', 14)
        atr_multiplier = self.get_param('atr_multiplier', 2.0)
        
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
        df['kc_middle'] = df['close'].ewm(span=kc_period).mean()  # EMA as middle line
        df['kc_upper'] = df['kc_middle'] + (atr_multiplier * df['atr'])
        df['kc_lower'] = df['kc_middle'] - (atr_multiplier * df['atr'])
        
        # Channel width and position
        df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle'] * 100
        df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
        
        # Breakout detection
        breakout_conf = self.get_param('breakout_confirmation', 0.1) / 100
        df['kc_upper_breakout_level'] = df['kc_upper'] * (1 + breakout_conf)
        df['kc_lower_breakout_level'] = df['kc_lower'] * (1 - breakout_conf)
        
        df['kc_upper_breakout'] = df['close'] > df['kc_upper_breakout_level']
        df['kc_lower_breakout'] = df['close'] < df['kc_lower_breakout_level']
        
        # Channel squeeze (when Bollinger Bands are inside Keltner Channels)
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        df['bb_std_dev'] = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * df['bb_std_dev'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * df['bb_std_dev'])
        
        df['squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
        
        # Squeeze breakout (high probability setup)
        df['squeeze_breakout_up'] = (
            df['kc_upper_breakout'] & 
            df['squeeze'].shift(1)  # Previous bar was in squeeze
        )
        
        df['squeeze_breakout_down'] = (
            df['kc_lower_breakout'] & 
            df['squeeze'].shift(1)  # Previous bar was in squeeze
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.6))
        
        # Momentum confirmation (price direction)
        df['momentum'] = df['close'] - df['close'].shift(5)  # 5-bar momentum
        df['bullish_momentum'] = df['momentum'] > 0
        df['bearish_momentum'] = df['momentum'] < 0
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate Keltner Channel breakout signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 7000):
                continue
            
            # Upper channel breakout
            if ((row['kc_upper_breakout'] or row['squeeze_breakout_up']) and 
                row['volume_spike'] and row['bullish_momentum'] and 
                self.is_market_open(row.name)):
                
                reason = "KC squeeze breakout up" if row['squeeze_breakout_up'] else "KC upper breakout"
                confidence = 0.85 if row['squeeze_breakout_up'] else 0.75
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"{reason} (Width: {row['kc_width']:.2f}%, Pos: {row['kc_position']:.2f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Lower channel breakout
            elif ((row['kc_lower_breakout'] or row['squeeze_breakout_down']) and 
                  row['volume_spike'] and row['bearish_momentum'] and 
                  self.is_market_open(row.name)):
                
                reason = "KC squeeze breakout down" if row['squeeze_breakout_down'] else "KC lower breakout"
                confidence = 0.85 if row['squeeze_breakout_down'] else 0.75
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"{reason} (Width: {row['kc_width']:.2f}%, Pos: {row['kc_position']:.2f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check Keltner Channel entry conditions."""
        if current_idx < 25:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if not current['volume_spike']:
            return False, "No volume confirmation"
        
        # Upper breakout with momentum
        if ((current['kc_upper_breakout'] or current['squeeze_breakout_up']) and 
            current['bullish_momentum']):
            setup_type = "squeeze" if current['squeeze_breakout_up'] else "standard"
            return True, f"KC upper breakout ({setup_type}, pos: {current['kc_position']:.2f})"
        
        # Lower breakout with momentum
        if ((current['kc_lower_breakout'] or current['squeeze_breakout_down']) and 
            current['bearish_momentum']):
            setup_type = "squeeze" if current['squeeze_breakout_down'] else "standard"
            return True, f"KC lower breakout ({setup_type}, pos: {current['kc_position']:.2f})"
        
        return False, "No KC breakout"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check Keltner Channel exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # Return to middle line
        if (entry_price > current['kc_upper'] and 
            current['close'] < current['kc_middle']):
            return True, "Price returned to KC middle"
        
        if (entry_price < current['kc_lower'] and 
            current['close'] > current['kc_middle']):
            return True, "Price returned to KC middle"
        
        # Opposite channel touch
        if (entry_price > current['kc_upper'] and 
            current['close'] < current['kc_lower']):
            return True, "Price touched opposite KC"
        
        if (entry_price < current['kc_lower'] and 
            current['close'] > current['kc_upper']):
            return True, "Price touched opposite KC"
        
        # Momentum reversal
        if (entry_price > current['kc_upper'] and 
            current['bearish_momentum']):
            return True, "Momentum turned bearish"
        
        if (entry_price < current['kc_lower'] and 
            current['bullish_momentum']):
            return True, "Momentum turned bullish"
        
        return False, "Hold KC breakout position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use ATR-based stop loss."""
        current = df.iloc[current_idx]
        
        if 'atr' in current and not pd.isna(current['atr']):
            atr_multiplier = self.get_param('stop_loss_atr', 1.5)
            return entry_price - (current['atr'] * atr_multiplier)
        
        # Fallback to middle line
        if 'kc_middle' in current:
            return current['kc_middle']
        
        # Final fallback
        return entry_price * 0.985
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use ATR-based target."""
        current = df.iloc[current_idx]
        
        if 'atr' in current and not pd.isna(current['atr']):
            atr_multiplier = self.get_param('target_atr', 3.0)
            return entry_price + (current['atr'] * atr_multiplier)
        
        # Fallback target
        return entry_price * 1.03