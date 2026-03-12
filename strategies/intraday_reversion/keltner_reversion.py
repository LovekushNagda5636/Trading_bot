"""
Keltner Channel Mean Reversion Strategy
Mean reversion strategy using Keltner Channels for entry and exit signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class KeltnerReversionStrategy(BaseStrategy):
    """
    Keltner Channel Mean Reversion Strategy.
    
    Entry: When price touches outer Keltner channels and shows reversal signs
    Exit: When price returns to middle line or continues away
    """
    
    @property
    def name(self) -> str:
        return "Keltner Reversion"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'kc_period': 20,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'volume_threshold': 1.2,
            'min_volume': 6000,
            'stop_loss_pct': 1.2,
            'target_middle_return': 0.8,  # Target 80% return to middle
            'rsi_confirmation': True,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'time_filter_start': "10:00",
            'time_filter_end': "14:30"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channel reversion indicators."""
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
        df['kc_middle'] = df['close'].ewm(span=kc_period).mean()
        df['kc_upper'] = df['kc_middle'] + (atr_multiplier * df['atr'])
        df['kc_lower'] = df['kc_middle'] - (atr_multiplier * df['atr'])
        
        # Channel position
        df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
        df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle'] * 100
        
        # Channel touches and penetrations
        df['touching_upper'] = df['close'] >= df['kc_upper'] * 0.999
        df['touching_lower'] = df['close'] <= df['kc_lower'] * 1.001
        df['outside_upper'] = df['close'] > df['kc_upper']
        df['outside_lower'] = df['close'] < df['kc_lower']
        
        # RSI confirmation (optional)
        if self.get_param('rsi_confirmation', True):
            rsi_period = self.get_param('rsi_period', 14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            rsi_oversold = self.get_param('rsi_oversold', 30)
            rsi_overbought = self.get_param('rsi_overbought', 70)
            
            df['rsi_oversold'] = df['rsi'] < rsi_oversold
            df['rsi_overbought'] = df['rsi'] > rsi_overbought
        else:
            df['rsi_oversold'] = True
            df['rsi_overbought'] = True
        
        # Mean reversion signals
        df['kc_bullish_reversion'] = (
            (df['touching_lower'] | df['outside_lower']) &
            df['rsi_oversold'] &
            (df['close'] > df['close'].shift(1))  # Price starting to reverse
        )
        
        df['kc_bearish_reversion'] = (
            (df['touching_upper'] | df['outside_upper']) &
            df['rsi_overbought'] &
            (df['close'] < df['close'].shift(1))  # Price starting to reverse
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.2))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "10:00")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "14:30")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Price momentum for reversal confirmation
        df['price_momentum_3'] = df['close'] - df['close'].shift(3)
        df['momentum_reversing_up'] = (df['price_momentum_3'] > 0) & (df['price_momentum_3'].shift(1) <= 0)
        df['momentum_reversing_down'] = (df['price_momentum_3'] < 0) & (df['price_momentum_3'].shift(1) >= 0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate Keltner Channel reversion signals."""
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
            
            # Bullish Keltner reversion signal
            if (row['kc_bullish_reversion'] and row['momentum_reversing_up'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Keltner bullish reversion (Pos: {row['kc_position']:.2f}, Width: {row['kc_width']:.2f}%)",
                    confidence=0.7,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish Keltner reversion signal
            elif (row['kc_bearish_reversion'] and row['momentum_reversing_down'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Keltner bearish reversion (Pos: {row['kc_position']:.2f}, Width: {row['kc_width']:.2f}%)",
                    confidence=0.7,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check Keltner Channel reversion entry conditions."""
        if current_idx < 25:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Time filter
        if not current['time_filter']:
            return False, "Outside trading hours"
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if current['volume'] < self.get_param('min_volume', 6000):
            return False, "Low volume"
        
        # Bullish reversion
        if current['kc_bullish_reversion'] and current['momentum_reversing_up']:
            return True, f"Keltner bullish reversion (pos: {current['kc_position']:.2f})"
        
        # Bearish reversion
        if current['kc_bearish_reversion'] and current['momentum_reversing_down']:
            return True, f"Keltner bearish reversion (pos: {current['kc_position']:.2f})"
        
        return False, "No Keltner reversion signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check Keltner Channel reversion exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Target reached (partial return to middle)
        target_return = self.get_param('target_middle_return', 0.8)
        
        if not pd.isna(current['kc_middle']):
            middle_distance = abs(entry_price - current['kc_middle'])
            current_distance = abs(current['close'] - current['kc_middle'])
            
            if current_distance <= middle_distance * (1 - target_return):
                return True, f"Target reached ({target_return*100}% return to middle)"
        
        # Price moved further away (failed reversion)
        if (entry_price < current['kc_middle'] and current['close'] < entry_price * 0.995):
            return True, "Price moved further from middle"
        
        if (entry_price > current['kc_middle'] and current['close'] > entry_price * 1.005):
            return True, "Price moved further from middle"
        
        # RSI reversal (if using RSI confirmation)
        if self.get_param('rsi_confirmation', True):
            if (entry_price < current['kc_middle'] and current['rsi_overbought']):
                return True, "RSI became overbought"
            
            if (entry_price > current['kc_middle'] and current['rsi_oversold']):
                return True, "RSI became oversold"
        
        # Time-based exit (after 45 minutes)
        time_in_trade = (current.name - entry_time).total_seconds() / 60  # minutes
        if time_in_trade > 45:
            return True, "Time-based exit (45 minutes)"
        
        return False, "Hold Keltner reversion position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate stop loss for Keltner reversion."""
        current = df.iloc[current_idx]
        
        # Use channel boundary as stop loss
        if not pd.isna(current['kc_upper']) and not pd.isna(current['kc_lower']):
            if entry_price < current['kc_middle']:  # Long position (entered near lower channel)
                # Stop below lower channel
                return current['kc_lower'] * 0.995
            else:  # Short position (entered near upper channel)
                # Stop above upper channel
                return current['kc_upper'] * 1.005
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.2) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target as partial return to middle line."""
        current = df.iloc[current_idx]
        
        # Target is partial return to middle line
        if not pd.isna(current['kc_middle']):
            target_return = self.get_param('target_middle_return', 0.8)
            middle_distance = current['kc_middle'] - entry_price
            return entry_price + (middle_distance * target_return)
        
        # Fallback target
        return entry_price * 1.015