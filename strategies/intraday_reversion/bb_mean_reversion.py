"""
Bollinger Bands Mean Reversion Strategy
Mean reversion strategy using Bollinger Bands for entry and exit signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class BollingerMeanReversionStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy.
    
    Entry: When price touches outer bands and shows reversal signs
    Exit: When price returns to middle band or continues away
    """
    
    @property
    def name(self) -> str:
        return "Bollinger Mean Reversion"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.1,
            'min_volume': 5000,
            'stop_loss_pct': 1.2,
            'target_middle_band': 0.8,  # Target 80% return to middle band
            'squeeze_filter': True,  # Avoid trades during squeeze
            'time_filter_start': "10:00",
            'time_filter_end': "14:30"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands mean reversion indicators."""
        df = df.copy()
        
        bb_period = self.get_param('bb_period', 20)
        bb_std = self.get_param('bb_std', 2.0)
        rsi_period = self.get_param('rsi_period', 14)
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        df['bb_std_dev'] = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * df['bb_std_dev'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * df['bb_std_dev'])
        
        # Band width and position
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # %B indicator
        df['percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Band touches and penetrations
        df['touching_upper'] = df['close'] >= df['bb_upper'] * 0.999  # Within 0.1% of upper band
        df['touching_lower'] = df['close'] <= df['bb_lower'] * 1.001  # Within 0.1% of lower band
        df['outside_upper'] = df['close'] > df['bb_upper']
        df['outside_lower'] = df['close'] < df['bb_lower']
        
        # Squeeze detection (avoid mean reversion during squeeze)
        if self.get_param('squeeze_filter', True):
            df['bb_width_percentile'] = df['bb_width'].rolling(window=50).rank(pct=True)
            df['bb_squeeze'] = df['bb_width_percentile'] < 0.2  # Bottom 20% of width
        else:
            df['bb_squeeze'] = False
        
        # RSI for confirmation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        rsi_oversold = self.get_param('rsi_oversold', 30)
        rsi_overbought = self.get_param('rsi_overbought', 70)
        
        df['rsi_oversold'] = df['rsi'] < rsi_oversold
        df['rsi_overbought'] = df['rsi'] > rsi_overbought
        
        # Mean reversion signals
        df['bullish_reversion'] = (
            (df['touching_lower'] | df['outside_lower']) &
            df['rsi_oversold'] &
            ~df['bb_squeeze'] &
            (df['close'] > df['close'].shift(1))  # Price starting to reverse
        )
        
        df['bearish_reversion'] = (
            (df['touching_upper'] | df['outside_upper']) &
            df['rsi_overbought'] &
            ~df['bb_squeeze'] &
            (df['close'] < df['close'].shift(1))  # Price starting to reverse
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.1))
        
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
        """Generate Bollinger Bands mean reversion signals."""
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
            
            # Bullish mean reversion signal
            if (row['bullish_reversion'] and row['momentum_reversing_up'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"BB bullish reversion (%B: {row['percent_b']:.2f}, RSI: {row['rsi']:.1f})",
                    confidence=0.7,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish mean reversion signal
            elif (row['bearish_reversion'] and row['momentum_reversing_down'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"BB bearish reversion (%B: {row['percent_b']:.2f}, RSI: {row['rsi']:.1f})",
                    confidence=0.7,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check Bollinger Bands mean reversion entry conditions."""
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
        if current['volume'] < self.get_param('min_volume', 5000):
            return False, "Low volume"
        
        # Squeeze filter
        if current['bb_squeeze']:
            return False, "Bollinger Bands in squeeze"
        
        # Bullish reversion
        if current['bullish_reversion'] and current['momentum_reversing_up']:
            return True, f"BB bullish reversion (%B: {current['percent_b']:.2f})"
        
        # Bearish reversion
        if current['bearish_reversion'] and current['momentum_reversing_down']:
            return True, f"BB bearish reversion (%B: {current['percent_b']:.2f})"
        
        return False, "No BB mean reversion signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check Bollinger Bands mean reversion exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Target reached (partial return to middle band)
        target_return = self.get_param('target_middle_band', 0.8)
        
        if not pd.isna(current['bb_middle']):
            middle_distance = abs(entry_price - current['bb_middle'])
            current_distance = abs(current['close'] - current['bb_middle'])
            
            if current_distance <= middle_distance * (1 - target_return):
                return True, f"Target reached ({target_return*100}% return to middle band)"
        
        # Price moved further away (failed reversion)
        if (entry_price < current['bb_middle'] and current['close'] < entry_price * 0.995):
            return True, "Price moved further from middle band"
        
        if (entry_price > current['bb_middle'] and current['close'] > entry_price * 1.005):
            return True, "Price moved further from middle band"
        
        # RSI reversal (momentum changed)
        if (entry_price < current['bb_middle'] and current['rsi_overbought']):
            return True, "RSI became overbought"
        
        if (entry_price > current['bb_middle'] and current['rsi_oversold']):
            return True, "RSI became oversold"
        
        # Time-based exit (after 45 minutes)
        time_in_trade = (current.name - entry_time).total_seconds() / 60  # minutes
        if time_in_trade > 45:
            return True, "Time-based exit (45 minutes)"
        
        return False, "Hold BB mean reversion position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate stop loss for mean reversion."""
        current = df.iloc[current_idx]
        
        # Use band as stop loss
        if not pd.isna(current['bb_upper']) and not pd.isna(current['bb_lower']):
            if entry_price < current['bb_middle']:  # Long position (entered near lower band)
                # Stop below lower band
                return current['bb_lower'] * 0.995
            else:  # Short position (entered near upper band)
                # Stop above upper band
                return current['bb_upper'] * 1.005
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.2) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target as partial return to middle band."""
        current = df.iloc[current_idx]
        
        # Target is partial return to middle band
        if not pd.isna(current['bb_middle']):
            target_return = self.get_param('target_middle_band', 0.8)
            middle_distance = current['bb_middle'] - entry_price
            return entry_price + (middle_distance * target_return)
        
        # Fallback target
        return entry_price * 1.012