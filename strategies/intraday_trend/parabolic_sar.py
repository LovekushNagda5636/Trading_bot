"""
Parabolic SAR Strategy
Trend following strategy using Parabolic Stop and Reverse indicator.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class ParabolicSARStrategy(BaseStrategy):
    """
    Parabolic SAR Strategy for trend following.
    
    Entry: When price crosses above/below SAR
    Exit: When SAR flips direction or market close
    """
    
    @property
    def name(self) -> str:
        return "Parabolic SAR"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'af_start': 0.02,
            'af_increment': 0.02,
            'af_max': 0.2,
            'min_volume': 6000,
            'trend_filter': True,
            'trend_ema': 50,
            'stop_loss_pct': 0.5,
            'target_rr': 2.5
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Parabolic SAR indicator."""
        df = df.copy()
        
        af_start = self.get_param('af_start', 0.02)
        af_increment = self.get_param('af_increment', 0.02)
        af_max = self.get_param('af_max', 0.2)
        
        # Initialize SAR calculation
        df['sar'] = np.nan
        df['af'] = af_start
        df['ep'] = np.nan  # Extreme Point
        df['trend'] = np.nan  # 1 for uptrend, -1 for downtrend
        
        # Calculate Parabolic SAR
        for i in range(1, len(df)):
            if i == 1:
                # Initialize first values
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    df.loc[df.index[i], 'trend'] = 1
                    df.loc[df.index[i], 'sar'] = df['low'].iloc[i-1]
                    df.loc[df.index[i], 'ep'] = df['high'].iloc[i]
                else:
                    df.loc[df.index[i], 'trend'] = -1
                    df.loc[df.index[i], 'sar'] = df['high'].iloc[i-1]
                    df.loc[df.index[i], 'ep'] = df['low'].iloc[i]
                continue
            
            prev_sar = df['sar'].iloc[i-1]
            prev_af = df['af'].iloc[i-1]
            prev_ep = df['ep'].iloc[i-1]
            prev_trend = df['trend'].iloc[i-1]
            
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            if prev_trend == 1:  # Uptrend
                # Calculate new SAR
                new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # Check for trend reversal
                if current_low <= new_sar:
                    # Trend reversal to downtrend
                    df.loc[df.index[i], 'trend'] = -1
                    df.loc[df.index[i], 'sar'] = prev_ep
                    df.loc[df.index[i], 'ep'] = current_low
                    df.loc[df.index[i], 'af'] = af_start
                else:
                    # Continue uptrend
                    df.loc[df.index[i], 'trend'] = 1
                    df.loc[df.index[i], 'sar'] = min(new_sar, df['low'].iloc[i-1], df['low'].iloc[i-2] if i > 1 else df['low'].iloc[i-1])
                    
                    # Update EP and AF
                    if current_high > prev_ep:
                        df.loc[df.index[i], 'ep'] = current_high
                        df.loc[df.index[i], 'af'] = min(prev_af + af_increment, af_max)
                    else:
                        df.loc[df.index[i], 'ep'] = prev_ep
                        df.loc[df.index[i], 'af'] = prev_af
            
            else:  # Downtrend
                # Calculate new SAR
                new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # Check for trend reversal
                if current_high >= new_sar:
                    # Trend reversal to uptrend
                    df.loc[df.index[i], 'trend'] = 1
                    df.loc[df.index[i], 'sar'] = prev_ep
                    df.loc[df.index[i], 'ep'] = current_high
                    df.loc[df.index[i], 'af'] = af_start
                else:
                    # Continue downtrend
                    df.loc[df.index[i], 'trend'] = -1
                    df.loc[df.index[i], 'sar'] = max(new_sar, df['high'].iloc[i-1], df['high'].iloc[i-2] if i > 1 else df['high'].iloc[i-1])
                    
                    # Update EP and AF
                    if current_low < prev_ep:
                        df.loc[df.index[i], 'ep'] = current_low
                        df.loc[df.index[i], 'af'] = min(prev_af + af_increment, af_max)
                    else:
                        df.loc[df.index[i], 'ep'] = prev_ep
                        df.loc[df.index[i], 'af'] = prev_af
        
        # Generate signals
        df['sar_bullish'] = (df['close'] > df['sar']) & (df['close'].shift(1) <= df['sar'].shift(1))
        df['sar_bearish'] = (df['close'] < df['sar']) & (df['close'].shift(1) >= df['sar'].shift(1))
        
        # Trend filter
        if self.get_param('trend_filter', True):
            trend_period = self.get_param('trend_ema', 50)
            df['trend_ema'] = df['close'].ewm(span=trend_period).mean()
            df['uptrend'] = df['close'] > df['trend_ema']
            df['downtrend'] = df['close'] < df['trend_ema']
        else:
            df['uptrend'] = True
            df['downtrend'] = True
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate Parabolic SAR signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 6000):
                continue
            
            # Bullish SAR signal
            if (row['sar_bullish'] and row['uptrend'] and 
                not pd.isna(row['sar'])):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Parabolic SAR bullish crossover",
                    confidence=0.75,
                    stop_loss=row['sar'],  # Use SAR as stop
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish SAR signal
            elif (row['sar_bearish'] and row['downtrend'] and 
                  not pd.isna(row['sar'])):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Parabolic SAR bearish crossover",
                    confidence=0.75,
                    stop_loss=row['sar'],  # Use SAR as stop
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check Parabolic SAR entry conditions."""
        if current_idx < 10:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if current['volume'] < self.get_param('min_volume', 6000):
            return False, "Low volume"
        
        # Bullish SAR entry
        if current['sar_bullish'] and current['uptrend']:
            return True, "Parabolic SAR bullish signal"
        
        # Bearish SAR entry
        if current['sar_bearish'] and current['downtrend']:
            return True, "Parabolic SAR bearish signal"
        
        return False, "No SAR signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check Parabolic SAR exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # SAR direction change
        if current['sar_bullish'] or current['sar_bearish']:
            return True, "SAR direction change"
        
        return False, "Hold with SAR"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use Parabolic SAR as dynamic stop loss."""
        current = df.iloc[current_idx]
        
        # SAR acts as trailing stop
        if 'sar' in current and not pd.isna(current['sar']):
            return current['sar']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 0.5) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on risk-reward ratio."""
        current = df.iloc[current_idx]
        
        if 'sar' in current and not pd.isna(current['sar']):
            risk = abs(entry_price - current['sar'])
            reward = risk * self.get_param('target_rr', 2.5)
            
            if entry_price > current['sar']:  # Long
                return entry_price + reward
            else:  # Short
                return entry_price - reward
        
        # Fallback target
        return entry_price * 1.025