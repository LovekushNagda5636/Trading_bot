"""
DMI (Directional Movement Index) Strategy
Trend following strategy using Directional Movement Index and ADX.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class DMIStrategy(BaseStrategy):
    """
    DMI Strategy using Directional Movement Index.
    
    Entry: When +DI crosses above -DI with strong ADX
    Exit: When +DI crosses below -DI or ADX weakens
    """
    
    @property
    def name(self) -> str:
        return "DMI"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'dmi_period': 14,
            'adx_threshold': 25,
            'min_volume': 7000,
            'stop_loss_pct': 1.0,
            'target_pct': 2.5,
            'di_separation_min': 5  # Minimum separation between +DI and -DI
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate DMI indicators."""
        df = df.copy()
        
        period = self.get_param('dmi_period', 14)
        
        # Calculate True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Calculate Directional Movement
        df['dm_plus'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0),
            0
        )
        
        df['dm_minus'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0),
            0
        )
        
        # Smooth the values using Wilder's smoothing
        df['tr_smooth'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
        df['dm_plus_smooth'] = df['dm_plus'].ewm(alpha=1/period, adjust=False).mean()
        df['dm_minus_smooth'] = df['dm_minus'].ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate Directional Indicators
        df['di_plus'] = (df['dm_plus_smooth'] / df['tr_smooth']) * 100
        df['di_minus'] = (df['dm_minus_smooth'] / df['tr_smooth']) * 100
        
        # Calculate DX and ADX
        df['dx'] = abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus']) * 100
        df['adx'] = df['dx'].ewm(alpha=1/period, adjust=False).mean()
        
        # Generate signals
        df['di_bullish_cross'] = (df['di_plus'] > df['di_minus']) & (df['di_plus'].shift(1) <= df['di_minus'].shift(1))
        df['di_bearish_cross'] = (df['di_plus'] < df['di_minus']) & (df['di_plus'].shift(1) >= df['di_minus'].shift(1))
        
        # ADX strength filter
        adx_threshold = self.get_param('adx_threshold', 25)
        df['adx_strong'] = df['adx'] > adx_threshold
        df['adx_rising'] = df['adx'] > df['adx'].shift(1)
        
        # DI separation filter
        di_separation_min = self.get_param('di_separation_min', 5)
        df['di_separation'] = abs(df['di_plus'] - df['di_minus'])
        df['di_separated'] = df['di_separation'] > di_separation_min
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate DMI signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            if row['volume'] < self.get_param('min_volume', 7000):
                continue
            
            # Bullish DMI signal
            if (row['di_bullish_cross'] and row['adx_strong'] and 
                row['di_separated'] and self.is_market_open(row.name)):
                
                confidence = 0.8 if row['adx_rising'] else 0.7
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"DMI bullish cross (+DI: {row['di_plus']:.1f}, -DI: {row['di_minus']:.1f}, ADX: {row['adx']:.1f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish DMI signal
            elif (row['di_bearish_cross'] and row['adx_strong'] and 
                  row['di_separated'] and self.is_market_open(row.name)):
                
                confidence = 0.8 if row['adx_rising'] else 0.7
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"DMI bearish cross (+DI: {row['di_plus']:.1f}, -DI: {row['di_minus']:.1f}, ADX: {row['adx']:.1f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check DMI entry conditions."""
        if current_idx < 30:  # Need enough data for DMI
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        if current['volume'] < self.get_param('min_volume', 7000):
            return False, "Low volume"
        
        if not current['adx_strong']:
            return False, "ADX too weak"
        
        if not current['di_separated']:
            return False, "DI lines too close"
        
        # Bullish DMI entry
        if current['di_bullish_cross']:
            return True, f"DMI bullish cross (ADX: {current['adx']:.1f})"
        
        # Bearish DMI entry
        if current['di_bearish_cross']:
            return True, f"DMI bearish cross (ADX: {current['adx']:.1f})"
        
        return False, "No DMI signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check DMI exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # Opposite DI crossover
        if current['di_bullish_cross'] or current['di_bearish_cross']:
            return True, "DMI crossover reversal"
        
        # ADX weakening significantly
        if not current['adx_strong']:
            return True, "ADX weakened below threshold"
        
        # DI lines converging (trend weakening)
        if not current['di_separated']:
            return True, "DI lines converging"
        
        return False, "Hold DMI position"
    
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
        """Calculate target based on ADX strength."""
        current = df.iloc[current_idx]
        
        # Base target
        target_pct = self.get_param('target_pct', 2.5) / 100
        
        # Adjust based on ADX strength
        if 'adx' in current and current['adx'] > 25:
            # Stronger ADX = higher target
            adx_adjustment = min((current['adx'] - 25) * 0.01, 0.5)  # Max 50% adjustment
            target_pct *= (1 + adx_adjustment)
        
        return entry_price * (1 + target_pct)