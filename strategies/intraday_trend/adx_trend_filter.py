"""
ADX Trend Filter Strategy
Uses ADX to identify strong trends and trades in the direction of the trend.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class ADXTrendFilterStrategy(BaseStrategy):
    """
    ADX Trend Filter Strategy for strong trend identification.
    
    Entry: When ADX > threshold and price follows trend direction
    Exit: When ADX weakens or trend reverses
    """
    
    @property
    def name(self) -> str:
        return "ADX Trend Filter"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_15
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 14,
            'adx_threshold': 25,
            'strong_trend': 40,
            'di_period': 14,
            'ema_period': 20,
            'stop_loss_pct': 1.5,
            'target_pct': 3.0,
            'min_volume': 8000
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX and directional indicators."""
        df = df.copy()
        
        period = self.get_param('adx_period', 14)
        
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
        
        # Smooth the values
        df['tr_smooth'] = df['tr'].rolling(window=period).mean()
        df['dm_plus_smooth'] = df['dm_plus'].rolling(window=period).mean()
        df['dm_minus_smooth'] = df['dm_minus'].rolling(window=period).mean()
        
        # Calculate DI+ and DI-
        df['di_plus'] = 100 * (df['dm_plus_smooth'] / df['tr_smooth'])
        df['di_minus'] = 100 * (df['dm_minus_smooth'] / df['tr_smooth'])
        
        # Calculate DX and ADX
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        # Trend direction
        df['bullish_trend'] = df['di_plus'] > df['di_minus']
        df['bearish_trend'] = df['di_minus'] > df['di_plus']
        
        # Strong trend identification
        adx_threshold = self.get_param('adx_threshold', 25)
        strong_threshold = self.get_param('strong_trend', 40)
        
        df['trending'] = df['adx'] > adx_threshold
        df['strong_trend'] = df['adx'] > strong_threshold
        
        # EMA for additional confirmation
        ema_period = self.get_param('ema_period', 20)
        df['ema'] = df['close'].ewm(span=ema_period).mean()
        df['above_ema'] = df['close'] > df['ema']
        df['below_ema'] = df['close'] < df['ema']
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate ADX trend signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # TODO: Add market session validation
            if row['volume'] < self.get_param('min_volume', 8000):
                continue
            
            if not row['trending']:
                continue
            
            # Bullish trend signal
            if (row['bullish_trend'] and row['above_ema'] and 
                row['trending'] and i > 0):
                
                strength = 0.7 if row['strong_trend'] else 0.6
                
                signal = Signal(
                    action="BUY",
                    strength=strength,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Strong bullish trend (ADX: {row['adx']:.1f})",
                    confidence=0.75,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish trend signal
            elif (row['bearish_trend'] and row['below_ema'] and 
                  row['trending'] and i > 0):
                
                strength = 0.7 if row['strong_trend'] else 0.6
                
                signal = Signal(
                    action="SELL",
                    strength=strength,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Strong bearish trend (ADX: {row['adx']:.1f})",
                    confidence=0.75,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check ADX trend entry conditions."""
        if current_idx < 30:
            return False, "Insufficient data for ADX calculation"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        if not current['trending']:
            return False, "No trending market (ADX too low)"
        
        # Volume filter
        if current['volume'] < self.get_param('min_volume', 8000):
            return False, "Volume too low"
        
        # Bullish trend entry
        if current['bullish_trend'] and current['above_ema']:
            return True, f"Bullish ADX trend (ADX: {current['adx']:.1f})"
        
        # Bearish trend entry
        if current['bearish_trend'] and current['below_ema']:
            return True, f"Bearish ADX trend (ADX: {current['adx']:.1f})"
        
        return False, "No clear trend direction"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check ADX trend exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Trend weakening
        if not current['trending']:
            return True, f"Trend weakening (ADX: {current['adx']:.1f})"
        
        # Directional change
        prev = df_with_indicators.iloc[current_idx - 1] if current_idx > 0 else current
        if (prev['bullish_trend'] and current['bearish_trend']) or \
           (prev['bearish_trend'] and current['bullish_trend']):
            return True, "Trend direction change"
        
        # TODO: Add trailing stop based on ADX strength
        
        return False, "Trend continues"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate stop loss based on EMA and volatility."""
        current = df.iloc[current_idx]
        stop_pct = self.get_param('stop_loss_pct', 1.5) / 100
        
        # TODO: Use ATR-based stop loss for better risk management
        # For now, use EMA as support/resistance
        if 'ema' in current:
            ema_stop = current['ema']
            pct_stop = entry_price * (1 - stop_pct)
            
            # Use the closer stop loss
            if entry_price > ema_stop:  # Long position
                return max(ema_stop, pct_stop)
            else:  # Short position
                return min(ema_stop, entry_price * (1 + stop_pct))
        
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on trend strength."""
        current = df.iloc[current_idx]
        base_target = self.get_param('target_pct', 3.0) / 100
        
        # Adjust target based on ADX strength
        if 'adx' in current:
            adx_multiplier = min(current['adx'] / 25, 2.0)  # Scale with ADX
            adjusted_target = base_target * adx_multiplier
        else:
            adjusted_target = base_target
        
        return entry_price * (1 + adjusted_target)