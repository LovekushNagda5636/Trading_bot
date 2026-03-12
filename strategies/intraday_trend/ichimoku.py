"""
Ichimoku Cloud Strategy
Comprehensive trend following strategy using Ichimoku Kinko Hyo system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class IchimokuStrategy(BaseStrategy):
    """
    Ichimoku Cloud Strategy for trend analysis.
    
    Entry: When price breaks above/below cloud with Tenkan-Kijun cross
    Exit: When cloud changes color or price returns to cloud
    """
    
    @property
    def name(self) -> str:
        return "Ichimoku Cloud"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_15
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_b_period': 52,
            'displacement': 26,
            'min_volume': 10000,
            'cloud_thickness_min': 0.2,  # Minimum cloud thickness %
            'stop_loss_pct': 1.0,
            'target_rr': 2.0
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud indicators."""
        df = df.copy()
        
        tenkan_period = self.get_param('tenkan_period', 9)
        kijun_period = self.get_param('kijun_period', 26)
        senkou_b_period = self.get_param('senkou_b_period', 52)
        displacement = self.get_param('displacement', 26)
        
        # Tenkan-sen (Conversion Line)
        df['tenkan_high'] = df['high'].rolling(window=tenkan_period).max()
        df['tenkan_low'] = df['low'].rolling(window=tenkan_period).min()
        df['tenkan_sen'] = (df['tenkan_high'] + df['tenkan_low']) / 2
        
        # Kijun-sen (Base Line)
        df['kijun_high'] = df['high'].rolling(window=kijun_period).max()
        df['kijun_low'] = df['low'].rolling(window=kijun_period).min()
        df['kijun_sen'] = (df['kijun_high'] + df['kijun_low']) / 2
        
        # Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        df['senkou_b_high'] = df['high'].rolling(window=senkou_b_period).max()
        df['senkou_b_low'] = df['low'].rolling(window=senkou_b_period).min()
        df['senkou_span_b'] = ((df['senkou_b_high'] + df['senkou_b_low']) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        df['chikou_span'] = df['close'].shift(-displacement)
        
        # Cloud analysis
        df['cloud_top'] = np.maximum(df['senkou_span_a'], df['senkou_span_b'])
        df['cloud_bottom'] = np.minimum(df['senkou_span_a'], df['senkou_span_b'])
        df['cloud_thickness'] = (df['cloud_top'] - df['cloud_bottom']) / df['close'] * 100
        
        # Cloud color (green when Senkou A > Senkou B)
        df['cloud_green'] = df['senkou_span_a'] > df['senkou_span_b']
        df['cloud_red'] = df['senkou_span_a'] < df['senkou_span_b']
        
        # Price position relative to cloud
        df['above_cloud'] = df['close'] > df['cloud_top']
        df['below_cloud'] = df['close'] < df['cloud_bottom']
        df['in_cloud'] = ~(df['above_cloud'] | df['below_cloud'])
        
        # Tenkan-Kijun cross
        df['tk_bullish_cross'] = (df['tenkan_sen'] > df['kijun_sen']) & (df['tenkan_sen'].shift(1) <= df['kijun_sen'].shift(1))
        df['tk_bearish_cross'] = (df['tenkan_sen'] < df['kijun_sen']) & (df['tenkan_sen'].shift(1) >= df['kijun_sen'].shift(1))
        
        # Chikou span confirmation
        df['chikou_above_price'] = df['chikou_span'] > df['close'].shift(displacement)
        df['chikou_below_price'] = df['chikou_span'] < df['close'].shift(displacement)
        
        # Signal generation
        df['ichimoku_bullish'] = (
            df['above_cloud'] & 
            df['tk_bullish_cross'] & 
            df['cloud_green'] &
            df['chikou_above_price'] &
            (df['cloud_thickness'] >= self.get_param('cloud_thickness_min', 0.2))
        )
        
        df['ichimoku_bearish'] = (
            df['below_cloud'] & 
            df['tk_bearish_cross'] & 
            df['cloud_red'] &
            df['chikou_below_price'] &
            (df['cloud_thickness'] >= self.get_param('cloud_thickness_min', 0.2))
        )
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate Ichimoku signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 10000):
                continue
            
            # Skip if in cloud (uncertain area)
            if row['in_cloud']:
                continue
            
            # Bullish Ichimoku signal
            if row['ichimoku_bullish']:
                signal = Signal(
                    action="BUY",
                    strength=0.9,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Ichimoku bullish: Above green cloud with TK cross",
                    confidence=0.85,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish Ichimoku signal
            elif row['ichimoku_bearish']:
                signal = Signal(
                    action="SELL",
                    strength=0.9,
                    price=row['close'],
                    timestamp=row.name,
                    reason="Ichimoku bearish: Below red cloud with TK cross",
                    confidence=0.85,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check Ichimoku entry conditions."""
        if current_idx < 60:  # Need more data for Ichimoku
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if current['volume'] < self.get_param('min_volume', 10000):
            return False, "Low volume"
        
        # Don't trade in cloud
        if current['in_cloud']:
            return False, "Price in cloud - uncertain area"
        
        # Bullish Ichimoku entry
        if current['ichimoku_bullish']:
            return True, "Ichimoku bullish setup"
        
        # Bearish Ichimoku entry
        if current['ichimoku_bearish']:
            return True, "Ichimoku bearish setup"
        
        return False, "No Ichimoku signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check Ichimoku exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # Price returned to cloud
        if current['in_cloud']:
            return True, "Price returned to cloud"
        
        # Cloud color change
        if current['tk_bullish_cross'] or current['tk_bearish_cross']:
            return True, "Tenkan-Kijun cross reversal"
        
        # Chikou span reversal
        if (entry_price > current['cloud_top'] and current['chikou_below_price']):
            return True, "Chikou span turned bearish"
        
        if (entry_price < current['cloud_bottom'] and current['chikou_above_price']):
            return True, "Chikou span turned bullish"
        
        return False, "Hold Ichimoku position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use cloud boundary as stop loss."""
        current = df.iloc[current_idx]
        
        # Use cloud as dynamic support/resistance
        if 'cloud_top' in current and 'cloud_bottom' in current:
            if entry_price > current['cloud_top']:  # Long position
                return current['cloud_top']
            elif entry_price < current['cloud_bottom']:  # Short position
                return current['cloud_bottom']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on cloud thickness and risk-reward."""
        current = df.iloc[current_idx]
        
        if ('cloud_top' in current and 'cloud_bottom' in current and 
            'cloud_thickness' in current):
            
            # Use cloud thickness as basis for target
            cloud_size = current['cloud_top'] - current['cloud_bottom']
            risk_reward = self.get_param('target_rr', 2.0)
            
            if entry_price > current['cloud_top']:  # Long
                risk = entry_price - current['cloud_top']
                return entry_price + (risk * risk_reward)
            elif entry_price < current['cloud_bottom']:  # Short
                risk = current['cloud_bottom'] - entry_price
                return entry_price - (risk * risk_reward)
        
        # Fallback target
        return entry_price * 1.025