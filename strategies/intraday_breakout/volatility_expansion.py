"""
Volatility Expansion Strategy
Breakout strategy based on volatility expansion after contraction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class VolatilityExpansionStrategy(BaseStrategy):
    """
    Volatility Expansion Strategy.
    
    Entry: When volatility expands after contraction period
    Exit: When volatility contracts again or end of day
    """
    
    @property
    def name(self) -> str:
        return "Volatility Expansion"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 14,
            'volatility_lookback': 20,
            'contraction_threshold': 0.7,  # ATR below 70% of average
            'expansion_threshold': 1.3,    # ATR above 130% of average
            'min_contraction_bars': 5,
            'volume_threshold': 1.5,
            'min_volume': 7000,
            'stop_loss_atr': 2.0,
            'target_atr': 3.0
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility expansion indicators."""
        df = df.copy()
        
        atr_period = self.get_param('atr_period', 14)
        volatility_lookback = self.get_param('volatility_lookback', 20)
        
        # Calculate ATR (Average True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
        
        # Calculate volatility metrics
        df['atr_avg'] = df['atr'].rolling(window=volatility_lookback).mean()
        df['atr_ratio'] = df['atr'] / df['atr_avg']
        
        # Volatility states
        contraction_threshold = self.get_param('contraction_threshold', 0.7)
        expansion_threshold = self.get_param('expansion_threshold', 1.3)
        
        df['low_volatility'] = df['atr_ratio'] < contraction_threshold
        df['high_volatility'] = df['atr_ratio'] > expansion_threshold
        df['normal_volatility'] = ~(df['low_volatility'] | df['high_volatility'])
        
        # Count consecutive low volatility bars
        df['contraction_count'] = 0
        min_contraction = self.get_param('min_contraction_bars', 5)
        
        for i in range(min_contraction, len(df)):
            if df['low_volatility'].iloc[i]:
                # Count consecutive low volatility bars
                count = 1
                for j in range(i-1, max(i-min_contraction*2, 0), -1):
                    if df['low_volatility'].iloc[j]:
                        count += 1
                    else:
                        break
                df.loc[df.index[i], 'contraction_count'] = count
        
        # Volatility expansion after contraction
        df['volatility_expansion'] = (
            df['high_volatility'] & 
            (df['contraction_count'].shift(1) >= min_contraction)
        )
        
        # Price direction during expansion
        df['price_change'] = df['close'] - df['close'].shift(1)
        df['expansion_bullish'] = df['volatility_expansion'] & (df['price_change'] > 0)
        df['expansion_bearish'] = df['volatility_expansion'] & (df['price_change'] < 0)
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.5))
        
        # Bollinger Band squeeze (additional confirmation)
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * df['bb_std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=20).quantile(0.2)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate volatility expansion signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 7000):
                continue
            
            # Skip if no volatility expansion
            if not row['volatility_expansion']:
                continue
            
            # Bullish volatility expansion
            if (row['expansion_bullish'] and row['volume_spike'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bullish volatility expansion (ATR ratio: {row['atr_ratio']:.2f}, Contraction: {int(row['contraction_count'])} bars)",
                    confidence=0.8,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish volatility expansion
            elif (row['expansion_bearish'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bearish volatility expansion (ATR ratio: {row['atr_ratio']:.2f}, Contraction: {int(row['contraction_count'])} bars)",
                    confidence=0.8,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check volatility expansion entry conditions."""
        if current_idx < 40:  # Need enough data for volatility calculations
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if not current['volume_spike']:
            return False, "No volume confirmation"
        
        # Volatility expansion check
        if not current['volatility_expansion']:
            return False, "No volatility expansion"
        
        # Bullish expansion
        if current['expansion_bullish']:
            return True, f"Bullish volatility expansion (ratio: {current['atr_ratio']:.2f})"
        
        # Bearish expansion
        if current['expansion_bearish']:
            return True, f"Bearish volatility expansion (ratio: {current['atr_ratio']:.2f})"
        
        return False, "No volatility expansion signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check volatility expansion exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # Volatility contraction
        if current['low_volatility']:
            return True, "Volatility contracted"
        
        # Return to normal volatility after significant time
        time_in_trade = (current.name - entry_time).total_seconds() / 60  # minutes
        if time_in_trade > 60 and current['normal_volatility']:  # 1 hour
            return True, "Volatility normalized after 1 hour"
        
        return False, "Hold volatility expansion position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use ATR-based stop loss."""
        current = df.iloc[current_idx]
        
        if 'atr' in current and not pd.isna(current['atr']):
            atr_multiplier = self.get_param('stop_loss_atr', 2.0)
            return entry_price - (current['atr'] * atr_multiplier)
        
        # Fallback
        return entry_price * 0.985
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use ATR-based target."""
        current = df.iloc[current_idx]
        
        if 'atr' in current and not pd.isna(current['atr']):
            atr_multiplier = self.get_param('target_atr', 3.0)
            return entry_price + (current['atr'] * atr_multiplier)
        
        # Fallback
        return entry_price * 1.025