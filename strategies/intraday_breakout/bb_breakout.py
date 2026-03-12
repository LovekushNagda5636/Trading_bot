"""
Bollinger Bands Breakout Strategy
Breakout strategy using Bollinger Bands for entry and exit signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class BollingerBandsBreakoutStrategy(BaseStrategy):
    """
    Bollinger Bands Breakout Strategy.
    
    Entry: When price breaks above upper band or below lower band with volume
    Exit: When price returns to middle band or opposite breakout
    """
    
    @property
    def name(self) -> str:
        return "Bollinger Bands Breakout"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'bb_period': 20,
            'bb_std': 2.0,
            'squeeze_threshold': 0.8,  # Band width percentile for squeeze
            'volume_threshold': 1.5,
            'min_volume': 6000,
            'breakout_confirmation': 0.05,  # % beyond band for confirmation
            'stop_loss_pct': 1.0,
            'target_bb_width': 2.0  # Target as multiple of band width
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands indicators."""
        df = df.copy()
        
        bb_period = self.get_param('bb_period', 20)
        bb_std = self.get_param('bb_std', 2.0)
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        df['bb_std_dev'] = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * df['bb_std_dev'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * df['bb_std_dev'])
        
        # Band width and position
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Squeeze detection (narrow bands)
        squeeze_threshold = self.get_param('squeeze_threshold', 0.8)
        df['bb_width_percentile'] = df['bb_width'].rolling(window=50).rank(pct=True)
        df['bb_squeeze'] = df['bb_width_percentile'] < (1 - squeeze_threshold)
        
        # Breakout detection
        breakout_conf = self.get_param('breakout_confirmation', 0.05) / 100
        df['bb_upper_breakout_level'] = df['bb_upper'] * (1 + breakout_conf)
        df['bb_lower_breakout_level'] = df['bb_lower'] * (1 - breakout_conf)
        
        df['bb_upper_breakout'] = df['close'] > df['bb_upper_breakout_level']
        df['bb_lower_breakout'] = df['close'] < df['bb_lower_breakout_level']
        
        # Breakout after squeeze (high probability setup)
        df['squeeze_breakout_up'] = (
            df['bb_upper_breakout'] & 
            df['bb_squeeze'].shift(1)  # Previous bar was in squeeze
        )
        
        df['squeeze_breakout_down'] = (
            df['bb_lower_breakout'] & 
            df['bb_squeeze'].shift(1)  # Previous bar was in squeeze
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.5))
        
        # %B indicator (position within bands)
        df['percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Band walking (consecutive closes outside bands)
        df['walking_upper'] = (df['close'] > df['bb_upper']).astype(int)
        df['walking_lower'] = (df['close'] < df['bb_lower']).astype(int)
        
        # Count consecutive walks
        df['upper_walk_count'] = 0
        df['lower_walk_count'] = 0
        
        for i in range(1, len(df)):
            if df['walking_upper'].iloc[i]:
                df.loc[df.index[i], 'upper_walk_count'] = df['upper_walk_count'].iloc[i-1] + 1
            
            if df['walking_lower'].iloc[i]:
                df.loc[df.index[i], 'lower_walk_count'] = df['lower_walk_count'].iloc[i-1] + 1
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate Bollinger Bands breakout signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 6000):
                continue
            
            # Upper band breakout (especially after squeeze)
            if ((row['bb_upper_breakout'] or row['squeeze_breakout_up']) and 
                row['volume_spike'] and self.is_market_open(row.name)):
                
                reason = "BB squeeze breakout up" if row['squeeze_breakout_up'] else "BB upper breakout"
                confidence = 0.85 if row['squeeze_breakout_up'] else 0.75
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"{reason} (Width: {row['bb_width']:.2f}%, %B: {row['percent_b']:.2f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Lower band breakout (especially after squeeze)
            elif ((row['bb_lower_breakout'] or row['squeeze_breakout_down']) and 
                  row['volume_spike'] and self.is_market_open(row.name)):
                
                reason = "BB squeeze breakout down" if row['squeeze_breakout_down'] else "BB lower breakout"
                confidence = 0.85 if row['squeeze_breakout_down'] else 0.75
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"{reason} (Width: {row['bb_width']:.2f}%, %B: {row['percent_b']:.2f})",
                    confidence=confidence,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check Bollinger Bands entry conditions."""
        if current_idx < 25:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if not current['volume_spike']:
            return False, "No volume confirmation"
        
        # Upper breakout (prioritize squeeze breakouts)
        if current['squeeze_breakout_up']:
            return True, f"BB squeeze breakout up (%B: {current['percent_b']:.2f})"
        elif current['bb_upper_breakout']:
            return True, f"BB upper breakout (%B: {current['percent_b']:.2f})"
        
        # Lower breakout (prioritize squeeze breakouts)
        if current['squeeze_breakout_down']:
            return True, f"BB squeeze breakout down (%B: {current['percent_b']:.2f})"
        elif current['bb_lower_breakout']:
            return True, f"BB lower breakout (%B: {current['percent_b']:.2f})"
        
        return False, "No BB breakout"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check Bollinger Bands exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # Return to middle band (mean reversion)
        if (entry_price > current['bb_upper'] and 
            current['close'] < current['bb_middle']):
            return True, "Price returned to BB middle"
        
        if (entry_price < current['bb_lower'] and 
            current['close'] > current['bb_middle']):
            return True, "Price returned to BB middle"
        
        # Opposite band touch (strong reversal)
        if (entry_price > current['bb_upper'] and 
            current['close'] < current['bb_lower']):
            return True, "Price touched opposite BB"
        
        if (entry_price < current['bb_lower'] and 
            current['close'] > current['bb_upper']):
            return True, "Price touched opposite BB"
        
        # Band walking exhaustion (after 3+ consecutive closes outside)
        if (entry_price > current['bb_upper'] and 
            current['upper_walk_count'] > 3 and 
            current['close'] < current['bb_upper']):
            return True, "Upper band walk exhausted"
        
        if (entry_price < current['bb_lower'] and 
            current['lower_walk_count'] > 3 and 
            current['close'] > current['bb_lower']):
            return True, "Lower band walk exhausted"
        
        return False, "Hold BB breakout position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use middle band or opposite band as stop loss."""
        current = df.iloc[current_idx]
        
        if 'bb_middle' in current and 'bb_upper' in current and 'bb_lower' in current:
            # For upper breakouts, stop at middle band
            if entry_price > current['bb_upper']:
                return current['bb_middle']
            # For lower breakouts, stop at middle band
            elif entry_price < current['bb_lower']:
                return current['bb_middle']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on band width."""
        current = df.iloc[current_idx]
        
        if ('bb_upper' in current and 'bb_lower' in current and 
            'bb_middle' in current):
            
            band_width = current['bb_upper'] - current['bb_lower']
            target_multiplier = self.get_param('target_bb_width', 2.0)
            
            # Target based on band width
            if entry_price > current['bb_upper']:  # Long breakout
                return entry_price + (band_width * target_multiplier)
            elif entry_price < current['bb_lower']:  # Short breakout
                return entry_price - (band_width * target_multiplier)
        
        # Fallback target
        return entry_price * 1.025