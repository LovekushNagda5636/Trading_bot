"""
Heikin Ashi Strategy
Trend following strategy using Heikin Ashi candlesticks for smoother trend identification.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class HeikinAshiStrategy(BaseStrategy):
    """
    Heikin Ashi Strategy for trend following.
    
    Entry: When Heikin Ashi shows strong trend with no lower shadows (bullish) or upper shadows (bearish)
    Exit: When Heikin Ashi shows trend reversal or doji formation
    """
    
    @property
    def name(self) -> str:
        return "Heikin Ashi"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'min_consecutive_candles': 2,
            'min_volume': 7000,
            'trend_filter': True,
            'trend_ema': 50,
            'stop_loss_pct': 1.0,
            'target_pct': 2.0,
            'doji_threshold': 0.1  # % for doji detection
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Heikin Ashi indicators."""
        df = df.copy()
        
        # Initialize Heikin Ashi values
        df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['ha_open'] = np.nan
        df['ha_high'] = np.nan
        df['ha_low'] = np.nan
        
        # Calculate Heikin Ashi Open
        df.loc[df.index[0], 'ha_open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        
        for i in range(1, len(df)):
            df.loc[df.index[i], 'ha_open'] = (df['ha_open'].iloc[i-1] + df['ha_close'].iloc[i-1]) / 2
        
        # Calculate Heikin Ashi High and Low
        df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
        df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        # Heikin Ashi candle properties
        df['ha_body'] = abs(df['ha_close'] - df['ha_open'])
        df['ha_upper_shadow'] = df['ha_high'] - np.maximum(df['ha_open'], df['ha_close'])
        df['ha_lower_shadow'] = np.minimum(df['ha_open'], df['ha_close']) - df['ha_low']
        
        # Candle colors
        df['ha_green'] = df['ha_close'] > df['ha_open']
        df['ha_red'] = df['ha_close'] < df['ha_open']
        
        # Doji detection
        doji_threshold = self.get_param('doji_threshold', 0.1) / 100
        df['ha_doji'] = df['ha_body'] < (df['ha_high'] - df['ha_low']) * doji_threshold
        
        # Strong trend signals
        df['ha_strong_bullish'] = (
            df['ha_green'] & 
            (df['ha_lower_shadow'] < df['ha_body'] * 0.1) &  # Very small lower shadow
            (df['ha_upper_shadow'] < df['ha_body'] * 0.3)    # Small upper shadow
        )
        
        df['ha_strong_bearish'] = (
            df['ha_red'] & 
            (df['ha_upper_shadow'] < df['ha_body'] * 0.1) &  # Very small upper shadow
            (df['ha_lower_shadow'] < df['ha_body'] * 0.3)    # Small lower shadow
        )
        
        # Consecutive candle counting
        df['consecutive_green'] = 0
        df['consecutive_red'] = 0
        
        for i in range(1, len(df)):
            if df['ha_green'].iloc[i]:
                if df['ha_green'].iloc[i-1]:
                    df.loc[df.index[i], 'consecutive_green'] = df['consecutive_green'].iloc[i-1] + 1
                else:
                    df.loc[df.index[i], 'consecutive_green'] = 1
            
            if df['ha_red'].iloc[i]:
                if df['ha_red'].iloc[i-1]:
                    df.loc[df.index[i], 'consecutive_red'] = df['consecutive_red'].iloc[i-1] + 1
                else:
                    df.loc[df.index[i], 'consecutive_red'] = 1
        
        # Trend filter
        if self.get_param('trend_filter', True):
            trend_period = self.get_param('trend_ema', 50)
            df['trend_ema'] = df['close'].ewm(span=trend_period).mean()
            df['uptrend'] = df['close'] > df['trend_ema']
            df['downtrend'] = df['close'] < df['trend_ema']
        else:
            df['uptrend'] = True
            df['downtrend'] = True
        
        # Entry signals
        min_consecutive = self.get_param('min_consecutive_candles', 2)
        
        df['ha_bullish_entry'] = (
            df['ha_strong_bullish'] &
            (df['consecutive_green'] >= min_consecutive) &
            df['uptrend'] &
            ~df['ha_doji']
        )
        
        df['ha_bearish_entry'] = (
            df['ha_strong_bearish'] &
            (df['consecutive_red'] >= min_consecutive) &
            df['downtrend'] &
            ~df['ha_doji']
        )
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate Heikin Ashi signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 7000):
                continue
            
            # Bullish Heikin Ashi signal
            if row['ha_bullish_entry']:
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Heikin Ashi bullish trend ({int(row['consecutive_green'])} consecutive green)",
                    confidence=0.75,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish Heikin Ashi signal
            elif row['ha_bearish_entry']:
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Heikin Ashi bearish trend ({int(row['consecutive_red'])} consecutive red)",
                    confidence=0.75,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check Heikin Ashi entry conditions."""
        if current_idx < 10:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if current['volume'] < self.get_param('min_volume', 7000):
            return False, "Low volume"
        
        # Bullish Heikin Ashi entry
        if current['ha_bullish_entry']:
            return True, f"Heikin Ashi bullish ({int(current['consecutive_green'])} green candles)"
        
        # Bearish Heikin Ashi entry
        if current['ha_bearish_entry']:
            return True, f"Heikin Ashi bearish ({int(current['consecutive_red'])} red candles)"
        
        return False, "No Heikin Ashi signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check Heikin Ashi exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:25").time():
            return True, "Market closing"
        
        # Doji formation (indecision)
        if current['ha_doji']:
            return True, "Heikin Ashi doji - trend indecision"
        
        # Color change (trend reversal)
        if current_idx > 0:
            previous = df_with_indicators.iloc[current_idx - 1]
            
            # Exit long on red candle after green trend
            if (entry_price > previous['close'] and 
                current['ha_red'] and previous['ha_green']):
                return True, "Heikin Ashi color change to red"
            
            # Exit short on green candle after red trend
            if (entry_price < previous['close'] and 
                current['ha_green'] and previous['ha_red']):
                return True, "Heikin Ashi color change to green"
        
        return False, "Hold Heikin Ashi position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use Heikin Ashi low/high as stop loss."""
        current = df.iloc[current_idx]
        
        # Use previous Heikin Ashi candle's extreme as stop
        if current_idx > 0:
            previous = df.iloc[current_idx - 1]
            
            if 'ha_low' in previous and 'ha_high' in previous:
                if entry_price > previous['ha_close']:  # Long position
                    return previous['ha_low']
                else:  # Short position
                    return previous['ha_high']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on Heikin Ashi body size."""
        current = df.iloc[current_idx]
        
        if 'ha_body' in current and current['ha_body'] > 0:
            # Target based on multiple of current candle body
            body_size = current['ha_body']
            target_multiplier = 2.0
            
            if current['ha_green']:  # Long
                return entry_price + (body_size * target_multiplier)
            else:  # Short
                return entry_price - (body_size * target_multiplier)
        
        # Fallback target
        target_pct = self.get_param('target_pct', 2.0) / 100
        return entry_price * (1 + target_pct)