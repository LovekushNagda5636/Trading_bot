"""
Doji Strategy
Price action strategy based on doji candlestick patterns indicating indecision and potential reversal.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class DojiStrategy(BaseStrategy):
    """
    Doji Strategy.
    
    Entry: When doji pattern forms at key levels with confirmation
    Exit: When pattern fails or target reached
    """
    
    @property
    def name(self) -> str:
        return "Doji"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'max_body_pct': 5,      # Maximum body as % of total range for doji
            'min_range_atr': 0.6,   # Minimum range as multiple of ATR
            'volume_threshold': 1.2,
            'min_volume': 5000,
            'atr_period': 14,
            'trend_period': 20,     # Period for trend identification
            'confirmation_bars': 2,  # Bars to wait for confirmation
            'stop_loss_pct': 1.2,
            'target_pct': 2.0,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate doji indicators."""
        df = df.copy()
        
        # Calculate ATR
        atr_period = self.get_param('atr_period', 14)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
        
        # Candlestick components
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Body percentage of total range
        df['body_pct'] = (df['body_size'] / df['total_range'] * 100).fillna(0)
        
        # Doji detection
        max_body_pct = self.get_param('max_body_pct', 5)
        min_range_atr = self.get_param('min_range_atr', 0.6)
        
        df['is_doji'] = (
            (df['body_pct'] <= max_body_pct) &
            (df['total_range'] >= df['atr'] * min_range_atr)
        )
        
        # Doji types
        df['standard_doji'] = (
            df['is_doji'] &
            (df['upper_wick'] > df['body_size'] * 2) &
            (df['lower_wick'] > df['body_size'] * 2)
        )
        
        df['dragonfly_doji'] = (
            df['is_doji'] &
            (df['lower_wick'] > df['upper_wick'] * 3) &
            (df['lower_wick'] > df['body_size'] * 5)
        )
        
        df['gravestone_doji'] = (
            df['is_doji'] &
            (df['upper_wick'] > df['lower_wick'] * 3) &
            (df['upper_wick'] > df['body_size'] * 5)
        )
        
        # Trend identification
        trend_period = self.get_param('trend_period', 20)
        df['ema'] = df['close'].ewm(span=trend_period).mean()
        df['uptrend'] = df['close'] > df['ema']
        df['downtrend'] = df['close'] < df['ema']
        
        # Trend strength
        df['trend_strength'] = abs(df['close'] - df['ema']) / df['ema']
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.2))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Doji reversal signals
        df['bullish_doji_reversal'] = (
            (df['standard_doji'] | df['dragonfly_doji']) &
            df['downtrend'] &
            (df['trend_strength'] > 0.02)  # Significant trend
        )
        
        df['bearish_doji_reversal'] = (
            (df['standard_doji'] | df['gravestone_doji']) &
            df['uptrend'] &
            (df['trend_strength'] > 0.02)  # Significant trend
        )
        
        # Confirmation tracking
        confirmation_bars = self.get_param('confirmation_bars', 2)
        df['bullish_confirmation'] = False
        df['bearish_confirmation'] = False
        
        for i in range(confirmation_bars, len(df)):
            # Check for bullish confirmation after doji
            if df['bullish_doji_reversal'].iloc[i - confirmation_bars]:
                # Look for higher closes in confirmation period
                confirmation_period = df.iloc[i - confirmation_bars + 1:i + 1]
                doji_close = df['close'].iloc[i - confirmation_bars]
                
                if any(confirmation_period['close'] > doji_close * 1.002):  # 0.2% higher
                    df.loc[df.index[i], 'bullish_confirmation'] = True
            
            # Check for bearish confirmation after doji
            if df['bearish_doji_reversal'].iloc[i - confirmation_bars]:
                # Look for lower closes in confirmation period
                confirmation_period = df.iloc[i - confirmation_bars + 1:i + 1]
                doji_close = df['close'].iloc[i - confirmation_bars]
                
                if any(confirmation_period['close'] < doji_close * 0.998):  # 0.2% lower
                    df.loc[df.index[i], 'bearish_confirmation'] = True
        
        # Doji strength
        df['doji_strength'] = 0.0
        
        for i in range(len(df)):
            if df['bullish_doji_reversal'].iloc[i] or df['bearish_doji_reversal'].iloc[i]:
                # Base strength from trend strength (stronger trend = stronger reversal potential)
                strength = min(df['trend_strength'].iloc[i] * 10, 1.0)
                
                # Boost for specific doji types
                if df['dragonfly_doji'].iloc[i] and df['bullish_doji_reversal'].iloc[i]:
                    strength *= 1.3  # Dragonfly is stronger bullish signal
                elif df['gravestone_doji'].iloc[i] and df['bearish_doji_reversal'].iloc[i]:
                    strength *= 1.3  # Gravestone is stronger bearish signal
                
                # Boost for volume
                if df['volume_spike'].iloc[i]:
                    strength *= 1.2
                
                # Boost for larger range
                if not np.isnan(df['atr'].iloc[i]):
                    range_factor = df['total_range'].iloc[i] / df['atr'].iloc[i]
                    if range_factor > 1.2:
                        strength *= 1.1
                
                df.loc[df.index[i], 'doji_strength'] = min(strength, 1.0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate doji reversal signals."""
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
            
            # Bullish doji confirmation signal
            if (row['bullish_confirmation'] and self.is_market_open(row.name)):
                
                doji_type = "dragonfly" if row['dragonfly_doji'] else "standard"
                
                signal = Signal(
                    action="BUY",
                    strength=0.7,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bullish {doji_type} doji reversal confirmed (Strength: {row['doji_strength']:.2f})",
                    confidence=0.6 + (row['doji_strength'] * 0.3),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish doji confirmation signal
            elif (row['bearish_confirmation'] and self.is_market_open(row.name)):
                
                doji_type = "gravestone" if row['gravestone_doji'] else "standard"
                
                signal = Signal(
                    action="SELL",
                    strength=0.7,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bearish {doji_type} doji reversal confirmed (Strength: {row['doji_strength']:.2f})",
                    confidence=0.6 + (row['doji_strength'] * 0.3),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check doji entry conditions."""
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
        if row['volume'] < self.get_param('min_volume', 5000):
            return False, "Low volume"
        
        # Bullish confirmation
        if current['bullish_confirmation']:
            doji_type = "dragonfly" if current['dragonfly_doji'] else "standard"
            return True, f"Bullish {doji_type} doji reversal confirmed"
        
        # Bearish confirmation
        if current['bearish_confirmation']:
            doji_type = "gravestone" if current['gravestone_doji'] else "standard"
            return True, f"Bearish {doji_type} doji reversal confirmed"
        
        return False, "No doji confirmation signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check doji exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Opposite doji signal
        if (entry_price > current['close'] and current['bearish_confirmation']):
            return True, "Opposite bearish doji confirmed"
        
        if (entry_price < current['close'] and current['bullish_confirmation']):
            return True, "Opposite bullish doji confirmed"
        
        # Trend resumption (price moved back in original trend direction)
        if current_idx >= 5:
            recent_trend = df_with_indicators.iloc[current_idx-5:current_idx]['close'].mean()
            
            if (entry_price > recent_trend and current['close'] < recent_trend * 0.995):
                return True, "Bullish reversal failed - trend resumed"
            
            if (entry_price < recent_trend and current['close'] > recent_trend * 1.005):
                return True, "Bearish reversal failed - trend resumed"
        
        # Time-based exit (after 3 hours)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 3:
            return True, "Time-based exit (3 hours)"
        
        return False, "Hold doji position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use doji extreme as stop loss."""
        confirmation_bars = self.get_param('confirmation_bars', 2)
        
        # Find the doji bar (confirmation_bars ago)
        if current_idx >= confirmation_bars:
            doji_idx = current_idx - confirmation_bars
            doji_bar = df.iloc[doji_idx]
            
            if entry_price > doji_bar['close']:  # Long position
                return doji_bar['low']
            else:  # Short position
                return doji_bar['high']
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.2) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on recent range."""
        if current_idx >= 20:
            recent_data = df.iloc[current_idx-20:current_idx]
            recent_range = recent_data['high'].max() - recent_data['low'].min()
            
            confirmation_bars = self.get_param('confirmation_bars', 2)
            if current_idx >= confirmation_bars:
                doji_idx = current_idx - confirmation_bars
                doji_close = df.iloc[doji_idx]['close']
                
                if entry_price > doji_close:  # Long
                    return entry_price + (recent_range * 0.5)
                else:  # Short
                    return entry_price - (recent_range * 0.5)
        
        # Fallback target
        target_pct = self.get_param('target_pct', 2.0) / 100
        return entry_price * (1 + target_pct)