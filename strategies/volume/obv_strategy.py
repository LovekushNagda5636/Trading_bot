"""
On Balance Volume (OBV) Strategy
Strategy based on On Balance Volume indicator for trend confirmation and divergence detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class OBVStrategy(BaseStrategy):
    """
    On Balance Volume Strategy.
    
    Entry: When OBV confirms price trends or shows divergence patterns
    Exit: When OBV diverges from price action or trend changes
    """
    
    @property
    def name(self) -> str:
        return "OBV Strategy"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'obv_ma_period': 20,
            'price_ma_period': 20,
            'divergence_lookback': 15,
            'trend_confirmation_period': 5,
            'min_price_move': 0.5,      # Minimum price move % for signals
            'volume_threshold': 1.2,
            'min_volume': 8000,
            'stop_loss_pct': 1.5,
            'target_pct': 2.5,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate OBV indicators."""
        df = df.copy()
        
        obv_ma_period = self.get_param('obv_ma_period', 20)
        price_ma_period = self.get_param('price_ma_period', 20)
        divergence_lookback = self.get_param('divergence_lookback', 15)
        trend_confirmation_period = self.get_param('trend_confirmation_period', 5)
        min_price_move = self.get_param('min_price_move', 0.5) / 100
        
        # Calculate OBV
        df['price_change'] = df['close'].diff()
        df['obv'] = 0.0
        
        # Initialize OBV calculation
        obv_value = 0
        for i in range(len(df)):
            if i == 0:
                obv_value = df['volume'].iloc[i]
            else:
                if df['price_change'].iloc[i] > 0:
                    obv_value += df['volume'].iloc[i]
                elif df['price_change'].iloc[i] < 0:
                    obv_value -= df['volume'].iloc[i]
                # If price unchanged, OBV remains same
            
            df.loc[df.index[i], 'obv'] = obv_value
        
        # OBV moving average and trend
        df['obv_ma'] = df['obv'].rolling(window=obv_ma_period).mean()
        df['obv_trend_up'] = df['obv'] > df['obv_ma']
        df['obv_trend_down'] = df['obv'] < df['obv_ma']
        
        # Price moving average and trend
        df['price_ma'] = df['close'].rolling(window=price_ma_period).mean()
        df['price_trend_up'] = df['close'] > df['price_ma']
        df['price_trend_down'] = df['close'] < df['price_ma']
        
        # OBV slope (rate of change)
        df['obv_slope'] = df['obv'].diff(5)  # 5-period slope
        df['price_slope'] = df['close'].diff(5)  # 5-period slope
        
        # Trend confirmation signals
        df['obv_bullish_confirmation'] = (
            df['price_trend_up'] &
            df['obv_trend_up'] &
            (df['obv_slope'] > 0) &
            (df['price_slope'] > 0) &
            (df['close'].pct_change() > min_price_move)
        )
        
        df['obv_bearish_confirmation'] = (
            df['price_trend_down'] &
            df['obv_trend_down'] &
            (df['obv_slope'] < 0) &
            (df['price_slope'] < 0) &
            (df['close'].pct_change() < -min_price_move)
        )
        
        # Divergence detection
        df['bullish_divergence'] = False
        df['bearish_divergence'] = False
        
        for i in range(divergence_lookback, len(df)):
            # Look back for divergence patterns
            lookback_data = df.iloc[i-divergence_lookback:i+1]
            
            # Find recent price and OBV extremes
            price_low_idx = lookback_data['close'].idxmin()
            price_high_idx = lookback_data['close'].idxmax()
            obv_low_idx = lookback_data['obv'].idxmin()
            obv_high_idx = lookback_data['obv'].idxmax()
            
            current_price = df['close'].iloc[i]
            current_obv = df['obv'].iloc[i]
            
            # Bullish divergence: Price makes lower low, OBV makes higher low
            if (price_low_idx == lookback_data.index[-1] and  # Recent price low
                current_price < lookback_data['close'].iloc[0] and  # Lower than start
                current_obv > lookback_data['obv'].min()):  # OBV higher than its low
                df.loc[df.index[i], 'bullish_divergence'] = True
            
            # Bearish divergence: Price makes higher high, OBV makes lower high
            if (price_high_idx == lookback_data.index[-1] and  # Recent price high
                current_price > lookback_data['close'].iloc[0] and  # Higher than start
                current_obv < lookback_data['obv'].max()):  # OBV lower than its high
                df.loc[df.index[i], 'bearish_divergence'] = True
        
        # OBV breakout signals
        df['obv_breakout_up'] = (
            (df['obv'] > df['obv'].rolling(window=20).max().shift(1)) &
            df['price_trend_up'] &
            (df['obv_slope'] > 0)
        )
        
        df['obv_breakout_down'] = (
            (df['obv'] < df['obv'].rolling(window=20).min().shift(1)) &
            df['price_trend_down'] &
            (df['obv_slope'] < 0)
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.2))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # OBV strength calculation
        df['obv_strength'] = 0.0
        
        for i in range(len(df)):
            strength = 0.0
            
            # Base strength from OBV slope magnitude
            if not np.isnan(df['obv_slope'].iloc[i]):
                slope_strength = min(abs(df['obv_slope'].iloc[i]) / df['volume'].iloc[i], 1.0)
                strength = slope_strength
            
            # Boost for trend confirmation
            if (df['obv_bullish_confirmation'].iloc[i] or 
                df['obv_bearish_confirmation'].iloc[i]):
                strength *= 1.3
            
            # Boost for divergence
            if (df['bullish_divergence'].iloc[i] or 
                df['bearish_divergence'].iloc[i]):
                strength *= 1.5
            
            # Boost for breakouts
            if (df['obv_breakout_up'].iloc[i] or 
                df['obv_breakout_down'].iloc[i]):
                strength *= 1.4
            
            # Boost for volume
            if df['volume_spike'].iloc[i]:
                strength *= 1.2
            
            df.loc[df.index[i], 'obv_strength'] = min(strength, 1.0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate OBV signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Time filter
            if not row['time_filter']:
                continue
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 8000):
                continue
            
            # OBV bullish confirmation
            if (row['obv_bullish_confirmation'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"OBV bullish confirmation (OBV slope: {row['obv_slope']:.0f}, Strength: {row['obv_strength']:.2f})",
                    confidence=0.7 + (row['obv_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # OBV bearish confirmation
            elif (row['obv_bearish_confirmation'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"OBV bearish confirmation (OBV slope: {row['obv_slope']:.0f}, Strength: {row['obv_strength']:.2f})",
                    confidence=0.7 + (row['obv_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bullish divergence
            elif (row['bullish_divergence'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"OBV bullish divergence (Strength: {row['obv_strength']:.2f})",
                    confidence=0.75 + (row['obv_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish divergence
            elif (row['bearish_divergence'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"OBV bearish divergence (Strength: {row['obv_strength']:.2f})",
                    confidence=0.75 + (row['obv_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # OBV breakouts
            elif (row['obv_breakout_up'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"OBV breakout up (Strength: {row['obv_strength']:.2f})",
                    confidence=0.8 + (row['obv_strength'] * 0.15),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            elif (row['obv_breakout_down'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"OBV breakout down (Strength: {row['obv_strength']:.2f})",
                    confidence=0.8 + (row['obv_strength'] * 0.15),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check OBV entry conditions."""
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
        if current['volume'] < self.get_param('min_volume', 8000):
            return False, "Low volume"
        
        # OBV confirmation signals
        if current['obv_bullish_confirmation']:
            return True, f"OBV bullish confirmation (slope: {current['obv_slope']:.0f})"
        
        if current['obv_bearish_confirmation']:
            return True, f"OBV bearish confirmation (slope: {current['obv_slope']:.0f})"
        
        # Divergence signals
        if current['bullish_divergence'] and current['volume_spike']:
            return True, "OBV bullish divergence"
        
        if current['bearish_divergence'] and current['volume_spike']:
            return True, "OBV bearish divergence"
        
        # Breakout signals
        if current['obv_breakout_up'] and current['volume_spike']:
            return True, "OBV breakout up"
        
        if current['obv_breakout_down'] and current['volume_spike']:
            return True, "OBV breakout down"
        
        return False, "No OBV signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check OBV exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # OBV trend reversal
        if (entry_price < current['close'] and current['obv_trend_down']):
            return True, "OBV trend turned down"
        
        if (entry_price > current['close'] and current['obv_trend_up']):
            return True, "OBV trend turned up"
        
        # OBV slope reversal
        if (entry_price < current['close'] and current['obv_slope'] < 0):
            return True, "OBV slope turned negative"
        
        if (entry_price > current['close'] and current['obv_slope'] > 0):
            return True, "OBV slope turned positive"
        
        # Opposite divergence
        if (entry_price < current['close'] and current['bearish_divergence']):
            return True, "OBV bearish divergence detected"
        
        if (entry_price > current['close'] and current['bullish_divergence']):
            return True, "OBV bullish divergence detected"
        
        # Time-based exit (after 3 hours)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 3:
            return True, "Time-based exit (3 hours)"
        
        return False, "Hold OBV position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use recent swing or OBV support/resistance as stop."""
        current = df.iloc[current_idx]
        
        # Use recent swing levels
        if current_idx >= 10:
            recent_data = df.iloc[current_idx-10:current_idx]
            
            if current['obv_trend_up']:  # Long position
                return recent_data['low'].min()
            elif current['obv_trend_down']:  # Short position
                return recent_data['high'].max()
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.5) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on OBV strength and recent range."""
        current = df.iloc[current_idx]
        
        # Base target
        target_pct = self.get_param('target_pct', 2.5) / 100
        
        # Adjust based on OBV strength
        if 'obv_strength' in current and not np.isnan(current['obv_strength']):
            strength_adjustment = current['obv_strength'] * 0.5
            target_pct *= (1 + strength_adjustment)
        
        # Higher target for divergence signals
        if current['bullish_divergence'] or current['bearish_divergence']:
            target_pct *= 1.3
        
        # Higher target for breakouts
        if current['obv_breakout_up'] or current['obv_breakout_down']:
            target_pct *= 1.4
        
        return entry_price * (1 + target_pct)