"""
Accumulation/Distribution Strategy
Strategy based on Accumulation/Distribution Line for detecting money flow and trend strength.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class AccumulationDistributionStrategy(BaseStrategy):
    """
    Accumulation/Distribution Strategy.
    
    Entry: When A/D Line confirms trends or shows divergence patterns
    Exit: When A/D Line diverges from price or shows distribution
    """
    
    @property
    def name(self) -> str:
        return "Accumulation Distribution"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'ad_ma_period': 20,
            'price_ma_period': 20,
            'divergence_lookback': 12,
            'trend_confirmation_period': 5,
            'min_price_move': 0.4,      # Minimum price move % for signals
            'volume_threshold': 1.3,
            'min_volume': 7000,
            'ad_threshold': 0.1,        # A/D momentum threshold
            'stop_loss_pct': 1.4,
            'target_pct': 2.8,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Accumulation/Distribution indicators."""
        df = df.copy()
        
        ad_ma_period = self.get_param('ad_ma_period', 20)
        price_ma_period = self.get_param('price_ma_period', 20)
        divergence_lookback = self.get_param('divergence_lookback', 12)
        min_price_move = self.get_param('min_price_move', 0.4) / 100
        ad_threshold = self.get_param('ad_threshold', 0.1)
        
        # Calculate Money Flow Multiplier
        df['money_flow_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        
        # Handle division by zero (when high == low)
        df['money_flow_multiplier'] = df['money_flow_multiplier'].fillna(0)
        
        # Calculate Money Flow Volume
        df['money_flow_volume'] = df['money_flow_multiplier'] * df['volume']
        
        # Calculate Accumulation/Distribution Line
        df['ad_line'] = df['money_flow_volume'].cumsum()
        
        # A/D Line moving average and trend
        df['ad_ma'] = df['ad_line'].rolling(window=ad_ma_period).mean()
        df['ad_trend_up'] = df['ad_line'] > df['ad_ma']
        df['ad_trend_down'] = df['ad_line'] < df['ad_ma']
        
        # A/D Line momentum (rate of change)
        df['ad_momentum'] = df['ad_line'].diff(5)  # 5-period momentum
        df['ad_momentum_pct'] = df['ad_momentum'] / df['ad_line'].abs()
        
        # Price indicators
        df['price_ma'] = df['close'].rolling(window=price_ma_period).mean()
        df['price_trend_up'] = df['close'] > df['price_ma']
        df['price_trend_down'] = df['close'] < df['price_ma']
        df['price_momentum'] = df['close'].diff(5)
        
        # Accumulation/Distribution phases
        df['accumulation'] = (
            (df['ad_momentum'] > 0) &
            (df['ad_momentum_pct'] > ad_threshold) &
            (df['money_flow_multiplier'] > 0.2)  # Buying pressure
        )
        
        df['distribution'] = (
            (df['ad_momentum'] < 0) &
            (df['ad_momentum_pct'] < -ad_threshold) &
            (df['money_flow_multiplier'] < -0.2)  # Selling pressure
        )
        
        # Trend confirmation signals
        df['ad_bullish_confirmation'] = (
            df['price_trend_up'] &
            df['ad_trend_up'] &
            df['accumulation'] &
            (df['close'].pct_change() > min_price_move)
        )
        
        df['ad_bearish_confirmation'] = (
            df['price_trend_down'] &
            df['ad_trend_down'] &
            df['distribution'] &
            (df['close'].pct_change() < -min_price_move)
        )
        
        # Divergence detection
        df['bullish_divergence'] = False
        df['bearish_divergence'] = False
        
        for i in range(divergence_lookback, len(df)):
            # Look back for divergence patterns
            lookback_data = df.iloc[i-divergence_lookback:i+1]
            
            current_price = df['close'].iloc[i]
            current_ad = df['ad_line'].iloc[i]
            
            # Find price and A/D extremes in lookback period
            price_min = lookback_data['close'].min()
            price_max = lookback_data['close'].max()
            ad_min = lookback_data['ad_line'].min()
            ad_max = lookback_data['ad_line'].max()
            
            # Bullish divergence: Price makes lower low, A/D makes higher low
            if (current_price <= price_min and  # Price at or near low
                current_ad > ad_min and  # A/D above its low
                df['price_trend_down'].iloc[i]):  # In downtrend
                df.loc[df.index[i], 'bullish_divergence'] = True
            
            # Bearish divergence: Price makes higher high, A/D makes lower high
            if (current_price >= price_max and  # Price at or near high
                current_ad < ad_max and  # A/D below its high
                df['price_trend_up'].iloc[i]):  # In uptrend
                df.loc[df.index[i], 'bearish_divergence'] = True
        
        # A/D Line breakout signals
        df['ad_breakout_up'] = (
            (df['ad_line'] > df['ad_line'].rolling(window=20).max().shift(1)) &
            df['accumulation'] &
            df['price_trend_up']
        )
        
        df['ad_breakout_down'] = (
            (df['ad_line'] < df['ad_line'].rolling(window=20).min().shift(1)) &
            df['distribution'] &
            df['price_trend_down']
        )
        
        # Money flow strength
        df['money_flow_strength'] = abs(df['money_flow_multiplier'])
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.3))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # A/D strength calculation
        df['ad_strength'] = 0.0
        
        for i in range(len(df)):
            strength = 0.0
            
            # Base strength from money flow multiplier
            mf_strength = abs(df['money_flow_multiplier'].iloc[i])
            strength = mf_strength
            
            # Boost for A/D momentum
            if not np.isnan(df['ad_momentum_pct'].iloc[i]):
                momentum_strength = min(abs(df['ad_momentum_pct'].iloc[i]) * 10, 1.0)
                strength = max(strength, momentum_strength)
            
            # Boost for trend confirmation
            if (df['ad_bullish_confirmation'].iloc[i] or 
                df['ad_bearish_confirmation'].iloc[i]):
                strength *= 1.3
            
            # Boost for divergence
            if (df['bullish_divergence'].iloc[i] or 
                df['bearish_divergence'].iloc[i]):
                strength *= 1.5
            
            # Boost for breakouts
            if (df['ad_breakout_up'].iloc[i] or 
                df['ad_breakout_down'].iloc[i]):
                strength *= 1.4
            
            # Boost for volume
            if df['volume_spike'].iloc[i]:
                strength *= 1.2
            
            # Boost for strong accumulation/distribution
            if df['accumulation'].iloc[i] or df['distribution'].iloc[i]:
                strength *= 1.2
            
            df.loc[df.index[i], 'ad_strength'] = min(strength, 1.0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate A/D signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Time filter
            if not row['time_filter']:
                continue
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 7000):
                continue
            
            # A/D bullish confirmation
            if (row['ad_bullish_confirmation'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"A/D bullish confirmation (MF: {row['money_flow_multiplier']:.2f}, Strength: {row['ad_strength']:.2f})",
                    confidence=0.7 + (row['ad_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # A/D bearish confirmation
            elif (row['ad_bearish_confirmation'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"A/D bearish confirmation (MF: {row['money_flow_multiplier']:.2f}, Strength: {row['ad_strength']:.2f})",
                    confidence=0.7 + (row['ad_strength'] * 0.2),
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
                    reason=f"A/D bullish divergence (Strength: {row['ad_strength']:.2f})",
                    confidence=0.75 + (row['ad_strength'] * 0.2),
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
                    reason=f"A/D bearish divergence (Strength: {row['ad_strength']:.2f})",
                    confidence=0.75 + (row['ad_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # A/D breakouts
            elif (row['ad_breakout_up'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"A/D breakout up (Accumulation, Strength: {row['ad_strength']:.2f})",
                    confidence=0.8 + (row['ad_strength'] * 0.15),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            elif (row['ad_breakout_down'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"A/D breakout down (Distribution, Strength: {row['ad_strength']:.2f})",
                    confidence=0.8 + (row['ad_strength'] * 0.15),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check A/D entry conditions."""
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
        if current['volume'] < self.get_param('min_volume', 7000):
            return False, "Low volume"
        
        # A/D confirmation signals
        if current['ad_bullish_confirmation']:
            return True, f"A/D bullish confirmation (MF: {current['money_flow_multiplier']:.2f})"
        
        if current['ad_bearish_confirmation']:
            return True, f"A/D bearish confirmation (MF: {current['money_flow_multiplier']:.2f})"
        
        # Divergence signals
        if current['bullish_divergence'] and current['volume_spike']:
            return True, "A/D bullish divergence"
        
        if current['bearish_divergence'] and current['volume_spike']:
            return True, "A/D bearish divergence"
        
        # Breakout signals
        if current['ad_breakout_up'] and current['volume_spike']:
            return True, "A/D breakout up (accumulation)"
        
        if current['ad_breakout_down'] and current['volume_spike']:
            return True, "A/D breakout down (distribution)"
        
        return False, "No A/D signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check A/D exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # A/D trend reversal
        if (entry_price < current['close'] and current['ad_trend_down']):
            return True, "A/D trend turned down"
        
        if (entry_price > current['close'] and current['ad_trend_up']):
            return True, "A/D trend turned up"
        
        # Phase change
        if (entry_price < current['close'] and current['distribution']):
            return True, "Distribution phase detected"
        
        if (entry_price > current['close'] and current['accumulation']):
            return True, "Accumulation phase detected"
        
        # Opposite divergence
        if (entry_price < current['close'] and current['bearish_divergence']):
            return True, "A/D bearish divergence detected"
        
        if (entry_price > current['close'] and current['bullish_divergence']):
            return True, "A/D bullish divergence detected"
        
        # Money flow reversal
        if (entry_price < current['close'] and current['money_flow_multiplier'] < -0.3):
            return True, "Strong selling pressure detected"
        
        if (entry_price > current['close'] and current['money_flow_multiplier'] > 0.3):
            return True, "Strong buying pressure detected"
        
        # Time-based exit (after 2.5 hours)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 2.5:
            return True, "Time-based exit (2.5 hours)"
        
        return False, "Hold A/D position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use recent swing or A/D support/resistance as stop."""
        current = df.iloc[current_idx]
        
        # Use recent swing levels
        if current_idx >= 10:
            recent_data = df.iloc[current_idx-10:current_idx]
            
            if current['ad_trend_up']:  # Long position
                return recent_data['low'].min()
            elif current['ad_trend_down']:  # Short position
                return recent_data['high'].max()
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.4) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on A/D strength and money flow."""
        current = df.iloc[current_idx]
        
        # Base target
        target_pct = self.get_param('target_pct', 2.8) / 100
        
        # Adjust based on A/D strength
        if 'ad_strength' in current and not np.isnan(current['ad_strength']):
            strength_adjustment = current['ad_strength'] * 0.6
            target_pct *= (1 + strength_adjustment)
        
        # Adjust based on money flow strength
        if 'money_flow_strength' in current and not np.isnan(current['money_flow_strength']):
            mf_adjustment = current['money_flow_strength'] * 0.4
            target_pct *= (1 + mf_adjustment)
        
        # Higher target for divergence signals
        if current['bullish_divergence'] or current['bearish_divergence']:
            target_pct *= 1.4
        
        # Higher target for breakouts
        if current['ad_breakout_up'] or current['ad_breakout_down']:
            target_pct *= 1.3
        
        return entry_price * (1 + target_pct)