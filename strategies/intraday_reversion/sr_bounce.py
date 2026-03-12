"""
Support/Resistance Bounce Strategy
Mean reversion strategy using dynamic support and resistance levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class SupportResistanceBounceStrategy(BaseStrategy):
    """
    Support/Resistance Bounce Strategy.
    
    Entry: When price bounces off identified support/resistance levels
    Exit: When price breaks through level or reaches opposite level
    """
    
    @property
    def name(self) -> str:
        return "Support Resistance Bounce"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'lookback_period': 50,
            'min_touches': 2,
            'level_tolerance': 0.2,  # % tolerance for level identification
            'bounce_confirmation': 0.1,  # % move away from level for confirmation
            'volume_threshold': 1.4,
            'min_volume': 7000,
            'stop_loss_pct': 1.0,
            'target_opposite_level': True,
            'rsi_confirmation': True,
            'rsi_period': 14,
            'time_filter_start': "09:45",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support/resistance bounce indicators."""
        df = df.copy()
        
        lookback_period = self.get_param('lookback_period', 50)
        min_touches = self.get_param('min_touches', 2)
        level_tolerance = self.get_param('level_tolerance', 0.2) / 100
        
        # Initialize support/resistance levels
        df['support_level'] = np.nan
        df['resistance_level'] = np.nan
        df['support_strength'] = 0
        df['resistance_strength'] = 0
        
        # Calculate support and resistance levels
        for i in range(lookback_period, len(df)):
            window_data = df.iloc[i-lookback_period:i]
            
            # Find potential support levels (local lows)
            lows = window_data['low'].values
            low_indices = []
            
            for j in range(2, len(lows)-2):
                if (lows[j] < lows[j-1] and lows[j] < lows[j-2] and 
                    lows[j] < lows[j+1] and lows[j] < lows[j+2]):
                    low_indices.append(j)
            
            # Find potential resistance levels (local highs)
            highs = window_data['high'].values
            high_indices = []
            
            for j in range(2, len(highs)-2):
                if (highs[j] > highs[j-1] and highs[j] > highs[j-2] and 
                    highs[j] > highs[j+1] and highs[j] > highs[j+2]):
                    high_indices.append(j)
            
            # Identify support levels
            if low_indices:
                low_levels = [lows[idx] for idx in low_indices]
                support_level, support_count = self._find_strongest_level(low_levels, level_tolerance, min_touches)
                
                if support_level is not None:
                    df.loc[df.index[i], 'support_level'] = support_level
                    df.loc[df.index[i], 'support_strength'] = support_count
            
            # Identify resistance levels
            if high_indices:
                high_levels = [highs[idx] for idx in high_indices]
                resistance_level, resistance_count = self._find_strongest_level(high_levels, level_tolerance, min_touches)
                
                if resistance_level is not None:
                    df.loc[df.index[i], 'resistance_level'] = resistance_level
                    df.loc[df.index[i], 'resistance_strength'] = resistance_count
        
        # Forward fill levels
        df['support_level'] = df['support_level'].fillna(method='ffill')
        df['resistance_level'] = df['resistance_level'].fillna(method='ffill')
        df['support_strength'] = df['support_strength'].fillna(method='ffill')
        df['resistance_strength'] = df['resistance_strength'].fillna(method='ffill')
        
        # Bounce detection
        bounce_confirmation = self.get_param('bounce_confirmation', 0.1) / 100
        
        df['near_support'] = (
            (~pd.isna(df['support_level'])) &
            (df['low'] <= df['support_level'] * (1 + level_tolerance)) &
            (df['low'] >= df['support_level'] * (1 - level_tolerance))
        )
        
        df['near_resistance'] = (
            (~pd.isna(df['resistance_level'])) &
            (df['high'] >= df['resistance_level'] * (1 - level_tolerance)) &
            (df['high'] <= df['resistance_level'] * (1 + level_tolerance))
        )
        
        df['support_bounce'] = (
            df['near_support'] &
            (df['close'] > df['support_level'] * (1 + bounce_confirmation)) &
            (df['close'] > df['close'].shift(1))
        )
        
        df['resistance_bounce'] = (
            df['near_resistance'] &
            (df['close'] < df['resistance_level'] * (1 - bounce_confirmation)) &
            (df['close'] < df['close'].shift(1))
        )
        
        # RSI confirmation (optional)
        if self.get_param('rsi_confirmation', True):
            rsi_period = self.get_param('rsi_period', 14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            df['rsi_oversold'] = df['rsi'] < 35
            df['rsi_overbought'] = df['rsi'] > 65
            
            # Add RSI confirmation to bounce signals
            df['support_bounce'] &= df['rsi_oversold']
            df['resistance_bounce'] &= df['rsi_overbought']
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.4))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:45")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        return df
    
    def _find_strongest_level(self, levels, tolerance, min_touches):
        """Find the strongest support/resistance level."""
        if len(levels) < min_touches:
            return None, 0
        
        level_groups = []
        
        for level in levels:
            # Find all levels within tolerance
            similar_levels = [l for l in levels if abs(l - level) / level <= tolerance]
            
            if len(similar_levels) >= min_touches:
                avg_level = np.mean(similar_levels)
                level_groups.append((avg_level, len(similar_levels)))
        
        if not level_groups:
            return None, 0
        
        # Return the level with most touches
        strongest_level = max(level_groups, key=lambda x: x[1])
        return strongest_level[0], strongest_level[1]
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate support/resistance bounce signals."""
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
            
            # Support bounce signal
            if (row['support_bounce'] and row['volume_spike'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Support bounce (Level: {row['support_level']:.2f}, Strength: {int(row['support_strength'])})",
                    confidence=0.7 + min(row['support_strength'] * 0.05, 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Resistance bounce signal
            elif (row['resistance_bounce'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Resistance bounce (Level: {row['resistance_level']:.2f}, Strength: {int(row['resistance_strength'])})",
                    confidence=0.7 + min(row['resistance_strength'] * 0.05, 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check support/resistance bounce entry conditions."""
        if current_idx < 60:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Time filter
        if not current['time_filter']:
            return False, "Outside trading hours"
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if not current['volume_spike']:
            return False, "No volume confirmation"
        
        # Support bounce
        if current['support_bounce']:
            return True, f"Support bounce (level: {current['support_level']:.2f}, strength: {int(current['support_strength'])})"
        
        # Resistance bounce
        if current['resistance_bounce']:
            return True, f"Resistance bounce (level: {current['resistance_level']:.2f}, strength: {int(current['resistance_strength'])})"
        
        return False, "No support/resistance bounce"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check support/resistance bounce exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Level break (failed bounce)
        level_tolerance = self.get_param('level_tolerance', 0.2) / 100
        
        if (entry_price > current['support_level'] and 
            current['close'] < current['support_level'] * (1 - level_tolerance)):
            return True, "Support level broken"
        
        if (entry_price < current['resistance_level'] and 
            current['close'] > current['resistance_level'] * (1 + level_tolerance)):
            return True, "Resistance level broken"
        
        # Target reached (opposite level)
        if self.get_param('target_opposite_level', True):
            if (entry_price > current['support_level'] and 
                not pd.isna(current['resistance_level']) and
                current['close'] >= current['resistance_level'] * 0.98):
                return True, "Reached resistance level"
            
            if (entry_price < current['resistance_level'] and 
                not pd.isna(current['support_level']) and
                current['close'] <= current['support_level'] * 1.02):
                return True, "Reached support level"
        
        # Time-based exit (after 1 hour)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 1:
            return True, "Time-based exit (1 hour)"
        
        return False, "Hold support/resistance bounce position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use support/resistance level as stop loss."""
        current = df.iloc[current_idx]
        
        level_tolerance = self.get_param('level_tolerance', 0.2) / 100
        
        # Use the bounced level as stop
        if (not pd.isna(current['support_level']) and 
            entry_price > current['support_level']):
            return current['support_level'] * (1 - level_tolerance)
        
        if (not pd.isna(current['resistance_level']) and 
            entry_price < current['resistance_level']):
            return current['resistance_level'] * (1 + level_tolerance)
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use opposite level as target."""
        current = df.iloc[current_idx]
        
        # Use opposite level as target
        if self.get_param('target_opposite_level', True):
            if (not pd.isna(current['support_level']) and 
                entry_price > current['support_level'] and
                not pd.isna(current['resistance_level'])):
                return current['resistance_level']
            
            if (not pd.isna(current['resistance_level']) and 
                entry_price < current['resistance_level'] and
                not pd.isna(current['support_level'])):
                return current['support_level']
        
        # Fallback target based on level distance
        if not pd.isna(current['support_level']) and not pd.isna(current['resistance_level']):
            level_distance = abs(current['resistance_level'] - current['support_level'])
            return entry_price + (level_distance * 0.5)  # 50% of level distance
        
        # Final fallback
        return entry_price * 1.02