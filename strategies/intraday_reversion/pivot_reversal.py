"""
Pivot Point Reversal Strategy
Mean reversion strategy using pivot points as reversal levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class PivotReversalStrategy(BaseStrategy):
    """
    Pivot Point Reversal Strategy.
    
    Entry: When price reverses at pivot support/resistance levels
    Exit: When price breaks through level or reaches opposite pivot
    """
    
    @property
    def name(self) -> str:
        return "Pivot Reversal"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'pivot_type': 'standard',  # 'standard', 'fibonacci', 'camarilla'
            'reversal_confirmation': 0.15,  # % move away from pivot for confirmation
            'volume_threshold': 1.5,
            'min_volume': 7000,
            'stop_loss_pct': 1.0,
            'target_next_level': True,
            'rsi_confirmation': True,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'time_filter_start': "10:00",
            'time_filter_end': "14:30"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pivot point reversal indicators."""
        df = df.copy()
        
        # Add date column for daily pivot calculation
        df['date'] = df.index.date
        
        # Calculate pivot points for each day
        df['pivot'] = np.nan
        df['r1'] = np.nan
        df['r2'] = np.nan
        df['r3'] = np.nan
        df['s1'] = np.nan
        df['s2'] = np.nan
        df['s3'] = np.nan
        
        pivot_type = self.get_param('pivot_type', 'standard')
        
        unique_dates = sorted(df['date'].unique())
        
        for i, date in enumerate(unique_dates):
            if i == 0:
                continue  # Skip first day (no previous day)
            
            # Get previous day's data
            prev_date = unique_dates[i-1]
            prev_day_data = df[df['date'] == prev_date]
            
            if len(prev_day_data) > 0:
                prev_high = prev_day_data['high'].max()
                prev_low = prev_day_data['low'].min()
                prev_close = prev_day_data['close'].iloc[-1]
                
                # Calculate pivot points based on type
                if pivot_type == 'standard':
                    pivot = (prev_high + prev_low + prev_close) / 3
                    r1 = 2 * pivot - prev_low
                    r2 = pivot + (prev_high - prev_low)
                    r3 = prev_high + 2 * (pivot - prev_low)
                    s1 = 2 * pivot - prev_high
                    s2 = pivot - (prev_high - prev_low)
                    s3 = prev_low - 2 * (prev_high - pivot)
                
                elif pivot_type == 'fibonacci':
                    pivot = (prev_high + prev_low + prev_close) / 3
                    r1 = pivot + 0.382 * (prev_high - prev_low)
                    r2 = pivot + 0.618 * (prev_high - prev_low)
                    r3 = pivot + (prev_high - prev_low)
                    s1 = pivot - 0.382 * (prev_high - prev_low)
                    s2 = pivot - 0.618 * (prev_high - prev_low)
                    s3 = pivot - (prev_high - prev_low)
                
                elif pivot_type == 'camarilla':
                    pivot = (prev_high + prev_low + prev_close) / 3
                    r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
                    r2 = prev_close + 1.1 * (prev_high - prev_low) / 6
                    r3 = prev_close + 1.1 * (prev_high - prev_low) / 4
                    s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
                    s2 = prev_close - 1.1 * (prev_high - prev_low) / 6
                    s3 = prev_close - 1.1 * (prev_high - prev_low) / 4
                
                # Apply to current day
                current_day_mask = df['date'] == date
                df.loc[current_day_mask, 'pivot'] = pivot
                df.loc[current_day_mask, 'r1'] = r1
                df.loc[current_day_mask, 'r2'] = r2
                df.loc[current_day_mask, 'r3'] = r3
                df.loc[current_day_mask, 's1'] = s1
                df.loc[current_day_mask, 's2'] = s2
                df.loc[current_day_mask, 's3'] = s3
        
        # Reversal detection at pivot levels
        reversal_conf = self.get_param('reversal_confirmation', 0.15) / 100
        
        # Support level reversals (bounces up from support)
        df['s1_reversal'] = (
            (df['low'] <= df['s1'] * 1.002) &  # Touched S1
            (df['close'] > df['s1'] * (1 + reversal_conf)) &  # Bounced up
            (df['close'] > df['close'].shift(1))  # Price rising
        )
        
        df['s2_reversal'] = (
            (df['low'] <= df['s2'] * 1.002) &
            (df['close'] > df['s2'] * (1 + reversal_conf)) &
            (df['close'] > df['close'].shift(1))
        )
        
        df['s3_reversal'] = (
            (df['low'] <= df['s3'] * 1.002) &
            (df['close'] > df['s3'] * (1 + reversal_conf)) &
            (df['close'] > df['close'].shift(1))
        )
        
        df['pivot_support_reversal'] = (
            (df['low'] <= df['pivot'] * 1.002) &
            (df['close'] > df['pivot'] * (1 + reversal_conf)) &
            (df['close'] > df['close'].shift(1))
        )
        
        # Resistance level reversals (bounces down from resistance)
        df['r1_reversal'] = (
            (df['high'] >= df['r1'] * 0.998) &  # Touched R1
            (df['close'] < df['r1'] * (1 - reversal_conf)) &  # Bounced down
            (df['close'] < df['close'].shift(1))  # Price falling
        )
        
        df['r2_reversal'] = (
            (df['high'] >= df['r2'] * 0.998) &
            (df['close'] < df['r2'] * (1 - reversal_conf)) &
            (df['close'] < df['close'].shift(1))
        )
        
        df['r3_reversal'] = (
            (df['high'] >= df['r3'] * 0.998) &
            (df['close'] < df['r3'] * (1 - reversal_conf)) &
            (df['close'] < df['close'].shift(1))
        )
        
        df['pivot_resistance_reversal'] = (
            (df['high'] >= df['pivot'] * 0.998) &
            (df['close'] < df['pivot'] * (1 - reversal_conf)) &
            (df['close'] < df['close'].shift(1))
        )
        
        # Combine all reversal signals
        df['bullish_pivot_reversal'] = (
            df['s1_reversal'] | df['s2_reversal'] | df['s3_reversal'] | df['pivot_support_reversal']
        )
        
        df['bearish_pivot_reversal'] = (
            df['r1_reversal'] | df['r2_reversal'] | df['r3_reversal'] | df['pivot_resistance_reversal']
        )
        
        # Identify which level caused the reversal
        df['reversal_level'] = 'none'
        df['reversal_level_value'] = np.nan
        df['next_target_level'] = np.nan
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            if row['s3_reversal']:
                df.loc[df.index[i], 'reversal_level'] = 'S3'
                df.loc[df.index[i], 'reversal_level_value'] = row['s3']
                df.loc[df.index[i], 'next_target_level'] = row['s2']
            elif row['s2_reversal']:
                df.loc[df.index[i], 'reversal_level'] = 'S2'
                df.loc[df.index[i], 'reversal_level_value'] = row['s2']
                df.loc[df.index[i], 'next_target_level'] = row['s1']
            elif row['s1_reversal']:
                df.loc[df.index[i], 'reversal_level'] = 'S1'
                df.loc[df.index[i], 'reversal_level_value'] = row['s1']
                df.loc[df.index[i], 'next_target_level'] = row['pivot']
            elif row['pivot_support_reversal']:
                df.loc[df.index[i], 'reversal_level'] = 'Pivot_Support'
                df.loc[df.index[i], 'reversal_level_value'] = row['pivot']
                df.loc[df.index[i], 'next_target_level'] = row['r1']
            elif row['r1_reversal']:
                df.loc[df.index[i], 'reversal_level'] = 'R1'
                df.loc[df.index[i], 'reversal_level_value'] = row['r1']
                df.loc[df.index[i], 'next_target_level'] = row['pivot']
            elif row['r2_reversal']:
                df.loc[df.index[i], 'reversal_level'] = 'R2'
                df.loc[df.index[i], 'reversal_level_value'] = row['r2']
                df.loc[df.index[i], 'next_target_level'] = row['r1']
            elif row['r3_reversal']:
                df.loc[df.index[i], 'reversal_level'] = 'R3'
                df.loc[df.index[i], 'reversal_level_value'] = row['r3']
                df.loc[df.index[i], 'next_target_level'] = row['r2']
            elif row['pivot_resistance_reversal']:
                df.loc[df.index[i], 'reversal_level'] = 'Pivot_Resistance'
                df.loc[df.index[i], 'reversal_level_value'] = row['pivot']
                df.loc[df.index[i], 'next_target_level'] = row['s1']
        
        # RSI confirmation (optional)
        if self.get_param('rsi_confirmation', True):
            rsi_period = self.get_param('rsi_period', 14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            rsi_oversold = self.get_param('rsi_oversold', 30)
            rsi_overbought = self.get_param('rsi_overbought', 70)
            
            df['rsi_oversold'] = df['rsi'] < rsi_oversold
            df['rsi_overbought'] = df['rsi'] > rsi_overbought
            
            # Add RSI confirmation
            df['bullish_pivot_reversal'] &= df['rsi_oversold']
            df['bearish_pivot_reversal'] &= df['rsi_overbought']
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.5))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "10:00")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "14:30")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate pivot reversal signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Skip if no pivot data
            if pd.isna(row['pivot']):
                continue
            
            # Time filter
            if not row['time_filter']:
                continue
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 7000):
                continue
            
            # Bullish pivot reversal signal
            if (row['bullish_pivot_reversal'] and row['volume_spike'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Pivot {row['reversal_level']} bullish reversal (Level: {row['reversal_level_value']:.2f})",
                    confidence=0.75,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish pivot reversal signal
            elif (row['bearish_pivot_reversal'] and row['volume_spike'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Pivot {row['reversal_level']} bearish reversal (Level: {row['reversal_level_value']:.2f})",
                    confidence=0.75,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check pivot reversal entry conditions."""
        if current_idx < 10:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Check pivot availability
        if pd.isna(current['pivot']):
            return False, "No pivot data"
        
        # Time filter
        if not current['time_filter']:
            return False, "Outside trading hours"
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if not current['volume_spike']:
            return False, "No volume confirmation"
        
        # Bullish reversal
        if current['bullish_pivot_reversal']:
            return True, f"Pivot {current['reversal_level']} bullish reversal"
        
        # Bearish reversal
        if current['bearish_pivot_reversal']:
            return True, f"Pivot {current['reversal_level']} bearish reversal"
        
        return False, "No pivot reversal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check pivot reversal exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Level break (failed reversal)
        if not pd.isna(current['reversal_level_value']):
            reversal_conf = self.get_param('reversal_confirmation', 0.15) / 100
            
            if (entry_price > current['reversal_level_value'] and 
                current['close'] < current['reversal_level_value'] * (1 - reversal_conf)):
                return True, "Reversal level broken"
            
            if (entry_price < current['reversal_level_value'] and 
                current['close'] > current['reversal_level_value'] * (1 + reversal_conf)):
                return True, "Reversal level broken"
        
        # Target reached (next pivot level)
        if (self.get_param('target_next_level', True) and 
            not pd.isna(current['next_target_level'])):
            
            if (entry_price < current['next_target_level'] and 
                current['close'] >= current['next_target_level'] * 0.995):
                return True, "Reached next pivot level"
            
            if (entry_price > current['next_target_level'] and 
                current['close'] <= current['next_target_level'] * 1.005):
                return True, "Reached next pivot level"
        
        # Time-based exit (after 45 minutes)
        time_in_trade = (current.name - entry_time).total_seconds() / 60  # minutes
        if time_in_trade > 45:
            return True, "Time-based exit (45 minutes)"
        
        return False, "Hold pivot reversal position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use reversal level as stop loss."""
        current = df.iloc[current_idx]
        
        # Use the reversal level as stop
        if not pd.isna(current['reversal_level_value']):
            reversal_conf = self.get_param('reversal_confirmation', 0.15) / 100
            
            if entry_price > current['reversal_level_value']:  # Long position
                return current['reversal_level_value'] * (1 - reversal_conf)
            else:  # Short position
                return current['reversal_level_value'] * (1 + reversal_conf)
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use next pivot level as target."""
        current = df.iloc[current_idx]
        
        # Use next level as target if available
        if (self.get_param('target_next_level', True) and 
            not pd.isna(current['next_target_level'])):
            return current['next_target_level']
        
        # Fallback target
        return entry_price * 1.02