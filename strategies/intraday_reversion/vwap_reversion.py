"""
VWAP Mean Reversion Strategy
Mean reversion strategy using VWAP as the mean reversion level.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class VWAPReversionStrategy(BaseStrategy):
    """
    VWAP Mean Reversion Strategy.
    
    Entry: When price deviates significantly from VWAP and shows reversal signs
    Exit: When price returns to VWAP or continues away
    """
    
    @property
    def name(self) -> str:
        return "VWAP Reversion"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'deviation_threshold': 1.0,  # % deviation from VWAP to trigger
            'max_deviation': 3.0,  # Maximum deviation before avoiding trade
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.2,
            'min_volume': 6000,
            'stop_loss_pct': 1.5,
            'target_vwap_return': 0.8,  # Target 80% return to VWAP
            'time_filter_start': "10:00",
            'time_filter_end': "14:30"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP reversion indicators."""
        df = df.copy()
        
        # Add date column for daily VWAP calculation
        df['date'] = df.index.date
        
        # Calculate VWAP for each day
        df['vwap'] = np.nan
        df['vwap_std'] = np.nan
        
        for date in df['date'].unique():
            day_mask = df['date'] == date
            day_data = df[day_mask].copy()
            
            if len(day_data) > 0:
                # Calculate typical price
                day_data['typical_price'] = (day_data['high'] + day_data['low'] + day_data['close']) / 3
                
                # Calculate cumulative VWAP
                day_data['cum_volume'] = day_data['volume'].cumsum()
                day_data['cum_tp_volume'] = (day_data['typical_price'] * day_data['volume']).cumsum()
                day_data['vwap'] = day_data['cum_tp_volume'] / day_data['cum_volume']
                
                # Calculate VWAP standard deviation
                day_data['price_vwap_diff'] = day_data['typical_price'] - day_data['vwap']
                day_data['cum_squared_diff'] = (day_data['price_vwap_diff'] ** 2 * day_data['volume']).cumsum()
                day_data['vwap_variance'] = day_data['cum_squared_diff'] / day_data['cum_volume']
                day_data['vwap_std'] = np.sqrt(day_data['vwap_variance'])
                
                # Update main dataframe
                df.loc[day_mask, 'vwap'] = day_data['vwap']
                df.loc[day_mask, 'vwap_std'] = day_data['vwap_std']
        
        # Price deviation from VWAP
        df['vwap_deviation_pct'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        df['vwap_z_score'] = (df['close'] - df['vwap']) / df['vwap_std']
        
        # Deviation thresholds
        deviation_threshold = self.get_param('deviation_threshold', 1.0)
        max_deviation = self.get_param('max_deviation', 3.0)
        
        df['oversold_vwap'] = (df['vwap_deviation_pct'] < -deviation_threshold) & (df['vwap_deviation_pct'] > -max_deviation)
        df['overbought_vwap'] = (df['vwap_deviation_pct'] > deviation_threshold) & (df['vwap_deviation_pct'] < max_deviation)
        
        # RSI for additional confirmation
        rsi_period = 14
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        rsi_oversold = self.get_param('rsi_oversold', 30)
        rsi_overbought = self.get_param('rsi_overbought', 70)
        
        df['rsi_oversold'] = df['rsi'] < rsi_oversold
        df['rsi_overbought'] = df['rsi'] > rsi_overbought
        
        # Reversal signals
        df['bullish_reversion_signal'] = (
            df['oversold_vwap'] & 
            df['rsi_oversold'] &
            (df['close'] > df['close'].shift(1))  # Price starting to reverse
        )
        
        df['bearish_reversion_signal'] = (
            df['overbought_vwap'] & 
            df['rsi_overbought'] &
            (df['close'] < df['close'].shift(1))  # Price starting to reverse
        )
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.get_param('volume_threshold', 1.2))
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "10:00")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "14:30")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Price momentum (for reversal confirmation)
        df['price_momentum'] = df['close'] - df['close'].shift(3)
        df['momentum_reversing_up'] = (df['price_momentum'] > 0) & (df['price_momentum'].shift(1) <= 0)
        df['momentum_reversing_down'] = (df['price_momentum'] < 0) & (df['price_momentum'].shift(1) >= 0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate VWAP reversion signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Skip if no VWAP data
            if pd.isna(row['vwap']):
                continue
            
            # Time filter
            if not row['time_filter']:
                continue
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 6000):
                continue
            
            # Bullish reversion signal
            if (row['bullish_reversion_signal'] and row['momentum_reversing_up'] and 
                self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"VWAP bullish reversion (Dev: {row['vwap_deviation_pct']:.2f}%, RSI: {row['rsi']:.1f})",
                    confidence=0.7,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish reversion signal
            elif (row['bearish_reversion_signal'] and row['momentum_reversing_down'] and 
                  self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.75,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"VWAP bearish reversion (Dev: {row['vwap_deviation_pct']:.2f}%, RSI: {row['rsi']:.1f})",
                    confidence=0.7,
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check VWAP reversion entry conditions."""
        if current_idx < 20:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Check VWAP availability
        if pd.isna(current['vwap']):
            return False, "No VWAP data"
        
        # Time filter
        if not current['time_filter']:
            return False, "Outside trading hours"
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Volume check
        if current['volume'] < self.get_param('min_volume', 6000):
            return False, "Low volume"
        
        # Bullish reversion
        if current['bullish_reversion_signal'] and current['momentum_reversing_up']:
            return True, f"VWAP bullish reversion (dev: {current['vwap_deviation_pct']:.2f}%)"
        
        # Bearish reversion
        if current['bearish_reversion_signal'] and current['momentum_reversing_down']:
            return True, f"VWAP bearish reversion (dev: {current['vwap_deviation_pct']:.2f}%)"
        
        return False, "No VWAP reversion signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check VWAP reversion exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Target reached (partial return to VWAP)
        target_return = self.get_param('target_vwap_return', 0.8)
        
        if not pd.isna(current['vwap']):
            vwap_distance = abs(entry_price - current['vwap'])
            current_distance = abs(current['close'] - current['vwap'])
            
            if current_distance <= vwap_distance * (1 - target_return):
                return True, f"Target reached ({target_return*100}% return to VWAP)"
        
        # Price moved further away (failed reversion)
        if (entry_price < current['vwap'] and current['close'] < entry_price * 0.995):
            return True, "Price moved further from VWAP"
        
        if (entry_price > current['vwap'] and current['close'] > entry_price * 1.005):
            return True, "Price moved further from VWAP"
        
        # Time-based exit (after 30 minutes)
        time_in_trade = (current.name - entry_time).total_seconds() / 60  # minutes
        if time_in_trade > 30:
            return True, "Time-based exit (30 minutes)"
        
        return False, "Hold VWAP reversion position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate stop loss for mean reversion."""
        current = df.iloc[current_idx]
        
        # Use a wider stop for mean reversion
        stop_pct = self.get_param('stop_loss_pct', 1.5) / 100
        
        # For mean reversion, stop should be further away from VWAP
        if not pd.isna(current['vwap']):
            if entry_price < current['vwap']:  # Long position (price below VWAP)
                # Stop below entry, further from VWAP
                return entry_price * (1 - stop_pct)
            else:  # Short position (price above VWAP)
                # Stop above entry, further from VWAP
                return entry_price * (1 + stop_pct)
        
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target as partial return to VWAP."""
        current = df.iloc[current_idx]
        
        # Target is partial return to VWAP
        if not pd.isna(current['vwap']):
            target_return = self.get_param('target_vwap_return', 0.8)
            vwap_distance = current['vwap'] - entry_price
            return entry_price + (vwap_distance * target_return)
        
        # Fallback target
        return entry_price * 1.015