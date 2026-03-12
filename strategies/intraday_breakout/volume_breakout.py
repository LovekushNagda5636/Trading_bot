"""
Volume Breakout Strategy
Breakout strategy based on unusual volume spikes with price movement.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class VolumeBreakoutStrategy(BaseStrategy):
    """
    Volume Breakout Strategy.
    
    Entry: When unusual volume spike occurs with significant price movement
    Exit: When volume normalizes or price reverses
    """
    
    @property
    def name(self) -> str:
        return "Volume Breakout"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'volume_spike_threshold': 3.0,  # 3x average volume
            'volume_ma_period': 20,
            'price_move_threshold': 0.5,  # Minimum % price move
            'min_volume': 10000,
            'volume_percentile_threshold': 90,  # Top 10% volume
            'stop_loss_pct': 1.0,
            'target_pct': 2.5,
            'volume_decay_threshold': 0.5,  # Exit when volume drops to 50% of spike
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume breakout indicators."""
        df = df.copy()
        
        volume_ma_period = self.get_param('volume_ma_period', 20)
        volume_spike_threshold = self.get_param('volume_spike_threshold', 3.0)
        price_move_threshold = self.get_param('price_move_threshold', 0.5)
        percentile_threshold = self.get_param('volume_percentile_threshold', 90)
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=volume_ma_period).mean()
        df['volume_std'] = df['volume'].rolling(window=volume_ma_period).std()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volume spike detection
        df['volume_spike'] = df['volume_ratio'] > volume_spike_threshold
        
        # Volume percentile (relative to recent history)
        df['volume_percentile'] = df['volume'].rolling(window=50).rank(pct=True) * 100
        df['high_volume_percentile'] = df['volume_percentile'] > percentile_threshold
        
        # Price movement analysis
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['price_range_pct'] = (df['high'] - df['low']) / df['open'] * 100
        
        # Significant price movement
        df['significant_price_move'] = abs(df['price_change_pct']) > price_move_threshold
        df['significant_range'] = df['price_range_pct'] > price_move_threshold
        
        # Combined volume-price signals
        df['volume_price_breakout_up'] = (
            (df['volume_spike'] | df['high_volume_percentile']) &
            df['significant_price_move'] &
            (df['price_change_pct'] > 0)
        )
        
        df['volume_price_breakout_down'] = (
            (df['volume_spike'] | df['high_volume_percentile']) &
            df['significant_price_move'] &
            (df['price_change_pct'] < 0)
        )
        
        # Volume momentum (rate of volume change)
        df['volume_momentum'] = df['volume'] - df['volume'].shift(1)
        df['volume_acceleration'] = df['volume_momentum'] - df['volume_momentum'].shift(1)
        
        # Price momentum confirmation
        df['price_momentum_5'] = df['close'] - df['close'].shift(5)
        df['bullish_momentum'] = df['price_momentum_5'] > 0
        df['bearish_momentum'] = df['price_momentum_5'] < 0
        
        # Volume-Price Trend (VPT) indicator
        df['vpt'] = 0.0
        for i in range(1, len(df)):
            price_change = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
            df.loc[df.index[i], 'vpt'] = df['vpt'].iloc[i-1] + (df['volume'].iloc[i] * price_change)
        
        df['vpt_ma'] = df['vpt'].rolling(window=10).mean()
        df['vpt_bullish'] = df['vpt'] > df['vpt_ma']
        df['vpt_bearish'] = df['vpt'] < df['vpt_ma']
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Volume profile analysis
        df['volume_above_avg'] = df['volume'] > df['volume_ma']
        df['consecutive_high_volume'] = 0
        
        for i in range(1, len(df)):
            if df['volume_above_avg'].iloc[i]:
                df.loc[df.index[i], 'consecutive_high_volume'] = df['consecutive_high_volume'].iloc[i-1] + 1
            else:
                df.loc[df.index[i], 'consecutive_high_volume'] = 0
        
        # Volume exhaustion detection
        df['volume_exhaustion'] = (
            (df['volume_ratio'] < 0.5) &  # Volume dropped significantly
            (df['volume_ratio'].shift(1) > 2.0)  # Previous bar had high volume
        )
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate volume breakout signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Time filter
            if not row['time_filter']:
                continue
            
            # Minimum volume filter
            if row['volume'] < self.get_param('min_volume', 10000):
                continue
            
            # Bullish volume breakout
            if (row['volume_price_breakout_up'] and row['bullish_momentum'] and 
                row['vpt_bullish'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Volume breakout up (Vol ratio: {row['volume_ratio']:.1f}x, Price: +{row['price_change_pct']:.2f}%)",
                    confidence=0.8 + min(row['volume_ratio'] * 0.02, 0.15),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish volume breakout
            elif (row['volume_price_breakout_down'] and row['bearish_momentum'] and 
                  row['vpt_bearish'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Volume breakout down (Vol ratio: {row['volume_ratio']:.1f}x, Price: {row['price_change_pct']:.2f}%)",
                    confidence=0.8 + min(row['volume_ratio'] * 0.02, 0.15),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check volume breakout entry conditions."""
        if current_idx < 25:
            return False, "Insufficient data"
        
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Time filter
        if not current['time_filter']:
            return False, "Outside trading hours"
        
        if not self.is_market_open(current.name):
            return False, "Market closed"
        
        # Minimum volume check
        if current['volume'] < self.get_param('min_volume', 10000):
            return False, "Volume too low"
        
        # Bullish volume breakout
        if (current['volume_price_breakout_up'] and current['bullish_momentum'] and 
            current['vpt_bullish']):
            return True, f"Volume breakout up (ratio: {current['volume_ratio']:.1f}x, price: +{current['price_change_pct']:.2f}%)"
        
        # Bearish volume breakout
        if (current['volume_price_breakout_down'] and current['bearish_momentum'] and 
            current['vpt_bearish']):
            return True, f"Volume breakout down (ratio: {current['volume_ratio']:.1f}x, price: {current['price_change_pct']:.2f}%)"
        
        return False, "No volume breakout"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check volume breakout exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Volume exhaustion
        if current['volume_exhaustion']:
            return True, "Volume exhaustion detected"
        
        # Volume decay (volume dropped significantly from spike)
        decay_threshold = self.get_param('volume_decay_threshold', 0.5)
        if current['volume_ratio'] < decay_threshold:
            return True, f"Volume decayed below {decay_threshold}x average"
        
        # VPT momentum reversal
        if (entry_price > current['close'] and current['vpt_bearish']):
            return True, "VPT turned bearish"
        
        if (entry_price < current['close'] and current['vpt_bullish']):
            return True, "VPT turned bullish"
        
        # Price momentum reversal
        if (entry_price > current['close'] and current['bearish_momentum']):
            return True, "Price momentum turned bearish"
        
        if (entry_price < current['close'] and current['bullish_momentum']):
            return True, "Price momentum turned bullish"
        
        # Time-based exit (after 1 hour if no strong volume)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 1 and current['volume_ratio'] < 1.5:
            return True, "Time-based exit (1 hour, low volume)"
        
        return False, "Hold volume breakout position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use recent low/high or percentage stop."""
        current = df.iloc[current_idx]
        
        # Use recent swing low/high (5 bars)
        if current_idx >= 5:
            recent_data = df.iloc[current_idx-5:current_idx]
            
            if entry_price > current['close']:  # Long position
                recent_low = recent_data['low'].min()
                return recent_low
            else:  # Short position
                recent_high = recent_data['high'].max()
                return recent_high
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.0) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use volume-based target calculation."""
        current = df.iloc[current_idx]
        
        # Base target on volume strength
        base_target_pct = self.get_param('target_pct', 2.5) / 100
        
        # Adjust target based on volume ratio (higher volume = higher target)
        volume_adjustment = min(current['volume_ratio'] * 0.2, 1.0)  # Max 100% adjustment
        adjusted_target_pct = base_target_pct * (1 + volume_adjustment)
        
        return entry_price * (1 + adjusted_target_pct)