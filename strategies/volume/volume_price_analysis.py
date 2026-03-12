"""
Volume Price Analysis Strategy
Strategy based on volume and price relationship analysis for confirmation of moves.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from ..base import BaseStrategy, MarketType, TimeFrame, Signal


class VolumePriceAnalysisStrategy(BaseStrategy):
    """
    Volume Price Analysis Strategy.
    
    Entry: When volume confirms price movements (high volume on breakouts, low volume on pullbacks)
    Exit: When volume diverges from price action
    """
    
    @property
    def name(self) -> str:
        return "Volume Price Analysis"
    
    def get_timeframe(self) -> TimeFrame:
        return TimeFrame.MINUTE_5
    
    def get_market_type(self) -> MarketType:
        return MarketType.EQUITY
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'volume_ma_period': 20,
            'price_ma_period': 20,
            'volume_spike_threshold': 2.0,  # 2x average volume
            'volume_dry_threshold': 0.5,    # 0.5x average volume
            'price_move_threshold': 0.5,    # Minimum price move %
            'vwap_period': 20,
            'min_volume': 10000,
            'stop_loss_pct': 1.5,
            'target_pct': 3.0,
            'time_filter_start': "09:30",
            'time_filter_end': "15:00"
        }
    
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume price analysis indicators."""
        df = df.copy()
        
        volume_ma_period = self.get_param('volume_ma_period', 20)
        price_ma_period = self.get_param('price_ma_period', 20)
        volume_spike_threshold = self.get_param('volume_spike_threshold', 2.0)
        volume_dry_threshold = self.get_param('volume_dry_threshold', 0.5)
        price_move_threshold = self.get_param('price_move_threshold', 0.5) / 100
        vwap_period = self.get_param('vwap_period', 20)
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_spike'] = df['volume_ratio'] > volume_spike_threshold
        df['volume_dry'] = df['volume_ratio'] < volume_dry_threshold
        
        # Price indicators
        df['price_ma'] = df['close'].rolling(window=price_ma_period).mean()
        df['price_change'] = df['close'].pct_change()
        df['price_move_up'] = df['price_change'] > price_move_threshold
        df['price_move_down'] = df['price_change'] < -price_move_threshold
        
        # VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['volume']).rolling(window=vwap_period).sum() / df['volume'].rolling(window=vwap_period).sum()
        
        # Price position relative to VWAP
        df['above_vwap'] = df['close'] > df['vwap']
        df['below_vwap'] = df['close'] < df['vwap']
        
        # Volume-Price patterns
        # 1. Volume Confirmation (high volume on price moves)
        df['bullish_volume_confirmation'] = (
            df['price_move_up'] & 
            df['volume_spike'] &
            df['above_vwap']
        )
        
        df['bearish_volume_confirmation'] = (
            df['price_move_down'] & 
            df['volume_spike'] &
            df['below_vwap']
        )
        
        # 2. Volume Divergence (low volume on price moves - potential reversal)
        df['bullish_volume_divergence'] = (
            df['price_move_up'] & 
            df['volume_dry'] &
            (df['close'] > df['price_ma'])
        )
        
        df['bearish_volume_divergence'] = (
            df['price_move_down'] & 
            df['volume_dry'] &
            (df['close'] < df['price_ma'])
        )
        
        # 3. Volume Accumulation/Distribution patterns
        df['money_flow_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['money_flow_multiplier'] = df['money_flow_multiplier'].fillna(0)
        df['money_flow_volume'] = df['money_flow_multiplier'] * df['volume']
        df['accumulation_distribution'] = df['money_flow_volume'].cumsum()
        
        # AD Line trend
        df['ad_ma'] = df['accumulation_distribution'].rolling(window=10).mean()
        df['ad_rising'] = df['accumulation_distribution'] > df['ad_ma']
        df['ad_falling'] = df['accumulation_distribution'] < df['ad_ma']
        
        # 4. Volume Breakout patterns
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        
        df['volume_breakout_up'] = (
            (df['close'] > df['resistance'].shift(1)) &
            df['volume_spike'] &
            df['ad_rising']
        )
        
        df['volume_breakout_down'] = (
            (df['close'] < df['support'].shift(1)) &
            df['volume_spike'] &
            df['ad_falling']
        )
        
        # 5. Volume Climax patterns (exhaustion)
        df['volume_climax_up'] = (
            df['price_move_up'] &
            (df['volume_ratio'] > volume_spike_threshold * 1.5) &  # Extreme volume
            (df['close'] < df['high'])  # Failed to close at high
        )
        
        df['volume_climax_down'] = (
            df['price_move_down'] &
            (df['volume_ratio'] > volume_spike_threshold * 1.5) &  # Extreme volume
            (df['close'] > df['low'])  # Failed to close at low
        )
        
        # Time filter
        start_time = pd.Timestamp(self.get_param('time_filter_start', "09:30")).time()
        end_time = pd.Timestamp(self.get_param('time_filter_end', "15:00")).time()
        df['time_filter'] = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        # Signal strength based on volume characteristics
        df['volume_strength'] = 0.0
        
        for i in range(len(df)):
            strength = 0.0
            
            # Base strength from volume ratio
            vol_ratio = df['volume_ratio'].iloc[i] if not np.isnan(df['volume_ratio'].iloc[i]) else 1.0
            strength = min(vol_ratio / 3, 1.0)  # Normalize
            
            # Boost for VWAP alignment
            if ((df['bullish_volume_confirmation'].iloc[i] and df['above_vwap'].iloc[i]) or
                (df['bearish_volume_confirmation'].iloc[i] and df['below_vwap'].iloc[i])):
                strength *= 1.3
            
            # Boost for AD line confirmation
            if ((df['bullish_volume_confirmation'].iloc[i] and df['ad_rising'].iloc[i]) or
                (df['bearish_volume_confirmation'].iloc[i] and df['ad_falling'].iloc[i])):
                strength *= 1.2
            
            # Boost for breakouts
            if df['volume_breakout_up'].iloc[i] or df['volume_breakout_down'].iloc[i]:
                strength *= 1.4
            
            df.loc[df.index[i], 'volume_strength'] = min(strength, 1.0)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate volume price analysis signals."""
        signals = []
        df_with_indicators = self.indicators(df)
        
        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            
            # Time filter
            if not row['time_filter']:
                continue
            
            # Volume filter
            if row['volume'] < self.get_param('min_volume', 10000):
                continue
            
            # Bullish volume confirmation
            if (row['bullish_volume_confirmation'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bullish volume confirmation (Vol ratio: {row['volume_ratio']:.1f}, Strength: {row['volume_strength']:.2f})",
                    confidence=0.75 + (row['volume_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Bearish volume confirmation
            elif (row['bearish_volume_confirmation'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.8,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Bearish volume confirmation (Vol ratio: {row['volume_ratio']:.1f}, Strength: {row['volume_strength']:.2f})",
                    confidence=0.75 + (row['volume_strength'] * 0.2),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            # Volume breakouts
            elif (row['volume_breakout_up'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="BUY",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Volume breakout up (Vol ratio: {row['volume_ratio']:.1f})",
                    confidence=0.8 + (row['volume_strength'] * 0.15),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
            
            elif (row['volume_breakout_down'] and self.is_market_open(row.name)):
                
                signal = Signal(
                    action="SELL",
                    strength=0.85,
                    price=row['close'],
                    timestamp=row.name,
                    reason=f"Volume breakout down (Vol ratio: {row['volume_ratio']:.1f})",
                    confidence=0.8 + (row['volume_strength'] * 0.15),
                    stop_loss=self.get_stoploss(df_with_indicators, i, row['close']),
                    target=self.get_target(df_with_indicators, i, row['close'])
                )
                signals.append(signal)
        
        return signals
    
    def should_enter(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """Check volume price analysis entry conditions."""
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
        if current['volume'] < self.get_param('min_volume', 10000):
            return False, "Low volume"
        
        # Bullish signals
        if current['bullish_volume_confirmation']:
            return True, f"Bullish volume confirmation (ratio: {current['volume_ratio']:.1f})"
        
        if current['volume_breakout_up']:
            return True, f"Volume breakout up (ratio: {current['volume_ratio']:.1f})"
        
        # Bearish signals
        if current['bearish_volume_confirmation']:
            return True, f"Bearish volume confirmation (ratio: {current['volume_ratio']:.1f})"
        
        if current['volume_breakout_down']:
            return True, f"Volume breakout down (ratio: {current['volume_ratio']:.1f})"
        
        return False, "No volume price signal"
    
    def should_exit(self, df: pd.DataFrame, current_idx: int, entry_price: float, 
                   entry_time: pd.Timestamp) -> Tuple[bool, str]:
        """Check volume price analysis exit conditions."""
        df_with_indicators = self.indicators(df)
        current = df_with_indicators.iloc[current_idx]
        
        # Market close
        if current.name.time() >= pd.Timestamp("15:20").time():
            return True, "Market closing"
        
        # Volume divergence (volume drying up on continued move)
        if (entry_price < current['close'] and current['bullish_volume_divergence']):
            return True, "Bullish volume divergence - potential reversal"
        
        if (entry_price > current['close'] and current['bearish_volume_divergence']):
            return True, "Bearish volume divergence - potential reversal"
        
        # Volume climax (exhaustion)
        if (entry_price < current['close'] and current['volume_climax_up']):
            return True, "Volume climax up - potential exhaustion"
        
        if (entry_price > current['close'] and current['volume_climax_down']):
            return True, "Volume climax down - potential exhaustion"
        
        # VWAP reversal
        if (entry_price > current['vwap'] and current['close'] < current['vwap']):
            return True, "Price fell below VWAP"
        
        if (entry_price < current['vwap'] and current['close'] > current['vwap']):
            return True, "Price rose above VWAP"
        
        # Time-based exit (after 2 hours)
        time_in_trade = (current.name - entry_time).total_seconds() / 3600  # hours
        if time_in_trade > 2:
            return True, "Time-based exit (2 hours)"
        
        return False, "Hold volume price position"
    
    def get_stoploss(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Use VWAP or recent swing as stop loss."""
        current = df.iloc[current_idx]
        
        # Use VWAP as dynamic stop
        if not np.isnan(current['vwap']):
            if entry_price > current['vwap']:  # Long position
                return current['vwap'] * 0.999
            else:  # Short position
                return current['vwap'] * 1.001
        
        # Fallback to percentage stop
        stop_pct = self.get_param('stop_loss_pct', 1.5) / 100
        return entry_price * (1 - stop_pct)
    
    def get_target(self, df: pd.DataFrame, current_idx: int, entry_price: float) -> float:
        """Calculate target based on volume strength and recent range."""
        current = df.iloc[current_idx]
        
        # Base target
        target_pct = self.get_param('target_pct', 3.0) / 100
        
        # Adjust based on volume strength
        if 'volume_strength' in current and not np.isnan(current['volume_strength']):
            # Higher volume strength = higher target potential
            strength_adjustment = current['volume_strength'] * 0.5
            target_pct *= (1 + strength_adjustment)
        
        # Adjust based on volume ratio
        if 'volume_ratio' in current and not np.isnan(current['volume_ratio']):
            if current['volume_ratio'] > 3:  # Very high volume
                target_pct *= 1.5
        
        return entry_price * (1 + target_pct)