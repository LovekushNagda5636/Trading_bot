"""
Nifty Futures Breakout Strategy for Intraday F&O Trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
import logging

from ..base import BaseStrategy
from trading_bot.core.models import Signal, SignalType
from trading_bot.analysis.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class NiftyFuturesBreakout(BaseStrategy):
    """
    Nifty Futures Breakout Strategy for intraday F&O trading.
    
    Logic:
    1. Identify opening range (9:15-9:45 AM)
    2. Wait for breakout above/below range
    3. Enter with momentum confirmation
    4. Use ATR-based stop loss and targets
    5. Square off before 3:20 PM
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Strategy parameters
        self.symbol = config.get('symbol', 'NIFTY')
        self.instrument_type = config.get('instrument_type', 'FUT')
        self.expiry = config.get('expiry', '2025-01-30')
        self.lot_size = config.get('lot_size', 25)
        self.max_lots = config.get('max_lots', 2)
        
        # Technical parameters
        self.breakout_period = config.get('breakout_period', 20)  # minutes
        self.volume_threshold = config.get('volume_threshold', 1.5)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        self.atr_period = config.get('atr_period', 14)
        
        # Trading session parameters
        self.opening_range_start = time(9, 15)
        self.opening_range_end = time(9, 45)
        self.square_off_time = time(15, 20)
        
        # State variables
        self.opening_high = None
        self.opening_low = None
        self.opening_range_set = False
        self.position_taken = False
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        
        # Technical indicators
        self.indicators = TechnicalIndicators()
        
        logger.info(f"Initialized Nifty Futures Breakout Strategy for {self.symbol}")
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on breakout logic."""
        if data.empty or len(data) < self.atr_period:
            return []
        
        signals = []
        current_time = datetime.now().time()
        latest = data.iloc[-1]
        
        # Check if market is open
        if not self._is_market_open(current_time):
            return []
        
        # Set opening range (9:15-9:45 AM)
        if not self.opening_range_set and self._is_opening_range_time(current_time):
            self._set_opening_range(data)
            return []
        
        # Wait for opening range to be set
        if not self.opening_range_set:
            return []
        
        # Check for breakout signals
        if not self.position_taken and current_time > self.opening_range_end:
            signal = self._check_breakout_signal(data, latest)
            if signal:
                signals.append(signal)
                self.position_taken = True
        
        # Check for exit signals
        elif self.position_taken:
            exit_signal = self._check_exit_signal(data, latest, current_time)
            if exit_signal:
                signals.append(exit_signal)
                self.position_taken = False
        
        return signals
    
    def _set_opening_range(self, data: pd.DataFrame) -> None:
        """Set the opening range high and low."""
        try:
            # Get data from opening range period
            opening_data = data[
                (data.index.time >= self.opening_range_start) &
                (data.index.time <= self.opening_range_end)
            ]
            
            if len(opening_data) >= 5:  # Minimum 5 candles
                self.opening_high = opening_data['high'].max()
                self.opening_low = opening_data['low'].min()
                self.opening_range_set = True
                
                logger.info(f"Opening range set: High={self.opening_high}, Low={self.opening_low}")
        
        except Exception as e:
            logger.error(f"Error setting opening range: {e}")
    
    def _check_breakout_signal(self, data: pd.DataFrame, latest: pd.Series) -> Optional[Signal]:
        """Check for breakout signals."""
        try:
            current_price = latest['close']
            volume = latest['volume']
            
            # Calculate ATR for stop loss and target
            atr = self.indicators.atr(data, period=self.atr_period).iloc[-1]
            
            # Calculate average volume
            avg_volume = data['volume'].tail(20).mean()
            volume_confirmed = volume > (avg_volume * self.volume_threshold)
            
            # Check for bullish breakout
            if (current_price > self.opening_high and 
                volume_confirmed and 
                self._momentum_confirmation(data, 'bullish')):
                
                # Calculate position size
                quantity = self.lot_size * min(self.max_lots, self._calculate_position_size(current_price, atr))
                
                # Set stop loss and target
                self.entry_price = current_price
                self.stop_loss = current_price - (atr * self.atr_multiplier)
                self.target = current_price + (atr * self.atr_multiplier * 1.5)  # 1.5:1 R:R
                
                return Signal(
                    symbol=f"{self.symbol}{self.expiry.replace('-', '')[2:]}FUT",
                    signal_type=SignalType.BUY,
                    price=current_price,
                    quantity=quantity,
                    stop_loss=self.stop_loss,
                    target=self.target,
                    confidence=0.8,
                    strategy=self.name,
                    metadata={
                        'breakout_type': 'bullish',
                        'opening_high': self.opening_high,
                        'opening_low': self.opening_low,
                        'atr': atr,
                        'volume_ratio': volume / avg_volume,
                        'lot_size': self.lot_size,
                        'lots': quantity // self.lot_size
                    }
                )
            
            # Check for bearish breakout
            elif (current_price < self.opening_low and 
                  volume_confirmed and 
                  self._momentum_confirmation(data, 'bearish')):
                
                # Calculate position size
                quantity = self.lot_size * min(self.max_lots, self._calculate_position_size(current_price, atr))
                
                # Set stop loss and target
                self.entry_price = current_price
                self.stop_loss = current_price + (atr * self.atr_multiplier)
                self.target = current_price - (atr * self.atr_multiplier * 1.5)  # 1.5:1 R:R
                
                return Signal(
                    symbol=f"{self.symbol}{self.expiry.replace('-', '')[2:]}FUT",
                    signal_type=SignalType.SELL,
                    price=current_price,
                    quantity=quantity,
                    stop_loss=self.stop_loss,
                    target=self.target,
                    confidence=0.8,
                    strategy=self.name,
                    metadata={
                        'breakout_type': 'bearish',
                        'opening_high': self.opening_high,
                        'opening_low': self.opening_low,
                        'atr': atr,
                        'volume_ratio': volume / avg_volume,
                        'lot_size': self.lot_size,
                        'lots': quantity // self.lot_size
                    }
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking breakout signal: {e}")
            return None
    
    def _check_exit_signal(self, data: pd.DataFrame, latest: pd.Series, current_time: time) -> Optional[Signal]:
        """Check for exit signals."""
        try:
            current_price = latest['close']
            
            # Time-based exit (square off before 3:20 PM)
            if current_time >= self.square_off_time:
                return Signal(
                    symbol=f"{self.symbol}{self.expiry.replace('-', '')[2:]}FUT",
                    signal_type=SignalType.EXIT,
                    price=current_price,
                    quantity=0,  # Will be determined by position manager
                    confidence=1.0,
                    strategy=self.name,
                    metadata={'exit_reason': 'time_based_square_off'}
                )
            
            # Stop loss hit
            if self.stop_loss:
                if ((self.entry_price > self.stop_loss and current_price <= self.stop_loss) or
                    (self.entry_price < self.stop_loss and current_price >= self.stop_loss)):
                    
                    return Signal(
                        symbol=f"{self.symbol}{self.expiry.replace('-', '')[2:]}FUT",
                        signal_type=SignalType.EXIT,
                        price=current_price,
                        quantity=0,
                        confidence=1.0,
                        strategy=self.name,
                        metadata={'exit_reason': 'stop_loss_hit'}
                    )
            
            # Target hit
            if self.target:
                if ((self.entry_price < self.target and current_price >= self.target) or
                    (self.entry_price > self.target and current_price <= self.target)):
                    
                    return Signal(
                        symbol=f"{self.symbol}{self.expiry.replace('-', '')[2:]}FUT",
                        signal_type=SignalType.EXIT,
                        price=current_price,
                        quantity=0,
                        confidence=1.0,
                        strategy=self.name,
                        metadata={'exit_reason': 'target_achieved'}
                    )
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking exit signal: {e}")
            return None
    
    def _momentum_confirmation(self, data: pd.DataFrame, direction: str) -> bool:
        """Check for momentum confirmation."""
        try:
            if len(data) < 5:
                return False
            
            # Use EMA crossover for momentum
            ema_fast = self.indicators.ema(data['close'], period=5).iloc[-1]
            ema_slow = self.indicators.ema(data['close'], period=10).iloc[-1]
            
            if direction == 'bullish':
                return ema_fast > ema_slow
            else:
                return ema_fast < ema_slow
        
        except Exception as e:
            logger.error(f"Error in momentum confirmation: {e}")
            return False
    
    def _calculate_position_size(self, price: float, atr: float) -> int:
        """Calculate position size based on risk management."""
        try:
            # Risk 1% of account per trade
            account_balance = 500000  # This should come from broker
            risk_per_trade = account_balance * 0.01
            
            # Calculate risk per lot
            risk_per_lot = atr * self.atr_multiplier * self.lot_size
            
            # Calculate number of lots
            max_lots_by_risk = int(risk_per_trade / risk_per_lot)
            
            return min(max_lots_by_risk, self.max_lots, 2)  # Max 2 lots
        
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1
    
    def _is_market_open(self, current_time: time) -> bool:
        """Check if market is open."""
        return time(9, 15) <= current_time <= time(15, 30)
    
    def _is_opening_range_time(self, current_time: time) -> bool:
        """Check if current time is within opening range period."""
        return self.opening_range_start <= current_time <= self.opening_range_end
    
    def reset_daily_state(self) -> None:
        """Reset strategy state for new trading day."""
        self.opening_high = None
        self.opening_low = None
        self.opening_range_set = False
        self.position_taken = False
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        
        logger.info("Daily state reset for Nifty Futures Breakout Strategy")
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information."""
        return {
            'name': self.name,
            'symbol': self.symbol,
            'instrument_type': self.instrument_type,
            'lot_size': self.lot_size,
            'max_lots': self.max_lots,
            'opening_range_set': self.opening_range_set,
            'opening_high': self.opening_high,
            'opening_low': self.opening_low,
            'position_taken': self.position_taken,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target': self.target
        }