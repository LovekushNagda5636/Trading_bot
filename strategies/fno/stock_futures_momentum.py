"""
Stock Futures Momentum Strategy for Intraday F&O Trading
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


class StockFuturesMomentum(BaseStrategy):
    """
    Stock Futures Momentum Strategy for intraday F&O trading.
    
    Logic:
    1. Use RSI + MACD for momentum confirmation
    2. Trade stock futures with high liquidity
    3. Use futures premium/discount for entry timing
    4. Strict risk management with ATR-based stops
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Strategy parameters
        self.symbol = config.get('symbol', 'RELIANCE')
        self.instrument_type = config.get('instrument_type', 'FUT')
        self.expiry = config.get('expiry', '2025-01-30')
        self.lot_size = config.get('lot_size', 250)
        self.max_lots = config.get('max_lots', 1)
        
        # Technical parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        self.volume_threshold = config.get('volume_threshold', 1.2)
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        
        # Trading session parameters
        self.square_off_time = time(15, 20)
        
        # State variables
        self.position_taken = False
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.position_type = None  # 'long' or 'short'
        
        # Technical indicators
        self.indicators = TechnicalIndicators()
        
        logger.info(f"Initialized Stock Futures Momentum Strategy for {self.symbol}")
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on momentum logic."""
        if data.empty or len(data) < max(self.rsi_period, self.macd_slow):
            return []
        
        signals = []
        current_time = datetime.now().time()
        latest = data.iloc[-1]
        
        # Check if market is open
        if not self._is_market_open(current_time):
            return []
        
        # Check for entry signals
        if not self.position_taken:
            signal = self._check_entry_signal(data, latest)
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
    
    def _check_entry_signal(self, data: pd.DataFrame, latest: pd.Series) -> Optional[Signal]:
        """Check for momentum entry signals."""
        try:
            current_price = latest['close']
            volume = latest['volume']
            
            # Calculate technical indicators
            rsi = self.indicators.rsi(data['close'], period=self.rsi_period).iloc[-1]
            macd_line, macd_signal, macd_histogram = self.indicators.macd(
                data['close'], 
                fast=self.macd_fast, 
                slow=self.macd_slow, 
                signal=self.macd_signal
            )
            
            macd_current = macd_line.iloc[-1]
            macd_signal_current = macd_signal.iloc[-1]
            macd_prev = macd_line.iloc[-2] if len(macd_line) > 1 else macd_current
            macd_signal_prev = macd_signal.iloc[-2] if len(macd_signal) > 1 else macd_signal_current
            
            # Calculate ATR for stop loss
            atr = self.indicators.atr(data, period=self.atr_period).iloc[-1]
            
            # Volume confirmation
            avg_volume = data['volume'].tail(20).mean()
            volume_confirmed = volume > (avg_volume * self.volume_threshold)
            
            # Check for bullish momentum
            bullish_rsi = rsi > 50 and rsi < self.rsi_overbought
            bullish_macd = (macd_current > macd_signal_current and 
                           macd_prev <= macd_signal_prev)  # MACD crossover
            
            if bullish_rsi and bullish_macd and volume_confirmed:
                # Calculate position size
                quantity = self.lot_size * self.max_lots
                
                # Set stop loss and target
                self.entry_price = current_price
                self.stop_loss = current_price - (atr * self.atr_multiplier)
                self.target = current_price + (atr * self.atr_multiplier * 1.5)  # 1.5:1 R:R
                self.position_type = 'long'
                
                return Signal(
                    symbol=f"{self.symbol}{self.expiry.replace('-', '')[2:]}FUT",
                    signal_type=SignalType.BUY,
                    price=current_price,
                    quantity=quantity,
                    stop_loss=self.stop_loss,
                    target=self.target,
                    confidence=0.75,
                    strategy=self.name,
                    metadata={
                        'momentum_type': 'bullish',
                        'rsi': rsi,
                        'macd': macd_current,
                        'macd_signal': macd_signal_current,
                        'atr': atr,
                        'volume_ratio': volume / avg_volume,
                        'lot_size': self.lot_size,
                        'lots': self.max_lots
                    }
                )
            
            # Check for bearish momentum
            bearish_rsi = rsi < 50 and rsi > self.rsi_oversold
            bearish_macd = (macd_current < macd_signal_current and 
                           macd_prev >= macd_signal_prev)  # MACD crossover down
            
            if bearish_rsi and bearish_macd and volume_confirmed:
                # Calculate position size
                quantity = self.lot_size * self.max_lots
                
                # Set stop loss and target
                self.entry_price = current_price
                self.stop_loss = current_price + (atr * self.atr_multiplier)
                self.target = current_price - (atr * self.atr_multiplier * 1.5)  # 1.5:1 R:R
                self.position_type = 'short'
                
                return Signal(
                    symbol=f"{self.symbol}{self.expiry.replace('-', '')[2:]}FUT",
                    signal_type=SignalType.SELL,
                    price=current_price,
                    quantity=quantity,
                    stop_loss=self.stop_loss,
                    target=self.target,
                    confidence=0.75,
                    strategy=self.name,
                    metadata={
                        'momentum_type': 'bearish',
                        'rsi': rsi,
                        'macd': macd_current,
                        'macd_signal': macd_signal_current,
                        'atr': atr,
                        'volume_ratio': volume / avg_volume,
                        'lot_size': self.lot_size,
                        'lots': self.max_lots
                    }
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking entry signal: {e}")
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
            if self.stop_loss and self.position_type:
                if ((self.position_type == 'long' and current_price <= self.stop_loss) or
                    (self.position_type == 'short' and current_price >= self.stop_loss)):
                    
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
            if self.target and self.position_type:
                if ((self.position_type == 'long' and current_price >= self.target) or
                    (self.position_type == 'short' and current_price <= self.target)):
                    
                    return Signal(
                        symbol=f"{self.symbol}{self.expiry.replace('-', '')[2:]}FUT",
                        signal_type=SignalType.EXIT,
                        price=current_price,
                        quantity=0,
                        confidence=1.0,
                        strategy=self.name,
                        metadata={'exit_reason': 'target_achieved'}
                    )
            
            # Momentum reversal exit
            rsi = self.indicators.rsi(data['close'], period=self.rsi_period).iloc[-1]
            
            if self.position_type == 'long' and rsi > self.rsi_overbought:
                return Signal(
                    symbol=f"{self.symbol}{self.expiry.replace('-', '')[2:]}FUT",
                    signal_type=SignalType.EXIT,
                    price=current_price,
                    quantity=0,
                    confidence=0.8,
                    strategy=self.name,
                    metadata={'exit_reason': 'momentum_reversal', 'rsi': rsi}
                )
            
            elif self.position_type == 'short' and rsi < self.rsi_oversold:
                return Signal(
                    symbol=f"{self.symbol}{self.expiry.replace('-', '')[2:]}FUT",
                    signal_type=SignalType.EXIT,
                    price=current_price,
                    quantity=0,
                    confidence=0.8,
                    strategy=self.name,
                    metadata={'exit_reason': 'momentum_reversal', 'rsi': rsi}
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Error checking exit signal: {e}")
            return None
    
    def _is_market_open(self, current_time: time) -> bool:
        """Check if market is open."""
        return time(9, 15) <= current_time <= time(15, 30)
    
    def reset_daily_state(self) -> None:
        """Reset strategy state for new trading day."""
        self.position_taken = False
        self.entry_price = None
        self.stop_loss = None
        self.target = None
        self.position_type = None
        
        logger.info(f"Daily state reset for {self.symbol} Futures Momentum Strategy")
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information."""
        return {
            'name': self.name,
            'symbol': self.symbol,
            'instrument_type': self.instrument_type,
            'lot_size': self.lot_size,
            'max_lots': self.max_lots,
            'position_taken': self.position_taken,
            'position_type': self.position_type,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target': self.target
        }