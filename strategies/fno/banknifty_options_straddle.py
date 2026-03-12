"""
Bank Nifty Options Straddle Strategy for Intraday F&O Trading
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


class BankNiftyOptionsStraddle(BaseStrategy):
    """
    Bank Nifty Options Straddle Strategy for intraday F&O trading.
    
    Logic:
    1. Sell ATM Call and Put options (Short Straddle)
    2. Collect premium when IV is high
    3. Manage delta neutrality
    4. Exit before expiry or on stop loss
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Strategy parameters
        self.symbol = config.get('symbol', 'BANKNIFTY')
        self.instrument_type = config.get('instrument_type', 'OPT')
        self.expiry = config.get('expiry', '2025-01-29')
        self.lot_size = config.get('lot_size', 15)
        self.max_lots = config.get('max_lots', 1)
        
        # Options parameters
        self.strategy_type = config.get('strategy_type', 'short_straddle')
        self.delta_neutral = config.get('delta_neutral', True)
        self.iv_threshold = config.get('iv_threshold', 20)  # Minimum IV to enter
        self.max_loss_pct = config.get('max_loss_pct', 0.5)  # 50% of premium
        
        # Trading session parameters
        self.entry_time_start = time(10, 0)   # Enter after 10 AM
        self.entry_time_end = time(14, 0)     # Last entry at 2 PM
        self.square_off_time = time(15, 20)
        
        # State variables
        self.position_taken = False
        self.call_strike = None
        self.put_strike = None
        self.call_premium = None
        self.put_premium = None
        self.total_premium_received = None
        self.atm_strike = None
        
        # Technical indicators
        self.indicators = TechnicalIndicators()
        
        logger.info(f"Initialized Bank Nifty Options Straddle Strategy")
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on options straddle logic."""
        if data.empty:
            return []
        
        signals = []
        current_time = datetime.now().time()
        latest = data.iloc[-1]
        
        # Check if market is open
        if not self._is_market_open(current_time):
            return []
        
        # Check for entry signals
        if not self.position_taken and self._is_entry_time(current_time):
            entry_signals = self._check_entry_signal(data, latest)
            if entry_signals:
                signals.extend(entry_signals)
                self.position_taken = True
        
        # Check for exit signals
        elif self.position_taken:
            exit_signals = self._check_exit_signals(data, latest, current_time)
            if exit_signals:
                signals.extend(exit_signals)
                self.position_taken = False
        
        return signals
    
    def _check_entry_signal(self, data: pd.DataFrame, latest: pd.Series) -> List[Signal]:
        """Check for straddle entry signals."""
        try:
            current_price = latest['close']
            
            # Calculate ATM strike (nearest 100 for Bank Nifty)
            self.atm_strike = round(current_price / 100) * 100
            
            # Check if IV is favorable (this would need options data)
            # For now, we'll use volatility as a proxy
            volatility = self._calculate_volatility(data)
            
            if volatility < self.iv_threshold:  # Low volatility = good for selling options
                return []
            
            # Generate short straddle signals
            signals = []
            
            # Short Call
            call_symbol = f"{self.symbol}{self.expiry.replace('-', '')[2:]}{self.atm_strike}CE"
            call_signal = Signal(
                symbol=call_symbol,
                signal_type=SignalType.SELL,  # Short Call
                price=0,  # Will be filled by options pricing
                quantity=self.lot_size * self.max_lots,
                confidence=0.7,
                strategy=self.name,
                metadata={
                    'option_type': 'CE',
                    'strike': self.atm_strike,
                    'strategy': 'short_straddle',
                    'leg': 'call',
                    'volatility': volatility
                }
            )
            signals.append(call_signal)
            
            # Short Put
            put_symbol = f"{self.symbol}{self.expiry.replace('-', '')[2:]}{self.atm_strike}PE"
            put_signal = Signal(
                symbol=put_symbol,
                signal_type=SignalType.SELL,  # Short Put
                price=0,  # Will be filled by options pricing
                quantity=self.lot_size * self.max_lots,
                confidence=0.7,
                strategy=self.name,
                metadata={
                    'option_type': 'PE',
                    'strike': self.atm_strike,
                    'strategy': 'short_straddle',
                    'leg': 'put',
                    'volatility': volatility
                }
            )
            signals.append(put_signal)
            
            self.call_strike = self.atm_strike
            self.put_strike = self.atm_strike
            
            logger.info(f"Generated short straddle signals at strike {self.atm_strike}")
            return signals
        
        except Exception as e:
            logger.error(f"Error checking entry signal: {e}")
            return []
    
    def _check_exit_signals(self, data: pd.DataFrame, latest: pd.Series, current_time: time) -> List[Signal]:
        """Check for exit signals."""
        try:
            signals = []
            
            # Time-based exit (square off before 3:20 PM)
            if current_time >= self.square_off_time:
                # Close Call position
                call_symbol = f"{self.symbol}{self.expiry.replace('-', '')[2:]}{self.call_strike}CE"
                call_exit = Signal(
                    symbol=call_symbol,
                    signal_type=SignalType.BUY,  # Buy to close short
                    price=0,
                    quantity=self.lot_size * self.max_lots,
                    confidence=1.0,
                    strategy=self.name,
                    metadata={'exit_reason': 'time_based_square_off', 'leg': 'call'}
                )
                signals.append(call_exit)
                
                # Close Put position
                put_symbol = f"{self.symbol}{self.expiry.replace('-', '')[2:]}{self.put_strike}PE"
                put_exit = Signal(
                    symbol=put_symbol,
                    signal_type=SignalType.BUY,  # Buy to close short
                    price=0,
                    quantity=self.lot_size * self.max_lots,
                    confidence=1.0,
                    strategy=self.name,
                    metadata={'exit_reason': 'time_based_square_off', 'leg': 'put'}
                )
                signals.append(put_exit)
                
                return signals
            
            # P&L based exit (this would need real options pricing)
            # For now, we'll use underlying movement as proxy
            current_price = latest['close']
            
            # If underlying moves too much, close the position
            if self.atm_strike:
                move_pct = abs(current_price - self.atm_strike) / self.atm_strike
                
                if move_pct > 0.03:  # 3% move - close position
                    # Close both legs
                    call_symbol = f"{self.symbol}{self.expiry.replace('-', '')[2:]}{self.call_strike}CE"
                    call_exit = Signal(
                        symbol=call_symbol,
                        signal_type=SignalType.BUY,
                        price=0,
                        quantity=self.lot_size * self.max_lots,
                        confidence=0.9,
                        strategy=self.name,
                        metadata={'exit_reason': 'stop_loss', 'leg': 'call'}
                    )
                    signals.append(call_exit)
                    
                    put_symbol = f"{self.symbol}{self.expiry.replace('-', '')[2:]}{self.put_strike}PE"
                    put_exit = Signal(
                        symbol=put_symbol,
                        signal_type=SignalType.BUY,
                        price=0,
                        quantity=self.lot_size * self.max_lots,
                        confidence=0.9,
                        strategy=self.name,
                        metadata={'exit_reason': 'stop_loss', 'leg': 'put'}
                    )
                    signals.append(put_exit)
            
            return signals
        
        except Exception as e:
            logger.error(f"Error checking exit signals: {e}")
            return []
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate historical volatility as proxy for IV."""
        try:
            if len(data) < 20:
                return 0
            
            # Calculate 20-day historical volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            return volatility
        
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0
    
    def _is_market_open(self, current_time: time) -> bool:
        """Check if market is open."""
        return time(9, 15) <= current_time <= time(15, 30)
    
    def _is_entry_time(self, current_time: time) -> bool:
        """Check if current time is within entry window."""
        return self.entry_time_start <= current_time <= self.entry_time_end
    
    def reset_daily_state(self) -> None:
        """Reset strategy state for new trading day."""
        self.position_taken = False
        self.call_strike = None
        self.put_strike = None
        self.call_premium = None
        self.put_premium = None
        self.total_premium_received = None
        self.atm_strike = None
        
        logger.info("Daily state reset for Bank Nifty Options Straddle Strategy")
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information."""
        return {
            'name': self.name,
            'symbol': self.symbol,
            'strategy_type': self.strategy_type,
            'lot_size': self.lot_size,
            'max_lots': self.max_lots,
            'position_taken': self.position_taken,
            'atm_strike': self.atm_strike,
            'call_strike': self.call_strike,
            'put_strike': self.put_strike,
            'total_premium_received': self.total_premium_received
        }