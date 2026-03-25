"""
Risk Manager - Core risk management and position sizing.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..core.models import Position, Trade, Signal
from ..execution.broker_interface import BrokerOrder
from .enhanced_position_sizer import EnhancedPositionSizer, PositionSizeResult
from .adaptive_stop_manager import AdaptiveStopManager, StopLossLevel, StopLossUpdate

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per day
    max_position_risk: float = 0.01   # 1% max risk per position
    max_positions: int = 5            # Maximum number of positions
    max_daily_loss: float = 0.05      # 5% max daily loss
    max_drawdown: float = 0.10        # 10% max drawdown
    max_correlation: float = 0.7      # Maximum correlation between positions
    min_account_balance: float = 50000  # Minimum account balance


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    current_portfolio_risk: float = 0.0
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    open_positions: int = 0
    used_margin: float = 0.0
    available_margin: float = 0.0


class RiskManager:
    """
    Core risk management system.
    Handles position sizing, risk limits, and portfolio risk controls.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_limits = RiskLimits(**config.get('risk_limits', {}))
        self.current_metrics = RiskMetrics()
        
        # Initialize enhanced components
        self.position_sizer = EnhancedPositionSizer(config.get('position_sizing', {}))
        self.stop_manager = AdaptiveStopManager(config.get('stop_loss', {}))
        
        # Risk tracking
        self.daily_trades = []
        self.position_history = []
        self.equity_curve = []
        self.peak_equity = 0.0
        
        # Emergency stop flag
        self.emergency_stop = False
        
        # Track win/loss statistics for Kelly Criterion
        self.win_count = 0
        self.loss_count = 0
        self.total_wins = 0.0
        self.total_losses = 0.0
        
        logger.info("Risk Manager initialized with Enhanced Position Sizer and Adaptive Stop Manager")
    
    def can_place_order(self, signal: Signal, account_balance: float, 
                       current_positions: List[Position]) -> Tuple[bool, str, int]:
        """
        Determine if an order can be placed based on risk limits.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            current_positions: List of current positions
            
        Returns:
            Tuple of (can_place, reason, suggested_quantity)
        """
        # Check emergency stop
        if self.emergency_stop:
            return False, "Emergency stop activated", 0
        
        # Check account balance
        if account_balance < self.risk_limits.min_account_balance:
            return False, f"Account balance below minimum: {account_balance}", 0
        
        # Check maximum positions
        if len(current_positions) >= self.risk_limits.max_positions:
            return False, f"Maximum positions reached: {len(current_positions)}", 0
        
        # Check daily loss limit
        if self.current_metrics.daily_pnl < -account_balance * self.risk_limits.max_daily_loss:
            return False, "Daily loss limit exceeded", 0
        
        # Check drawdown limit
        if self.current_metrics.current_drawdown > self.risk_limits.max_drawdown:
            return False, "Maximum drawdown exceeded", 0
        
        # Calculate position size
        position_size = self.calculate_position_size(
            signal, account_balance, current_positions
        )
        
        if position_size <= 0:
            return False, "Position size calculation resulted in zero or negative size", 0
        
        # Check portfolio risk
        position_risk = self.calculate_position_risk(signal, position_size, account_balance)
        
        if position_risk > self.risk_limits.max_position_risk:
            return False, f"Position risk too high: {position_risk:.2%}", 0
        
        # Check correlation with existing positions
        if self.check_correlation_risk(signal.symbol, current_positions):
            return False, "High correlation with existing positions", 0
        
        return True, "Risk checks passed", position_size
    
    def calculate_position_size(self, signal: Signal, account_balance: float, 
                              current_positions: List[Position],
                              regime: str = 'sideways',
                              confidence: float = 0.7,
                              atr: Optional[float] = None) -> int:
        """
        Calculate optimal position size using Enhanced Kelly Criterion.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            current_positions: List of current positions
            regime: Current market regime
            confidence: Model confidence (0-1)
            atr: Average True Range
            
        Returns:
            Position size in shares
        """
        try:
            # Calculate win probability and win/loss ratio from historical data
            win_probability = self._calculate_win_probability()
            avg_win_loss_ratio = self._calculate_win_loss_ratio()
            
            # Use enhanced position sizer
            result = self.position_sizer.calculate_position_size(
                capital=account_balance,
                entry_price=signal.price,
                stop_loss=signal.stop_loss if signal.stop_loss else signal.price * 0.99,
                win_probability=win_probability,
                avg_win_loss_ratio=avg_win_loss_ratio,
                confidence=confidence,
                regime=regime,
                max_positions=self.risk_limits.max_positions
            )
            
            logger.info(f"Position size calculated: {result.quantity} shares | {result.reasoning}")
            
            return result.quantity
            
        except Exception as e:
            logger.error(f"Error in enhanced position sizing: {e}")
            # Fallback to simple calculation
            return self._simple_position_size(signal, account_balance)
    
    def _simple_position_size(self, signal: Signal, account_balance: float) -> int:
        """Fallback simple position sizing."""
        risk_amount = account_balance * self.risk_limits.max_position_risk
        
        if signal.stop_loss and signal.price:
            stop_distance = abs(signal.price - signal.stop_loss)
            if stop_distance > 0:
                return int(risk_amount / stop_distance)
        
        # Default: 2% of account
        return int((account_balance * 0.02) / signal.price)
    
    def _calculate_win_probability(self) -> float:
        """Calculate win probability from historical trades."""
        total_trades = self.win_count + self.loss_count
        if total_trades == 0:
            return 0.55  # Default 55% win rate
        
        return self.win_count / total_trades
    
    def _calculate_win_loss_ratio(self) -> float:
        """Calculate average win/loss ratio."""
        if self.loss_count == 0 or self.total_losses == 0:
            return 1.5  # Default 1.5:1 ratio
        
        avg_win = self.total_wins / self.win_count if self.win_count > 0 else 0
        avg_loss = abs(self.total_losses / self.loss_count)
        
        if avg_loss == 0:
            return 1.5
        
        return avg_win / avg_loss
    
    def calculate_position_risk(self, signal: Signal, quantity: int, 
                              account_balance: float) -> float:
        """
        Calculate risk percentage for a position.
        
        Args:
            signal: Trading signal
            quantity: Position quantity
            account_balance: Current account balance
            
        Returns:
            Risk as percentage of account balance
        """
        if not signal.stop_loss or quantity <= 0:
            return 0.0
        
        # Calculate potential loss
        stop_distance = abs(signal.price - signal.stop_loss)
        potential_loss = stop_distance * quantity
        
        # Risk as percentage of account
        risk_pct = potential_loss / account_balance
        
        return risk_pct
    
    def check_correlation_risk(self, symbol: str, current_positions: List[Position]) -> bool:
        """
        Check if new position would create excessive correlation risk.
        
        Args:
            symbol: Symbol to check
            current_positions: Current positions
            
        Returns:
            True if correlation risk is too high
        """
        # TODO: Implement proper correlation calculation
        # This would require historical price data and correlation matrix
        
        # Simple sector/industry check for now
        sector_exposure = {}
        
        for position in current_positions:
            # Extract sector from symbol (simplified)
            sector = self._get_sector(position.symbol)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + 1
        
        new_sector = self._get_sector(symbol)
        
        # Check if adding this position would exceed sector concentration
        if sector_exposure.get(new_sector, 0) >= 2:  # Max 2 positions per sector
            return True
        
        return False
    
    def _get_sector(self, symbol: str) -> str:
        """
        Get sector for a symbol (simplified implementation).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Sector name
        """
        # TODO: Implement proper sector mapping
        # This should use a database or API to get actual sector information
        
        # Simplified sector mapping based on symbol patterns
        if any(bank in symbol.upper() for bank in ['BANK', 'HDFC', 'ICICI', 'AXIS', 'SBI']):
            return 'BANKING'
        elif any(it in symbol.upper() for it in ['TCS', 'INFY', 'WIPRO', 'TECH']):
            return 'IT'
        elif any(pharma in symbol.upper() for pharma in ['PHARMA', 'SUN', 'CIPLA', 'REDDY']):
            return 'PHARMA'
        else:
            return 'OTHER'
    
    def update_metrics(self, account_balance: float, positions: List[Position], 
                      daily_pnl: float) -> None:
        """
        Update current risk metrics.
        
        Args:
            account_balance: Current account balance
            positions: Current positions
            daily_pnl: Daily P&L
        """
        self.current_metrics.daily_pnl = daily_pnl
        self.current_metrics.open_positions = len(positions)
        
        # Calculate portfolio risk
        total_risk = 0.0
        for position in positions:
            # Estimate position risk (simplified)
            position_value = abs(position.quantity * position.average_price)
            position_risk = position_value * 0.02  # Assume 2% risk per position
            total_risk += position_risk
        
        self.current_metrics.current_portfolio_risk = total_risk / account_balance
        
        # Update equity curve and drawdown
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': account_balance,
            'pnl': daily_pnl
        })
        
        # Calculate drawdown
        if account_balance > self.peak_equity:
            self.peak_equity = account_balance
        
        self.current_metrics.current_drawdown = (self.peak_equity - account_balance) / self.peak_equity
        
        # Check for emergency stop conditions
        self._check_emergency_conditions(account_balance)
    
    def _check_emergency_conditions(self, account_balance: float) -> None:
        """
        Check for emergency stop conditions.
        
        Args:
            account_balance: Current account balance
        """
        # Emergency stop conditions
        emergency_conditions = [
            self.current_metrics.daily_pnl < -account_balance * 0.1,  # 10% daily loss
            self.current_metrics.current_drawdown > 0.15,  # 15% drawdown
            account_balance < self.risk_limits.min_account_balance * 0.8  # 80% of min balance
        ]
        
        if any(emergency_conditions):
            self.emergency_stop = True
            logger.critical("EMERGENCY STOP ACTIVATED - Risk limits breached")
    
    def get_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report.
        
        Returns:
            Dict containing risk metrics and analysis
        """
        return {
            'current_metrics': {
                'portfolio_risk': f"{self.current_metrics.current_portfolio_risk:.2%}",
                'daily_pnl': self.current_metrics.daily_pnl,
                'drawdown': f"{self.current_metrics.current_drawdown:.2%}",
                'open_positions': self.current_metrics.open_positions,
                'emergency_stop': self.emergency_stop
            },
            'risk_limits': {
                'max_portfolio_risk': f"{self.risk_limits.max_portfolio_risk:.2%}",
                'max_position_risk': f"{self.risk_limits.max_position_risk:.2%}",
                'max_positions': self.risk_limits.max_positions,
                'max_daily_loss': f"{self.risk_limits.max_daily_loss:.2%}",
                'max_drawdown': f"{self.risk_limits.max_drawdown:.2%}"
            },
            'utilization': {
                'portfolio_risk_used': f"{(self.current_metrics.current_portfolio_risk / self.risk_limits.max_portfolio_risk) * 100:.1f}%",
                'positions_used': f"{(self.current_metrics.open_positions / self.risk_limits.max_positions) * 100:.1f}%",
                'daily_loss_used': f"{abs(self.current_metrics.daily_pnl) / (self.peak_equity * self.risk_limits.max_daily_loss) * 100:.1f}%" if self.peak_equity > 0 else "0%"
            }
        }
    
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics at market open."""
        self.current_metrics.daily_pnl = 0.0
        self.daily_trades = []
        
        # Reset emergency stop if conditions have improved
        if not any([
            self.current_metrics.current_drawdown > self.risk_limits.max_drawdown,
            self.current_metrics.current_portfolio_risk > self.risk_limits.max_portfolio_risk
        ]):
            self.emergency_stop = False
            logger.info("Emergency stop reset - conditions improved")
    
    def add_trade(self, trade: Trade) -> None:
        """
        Add completed trade to tracking and update statistics.
        
        Args:
            trade: Completed trade
        """
        self.daily_trades.append(trade)
        
        # Update daily P&L
        self.current_metrics.daily_pnl += trade.pnl
        
        # Update win/loss statistics for Kelly Criterion
        if trade.pnl > 0:
            self.win_count += 1
            self.total_wins += trade.pnl
        else:
            self.loss_count += 1
            self.total_losses += trade.pnl
        
        logger.info(f"Trade added: {trade.symbol} P&L: {trade.pnl} | Win rate: {self._calculate_win_probability():.1%}")
    
    def calculate_initial_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        support_resistance: Optional[float] = None,
        volatility: float = 0.15
    ) -> StopLossLevel:
        """
        Calculate initial stop-loss using adaptive stop manager.
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            atr: Average True Range
            support_resistance: Key S/R level
            volatility: Current volatility
            
        Returns:
            StopLossLevel with optimal stop
        """
        return self.stop_manager.calculate_initial_stop(
            entry_price=entry_price,
            direction=direction,
            atr=atr,
            support_resistance=support_resistance,
            current_volatility=volatility
        )
    
    def update_stop_loss(
        self,
        position: Position,
        current_price: float,
        atr: float,
        volatility: float
    ) -> Optional[StopLossUpdate]:
        """
        Update stop-loss for an existing position.
        
        Args:
            position: Current position
            current_price: Current market price
            atr: Current ATR
            volatility: Current volatility
            
        Returns:
            StopLossUpdate if stop should be updated
        """
        try:
            # Determine direction
            direction = 'long' if position.quantity > 0 else 'short'
            
            # Calculate initial risk
            initial_risk = abs(position.average_price - position.stop_loss) if hasattr(position, 'stop_loss') else position.average_price * 0.01
            
            # Get comprehensive stop update
            update = self.stop_manager.get_comprehensive_stop_update(
                entry_price=position.average_price,
                entry_time=position.entry_time if hasattr(position, 'entry_time') else datetime.now(),
                current_price=current_price,
                current_stop=position.stop_loss if hasattr(position, 'stop_loss') else position.average_price * 0.99,
                direction=direction,
                initial_risk=initial_risk,
                atr=atr,
                current_volatility=volatility
            )
            
            if update.should_update:
                logger.info(f"Stop-loss update for {position.symbol}: {update.reasoning}")
                return update
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating stop-loss: {e}")
            return None
    
    def get_position_limits(self, symbol: str) -> Dict[str, int]:
        """
        Get position limits for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with position limits
        """
        # TODO: Implement symbol-specific limits
        # This could include sector limits, volatility-based limits, etc.
        
        return {
            'max_quantity': 1000,  # Default max quantity
            'max_value': 100000,   # Default max value (₹1 lakh)
            'max_risk': 0.01       # Default max risk (1%)
        }