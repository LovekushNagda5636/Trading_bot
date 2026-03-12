"""
Portfolio Manager - Manages overall portfolio risk and allocation.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    total_positions: int
    winning_positions: int
    losing_positions: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    exposure: float


class PortfolioManager:
    """
    Manages overall portfolio risk, allocation, and performance tracking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)  # 2%
        self.max_positions = config.get('max_positions', 5)
        self.max_sector_exposure = config.get('max_sector_exposure', 0.3)  # 30%
        self.max_single_position = config.get('max_single_position', 0.1)  # 10%
        
        # Portfolio tracking
        self.positions = {}
        self.closed_trades = []
        self.daily_pnl_history = []
        self.portfolio_value_history = []
        
        # Risk metrics
        self.current_exposure = 0.0
        self.current_risk = 0.0
        self.max_drawdown_seen = 0.0
        self.peak_portfolio_value = 0.0
        
        logger.info("Portfolio manager initialized")
    
    def update_portfolio(self, positions: List[Any], account_balance: float) -> None:
        """Update portfolio with current positions and balance."""
        try:
            # Update positions
            self.positions = {pos.symbol: pos for pos in positions}
            
            # Calculate portfolio metrics
            total_position_value = sum(
                abs(pos.quantity * pos.current_price) for pos in positions
            )
            
            total_pnl = sum(pos.unrealized_pnl for pos in positions)
            
            # Update exposure
            self.current_exposure = total_position_value / account_balance if account_balance > 0 else 0
            
            # Update portfolio value history
            current_portfolio_value = account_balance + total_pnl
            self.portfolio_value_history.append({
                'timestamp': datetime.now(),
                'value': current_portfolio_value,
                'pnl': total_pnl
            })
            
            # Keep only recent history (last 1000 entries)
            if len(self.portfolio_value_history) > 1000:
                self.portfolio_value_history = self.portfolio_value_history[-500:]
            
            # Update peak and drawdown
            if current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_portfolio_value
            
            current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
            if current_drawdown > self.max_drawdown_seen:
                self.max_drawdown_seen = current_drawdown
            
            logger.debug(f"Portfolio updated: {len(positions)} positions, exposure: {self.current_exposure:.2%}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
    
    def check_position_limits(self, 
                             symbol: str, 
                             position_value: float, 
                             account_balance: float) -> Dict[str, Any]:
        """Check if new position violates portfolio limits."""
        try:
            # Check maximum positions
            if len(self.positions) >= self.max_positions:
                return {
                    'allowed': False,
                    'reason': f'Maximum positions limit ({self.max_positions}) reached'
                }
            
            # Check single position limit
            position_percentage = position_value / account_balance
            if position_percentage > self.max_single_position:
                return {
                    'allowed': False,
                    'reason': f'Position exceeds {self.max_single_position:.1%} limit'
                }
            
            # Check total exposure limit
            current_total_value = sum(
                abs(pos.quantity * pos.current_price) for pos in self.positions.values()
            )
            new_total_exposure = (current_total_value + position_value) / account_balance
            
            if new_total_exposure > self.max_portfolio_risk:
                return {
                    'allowed': False,
                    'reason': f'Portfolio exposure would exceed {self.max_portfolio_risk:.1%} limit'
                }
            
            # Check sector exposure (simplified - assumes symbol prefix indicates sector)
            sector = self._get_sector(symbol)
            sector_exposure = self._calculate_sector_exposure(sector, position_value, account_balance)
            
            if sector_exposure > self.max_sector_exposure:
                return {
                    'allowed': False,
                    'reason': f'Sector exposure would exceed {self.max_sector_exposure:.1%} limit'
                }
            
            return {
                'allowed': True,
                'position_percentage': position_percentage,
                'new_total_exposure': new_total_exposure,
                'sector_exposure': sector_exposure
            }
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return {
                'allowed': False,
                'reason': f'Error checking limits: {str(e)}'
            }
    
    def calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk."""
        try:
            total_risk = 0.0
            
            for position in self.positions.values():
                # Calculate position risk based on stop loss
                if hasattr(position, 'stop_loss') and position.stop_loss > 0:
                    if position.quantity > 0:  # Long position
                        risk_per_share = max(0, position.current_price - position.stop_loss)
                    else:  # Short position
                        risk_per_share = max(0, position.stop_loss - position.current_price)
                    
                    position_risk = abs(position.quantity) * risk_per_share
                    total_risk += position_risk
            
            return total_risk
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return 0.0
    
    def get_portfolio_metrics(self, account_balance: float) -> PortfolioMetrics:
        """Get comprehensive portfolio metrics."""
        try:
            # Calculate basic metrics
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            total_pnl = total_unrealized_pnl + total_realized_pnl
            
            # Calculate position statistics
            winning_positions = len([pos for pos in self.positions.values() if pos.unrealized_pnl > 0])
            losing_positions = len([pos for pos in self.positions.values() if pos.unrealized_pnl < 0])
            total_positions = len(self.positions)
            
            win_rate = winning_positions / total_positions if total_positions > 0 else 0
            
            # Calculate daily P&L
            daily_pnl = self._calculate_daily_pnl()
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Calculate total portfolio value
            total_position_value = sum(
                abs(pos.quantity * pos.current_price) for pos in self.positions.values()
            )
            
            return PortfolioMetrics(
                total_value=account_balance + total_pnl,
                total_pnl=total_pnl,
                unrealized_pnl=total_unrealized_pnl,
                realized_pnl=total_realized_pnl,
                daily_pnl=daily_pnl,
                total_positions=total_positions,
                winning_positions=winning_positions,
                losing_positions=losing_positions,
                win_rate=win_rate,
                max_drawdown=self.max_drawdown_seen,
                sharpe_ratio=sharpe_ratio,
                exposure=self.current_exposure
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return PortfolioMetrics(
                total_value=account_balance,
                total_pnl=0, unrealized_pnl=0, realized_pnl=0, daily_pnl=0,
                total_positions=0, winning_positions=0, losing_positions=0,
                win_rate=0, max_drawdown=0, sharpe_ratio=0, exposure=0
            )
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol (simplified implementation)."""
        # Simplified sector mapping based on symbol
        sector_map = {
            'RELIANCE': 'Energy',
            'TCS': 'IT',
            'INFY': 'IT',
            'HDFC': 'Banking',
            'HDFCBANK': 'Banking',
            'ICICIBANK': 'Banking',
            'SBIN': 'Banking',
            'ITC': 'FMCG',
            'BHARTIARTL': 'Telecom',
            'LT': 'Infrastructure'
        }
        
        return sector_map.get(symbol, 'Other')
    
    def _calculate_sector_exposure(self, sector: str, new_position_value: float, account_balance: float) -> float:
        """Calculate sector exposure including new position."""
        current_sector_value = sum(
            abs(pos.quantity * pos.current_price) 
            for pos in self.positions.values()
            if self._get_sector(pos.symbol) == sector
        )
        
        total_sector_value = current_sector_value + new_position_value
        return total_sector_value / account_balance
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L."""
        today = datetime.now().date()
        
        # Get today's P&L from history
        today_entries = [
            entry for entry in self.portfolio_value_history
            if entry['timestamp'].date() == today
        ]
        
        if len(today_entries) >= 2:
            return today_entries[-1]['pnl'] - today_entries[0]['pnl']
        
        return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (simplified)."""
        if len(self.portfolio_value_history) < 30:  # Need at least 30 data points
            return 0.0
        
        try:
            # Calculate daily returns
            returns = []
            for i in range(1, len(self.portfolio_value_history)):
                prev_value = self.portfolio_value_history[i-1]['value']
                curr_value = self.portfolio_value_history[i]['value']
                
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            
            if not returns:
                return 0.0
            
            # Calculate mean and std
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_return = variance ** 0.5
            
            # Sharpe ratio (assuming risk-free rate = 0)
            if std_return > 0:
                return mean_return / std_return * (252 ** 0.5)  # Annualized
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get portfolio risk summary."""
        return {
            'current_exposure': self.current_exposure,
            'max_exposure_limit': self.max_portfolio_risk,
            'current_positions': len(self.positions),
            'max_positions_limit': self.max_positions,
            'current_risk': self.calculate_portfolio_risk(),
            'max_drawdown': self.max_drawdown_seen,
            'peak_portfolio_value': self.peak_portfolio_value,
            'risk_utilization': self.current_exposure / self.max_portfolio_risk
        }