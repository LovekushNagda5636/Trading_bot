"""
Position Sizer - Calculates optimal position sizes based on risk parameters.
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENTAGE = "fixed_percentage"
    RISK_PERCENTAGE = "risk_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


class PositionSizer:
    """
    Calculates optimal position sizes based on risk management rules.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = PositionSizingMethod(config.get('method', 'risk_percentage'))
        self.max_position_risk = config.get('max_position_risk', 0.01)  # 1%
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)  # 2%
        self.fixed_amount = config.get('fixed_amount', 10000)  # ₹10,000
        self.fixed_percentage = config.get('fixed_percentage', 0.05)  # 5%
        
        logger.info(f"Position sizer initialized with method: {self.method.value}")
    
    def calculate_position_size(self, 
                              account_balance: float,
                              entry_price: float,
                              stop_loss: float,
                              symbol: str = "",
                              current_portfolio_risk: float = 0.0) -> Dict[str, Any]:
        """
        Calculate optimal position size.
        
        Args:
            account_balance: Available account balance
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            symbol: Trading symbol (for symbol-specific rules)
            current_portfolio_risk: Current portfolio risk exposure
            
        Returns:
            Dict with position size details
        """
        try:
            if self.method == PositionSizingMethod.FIXED_AMOUNT:
                return self._fixed_amount_sizing(entry_price)
            
            elif self.method == PositionSizingMethod.FIXED_PERCENTAGE:
                return self._fixed_percentage_sizing(account_balance, entry_price)
            
            elif self.method == PositionSizingMethod.RISK_PERCENTAGE:
                return self._risk_percentage_sizing(
                    account_balance, entry_price, stop_loss, current_portfolio_risk
                )
            
            elif self.method == PositionSizingMethod.VOLATILITY_ADJUSTED:
                return self._volatility_adjusted_sizing(
                    account_balance, entry_price, stop_loss, symbol
                )
            
            else:
                # Default to risk percentage
                return self._risk_percentage_sizing(
                    account_balance, entry_price, stop_loss, current_portfolio_risk
                )
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'quantity': 0,
                'position_value': 0,
                'risk_amount': 0,
                'risk_percentage': 0,
                'method': self.method.value,
                'error': str(e)
            }
    
    def _fixed_amount_sizing(self, entry_price: float) -> Dict[str, Any]:
        """Calculate position size using fixed amount method."""
        quantity = int(self.fixed_amount / entry_price)
        position_value = quantity * entry_price
        
        return {
            'quantity': quantity,
            'position_value': position_value,
            'risk_amount': 0,  # Not applicable for fixed amount
            'risk_percentage': 0,
            'method': self.method.value
        }
    
    def _fixed_percentage_sizing(self, account_balance: float, entry_price: float) -> Dict[str, Any]:
        """Calculate position size using fixed percentage method."""
        position_value = account_balance * self.fixed_percentage
        quantity = int(position_value / entry_price)
        actual_position_value = quantity * entry_price
        
        return {
            'quantity': quantity,
            'position_value': actual_position_value,
            'risk_amount': 0,  # Not applicable for fixed percentage
            'risk_percentage': actual_position_value / account_balance,
            'method': self.method.value
        }
    
    def _risk_percentage_sizing(self, 
                               account_balance: float,
                               entry_price: float,
                               stop_loss: float,
                               current_portfolio_risk: float) -> Dict[str, Any]:
        """Calculate position size using risk percentage method."""
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return {
                'quantity': 0,
                'position_value': 0,
                'risk_amount': 0,
                'risk_percentage': 0,
                'method': self.method.value,
                'error': 'Invalid stop loss price'
            }
        
        # Calculate maximum risk amount
        max_position_risk_amount = account_balance * self.max_position_risk
        
        # Check portfolio risk limits
        remaining_portfolio_risk = self.max_portfolio_risk - current_portfolio_risk
        if remaining_portfolio_risk <= 0:
            return {
                'quantity': 0,
                'position_value': 0,
                'risk_amount': 0,
                'risk_percentage': 0,
                'method': self.method.value,
                'error': 'Portfolio risk limit exceeded'
            }
        
        # Use the smaller of position risk or remaining portfolio risk
        max_risk_amount = min(
            max_position_risk_amount,
            account_balance * remaining_portfolio_risk
        )
        
        # Calculate quantity based on risk
        quantity = int(max_risk_amount / risk_per_share)
        
        if quantity <= 0:
            return {
                'quantity': 0,
                'position_value': 0,
                'risk_amount': 0,
                'risk_percentage': 0,
                'method': self.method.value,
                'error': 'Calculated quantity is zero'
            }
        
        position_value = quantity * entry_price
        actual_risk_amount = quantity * risk_per_share
        risk_percentage = actual_risk_amount / account_balance
        
        return {
            'quantity': quantity,
            'position_value': position_value,
            'risk_amount': actual_risk_amount,
            'risk_percentage': risk_percentage,
            'method': self.method.value,
            'risk_per_share': risk_per_share
        }
    
    def _volatility_adjusted_sizing(self,
                                   account_balance: float,
                                   entry_price: float,
                                   stop_loss: float,
                                   symbol: str) -> Dict[str, Any]:
        """Calculate position size using volatility-adjusted method."""
        # For now, use risk percentage method
        # TODO: Implement volatility calculation using historical data
        return self._risk_percentage_sizing(account_balance, entry_price, stop_loss, 0.0)
    
    def validate_position_size(self,
                              quantity: int,
                              entry_price: float,
                              account_balance: float,
                              current_positions: int = 0) -> Dict[str, Any]:
        """
        Validate if a position size is acceptable.
        
        Args:
            quantity: Proposed quantity
            entry_price: Entry price
            account_balance: Available balance
            current_positions: Number of current positions
            
        Returns:
            Dict with validation result
        """
        position_value = quantity * entry_price
        
        # Check minimum position size
        min_position_value = self.config.get('min_position_value', 1000)
        if position_value < min_position_value:
            return {
                'valid': False,
                'reason': f'Position value ₹{position_value:,.2f} below minimum ₹{min_position_value:,.2f}'
            }
        
        # Check maximum position size
        max_position_percentage = self.config.get('max_position_percentage', 0.1)  # 10%
        if position_value > account_balance * max_position_percentage:
            return {
                'valid': False,
                'reason': f'Position exceeds {max_position_percentage*100}% of account balance'
            }
        
        # Check maximum number of positions
        max_positions = self.config.get('max_positions', 5)
        if current_positions >= max_positions:
            return {
                'valid': False,
                'reason': f'Maximum positions limit ({max_positions}) reached'
            }
        
        # Check available margin
        required_margin = position_value * 0.2  # Assuming 20% margin
        if required_margin > account_balance:
            return {
                'valid': False,
                'reason': 'Insufficient margin available'
            }
        
        return {
            'valid': True,
            'position_value': position_value,
            'required_margin': required_margin,
            'position_percentage': position_value / account_balance
        }
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of position sizing configuration."""
        return {
            'method': self.method.value,
            'max_position_risk': self.max_position_risk,
            'max_portfolio_risk': self.max_portfolio_risk,
            'fixed_amount': self.fixed_amount,
            'fixed_percentage': self.fixed_percentage,
            'min_position_value': self.config.get('min_position_value', 1000),
            'max_position_percentage': self.config.get('max_position_percentage', 0.1),
            'max_positions': self.config.get('max_positions', 5)
        }