"""
Enhanced Position Sizer with Kelly Criterion and Regime Awareness
Implements advanced position sizing for optimal capital allocation
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    quantity: int
    position_value: float
    risk_amount: float
    kelly_fraction: float
    confidence_adjusted: float
    regime_adjusted: float
    final_size_pct: float
    reasoning: str


class EnhancedPositionSizer:
    """
    Advanced position sizing using Enhanced Kelly Criterion.
    Adjusts for confidence, regime, and risk constraints.
    """
    
    def __init__(self, config: Dict):
        self.max_position_pct = config.get('max_position_pct', 0.20)  # 20% max
        self.min_position_pct = config.get('min_position_pct', 0.02)  # 2% min
        self.kelly_fraction = config.get('kelly_fraction', 0.25)  # Use 25% of Kelly
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2% risk
        
        # Regime adjustments
        self.regime_multipliers = {
            'trending_bull': 1.5,
            'trending_bear': 1.2,
            'sideways': 0.8,
            'high_volatility': 0.6,
            'low_volatility': 1.0
        }
        
        logger.info("Enhanced Position Sizer initialized")
    
    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        win_probability: float,
        avg_win_loss_ratio: float,
        confidence: float = 0.7,
        regime: str = 'sideways',
        max_positions: int = 5
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using Enhanced Kelly Criterion.
        
        Args:
            capital: Available trading capital
            entry_price: Entry price for the trade
            stop_loss: Stop-loss price
            win_probability: Probability of winning (0-1)
            avg_win_loss_ratio: Average win/loss ratio
            confidence: Model confidence (0-1)
            regime: Current market regime
            max_positions: Maximum number of concurrent positions
            
        Returns:
            PositionSizeResult with calculated position size
        """
        try:
            # Step 1: Calculate Kelly Criterion
            kelly_pct = self._calculate_kelly(
                win_probability, avg_win_loss_ratio
            )
            
            # Step 2: Apply confidence adjustment
            confidence_adjusted = kelly_pct * confidence
            
            # Step 3: Apply regime adjustment
            regime_multiplier = self.regime_multipliers.get(regime, 1.0)
            regime_adjusted = confidence_adjusted * regime_multiplier
            
            # Step 4: Apply Kelly fraction (fractional Kelly)
            fractional_kelly = regime_adjusted * self.kelly_fraction
            
            # Step 5: Apply position limits
            final_size_pct = np.clip(
                fractional_kelly,
                self.min_position_pct,
                self.max_position_pct
            )
            
            # Step 6: Adjust for number of positions
            if max_positions > 1:
                # Reserve capital for other positions
                final_size_pct = min(final_size_pct, 0.8 / max_positions)
            
            # Step 7: Calculate risk-based position size
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share == 0:
                risk_per_share = entry_price * 0.01  # Default 1% risk
            
            max_risk_amount = capital * self.max_risk_per_trade
            risk_based_quantity = int(max_risk_amount / risk_per_share)
            
            # Step 8: Calculate Kelly-based position size
            position_value = capital * final_size_pct
            kelly_based_quantity = int(position_value / entry_price)
            
            # Step 9: Take the minimum of risk-based and Kelly-based
            final_quantity = min(risk_based_quantity, kelly_based_quantity)
            
            # Ensure minimum viable position
            if final_quantity < 1:
                final_quantity = 1
            
            # Calculate actual values
            actual_position_value = final_quantity * entry_price
            actual_risk_amount = final_quantity * risk_per_share
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                kelly_pct, confidence_adjusted, regime_adjusted,
                final_size_pct, regime, confidence
            )
            
            return PositionSizeResult(
                quantity=final_quantity,
                position_value=actual_position_value,
                risk_amount=actual_risk_amount,
                kelly_fraction=kelly_pct,
                confidence_adjusted=confidence_adjusted,
                regime_adjusted=regime_adjusted,
                final_size_pct=final_size_pct,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            # Return minimum safe position
            return PositionSizeResult(
                quantity=1,
                position_value=entry_price,
                risk_amount=abs(entry_price - stop_loss),
                kelly_fraction=0.02,
                confidence_adjusted=0.02,
                regime_adjusted=0.02,
                final_size_pct=0.02,
                reasoning="Error in calculation, using minimum position"
            )
    
    def _calculate_kelly(
        self,
        win_prob: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly Criterion percentage.
        
        Formula: f* = (p * b - q) / b
        Where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = win/loss ratio
        """
        try:
            # Ensure valid inputs
            win_prob = np.clip(win_prob, 0.01, 0.99)
            win_loss_ratio = max(win_loss_ratio, 0.1)
            
            loss_prob = 1 - win_prob
            
            # Kelly formula
            kelly = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
            
            # Kelly can be negative (don't trade) or very large
            kelly = np.clip(kelly, 0.0, 1.0)
            
            return kelly
            
        except Exception as e:
            logger.error(f"Error in Kelly calculation: {e}")
            return 0.02  # Default 2%
    
    def _generate_reasoning(
        self,
        kelly_pct: float,
        confidence_adj: float,
        regime_adj: float,
        final_pct: float,
        regime: str,
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for position size."""
        reasoning_parts = []
        
        reasoning_parts.append(f"Kelly: {kelly_pct:.1%}")
        reasoning_parts.append(f"Confidence adj: {confidence_adj:.1%} (conf={confidence:.0%})")
        reasoning_parts.append(f"Regime adj: {regime_adj:.1%} (regime={regime})")
        reasoning_parts.append(f"Final: {final_pct:.1%}")
        
        if final_pct == self.max_position_pct:
            reasoning_parts.append("(capped at max)")
        elif final_pct == self.min_position_pct:
            reasoning_parts.append("(at minimum)")
        
        return " | ".join(reasoning_parts)
    
    def calculate_for_options(
        self,
        capital: float,
        option_premium: float,
        lot_size: int,
        win_probability: float,
        avg_win_loss_ratio: float,
        confidence: float = 0.7,
        regime: str = 'sideways'
    ) -> PositionSizeResult:
        """
        Calculate position size for options trading.
        
        Args:
            capital: Available capital
            option_premium: Premium per option
            lot_size: Lot size for the option
            win_probability: Win probability
            avg_win_loss_ratio: Win/loss ratio
            confidence: Model confidence
            regime: Market regime
            
        Returns:
            PositionSizeResult for options
        """
        try:
            # Calculate Kelly
            kelly_pct = self._calculate_kelly(win_probability, avg_win_loss_ratio)
            
            # Apply adjustments
            confidence_adjusted = kelly_pct * confidence
            regime_multiplier = self.regime_multipliers.get(regime, 1.0)
            regime_adjusted = confidence_adjusted * regime_multiplier
            
            # Fractional Kelly
            fractional_kelly = regime_adjusted * self.kelly_fraction
            
            # Apply limits
            final_size_pct = np.clip(
                fractional_kelly,
                self.min_position_pct,
                self.max_position_pct
            )
            
            # Calculate number of lots
            position_value = capital * final_size_pct
            cost_per_lot = option_premium * lot_size
            
            if cost_per_lot == 0:
                num_lots = 0
            else:
                num_lots = int(position_value / cost_per_lot)
            
            # Ensure at least 1 lot if affordable
            if num_lots == 0 and cost_per_lot <= capital * self.max_position_pct:
                num_lots = 1
            
            # Calculate actual values
            actual_quantity = num_lots * lot_size
            actual_position_value = num_lots * cost_per_lot
            actual_risk_amount = actual_position_value  # Max loss is premium paid
            
            reasoning = f"Options: {num_lots} lots × {lot_size} = {actual_quantity} qty | Premium: ₹{option_premium:.2f} | Total: ₹{actual_position_value:.2f}"
            
            return PositionSizeResult(
                quantity=actual_quantity,
                position_value=actual_position_value,
                risk_amount=actual_risk_amount,
                kelly_fraction=kelly_pct,
                confidence_adjusted=confidence_adjusted,
                regime_adjusted=regime_adjusted,
                final_size_pct=final_size_pct,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error calculating options position size: {e}")
            return PositionSizeResult(
                quantity=0,
                position_value=0,
                risk_amount=0,
                kelly_fraction=0,
                confidence_adjusted=0,
                regime_adjusted=0,
                final_size_pct=0,
                reasoning=f"Error: {e}"
            )
    
    def adjust_for_correlation(
        self,
        base_size: PositionSizeResult,
        existing_positions: list,
        correlation: float
    ) -> PositionSizeResult:
        """
        Adjust position size based on correlation with existing positions.
        
        Args:
            base_size: Base position size calculation
            existing_positions: List of existing positions
            correlation: Correlation coefficient (-1 to 1)
            
        Returns:
            Adjusted PositionSizeResult
        """
        try:
            # If high positive correlation, reduce size
            if correlation > 0.7:
                adjustment_factor = 0.5  # Reduce by 50%
            elif correlation > 0.5:
                adjustment_factor = 0.7  # Reduce by 30%
            elif correlation < -0.5:
                # Negative correlation is good for diversification
                adjustment_factor = 1.2  # Increase by 20%
            else:
                adjustment_factor = 1.0  # No adjustment
            
            adjusted_quantity = int(base_size.quantity * adjustment_factor)
            adjusted_quantity = max(1, adjusted_quantity)  # At least 1
            
            # Recalculate values
            adjusted_position_value = adjusted_quantity * (base_size.position_value / base_size.quantity)
            adjusted_risk = adjusted_quantity * (base_size.risk_amount / base_size.quantity)
            
            reasoning = f"{base_size.reasoning} | Correlation adj: {adjustment_factor:.1%} (corr={correlation:.2f})"
            
            return PositionSizeResult(
                quantity=adjusted_quantity,
                position_value=adjusted_position_value,
                risk_amount=adjusted_risk,
                kelly_fraction=base_size.kelly_fraction,
                confidence_adjusted=base_size.confidence_adjusted,
                regime_adjusted=base_size.regime_adjusted,
                final_size_pct=base_size.final_size_pct * adjustment_factor,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error adjusting for correlation: {e}")
            return base_size
