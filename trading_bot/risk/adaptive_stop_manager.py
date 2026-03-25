"""
Adaptive Stop-Loss Manager
Implements intelligent stop-loss management with multiple strategies
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class StopLossLevel:
    """Stop-loss level with reasoning."""
    price: float
    type: str  # 'initial', 'trailing', 'time_based', 'volatility_adjusted'
    distance_pct: float
    reasoning: str
    activated_at: Optional[datetime] = None


@dataclass
class StopLossUpdate:
    """Result of stop-loss update calculation."""
    new_stop: float
    should_update: bool
    stop_type: str
    reasoning: str
    risk_reward_ratio: float


class AdaptiveStopManager:
    """
    Manages adaptive stop-losses with multiple strategies:
    1. Initial stop-loss (ATR, percentage, support/resistance)
    2. Trailing stop-loss (activated after profit threshold)
    3. Time-based stops (exit if no profit after X minutes)
    4. Volatility-adjusted stops (widen/tighten based on volatility)
    """
    
    def __init__(self, config: Dict):
        # Initial stop-loss config
        self.atr_multiplier = config.get('atr_multiplier', 1.5)
        self.percentage_stop = config.get('percentage_stop', 0.0075)  # 0.75%
        
        # Trailing stop config
        self.trailing_activation_ratio = config.get('trailing_activation_ratio', 1.0)  # After 1x risk in profit
        self.trailing_distance_pct = config.get('trailing_distance_pct', 0.004)  # 0.4%
        self.trailing_ratchet_pct = config.get('trailing_ratchet_pct', 0.005)  # 0.5% profit increments
        
        # Time-based config
        self.no_profit_timeout_minutes = config.get('no_profit_timeout_minutes', 30)
        self.afternoon_tighten_time = config.get('afternoon_tighten_time', '14:30')
        self.force_exit_time = config.get('force_exit_time', '15:20')
        
        # Volatility adjustment config
        self.high_volatility_multiplier = config.get('high_volatility_multiplier', 1.3)
        self.low_volatility_multiplier = config.get('low_volatility_multiplier', 0.8)
        self.volatility_threshold_high = config.get('volatility_threshold_high', 0.25)  # 25%
        self.volatility_threshold_low = config.get('volatility_threshold_low', 0.10)  # 10%
        
        logger.info("Adaptive Stop Manager initialized")
    
    def calculate_initial_stop(
        self,
        entry_price: float,
        direction: str,  # 'long' or 'short'
        atr: float,
        support_resistance: Optional[float] = None,
        current_volatility: float = 0.15
    ) -> StopLossLevel:
        """
        Calculate initial stop-loss using multiple methods and choose the best.
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            atr: Average True Range
            support_resistance: Key support (long) or resistance (short) level
            current_volatility: Current market volatility
            
        Returns:
            StopLossLevel with optimal initial stop
        """
        try:
            stops = []
            
            # Method 1: ATR-based stop
            atr_distance = atr * self.atr_multiplier
            if direction == 'long':
                atr_stop = entry_price - atr_distance
            else:
                atr_stop = entry_price + atr_distance
            
            atr_stop_pct = abs(atr_stop - entry_price) / entry_price
            stops.append({
                'price': atr_stop,
                'distance_pct': atr_stop_pct,
                'method': 'ATR',
                'reasoning': f"ATR-based: {self.atr_multiplier}x ATR = ₹{atr_distance:.2f}"
            })
            
            # Method 2: Percentage-based stop
            if direction == 'long':
                pct_stop = entry_price * (1 - self.percentage_stop)
            else:
                pct_stop = entry_price * (1 + self.percentage_stop)
            
            stops.append({
                'price': pct_stop,
                'distance_pct': self.percentage_stop,
                'method': 'Percentage',
                'reasoning': f"Percentage-based: {self.percentage_stop:.2%}"
            })
            
            # Method 3: Support/Resistance-based stop
            if support_resistance is not None:
                if direction == 'long':
                    sr_stop = support_resistance * 0.998  # Slightly below support
                else:
                    sr_stop = support_resistance * 1.002  # Slightly above resistance
                
                sr_stop_pct = abs(sr_stop - entry_price) / entry_price
                stops.append({
                    'price': sr_stop,
                    'distance_pct': sr_stop_pct,
                    'method': 'Support/Resistance',
                    'reasoning': f"S/R-based: ₹{support_resistance:.2f}"
                })
            
            # Choose the tightest stop that still allows breathing room
            # But not too tight (min 0.3%)
            valid_stops = [s for s in stops if s['distance_pct'] >= 0.003]
            
            if not valid_stops:
                valid_stops = stops  # Use all if none meet minimum
            
            # Choose tightest valid stop
            best_stop = min(valid_stops, key=lambda x: x['distance_pct'])
            
            # Adjust for volatility
            adjusted_stop = self._adjust_for_volatility(
                best_stop['price'],
                entry_price,
                direction,
                current_volatility
            )
            
            adjusted_distance_pct = abs(adjusted_stop - entry_price) / entry_price
            
            reasoning = f"{best_stop['method']}: {best_stop['reasoning']} | Vol adj: {current_volatility:.1%}"
            
            return StopLossLevel(
                price=adjusted_stop,
                type='initial',
                distance_pct=adjusted_distance_pct,
                reasoning=reasoning,
                activated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating initial stop: {e}")
            # Return safe default
            if direction == 'long':
                default_stop = entry_price * 0.99
            else:
                default_stop = entry_price * 1.01
            
            return StopLossLevel(
                price=default_stop,
                type='initial',
                distance_pct=0.01,
                reasoning="Default 1% stop (error in calculation)",
                activated_at=datetime.now()
            )
    
    def update_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        direction: str,
        initial_risk: float,
        atr: Optional[float] = None
    ) -> StopLossUpdate:
        """
        Update trailing stop-loss if conditions are met.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            current_stop: Current stop-loss price
            direction: 'long' or 'short'
            initial_risk: Initial risk amount (entry - initial_stop)
            atr: Current ATR (optional, for ATR-based trailing)
            
        Returns:
            StopLossUpdate with new stop if applicable
        """
        try:
            # Calculate current profit
            if direction == 'long':
                profit = current_price - entry_price
                profit_pct = profit / entry_price
            else:
                profit = entry_price - current_price
                profit_pct = profit / entry_price
            
            # Check if trailing should be activated
            profit_risk_ratio = profit / initial_risk if initial_risk > 0 else 0
            
            if profit_risk_ratio < self.trailing_activation_ratio:
                # Not enough profit yet to activate trailing
                return StopLossUpdate(
                    new_stop=current_stop,
                    should_update=False,
                    stop_type='initial',
                    reasoning=f"Trailing not activated (profit/risk: {profit_risk_ratio:.2f} < {self.trailing_activation_ratio})",
                    risk_reward_ratio=profit_risk_ratio
                )
            
            # Calculate trailing stop
            if atr is not None:
                # ATR-based trailing
                trail_distance = atr * 0.5  # 0.5x ATR for trailing
            else:
                # Percentage-based trailing
                trail_distance = current_price * self.trailing_distance_pct
            
            if direction == 'long':
                new_stop = current_price - trail_distance
                # Only move stop up, never down
                if new_stop > current_stop:
                    should_update = True
                    reasoning = f"Trailing activated: New stop ₹{new_stop:.2f} (was ₹{current_stop:.2f})"
                else:
                    new_stop = current_stop
                    should_update = False
                    reasoning = f"Trailing active but no update (price not high enough)"
            else:  # short
                new_stop = current_price + trail_distance
                # Only move stop down, never up
                if new_stop < current_stop:
                    should_update = True
                    reasoning = f"Trailing activated: New stop ₹{new_stop:.2f} (was ₹{current_stop:.2f})"
                else:
                    new_stop = current_stop
                    should_update = False
                    reasoning = f"Trailing active but no update (price not low enough)"
            
            return StopLossUpdate(
                new_stop=new_stop,
                should_update=should_update,
                stop_type='trailing',
                reasoning=reasoning,
                risk_reward_ratio=profit_risk_ratio
            )
            
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
            return StopLossUpdate(
                new_stop=current_stop,
                should_update=False,
                stop_type='error',
                reasoning=f"Error: {e}",
                risk_reward_ratio=0
            )

    
    def check_time_based_exit(
        self,
        entry_time: datetime,
        current_price: float,
        entry_price: float,
        direction: str,
        current_stop: float
    ) -> StopLossUpdate:
        """
        Check if time-based exit conditions are met.
        
        Args:
            entry_time: Time when position was entered
            current_price: Current market price
            entry_price: Entry price
            direction: 'long' or 'short'
            current_stop: Current stop-loss
            
        Returns:
            StopLossUpdate with time-based adjustments
        """
        try:
            current_time = datetime.now()
            time_in_position = (current_time - entry_time).total_seconds() / 60  # minutes
            
            # Calculate current profit
            if direction == 'long':
                profit_pct = (current_price - entry_price) / entry_price
            else:
                profit_pct = (entry_price - current_price) / entry_price
            
            # Check 1: No profit timeout
            if time_in_position >= self.no_profit_timeout_minutes and profit_pct <= 0:
                # Exit at market if no profit after timeout
                return StopLossUpdate(
                    new_stop=current_price,  # Exit at market
                    should_update=True,
                    stop_type='time_based_no_profit',
                    reasoning=f"No profit after {self.no_profit_timeout_minutes} minutes - exit at market",
                    risk_reward_ratio=0
                )
            
            # Check 2: Afternoon tightening (after 2:30 PM)
            afternoon_time = datetime.strptime(self.afternoon_tighten_time, '%H:%M').time()
            if current_time.time() >= afternoon_time:
                # Tighten stop to breakeven or better
                if direction == 'long':
                    tightened_stop = max(current_stop, entry_price)
                    if tightened_stop > current_stop:
                        return StopLossUpdate(
                            new_stop=tightened_stop,
                            should_update=True,
                            stop_type='time_based_afternoon',
                            reasoning=f"Afternoon tightening: Move stop to breakeven",
                            risk_reward_ratio=profit_pct / 0.01 if profit_pct > 0 else 0
                        )
                else:  # short
                    tightened_stop = min(current_stop, entry_price)
                    if tightened_stop < current_stop:
                        return StopLossUpdate(
                            new_stop=tightened_stop,
                            should_update=True,
                            stop_type='time_based_afternoon',
                            reasoning=f"Afternoon tightening: Move stop to breakeven",
                            risk_reward_ratio=profit_pct / 0.01 if profit_pct > 0 else 0
                        )
            
            # Check 3: Force exit time (3:20 PM)
            force_exit_time = datetime.strptime(self.force_exit_time, '%H:%M').time()
            if current_time.time() >= force_exit_time:
                return StopLossUpdate(
                    new_stop=current_price,  # Exit at market
                    should_update=True,
                    stop_type='time_based_force_exit',
                    reasoning=f"Force exit at {self.force_exit_time} - close all positions",
                    risk_reward_ratio=profit_pct / 0.01 if profit_pct > 0 else 0
                )
            
            # No time-based update needed
            return StopLossUpdate(
                new_stop=current_stop,
                should_update=False,
                stop_type='time_based_no_action',
                reasoning=f"Time-based: No action needed (in position {time_in_position:.0f}m)",
                risk_reward_ratio=profit_pct / 0.01 if profit_pct > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Error in time-based exit check: {e}")
            return StopLossUpdate(
                new_stop=current_stop,
                should_update=False,
                stop_type='error',
                reasoning=f"Error: {e}",
                risk_reward_ratio=0
            )
    
    def _adjust_for_volatility(
        self,
        stop_price: float,
        entry_price: float,
        direction: str,
        current_volatility: float
    ) -> float:
        """
        Adjust stop-loss based on current volatility.
        
        Args:
            stop_price: Calculated stop price
            entry_price: Entry price
            direction: 'long' or 'short'
            current_volatility: Current market volatility
            
        Returns:
            Adjusted stop price
        """
        try:
            # Determine volatility regime
            if current_volatility >= self.volatility_threshold_high:
                # High volatility - widen stops
                multiplier = self.high_volatility_multiplier
            elif current_volatility <= self.volatility_threshold_low:
                # Low volatility - tighten stops
                multiplier = self.low_volatility_multiplier
            else:
                # Normal volatility - no adjustment
                multiplier = 1.0
            
            # Calculate adjusted distance
            original_distance = abs(stop_price - entry_price)
            adjusted_distance = original_distance * multiplier
            
            # Apply adjusted distance
            if direction == 'long':
                adjusted_stop = entry_price - adjusted_distance
            else:
                adjusted_stop = entry_price + adjusted_distance
            
            return adjusted_stop
            
        except Exception as e:
            logger.error(f"Error adjusting for volatility: {e}")
            return stop_price  # Return original if error
    
    def calculate_breakeven_stop(
        self,
        entry_price: float,
        direction: str,
        commission_pct: float = 0.0006  # 0.06% round-trip
    ) -> float:
        """
        Calculate breakeven stop price (including commissions).
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            commission_pct: Total commission percentage
            
        Returns:
            Breakeven stop price
        """
        try:
            if direction == 'long':
                # Need to cover entry and exit commissions
                breakeven = entry_price * (1 + commission_pct)
            else:
                breakeven = entry_price * (1 - commission_pct)
            
            return breakeven
            
        except Exception as e:
            logger.error(f"Error calculating breakeven: {e}")
            return entry_price
    
    def should_move_to_breakeven(
        self,
        entry_price: float,
        current_price: float,
        initial_risk: float,
        direction: str,
        profit_threshold: float = 0.5  # Move to BE after 0.5x risk in profit
    ) -> Tuple[bool, float]:
        """
        Determine if stop should be moved to breakeven.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            initial_risk: Initial risk amount
            direction: 'long' or 'short'
            profit_threshold: Profit threshold as multiple of initial risk
            
        Returns:
            Tuple of (should_move, breakeven_price)
        """
        try:
            # Calculate current profit
            if direction == 'long':
                profit = current_price - entry_price
            else:
                profit = entry_price - current_price
            
            # Check if profit exceeds threshold
            if profit >= initial_risk * profit_threshold:
                breakeven = self.calculate_breakeven_stop(entry_price, direction)
                return True, breakeven
            
            return False, entry_price
            
        except Exception as e:
            logger.error(f"Error checking breakeven move: {e}")
            return False, entry_price
    
    def get_comprehensive_stop_update(
        self,
        entry_price: float,
        entry_time: datetime,
        current_price: float,
        current_stop: float,
        direction: str,
        initial_risk: float,
        atr: float,
        current_volatility: float,
        support_resistance: Optional[float] = None
    ) -> StopLossUpdate:
        """
        Comprehensive stop-loss update considering all factors.
        
        This is the main method to call for stop updates.
        It considers:
        1. Trailing stops
        2. Time-based exits
        3. Volatility adjustments
        4. Breakeven moves
        
        Args:
            entry_price: Entry price
            entry_time: Entry time
            current_price: Current price
            current_stop: Current stop-loss
            direction: 'long' or 'short'
            initial_risk: Initial risk amount
            atr: Current ATR
            current_volatility: Current volatility
            support_resistance: Key S/R level
            
        Returns:
            StopLossUpdate with best recommendation
        """
        try:
            updates = []
            
            # Check 1: Trailing stop
            trailing_update = self.update_trailing_stop(
                entry_price, current_price, current_stop,
                direction, initial_risk, atr
            )
            if trailing_update.should_update:
                updates.append(trailing_update)
            
            # Check 2: Time-based exit
            time_update = self.check_time_based_exit(
                entry_time, current_price, entry_price,
                direction, current_stop
            )
            if time_update.should_update:
                updates.append(time_update)
            
            # Check 3: Breakeven move
            should_move_be, be_price = self.should_move_to_breakeven(
                entry_price, current_price, initial_risk, direction
            )
            if should_move_be:
                if direction == 'long' and be_price > current_stop:
                    updates.append(StopLossUpdate(
                        new_stop=be_price,
                        should_update=True,
                        stop_type='breakeven',
                        reasoning=f"Move to breakeven: ₹{be_price:.2f}",
                        risk_reward_ratio=0.5
                    ))
                elif direction == 'short' and be_price < current_stop:
                    updates.append(StopLossUpdate(
                        new_stop=be_price,
                        should_update=True,
                        stop_type='breakeven',
                        reasoning=f"Move to breakeven: ₹{be_price:.2f}",
                        risk_reward_ratio=0.5
                    ))
            
            # If no updates, return current stop
            if not updates:
                return StopLossUpdate(
                    new_stop=current_stop,
                    should_update=False,
                    stop_type='no_update',
                    reasoning="No stop update needed",
                    risk_reward_ratio=0
                )
            
            # Prioritize updates:
            # 1. Force exit (highest priority)
            # 2. No profit timeout
            # 3. Trailing stop
            # 4. Breakeven
            # 5. Afternoon tightening
            
            priority_order = [
                'time_based_force_exit',
                'time_based_no_profit',
                'trailing',
                'breakeven',
                'time_based_afternoon'
            ]
            
            for priority_type in priority_order:
                for update in updates:
                    if update.stop_type == priority_type:
                        return update
            
            # Return first update if no priority match
            return updates[0]
            
        except Exception as e:
            logger.error(f"Error in comprehensive stop update: {e}")
            return StopLossUpdate(
                new_stop=current_stop,
                should_update=False,
                stop_type='error',
                reasoning=f"Error: {e}",
                risk_reward_ratio=0
            )
