"""
Test script for Enhanced Risk Management components
Demonstrates the new Kelly Criterion position sizer and Adaptive Stop Manager
"""

import sys
sys.path.append('..')

from trading_bot.risk.enhanced_position_sizer import EnhancedPositionSizer
from trading_bot.risk.adaptive_stop_manager import AdaptiveStopManager
from datetime import datetime

def test_position_sizer():
    """Test Enhanced Position Sizer with Kelly Criterion."""
    print("=" * 60)
    print("TESTING ENHANCED POSITION SIZER")
    print("=" * 60)
    
    config = {
        'max_position_pct': 0.20,
        'min_position_pct': 0.02,
        'kelly_fraction': 0.25,
        'max_risk_per_trade': 0.02
    }
    
    sizer = EnhancedPositionSizer(config)
    
    # Test Case 1: High confidence trending bull market
    print("\nTest 1: High Confidence Trending Bull")
    result = sizer.calculate_position_size(
        capital=25000,
        entry_price=1500,
        stop_loss=1485,  # 1% stop
        win_probability=0.65,
        avg_win_loss_ratio=1.8,
        confidence=0.85,
        regime='trending_bull',
        max_positions=3
    )
    
    print(f"  Quantity: {result.quantity} shares")
    print(f"  Position Value: ₹{result.position_value:.2f}")
    print(f"  Risk Amount: ₹{result.risk_amount:.2f}")
    print(f"  Final Size: {result.final_size_pct:.1%} of capital")
    print(f"  Reasoning: {result.reasoning}")
    
    # Test Case 2: Low confidence sideways market
    print("\nTest 2: Low Confidence Sideways Market")
    result = sizer.calculate_position_size(
        capital=25000,
        entry_price=2450,
        stop_loss=2430,
        win_probability=0.52,
        avg_win_loss_ratio=1.3,
        confidence=0.60,
        regime='sideways',
        max_positions=5
    )
    
    print(f"  Quantity: {result.quantity} shares")
    print(f"  Position Value: ₹{result.position_value:.2f}")
    print(f"  Risk Amount: ₹{result.risk_amount:.2f}")
    print(f"  Final Size: {result.final_size_pct:.1%} of capital")
    print(f"  Reasoning: {result.reasoning}")
    
    # Test Case 3: Options position
    print("\nTest 3: Options Position Sizing")
    result = sizer.calculate_for_options(
        capital=25000,
        option_premium=45.50,
        lot_size=25,
        win_probability=0.60,
        avg_win_loss_ratio=2.0,
        confidence=0.75,
        regime='trending_bull'
    )
    
    print(f"  Quantity: {result.quantity} options")
    print(f"  Position Value: ₹{result.position_value:.2f}")
    print(f"  Risk Amount: ₹{result.risk_amount:.2f}")
    print(f"  Reasoning: {result.reasoning}")


def test_stop_manager():
    """Test Adaptive Stop Manager."""
    print("\n" + "=" * 60)
    print("TESTING ADAPTIVE STOP MANAGER")
    print("=" * 60)
    
    config = {
        'atr_multiplier': 1.5,
        'percentage_stop': 0.0075,
        'trailing_activation_ratio': 1.0,
        'trailing_distance_pct': 0.004,
        'no_profit_timeout_minutes': 30,
        'afternoon_tighten_time': '14:30',
        'force_exit_time': '15:20'
    }
    
    manager = AdaptiveStopManager(config)
    
    # Test Case 1: Initial stop calculation
    print("\nTest 1: Initial Stop-Loss Calculation")
    stop = manager.calculate_initial_stop(
        entry_price=1500,
        direction='long',
        atr=15,
        support_resistance=1480,
        current_volatility=0.18
    )
    
    print(f"  Stop Price: ₹{stop.price:.2f}")
    print(f"  Stop Type: {stop.type}")
    print(f"  Distance: {stop.distance_pct:.2%}")
    print(f"  Reasoning: {stop.reasoning}")
    
    # Test Case 2: Trailing stop update (in profit)
    print("\nTest 2: Trailing Stop Update (In Profit)")
    update = manager.update_trailing_stop(
        entry_price=1500,
        current_price=1530,  # +2% profit
        current_stop=1485,
        direction='long',
        initial_risk=15,
        atr=15
    )
    
    print(f"  Should Update: {update.should_update}")
    print(f"  New Stop: ₹{update.new_stop:.2f}")
    print(f"  Stop Type: {update.stop_type}")
    print(f"  Risk/Reward: {update.risk_reward_ratio:.2f}")
    print(f"  Reasoning: {update.reasoning}")
    
    # Test Case 3: Time-based exit check
    print("\nTest 3: Time-Based Exit Check")
    entry_time = datetime.now()
    update = manager.check_time_based_exit(
        entry_time=entry_time,
        current_price=1505,
        entry_price=1500,
        direction='long',
        current_stop=1485
    )
    
    print(f"  Should Update: {update.should_update}")
    print(f"  Stop Type: {update.stop_type}")
    print(f"  Reasoning: {update.reasoning}")
    
    # Test Case 4: Comprehensive stop update
    print("\nTest 4: Comprehensive Stop Update")
    update = manager.get_comprehensive_stop_update(
        entry_price=1500,
        entry_time=entry_time,
        current_price=1525,
        current_stop=1485,
        direction='long',
        initial_risk=15,
        atr=15,
        current_volatility=0.18
    )
    
    print(f"  Should Update: {update.should_update}")
    print(f"  New Stop: ₹{update.new_stop:.2f}")
    print(f"  Stop Type: {update.stop_type}")
    print(f"  Reasoning: {update.reasoning}")


if __name__ == "__main__":
    test_position_sizer()
    test_stop_manager()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
