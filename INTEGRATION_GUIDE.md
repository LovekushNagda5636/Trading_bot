# Integration Guide: Enhanced Risk Management
## How to integrate the new components into your trading bot

---

## Quick Start

The new risk management components are already integrated into `RiskManager`. You just need to update your configuration and use the enhanced methods.

---

## Step 1: Update Configuration

Add the new configuration sections to your `config/angel_one_config.json` or `config/trading_development.json`:

```json
{
  "risk_limits": {
    "max_portfolio_risk": 0.02,
    "max_position_risk": 0.01,
    "max_positions": 5,
    "max_daily_loss": 0.05,
    "max_drawdown": 0.10,
    "max_correlation": 0.7,
    "min_account_balance": 50000
  },
  "position_sizing": {
    "max_position_pct": 0.20,
    "min_position_pct": 0.02,
    "kelly_fraction": 0.25,
    "max_risk_per_trade": 0.02
  },
  "stop_loss": {
    "atr_multiplier": 1.5,
    "percentage_stop": 0.0075,
    "trailing_activation_ratio": 1.0,
    "trailing_distance_pct": 0.004,
    "trailing_ratchet_pct": 0.005,
    "no_profit_timeout_minutes": 30,
    "afternoon_tighten_time": "14:30",
    "force_exit_time": "15:20",
    "high_volatility_multiplier": 1.3,
    "low_volatility_multiplier": 0.8,
    "volatility_threshold_high": 0.25,
    "volatility_threshold_low": 0.10
  }
}
```

---

## Step 2: Update Trading Engine

Modify your `trading_bot/engine/trading_engine.py` to use the enhanced risk management:

### A. Position Sizing

**Before**:
```python
# Old way
quantity = risk_manager.calculate_position_size(
    signal, account_balance, current_positions
)
```

**After**:
```python
# New way with regime and confidence
quantity = risk_manager.calculate_position_size(
    signal=signal,
    account_balance=account_balance,
    current_positions=current_positions,
    regime=self.detect_market_regime(),  # 'trending_bull', 'sideways', etc.
    confidence=signal.confidence if hasattr(signal, 'confidence') else 0.7,
    atr=self.calculate_atr(signal.symbol)
)
```

### B. Initial Stop-Loss

**Before**:
```python
# Old way - manual calculation
stop_loss = entry_price * 0.99  # Fixed 1% stop
```

**After**:
```python
# New way - adaptive stop
stop_level = risk_manager.calculate_initial_stop_loss(
    entry_price=entry_price,
    direction='long' if signal.direction == 'BUY' else 'short',
    atr=self.calculate_atr(signal.symbol),
    support_resistance=self.find_nearest_support(signal.symbol),
    volatility=self.calculate_volatility(signal.symbol)
)

stop_loss = stop_level.price
logger.info(f"Initial stop: {stop_level.reasoning}")
```

### C. Stop-Loss Updates (in position monitoring loop)

**Add this to your position monitoring**:
```python
# Update stops for all open positions
for position in open_positions:
    current_price = self.get_current_price(position.symbol)
    atr = self.calculate_atr(position.symbol)
    volatility = self.calculate_volatility(position.symbol)
    
    # Check if stop should be updated
    update = risk_manager.update_stop_loss(
        position=position,
        current_price=current_price,
        atr=atr,
        volatility=volatility
    )
    
    if update and update.should_update:
        logger.info(f"Updating stop for {position.symbol}: {update.reasoning}")
        
        # Update the stop-loss order
        self.broker.modify_stop_loss(
            position_id=position.id,
            new_stop=update.new_stop
        )
        
        # Update position object
        position.stop_loss = update.new_stop
        
        # Check if it's a force exit
        if update.stop_type in ['time_based_force_exit', 'time_based_no_profit']:
            logger.warning(f"Force exit triggered for {position.symbol}")
            self.close_position(position, reason=update.reasoning)
```

---

## Step 3: Add Helper Methods

Add these helper methods to your trading engine:

```python
def detect_market_regime(self) -> str:
    """
    Detect current market regime.
    Returns: 'trending_bull', 'trending_bear', 'sideways', 
             'high_volatility', 'low_volatility'
    """
    # Simple implementation - enhance as needed
    nifty_data = self.get_recent_data('NIFTY50', periods=50)
    
    # Calculate trend
    sma_20 = nifty_data['close'].rolling(20).mean().iloc[-1]
    sma_50 = nifty_data['close'].rolling(50).mean().iloc[-1]
    current_price = nifty_data['close'].iloc[-1]
    
    # Calculate volatility
    returns = nifty_data['close'].pct_change()
    volatility = returns.std() * np.sqrt(252)  # Annualized
    
    # Determine regime
    if volatility > 0.25:
        return 'high_volatility'
    elif volatility < 0.10:
        return 'low_volatility'
    elif current_price > sma_20 > sma_50:
        return 'trending_bull'
    elif current_price < sma_20 < sma_50:
        return 'trending_bear'
    else:
        return 'sideways'

def calculate_atr(self, symbol: str, period: int = 14) -> float:
    """Calculate Average True Range."""
    data = self.get_recent_data(symbol, periods=period + 1)
    
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    
    return atr

def calculate_volatility(self, symbol: str, period: int = 20) -> float:
    """Calculate historical volatility."""
    data = self.get_recent_data(symbol, periods=period + 1)
    returns = data['close'].pct_change()
    volatility = returns.std() * np.sqrt(252)  # Annualized
    
    return volatility

def find_nearest_support(self, symbol: str) -> Optional[float]:
    """Find nearest support level."""
    data = self.get_recent_data(symbol, periods=50)
    
    # Simple implementation: recent swing lows
    lows = data['low'].rolling(5, center=True).min()
    swing_lows = data['low'][data['low'] == lows]
    
    current_price = data['close'].iloc[-1]
    supports = swing_lows[swing_lows < current_price]
    
    if len(supports) > 0:
        return supports.iloc[-1]  # Most recent support
    
    return None
```

---

## Step 4: Update Signal Generation

Enhance your signals to include confidence scores:

```python
class Signal:
    def __init__(self, ...):
        # ... existing fields ...
        self.confidence = 0.7  # Add confidence field
        
    def calculate_confidence(self, indicators: Dict) -> float:
        """
        Calculate signal confidence based on multiple factors.
        Returns: 0.0 to 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Multiple indicator agreement
        if indicators.get('rsi_signal') == indicators.get('macd_signal'):
            confidence += 0.1
        
        # Factor 2: Volume confirmation
        if indicators.get('volume_ratio', 1.0) > 1.5:
            confidence += 0.1
        
        # Factor 3: Trend alignment
        if indicators.get('trend_aligned'):
            confidence += 0.15
        
        # Factor 4: Support/resistance proximity
        if indicators.get('near_key_level'):
            confidence += 0.1
        
        # Factor 5: Time of day (avoid first/last 15 min)
        current_time = datetime.now().time()
        if time(9, 30) <= current_time <= time(15, 15):
            confidence += 0.05
        
        return min(confidence, 1.0)  # Cap at 1.0
```

---

## Step 5: Monitor and Log

Add comprehensive logging to track the new risk management:

```python
# After position entry
logger.info(f"""
Position Opened:
  Symbol: {position.symbol}
  Direction: {position.direction}
  Quantity: {position.quantity}
  Entry: ₹{position.entry_price:.2f}
  Stop: ₹{position.stop_loss:.2f} ({stop_level.reasoning})
  Position Size: {position_size_result.final_size_pct:.1%} of capital
  Kelly Fraction: {position_size_result.kelly_fraction:.2%}
  Regime: {current_regime}
  Confidence: {signal.confidence:.0%}
""")

# During position monitoring
logger.info(f"""
Position Update:
  Symbol: {position.symbol}
  Current: ₹{current_price:.2f}
  P&L: ₹{position.unrealized_pnl:.2f} ({position.unrealized_pnl_pct:.2%})
  Stop: ₹{position.stop_loss:.2f}
  Time in Position: {time_in_position} minutes
  Stop Update: {update.reasoning if update else 'No update'}
""")
```

---

## Step 6: Backtesting Integration

Update your backtest engine to use the new risk management:

```python
# In backtest_engine.py

def run_backtest(self, ...):
    # ... existing code ...
    
    # Use enhanced position sizing
    quantity = self.risk_manager.calculate_position_size(
        signal=signal,
        account_balance=self.current_capital,
        current_positions=self.positions,
        regime=self.detect_regime(current_date),
        confidence=signal.confidence,
        atr=self.calculate_atr(signal.symbol, current_date)
    )
    
    # Use adaptive stops
    stop_level = self.risk_manager.calculate_initial_stop_loss(
        entry_price=entry_price,
        direction='long' if signal.direction == 'BUY' else 'short',
        atr=self.calculate_atr(signal.symbol, current_date),
        volatility=self.calculate_volatility(signal.symbol, current_date)
    )
    
    # Simulate stop updates during holding period
    for bar in holding_period:
        update = self.risk_manager.update_stop_loss(
            position=position,
            current_price=bar['close'],
            atr=self.calculate_atr(signal.symbol, bar['date']),
            volatility=self.calculate_volatility(signal.symbol, bar['date'])
        )
        
        if update and update.should_update:
            position.stop_loss = update.new_stop
            
            # Check if stop hit
            if (position.direction == 'long' and bar['low'] <= position.stop_loss) or \
               (position.direction == 'short' and bar['high'] >= position.stop_loss):
                self.close_position(position, bar['date'], position.stop_loss, 'stop_loss')
                break
```

---

## Step 7: Testing

Test the integration thoroughly:

```bash
# 1. Run the test script
cd examples
python test_risk_management.py

# 2. Run a backtest with the new risk management
python -m trading_bot.backtesting.backtest_engine --start 2024-01-01 --end 2024-12-31

# 3. Paper trade for a day
python continuous_trading_bot.py --mode paper

# 4. Monitor logs for risk management decisions
tail -f logs/trading_bot.log | grep -E "(Position|Stop|Kelly|Regime)"
```

---

## Common Issues and Solutions

### Issue 1: ATR calculation returns NaN
**Solution**: Ensure you have enough historical data (at least 15 bars)

```python
def calculate_atr(self, symbol: str, period: int = 14) -> float:
    data = self.get_recent_data(symbol, periods=period + 1)
    
    if len(data) < period + 1:
        logger.warning(f"Insufficient data for ATR calculation: {len(data)} bars")
        # Return default ATR based on price
        return data['close'].iloc[-1] * 0.01  # 1% of price
    
    # ... rest of calculation ...
```

### Issue 2: Regime detection is too volatile
**Solution**: Add smoothing and hysteresis

```python
def detect_market_regime(self) -> str:
    current_regime = self._calculate_regime()
    
    # Only change regime if it persists for 3 bars
    if not hasattr(self, '_regime_history'):
        self._regime_history = []
    
    self._regime_history.append(current_regime)
    self._regime_history = self._regime_history[-3:]  # Keep last 3
    
    if len(self._regime_history) == 3 and len(set(self._regime_history)) == 1:
        return current_regime
    
    # Return previous regime if not confirmed
    return getattr(self, '_confirmed_regime', 'sideways')
```

### Issue 3: Position sizes are too small
**Solution**: Adjust Kelly fraction or check win rate data

```python
# Increase Kelly fraction for more aggressive sizing
config['position_sizing']['kelly_fraction'] = 0.35  # Use 35% of Kelly

# Or check if win rate is being calculated correctly
logger.info(f"Win rate: {risk_manager._calculate_win_probability():.1%}")
logger.info(f"Win/Loss ratio: {risk_manager._calculate_win_loss_ratio():.2f}")
```

---

## Performance Monitoring

Track these metrics to evaluate the new risk management:

```python
# Daily summary
daily_metrics = {
    'total_trades': len(completed_trades),
    'win_rate': risk_manager._calculate_win_probability(),
    'avg_win_loss_ratio': risk_manager._calculate_win_loss_ratio(),
    'avg_position_size_pct': np.mean([t.position_size_pct for t in completed_trades]),
    'stops_hit': len([t for t in completed_trades if t.exit_reason == 'stop_loss']),
    'trailing_stops_activated': len([t for t in completed_trades if t.trailing_activated]),
    'time_based_exits': len([t for t in completed_trades if 'time' in t.exit_reason]),
    'avg_holding_time': np.mean([t.holding_time_minutes for t in completed_trades])
}

logger.info(f"Daily Risk Management Summary: {daily_metrics}")
```

---

## Next Steps

1. Run paper trading for 1 week to validate
2. Monitor position sizing decisions
3. Track stop-loss effectiveness
4. Adjust configuration based on results
5. Proceed to Phase 2 (ML enhancements)

---

## Support

If you encounter issues:
1. Check logs for error messages
2. Verify configuration is correct
3. Ensure all dependencies are installed
4. Test individual components using `examples/test_risk_management.py`

---

**Remember**: The new risk management is more sophisticated but requires proper configuration and monitoring. Start with conservative settings and adjust based on results.
