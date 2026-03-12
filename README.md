# 🤖 Angel One Self-Learning Trading Bot v2.0

An AI-powered intraday trading bot that **learns from every trade** and gets smarter over time. Trades **Equity, Futures, and Options** on the Indian markets via Angel One SmartAPI.

## ✨ What's New in v2.0 — Self-Learning Engine

| Feature | Before | After |
|---------|--------|-------|
| Strategy scoring | Fixed weights | **Adaptive weights** learned from outcomes |
| Stop-loss | Fixed 1% | **Dynamic SL** based on volatility + learned multiplier |
| Targets | Fixed 1.5% / 3% | **Adaptive targets** + trailing stops |
| Position sizing | Fixed % of budget | **Kelly criterion** scaled by confidence |
| Market types | Equity only | **Equity + F&O Options + Futures** |
| Signal filtering | Score > 35 | **Learned optimal threshold** per regime |
| Entry timing | Any time | **Avoids historically bad hours** |
| Strategy selection | All equal | **Weights adjusted** by win rate per strategy |
| Risk management | Fixed daily loss | **Regime-aware** position limits |
| Learning | None | **End-of-day learning cycle** with mistake classification |

## 🧠 How Self-Learning Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADING BOT LOOP                              │
│                                                                  │
│  1. SCAN Market Data (Angel One API)                            │
│  2. DETECT Market Regime (trending/sideways/volatile)           │
│  3. GENERATE Signals (8 equity + 4 options + 3 futures strats)  │
│  4. ENHANCE Signals (apply learned strategy weights)            │
│  5. FILTER (learned min score + journal rules + regime limits)  │
│  6. EXECUTE (adaptive SL/target/qty based on confidence)        │
│  7. MONITOR (trailing stops, partial exits)                     │
│  8. RECORD (full context: regime, strategies, timing, outcome)  │
│  9. LEARN (end-of-day: update weights, discover combos)         │
│                                                                  │
│  The bot improves parameters after every session.               │
└─────────────────────────────────────────────────────────────────┘
```

### What It Learns From

- **Winning strategies**: Which strategies win in which market regime
- **Losing strategies**: Which strategies to avoid (weight → 0.1x)
- **Best hours**: Which hours of the day produce most wins
- **Worst hours**: Which hours to avoid trading
- **SL placement**: Too many tight SL hits → widen SL multiplier
- **Target placement**: Too many premature exits → raise targets
- **Score threshold**: Finds optimal signal score that maximizes win rate
- **Strategy combinations**: Discovers pairs of strategies that win together
- **Regime awareness**: Adjusts aggressiveness per market regime

### Mistake Classification

Every losing trade is classified:
| Mistake | Meaning | What Bot Learns |
|---------|---------|---------------------|
| `tight_sl` | SL hit but trade reversed | Widen SL for this regime |
| `wrong_direction` | Trade went opposite way | Downweight strategy in regime |
| `late_entry` | Entered after 2 PM | Tighten entry cutoff hour |
| `weak_signal` | Low score trade lost | Raise minimum score threshold |
| `low_volatility_entry` | No range for profit | Skip entries when range < 0.5% |
| `premature_exit` | Left money on table | Raise target multiplier |

## 📊 Markets Covered

### Equity (Intraday)
- **130+ F&O stocks** scanned every 5 minutes
- 8 strategies: Gap, Momentum, Reversal, Volume Surge, Breakout, VWAP, Prev Day, Narrow Range

### Index Options (NIFTY/BANKNIFTY)
- ATM Calls/Puts (directional plays)
- Long Straddles (volatility plays)
- Short Straddles (sideways plays)
- OTM Breakout plays

### Stock Futures
- Momentum futures signals
- Breakout/breakdown futures
- Sector rotation plays

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure (edit config/angel_one_config.json with your credentials)

# 3. Run the bot
python continuous_trading_bot.py
```

## 📁 Project Structure

```
Trading bot/
├── continuous_trading_bot.py      # Main bot (ENTRY POINT)
├── angel_one_auth_service.py      # Angel One authentication
├── config/
│   └── angel_one_config.json      # API keys & trading config
├── data/
│   ├── journal/                   # Trade journal (persistent learning)
│   │   ├── trades_journal.json    # Every trade with full context
│   │   ├── strategy_stats.json    # Per-strategy performance
│   │   ├── daily_stats.json       # Daily aggregates
│   │   └── learned_rules.json     # Current learned parameters
│   └── models/
│       ├── adaptive_params.json   # Adaptive engine parameters
│       └── composite_strategies.json  # Auto-discovered strategies
├── trading_bot/
│   └── ml/
│       ├── trade_journal.py       # Trade journaling & mistake classification
│       ├── adaptive_strategy_engine.py  # Self-learning strategy tuner
│       ├── regime_detector.py     # Market regime detection
│       ├── fno_scanner.py         # F&O / Options / Futures scanner
│       ├── reinforcement_learning.py  # DQN RL agent
│       ├── strategy_generator.py     # Genetic algorithm strategy gen
│       ├── model_trainer.py          # ML model training pipeline
│       ├── feature_engineering.py    # Feature extraction
│       └── market_intelligence.py    # Pattern & anomaly detection
├── strategies/                    # 50+ strategy implementations
│   ├── intraday_breakout/
│   ├── intraday_trend/
│   ├── intraday_reversion/
│   ├── price_action/
│   └── fno/
├── trades.json                    # Current session trades
├── logs/                          # Trading logs
└── dashboard.html                 # P&L Dashboard
```

## 📈 Learning Data Files

After the bot runs, you'll find these learning files:

| File | Contains |
|------|----------|
| `data/journal/trades_journal.json` | Full history of every trade with market context |
| `data/journal/strategy_stats.json` | Win rate, profit factor per strategy per regime |
| `data/journal/learned_rules.json` | Current learned rules (min score, weights, etc.) |
| `data/models/adaptive_params.json` | Adaptive parameters (SL/target multipliers, etc.) |
| `data/journal/learning_report_YYYYMMDD.json` | Daily learning reports |

## ⚙️ Configuration

Key config parameters in `config/angel_one_config.json`:

```json
{
  "trading": {
    "budget": 25000,
    "max_positions": 5,
    "max_daily_loss_pct": 5.0,
    "square_off_time": "15:20"
  }
}
```

## 🛡️ Risk Management

- **Max daily loss**: 5% of budget (configurable)
- **Per-trade risk**: 2% of budget (Kelly-scaled)
- **Regime-based position limits**: Fewer trades in volatile/sideways markets
- **Trailing stop-losses**: Lock in profits as trade moves in favor
- **Partial exits**: Book 50% at target 1, trail the rest
- **EOD auto square-off**: All positions closed by 15:25

## 📝 Notes

- This is a **paper trading** bot — it simulates trades for P&L tracking
- It uses Angel One SmartAPI for real-time market data
- The bot gets smarter with each session — the more it runs, the better it performs
- All learning data is persisted to disk and survives restarts