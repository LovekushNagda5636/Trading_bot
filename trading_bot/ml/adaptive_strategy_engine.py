"""
Adaptive Strategy Engine — Self-learning strategy selector and tuner.
Uses trade journal data to dynamically weight strategies, adjust parameters,
and create new composite strategies from winning combinations.
"""

import numpy as np
import json
import os
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict

# Lazy imports to avoid circular dependencies
def get_ml_components():
    from .strategy_generator import AIStrategyGenerator
    from .feature_engineering import FeatureEngineer
    from .market_intelligence import MarketIntelligence
    return AIStrategyGenerator, FeatureEngineer, MarketIntelligence

logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AdaptiveParams:
    """Parameters that the engine learns and adapts over time."""
    # Signal thresholds
    min_score: float = 35.0
    high_confidence_score: float = 60.0
    
    # Position sizing
    base_position_pct: float = 0.15  # % of budget per trade
    max_position_pct: float = 0.50
    
    # Risk management
    sl_multiplier: float = 1.0       # Applied to base SL
    target_multiplier: float = 1.0   # Applied to base target
    trailing_sl_pct: float = 0.5     # Trailing stop % from high
    
    # Timing
    avoid_after_hour: int = 15       # No entries after this hour
    prefer_before_hour: int = 11     # Preference for early entries
    
    # Strategy weights
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    
    # Regime-specific settings
    regime_aggressiveness: Dict[str, float] = field(default_factory=lambda: {
        "trending_up": 1.3,
        "trending_down": 1.1,
        "sideways": 0.7,
        "volatile": 0.5,
        "unknown": 1.0,
    })
    
    # Max concurrent positions per regime
    regime_max_positions: Dict[str, int] = field(default_factory=lambda: {
        "trending_up": 5,
        "trending_down": 4,
        "sideways": 2,
        "volatile": 1,
        "unknown": 3,
    })


@dataclass
class CompositeStrategy:
    """A strategy created by combining learned patterns."""
    name: str
    description: str
    entry_conditions: Dict[str, Any] = field(default_factory=dict)
    exit_conditions: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, float] = field(default_factory=dict)
    created_at: str = ""
    version: int = 1


class AdaptiveStrategyEngine:
    """
    Self-learning strategy engine that:
    1. Adjusts signal scoring weights based on past performance
    2. Creates new strategies from winning patterns
    3. Adapts risk parameters dynamically
    4. Routes signals through a learned confidence filter
    """

    def __init__(self, journal=None):
        self.params_file = MODELS_DIR / "adaptive_params.json"
        self.composites_file = MODELS_DIR / "composite_strategies.json"
        self.params = self._load_params()
        self.composites: List[CompositeStrategy] = self._load_composites()
        self.journal = journal
        
        # ── AI/ML Components ──
        AI_Gen, Feature_Eng, Market_Intel = get_ml_components()
        self.fe = Feature_Eng()
        self.generator = AI_Gen(self.fe)
        self.intel = Market_Intel(self.fe)
        
        # Load AI population
        self.pop_file = MODELS_DIR / "ai_population.json"
        if self.pop_file.exists():
            self.generator.load_population(str(self.pop_file))
        
        # Runtime state
        self._recent_signals: List[Dict] = []
        self._signal_outcomes: List[Dict] = []
        self._regime_history: List[str] = []
        
        logger.info(f"🧠 Adaptive Engine initialized | "
                     f"Min score: {self.params.min_score} | "
                     f"SL mult: {self.params.sl_multiplier}x | "
                     f"Target mult: {self.params.target_multiplier}x")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_params(self) -> AdaptiveParams:
        if self.params_file.exists():
            try:
                with open(self.params_file, 'r') as f:
                    data = json.load(f)
                return AdaptiveParams(**data)
            except Exception as e:
                logger.warning(f"Failed to load adaptive params: {e}")
        return AdaptiveParams()

    def _save_params(self):
        try:
            dir_name = str(self.params_file.parent)
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix="params_", dir=dir_name
            )
            with os.fdopen(fd, 'w') as f:
                json.dump(asdict(self.params), f, indent=2)
            os.replace(tmp_path, str(self.params_file))
        except Exception as e:
            logger.error(f"Failed to save adaptive params: {e}")
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    def _load_composites(self) -> List[CompositeStrategy]:
        if self.composites_file.exists():
            try:
                with open(self.composites_file, 'r') as f:
                    data = json.load(f)
                return [CompositeStrategy(**c) for c in data]
            except Exception:
                pass
        return []

    def _save_composites(self):
        try:
            dir_name = str(self.composites_file.parent)
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix="composites_", dir=dir_name
            )
            with os.fdopen(fd, 'w') as f:
                json.dump([asdict(c) for c in self.composites], f, indent=2)
            os.replace(tmp_path, str(self.composites_file))
        except Exception as e:
            logger.error(f"Failed to save composites: {e}")
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    # ── AI Strategy Execution ────────────────────────────────────────────────

    def get_ai_signals(self, symbol: str, stock_data: Dict, market_context: Dict = None) -> List[Dict]:
        """Run AI-generated strategies for a stock."""
        signals = []
        try:
            # Update feature engineer
            # Note: In a real tick-by-tick system, this would be updated elsewhere
            # but for our scan-based system, we update it here if needed.
            
            # Extract features
            features = self.fe.extract_features(symbol)
            if not features:
                return []
            
            # Update market intelligence
            # self.intel.update_market_data(symbol, None, features)
            
            # Run top 5 AI strategies
            ai_strats = self.generator.generate_strategies(num_strategies=5)
            for strat in ai_strats:
                # Mock a TradingSignal for the strategy check
                from trading_bot.core.models import TradingSignal, SignalType
                
                for side in ["BUY", "SELL"]:
                    sig = TradingSignal(symbol=symbol, signal_type=SignalType(side))
                    if strat._rule_based_entry_decision(features, sig, None):
                        signals.append({
                            "strategy": f"AI_{strat.strategy_config.strategy_id}",
                            "direction": side,
                            "score": 45,  # Base score for AI signals
                            "reason": f"AI Pattern: {strat.gene.entry_logic} combo of {len(strat.gene.entry_indicators)} indicators",
                            "is_ai": True
                        })
        except Exception as e:
            logger.error(f"AI signal error for {symbol}: {e}")
            
        return signals

    # ── Signal Enhancement ───────────────────────────────────────────────────

    def enhance_signal(self, signals: List[Dict], stock_data: Dict,
                       market_context: Dict = None) -> List[Dict]:
        """
        Re-score signals using learned weights and adaptive parameters.
        Time-of-day adjustments are handled upstream in StrategyEngine.analyze().
        """
        ctx = market_context or {}
        regime = ctx.get("regime", "unknown")
        aggressiveness = self.params.regime_aggressiveness.get(regime, 1.0)
        
        enhanced = []
        for sig in signals:
            strat_name = sig.get("strategy", "Unknown")
            base_score = sig.get("score", 0)
            
            # Apply learned strategy weight
            weight = self.params.strategy_weights.get(strat_name, 1.0)
            
            # Apply regime adjustment
            adjusted_score = base_score * weight * aggressiveness
            
            # Volume confirmation boost (small, since StrategyEngine already factors volume)
            volume = stock_data.get("volume", 0)
            if volume > 2_000_000:
                adjusted_score *= 1.05
            elif volume < 100_000:
                adjusted_score *= 0.85
            
            sig["raw_score"] = base_score
            sig["adjusted_score"] = round(adjusted_score, 1)
            sig["strategy_weight"] = weight
            sig["regime_factor"] = aggressiveness
            enhanced.append(sig)
        
        return enhanced

    def should_take_trade(self, opp: Dict, market_context: Dict = None) -> Tuple[bool, str, float]:
        """
        Master gate — decides whether to enter a trade based on all learned rules.
        Returns (should_trade, reason, confidence).
        """
        ctx = market_context or {}
        
        # 1. Check score threshold (learned)
        score = opp.get("score", 0)
        if score < self.params.min_score:
            return False, f"Score {score:.0f} < learned min {self.params.min_score:.0f}", 0
        
        # 2. Check journal-based rules
        if self.journal:
            avoid, reason = self.journal.should_avoid_trade(opp, ctx)
            if avoid:
                return False, f"Journal rule: {reason}", 0
        
        # 3. Check regime limits
        regime = ctx.get("regime", "unknown")
        max_pos = self.params.regime_max_positions.get(regime, 3)
        current_pos = ctx.get("open_positions", 0)
        if current_pos >= max_pos:
            return False, f"Max positions ({max_pos}) for {regime} regime reached", 0
        
        # 4. Check daily loss limit
        daily_pnl = ctx.get("daily_pnl", 0)
        budget = ctx.get("budget", 25000)
        max_daily_loss = budget * 0.05  # 5% max daily loss
        if daily_pnl < -max_daily_loss:
            return False, f"Daily loss limit ₹{max_daily_loss:.0f} reached", 0
        
        # 5. Compute confidence
        confidence = self._compute_confidence(opp, ctx)
        
        if confidence < 0.3:
            return False, f"Low confidence {confidence:.2f}", confidence
        
        return True, "All checks passed", confidence

    def _compute_confidence(self, opp: Dict, ctx: Dict) -> float:
        """Compute trade confidence from multiple factors including RSI alignment."""
        factors = []
        
        # Score factor (30%)
        score = opp.get("score", 0)
        score_factor = min(1.0, score / 80.0)
        factors.append(score_factor * 0.30)
        
        # Strategy count factor — reduced weight since dedup handles correlation (10%)
        num_strats = opp.get("num_strategies", 1)
        strat_factor = min(1.0, num_strats / 4.0)
        factors.append(strat_factor * 0.10)
        
        # Risk/reward factor (20%)
        rr = opp.get("risk_reward", 1.0)
        rr_factor = min(1.0, rr / 3.0)
        factors.append(rr_factor * 0.20)
        
        # Strategy confidence from journal (15%)
        if self.journal:
            strategies = opp.get("strategies", [])
            if strategies:
                strat_conf = np.mean([self.journal.get_strategy_weight(s) for s in strategies])
                factors.append(min(1.0, strat_conf) * 0.15)
            else:
                factors.append(0.5 * 0.15)
        else:
            factors.append(0.5 * 0.15)
        
        # Volume factor (10%)
        volume = opp.get("volume", 0)
        vol_factor = min(1.0, volume / 2_000_000)
        factors.append(vol_factor * 0.10)
        
        # ── NEW: RSI alignment factor (15%) ──
        rsi = opp.get("rsi", 50.0)
        direction = opp.get("direction", ctx.get("direction", "BUY"))
        if direction == "BUY":
            # Buying: best at RSI 30-60 (oversold bounce), worst at RSI > 80
            if rsi < 25:
                rsi_factor = 0.9  # Very oversold, good for bounce
            elif rsi < 60:
                rsi_factor = 1.0  # Ideal buy zone
            elif rsi < 75:
                rsi_factor = 0.6  # Getting risky
            else:
                rsi_factor = 0.2  # Overbought, bad time to buy
        else:
            # Selling: best at RSI 40-70 (overbought), worst at RSI < 20
            if rsi > 75:
                rsi_factor = 0.9  # Very overbought, good for short
            elif rsi > 40:
                rsi_factor = 1.0  # Ideal sell zone
            elif rsi > 25:
                rsi_factor = 0.6  # Getting risky
            else:
                rsi_factor = 0.2  # Oversold, bad time to short
        factors.append(rsi_factor * 0.15)
        
        return round(sum(factors), 3)

    # ── Dynamic Risk Parameters ──────────────────────────────────────────────

    def get_dynamic_targets(self, ltp: float, direction: str,
                            stock_data: Dict, market_context: Dict = None) -> Dict:
        """
        Compute adaptive SL/target levels using learned parameters.
        """
        ctx = market_context or {}
        regime = ctx.get("regime", "unknown")
        
        # Base percentages
        base_sl_pct = 0.01      # 1% SL
        base_t1_pct = 0.015     # 1.5% target 1
        base_t2_pct = 0.030     # 3.0% target 2
        
        # Adapt based on day range
        day_range = stock_data.get("high", ltp) - stock_data.get("low", ltp)
        day_range_pct = (day_range / ltp) if ltp > 0 else 0
        
        if day_range_pct > 0.03:  # Wide range day
            base_sl_pct *= 1.5
            base_t1_pct *= 1.3
            base_t2_pct *= 1.3
        elif day_range_pct < 0.01:  # Tight range day
            base_sl_pct *= 0.7
            base_t1_pct *= 0.7
            base_t2_pct *= 0.8
        
        # Apply learned multipliers
        sl_pct = base_sl_pct * self.params.sl_multiplier
        t1_pct = base_t1_pct * self.params.target_multiplier
        t2_pct = base_t2_pct * self.params.target_multiplier
        
        # Regime adjustments
        if regime == "volatile":
            sl_pct *= 1.5
            t1_pct *= 1.3
        elif regime == "sideways":
            sl_pct *= 0.8
            t1_pct *= 0.8
        
        if direction == "BUY":
            return {
                "stop_loss": round(ltp * (1 - sl_pct), 2),
                "target_1": round(ltp * (1 + t1_pct), 2),
                "target_2": round(ltp * (1 + t2_pct), 2),
                "trailing_sl_pct": self.params.trailing_sl_pct,
            }
        else:
            return {
                "stop_loss": round(ltp * (1 + sl_pct), 2),
                "target_1": round(ltp * (1 - t1_pct), 2),
                "target_2": round(ltp * (1 - t2_pct), 2),
                "trailing_sl_pct": self.params.trailing_sl_pct,
            }

    def get_position_size(self, ltp: float, sl_price: float, budget: float,
                          confidence: float = 0.5) -> int:
        """
        Adaptive position sizing based on confidence and risk.
        Uses fractional Kelly criterion for optimal sizing.
        """
        risk_per_share = abs(ltp - sl_price)
        if risk_per_share <= 0:
            return max(1, int(budget * 0.10 / ltp))
        
        # Base position = 2% risk of budget
        max_risk = budget * 0.02
        base_qty = int(max_risk / risk_per_share)
        
        # Scale by confidence (Kelly-like)
        # Higher confidence → larger position
        kelly_fraction = confidence * 0.5  # Half-Kelly for safety
        adjusted_qty = int(base_qty * max(0.3, kelly_fraction * 2))
        
        # Cap by max position size
        max_qty = int(budget * self.params.max_position_pct / ltp)
        min_qty = 1
        
        return max(min_qty, min(adjusted_qty, max_qty))

    # ── Self-Learning Loop ───────────────────────────────────────────────────

    def learn_from_session(self):
        """
        Called at end of day or after N trades.
        Analyzes journal data and updates all adaptive parameters.
        """
        if not self.journal:
            logger.warning("No journal connected, cannot learn")
            return
        
        logger.info("🧠 Starting self-learning cycle...")
        
        # 1. Update score threshold
        new_min = self.journal.get_learned_min_score()
        if new_min != self.params.min_score:
            logger.info(f"📊 Min score: {self.params.min_score} → {new_min}")
            self.params.min_score = new_min
        
        # 2. Update SL/target multipliers
        new_sl = self.journal.get_optimal_sl_multiplier()
        new_tgt = self.journal.get_optimal_target_multiplier()
        if new_sl != self.params.sl_multiplier:
            logger.info(f"📊 SL multiplier: {self.params.sl_multiplier}x → {new_sl}x")
            self.params.sl_multiplier = new_sl
        if new_tgt != self.params.target_multiplier:
            logger.info(f"📊 Target multiplier: {self.params.target_multiplier}x → {new_tgt}x")
            self.params.target_multiplier = new_tgt
        
        # 3. Update strategy weights
        for strat_name in self.journal.strategy_stats:
            new_weight = self.journal.get_strategy_weight(strat_name)
            old_weight = self.params.strategy_weights.get(strat_name, 1.0)
            if abs(new_weight - old_weight) > 0.05:
                logger.info(f"📊 {strat_name} weight: {old_weight} → {new_weight}")
                self.params.strategy_weights[strat_name] = new_weight
        
        # 4. Update timing preferences
        best_hours = self.journal.get_best_entry_hours()
        worst_hours = self.journal.get_worst_entry_hours()
        if worst_hours:
            # Set avoid_after_hour to earliest consistently bad hour
            afternoon_bad = [h for h in worst_hours if h >= 13]
            if afternoon_bad:
                self.params.avoid_after_hour = min(afternoon_bad)
        
        # 5. Update regime weights
        regime_weights = self.journal.get_regime_weights()
        for regime, weight in regime_weights.items():
            self.params.regime_aggressiveness[regime] = max(0.3, min(2.0, weight * 1.5))
        
        # 6. Discover new composite strategies
        self._discover_composites()
        
        # 7. Evolve AI Strategies
        logger.info("🧬 Evolving AI strategy population...")
        # Prepare performance data for the generator from journal
        perf_data = {}
        if self.journal:
            # Map AI strategy performance from journal to generator format
            from .strategy_generator import StrategyPerformance
            # This is a bit simplified - in reality we track which AI ID mapped to what
            for i in range(self.generator.population_size):
                strat_id = f"generated_{i}" # This needs to match internal generator naming
                # Find trades for this AI strat
                trades = [t for t in self.journal.trades if any(s.startswith(f"AI_ai_generated_{strat_id}") for s in t.strategies_used)]
                if trades:
                    wins = sum(1 for t in trades if t.pnl > 0)
                    perf_data[strat_id] = StrategyPerformance(
                        total_trades=len(trades),
                        win_rate=wins/len(trades),
                        total_return=sum(t.pnl for t in trades) / 25000, # Normalize
                    )
        
        self.generator.evolve_population(perf_data)
        self.generator.save_population(str(self.pop_file))
        
        # 8. Retrain Market Intel
        self.intel._retrain_models()

        # 9. Update journal learned rules
        self.journal.update_learned_rules()
        
        # 10. GUARDRAILS: Clamp all parameters to safe ranges
        self._enforce_parameter_bounds()
        
        # 11. Persist
        self._save_params()
        self._save_composites()
        
        logger.info("✅ Self-learning cycle complete")
        return self.get_learning_report()

    def _enforce_parameter_bounds(self):
        """Clamp learned parameters to safe ranges to prevent catastrophic drift."""
        p = self.params
        old_vals = {
            'min_score': p.min_score,
            'sl_multiplier': p.sl_multiplier,
            'target_multiplier': p.target_multiplier,
            'avoid_after_hour': p.avoid_after_hour,
            'trailing_sl_pct': p.trailing_sl_pct,
            'base_position_pct': p.base_position_pct,
            'max_position_pct': p.max_position_pct,
        }
        
        # Score thresholds
        p.min_score = max(25.0, min(80.0, p.min_score))
        p.high_confidence_score = max(40.0, min(90.0, p.high_confidence_score))
        
        # Risk multipliers
        p.sl_multiplier = max(0.5, min(2.0, p.sl_multiplier))
        p.target_multiplier = max(0.5, min(2.5, p.target_multiplier))
        p.trailing_sl_pct = max(0.2, min(2.0, p.trailing_sl_pct))
        
        # Position sizing
        p.base_position_pct = max(0.05, min(0.25, p.base_position_pct))
        p.max_position_pct = max(0.10, min(0.50, p.max_position_pct))
        
        # Timing
        p.avoid_after_hour = max(13, min(15, p.avoid_after_hour))
        p.prefer_before_hour = max(9, min(12, p.prefer_before_hour))
        
        # Strategy weights: clamp to [0.1, 3.0]
        for k in list(p.strategy_weights.keys()):
            p.strategy_weights[k] = max(0.1, min(3.0, p.strategy_weights[k]))
        
        # Regime aggressiveness: clamp to [0.2, 2.0]
        for k in list(p.regime_aggressiveness.keys()):
            p.regime_aggressiveness[k] = max(0.2, min(2.0, p.regime_aggressiveness[k]))
        
        # Regime max positions: clamp to [1, 5]
        for k in list(p.regime_max_positions.keys()):
            p.regime_max_positions[k] = max(1, min(5, p.regime_max_positions[k]))
        
        # Log any clamped values
        for key, old_val in old_vals.items():
            new_val = getattr(p, key)
            if old_val != new_val:
                logger.warning(
                    f"🛡️ Parameter '{key}' clamped: {old_val} → {new_val} (safety bounds)"
                )

    def _discover_composites(self):
        """
        Analyze trade history to discover strategy combinations that work well.
        Creates new composite strategies from winning patterns.
        """
        if not self.journal or len(self.journal.trades) < 20:
            return
        
        closed_trades = [t for t in self.journal.trades if t.status != "OPEN"]
        
        # Find strategy pairs that win together
        pair_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0})
        
        for trade in closed_trades:
            strats = sorted(trade.strategies_used)
            if len(strats) >= 2:
                for i in range(len(strats)):
                    for j in range(i + 1, len(strats)):
                        pair = f"{strats[i]}+{strats[j]}"
                        pair_stats[pair]["trades"] += 1
                        if trade.pnl > 0:
                            pair_stats[pair]["wins"] += 1
                        pair_stats[pair]["total_pnl"] += trade.pnl
        
        # Create composites for winning pairs
        existing_names = {c.name for c in self.composites}
        
        for pair_name, stats in pair_stats.items():
            if stats["trades"] >= 5 and stats["wins"] / stats["trades"] > 0.6:
                comp_name = f"Composite_{pair_name.replace('+', '_')}"
                if comp_name not in existing_names:
                    strat_a, strat_b = pair_name.split("+")
                    composite = CompositeStrategy(
                        name=comp_name,
                        description=f"Auto-discovered: {strat_a} + {strat_b} combo "
                                    f"(WR: {stats['wins']/stats['trades']*100:.0f}%)",
                        entry_conditions={
                            "required_strategies": [strat_a, strat_b],
                            "min_score": 30,
                            "logic": "AND",
                        },
                        exit_conditions={
                            "sl_pct": 1.0 * self.params.sl_multiplier,
                            "target_pct": 1.5 * self.params.target_multiplier,
                        },
                        performance={
                            "trades": stats["trades"],
                            "win_rate": round(stats["wins"] / stats["trades"], 3),
                            "total_pnl": round(stats["total_pnl"], 2),
                        },
                        created_at=datetime.now().isoformat(),
                    )
                    self.composites.append(composite)
                    logger.info(f"🆕 Discovered composite strategy: {comp_name} "
                               f"(WR: {stats['wins']/stats['trades']*100:.0f}%, "
                               f"P&L: ₹{stats['total_pnl']:.0f})")

    def check_composite_match(self, strategies: List[str]) -> Optional[CompositeStrategy]:
        """Check if current signal matches any composite strategy."""
        for comp in self.composites:
            required = comp.entry_conditions.get("required_strategies", [])
            if required and all(r in strategies for r in required):
                return comp
        return None

    # ── Reporting ────────────────────────────────────────────────────────────

    def get_learning_report(self) -> Dict:
        """Get human-readable learning status report."""
        report = {
            "adaptive_params": {
                "min_score": self.params.min_score,
                "sl_multiplier": self.params.sl_multiplier,
                "target_multiplier": self.params.target_multiplier,
                "avoid_after_hour": self.params.avoid_after_hour,
                "trailing_sl_pct": self.params.trailing_sl_pct,
            },
            "strategy_weights": dict(self.params.strategy_weights),
            "regime_settings": dict(self.params.regime_aggressiveness),
            "composite_strategies": len(self.composites),
            "composites": [
                {"name": c.name, "wr": c.performance.get("win_rate", 0),
                 "trades": c.performance.get("trades", 0)}
                for c in self.composites
            ],
        }
        
        if self.journal:
            report["journal_summary"] = self.journal.get_learning_summary()
        
        return report
