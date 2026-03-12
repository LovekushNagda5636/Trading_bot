"""
Trade Journal — Persistent learning memory for the trading bot.
Records every trade with rich context so the ML system can learn from mistakes.
"""

import json
import os
import tempfile
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import defaultdict

logger = logging.getLogger(__name__)

JOURNAL_DIR = Path("data/journal")
JOURNAL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TradeRecord:
    """Complete record of a single trade for learning."""
    trade_id: str
    symbol: str
    direction: str  # BUY / SELL
    instrument_type: str  # EQ / FUT / OPT
    entry_price: float
    entry_time: str
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    qty: int = 1
    target: float = 0.0
    stop_loss: float = 0.0
    status: str = "OPEN"  # OPEN / TARGET_HIT / SL_HIT / EOD_SQUARE_OFF / TRAIL_EXIT
    pnl: float = 0.0
    pnl_pct: float = 0.0
    # Strategy context
    strategies_used: List[str] = field(default_factory=list)
    signal_score: float = 0.0
    num_agreeing_strategies: int = 0
    reasons: List[str] = field(default_factory=list)
    # Market context at entry
    market_regime: str = "unknown"  # trending_up / trending_down / sideways / volatile
    market_volatility: float = 0.0
    sector_trend: str = "neutral"
    vix_level: float = 0.0
    nifty_change_pct: float = 0.0
    # Price context
    day_range_pct: float = 0.0
    volume_ratio: float = 0.0  # volume vs average
    gap_pct: float = 0.0
    vwap_deviation: float = 0.0
    rsi_at_entry: float = 50.0
    # Timing
    entry_hour: int = 0
    entry_minute: int = 0
    day_of_week: int = 0
    # Post-trade analysis
    max_favorable_excursion: float = 0.0  # Max profit during trade
    max_adverse_excursion: float = 0.0    # Max loss during trade
    time_in_trade_minutes: float = 0.0
    # Learning tags
    mistake_type: str = ""  # premature_exit / late_entry / wrong_direction / bad_sl / bad_target
    lesson_learned: str = ""


@dataclass
class StrategyPerformanceRecord:
    """Tracks how each strategy performs over time."""
    strategy_name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    # By market regime
    regime_performance: Dict[str, Dict] = field(default_factory=dict)
    # By time of day
    hourly_performance: Dict[int, Dict] = field(default_factory=dict)
    # By day of week
    daily_performance: Dict[int, Dict] = field(default_factory=dict)
    # Confidence score (learned)
    confidence_score: float = 0.5
    last_updated: str = ""


class TradeJournal:
    """
    Persistent trade journal that records everything and computes
    analytics the ML system can use for self-improvement.
    """

    def __init__(self):
        self.journal_file = JOURNAL_DIR / "trades_journal.json"
        self.strategy_stats_file = JOURNAL_DIR / "strategy_stats.json"
        self.daily_stats_file = JOURNAL_DIR / "daily_stats.json"
        self.learning_file = JOURNAL_DIR / "learned_rules.json"
        
        self.trades: List[TradeRecord] = []
        self.strategy_stats: Dict[str, StrategyPerformanceRecord] = {}
        self.daily_stats: Dict[str, Dict] = {}
        self.learned_rules: Dict[str, Any] = {}
        
        self._load_all()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_all(self):
        """Load all journal data from disk."""
        self.trades = self._load_trades()
        self.strategy_stats = self._load_strategy_stats()
        self.daily_stats = self._load_json(self.daily_stats_file, {})
        self.learned_rules = self._load_json(self.learning_file, {
            "min_score_threshold": 35.0,
            "preferred_strategies": [],
            "avoided_strategies": [],
            "best_entry_hours": [],
            "worst_entry_hours": [],
            "regime_weights": {},
            "dynamic_sl_multiplier": 1.0,
            "dynamic_target_multiplier": 1.0,
            "max_positions_override": None,
            "strategy_weights": {},
            "version": 1,
        })
        logger.info(f"📓 Journal loaded: {len(self.trades)} trades, "
                     f"{len(self.strategy_stats)} strategies tracked")

    def _load_trades(self) -> List[TradeRecord]:
        if not self.journal_file.exists():
            return []
        try:
            with open(self.journal_file, 'r') as f:
                data = json.load(f)
            return [TradeRecord(**t) for t in data]
        except Exception as e:
            logger.warning(f"Failed to load journal: {e}")
            return []

    def _load_strategy_stats(self) -> Dict[str, StrategyPerformanceRecord]:
        if not self.strategy_stats_file.exists():
            return {}
        try:
            with open(self.strategy_stats_file, 'r') as f:
                data = json.load(f)
            return {k: StrategyPerformanceRecord(**v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load strategy stats: {e}")
            return {}

    def _load_json(self, filepath: Path, default: Any) -> Any:
        if not filepath.exists():
            return default
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return default

    def _atomic_write(self, filepath: Path, data):
        """Write data to file atomically via temp file + rename."""
        dir_name = str(filepath.parent)
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", prefix="journal_", dir=dir_name)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, str(filepath))
        except Exception as e:
            logger.error(f"Failed to write {filepath.name}: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _save_all(self):
        """Persist everything to disk atomically."""
        try:
            self._atomic_write(
                self.journal_file,
                [asdict(t) for t in self.trades]
            )
            self._atomic_write(
                self.strategy_stats_file,
                {k: asdict(v) for k, v in self.strategy_stats.items()}
            )
            self._atomic_write(self.daily_stats_file, self.daily_stats)
            self._atomic_write(self.learning_file, self.learned_rules)
        except Exception as e:
            logger.error(f"Failed to save journal data: {e}")

    # ── Record Trades ────────────────────────────────────────────────────────

    def record_entry(self, opp: Dict, market_context: Dict = None) -> TradeRecord:
        """Record a new trade entry with full market context."""
        now = datetime.now()
        ctx = market_context or {}
        
        # Compute additional context from opportunity data
        ltp = opp.get("ltp", 0)
        open_p = opp.get("open", ltp)
        high = opp.get("high", ltp)
        low = opp.get("low", ltp)
        
        day_range_pct = ((high - low) / open_p * 100) if open_p > 0 else 0
        gap_pct = ((open_p - opp.get("prev_close", open_p)) / opp.get("prev_close", open_p) * 100) \
                  if opp.get("prev_close", 0) > 0 else 0
        vwap = (high + low + ltp) / 3 if high > 0 else ltp
        vwap_dev = ((ltp - vwap) / vwap * 100) if vwap > 0 else 0
        
        record = TradeRecord(
            trade_id=f"J{int(now.timestamp())}_{opp.get('symbol', 'UNK')}",
            symbol=opp.get("symbol", ""),
            direction=opp.get("direction", "BUY"),
            instrument_type=opp.get("instrument_type", "EQ"),
            entry_price=ltp,
            entry_time=now.isoformat(),
            qty=opp.get("qty", 1),
            target=opp.get("target_1", 0),
            stop_loss=opp.get("stop_loss", 0),
            status="OPEN",
            strategies_used=opp.get("strategies", []),
            signal_score=opp.get("score", 0),
            num_agreeing_strategies=opp.get("num_strategies", 0),
            reasons=opp.get("reasons", []),
            # Market context
            market_regime=ctx.get("regime", "unknown"),
            market_volatility=ctx.get("volatility", 0),
            nifty_change_pct=ctx.get("nifty_change_pct", 0),
            vix_level=ctx.get("vix", 0),
            # Price context
            day_range_pct=round(day_range_pct, 2),
            volume_ratio=opp.get("volume_ratio", 1.0),
            gap_pct=round(gap_pct, 2),
            vwap_deviation=round(vwap_dev, 2),
            # Timing
            entry_hour=now.hour,
            entry_minute=now.minute,
            day_of_week=now.weekday(),
        )
        
        self.trades.append(record)
        self._save_all()
        return record

    def record_exit(self, trade_id: str, exit_price: float, status: str,
                    max_favorable: float = 0, max_adverse: float = 0):
        """Record trade exit and compute post-trade analytics."""
        now = datetime.now()
        
        for t in self.trades:
            if t.trade_id == trade_id and t.status == "OPEN":
                t.exit_price = exit_price
                t.exit_time = now.isoformat()
                t.status = status
                
                # Calculate P&L
                if t.direction == "BUY":
                    t.pnl = round((exit_price - t.entry_price) * t.qty, 2)
                else:
                    t.pnl = round((t.entry_price - exit_price) * t.qty, 2)
                
                t.pnl_pct = round((t.pnl / (t.entry_price * t.qty)) * 100, 2) if t.entry_price > 0 else 0
                t.max_favorable_excursion = max_favorable
                t.max_adverse_excursion = max_adverse
                
                # Time in trade
                try:
                    entry_dt = datetime.fromisoformat(t.entry_time)
                    t.time_in_trade_minutes = (now - entry_dt).total_seconds() / 60
                except (ValueError, TypeError):
                    t.time_in_trade_minutes = 0
                
                # Classify mistakes
                t.mistake_type = self._classify_mistake(t)
                t.lesson_learned = self._generate_lesson(t)
                
                # Update strategy stats
                self._update_strategy_stats(t)
                
                # Update daily stats
                self._update_daily_stats(t)
                
                self._save_all()
                
                logger.info(f"📓 Trade closed: {t.symbol} | P&L: ₹{t.pnl} ({t.pnl_pct}%) | "
                           f"Mistake: {t.mistake_type or 'None'}")
                return t
        
        return None

    # ── Mistake Classification ───────────────────────────────────────────────

    def _classify_mistake(self, trade: TradeRecord) -> str:
        """Classify the type of mistake made (if any)."""
        if trade.pnl >= 0:
            # Even winning trades can have mistakes
            if trade.max_favorable_excursion > 0:
                potential_gain = trade.max_favorable_excursion * trade.qty
                actual_gain = trade.pnl
                if potential_gain > 0 and actual_gain < potential_gain * 0.3:
                    return "premature_exit"
            return ""
        
        # ── NEW: Detect chasing exhausted moves ──
        # If entry was after a big move and it immediately reversed
        entry_price = trade.entry_price
        open_price = getattr(trade, 'open_price', 0) or entry_price
        if open_price > 0:
            move_at_entry = abs(entry_price - open_price) / open_price * 100
            if move_at_entry > 3.0:
                return "chased_exhausted_move"
        
        # ── NEW: Detect signal inflation ──
        # Too many strategies agreed but trade still lost → likely correlated noise
        if len(trade.strategies_used) >= 4 and trade.signal_score >= 70:
            return "signal_inflation"
        
        # Loss analysis
        if trade.status == "SL_HIT":
            # Was the stop loss too tight?
            if abs(trade.pnl_pct) < 0.5 and trade.max_favorable_excursion > 0:
                return "tight_sl"
            # Did it reverse after hitting SL?
            if trade.max_adverse_excursion > 0:
                return "bad_sl_placement"
            return "correct_sl"
        
        if trade.status == "EOD_SQUARE_OFF":
            if abs(trade.pnl_pct) < 0.3:
                return "flat_trade"
            return "wrong_direction"
        
        # General loss
        if trade.day_range_pct < 0.5:
            return "low_volatility_entry"
        
        if trade.entry_hour >= 14:
            return "late_entry"
        
        if trade.signal_score < 40:
            return "weak_signal"
        
        return "wrong_direction"

    def _generate_lesson(self, trade: TradeRecord) -> str:
        """Generate a learning note from the trade outcome."""
        if trade.pnl >= 0:
            if trade.mistake_type == "premature_exit":
                return f"Hold longer when {', '.join(trade.strategies_used)} align. " \
                       f"Left money on table."
            return f"Good trade. {', '.join(trade.strategies_used)} worked in " \
                   f"{trade.market_regime} regime."
        
        lessons = {
            "tight_sl": f"Widen SL for {trade.symbol} in {trade.market_regime} regime. "
                        f"Range was {trade.day_range_pct}%.",
            "bad_sl_placement": f"Use ATR-based SL instead of fixed % for {trade.symbol}.",
            "wrong_direction": f"Strategy {', '.join(trade.strategies_used)} failed in "
                              f"{trade.market_regime}. Avoid in similar conditions.",
            "low_volatility_entry": f"Don't enter {trade.symbol} when range < 0.5%. "
                                    f"Wait for expansion.",
            "late_entry": f"Avoid entries after 2 PM. Trade showed {trade.pnl_pct}% loss.",
            "weak_signal": f"Score {trade.signal_score} was too low. "
                          f"Raise threshold for {trade.market_regime} regime.",
            "flat_trade": f"No trend developed. Skip narrow-range signals in afternoon.",
            "chased_exhausted_move": f"Entered {trade.symbol} after move was already extended. "
                                     f"Don't chase moves > 3%. Wait for pullback or skip.",
            "signal_inflation": f"{len(trade.strategies_used)} correlated strategies falsely "
                               f"inflated score to {trade.signal_score}. "
                               f"Require diverse (uncorrelated) strategy signals.",
        }
        return lessons.get(trade.mistake_type, f"Loss of ₹{trade.pnl} on {trade.symbol}")

    # ── Strategy Stats ───────────────────────────────────────────────────────

    def _update_strategy_stats(self, trade: TradeRecord):
        """Update per-strategy performance tracking."""
        for strat_name in trade.strategies_used:
            if strat_name not in self.strategy_stats:
                self.strategy_stats[strat_name] = StrategyPerformanceRecord(
                    strategy_name=strat_name
                )
            
            stats = self.strategy_stats[strat_name]
            stats.total_trades += 1
            stats.total_pnl += trade.pnl
            
            if trade.pnl > 0:
                stats.wins += 1
            elif trade.pnl < 0:
                stats.losses += 1
            else:
                stats.draws += 1
            
            stats.win_rate = stats.wins / stats.total_trades if stats.total_trades > 0 else 0
            stats.avg_pnl = stats.total_pnl / stats.total_trades if stats.total_trades > 0 else 0
            
            # Track wins/losses separately
            winning_trades = [t for t in self.trades if strat_name in t.strategies_used and t.pnl > 0]
            losing_trades = [t for t in self.trades if strat_name in t.strategies_used and t.pnl < 0]
            
            stats.avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            stats.avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
            total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
            stats.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Regime performance
            regime = trade.market_regime
            if regime not in stats.regime_performance:
                stats.regime_performance[regime] = {"trades": 0, "wins": 0, "pnl": 0}
            stats.regime_performance[regime]["trades"] += 1
            if trade.pnl > 0:
                stats.regime_performance[regime]["wins"] += 1
            stats.regime_performance[regime]["pnl"] += trade.pnl
            
            # Hourly performance
            hour = trade.entry_hour
            if hour not in stats.hourly_performance:
                stats.hourly_performance[hour] = {"trades": 0, "wins": 0, "pnl": 0}
            stats.hourly_performance[hour]["trades"] += 1
            if trade.pnl > 0:
                stats.hourly_performance[hour]["wins"] += 1
            stats.hourly_performance[hour]["pnl"] += trade.pnl
            
            # Day of week performance
            dow = trade.day_of_week
            if dow not in stats.daily_performance:
                stats.daily_performance[dow] = {"trades": 0, "wins": 0, "pnl": 0}
            stats.daily_performance[dow]["trades"] += 1
            if trade.pnl > 0:
                stats.daily_performance[dow]["wins"] += 1
            stats.daily_performance[dow]["pnl"] += trade.pnl
            
            # Update confidence based on recent performance
            recent = [t for t in self.trades[-50:] if strat_name in t.strategies_used]
            if len(recent) >= 5:
                recent_wr = sum(1 for t in recent if t.pnl > 0) / len(recent)
                stats.confidence_score = round(recent_wr * 0.6 + stats.win_rate * 0.4, 3)
            
            stats.last_updated = datetime.now().isoformat()

    def _update_daily_stats(self, trade: TradeRecord):
        """Update daily aggregate statistics."""
        date_key = datetime.now().strftime("%Y-%m-%d")
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                "trades": 0, "wins": 0, "losses": 0, "total_pnl": 0,
                "max_win": 0, "max_loss": 0, "strategies_used": [],
            }
        
        day = self.daily_stats[date_key]
        day["trades"] += 1
        day["total_pnl"] += trade.pnl
        
        if trade.pnl > 0:
            day["wins"] += 1
            day["max_win"] = max(day["max_win"], trade.pnl)
        elif trade.pnl < 0:
            day["losses"] += 1
            day["max_loss"] = min(day["max_loss"], trade.pnl)
        
        for s in trade.strategies_used:
            if s not in day["strategies_used"]:
                day["strategies_used"].append(s)

    # ── Learning Queries ─────────────────────────────────────────────────────

    def get_best_strategies(self, min_trades: int = 5) -> List[str]:
        """Get strategies ranked by confidence score (learned from outcomes)."""
        eligible = {k: v for k, v in self.strategy_stats.items() 
                    if v.total_trades >= min_trades}
        sorted_strats = sorted(eligible.items(), 
                               key=lambda x: x[1].confidence_score, reverse=True)
        return [name for name, _ in sorted_strats]

    def get_worst_strategies(self, min_trades: int = 5) -> List[str]:
        """Get strategies that consistently lose money."""
        eligible = {k: v for k, v in self.strategy_stats.items() 
                    if v.total_trades >= min_trades and v.win_rate < 0.35}
        return list(eligible.keys())

    def get_best_entry_hours(self) -> List[int]:
        """Get hours of day that historically produce winning trades."""
        hour_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
        for t in self.trades:
            if t.status != "OPEN":
                hour_stats[t.entry_hour]["trades"] += 1
                if t.pnl > 0:
                    hour_stats[t.entry_hour]["wins"] += 1
                hour_stats[t.entry_hour]["pnl"] += t.pnl
        
        good_hours = []
        for hour, stats in hour_stats.items():
            if stats["trades"] >= 3:
                wr = stats["wins"] / stats["trades"]
                if wr >= 0.5 and stats["pnl"] > 0:
                    good_hours.append(hour)
        
        return sorted(good_hours)

    def get_worst_entry_hours(self) -> List[int]:
        """Get hours of day that historically produce losing trades."""
        hour_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
        for t in self.trades:
            if t.status != "OPEN":
                hour_stats[t.entry_hour]["trades"] += 1
                if t.pnl > 0:
                    hour_stats[t.entry_hour]["wins"] += 1
                hour_stats[t.entry_hour]["pnl"] += t.pnl
        
        bad_hours = []
        for hour, stats in hour_stats.items():
            if stats["trades"] >= 3:
                wr = stats["wins"] / stats["trades"]
                if wr < 0.35 or stats["pnl"] < 0:
                    bad_hours.append(hour)
        
        return sorted(bad_hours)

    def get_optimal_sl_multiplier(self) -> float:
        """Learn optimal SL multiplier from trade history."""
        sl_hits = [t for t in self.trades 
                   if t.status == "SL_HIT" and t.mistake_type in ("tight_sl", "bad_sl_placement")]
        if len(sl_hits) < 3:
            return 1.0
        
        # If too many tight SLs, widen
        tight_ratio = sum(1 for t in sl_hits if t.mistake_type == "tight_sl") / len(sl_hits)
        if tight_ratio > 0.5:
            return min(2.0, 1.0 + tight_ratio * 0.5)
        return 1.0

    def get_optimal_target_multiplier(self) -> float:
        """Learn optimal target multiplier from trade history."""
        premature_exits = [t for t in self.trades if t.mistake_type == "premature_exit"]
        if len(premature_exits) < 3:
            return 1.0
        
        # If many premature exits, raise targets
        total_closed = sum(1 for t in self.trades if t.status != "OPEN")
        if total_closed > 0:
            ratio = len(premature_exits) / total_closed
            if ratio > 0.3:
                return min(2.0, 1.0 + ratio * 0.5)
        return 1.0

    def get_regime_weights(self) -> Dict[str, float]:
        """Get learned weights for different market regimes."""
        regime_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
        for t in self.trades:
            if t.status != "OPEN":
                regime_stats[t.market_regime]["trades"] += 1
                if t.pnl > 0:
                    regime_stats[t.market_regime]["wins"] += 1
                regime_stats[t.market_regime]["pnl"] += t.pnl
        
        weights = {}
        for regime, stats in regime_stats.items():
            if stats["trades"] >= 3:
                wr = stats["wins"] / stats["trades"]
                avg_pnl = stats["pnl"] / stats["trades"]
                weights[regime] = round(wr * 0.7 + (1 if avg_pnl > 0 else 0) * 0.3, 3)
        
        return weights

    def get_learned_min_score(self) -> float:
        """Learn the optimal minimum score threshold from history."""
        if len(self.trades) < 10:
            return 35.0
        
        scores_and_outcomes = [(t.signal_score, t.pnl > 0) 
                               for t in self.trades if t.status != "OPEN"]
        if not scores_and_outcomes:
            return 35.0
        
        # Find the score threshold that maximizes win rate
        best_threshold = 35.0
        best_metric = 0
        
        for threshold in range(20, 90, 5):
            filtered = [(s, w) for s, w in scores_and_outcomes if s >= threshold]
            if len(filtered) < 3:
                continue
            wr = sum(1 for _, w in filtered if w) / len(filtered)
            # Metric = win_rate * sqrt(num_trades)
            metric = wr * np.sqrt(len(filtered))
            if metric > best_metric:
                best_metric = metric
                best_threshold = float(threshold)
        
        return best_threshold

    def get_strategy_weight(self, strategy_name: str) -> float:
        """Get learned weight for a specific strategy (0.0 to 2.0)."""
        if strategy_name not in self.strategy_stats:
            return 1.0
        stats = self.strategy_stats[strategy_name]
        if stats.total_trades < 3:
            return 1.0
        
        # Weight based on confidence and profit factor
        weight = stats.confidence_score
        if stats.profit_factor > 1.5:
            weight *= 1.3
        elif stats.profit_factor < 0.8:
            weight *= 0.6
        
        return round(min(2.0, max(0.1, weight)), 3)

    def should_avoid_trade(self, opp: Dict, market_context: Dict = None) -> Tuple[bool, str]:
        """Check if learned rules suggest avoiding this trade."""
        ctx = market_context or {}
        
        # Check worst strategies
        worst = self.get_worst_strategies()
        strategies = opp.get("strategies", [])
        if strategies and all(s in worst for s in strategies):
            return True, f"All strategies ({', '.join(strategies)}) have poor track record"
        
        # Check worst hours
        hour = datetime.now().hour
        worst_hours = self.get_worst_entry_hours()
        if hour in worst_hours:
            return True, f"Hour {hour} historically unprofitable"
        
        # Check if regime is bad for these strategies
        regime = ctx.get("regime", "unknown")
        regime_weights = self.get_regime_weights()
        if regime in regime_weights and regime_weights[regime] < 0.3:
            return True, f"Regime '{regime}' has poor win rate"
        
        # Check score threshold
        learned_min = self.get_learned_min_score()
        if opp.get("score", 0) < learned_min:
            return True, f"Score {opp.get('score', 0)} below learned threshold {learned_min}"
        
        return False, ""

    # ── Summary ──────────────────────────────────────────────────────────────

    def get_learning_summary(self) -> Dict:
        """Get a comprehensive summary of what the bot has learned."""
        closed_trades = [t for t in self.trades if t.status != "OPEN"]
        if not closed_trades:
            return {"status": "No closed trades yet, still learning..."}
        
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl < 0]
        
        mistakes = defaultdict(int)
        for t in closed_trades:
            if t.mistake_type:
                mistakes[t.mistake_type] += 1
        
        return {
            "total_trades": len(closed_trades),
            "total_pnl": round(sum(t.pnl for t in closed_trades), 2),
            "win_rate": round(len(wins) / len(closed_trades) * 100, 1) if closed_trades else 0,
            "avg_win": round(np.mean([t.pnl for t in wins]), 2) if wins else 0,
            "avg_loss": round(np.mean([t.pnl for t in losses]), 2) if losses else 0,
            "best_strategies": self.get_best_strategies()[:5],
            "worst_strategies": self.get_worst_strategies()[:5],
            "best_hours": self.get_best_entry_hours(),
            "worst_hours": self.get_worst_entry_hours(),
            "learned_min_score": self.get_learned_min_score(),
            "optimal_sl_multiplier": self.get_optimal_sl_multiplier(),
            "optimal_target_multiplier": self.get_optimal_target_multiplier(),
            "common_mistakes": dict(mistakes),
            "regime_weights": self.get_regime_weights(),
            "lessons": [t.lesson_learned for t in closed_trades[-10:] if t.lesson_learned],
        }

    def update_learned_rules(self):
        """Recompute and persist all learned rules from trade history."""
        self.learned_rules.update({
            "min_score_threshold": self.get_learned_min_score(),
            "preferred_strategies": self.get_best_strategies()[:10],
            "avoided_strategies": self.get_worst_strategies(),
            "best_entry_hours": self.get_best_entry_hours(),
            "worst_entry_hours": self.get_worst_entry_hours(),
            "regime_weights": self.get_regime_weights(),
            "dynamic_sl_multiplier": self.get_optimal_sl_multiplier(),
            "dynamic_target_multiplier": self.get_optimal_target_multiplier(),
            "strategy_weights": {k: self.get_strategy_weight(k) 
                                 for k in self.strategy_stats.keys()},
            "version": self.learned_rules.get("version", 0) + 1,
            "last_updated": datetime.now().isoformat(),
        })
        self._save_all()
        logger.info(f"🧠 Updated learned rules v{self.learned_rules['version']}")
        return self.learned_rules


# NOTE: Do NOT use a global instance. Each consumer should
# create or receive a TradeJournal instance explicitly to
# avoid dual-state bugs between the global and local copies.
