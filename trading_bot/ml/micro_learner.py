"""
Micro-Learner — Performs incremental learning updates after EACH trade,
rather than waiting for end-of-day batch processing.

Tracks per-{strategy × regime × hour} performance metrics and
updates adaptive parameters in near real-time.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import math

logger = logging.getLogger(__name__)


class MicroLearner:
    """
    Incremental per-trade learning engine.

    After each trade closes, immediately updates:
    1. Strategy × Regime × Hour win rate matrix
    2. Optimal score thresholds per regime
    3. SL/target multiplier adaptation
    4. Strategy weight adjustments
    5. Hourly performance profiles

    Uses Exponential Moving Average (EMA) for smooth, recency-biased updates.
    """

    DECAY_FACTOR = 0.05  # EMA decay — recent trades matter more
    MIN_TRADES_FOR_LEARNING = 5  # Minimum trades before acting on learned data
    PERSISTENCE_FILE = "data/models/micro_learned.json"

    def __init__(self, persistence_dir: str = "."):
        self.base_dir = Path(persistence_dir)
        self.persistence_path = self.base_dir / self.PERSISTENCE_FILE

        # ── Core learning matrices ────────────────────────────────────────
        # Key: (strategy, regime, hour) → metrics dict
        self._performance_matrix: Dict[str, Dict] = defaultdict(
            lambda: {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "ema_win_rate": 0.5,
                "ema_pnl": 0.0,
                "avg_winner": 0.0,
                "avg_loser": 0.0,
                "total_pnl": 0.0,
            }
        )

        # Per-strategy adaptations
        self._strategy_weights: Dict[str, float] = {}

        # Per-hour performance
        self._hourly_performance: Dict[int, Dict] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0}
        )

        # SL/Target observations
        self._sl_observations: List[Dict] = []  # Recent SL-hit trades
        self._target_observations: List[Dict] = []  # Recent target-hit trades

        # Score threshold learning
        self._score_outcomes: List[Tuple[float, bool]] = []  # (score, was_profitable)

        self._total_trades_processed = 0

        # Load persisted state
        self._load()

    # ── Per-Trade Learning ────────────────────────────────────────────────────

    def learn_from_trade(self, trade: Dict):
        """
        Process a single closed trade and update all learning systems.

        Expected trade dict keys:
            symbol, strategy, pnl, entry_price, exit_price,
            entry_time, exit_time, regime, signal_score,
            sl_hit (bool), target_hit (bool), qty
        """
        try:
            strategy = trade.get("strategy", "unknown")
            regime = trade.get("regime", "unknown")
            pnl = float(trade.get("pnl", 0))
            is_win = pnl > 0

            # Extract hour from entry time
            entry_time = trade.get("entry_time", "")
            hour = self._extract_hour(entry_time)

            # ── Update performance matrix ─────────────────────────────────
            key = f"{strategy}|{regime}|{hour}"
            cell = self._performance_matrix[key]

            cell["trades"] += 1
            if is_win:
                cell["wins"] += 1
                cell["avg_winner"] = self._ema(cell["avg_winner"], pnl, self.DECAY_FACTOR)
            else:
                cell["losses"] += 1
                cell["avg_loser"] = self._ema(cell["avg_loser"], abs(pnl), self.DECAY_FACTOR)

            cell["total_pnl"] += pnl

            # EMA win rate
            win_val = 1.0 if is_win else 0.0
            cell["ema_win_rate"] = self._ema(cell["ema_win_rate"], win_val, self.DECAY_FACTOR)

            # EMA P&L
            cell["ema_pnl"] = self._ema(cell["ema_pnl"], pnl, self.DECAY_FACTOR)

            # ── Update strategy weights ───────────────────────────────────
            self._update_strategy_weight(strategy, is_win, pnl)

            # ── Update hourly performance ─────────────────────────────────
            self._hourly_performance[hour]["trades"] += 1
            if is_win:
                self._hourly_performance[hour]["wins"] += 1
            self._hourly_performance[hour]["total_pnl"] += pnl

            # ── SL/Target observations ────────────────────────────────────
            if trade.get("sl_hit"):
                self._sl_observations.append({
                    "pnl": pnl,
                    "sl_pct": trade.get("sl_pct", 0),
                    "regime": regime,
                    "reversed_after": trade.get("reversed_after_exit", False),
                })
                if len(self._sl_observations) > 200:
                    self._sl_observations = self._sl_observations[-200:]

            if trade.get("target_hit"):
                self._target_observations.append({
                    "pnl": pnl,
                    "target_pct": trade.get("target_pct", 0),
                    "regime": regime,
                    "could_have_gained_more": trade.get("max_after_exit", 0) > pnl,
                })
                if len(self._target_observations) > 200:
                    self._target_observations = self._target_observations[-200:]

            # ── Score threshold learning ──────────────────────────────────
            score = trade.get("signal_score", 0)
            if score > 0:
                self._score_outcomes.append((score, is_win))
                if len(self._score_outcomes) > 500:
                    self._score_outcomes = self._score_outcomes[-500:]

            self._total_trades_processed += 1

            # ── Persist every 5 trades ────────────────────────────────────
            if self._total_trades_processed % 5 == 0:
                self._save()

            logger.debug(f"Micro-learned from trade: {strategy} | {regime} | hr={hour} | pnl={pnl:.2f}")

        except Exception as e:
            logger.error(f"Micro-learning error: {e}")

    # ── Query Learned Data ────────────────────────────────────────────────────

    def get_strategy_weight(self, strategy: str) -> float:
        """Get learned weight multiplier for a strategy (0.1 to 2.0)."""
        return self._strategy_weights.get(strategy, 1.0)

    def get_win_rate(self, strategy: str, regime: str, hour: int = -1) -> float:
        """Get EMA win rate for a specific strategy/regime/hour combo."""
        if hour >= 0:
            key = f"{strategy}|{regime}|{hour}"
            cell = self._performance_matrix.get(key)
            if cell and cell["trades"] >= self.MIN_TRADES_FOR_LEARNING:
                return cell["ema_win_rate"]

        # Fall back to strategy+regime (any hour)
        matching = [
            v for k, v in self._performance_matrix.items()
            if k.startswith(f"{strategy}|{regime}|") and v["trades"] >= self.MIN_TRADES_FOR_LEARNING
        ]
        if matching:
            return np.mean([m["ema_win_rate"] for m in matching])

        return 0.5  # Default

    def is_bad_hour(self, hour: int, min_trades: int = 10) -> bool:
        """Check if an hour historically produces losses."""
        perf = self._hourly_performance.get(hour)
        if not perf or perf["trades"] < min_trades:
            return False
        win_rate = perf["wins"] / perf["trades"]
        return win_rate < 0.35 or perf["total_pnl"] < 0

    def get_optimal_score_threshold(self, regime: str = "any") -> float:
        """
        Find the minimum signal score that historically maximizes win rate.
        Uses binary search on score threshold.
        """
        if len(self._score_outcomes) < 20:
            return 35.0  # Default

        outcomes = self._score_outcomes[-300:]

        best_threshold = 35.0
        best_metric = -1.0

        for threshold in range(20, 80, 5):
            qualifying = [(s, w) for s, w in outcomes if s >= threshold]
            if len(qualifying) < 5:
                continue
            wins = sum(1 for _, w in qualifying if w)
            win_rate = wins / len(qualifying)
            # Optimize for: win_rate weighted by number of qualifying trades
            metric = win_rate * math.log(len(qualifying) + 1)
            if metric > best_metric:
                best_metric = metric
                best_threshold = float(threshold)

        return best_threshold

    def get_sl_multiplier_recommendation(self, regime: str = "any") -> float:
        """
        Recommend SL multiplier adjustment based on SL-hit patterns.
        Returns: multiplier (e.g. 1.2 means widen SL by 20%)
        """
        recent_sl = self._sl_observations[-50:]
        if len(recent_sl) < 5:
            return 1.0

        # If many SL-hit trades later reversed → SL is too tight
        reversed_count = sum(1 for obs in recent_sl if obs.get("reversed_after"))
        reversed_pct = reversed_count / len(recent_sl)

        if reversed_pct > 0.5:
            return 1.3  # Widen SL by 30%
        elif reversed_pct > 0.3:
            return 1.15  # Widen by 15%
        elif reversed_pct < 0.1:
            return 0.9  # Can tighten SL slightly
        else:
            return 1.0

    def get_target_multiplier_recommendation(self) -> float:
        """
        Recommend target multiplier based on target-hit patterns.
        Returns: multiplier (e.g. 1.2 means raise targets by 20%)
        """
        recent = self._target_observations[-50:]
        if len(recent) < 5:
            return 1.0

        # If many target-hit trades could have gained more → raise targets
        premature_count = sum(1 for obs in recent if obs.get("could_have_gained_more"))
        premature_pct = premature_count / len(recent)

        if premature_pct > 0.6:
            return 1.3
        elif premature_pct > 0.4:
            return 1.15
        elif premature_pct < 0.2:
            return 0.9  # Targets too ambitious
        else:
            return 1.0

    def get_strategy_regime_heatmap(self) -> Dict[str, Dict[str, float]]:
        """
        Return a strategy × regime heatmap of win rates.
        Useful for the dashboard visualization.
        """
        heatmap: Dict[str, Dict[str, float]] = defaultdict(dict)

        for key, cell in self._performance_matrix.items():
            parts = key.split("|")
            if len(parts) < 2:
                continue
            strategy, regime = parts[0], parts[1]

            if cell["trades"] >= self.MIN_TRADES_FOR_LEARNING:
                if regime not in heatmap[strategy]:
                    heatmap[strategy][regime] = cell["ema_win_rate"]
                else:
                    # Average across hours
                    heatmap[strategy][regime] = (
                        heatmap[strategy][regime] + cell["ema_win_rate"]
                    ) / 2

        return dict(heatmap)

    def get_hourly_heatmap(self) -> Dict[int, Dict[str, float]]:
        """Return hourly performance heatmap."""
        result = {}
        for hour, perf in sorted(self._hourly_performance.items()):
            if perf["trades"] > 0:
                result[hour] = {
                    "trades": perf["trades"],
                    "win_rate": round(perf["wins"] / perf["trades"], 3),
                    "total_pnl": round(perf["total_pnl"], 2),
                    "avg_pnl": round(perf["total_pnl"] / perf["trades"], 2),
                }
        return result

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of all micro-learning state."""
        return {
            "total_trades_processed": self._total_trades_processed,
            "active_strategy_weights": dict(self._strategy_weights),
            "optimal_score_threshold": self.get_optimal_score_threshold(),
            "sl_multiplier_rec": self.get_sl_multiplier_recommendation(),
            "target_multiplier_rec": self.get_target_multiplier_recommendation(),
            "bad_hours": [h for h in range(9, 16) if self.is_bad_hour(h)],
            "matrix_cells": len(self._performance_matrix),
            "hourly_data_points": sum(v["trades"] for v in self._hourly_performance.values()),
        }

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _update_strategy_weight(self, strategy: str, is_win: bool, pnl: float):
        """Update EMA-based strategy weight."""
        current = self._strategy_weights.get(strategy, 1.0)
        adjustment = 0.02 if is_win else -0.02

        # Larger adjustment for bigger wins/losses
        if abs(pnl) > 500:
            adjustment *= 1.5

        new_weight = current + adjustment
        # Clamp between 0.1 and 2.0
        self._strategy_weights[strategy] = max(0.1, min(2.0, new_weight))

    @staticmethod
    def _ema(current: float, new_value: float, alpha: float) -> float:
        """Exponential Moving Average update."""
        return current * (1 - alpha) + new_value * alpha

    @staticmethod
    def _extract_hour(time_str) -> int:
        """Extract hour from various time formats."""
        if isinstance(time_str, str):
            for fmt in ("%H:%M:%S", "%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(time_str.strip()[:19], fmt).hour
                except ValueError:
                    continue
        elif isinstance(time_str, datetime):
            return time_str.hour

        return datetime.now().hour  # Fallback

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        """Save learned state to disk."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": 1,
                "total_trades": self._total_trades_processed,
                "strategy_weights": dict(self._strategy_weights),
                "hourly_performance": {
                    str(k): v for k, v in self._hourly_performance.items()
                },
                "performance_matrix": dict(self._performance_matrix),
                "score_outcomes": self._score_outcomes[-200:],
                "timestamp": datetime.now().isoformat(),
            }

            # Atomic write
            tmp_path = self.persistence_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(self.persistence_path)

        except Exception as e:
            logger.error(f"Error saving micro-learned data: {e}")

    def _load(self):
        """Load persisted state from disk."""
        try:
            if not self.persistence_path.exists():
                return

            with open(self.persistence_path, "r") as f:
                data = json.load(f)

            self._total_trades_processed = data.get("total_trades", 0)
            self._strategy_weights = data.get("strategy_weights", {})

            # Hourly performance
            for k, v in data.get("hourly_performance", {}).items():
                self._hourly_performance[int(k)] = v

            # Performance matrix
            for k, v in data.get("performance_matrix", {}).items():
                self._performance_matrix[k] = v

            # Score outcomes
            self._score_outcomes = [
                (s, w) for s, w in data.get("score_outcomes", [])
            ]

            logger.info(
                f"Loaded micro-learned state: {self._total_trades_processed} trades, "
                f"{len(self._performance_matrix)} matrix cells"
            )

        except Exception as e:
            logger.error(f"Error loading micro-learned data: {e}")
