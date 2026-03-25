"""
Ensemble Scorer — Combines rule-based strategy scores with ML model
predictions and RL agent Q-values into a unified confidence signal.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class EnsembleScorer:
    """
    Combines multiple signal sources into a single confidence score.

    Sources:
        1. Rule-based strategy score (from existing StrategyEngine)
        2. ML model probability (from ModelTrainer / sklearn models)
        3. Regime alignment bonus (from MarketRegimeDetector)
        4. Candle indicator confluence (from CandleManager)
        5. Journal-learned weight adjustments (from AdaptiveStrategyEngine)

    Output:
        A single float in [0, 100] representing overall confidence.
    """

    # Weight presets — adaptive weights are updated by the learning loop
    DEFAULT_WEIGHTS = {
        "rule_based": 0.35,
        "ml_model": 0.25,
        "indicator_confluence": 0.20,
        "regime_alignment": 0.10,
        "journal_learned": 0.10,
    }

    def __init__(self):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self._scores_history: List[Dict] = []
        self._performance_by_source: Dict[str, Dict] = {
            source: {"correct": 0, "total": 0} for source in self.DEFAULT_WEIGHTS
        }

    def score_opportunity(
        self,
        *,
        rule_score: float = 0.0,
        ml_probability: Optional[float] = None,
        candle_indicators: Optional[Dict] = None,
        regime: Optional[str] = None,
        strategy_name: Optional[str] = None,
        journal_adjustment: float = 0.0,
        ltp: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compute ensemble confidence for a trading opportunity.

        Args:
            rule_score: Raw score from StrategyEngine (0-100)
            ml_probability: Predicted probability of profitable trade (0.0-1.0)
            candle_indicators: Dict of real indicator values from CandleManager
            regime: Current market regime string
            strategy_name: Name of the strategy generating this signal
            journal_adjustment: Learned weight modifier from adaptive engine
            ltp: Last traded price

        Returns:
            Dict with 'ensemble_score', 'confidence', 'components', 'recommendation'
        """
        components = {}

        # ── 1. Rule-based score (already 0-100) ────────────────────────────
        rule_normalized = max(0, min(rule_score, 100))
        components["rule_based"] = rule_normalized

        # ── 2. ML Model probability ────────────────────────────────────────
        if ml_probability is not None:
            ml_score = ml_probability * 100
        else:
            ml_score = 50.0  # Neutral (no model available)
        components["ml_model"] = ml_score

        # ── 3. Candle indicator confluence ─────────────────────────────────
        confluence_score = self._compute_confluence(candle_indicators, ltp)
        components["indicator_confluence"] = confluence_score

        # ── 4. Regime alignment ────────────────────────────────────────────
        regime_score = self._compute_regime_alignment(regime, strategy_name)
        components["regime_alignment"] = regime_score

        # ── 5. Journal learned adjustment ──────────────────────────────────
        # journal_adjustment is a multiplier (e.g. 1.2 → +20%)
        journal_score = 50.0 + (journal_adjustment * 50.0)
        journal_score = max(0, min(journal_score, 100))
        components["journal_learned"] = journal_score

        # ── Weighted ensemble ──────────────────────────────────────────────
        ensemble = 0.0
        for source, weight in self.weights.items():
            ensemble += components.get(source, 50.0) * weight

        ensemble = max(0, min(ensemble, 100))

        # ── Confidence level ───────────────────────────────────────────────
        if ensemble >= 75:
            confidence = "HIGH"
        elif ensemble >= 55:
            confidence = "MEDIUM"
        elif ensemble >= 40:
            confidence = "LOW"
        else:
            confidence = "VERY_LOW"

        # ── Recommendation ─────────────────────────────────────────────────
        if ensemble >= 65 and confidence in ("HIGH", "MEDIUM"):
            recommendation = "STRONG_ENTRY"
        elif ensemble >= 50:
            recommendation = "ENTER"
        elif ensemble >= 35:
            recommendation = "WATCH"
        else:
            recommendation = "SKIP"

        result = {
            "ensemble_score": round(ensemble, 2),
            "confidence": confidence,
            "recommendation": recommendation,
            "components": {k: round(v, 2) for k, v in components.items()},
            "weights": dict(self.weights),
            "timestamp": datetime.now().isoformat(),
        }

        # Store for learning
        self._scores_history.append(result)
        if len(self._scores_history) > 5000:
            self._scores_history = self._scores_history[-5000:]

        return result

    def _compute_confluence(self, indicators: Optional[Dict], ltp: float) -> float:
        """
        Compute indicator confluence score from real candle indicators.
        Each indicator casts a bullish/bearish vote; confluence measures agreement.
        """
        if not indicators or ltp <= 0:
            return 50.0  # Neutral

        votes = []

        # RSI vote
        rsi = indicators.get("rsi_14")
        if rsi is not None:
            if rsi < 30:
                votes.append(("rsi_oversold", 1.0))  # Bullish
            elif rsi > 70:
                votes.append(("rsi_overbought", -1.0))  # Bearish
            elif rsi < 45:
                votes.append(("rsi_weak_bull", 0.4))
            elif rsi > 55:
                votes.append(("rsi_weak_bear", -0.4))
            else:
                votes.append(("rsi_neutral", 0.0))

        # MACD vote
        macd_bull = indicators.get("macd_bullish")
        if macd_bull is not None:
            votes.append(("macd", 0.8 if macd_bull > 0.5 else -0.8))

        histogram = indicators.get("macd_histogram")
        if histogram is not None:
            if histogram > 0:
                votes.append(("macd_hist", min(histogram * 100, 1.0)))
            else:
                votes.append(("macd_hist", max(histogram * 100, -1.0)))

        # VWAP vote
        vwap_dev = indicators.get("vwap_deviation")
        if vwap_dev is not None:
            if vwap_dev > 0.5:
                votes.append(("vwap_above", 0.6))
            elif vwap_dev < -0.5:
                votes.append(("vwap_below", -0.6))
            else:
                votes.append(("vwap_near", 0.0))

        # EMA alignment
        ema_alignment = indicators.get("ema_alignment")
        if ema_alignment is not None:
            votes.append(("ema", 0.7 * ema_alignment))

        # SuperTrend
        st_trend = indicators.get("supertrend_trend")
        if st_trend is not None:
            votes.append(("supertrend", 0.8 * st_trend))

        # Bollinger Band position
        bb_pos = indicators.get("bb_position")
        if bb_pos is not None:
            if bb_pos < 0.15:
                votes.append(("bb_low", 0.7))  # Near lower band → bullish
            elif bb_pos > 0.85:
                votes.append(("bb_high", -0.7))  # Near upper band → bearish
            else:
                votes.append(("bb_mid", 0.0))

        if not votes:
            return 50.0

        # Compute agreement score
        vote_values = [v[1] for v in votes]
        avg_vote = np.mean(vote_values)   # -1 to +1
        agreement = np.std(vote_values)    # Lower std = more agreement

        # Convert to 0-100 scale
        # avg_vote in [-1, 1] → score in [0, 100]
        base_score = (avg_vote + 1) / 2 * 100

        # Boost score when indicators agree (low std)
        agreement_bonus = max(0, (0.5 - agreement)) * 20  # Up to +10 bonus

        score = base_score + agreement_bonus
        return max(0, min(score, 100))

    def _compute_regime_alignment(self, regime: Optional[str], strategy: Optional[str]) -> float:
        """
        Score how well the strategy aligns with the current market regime.
        """
        if not regime or not strategy:
            return 50.0

        regime = regime.lower() if regime else ""
        strategy = strategy.lower() if strategy else ""

        # Strategy-regime compatibility matrix
        GOOD_COMBOS = {
            ("trending", "momentum"): 85,
            ("trending", "trend"): 85,
            ("trending", "breakout"): 80,
            ("trending_bull", "momentum"): 90,
            ("trending_bull", "breakout"): 85,
            ("trending_bear", "reversal"): 75,
            ("sideways", "reversion"): 85,
            ("sideways", "mean_reversion"): 85,
            ("sideways", "range"): 80,
            ("volatile", "volatility"): 80,
            ("high_volatility", "gap"): 75,
            ("low_volatility", "narrow_range"): 80,
        }

        BAD_COMBOS = {
            ("trending", "reversion"): 25,
            ("trending", "mean_reversion"): 25,
            ("sideways", "momentum"): 30,
            ("sideways", "breakout"): 30,
            ("high_volatility", "narrow_range"): 20,
            ("low_volatility", "breakout"): 30,
        }

        # Check for matches (fuzzy: check if regime/strategy key is contained)
        for (r, s), score in GOOD_COMBOS.items():
            if r in regime and s in strategy:
                return float(score)

        for (r, s), score in BAD_COMBOS.items():
            if r in regime and s in strategy:
                return float(score)

        return 50.0  # Neutral

    # ── Learning / Weight Adaptation ──────────────────────────────────────────

    def record_outcome(self, scored_signal: Dict, pnl: float):
        """
        Record trade outcome to learn optimal weights.
        Called when a trade that was scored by this ensemble closes.
        """
        try:
            is_profitable = pnl > 0
            components = scored_signal.get("components", {})

            for source in self.DEFAULT_WEIGHTS:
                source_score = components.get(source, 50.0)
                predicted_good = source_score >= 55.0  # Above neutral

                self._performance_by_source[source]["total"] += 1
                if predicted_good == is_profitable:
                    self._performance_by_source[source]["correct"] += 1

        except Exception as e:
            logger.error(f"Error recording ensemble outcome: {e}")

    def adapt_weights(self, min_samples: int = 30):
        """
        Re-compute source weights based on each source's historical accuracy.
        Called periodically (e.g. end of day).
        """
        try:
            accuracies = {}
            for source, perf in self._performance_by_source.items():
                total = perf["total"]
                if total >= min_samples:
                    acc = perf["correct"] / total
                    accuracies[source] = acc
                else:
                    accuracies[source] = 0.5  # Default neutral

            # Convert accuracies to weights (softmax-like)
            values = np.array(list(accuracies.values()))
            if values.max() - values.min() < 0.01:
                return  # All sources equally good, keep defaults

            # Shift to positive, then normalize
            shifted = values - values.min() + 0.1
            normalized = shifted / shifted.sum()

            for i, source in enumerate(accuracies.keys()):
                old_w = self.weights.get(source, 0)
                new_w = float(normalized[i])
                # Smooth update (don't swing weights too drastically)
                self.weights[source] = old_w * 0.7 + new_w * 0.3

            logger.info(f"Ensemble weights updated: {self.weights}")

        except Exception as e:
            logger.error(f"Error adapting ensemble weights: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get ensemble scorer performance summary."""
        return {
            "weights": dict(self.weights),
            "performance_by_source": dict(self._performance_by_source),
            "total_scored": len(self._scores_history),
            "recent_avg_score": (
                round(np.mean([s["ensemble_score"] for s in self._scores_history[-50:]]), 2)
                if self._scores_history
                else 0
            ),
        }
