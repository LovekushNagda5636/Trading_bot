"""
Multi-Timeframe Confluence Analyzer — Confirms signals across 5m, 15m,
and 1h candle timeframes to improve entry quality and reduce false signals.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """
    Performs multi-timeframe analysis (MTF) to validate trading signals.

    Principle: Only enter when the lower timeframe signal aligns with the
    higher timeframe trend. E.g. BUY on 5m only if 15m is also bullish.

    Timeframes analyzed:
        - 5m  (execution timeframe — entry/exit timing)
        - 15m (confirmation timeframe — trend direction)
        - 1h  (context timeframe — overall bias)
    """

    TIMEFRAMES = ["5m", "15m", "1h"]

    # Minimum confluence thresholds
    MIN_CONFLUENCE = 0.60  # At least 60% of timeframes must agree
    STRONG_CONFLUENCE = 0.80  # 80%+ = high-confidence entry

    def __init__(self, candle_manager=None):
        """
        Args:
            candle_manager: Instance of CandleManager for fetching indicators
        """
        self.candle_manager = candle_manager

    def set_candle_manager(self, candle_manager):
        self.candle_manager = candle_manager

    def analyze(
        self,
        symbol: str,
        signal_direction: str = "BUY",
    ) -> Dict[str, Any]:
        """
        Perform multi-timeframe analysis for a symbol.

        Args:
            symbol: Stock symbol
            signal_direction: 'BUY' or 'SELL' — the proposed trade direction

        Returns:
            Dict with confluence_score, recommendation, and per-timeframe details
        """
        if not self.candle_manager:
            return {
                "confluence_score": 0.5,
                "recommendation": "NO_DATA",
                "details": {},
            }

        is_buy = signal_direction.upper() == "BUY"

        tf_results = {}
        for tf in self.TIMEFRAMES:
            indicators = self.candle_manager.compute_all_indicators(symbol, tf)
            if not indicators:
                tf_results[tf] = {
                    "available": False,
                    "trend": "unknown",
                    "strength": 0.0,
                    "votes": [],
                }
                continue

            trend_info = self._assess_timeframe(indicators, is_buy)
            tf_results[tf] = trend_info

        # ── Compute confluence ─────────────────────────────────────────────
        available_tfs = [
            tf for tf in self.TIMEFRAMES if tf_results[tf].get("available", False)
        ]

        if not available_tfs:
            return {
                "confluence_score": 0.5,
                "recommendation": "INSUFFICIENT_DATA",
                "details": tf_results,
            }

        # Weighted confluence: higher timeframes matter more for trend
        tf_weights = {"5m": 0.30, "15m": 0.40, "1h": 0.30}
        weighted_sum = 0.0
        weight_total = 0.0

        for tf in available_tfs:
            w = tf_weights.get(tf, 0.33)
            strength = tf_results[tf].get("strength", 0.0)
            weighted_sum += strength * w
            weight_total += w

        confluence = weighted_sum / weight_total if weight_total > 0 else 0.5

        # ── Recommendation ─────────────────────────────────────────────────
        if confluence >= self.STRONG_CONFLUENCE:
            recommendation = "STRONG_CONFLUENCE"
        elif confluence >= self.MIN_CONFLUENCE:
            recommendation = "CONFIRMED"
        elif confluence >= 0.40:
            recommendation = "WEAK"
        else:
            recommendation = "DIVERGENT"

        # ── Check for critical divergences ─────────────────────────────────
        # If 1h trend is opposite to signal, flag it
        higher_tf = tf_results.get("1h", {}) or tf_results.get("15m", {})
        if higher_tf.get("available"):
            higher_trend = higher_tf.get("trend", "neutral")
            if is_buy and higher_trend == "bearish":
                recommendation = "COUNTER_TREND"
                confluence *= 0.7  # Penalty
            elif not is_buy and higher_trend == "bullish":
                recommendation = "COUNTER_TREND"
                confluence *= 0.7

        return {
            "confluence_score": round(confluence, 4),
            "recommendation": recommendation,
            "direction": signal_direction,
            "details": tf_results,
            "available_timeframes": len(available_tfs),
            "timestamp": datetime.now().isoformat(),
        }

    def _assess_timeframe(self, indicators: Dict, is_buy: bool) -> Dict:
        """
        Assess a single timeframe's trend alignment with the proposed direction.

        Returns strength in [0, 1]: 1.0 = perfectly aligned, 0.0 = fully opposed.
        """
        votes = []

        # ── RSI alignment ──────────────────────────────────────────────────
        rsi = indicators.get("rsi_14")
        if rsi is not None:
            if is_buy:
                # For BUY: RSI < 70 is good, < 30 is great (oversold)
                if rsi < 30:
                    votes.append(("rsi", 1.0, "oversold"))
                elif rsi < 50:
                    votes.append(("rsi", 0.7, "below_midline"))
                elif rsi < 70:
                    votes.append(("rsi", 0.4, "above_midline"))
                else:
                    votes.append(("rsi", 0.1, "overbought"))
            else:
                if rsi > 70:
                    votes.append(("rsi", 1.0, "overbought"))
                elif rsi > 50:
                    votes.append(("rsi", 0.7, "above_midline"))
                elif rsi > 30:
                    votes.append(("rsi", 0.4, "below_midline"))
                else:
                    votes.append(("rsi", 0.1, "oversold"))

        # ── EMA alignment ──────────────────────────────────────────────────
        ema_align = indicators.get("ema_alignment")
        if ema_align is not None:
            if is_buy:
                votes.append(
                    ("ema", 0.9 if ema_align > 0 else 0.2, "bullish" if ema_align > 0 else "bearish")
                )
            else:
                votes.append(
                    ("ema", 0.9 if ema_align < 0 else 0.2, "bearish" if ema_align < 0 else "bullish")
                )

        # ── MACD alignment ─────────────────────────────────────────────────
        macd_bull = indicators.get("macd_bullish")
        if macd_bull is not None:
            aligned = (is_buy and macd_bull > 0.5) or (not is_buy and macd_bull < 0.5)
            votes.append(("macd", 0.85 if aligned else 0.15, "aligned" if aligned else "opposed"))

        # ── SuperTrend alignment ───────────────────────────────────────────
        st = indicators.get("supertrend_trend")
        if st is not None:
            aligned = (is_buy and st > 0) or (not is_buy and st < 0)
            votes.append(("supertrend", 0.9 if aligned else 0.1, "aligned" if aligned else "opposed"))

        # ── VWAP alignment ─────────────────────────────────────────────────
        vwap_dev = indicators.get("vwap_deviation")
        if vwap_dev is not None:
            if is_buy:
                # Buying near VWAP (small negative deviation) or above is fine
                if vwap_dev > 0.2:
                    votes.append(("vwap", 0.7, "above"))
                elif vwap_dev > -0.3:
                    votes.append(("vwap", 0.8, "near"))
                else:
                    votes.append(("vwap", 0.4, "below"))
            else:
                if vwap_dev < -0.2:
                    votes.append(("vwap", 0.7, "below"))
                elif vwap_dev < 0.3:
                    votes.append(("vwap", 0.8, "near"))
                else:
                    votes.append(("vwap", 0.4, "above"))

        # ── Bollinger Band position ────────────────────────────────────────
        bb_pos = indicators.get("bb_position")
        if bb_pos is not None:
            if is_buy:
                if bb_pos < 0.2:
                    votes.append(("bb", 0.9, "near_lower"))
                elif bb_pos < 0.5:
                    votes.append(("bb", 0.6, "lower_half"))
                else:
                    votes.append(("bb", 0.3, "upper_half"))
            else:
                if bb_pos > 0.8:
                    votes.append(("bb", 0.9, "near_upper"))
                elif bb_pos > 0.5:
                    votes.append(("bb", 0.6, "upper_half"))
                else:
                    votes.append(("bb", 0.3, "lower_half"))

        if not votes:
            return {
                "available": False,
                "trend": "unknown",
                "strength": 0.0,
                "votes": [],
            }

        # ── Aggregate ──────────────────────────────────────────────────────
        avg_strength = sum(v[1] for v in votes) / len(votes)

        if avg_strength >= 0.65:
            trend = "bullish" if is_buy else "bearish"
        elif avg_strength <= 0.35:
            trend = "bearish" if is_buy else "bullish"
        else:
            trend = "neutral"

        return {
            "available": True,
            "trend": trend,
            "strength": round(avg_strength, 4),
            "votes": [(v[0], round(v[1], 2), v[2]) for v in votes],
        }

    def should_enter(
        self, symbol: str, signal_direction: str = "BUY"
    ) -> bool:
        """
        Quick check: should we enter based on MTF confluence?
        Returns True only if confluence meets minimum threshold.
        """
        result = self.analyze(symbol, signal_direction)
        return (
            result["confluence_score"] >= self.MIN_CONFLUENCE
            and result["recommendation"] not in ("DIVERGENT", "COUNTER_TREND", "NO_DATA")
        )

    def get_entry_quality(self, symbol: str, signal_direction: str = "BUY") -> str:
        """
        Get a human-readable entry quality label.
        """
        result = self.analyze(symbol, signal_direction)
        score = result["confluence_score"]

        if score >= 0.85:
            return "EXCELLENT"
        elif score >= 0.70:
            return "GOOD"
        elif score >= 0.55:
            return "FAIR"
        elif score >= 0.40:
            return "POOR"
        else:
            return "AVOID"
