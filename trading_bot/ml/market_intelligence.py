"""
Market Intelligence System - AI that understands market patterns and dynamics.
Continuously learns from market data and trading outcomes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import structlog
from dataclasses import dataclass, field
import json
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from ..core.models import MarketTick, OHLC, TradingSignal, TradeDecision, Position
from .feature_engineering import FeatureEngineer, MarketFeatures

logger = structlog.get_logger(__name__)


@dataclass
class MarketRegime:
    """Represents a market regime/state."""
    regime_id: str
    name: str
    characteristics: Dict[str, float]
    volatility_range: Tuple[float, float]
    trend_strength: float
    volume_profile: str  # "high", "medium", "low"
    optimal_strategies: List[str]
    risk_level: str  # "low", "medium", "high"
    duration_hours: float = 0.0
    frequency: float = 0.0  # How often this regime occurs


@dataclass
class MarketPattern:
    """Represents a discovered market pattern."""
    pattern_id: str
    name: str
    description: str
    preconditions: Dict[str, Any]
    expected_outcome: str  # "bullish", "bearish", "neutral"
    confidence: float
    success_rate: float
    avg_duration: float
    risk_reward_ratio: float
    occurrences: int = 0


@dataclass
class TradingInsight:
    """AI-generated trading insight."""
    insight_id: str
    type: str  # "pattern", "anomaly", "regime_change", "opportunity"
    symbol: str
    description: str
    confidence: float
    expected_impact: str  # "positive", "negative", "neutral"
    time_horizon: str  # "immediate", "short_term", "medium_term"
    supporting_evidence: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class MarketIntelligence:
    """
    AI-powered market intelligence system that learns market patterns,
    detects anomalies, and provides trading insights.
    """
    
    def __init__(self, feature_engineer: FeatureEngineer):
        self.feature_engineer = feature_engineer
        
        # Market understanding
        self.market_regimes: Dict[str, MarketRegime] = {}
        self.discovered_patterns: Dict[str, MarketPattern] = {}
        self.current_regime: Optional[str] = None
        
        # Learning components
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.regime_classifier = KMeans(n_clusters=5, random_state=42)
        self.scaler = StandardScaler()
        
        # Data storage
        self.market_history: List[Dict] = []
        self.trading_outcomes: List[Dict] = []
        self.insights_history: List[TradingInsight] = []
        
        # Learning state
        self.is_trained = False
        self.last_training_time = None
        self.training_data_size = 0
        
        # Initialize default regimes
        self._initialize_default_regimes()
        
        logger.info("Market Intelligence system initialized")
    
    def _initialize_default_regimes(self):
        """Initialize default market regimes."""
        self.market_regimes = {
            "trending_bull": MarketRegime(
                regime_id="trending_bull",
                name="Trending Bull Market",
                characteristics={
                    "trend_strength": 0.8,
                    "volatility": 0.15,
                    "volume_growth": 1.2,
                    "momentum": 0.7
                },
                volatility_range=(0.1, 0.25),
                trend_strength=0.8,
                volume_profile="high",
                optimal_strategies=["momentum", "trend_following"],
                risk_level="medium"
            ),
            "trending_bear": MarketRegime(
                regime_id="trending_bear",
                name="Trending Bear Market",
                characteristics={
                    "trend_strength": -0.8,
                    "volatility": 0.25,
                    "volume_growth": 1.1,
                    "momentum": -0.7
                },
                volatility_range=(0.15, 0.4),
                trend_strength=-0.8,
                volume_profile="medium",
                optimal_strategies=["short_selling", "defensive"],
                risk_level="high"
            ),
            "sideways": MarketRegime(
                regime_id="sideways",
                name="Sideways/Range-bound Market",
                characteristics={
                    "trend_strength": 0.1,
                    "volatility": 0.12,
                    "volume_growth": 0.9,
                    "momentum": 0.0
                },
                volatility_range=(0.08, 0.18),
                trend_strength=0.1,
                volume_profile="low",
                optimal_strategies=["mean_reversion", "range_trading"],
                risk_level="low"
            ),
            "high_volatility": MarketRegime(
                regime_id="high_volatility",
                name="High Volatility Market",
                characteristics={
                    "trend_strength": 0.3,
                    "volatility": 0.35,
                    "volume_growth": 1.5,
                    "momentum": 0.2
                },
                volatility_range=(0.3, 0.6),
                trend_strength=0.3,
                volume_profile="high",
                optimal_strategies=["volatility_trading", "options"],
                risk_level="high"
            ),
            "low_volatility": MarketRegime(
                regime_id="low_volatility",
                name="Low Volatility Market",
                characteristics={
                    "trend_strength": 0.2,
                    "volatility": 0.08,
                    "volume_growth": 0.8,
                    "momentum": 0.1
                },
                volatility_range=(0.05, 0.12),
                trend_strength=0.2,
                volume_profile="low",
                optimal_strategies=["carry_trade", "low_risk"],
                risk_level="low"
            )
        }
    
    def update_market_data(self, symbol: str, tick: MarketTick, features: MarketFeatures):
        """Update market intelligence with new data."""
        try:
            # Store market data point
            data_point = {
                'timestamp': tick.timestamp,
                'symbol': symbol,
                'price': float(tick.price),
                'volume': tick.volume,
                'features': features,
                'regime': self.current_regime
            }
            
            self.market_history.append(data_point)
            
            # Limit history size
            if len(self.market_history) > 10000:
                self.market_history = self.market_history[-10000:]
            
            # Detect current regime
            self._detect_current_regime(symbol, features)
            
            # Look for patterns
            self._detect_patterns(symbol, features)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(symbol, features)
            
            # Generate insights
            if anomalies or self._regime_changed():
                insights = self._generate_insights(symbol, features, anomalies)
                self.insights_history.extend(insights)
            
        except Exception as e:
            logger.error(f"Error updating market intelligence: {e}")
    
    def _detect_current_regime(self, symbol: str, features: MarketFeatures):
        """Detect current market regime."""
        try:
            # Extract regime indicators
            volatility = features.volatility_features.get("volatility_20d", 0.2)
            trend_strength = features.regime_features.get("trend_regime", 0)
            volume_ratio = features.volume_features.get("volume_ratio_20", 1.0)
            momentum = features.price_features.get("momentum_10d", 0)
            
            # Simple rule-based regime detection
            if volatility > 0.3:
                new_regime = "high_volatility"
            elif volatility < 0.1:
                new_regime = "low_volatility"
            elif trend_strength > 0.7 and momentum > 0.02:
                new_regime = "trending_bull"
            elif trend_strength > 0.7 and momentum < -0.02:
                new_regime = "trending_bear"
            else:
                new_regime = "sideways"
            
            if new_regime != self.current_regime:
                logger.info(f"Market regime changed from {self.current_regime} to {new_regime}")
                self.current_regime = new_regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
    
    def _detect_patterns(self, symbol: str, features: MarketFeatures):
        """Detect recurring market patterns."""
        try:
            # Look for common patterns
            patterns_found = []
            
            # RSI Divergence Pattern
            rsi = features.technical_features.get("rsi", 50)
            price_momentum = features.price_features.get("momentum_5d", 0)
            
            if rsi > 70 and price_momentum < 0:
                patterns_found.append("bearish_rsi_divergence")
            elif rsi < 30 and price_momentum > 0:
                patterns_found.append("bullish_rsi_divergence")
            
            # Bollinger Band Squeeze
            bb_squeeze = features.technical_features.get("bb_squeeze", 0)
            if bb_squeeze < 0.02:  # Very narrow bands
                patterns_found.append("bollinger_squeeze")
            
            # Volume Breakout Pattern
            volume_ratio = features.volume_features.get("volume_ratio_5", 1.0)
            price_change = features.price_features.get("return_1d", 0)
            
            if volume_ratio > 2.0 and abs(price_change) > 0.02:
                if price_change > 0:
                    patterns_found.append("bullish_volume_breakout")
                else:
                    patterns_found.append("bearish_volume_breakout")
            
            # Update pattern statistics
            for pattern_name in patterns_found:
                if pattern_name not in self.discovered_patterns:
                    self.discovered_patterns[pattern_name] = MarketPattern(
                        pattern_id=pattern_name,
                        name=pattern_name.replace("_", " ").title(),
                        description=f"Detected {pattern_name} pattern",
                        preconditions={},
                        expected_outcome="neutral",
                        confidence=0.5,
                        success_rate=0.5,
                        avg_duration=4.0,
                        risk_reward_ratio=1.0
                    )
                
                self.discovered_patterns[pattern_name].occurrences += 1
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
    
    def _detect_anomalies(self, symbol: str, features: MarketFeatures) -> List[str]:
        """Detect market anomalies."""
        try:
            anomalies = []
            
            # Price anomalies
            price_percentile = features.price_features.get("price_percentile_100d", 0.5)
            if price_percentile > 0.95:
                anomalies.append("price_at_100d_high")
            elif price_percentile < 0.05:
                anomalies.append("price_at_100d_low")
            
            # Volume anomalies
            volume_ratio = features.volume_features.get("volume_ratio_20", 1.0)
            if volume_ratio > 5.0:
                anomalies.append("extreme_volume_spike")
            elif volume_ratio < 0.2:
                anomalies.append("extremely_low_volume")
            
            # Volatility anomalies
            vol_clustering = features.volatility_features.get("vol_clustering", 1.0)
            if vol_clustering > 3.0:
                anomalies.append("volatility_spike")
            
            # Microstructure anomalies
            spread = features.microstructure_features.get("avg_spread", 0.001)
            if spread > 0.01:  # 1% spread is unusual
                anomalies.append("wide_bid_ask_spread")
            
            order_imbalance = features.microstructure_features.get("order_imbalance", 0)
            if abs(order_imbalance) > 0.8:
                anomalies.append("extreme_order_imbalance")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _regime_changed(self) -> bool:
        """Check if market regime recently changed."""
        if len(self.market_history) < 2:
            return False
        
        current_regime = self.market_history[-1].get('regime')
        previous_regime = self.market_history[-2].get('regime')
        
        return current_regime != previous_regime
    
    def _generate_insights(self, symbol: str, features: MarketFeatures, anomalies: List[str]) -> List[TradingInsight]:
        """Generate AI trading insights."""
        insights = []
        
        try:
            # Regime change insights
            if self._regime_changed():
                insight = TradingInsight(
                    insight_id=f"regime_change_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    type="regime_change",
                    symbol=symbol,
                    description=f"Market regime changed to {self.current_regime}",
                    confidence=0.8,
                    expected_impact="neutral",
                    time_horizon="short_term",
                    supporting_evidence=[f"Regime indicators suggest {self.current_regime}"]
                )
                insights.append(insight)
            
            # Anomaly insights
            for anomaly in anomalies:
                if anomaly == "price_at_100d_high":
                    insight = TradingInsight(
                        insight_id=f"anomaly_{anomaly}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        type="anomaly",
                        symbol=symbol,
                        description="Price reached 100-day high - potential reversal or continuation",
                        confidence=0.7,
                        expected_impact="neutral",
                        time_horizon="immediate",
                        supporting_evidence=["Price at 100-day percentile > 95%"]
                    )
                    insights.append(insight)
                
                elif anomaly == "extreme_volume_spike":
                    insight = TradingInsight(
                        insight_id=f"anomaly_{anomaly}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        type="anomaly",
                        symbol=symbol,
                        description="Extreme volume spike detected - significant news or event likely",
                        confidence=0.9,
                        expected_impact="positive",
                        time_horizon="immediate",
                        supporting_evidence=["Volume ratio > 5x average"]
                    )
                    insights.append(insight)
            
            # Pattern-based insights
            for pattern_name, pattern in self.discovered_patterns.items():
                if pattern.occurrences > 0:  # Recently detected
                    insight = TradingInsight(
                        insight_id=f"pattern_{pattern_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        type="pattern",
                        symbol=symbol,
                        description=f"Detected {pattern.name} pattern",
                        confidence=pattern.confidence,
                        expected_impact=pattern.expected_outcome,
                        time_horizon="short_term",
                        supporting_evidence=[f"Pattern success rate: {pattern.success_rate:.1%}"]
                    )
                    insights.append(insight)
            
            # Opportunity insights based on current regime
            if self.current_regime in self.market_regimes:
                regime = self.market_regimes[self.current_regime]
                if regime.optimal_strategies:
                    insight = TradingInsight(
                        insight_id=f"opportunity_{self.current_regime}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        type="opportunity",
                        symbol=symbol,
                        description=f"Current regime favors {', '.join(regime.optimal_strategies)} strategies",
                        confidence=0.6,
                        expected_impact="positive",
                        time_horizon="medium_term",
                        supporting_evidence=[f"Market in {regime.name} regime"]
                    )
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights
    
    def learn_from_trade_outcome(self, trade_decision: TradeDecision, outcome: Dict[str, Any]):
        """Learn from trading outcomes to improve intelligence."""
        try:
            # Store trading outcome
            learning_data = {
                'timestamp': datetime.now(),
                'symbol': trade_decision.symbol,
                'action': trade_decision.action.value,
                'quantity': trade_decision.quantity,
                'strategy_id': trade_decision.strategy_id,
                'outcome': outcome,
                'regime': self.current_regime,
                'market_conditions': self._get_current_market_conditions(trade_decision.symbol)
            }
            
            self.trading_outcomes.append(learning_data)
            
            # Update pattern success rates
            self._update_pattern_performance(outcome)
            
            # Update regime effectiveness
            self._update_regime_performance(outcome)
            
            # Trigger retraining if enough new data
            if len(self.trading_outcomes) % 100 == 0:
                self._retrain_models()
            
        except Exception as e:
            logger.error(f"Error learning from trade outcome: {e}")
    
    def _get_current_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Get current market conditions for learning."""
        try:
            features = self.feature_engineer.extract_features(symbol)
            if not features:
                return {}
            
            return {
                'volatility': features.volatility_features.get("volatility_20d", 0.2),
                'trend_strength': features.regime_features.get("trend_regime", 0),
                'volume_ratio': features.volume_features.get("volume_ratio_20", 1.0),
                'rsi': features.technical_features.get("rsi", 50),
                'regime': self.current_regime
            }
        except Exception as e:
            logger.error(f"Error getting market conditions: {e}")
            return {}
    
    def _update_pattern_performance(self, outcome: Dict[str, Any]):
        """Update pattern performance based on trade outcomes."""
        try:
            pnl = outcome.get('pnl', 0)
            is_profitable = pnl > 0
            
            # Update patterns that were active during this trade
            for pattern in self.discovered_patterns.values():
                if pattern.occurrences > 0:  # Pattern was recently active
                    # Simple update - in practice, you'd track which patterns led to which trades
                    if is_profitable:
                        pattern.success_rate = (pattern.success_rate * 0.9) + (1.0 * 0.1)
                    else:
                        pattern.success_rate = (pattern.success_rate * 0.9) + (0.0 * 0.1)
                    
                    pattern.confidence = min(pattern.success_rate + 0.1, 1.0)
        
        except Exception as e:
            logger.error(f"Error updating pattern performance: {e}")
    
    def _update_regime_performance(self, outcome: Dict[str, Any]):
        """Update regime performance tracking."""
        try:
            if not self.current_regime:
                return
            
            pnl = outcome.get('pnl', 0)
            
            # Update regime statistics (simplified)
            regime = self.market_regimes.get(self.current_regime)
            if regime:
                # Track regime performance - in practice, you'd maintain detailed statistics
                pass
        
        except Exception as e:
            logger.error(f"Error updating regime performance: {e}")
    
    def _retrain_models(self):
        """Retrain ML models with new data."""
        try:
            if len(self.trading_outcomes) < 50:
                return
            
            logger.info("Retraining market intelligence models...")
            
            # Prepare training data
            X = []
            y = []
            
            for outcome in self.trading_outcomes[-500:]:  # Use recent data
                conditions = outcome.get('market_conditions', {})
                if conditions:
                    feature_vector = [
                        conditions.get('volatility', 0.2),
                        conditions.get('trend_strength', 0),
                        conditions.get('volume_ratio', 1.0),
                        conditions.get('rsi', 50) / 100.0,
                    ]
                    X.append(feature_vector)
                    
                    # Binary outcome: profitable or not
                    pnl = outcome['outcome'].get('pnl', 0)
                    y.append(1 if pnl > 0 else 0)
            
            if len(X) > 20:
                X = np.array(X)
                
                # Retrain anomaly detector
                self.anomaly_detector.fit(X)
                
                # Retrain regime classifier
                if len(X) > 10:
                    self.regime_classifier.fit(X)
                
                self.is_trained = True
                self.last_training_time = datetime.now()
                self.training_data_size = len(X)
                
                logger.info(f"Models retrained with {len(X)} samples")
        
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def get_current_insights(self, symbol: Optional[str] = None, limit: int = 10) -> List[TradingInsight]:
        """Get current trading insights."""
        insights = self.insights_history[-100:]  # Recent insights
        
        if symbol:
            insights = [i for i in insights if i.symbol == symbol]
        
        # Sort by confidence and recency
        insights.sort(key=lambda x: (x.confidence, x.timestamp), reverse=True)
        
        return insights[:limit]
    
    def get_market_regime_info(self) -> Dict[str, Any]:
        """Get current market regime information."""
        if not self.current_regime:
            return {}
        
        regime = self.market_regimes.get(self.current_regime)
        if not regime:
            return {}
        
        return {
            'current_regime': self.current_regime,
            'regime_name': regime.name,
            'characteristics': regime.characteristics,
            'optimal_strategies': regime.optimal_strategies,
            'risk_level': regime.risk_level,
            'volatility_range': regime.volatility_range
        }
    
    def get_discovered_patterns(self) -> Dict[str, Dict]:
        """Get information about discovered patterns."""
        pattern_info = {}
        
        for pattern_id, pattern in self.discovered_patterns.items():
            pattern_info[pattern_id] = {
                'name': pattern.name,
                'description': pattern.description,
                'success_rate': pattern.success_rate,
                'confidence': pattern.confidence,
                'occurrences': pattern.occurrences,
                'expected_outcome': pattern.expected_outcome
            }
        
        return pattern_info
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of market intelligence state."""
        return {
            'current_regime': self.current_regime,
            'total_patterns_discovered': len(self.discovered_patterns),
            'recent_insights_count': len([i for i in self.insights_history if 
                                        (datetime.now() - i.timestamp).total_seconds() < 3600]),
            'is_trained': self.is_trained,
            'last_training_time': self.last_training_time,
            'training_data_size': self.training_data_size,
            'market_data_points': len(self.market_history),
            'trading_outcomes_learned': len(self.trading_outcomes)
        }


# Global market intelligence instance
market_intelligence = None