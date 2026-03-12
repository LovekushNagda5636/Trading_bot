"""
AI-Powered Trading Strategy that combines multiple ML approaches.
Uses reinforcement learning, market intelligence, and continuous learning.
"""

from decimal import Decimal
from typing import List, Optional, Dict, Any
import structlog
import asyncio

from .base import Strategy, MarketContext, ExitCondition
from ..core.models import TradingSignal, SignalType, OrderSide
from ..core.config import StrategyConfig
from ..ml.reinforcement_learning import RLTradingAgent
from ..ml.market_intelligence import MarketIntelligence, TradingInsight
from ..ml.model_trainer import ModelTrainer
from ..ml.feature_engineering import FeatureEngineer

logger = structlog.get_logger(__name__)


class AIStrategy(Strategy):
    """
    Advanced AI-powered trading strategy that combines:
    - Reinforcement Learning for decision making
    - Market Intelligence for pattern recognition
    - Continuous Learning from outcomes
    - Feature Engineering for rich market understanding
    """
    
    def __init__(
        self, 
        strategy_config: StrategyConfig,
        feature_engineer: FeatureEngineer,
        market_intelligence: MarketIntelligence,
        model_trainer: ModelTrainer
    ):
        super().__init__(strategy_config)
        
        self.feature_engineer = feature_engineer
        self.market_intelligence = market_intelligence
        self.model_trainer = model_trainer
        
        # Initialize RL agent
        self.rl_agent = RLTradingAgent(strategy_config, feature_engineer)
        
        # AI strategy specific parameters
        self.use_rl = self.parameters.get('use_rl', True)
        self.use_ml_predictions = self.parameters.get('use_ml_predictions', True)
        self.use_market_intelligence = self.parameters.get('use_market_intelligence', True)
        self.confidence_threshold = self.parameters.get('confidence_threshold', 0.6)
        self.ensemble_voting = self.parameters.get('ensemble_voting', True)
        
        # Decision weights
        self.rl_weight = self.parameters.get('rl_weight', 0.4)
        self.ml_weight = self.parameters.get('ml_weight', 0.3)
        self.intelligence_weight = self.parameters.get('intelligence_weight', 0.3)
        
        # Learning configuration
        self.continuous_learning = self.parameters.get('continuous_learning', True)
        self.adaptation_rate = self.parameters.get('adaptation_rate', 0.1)
        
        # Performance tracking
        self.ai_decisions = []
        self.learning_outcomes = []
        
        logger.info(f"AI Strategy initialized: {self.strategy_id}")
        logger.info(f"RL enabled: {self.use_rl}, ML predictions: {self.use_ml_predictions}")
        logger.info(f"Market intelligence: {self.use_market_intelligence}")
    
    def should_enter_trade(self, signal: TradingSignal, context: MarketContext) -> bool:
        """
        AI-powered entry decision using multiple ML approaches.
        """
        try:
            # Get market features
            features = self.feature_engineer.extract_features(signal.symbol)
            if not features:
                logger.debug(f"No features available for {signal.symbol}")
                return False
            
            # Collect decisions from different AI components
            decisions = []
            confidences = []
            
            # 1. Reinforcement Learning Decision
            if self.use_rl:
                rl_decision = self.rl_agent.should_enter_trade(signal, context)
                rl_confidence = self._get_rl_confidence(signal.symbol, context)
                decisions.append(rl_decision)
                confidences.append(rl_confidence * self.rl_weight)
                logger.debug(f"RL decision: {rl_decision} (confidence: {rl_confidence:.3f})")
            
            # 2. ML Model Prediction
            if self.use_ml_predictions:
                ml_prediction = self.model_trainer.predict(features)
                if ml_prediction:
                    ml_decision, ml_confidence = ml_prediction
                    ml_trade_decision = ml_decision == 1  # 1 means profitable trade predicted
                    decisions.append(ml_trade_decision)
                    confidences.append(ml_confidence * self.ml_weight)
                    logger.debug(f"ML decision: {ml_trade_decision} (confidence: {ml_confidence:.3f})")
            
            # 3. Market Intelligence Insights
            if self.use_market_intelligence:
                intelligence_decision, intelligence_confidence = self._get_intelligence_decision(
                    signal, context, features
                )
                decisions.append(intelligence_decision)
                confidences.append(intelligence_confidence * self.intelligence_weight)
                logger.debug(f"Intelligence decision: {intelligence_decision} (confidence: {intelligence_confidence:.3f})")
            
            # Ensemble decision making
            final_decision = self._make_ensemble_decision(decisions, confidences, signal, context)
            
            # Store decision for learning
            self._store_ai_decision(signal, context, decisions, confidences, final_decision)
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Error in AI strategy entry decision: {e}")
            return False
    
    def _get_rl_confidence(self, symbol: str, context: MarketContext) -> float:
        """Get confidence from RL agent."""
        try:
            state = self.rl_agent.get_state(symbol, context)
            if state is None:
                return 0.5
            
            # Get Q-values and calculate confidence
            import torch
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.rl_agent.device)
                q_values = self.rl_agent.q_network(state_tensor)
                confidence = torch.softmax(q_values, dim=1).max().item()
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error getting RL confidence: {e}")
            return 0.5
    
    def _get_intelligence_decision(
        self, 
        signal: TradingSignal, 
        context: MarketContext, 
        features
    ) -> tuple[bool, float]:
        """Get decision from market intelligence."""
        try:
            # Get current insights
            insights = self.market_intelligence.get_current_insights(signal.symbol, limit=5)
            
            if not insights:
                return False, 0.3  # No insights, low confidence neutral
            
            # Analyze insights
            positive_insights = 0
            negative_insights = 0
            total_confidence = 0
            
            for insight in insights:
                total_confidence += insight.confidence
                
                if insight.expected_impact == "positive":
                    positive_insights += insight.confidence
                elif insight.expected_impact == "negative":
                    negative_insights += insight.confidence
            
            # Get market regime information
            regime_info = self.market_intelligence.get_market_regime_info()
            regime_confidence = 0.5
            
            if regime_info:
                current_regime = regime_info.get('current_regime')
                optimal_strategies = regime_info.get('optimal_strategies', [])
                
                # Check if our strategy type aligns with regime
                if signal.signal_type == SignalType.BUY:
                    if 'momentum' in optimal_strategies or 'trend_following' in optimal_strategies:
                        regime_confidence = 0.8
                    elif 'mean_reversion' in optimal_strategies:
                        regime_confidence = 0.3
                elif signal.signal_type == SignalType.SELL:
                    if 'short_selling' in optimal_strategies or 'defensive' in optimal_strategies:
                        regime_confidence = 0.8
                    elif 'mean_reversion' in optimal_strategies:
                        regime_confidence = 0.7
            
            # Combine insights and regime analysis
            if positive_insights > negative_insights:
                decision = signal.signal_type == SignalType.BUY
                confidence = min((positive_insights / len(insights)) * regime_confidence, 1.0)
            elif negative_insights > positive_insights:
                decision = signal.signal_type == SignalType.SELL
                confidence = min((negative_insights / len(insights)) * regime_confidence, 1.0)
            else:
                decision = False
                confidence = 0.4
            
            return decision, confidence
            
        except Exception as e:
            logger.error(f"Error getting intelligence decision: {e}")
            return False, 0.3
    
    def _make_ensemble_decision(
        self, 
        decisions: List[bool], 
        confidences: List[float], 
        signal: TradingSignal, 
        context: MarketContext
    ) -> bool:
        """Make final ensemble decision."""
        try:
            if not decisions:
                return False
            
            if self.ensemble_voting:
                # Weighted voting
                weighted_score = sum(
                    1.0 if decision else 0.0 * confidence 
                    for decision, confidence in zip(decisions, confidences)
                )
                total_weight = sum(confidences)
                
                if total_weight > 0:
                    final_confidence = weighted_score / total_weight
                else:
                    final_confidence = 0.5
                
                # Decision based on confidence threshold
                final_decision = final_confidence > self.confidence_threshold
                
                logger.info(f"Ensemble decision for {signal.symbol}: {final_decision} "
                           f"(confidence: {final_confidence:.3f}, threshold: {self.confidence_threshold})")
                
                return final_decision
            else:
                # Simple majority voting
                positive_votes = sum(1 for decision in decisions if decision)
                return positive_votes > len(decisions) / 2
            
        except Exception as e:
            logger.error(f"Error making ensemble decision: {e}")
            return False
    
    def _store_ai_decision(
        self, 
        signal: TradingSignal, 
        context: MarketContext, 
        decisions: List[bool], 
        confidences: List[float], 
        final_decision: bool
    ):
        """Store AI decision for learning."""
        try:
            decision_record = {
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'signal_type': signal.signal_type.value,
                'signal_strength': signal.strength,
                'individual_decisions': decisions,
                'individual_confidences': confidences,
                'final_decision': final_decision,
                'market_price': float(context.current_price),
                'strategy_id': self.strategy_id
            }
            
            self.ai_decisions.append(decision_record)
            
            # Limit history
            if len(self.ai_decisions) > 1000:
                self.ai_decisions = self.ai_decisions[-1000:]
            
        except Exception as e:
            logger.error(f"Error storing AI decision: {e}")
    
    def calculate_position_size(self, context: MarketContext) -> int:
        """Calculate position size using AI insights."""
        try:
            # Base position size
            base_quantity = super().calculate_position_size(context)
            
            # Get AI confidence for position sizing
            if self.ai_decisions:
                recent_decision = self.ai_decisions[-1]
                avg_confidence = sum(recent_decision['individual_confidences']) / len(recent_decision['individual_confidences'])
                
                # Adjust position size based on confidence
                confidence_multiplier = min(avg_confidence * 1.5, 1.2)  # Max 20% increase
                adjusted_quantity = int(base_quantity * confidence_multiplier)
                
                return max(1, adjusted_quantity)
            
            return base_quantity
            
        except Exception as e:
            logger.error(f"Error calculating AI position size: {e}")
            return super().calculate_position_size(context)
    
    def get_exit_conditions(self, context: MarketContext) -> List[ExitCondition]:
        """Get AI-enhanced exit conditions."""
        conditions = super().get_exit_conditions(context)
        
        # Add AI-specific exit conditions
        conditions.append(ExitCondition(
            condition_type="ai_confidence_drop",
            enabled=True
        ))
        
        conditions.append(ExitCondition(
            condition_type="regime_change",
            enabled=True
        ))
        
        return conditions
    
    def should_exit_trade(self, position, context: MarketContext) -> bool:
        """AI-enhanced exit decision."""
        # Check standard exit conditions
        if super().should_exit_trade(position, context):
            return True
        
        try:
            # AI-specific exit logic
            
            # 1. Check if market regime changed unfavorably
            regime_info = self.market_intelligence.get_market_regime_info()
            if regime_info:
                optimal_strategies = regime_info.get('optimal_strategies', [])
                if position.is_long and 'short_selling' in optimal_strategies:
                    logger.info(f"Exiting long position due to regime change: {regime_info.get('current_regime')}")
                    return True
                elif position.is_short and 'momentum' in optimal_strategies:
                    logger.info(f"Exiting short position due to regime change: {regime_info.get('current_regime')}")
                    return True
            
            # 2. Check recent insights
            insights = self.market_intelligence.get_current_insights(position.symbol, limit=3)
            negative_insights = [i for i in insights if i.expected_impact == "negative" and i.confidence > 0.7]
            
            if len(negative_insights) >= 2:
                logger.info(f"Exiting position due to negative insights: {len(negative_insights)}")
                return True
            
            # 3. RL agent exit decision
            if self.use_rl:
                # Update RL agent with current context and let it decide
                self.rl_agent.update_experience(position.symbol, context, done=False)
                
                # Get RL confidence - if very low, consider exit
                rl_confidence = self._get_rl_confidence(position.symbol, context)
                if rl_confidence < 0.3:
                    logger.info(f"Exiting position due to low RL confidence: {rl_confidence:.3f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in AI exit decision: {e}")
            return False
    
    async def learn_from_outcome(self, trade_decision, trade_outcome: Dict[str, Any]):
        """Learn from trading outcome using all AI components."""
        try:
            # Store outcome for analysis
            self.learning_outcomes.append({
                'decision': trade_decision,
                'outcome': trade_outcome,
                'timestamp': trade_decision.timestamp
            })
            
            # Update RL agent
            if self.use_rl and hasattr(self, 'rl_agent'):
                # Create context for RL learning
                context = MarketContext(
                    symbol=trade_decision.symbol,
                    current_price=Decimal(str(trade_outcome.get('exit_price', 0))),
                    available_capital=Decimal('100000'),  # Placeholder
                    portfolio_exposure=Decimal('0')
                )
                self.rl_agent.update_experience(trade_decision.symbol, context, done=True)
            
            # Update market intelligence
            if self.use_market_intelligence:
                self.market_intelligence.learn_from_trade_outcome(trade_decision, trade_outcome)
            
            # Update ML model trainer
            if self.use_ml_predictions and self.continuous_learning:
                features = self.feature_engineer.extract_features(trade_decision.symbol)
                if features:
                    self.model_trainer.add_training_sample(
                        trade_decision.symbol, 
                        features, 
                        trade_decision, 
                        trade_outcome
                    )
            
            # Adapt strategy parameters based on performance
            if self.continuous_learning:
                await self._adapt_strategy_parameters()
            
            logger.info(f"AI strategy learned from outcome: PnL={trade_outcome.get('pnl', 0):.2f}")
            
        except Exception as e:
            logger.error(f"Error learning from outcome: {e}")
    
    async def _adapt_strategy_parameters(self):
        """Adapt strategy parameters based on recent performance."""
        try:
            if len(self.learning_outcomes) < 10:
                return
            
            # Analyze recent performance
            recent_outcomes = self.learning_outcomes[-20:]
            profitable_trades = [o for o in recent_outcomes if o['outcome'].get('pnl', 0) > 0]
            win_rate = len(profitable_trades) / len(recent_outcomes)
            
            # Adapt confidence threshold
            if win_rate > 0.7:
                # High win rate - can be more aggressive
                self.confidence_threshold = max(0.5, self.confidence_threshold - self.adaptation_rate * 0.1)
            elif win_rate < 0.4:
                # Low win rate - be more conservative
                self.confidence_threshold = min(0.8, self.confidence_threshold + self.adaptation_rate * 0.1)
            
            # Adapt component weights based on individual performance
            # This is a simplified adaptation - in practice, you'd track which components
            # contributed to successful vs unsuccessful trades
            
            logger.debug(f"Adapted strategy parameters: confidence_threshold={self.confidence_threshold:.3f}")
            
        except Exception as e:
            logger.error(f"Error adapting strategy parameters: {e}")
    
    def get_ai_performance_summary(self) -> Dict[str, Any]:
        """Get AI strategy performance summary."""
        try:
            summary = {
                'strategy_id': self.strategy_id,
                'ai_decisions_count': len(self.ai_decisions),
                'learning_outcomes_count': len(self.learning_outcomes),
                'current_confidence_threshold': self.confidence_threshold,
                'component_weights': {
                    'rl_weight': self.rl_weight,
                    'ml_weight': self.ml_weight,
                    'intelligence_weight': self.intelligence_weight
                },
                'components_enabled': {
                    'rl': self.use_rl,
                    'ml_predictions': self.use_ml_predictions,
                    'market_intelligence': self.use_market_intelligence,
                    'continuous_learning': self.continuous_learning
                }
            }
            
            # Add RL performance if available
            if self.use_rl and hasattr(self.rl_agent, 'get_performance_metrics'):
                summary['rl_performance'] = self.rl_agent.get_performance_metrics()
            
            # Add recent decision analysis
            if self.ai_decisions:
                recent_decisions = self.ai_decisions[-10:]
                avg_confidence = sum(
                    sum(d['individual_confidences']) / len(d['individual_confidences'])
                    for d in recent_decisions
                ) / len(recent_decisions)
                summary['avg_recent_confidence'] = avg_confidence
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting AI performance summary: {e}")
            return {'error': str(e)}
    
    def save_ai_state(self, filepath: str):
        """Save AI strategy state."""
        try:
            # Save RL model
            if self.use_rl:
                rl_filepath = filepath.replace('.pkl', '_rl.pkl')
                self.rl_agent.save_model(rl_filepath)
            
            # Save decision history and parameters
            import pickle
            state_data = {
                'ai_decisions': self.ai_decisions,
                'learning_outcomes': self.learning_outcomes,
                'confidence_threshold': self.confidence_threshold,
                'component_weights': {
                    'rl_weight': self.rl_weight,
                    'ml_weight': self.ml_weight,
                    'intelligence_weight': self.intelligence_weight
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state_data, f)
            
            logger.info(f"AI strategy state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving AI state: {e}")
    
    def load_ai_state(self, filepath: str):
        """Load AI strategy state."""
        try:
            # Load RL model
            if self.use_rl:
                rl_filepath = filepath.replace('.pkl', '_rl.pkl')
                self.rl_agent.load_model(rl_filepath)
            
            # Load decision history and parameters
            import pickle
            with open(filepath, 'rb') as f:
                state_data = pickle.load(f)
            
            self.ai_decisions = state_data.get('ai_decisions', [])
            self.learning_outcomes = state_data.get('learning_outcomes', [])
            self.confidence_threshold = state_data.get('confidence_threshold', 0.6)
            
            weights = state_data.get('component_weights', {})
            self.rl_weight = weights.get('rl_weight', 0.4)
            self.ml_weight = weights.get('ml_weight', 0.3)
            self.intelligence_weight = weights.get('intelligence_weight', 0.3)
            
            logger.info(f"AI strategy state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading AI state: {e}")