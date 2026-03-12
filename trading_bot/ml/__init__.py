"""Machine Learning components for intelligent trading strategies.

Imports are lazy to avoid dependency issues with pydantic/torch/etc.
Use direct imports from submodules instead:
    from trading_bot.ml.trade_journal import TradeJournal
    from trading_bot.ml.adaptive_strategy_engine import AdaptiveStrategyEngine
"""

# Only export names — actual imports happen lazily when accessed
__all__ = [
    'MarketIntelligence',
    'AIStrategyGenerator',
    'RLTradingAgent',
    'FeatureEngineer',
    'ModelTrainer',
    'TradeJournal',
    'AdaptiveStrategyEngine',
    'MarketRegimeDetector',
    'FnOScanner',
]


def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies at package init."""
    if name == 'MarketIntelligence':
        from .market_intelligence import MarketIntelligence
        return MarketIntelligence
    elif name == 'AIStrategyGenerator':
        from .strategy_generator import AIStrategyGenerator
        return AIStrategyGenerator
    elif name == 'RLTradingAgent':
        from .reinforcement_learning import RLTradingAgent
        return RLTradingAgent
    elif name == 'FeatureEngineer':
        from .feature_engineering import FeatureEngineer
        return FeatureEngineer
    elif name == 'ModelTrainer':
        from .model_trainer import ModelTrainer
        return ModelTrainer
    elif name == 'TradeJournal':
        from .trade_journal import TradeJournal
        return TradeJournal
    elif name == 'AdaptiveStrategyEngine':
        from .adaptive_strategy_engine import AdaptiveStrategyEngine
        return AdaptiveStrategyEngine
    elif name == 'MarketRegimeDetector':
        from .regime_detector import MarketRegimeDetector
        return MarketRegimeDetector
    elif name == 'FnOScanner':
        from .fno_scanner import FnOScanner
        return FnOScanner
    raise AttributeError(f"module 'trading_bot.ml' has no attribute {name}")