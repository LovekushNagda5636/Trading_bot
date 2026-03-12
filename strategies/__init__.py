"""
Trading Strategies Library

A comprehensive collection of trading strategies for Indian markets.
Supports equity, futures, and options trading with various timeframes.
"""

from .base import BaseStrategy, MarketType, TimeFrame, Signal, StrategyParams, MarketSession
from .registry import strategy_registry, get_strategy, create_strategy, list_strategies, discover_strategies

__version__ = "1.0.0"

__all__ = [
    'BaseStrategy',
    'MarketType', 
    'TimeFrame',
    'Signal',
    'StrategyParams',
    'MarketSession',
    'strategy_registry',
    'get_strategy',
    'create_strategy', 
    'list_strategies',
    'discover_strategies'
]