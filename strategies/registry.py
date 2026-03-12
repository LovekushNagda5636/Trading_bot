"""
Strategy Registry - Central registry for all trading strategies.
Handles strategy discovery, loading, and management.
"""

import importlib
import inspect
from typing import Dict, List, Type, Any, Optional
from pathlib import Path
import logging

from .base import BaseStrategy, MarketType, TimeFrame

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Central registry for all trading strategies.
    Automatically discovers and registers strategies from the strategies directory.
    """
    
    def __init__(self):
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._strategy_metadata: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
    
    def discover_strategies(self) -> None:
        """Discover and register all strategies from the strategies directory."""
        if self._loaded:
            return
        
        strategies_dir = Path(__file__).parent
        logger.info(f"Discovering strategies in: {strategies_dir}")
        
        # Strategy directories to scan
        strategy_dirs = [
            'intraday_trend',
            'intraday_breakout', 
            'intraday_reversion',
            'price_action',
            'volume',
            'scalping',
            'gap',
            'index_sector',
            'swing',
            'derivatives',
            'quant'
        ]
        
        for strategy_dir in strategy_dirs:
            dir_path = strategies_dir / strategy_dir
            if dir_path.exists():
                self._discover_strategies_in_dir(dir_path, strategy_dir)
        
        self._loaded = True
        logger.info(f"Discovered {len(self._strategies)} strategies")
    
    def _discover_strategies_in_dir(self, dir_path: Path, category: str) -> None:
        """Discover strategies in a specific directory."""
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            try:
                # Import the module
                module_name = f"strategies.{category}.{py_file.stem}"
                module = importlib.import_module(module_name)
                
                # Find strategy classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseStrategy) and 
                        obj != BaseStrategy):
                        
                        strategy_name = obj.__name__
                        self._strategies[strategy_name] = obj
                        
                        # Store metadata
                        self._strategy_metadata[strategy_name] = {
                            'category': category,
                            'module': module_name,
                            'file': str(py_file),
                            'timeframe': getattr(obj, '_timeframe', None),
                            'market_type': getattr(obj, '_market_type', None)
                        }
                        
                        logger.debug(f"Registered strategy: {strategy_name} from {category}")
                        
            except Exception as e:
                logger.error(f"Error loading strategy from {py_file}: {e}")
    
    def get_strategy(self, name: str) -> Optional[Type[BaseStrategy]]:
        """Get a strategy class by name."""
        if not self._loaded:
            self.discover_strategies()
        return self._strategies.get(name)
    
    def create_strategy(self, name: str, params: Dict[str, Any]) -> Optional[BaseStrategy]:
        """Create a strategy instance with given parameters."""
        strategy_class = self.get_strategy(name)
        if strategy_class:
            try:
                return strategy_class(params)
            except Exception as e:
                logger.error(f"Error creating strategy {name}: {e}")
                return None
        return None
    
    def list_strategies(self) -> List[str]:
        """List all available strategy names."""
        if not self._loaded:
            self.discover_strategies()
        return list(self._strategies.keys())
    
    def list_strategies_by_category(self) -> Dict[str, List[str]]:
        """List strategies grouped by category."""
        if not self._loaded:
            self.discover_strategies()
        
        categories = {}
        for name, metadata in self._strategy_metadata.items():
            category = metadata['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        
        return categories
    
    def list_strategies_by_market_type(self, market_type: MarketType) -> List[str]:
        """List strategies for a specific market type."""
        if not self._loaded:
            self.discover_strategies()
        
        strategies = []
        for name, strategy_class in self._strategies.items():
            try:
                # Create temporary instance to check market type
                temp_instance = strategy_class({})
                if temp_instance.get_market_type() == market_type:
                    strategies.append(name)
            except:
                pass
        
        return strategies
    
    def list_strategies_by_timeframe(self, timeframe: TimeFrame) -> List[str]:
        """List strategies for a specific timeframe."""
        if not self._loaded:
            self.discover_strategies()
        
        strategies = []
        for name, strategy_class in self._strategies.items():
            try:
                # Create temporary instance to check timeframe
                temp_instance = strategy_class({})
                if temp_instance.get_timeframe() == timeframe:
                    strategies.append(name)
            except:
                pass
        
        return strategies
    
    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a strategy."""
        if name not in self._strategies:
            return None
        
        strategy_class = self._strategies[name]
        metadata = self._strategy_metadata.get(name, {})
        
        try:
            # Create temporary instance to get info
            temp_instance = strategy_class({})
            
            info = {
                'name': name,
                'class': strategy_class.__name__,
                'category': metadata.get('category', 'unknown'),
                'module': metadata.get('module', ''),
                'file': metadata.get('file', ''),
                'timeframe': temp_instance.get_timeframe().value,
                'market_type': temp_instance.get_market_type().value,
                'default_params': temp_instance.get_default_params(),
                'docstring': strategy_class.__doc__ or 'No description available'
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting info for strategy {name}: {e}")
            return None
    
    def validate_strategy(self, name: str) -> bool:
        """Validate that a strategy implements all required methods."""
        strategy_class = self.get_strategy(name)
        if not strategy_class:
            return False
        
        required_methods = [
            'get_timeframe', 'get_market_type', 'get_default_params',
            'indicators', 'generate_signals', 'should_enter', 
            'should_exit', 'get_stoploss', 'get_target'
        ]
        
        for method in required_methods:
            if not hasattr(strategy_class, method):
                logger.error(f"Strategy {name} missing required method: {method}")
                return False
        
        return True
    
    def get_strategies_summary(self) -> Dict[str, Any]:
        """Get summary of all registered strategies."""
        if not self._loaded:
            self.discover_strategies()
        
        categories = self.list_strategies_by_category()
        
        summary = {
            'total_strategies': len(self._strategies),
            'categories': {cat: len(strategies) for cat, strategies in categories.items()},
            'by_market_type': {},
            'by_timeframe': {}
        }
        
        # Count by market type
        for market_type in MarketType:
            strategies = self.list_strategies_by_market_type(market_type)
            summary['by_market_type'][market_type.value] = len(strategies)
        
        # Count by timeframe
        for timeframe in TimeFrame:
            strategies = self.list_strategies_by_timeframe(timeframe)
            summary['by_timeframe'][timeframe.value] = len(strategies)
        
        return summary


# Global registry instance
strategy_registry = StrategyRegistry()


def get_strategy(name: str) -> Optional[Type[BaseStrategy]]:
    """Convenience function to get a strategy."""
    return strategy_registry.get_strategy(name)


def create_strategy(name: str, params: Dict[str, Any]) -> Optional[BaseStrategy]:
    """Convenience function to create a strategy instance."""
    return strategy_registry.create_strategy(name, params)


def list_strategies() -> List[str]:
    """Convenience function to list all strategies."""
    return strategy_registry.list_strategies()


def discover_strategies() -> None:
    """Convenience function to discover strategies."""
    strategy_registry.discover_strategies()