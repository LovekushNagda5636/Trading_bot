"""
Configuration management for the trading bot.
Supports environment-specific configurations and runtime parameter updates.
"""

import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import logging
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import field_validator

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Environment
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    # Database
    database_url: str = "sqlite:///trading_bot.db"
    database_echo: bool = False
    
    # Trading
    trading_enabled: bool = False
    paper_trading: bool = True
    max_positions: int = 10
    
    # Risk Management
    max_daily_loss: Decimal = Decimal('10000')
    max_position_size: Decimal = Decimal('100000')
    max_portfolio_exposure: Decimal = Decimal('500000')
    
    # Market Data
    market_data_timeout: int = 30
    reconnection_interval: int = 5
    
    # Performance
    max_ticks_per_second: int = 1000
    order_execution_timeout: int = 500  # milliseconds
    
    # Monitoring
    enable_alerts: bool = True
    alert_email: Optional[str] = None
    alert_sms: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_prefix = "TRADING_BOT_"


@dataclass
class BrokerConfig:
    """Broker-specific configuration."""
    broker_name: str = ""
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    base_url: str = ""
    rate_limit: int = 100  # requests per minute
    timeout: int = 30  # seconds
    enabled: bool = False


@dataclass
class StrategyConfig:
    """Strategy configuration parameters."""
    strategy_id: str = ""
    strategy_type: str = ""
    enabled: bool = False
    allocation: Decimal = Decimal('0.1')  # 10% allocation
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not 0 < self.allocation <= 1:
            return False
        return True


@dataclass
class TradingConfig:
    """Main trading configuration."""
    trading_enabled: bool = False
    paper_trading: bool = True
    max_positions: int = 10
    risk_limits: Dict[str, Any] = field(default_factory=dict)
    strategy_configs: List[StrategyConfig] = field(default_factory=list)
    broker_configs: List[BrokerConfig] = field(default_factory=list)
    
    def get_active_strategies(self) -> List[StrategyConfig]:
        """Get list of enabled strategies."""
        return [s for s in self.strategy_configs if s.enabled]
    
    def get_active_brokers(self) -> List[BrokerConfig]:
        """Get list of enabled brokers."""
        return [b for b in self.broker_configs if b.enabled]


class ConfigManager:
    """
    Configuration manager with hot-reload capabilities.
    Supports environment-specific configs and runtime updates.
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self._settings = Settings()
        self._trading_config = TradingConfig()
        self._config_watchers: List[callable] = []
        
        self._load_configurations()
    
    @property
    def settings(self) -> Settings:
        """Get application settings."""
        return self._settings
    
    @property
    def trading_config(self) -> TradingConfig:
        """Get trading configuration."""
        return self._trading_config
    
    def _load_configurations(self) -> None:
        """Load all configuration files."""
        try:
            self._load_trading_config()
            self._load_broker_configs()
            self._load_strategy_configs()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _load_trading_config(self) -> None:
        """Load main trading configuration."""
        config_file = self.config_dir / f"trading_{self._settings.environment}.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                self._trading_config = TradingConfig(**config_data)
        else:
            # Create default config
            self._save_trading_config()
    
    def _load_broker_configs(self) -> None:
        """Load broker configurations."""
        broker_file = self.config_dir / "brokers.json"
        
        if broker_file.exists():
            with open(broker_file, 'r') as f:
                brokers_data = json.load(f)
                self._trading_config.broker_configs = [
                    BrokerConfig(**broker) for broker in brokers_data.get('brokers', [])
                ]
    
    def _load_strategy_configs(self) -> None:
        """Load strategy configurations."""
        strategy_file = self.config_dir / "strategies.json"
        
        if strategy_file.exists():
            with open(strategy_file, 'r') as f:
                strategies_data = json.load(f)
                self._trading_config.strategy_configs = [
                    StrategyConfig(**strategy) for strategy in strategies_data.get('strategies', [])
                ]
    
    def _save_trading_config(self) -> None:
        """Save trading configuration to file."""
        config_file = self.config_dir / f"trading_{self._settings.environment}.json"
        
        config_data = {
            'trading_enabled': self._trading_config.trading_enabled,
            'paper_trading': self._trading_config.paper_trading,
            'max_positions': self._trading_config.max_positions,
            'risk_limits': self._trading_config.risk_limits
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def update_trading_config(self, **kwargs) -> bool:
        """Update trading configuration parameters."""
        try:
            for key, value in kwargs.items():
                if hasattr(self._trading_config, key):
                    setattr(self._trading_config, key, value)
            
            self._save_trading_config()
            self._notify_config_change('trading_config', kwargs)
            return True
        except Exception as e:
            logger.error(f"Error updating trading config: {e}")
            return False
    
    def add_strategy_config(self, strategy_config: StrategyConfig) -> bool:
        """Add a new strategy configuration."""
        try:
            if not strategy_config.validate_parameters():
                raise ValueError("Invalid strategy parameters")
            
            # Check for duplicate strategy IDs
            existing_ids = [s.strategy_id for s in self._trading_config.strategy_configs]
            if strategy_config.strategy_id in existing_ids:
                raise ValueError(f"Strategy ID {strategy_config.strategy_id} already exists")
            
            self._trading_config.strategy_configs.append(strategy_config)
            self._save_strategy_configs()
            self._notify_config_change('strategy_added', strategy_config)
            return True
        except Exception as e:
            logger.error(f"Error adding strategy config: {e}")
            return False
    
    def update_strategy_config(self, strategy_id: str, **kwargs) -> bool:
        """Update an existing strategy configuration."""
        try:
            for strategy in self._trading_config.strategy_configs:
                if strategy.strategy_id == strategy_id:
                    for key, value in kwargs.items():
                        if hasattr(strategy, key):
                            setattr(strategy, key, value)
                    
                    if not strategy.validate_parameters():
                        raise ValueError("Invalid strategy parameters")
                    
                    self._save_strategy_configs()
                    self._notify_config_change('strategy_updated', strategy)
                    return True
            
            raise ValueError(f"Strategy {strategy_id} not found")
        except Exception as e:
            logger.error(f"Error updating strategy config: {e}")
            return False
    
    def _save_strategy_configs(self) -> None:
        """Save strategy configurations to file."""
        strategy_file = self.config_dir / "strategies.json"
        
        strategies_data = {
            'strategies': [
                {
                    'strategy_id': s.strategy_id,
                    'strategy_type': s.strategy_type,
                    'enabled': s.enabled,
                    'allocation': str(s.allocation),
                    'parameters': s.parameters
                }
                for s in self._trading_config.strategy_configs
            ]
        }
        
        with open(strategy_file, 'w') as f:
            json.dump(strategies_data, f, indent=2)
    
    def add_config_watcher(self, callback: callable) -> None:
        """Add a callback to be notified of configuration changes."""
        self._config_watchers.append(callback)
    
    def _notify_config_change(self, change_type: str, data: Any) -> None:
        """Notify all watchers of configuration changes."""
        for callback in self._config_watchers:
            try:
                callback(change_type, data)
            except Exception as e:
                logger.error(f"Error in config watcher callback: {e}")
    
    def reload_config(self) -> bool:
        """Reload configuration from files."""
        try:
            self._load_configurations()
            self._notify_config_change('config_reloaded', None)
            return True
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            return False
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return any errors."""
        errors = []
        
        # Validate strategy allocations don't exceed 100%
        total_allocation = sum(s.allocation for s in self._trading_config.strategy_configs if s.enabled)
        if total_allocation > 1:
            errors.append(f"Total strategy allocation ({total_allocation:.2%}) exceeds 100%")
        
        # Validate broker configurations
        active_brokers = self._trading_config.get_active_brokers()
        if self._trading_config.trading_enabled and not active_brokers:
            errors.append("Trading enabled but no active brokers configured")
        
        # Validate risk limits
        if self._settings.max_daily_loss <= 0:
            errors.append("Max daily loss must be positive")
        
        return errors


# Global configuration manager instance
config_manager = ConfigManager()