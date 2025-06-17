"""
Centralized configuration manager for the trading bot.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from .base_config import (
    BaseConfig,
    ConfigError,
    DatabaseConfig,
    ExchangeConfig,
    LoggingConfig,
    MLConfig,
    NotificationConfig,
    TradingConfig
)

T = TypeVar('T', bound=BaseConfig)


class AppConfig(BaseConfig):
    """Main application configuration that combines all sub-configurations."""
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    exchange: ExchangeConfig = ExchangeConfig()
    trading: TradingConfig = TradingConfig()
    ml: MLConfig = MLConfig()
    notifications: NotificationConfig = NotificationConfig()
    logging: LoggingConfig = LoggingConfig()
    
    def validate_config(self) -> None:
        """Validate the entire configuration."""
        # Validate environment
        if self.environment not in ["development", "staging", "production"]:
            raise ConfigError(f"Invalid environment: {self.environment}")
        
        # Validate production requirements
        if self.environment == "production":
            if not self.exchange.api_key or not self.exchange.api_secret:
                raise ConfigError("API credentials required for production")
            
            if self.exchange.testnet:
                raise ConfigError("Testnet cannot be enabled in production")
            
            if not self.notifications.discord_webhook_url:
                raise ConfigError("Discord notifications required for production")
        
        # Validate trading configuration consistency
        if self.trading.max_position_pct >= self.trading.max_total_exposure_pct:
            raise ConfigError("max_position_pct must be less than max_total_exposure_pct")
        
        # Validate ML model files exist
        if not Path(self.ml.model_path).exists():
            raise ConfigError(f"ML model file not found: {self.ml.model_path}")
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


class ConfigManager:
    """Singleton configuration manager."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None
    
    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from environment and files."""
        try:
            # Load environment file if it exists
            env_file = Path(".env")
            if env_file.exists():
                from dotenv import load_dotenv
                load_dotenv(env_file)
            
            # Create main configuration
            self._config = AppConfig()
            
            # Validate configuration
            self._config.validate_config()
            
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}")
    
    @property
    def config(self) -> AppConfig:
        """Get the current configuration."""
        if self._config is None:
            self._load_config()
        return self._config
    
    def reload_config(self) -> None:
        """Reload configuration from files."""
        self._config = None
        self._load_config()
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.config.dict()
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        if self._config is None:
            self._load_config()
        
        # Create new config with updates
        config_dict = self._config.dict()
        config_dict.update(updates)
        
        try:
            self._config = AppConfig(**config_dict)
            self._config.validate_config()
        except Exception as e:
            raise ConfigError(f"Failed to update configuration: {e}")
    
    def get_nested_value(self, path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation."""
        return self.config.get_nested_value(path, default)


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config_manager.config


def reload_config() -> None:
    """Reload the global configuration."""
    config_manager.reload_config()


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config().database


def get_exchange_config() -> ExchangeConfig:
    """Get exchange configuration."""
    return get_config().exchange


def get_trading_config() -> TradingConfig:
    """Get trading configuration."""
    return get_config().trading


def get_ml_config() -> MLConfig:
    """Get machine learning configuration."""
    return get_config().ml


def get_notification_config() -> NotificationConfig:
    """Get notification configuration."""
    return get_config().notifications


def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return get_config().logging