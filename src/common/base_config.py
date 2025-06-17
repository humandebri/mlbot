"""
Base configuration management with validation and environment loading.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

T = TypeVar('T', bound='BaseConfig')


class ConfigError(Exception):
    """Configuration related errors."""
    pass


class BaseConfig(BaseSettings, ABC):
    """Abstract base class for all configuration classes."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid"  # Prevent typos in config
    )
    
    @classmethod
    def load_from_env(cls: Type[T], env_prefix: str = "") -> T:
        """Load configuration from environment variables."""
        try:
            return cls(_env_prefix=env_prefix)
        except Exception as e:
            raise ConfigError(f"Failed to load {cls.__name__}: {e}")
    
    @abstractmethod
    def validate_config(self) -> None:
        """Validate configuration after loading."""
        pass
    
    def get_nested_value(self, path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation."""
        parts = path.split('.')
        value = self
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        
        return value


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    
    # Redis settings
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    redis_timeout: float = Field(default=5.0, env="REDIS_TIMEOUT")
    
    # DuckDB settings
    duckdb_path: str = Field(default="data/market_data.duckdb", env="DUCKDB_PATH")
    duckdb_memory_limit: str = Field(default="2GB", env="DUCKDB_MEMORY_LIMIT")
    duckdb_threads: int = Field(default=4, env="DUCKDB_THREADS")
    
    @validator('duckdb_path')
    def validate_duckdb_path(cls, v):
        """Ensure DuckDB directory exists."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)


class TradingConfig(BaseModel):
    """Trading configuration settings."""
    
    # Position sizing
    max_position_pct: float = Field(default=0.3, ge=0.01, le=1.0)
    max_total_exposure_pct: float = Field(default=0.6, ge=0.01, le=1.0)
    max_daily_loss_pct: float = Field(default=0.1, ge=0.01, le=0.5)
    leverage: float = Field(default=3.0, ge=1.0, le=10.0)
    
    # Risk management
    stop_loss_pct: float = Field(default=0.02, ge=0.005, le=0.1)
    take_profit_pct: float = Field(default=0.03, ge=0.01, le=0.2)
    trailing_stop_pct: float = Field(default=0.015, ge=0.005, le=0.05)
    
    # Trading limits
    max_trades_per_hour: int = Field(default=50, ge=1, le=200)
    max_trades_per_day: int = Field(default=200, ge=1, le=1000)
    cooldown_period_seconds: int = Field(default=300, ge=60, le=3600)
    
    # Prediction thresholds
    min_prediction_confidence: float = Field(default=0.6, ge=0.5, le=0.95)
    min_expected_pnl: float = Field(default=0.001, ge=0.0001, le=0.01)
    
    @validator('max_total_exposure_pct')
    def validate_total_exposure(cls, v, values):
        """Ensure total exposure is greater than max position."""
        if 'max_position_pct' in values and v <= values['max_position_pct']:
            raise ValueError('total_exposure_pct must be greater than max_position_pct')
        return v


class ExchangeConfig(BaseModel):
    """Exchange configuration settings."""
    
    # API endpoints
    base_url: str = Field(default="https://api.bybit.com", env="BYBIT_BASE_URL")
    testnet_url: str = Field(default="https://api-testnet.bybit.com", env="BYBIT_TESTNET_URL")
    ws_url: str = Field(default="wss://stream.bybit.com/v5/public/linear", env="BYBIT_WS_URL")
    testnet_ws_url: str = Field(default="wss://stream-testnet.bybit.com/v5/public/linear", env="BYBIT_TESTNET_WS_URL")
    
    # Credentials
    api_key: Optional[str] = Field(default=None, env="BYBIT_API_KEY")
    api_secret: Optional[str] = Field(default=None, env="BYBIT_API_SECRET")
    testnet: bool = Field(default=True, env="BYBIT_TESTNET")
    
    # Rate limiting
    requests_per_second: int = Field(default=5, ge=1, le=20)
    requests_per_minute: int = Field(default=300, ge=10, le=1200)
    
    # Trading symbols
    symbols: List[str] = Field(default=["BTCUSDT", "ETHUSDT", "ICPUSDT"])
    
    # WebSocket settings
    ping_interval: int = Field(default=20, ge=10, le=60)
    max_reconnect_attempts: int = Field(default=10, ge=3, le=50)
    reconnect_delay: float = Field(default=5.0, ge=1.0, le=30.0)
    
    def get_api_url(self) -> str:
        """Get appropriate API URL based on testnet setting."""
        return self.testnet_url if self.testnet else self.base_url
    
    def get_ws_url(self) -> str:
        """Get appropriate WebSocket URL based on testnet setting."""
        return self.testnet_ws_url if self.testnet else self.ws_url
    
    @validator('api_key', 'api_secret')
    def validate_credentials(cls, v):
        """Validate API credentials are provided."""
        if v is None or v.strip() == "":
            raise ValueError("API credentials must be provided")
        return v.strip()


class MLConfig(BaseModel):
    """Machine learning configuration settings."""
    
    # Model settings
    model_path: str = Field(default="models/v3.1_improved/model.onnx", env="ML_MODEL_PATH")
    scaler_path: str = Field(default="models/v3.1_improved/scaler.pkl", env="ML_SCALER_PATH")
    feature_count: int = Field(default=44, ge=1, le=500)
    
    # Inference settings
    batch_size: int = Field(default=32, ge=1, le=1000)
    max_inference_time_ms: float = Field(default=100.0, ge=1.0, le=1000.0)
    
    # Feature engineering
    lookback_windows: List[int] = Field(default=[1, 5, 15, 30, 60])
    delta_values: List[float] = Field(default=[0.001, 0.002, 0.005])
    
    # Model validation
    min_model_accuracy: float = Field(default=0.65, ge=0.5, le=0.95)
    retrain_threshold_days: int = Field(default=7, ge=1, le=30)
    
    @validator('model_path', 'scaler_path')
    def validate_model_files(cls, v):
        """Validate model files exist."""
        if not Path(v).exists():
            raise ValueError(f"Model file not found: {v}")
        return str(Path(v).resolve())


class NotificationConfig(BaseModel):
    """Notification configuration settings."""
    
    # Discord settings
    discord_webhook_url: Optional[str] = Field(default=None, env="DISCORD_WEBHOOK_URL")
    discord_username: str = Field(default="MLBot", env="DISCORD_USERNAME")
    
    # Notification levels
    notify_trades: bool = Field(default=True, env="NOTIFY_TRADES")
    notify_errors: bool = Field(default=True, env="NOTIFY_ERRORS")
    notify_system_status: bool = Field(default=True, env="NOTIFY_SYSTEM_STATUS")
    notify_daily_reports: bool = Field(default=True, env="NOTIFY_DAILY_REPORTS")
    
    # Rate limiting for notifications
    max_notifications_per_hour: int = Field(default=20, ge=1, le=100)
    error_notification_cooldown: int = Field(default=300, ge=60, le=3600)
    
    @validator('discord_webhook_url')
    def validate_discord_webhook(cls, v):
        """Validate Discord webhook URL format."""
        if v and not v.startswith('https://discord.com/api/webhooks/'):
            raise ValueError("Invalid Discord webhook URL format")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    
    # Log levels
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    console_log_level: str = Field(default="INFO", env="CONSOLE_LOG_LEVEL")
    file_log_level: str = Field(default="DEBUG", env="FILE_LOG_LEVEL")
    
    # Log files
    log_dir: str = Field(default="logs", env="LOG_DIR")
    log_file: str = Field(default="trading_bot.log", env="LOG_FILE")
    max_log_size_mb: int = Field(default=100, ge=1, le=1000)
    backup_count: int = Field(default=5, ge=1, le=20)
    
    # Log format
    json_logs: bool = Field(default=True, env="JSON_LOGS")
    include_trace: bool = Field(default=False, env="INCLUDE_TRACE")
    
    @validator('log_level', 'console_log_level', 'file_log_level')
    def validate_log_levels(cls, v):
        """Validate log level values."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    @validator('log_dir')
    def create_log_dir(cls, v):
        """Ensure log directory exists."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v