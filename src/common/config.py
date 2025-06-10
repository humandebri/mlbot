"""Configuration management for the trading bot."""

from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BybitConfig(BaseModel):
    """Bybit API configuration."""
    
    base_url: str = "https://api.bybit.com"
    testnet_url: str = "https://api-testnet.bybit.com"
    ws_url: str = "wss://stream.bybit.com/v5/public/linear"
    testnet_ws_url: str = "wss://stream-testnet.bybit.com/v5/public/linear"
    
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True
    
    # Rate limiting
    requests_per_second: int = 10
    requests_per_minute: int = 600
    
    # Symbols to trade
    symbols: List[str] = ["BTCUSDT", "ETHUSDT"]
    
    # WebSocket settings
    ping_interval: int = 20
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 5.0


class RedisConfig(BaseModel):
    """Redis configuration."""
    
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    # Connection pool settings
    max_connections: int = 20
    retry_on_timeout: bool = True
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    
    # Streams settings
    stream_maxlen: int = 10000
    batch_size: int = 100


class DuckDBConfig(BaseModel):
    """DuckDB configuration."""
    
    database_path: str = "data/trading_bot.db"
    memory_limit: str = "2GB"
    threads: int = 4
    
    # Parquet settings
    parquet_compression: str = "snappy"
    row_group_size: int = 100000


class ModelConfig(BaseModel):
    """ML model configuration."""
    
    model_path: str = "models/catboost_model.onnx"
    feature_columns: List[str] = []
    target_column: str = "expPNL"
    
    # Training settings
    train_size: float = 0.8
    validation_size: float = 0.1
    test_size: float = 0.1
    
    # CatBoost parameters
    iterations: int = 1000
    learning_rate: float = 0.1
    depth: int = 6
    l2_leaf_reg: float = 3.0
    
    # Feature engineering
    delta_values: List[float] = [0.02, 0.05, 0.10]
    lookahead_windows: List[int] = [60, 300, 900]
    
    # Inference settings
    inference_timeout: float = 0.001  # 1ms
    batch_inference: bool = True
    max_batch_size: int = 1000


class TradingConfig(BaseModel):
    """Trading configuration."""
    
    # Risk management
    max_position_pct: float = 0.10
    max_drawdown_pct: float = 0.05
    emergency_stop_pct: float = 0.10
    
    # Order settings
    min_order_size: float = 0.001
    max_order_size: float = 1.0
    order_timeout: int = 300  # seconds
    
    # Fees (Bybit VIP0)
    taker_fee: float = 0.0055
    maker_fee: float = -0.0015
    
    # Thresholds
    min_expected_pnl: float = 0.0
    min_fill_probability: float = 0.3
    
    # Position management
    default_hold_time: int = 300  # seconds
    max_open_positions: int = 5


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File settings
    log_file: str = "logs/trading_bot.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Structured logging
    use_json: bool = True
    add_caller_info: bool = True


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration."""
    
    prometheus_port: int = 9090
    health_check_port: int = 8080
    
    # Metrics collection
    collect_system_metrics: bool = True
    collect_business_metrics: bool = True
    metrics_interval: int = 10  # seconds
    
    # Alerting
    alert_webhook: Optional[str] = None
    slack_webhook: Optional[str] = None
    
    # Health checks
    redis_health_timeout: float = 1.0
    duckdb_health_timeout: float = 2.0
    model_health_timeout: float = 0.1


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Component configurations
    bybit: BybitConfig = Field(default_factory=BybitConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    duckdb: DuckDBConfig = Field(default_factory=DuckDBConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_testnet(self) -> bool:
        """Check if using testnet."""
        return self.bybit.testnet
    
    def get_bybit_urls(self) -> Dict[str, str]:
        """Get Bybit URLs based on environment."""
        if self.is_testnet:
            return {
                "rest": self.bybit.testnet_url,
                "websocket": self.bybit.testnet_ws_url,
            }
        return {
            "rest": self.bybit.base_url,
            "websocket": self.bybit.ws_url,
        }
    
    def ensure_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            Path(self.duckdb.database_path).parent,
            Path(self.model.model_path).parent,
            Path(self.logging.log_file).parent,
            Path("data"),
            Path("logs"),
            Path("models"),
            Path("artifacts"),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()