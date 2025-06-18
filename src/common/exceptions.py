"""
Custom exceptions for the trading bot.
"""

from typing import Any, Dict, Optional


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context
        }


class ConfigurationError(TradingBotError):
    """Raised when there's a configuration problem."""
    pass


class ExchangeError(TradingBotError):
    """Base exception for exchange-related errors."""
    pass


class APIError(ExchangeError):
    """Raised when API calls fail."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_data = response_data or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary with API details."""
        data = super().to_dict()
        data.update({
            "status_code": self.status_code,
            "response_data": self.response_data
        })
        return data


class WebSocketError(ExchangeError):
    """Raised when WebSocket operations fail."""
    pass


class ConnectionError(ExchangeError):
    """Raised when connection to exchange fails."""
    pass


class DataError(TradingBotError):
    """Base exception for data-related errors."""
    pass


class DatabaseError(DataError):
    """Raised when database operations fail."""
    pass


class RedisError(DatabaseError):
    """Raised when Redis operations fail."""
    pass


class DuckDBError(DatabaseError):
    """Raised when DuckDB operations fail."""
    pass


class MLError(TradingBotError):
    """Base exception for machine learning errors."""
    pass


class ModelError(MLError):
    """Raised when model operations fail."""
    pass


class FeatureError(MLError):
    """Raised when feature engineering fails."""
    pass


class PredictionError(MLError):
    """Raised when predictions fail."""
    pass


class TradingError(TradingBotError):
    """Base exception for trading-related errors."""
    pass


class RiskManagementError(TradingError):
    """Raised when risk limits are violated."""
    pass


class OrderError(TradingError):
    """Raised when order operations fail."""
    pass


class PositionError(TradingError):
    """Raised when position management fails."""
    pass


class NotificationError(TradingBotError):
    """Raised when notification sending fails."""
    pass


class ValidationError(TradingBotError):
    """Raised when data validation fails."""
    pass


class TimeoutError(TradingBotError):
    """Raised when operations timeout."""
    pass


class RateLimitError(TradingBotError):
    """Raised when rate limits are exceeded."""
    pass