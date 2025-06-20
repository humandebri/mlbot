"""Structured logging setup for the trading bot."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.typing import Processor

from .config import settings


def setup_logging() -> None:
    """Setup structured logging with both standard and structured loggers."""
    
    # Ensure log directory exists
    log_file = Path(settings.logging.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure standard library logging
    _setup_stdlib_logging()
    
    # Configure structlog
    _setup_structlog()


def _setup_stdlib_logging() -> None:
    """Setup standard library logging."""
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.logging.level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.logging.level.upper()))
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        settings.logging.log_file,
        maxBytes=settings.logging.max_file_size,
        backupCount=settings.logging.backup_count,
    )
    file_handler.setLevel(getattr(logging, settings.logging.level.upper()))
    
    # Formatters
    if settings.logging.use_json:
        formatter = logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        formatter = logging.Formatter(settings.logging.format)
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("aiohttp").setLevel(logging.INFO)


def _setup_structlog() -> None:
    """Setup structlog configuration."""
    
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    
    if settings.logging.add_caller_info:
        processors.append(structlog.processors.CallsiteParameterAdder())
    
    processors.extend([
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.dict_tracebacks,
    ])
    
    if settings.debug:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.logging.level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class TradingLogger:
    """Specialized logger for trading events."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def log_order(
        self,
        action: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        order_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log order-related events."""
        self.logger.info(
            "Order event",
            action=action,
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            order_id=order_id,
            **kwargs,
        )
    
    def log_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        order_id: str,
        fee: float,
        **kwargs: Any,
    ) -> None:
        """Log order fill events."""
        self.logger.info(
            "Order filled",
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            order_id=order_id,
            fee=fee,
            **kwargs,
        )
    
    def log_position(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        mark_price: float,
        pnl: float,
        **kwargs: Any,
    ) -> None:
        """Log position updates."""
        self.logger.info(
            "Position update",
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            mark_price=mark_price,
            pnl=pnl,
            **kwargs,
        )
    
    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        features: Dict[str, float],
        delta: float,
        lookahead: int,
        **kwargs: Any,
    ) -> None:
        """Log trading signals."""
        self.logger.info(
            "Trading signal",
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            features=features,
            delta=delta,
            lookahead=lookahead,
            **kwargs,
        )
    
    def log_error(
        self,
        error_type: str,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Log errors with context."""
        self.logger.error(
            "Error occurred",
            error_type=error_type,
            message=message,
            exception=str(exception) if exception else None,
            **kwargs,
        )
    
    def log_performance(
        self,
        period: str,
        total_pnl: float,
        win_rate: float,
        sharpe_ratio: float,
        max_drawdown: float,
        **kwargs: Any,
    ) -> None:
        """Log performance metrics."""
        self.logger.info(
            "Performance metrics",
            period=period,
            total_pnl=total_pnl,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            **kwargs,
        )