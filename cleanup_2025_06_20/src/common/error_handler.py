"""
Centralized error handling for the trading bot.
"""

import sys
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .config_manager import get_notification_config
from .exceptions import TradingBotError
from .logging import get_logger
from .types import ErrorCallback, SystemStatus

logger = get_logger(__name__)


class ErrorHandler:
    """Centralized error handler with notification and recovery capabilities."""
    
    def __init__(self):
        self.error_callbacks: List[ErrorCallback] = []
        self.recovery_handlers: Dict[Type[Exception], Callable] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_error_times: Dict[str, float] = {}
        self.notification_config = get_notification_config()
    
    def register_callback(self, callback: ErrorCallback) -> None:
        """Register error callback."""
        self.error_callbacks.append(callback)
    
    def register_recovery_handler(
        self, 
        exception_type: Type[Exception], 
        handler: Callable
    ) -> None:
        """Register recovery handler for specific exception type."""
        self.recovery_handlers[exception_type] = handler
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        notify: bool = True,
        attempt_recovery: bool = True
    ) -> bool:
        """
        Handle error with logging, notification, and recovery.
        
        Args:
            error: Exception to handle
            context: Additional context information
            notify: Whether to send notifications
            attempt_recovery: Whether to attempt recovery
            
        Returns:
            True if error was recovered, False otherwise
        """
        error_type = type(error).__name__
        error_key = f"{error_type}:{str(error)}"
        
        # Update error statistics
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        import time
        self.last_error_times[error_key] = time.time()
        
        # Prepare error details
        error_details = {
            "error_type": error_type,
            "error_message": str(error),
            "error_count": self.error_counts[error_key],
            "context": context or {},
            "traceback": traceback.format_exc()
        }
        
        # Log error
        logger.error(
            f"Error handled: {error_type}",
            exception=error,
            **error_details
        )
        
        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as callback_error:
                logger.error(
                    "Error in error callback",
                    exception=callback_error
                )
        
        # Send notification if enabled
        if notify and self.notification_config.notify_errors:
            self._send_error_notification(error_details)
        
        # Attempt recovery
        recovered = False
        if attempt_recovery:
            recovered = self._attempt_recovery(error, context)
        
        return recovered
    
    def _send_error_notification(self, error_details: Dict[str, Any]) -> None:
        """Send error notification."""
        try:
            # Import here to avoid circular imports
            from .discord_notifier import discord_notifier
            
            # Check notification rate limiting
            import time
            error_key = f"{error_details['error_type']}:{error_details['error_message']}"
            last_time = self.last_error_times.get(f"notif_{error_key}", 0)
            
            if time.time() - last_time < self.notification_config.error_notification_cooldown:
                return
            
            self.last_error_times[f"notif_{error_key}"] = time.time()
            
            # Format notification
            title = f"🚨 Error: {error_details['error_type']}"
            description = (
                f"**Message:** {error_details['error_message']}\n"
                f"**Count:** {error_details['error_count']}\n"
                f"**Context:** {error_details['context']}"
            )
            
            # Truncate if too long
            if len(description) > 1900:
                description = description[:1900] + "..."
            
            discord_notifier.send_notification(
                title=title,
                description=description,
                color="ff0000"  # Red
            )
            
        except Exception as e:
            logger.error("Failed to send error notification", exception=e)
    
    def _attempt_recovery(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Attempt to recover from error."""
        error_type = type(error)
        
        # Try specific recovery handler
        if error_type in self.recovery_handlers:
            try:
                logger.info(f"Attempting recovery for {error_type.__name__}")
                recovery_result = self.recovery_handlers[error_type](error, context)
                
                if recovery_result:
                    logger.info(f"Successfully recovered from {error_type.__name__}")
                    return True
                else:
                    logger.warning(f"Recovery failed for {error_type.__name__}")
                    
            except Exception as recovery_error:
                logger.error(
                    f"Recovery handler failed for {error_type.__name__}",
                    exception=recovery_error
                )
        
        # Try base class recovery handlers
        for registered_type, handler in self.recovery_handlers.items():
            if issubclass(error_type, registered_type) and registered_type != error_type:
                try:
                    logger.info(f"Attempting base recovery for {error_type.__name__} using {registered_type.__name__}")
                    recovery_result = handler(error, context)
                    
                    if recovery_result:
                        logger.info(f"Successfully recovered from {error_type.__name__} using base handler")
                        return True
                        
                except Exception as recovery_error:
                    logger.error(
                        f"Base recovery handler failed for {error_type.__name__}",
                        exception=recovery_error
                    )
        
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts.copy(),
            "last_error_times": self.last_error_times.copy(),
            "total_errors": sum(self.error_counts.values()),
            "unique_errors": len(self.error_counts)
        }
    
    def clear_statistics(self) -> None:
        """Clear error statistics."""
        self.error_counts.clear()
        self.last_error_times.clear()


# Global error handler instance
error_handler = ErrorHandler()


@contextmanager
def error_context(
    context: Optional[Dict[str, Any]] = None,
    notify: bool = True,
    attempt_recovery: bool = True,
    re_raise: bool = True
):
    """
    Context manager for error handling.
    
    Args:
        context: Additional context information
        notify: Whether to send notifications
        attempt_recovery: Whether to attempt recovery
        re_raise: Whether to re-raise the exception
    """
    try:
        yield
    except Exception as e:
        recovered = error_handler.handle_error(
            error=e,
            context=context,
            notify=notify,
            attempt_recovery=attempt_recovery
        )
        
        if re_raise and not recovered:
            raise


def handle_critical_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    shutdown: bool = False
) -> None:
    """
    Handle critical errors that may require system shutdown.
    
    Args:
        error: Critical error
        context: Additional context
        shutdown: Whether to shutdown the system
    """
    logger.critical(
        f"Critical error occurred: {type(error).__name__}",
        exception=error,
        context=context or {}
    )
    
    # Always notify critical errors
    error_handler.handle_error(
        error=error,
        context={**(context or {}), "critical": True},
        notify=True,
        attempt_recovery=False
    )
    
    if shutdown:
        logger.critical("System shutdown requested due to critical error")
        # Import here to avoid circular imports
        try:
            from .discord_notifier import discord_notifier
            discord_notifier.send_notification(
                title="💥 Critical System Error",
                description=f"System shutdown due to: {str(error)}",
                color="000000"  # Black
            )
        except:
            pass
        
        sys.exit(1)


def setup_exception_hooks() -> None:
    """Setup global exception hooks."""
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow KeyboardInterrupt to work normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical(
            "Uncaught exception",
            exception=exc_value,
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        handle_critical_error(
            error=exc_value,
            context={"uncaught": True},
            shutdown=True
        )
    
    sys.excepthook = handle_exception


# Default recovery handlers
def default_api_error_recovery(error: Exception, context: Optional[Dict[str, Any]]) -> bool:
    """Default recovery for API errors."""
    logger.info("Attempting API error recovery with exponential backoff")
    
    import time
    import asyncio
    
    # Simple exponential backoff
    backoff_time = min(2 ** context.get('retry_count', 0), 60)
    
    if asyncio.iscoroutinefunction(context.get('retry_function')):
        # For async functions, we can't directly sleep here
        # The caller should handle the backoff
        return False
    else:
        time.sleep(backoff_time)
        return True


def default_database_error_recovery(error: Exception, context: Optional[Dict[str, Any]]) -> bool:
    """Default recovery for database errors."""
    logger.info("Attempting database error recovery")
    
    # Try to reconnect
    try:
        if context and 'reconnect_function' in context:
            context['reconnect_function']()
            return True
    except Exception as e:
        logger.error("Database reconnection failed", exception=e)
    
    return False


# Register default recovery handlers
error_handler.register_recovery_handler(Exception, lambda e, c: False)  # Default: no recovery