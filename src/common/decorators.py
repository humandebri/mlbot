"""
Common decorators for the trading bot.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from .exceptions import RateLimitError, TimeoutError, TradingBotError
from .logging import get_logger

F = TypeVar('F', bound=Callable[..., Any])
logger = get_logger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_failure: Optional[Callable] = None
):
    """
    Retry decorator for functions that may fail temporarily.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
        on_failure: Callback function called on final failure
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        if on_failure:
                            on_failure(e, attempt + 1)
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts",
                            exception=e,
                            attempts=max_attempts
                        )
                        raise
                    
                    logger.warning(
                        f"Function {func.__name__} failed, retrying in {current_delay}s",
                        exception=e,
                        attempt=attempt + 1,
                        max_attempts=max_attempts
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        if on_failure:
                            on_failure(e, attempt + 1)
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts",
                            exception=e,
                            attempts=max_attempts
                        )
                        raise
                    
                    logger.warning(
                        f"Function {func.__name__} failed, retrying in {current_delay}s",
                        exception=e,
                        attempt=attempt + 1,
                        max_attempts=max_attempts
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def timeout(seconds: float):
    """
    Timeout decorator for async functions.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {seconds} seconds"
                )
        
        return wrapper
    
    return decorator


def rate_limit(calls_per_second: float = 1.0):
    """
    Rate limiting decorator.
    
    Args:
        calls_per_second: Maximum calls per second allowed
    """
    min_interval = 1.0 / calls_per_second
    last_called = {}
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            now = time.time()
            key = f"{func.__module__}.{func.__name__}"
            
            if key in last_called:
                elapsed = now - last_called[key]
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    await asyncio.sleep(sleep_time)
            
            last_called[key] = time.time()
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            now = time.time()
            key = f"{func.__module__}.{func.__name__}"
            
            if key in last_called:
                elapsed = now - last_called[key]
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    time.sleep(sleep_time)
            
            last_called[key] = time.time()
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def log_execution(
    level: str = "INFO",
    include_args: bool = False,
    include_result: bool = False,
    include_time: bool = True
):
    """
    Logging decorator for function execution.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        include_args: Whether to log function arguments
        include_result: Whether to log function result
        include_time: Whether to log execution time
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time() if include_time else None
            
            # Log function entry
            log_data = {"function": func.__name__}
            if include_args:
                log_data["args"] = str(args)
                log_data["kwargs"] = str(kwargs)
            
            logger.log(level, f"Executing {func.__name__}", **log_data)
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful completion
                completion_data = {"function": func.__name__}
                if include_time and start_time:
                    completion_data["execution_time"] = f"{time.time() - start_time:.3f}s"
                if include_result:
                    completion_data["result"] = str(result)
                
                logger.log(level, f"Completed {func.__name__}", **completion_data)
                return result
                
            except Exception as e:
                # Log function failure
                error_data = {"function": func.__name__, "exception": str(e)}
                if include_time and start_time:
                    error_data["execution_time"] = f"{time.time() - start_time:.3f}s"
                
                logger.error(f"Failed {func.__name__}", **error_data)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time() if include_time else None
            
            # Log function entry
            log_data = {"function": func.__name__}
            if include_args:
                log_data["args"] = str(args)
                log_data["kwargs"] = str(kwargs)
            
            logger.log(level, f"Executing {func.__name__}", **log_data)
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                completion_data = {"function": func.__name__}
                if include_time and start_time:
                    completion_data["execution_time"] = f"{time.time() - start_time:.3f}s"
                if include_result:
                    completion_data["result"] = str(result)
                
                logger.log(level, f"Completed {func.__name__}", **completion_data)
                return result
                
            except Exception as e:
                # Log function failure
                error_data = {"function": func.__name__, "exception": str(e)}
                if include_time and start_time:
                    error_data["execution_time"] = f"{time.time() - start_time:.3f}s"
                
                logger.error(f"Failed {func.__name__}", **error_data)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def validate_types(**expected_types):
    """
    Type validation decorator.
    
    Args:
        **expected_types: Keyword arguments mapping parameter names to expected types
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature for parameter names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate types
            for param_name, expected_type in expected_types.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' expected {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def cache_result(ttl: float = 300.0):
    """
    Simple caching decorator with TTL.
    
    Args:
        ttl: Time to live in seconds
    """
    cache = {}
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            
            # Check cache
            if key in cache:
                cached_time, cached_result = cache[key]
                if now - cached_time < ttl:
                    return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache[key] = (now, result)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            
            # Check cache
            if key in cache:
                cached_time, cached_result = cache[key]
                if now - cached_time < ttl:
                    return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = (now, result)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def error_handler(
    exception_types: Union[Type[Exception], tuple] = Exception,
    default_return: Any = None,
    re_raise: bool = False,
    log_error: bool = True
):
    """
    Error handling decorator.
    
    Args:
        exception_types: Exception types to catch
        default_return: Default value to return on error
        re_raise: Whether to re-raise the exception after handling
        log_error: Whether to log the error
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exception_types as e:
                if log_error:
                    logger.error(
                        f"Error in {func.__name__}",
                        exception=e,
                        function=func.__name__
                    )
                
                if re_raise:
                    raise
                
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                if log_error:
                    logger.error(
                        f"Error in {func.__name__}",
                        exception=e,
                        function=func.__name__
                    )
                
                if re_raise:
                    raise
                
                return default_return
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator