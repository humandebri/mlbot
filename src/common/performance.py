"""
Performance monitoring and optimization utilities.
"""

import asyncio
import functools
import gc
import psutil
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .logging import get_logger
from .utils import CircularBuffer

logger = get_logger(__name__)
F = TypeVar('F', bound=Callable[..., Any])


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.execution_times: Dict[str, CircularBuffer] = defaultdict(
            lambda: CircularBuffer(max_samples)
        )
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.memory_usage: CircularBuffer = CircularBuffer(max_samples)
        self.cpu_usage: CircularBuffer = CircularBuffer(max_samples)
        
        # Start background monitoring
        self._monitoring = True
        # Only start monitoring if we're in an async context
        try:
            asyncio.create_task(self._monitor_system_resources())
        except RuntimeError:
            # No event loop running, skip background monitoring
            pass
    
    async def _monitor_system_resources(self) -> None:
        """Monitor system resources in background."""
        while self._monitoring:
            try:
                # Memory usage
                memory_percent = psutil.virtual_memory().percent
                self.memory_usage.append(memory_percent)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.append(cpu_percent)
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error("Error monitoring system resources", exception=e)
                await asyncio.sleep(60)  # Wait longer on error
    
    def record_execution(self, function_name: str, execution_time: float) -> None:
        """Record function execution time."""
        self.execution_times[function_name].append(execution_time)
        self.call_counts[function_name] += 1
    
    def record_error(self, function_name: str) -> None:
        """Record function error."""
        self.error_counts[function_name] += 1
    
    def get_function_stats(self, function_name: str) -> Dict[str, Any]:
        """Get statistics for a specific function."""
        times = self.execution_times[function_name].get_items()
        
        if not times:
            return {
                "function": function_name,
                "call_count": self.call_counts[function_name],
                "error_count": self.error_counts[function_name],
                "avg_time": 0.0,
                "min_time": 0.0,
                "max_time": 0.0,
                "total_time": 0.0
            }
        
        return {
            "function": function_name,
            "call_count": self.call_counts[function_name],
            "error_count": self.error_counts[function_name],
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times),
            "samples": len(times)
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system resource statistics."""
        memory_samples = self.memory_usage.get_items()
        cpu_samples = self.cpu_usage.get_items()
        
        stats = {
            "memory": {
                "current": psutil.virtual_memory().percent,
                "available": psutil.virtual_memory().available / (1024**3),  # GB
                "total": psutil.virtual_memory().total / (1024**3),  # GB
            },
            "cpu": {
                "current": psutil.cpu_percent(),
                "count": psutil.cpu_count(),
            },
            "disk": {
                "usage": psutil.disk_usage('/').percent,
                "free": psutil.disk_usage('/').free / (1024**3),  # GB
            }
        }
        
        if memory_samples:
            stats["memory"].update({
                "avg": sum(memory_samples) / len(memory_samples),
                "max": max(memory_samples),
                "min": min(memory_samples)
            })
        
        if cpu_samples:
            stats["cpu"].update({
                "avg": sum(cpu_samples) / len(cpu_samples),
                "max": max(cpu_samples),
                "min": min(cpu_samples)
            })
        
        return stats
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        # Function statistics
        function_stats = {}
        for func_name in self.call_counts.keys():
            function_stats[func_name] = self.get_function_stats(func_name)
        
        # Find top performers and problematic functions
        sorted_by_time = sorted(
            function_stats.values(),
            key=lambda x: x["total_time"],
            reverse=True
        )
        
        sorted_by_errors = sorted(
            function_stats.values(),
            key=lambda x: x["error_count"],
            reverse=True
        )
        
        return {
            "system": self.get_system_stats(),
            "functions": {
                "total_functions": len(function_stats),
                "top_time_consumers": sorted_by_time[:10],
                "most_errors": sorted_by_errors[:10],
                "all_stats": function_stats
            },
            "summary": {
                "total_calls": sum(self.call_counts.values()),
                "total_errors": sum(self.error_counts.values()),
                "error_rate": sum(self.error_counts.values()) / max(sum(self.call_counts.values()), 1)
            }
        }
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False


# Global performance monitor
performance_monitor = PerformanceMonitor()


def profile_performance(include_memory: bool = False, include_cpu: bool = False):
    """
    Decorator to profile function performance.
    
    Args:
        include_memory: Whether to track memory usage
        include_cpu: Whether to track CPU usage
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.virtual_memory().percent if include_memory else None
            start_cpu = psutil.cpu_percent() if include_cpu else None
            
            try:
                result = await func(*args, **kwargs)
                
                # Record successful execution
                execution_time = time.time() - start_time
                performance_monitor.record_execution(func.__name__, execution_time)
                
                # Log performance info
                log_data = {"execution_time": f"{execution_time:.3f}s"}
                
                if include_memory and start_memory is not None:
                    memory_diff = psutil.virtual_memory().percent - start_memory
                    log_data["memory_change"] = f"{memory_diff:+.1f}%"
                
                if include_cpu and start_cpu is not None:
                    cpu_diff = psutil.cpu_percent() - start_cpu
                    log_data["cpu_change"] = f"{cpu_diff:+.1f}%"
                
                logger.debug(f"Performance: {func.__name__}", **log_data)
                
                return result
                
            except Exception as e:
                # Record error
                performance_monitor.record_error(func.__name__)
                
                execution_time = time.time() - start_time
                logger.error(
                    f"Performance error in {func.__name__}",
                    exception=e,
                    execution_time=f"{execution_time:.3f}s"
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.virtual_memory().percent if include_memory else None
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful execution
                execution_time = time.time() - start_time
                performance_monitor.record_execution(func.__name__, execution_time)
                
                # Log performance info
                log_data = {"execution_time": f"{execution_time:.3f}s"}
                
                if include_memory and start_memory is not None:
                    memory_diff = psutil.virtual_memory().percent - start_memory
                    log_data["memory_change"] = f"{memory_diff:+.1f}%"
                
                logger.debug(f"Performance: {func.__name__}", **log_data)
                
                return result
                
            except Exception as e:
                # Record error
                performance_monitor.record_error(func.__name__)
                
                execution_time = time.time() - start_time
                logger.error(
                    f"Performance error in {func.__name__}",
                    exception=e,
                    execution_time=f"{execution_time:.3f}s"
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


@contextmanager
def performance_context(operation_name: str):
    """Context manager for measuring operation performance."""
    start_time = time.time()
    start_memory = psutil.virtual_memory().percent
    
    try:
        yield
        
        # Record successful operation
        execution_time = time.time() - start_time
        performance_monitor.record_execution(operation_name, execution_time)
        
        memory_diff = psutil.virtual_memory().percent - start_memory
        logger.debug(
            f"Operation completed: {operation_name}",
            execution_time=f"{execution_time:.3f}s",
            memory_change=f"{memory_diff:+.1f}%"
        )
        
    except Exception as e:
        # Record error
        performance_monitor.record_error(operation_name)
        
        execution_time = time.time() - start_time
        logger.error(
            f"Operation failed: {operation_name}",
            exception=e,
            execution_time=f"{execution_time:.3f}s"
        )
        raise


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def force_garbage_collection() -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        before_objects = len(gc.get_objects())
        collected = gc.collect()
        after_objects = len(gc.get_objects())
        
        stats = {
            "objects_before": before_objects,
            "objects_after": after_objects,
            "objects_collected": before_objects - after_objects,
            "garbage_collected": collected
        }
        
        logger.info("Garbage collection completed", **stats)
        return stats
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get detailed memory usage information."""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "system_total_gb": memory.total / (1024**3),
            "system_available_gb": memory.available / (1024**3),
            "system_used_percent": memory.percent,
            "process_rss_mb": process_memory.rss / (1024**2),
            "process_vms_mb": process_memory.vms / (1024**2),
            "process_percent": process.memory_percent()
        }
    
    @staticmethod
    def optimize_memory():
        """Perform memory optimization."""
        # Force garbage collection
        MemoryOptimizer.force_garbage_collection()
        
        # Log memory usage after optimization
        memory_info = MemoryOptimizer.get_memory_usage()
        logger.info("Memory optimization completed", **memory_info)


class CPUOptimizer:
    """CPU optimization utilities."""
    
    @staticmethod
    def get_cpu_usage() -> Dict[str, Any]:
        """Get detailed CPU usage information."""
        return {
            "usage_percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True),
            "freq_current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    
    @staticmethod
    async def yield_control():
        """Yield control to other tasks."""
        await asyncio.sleep(0)
    
    @staticmethod
    def set_process_priority(priority: int = 0):
        """Set process priority (-20 to 20, lower is higher priority)."""
        try:
            process = psutil.Process()
            process.nice(priority)
            logger.info(f"Process priority set to {priority}")
        except Exception as e:
            logger.error("Failed to set process priority", exception=e)


def optimize_performance():
    """Run comprehensive performance optimization."""
    logger.info("Starting performance optimization")
    
    # Memory optimization
    MemoryOptimizer.optimize_memory()
    
    # CPU information
    cpu_info = CPUOptimizer.get_cpu_usage()
    logger.info("CPU status", **cpu_info)
    
    # Get performance report
    perf_report = performance_monitor.get_performance_report()
    logger.info(
        "Performance summary",
        total_calls=perf_report["summary"]["total_calls"],
        total_errors=perf_report["summary"]["total_errors"],
        error_rate=f"{perf_report['summary']['error_rate']:.2%}"
    )
    
    logger.info("Performance optimization completed")
    return perf_report