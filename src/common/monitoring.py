"""Monitoring, health checks, and metrics collection."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

import psutil
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from prometheus_client.core import CollectorRegistry

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)

# Prometheus metrics
REGISTRY = CollectorRegistry()

# System metrics
CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU usage percentage", registry=REGISTRY)
MEMORY_USAGE = Gauge("system_memory_usage_bytes", "Memory usage in bytes", registry=REGISTRY)
DISK_USAGE = Gauge("system_disk_usage_percent", "Disk usage percentage", registry=REGISTRY)
SYSTEM_HEALTH = Gauge("system_health_score", "Overall system health score (0-1)", registry=REGISTRY)

# Application metrics
WEBSOCKET_CONNECTIONS = Gauge(
    "websocket_connections_active", "Active WebSocket connections", ["symbol"], registry=REGISTRY
)
MESSAGES_RECEIVED = Counter(
    "messages_received_total", "Total messages received", ["source", "symbol"], registry=REGISTRY
)
MESSAGES_PROCESSED = Counter(
    "messages_processed_total", "Total messages processed", ["component", "symbol"], registry=REGISTRY
)
PROCESSING_TIME = Histogram(
    "processing_time_seconds", "Processing time in seconds", ["component"], registry=REGISTRY
)
MODEL_PREDICTIONS = Counter(
    "trading_predictions_total", "Total model predictions made", ["symbol"], registry=REGISTRY
)
MODEL_INFERENCE_TIME = Histogram(
    "model_inference_time_seconds", "Model inference time in seconds", registry=REGISTRY
)

# Trading metrics
ORDERS_PLACED = Counter(
    "orders_placed_total", "Total orders placed", ["symbol", "side"], registry=REGISTRY
)
ORDERS_FILLED = Counter(
    "orders_filled_total", "Total orders filled", ["symbol", "side"], registry=REGISTRY
)
ORDERS_CANCELLED = Counter(
    "orders_cancelled_total", "Total orders cancelled", ["symbol", "reason"], registry=REGISTRY
)
ORDER_LATENCY = Histogram(
    "order_latency_seconds", "Order placement latency in seconds", ["symbol"], registry=REGISTRY
)
POSITION_PNL = Gauge("position_pnl", "Current position PnL", ["symbol"], registry=REGISTRY)
PORTFOLIO_VALUE = Gauge("portfolio_value", "Total portfolio value", registry=REGISTRY)
DRAWDOWN = Gauge("max_drawdown_percent", "Maximum drawdown percentage", registry=REGISTRY)

# Model metrics
MODEL_ERRORS = Counter(
    "model_errors_total", "Total model errors", ["symbol"], registry=REGISTRY
)
SIGNALS_GENERATED = Counter(
    "signals_generated_total", "Total trading signals generated", ["symbol", "signal_type"], registry=REGISTRY
)

# Risk management metrics
RISK_VIOLATIONS = Counter(
    "risk_violations_total", "Total risk violations", ["violation_type"], registry=REGISTRY
)
POSITION_VALUE = Gauge(
    "position_value_usd", "Current position value in USD", ["symbol"], registry=REGISTRY
)
DAILY_PNL = Gauge("daily_pnl_usd", "Daily PnL in USD", registry=REGISTRY)

# Position management metrics
ACTIVE_POSITIONS = Gauge(
    "active_positions_count", "Number of active positions", ["symbol"], registry=REGISTRY
)
TOTAL_PNL = Gauge("total_pnl_usd", "Total PnL in USD", registry=REGISTRY)
WIN_RATE = Gauge("win_rate_percent", "Win rate percentage", registry=REGISTRY)

# Error metrics
ERRORS_TOTAL = Counter(
    "errors_total", "Total errors", ["component", "error_type"], registry=REGISTRY
)
WEBSOCKET_RECONNECTS = Counter(
    "websocket_reconnects_total", "WebSocket reconnections", ["symbol"], registry=REGISTRY
)

# Additional metrics for liquidation processing
LIQUIDATION_VOLUME_5S = Gauge(
    "liquidation_volume_5s", "Liquidation volume in last 5 seconds", ["symbol"], registry=REGISTRY
)
LIQUIDATION_SPIKE_Z_SCORE = Gauge(
    "liquidation_spike_z_score", "Liquidation spike Z-score", ["symbol"], registry=REGISTRY
)
LIQUIDATION_SIDE_RATIO = Gauge(
    "liquidation_side_ratio", "Liquidation side ratio (sell/total)", ["symbol"], registry=REGISTRY
)
LIQUIDATION_SPIKES_DETECTED = Counter(
    "liquidation_spikes_detected", "Total liquidation spikes detected", ["symbol", "type"], registry=REGISTRY
)
LIQUIDATION_SPIKE_SEVERITY = Histogram(
    "liquidation_spike_severity", "Liquidation spike severity distribution", ["symbol"], registry=REGISTRY
)

# Archiver metrics
ARCHIVE_BATCH_SIZE = Histogram(
    "archive_batch_size", "Archive batch size distribution", ["data_type"], registry=REGISTRY
)
BATCH_FLUSH_SIZE = Histogram(
    "batch_flush_size", "Batch flush size distribution", ["data_type"], registry=REGISTRY
)
REDIS_WRITE_ERRORS = Counter(
    "redis_write_errors", "Redis write errors", ["data_type"], registry=REGISTRY
)
ARCHIVER_ERRORS = Counter(
    "archiver_errors", "Archiver errors", ["error_type"], registry=REGISTRY
)
INGESTOR_ERRORS = Counter(
    "ingestor_errors", "Ingestor processing errors", ["error_type"], registry=REGISTRY
)

# FeatureHub metrics
FEATURE_VECTOR_SIZE = Gauge(
    "feature_vector_size", "Size of feature vector", ["symbol"], registry=REGISTRY
)
FEATURE_COMPUTATION_LATENCY = Histogram(
    "feature_computation_latency", "Feature computation latency", ["symbol"], registry=REGISTRY
)
FEATURE_HUB_ERRORS = Counter(
    "feature_hub_errors", "FeatureHub processing errors", ["error_type"], registry=REGISTRY
)


class HealthChecker:
    """Health check manager for all components."""
    
    def __init__(self):
        self.health_status: Dict[str, bool] = {}
        self.last_check: Dict[str, float] = {}
    
    async def check_redis_health(self) -> bool:
        """Check Redis connection health."""
        try:
            from ..common.database import get_redis_client
            
            redis_client = await get_redis_client()
            await asyncio.wait_for(
                redis_client.ping(), timeout=settings.monitoring.redis_health_timeout
            )
            return True
        except Exception as e:
            logger.error("Redis health check failed", exception=e)
            return False
    
    async def check_duckdb_health(self) -> bool:
        """Check DuckDB connection health."""
        try:
            from ..common.database import get_duckdb_connection
            
            conn = get_duckdb_connection()
            await asyncio.wait_for(
                asyncio.to_thread(conn.execute, "SELECT 1"),
                timeout=settings.monitoring.duckdb_health_timeout
            )
            return True
        except Exception as e:
            logger.error("DuckDB health check failed", exception=e)
            return False
    
    async def check_model_health(self) -> bool:
        """Check model server health."""
        try:
            # This would typically make a test prediction
            # For now, just check if model file exists
            from pathlib import Path
            
            model_path = Path(settings.model.model_path)
            return model_path.exists()
        except Exception as e:
            logger.error("Model health check failed", exception=e)
            return False
    
    async def run_all_checks(self) -> Dict[str, bool]:
        """Run all health checks."""
        checks = {
            "redis": self.check_redis_health(),
            "duckdb": self.check_duckdb_health(),
            "model": self.check_model_health(),
        }
        
        results = {}
        for name, check_coro in checks.items():
            try:
                results[name] = await check_coro
                self.health_status[name] = results[name]
                self.last_check[name] = time.time()
            except Exception as e:
                logger.error(f"Health check failed for {name}", exception=e)
                results[name] = False
                self.health_status[name] = False
        
        return results
    
    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        return all(self.health_status.values())
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed health status."""
        return {
            "healthy": self.is_healthy(),
            "components": {
                name: {
                    "healthy": status,
                    "last_check": self.last_check.get(name, 0),
                }
                for name, status in self.health_status.items()
            },
        }


class MetricsCollector:
    """System and application metrics collector."""
    
    def __init__(self):
        self.running = False
        self.task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start metrics collection."""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._collect_loop())
        logger.info("Metrics collection started")
    
    async def stop(self) -> None:
        """Stop metrics collection."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collection stopped")
    
    async def _collect_loop(self) -> None:
        """Main metrics collection loop."""
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(settings.monitoring.metrics_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in metrics collection", exception=e)
                await asyncio.sleep(settings.monitoring.metrics_interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        if not settings.monitoring.collect_system_metrics:
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            CPU_USAGE.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)
            
            # Disk usage
            disk = psutil.disk_usage("/")
            DISK_USAGE.set(disk.percent)
            
        except Exception as e:
            logger.error("Error collecting system metrics", exception=e)


@asynccontextmanager
async def measure_time(component: str) -> AsyncGenerator[None, None]:
    """Context manager to measure processing time."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        PROCESSING_TIME.labels(component=component).observe(duration)


def increment_counter(metric: Counter, **labels: str) -> None:
    """Safely increment a counter metric."""
    try:
        metric.labels(**labels).inc()
    except Exception as e:
        logger.error(f"Error incrementing counter {metric._name}", exception=e)


def set_gauge(metric: Gauge, value: float, **labels: str) -> None:
    """Safely set a gauge metric."""
    try:
        if labels:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)
    except Exception as e:
        logger.error(f"Error setting gauge {metric._name}", exception=e)


def observe_histogram(metric: Histogram, value: float, **labels: str) -> None:
    """Safely observe a histogram metric."""
    try:
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)
    except Exception as e:
        logger.error(f"Error observing histogram {metric._name}", exception=e)


async def start_monitoring() -> tuple[HealthChecker, MetricsCollector]:
    """Start monitoring services."""
    # Start Prometheus metrics server
    if settings.monitoring.prometheus_port:
        start_http_server(settings.monitoring.prometheus_port, registry=REGISTRY)
        logger.info(f"Prometheus metrics server started on port {settings.monitoring.prometheus_port}")
    
    # Initialize health checker and metrics collector
    health_checker = HealthChecker()
    metrics_collector = MetricsCollector()
    
    # Start metrics collection
    await metrics_collector.start()
    
    # Initial health check
    await health_checker.run_all_checks()
    
    return health_checker, metrics_collector


def setup_metrics_server(port: Optional[int] = None) -> None:
    """Setup and start the Prometheus metrics server.
    
    Args:
        port: Port to run the metrics server on. If None, uses config value.
    """
    if port is None:
        port = settings.monitoring.prometheus_port
    
    if port:
        try:
            start_http_server(port, registry=REGISTRY)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server on port {port}", exception=e)
            raise