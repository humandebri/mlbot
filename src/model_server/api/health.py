"""
Health monitoring and metrics API endpoints.

Provides comprehensive health checks and performance metrics for:
- Service health and readiness
- Model status and performance
- System resource monitoring
- Real-time metrics and statistics
"""

from fastapi import APIRouter, Depends, HTTPException
import psutil
import time
from datetime import datetime
from typing import Optional, Dict, Any

from ..schemas import HealthResponse, MetricsResponse, ModelInfo
from ..services.prediction_service import PredictionService
from ...common.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["Health & Monitoring"])


async def get_prediction_service() -> Optional[PredictionService]:
    """Get prediction service instance (optional for health checks)."""
    from .prediction import prediction_service
    return prediction_service


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    description="Comprehensive health check including model status and system metrics"
)
async def health_check(
    service: Optional[PredictionService] = Depends(get_prediction_service)
) -> HealthResponse:
    """
    Comprehensive health check for the Model Server.
    
    Returns:
    - Service status and readiness
    - Model information and performance
    - System resource usage
    - Service uptime and statistics
    """
    
    try:
        # Basic service info
        now = datetime.now()
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        system_metrics = {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_mb": memory.used / 1024 / 1024,
            "memory_usage_percent": memory.percent,
            "memory_available_mb": memory.available / 1024 / 1024,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        # Service status
        if service is None:
            return HealthResponse(
                status="unhealthy",
                timestamp=now,
                version="1.0.0",
                model_loaded=False,
                model_info=None,
                uptime_seconds=0.0,
                total_predictions=0,
                average_inference_time_ms=0.0,
                cache_size=0,
                system_metrics=system_metrics
            )
        
        # Get service performance stats
        performance_stats = service.get_performance_stats()
        
        # Model information
        model_info = None
        if service.model_info:
            model_info = service.model_info
        
        # Determine status
        status = "healthy"
        if not service.is_ready:
            status = "starting"
        elif performance_stats["error_rate"] > 0.1:  # More than 10% error rate
            status = "degraded"
        elif cpu_usage > 80 or memory.percent > 90:
            status = "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=now,
            version="1.0.0",
            model_loaded=service.is_ready,
            model_info=model_info,
            uptime_seconds=performance_stats["uptime_seconds"],
            total_predictions=performance_stats["total_predictions"],
            average_inference_time_ms=performance_stats["average_inference_time_ms"],
            cache_size=performance_stats["cache_size"],
            system_metrics=system_metrics
        )
        
    except Exception as e:
        logger.error("Health check error", exception=e)
        
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            model_loaded=False,
            model_info=None,
            uptime_seconds=0.0,
            total_predictions=0,
            average_inference_time_ms=0.0,
            cache_size=0,
            system_metrics={"error": str(e)}
        )


@router.get(
    "/ready",
    summary="Readiness Check",
    description="Simple readiness check for load balancers"
)
async def readiness_check(
    service: Optional[PredictionService] = Depends(get_prediction_service)
) -> dict:
    """
    Simple readiness check for Kubernetes/load balancer probes.
    
    Returns 200 if service is ready to accept requests.
    """
    
    if service is None or not service.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "ready",
        "timestamp": datetime.now(),
        "model_version": service.model_version
    }


@router.get(
    "/live",
    summary="Liveness Check",
    description="Simple liveness check for container orchestration"
)
async def liveness_check() -> dict:
    """
    Simple liveness check for Kubernetes liveness probes.
    
    Returns 200 if service process is alive.
    """
    
    return {
        "status": "alive",
        "timestamp": datetime.now(),
        "pid": psutil.Process().pid
    }


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Performance Metrics",
    description="Detailed performance metrics and statistics"
)
async def get_metrics(
    service: Optional[PredictionService] = Depends(get_prediction_service)
) -> MetricsResponse:
    """
    Get detailed performance metrics and statistics.
    
    Returns comprehensive metrics including:
    - Prediction throughput and latency
    - Error rates and cache performance
    - System resource usage
    - Model performance statistics
    """
    
    try:
        if service is None:
            raise HTTPException(status_code=503, detail="Service not available")
        
        # Get performance stats
        stats = service.get_performance_stats()
        
        # Calculate predictions per second
        uptime = stats["uptime_seconds"]
        predictions_per_second = stats["total_predictions"] / max(uptime, 1)
        
        # Inference time statistics
        inference_times = {
            "mean_ms": stats["average_inference_time_ms"],
            "p95_ms": stats["p95_inference_time_ms"],
            "p99_ms": stats["p99_inference_time_ms"]
        }
        
        # Add p50 if we have inference times
        if service.inference_times:
            import numpy as np
            inference_times["p50_ms"] = float(np.percentile(service.inference_times, 50))
        else:
            inference_times["p50_ms"] = 0.0
        
        # System resources
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        system_resources = {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory.percent,
            "memory_used_mb": memory.used / 1024 / 1024,
            "processes": len(psutil.pids()),
            "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        }
        
        # Model performance (simplified)
        model_performance = {
            "total_predictions": stats["total_predictions"],
            "successful_predictions": stats["total_predictions"] - stats["total_errors"],
            "model_version": service.model_version,
            "cache_utilization": stats["cache_size"] / 1000  # Assuming max cache size of 1000
        }
        
        return MetricsResponse(
            total_predictions=stats["total_predictions"],
            predictions_per_second=predictions_per_second,
            inference_times=inference_times,
            error_rate=stats["error_rate"],
            cache_hit_rate=0.0,  # TODO: Implement cache hit tracking
            model_performance=model_performance,
            system_resources=system_resources
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Metrics collection error", exception=e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to collect metrics: {str(e)}"
        )


@router.get(
    "/model",
    summary="Model Information",
    description="Current model information and metadata"
)
async def get_model_info(
    service: Optional[PredictionService] = Depends(get_prediction_service)
) -> dict:
    """
    Get current model information and metadata.
    
    Returns model version, performance stats, and configuration.
    """
    
    if service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    
    try:
        model_info = {
            "model_version": service.model_version,
            "model_path": service.model_path,
            "model_loaded": service.is_ready,
            "load_time": getattr(service, 'last_load_time', None),
            "auto_reload_enabled": service.auto_reload_enabled
        }
        
        if service.model_info:
            model_info.update({
                "model_type": service.model_info.model_type,
                "feature_count": service.model_info.feature_count,
                "model_size_mb": service.model_info.model_size_mb,
                "created_at": service.model_info.created_at,
                "training_metrics": service.model_info.training_metrics
            })
        
        return model_info
        
    except Exception as e:
        logger.error("Model info error", exception=e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Real-time Statistics",
    description="Real-time service statistics and counters"
)
async def get_realtime_stats(
    service: Optional[PredictionService] = Depends(get_prediction_service)
) -> dict:
    """
    Get real-time service statistics.
    
    Returns current counters and performance indicators.
    """
    
    if service is None:
        return {
            "service_available": False,
            "timestamp": datetime.now(),
            "uptime_seconds": 0
        }
    
    try:
        stats = service.get_performance_stats()
        
        # Recent performance (last 100 predictions)
        recent_times = service.inference_times[-100:] if service.inference_times else []
        recent_avg = sum(recent_times) / len(recent_times) if recent_times else 0
        
        return {
            "service_available": True,
            "timestamp": datetime.now(),
            "uptime_seconds": stats["uptime_seconds"],
            "total_predictions": stats["total_predictions"],
            "total_errors": stats["total_errors"],
            "recent_average_inference_ms": recent_avg,
            "cache_size": stats["cache_size"],
            "model_version": service.model_version,
            "memory_usage_mb": psutil.virtual_memory().used / 1024 / 1024,
            "cpu_usage_percent": psutil.cpu_percent()
        }
        
    except Exception as e:
        logger.error("Real-time stats error", exception=e)
        return {
            "service_available": False,
            "error": str(e),
            "timestamp": datetime.now()
        }