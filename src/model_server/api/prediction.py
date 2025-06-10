"""
Prediction API endpoints for real-time and batch inference.

High-performance REST API endpoints providing:
- Single prediction with sub-millisecond latency
- Batch prediction with optimal throughput
- Configurable prediction parameters
- Comprehensive error handling
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Optional
import asyncio
import time
from datetime import datetime

from ..schemas import (
    FeatureInput, BatchFeatureInput, PredictionResponse, BatchPredictionResponse,
    PredictionConfig, BatchConfig, ErrorResponse
)
from ..services.prediction_service import PredictionService
from ...common.logging import get_logger

logger = get_logger(__name__)

# Global prediction service instance
prediction_service: Optional[PredictionService] = None

router = APIRouter(prefix="/predict", tags=["Prediction"])


async def get_prediction_service() -> PredictionService:
    """Dependency to get prediction service instance."""
    global prediction_service
    
    if prediction_service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not initialized"
        )
    
    if not prediction_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not ready"
        )
    
    return prediction_service


def set_prediction_service(service: PredictionService) -> None:
    """Set the global prediction service instance."""
    global prediction_service
    prediction_service = service


@router.post(
    "/single",
    response_model=PredictionResponse,
    summary="Single Prediction",
    description="Make a single real-time prediction with sub-millisecond latency"
)
async def predict_single(
    feature_input: FeatureInput,
    config: Optional[PredictionConfig] = None,
    service: PredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Make a single prediction for liquidation-driven trading.
    
    - **features**: Dictionary of feature names and values
    - **symbol**: Trading symbol (default: BTCUSDT)
    - **timestamp**: Optional timestamp for the prediction
    - **config**: Optional prediction configuration
    
    Returns prediction with confidence score and metadata.
    """
    try:
        logger.debug("Single prediction request", 
                    symbol=feature_input.symbol,
                    feature_count=len(feature_input.features))
        
        start_time = time.perf_counter()
        
        # Make prediction
        result = await service.predict(feature_input, config)
        
        # Log performance
        request_time = (time.perf_counter() - start_time) * 1000
        logger.debug("Single prediction completed",
                    symbol=feature_input.symbol,
                    request_time_ms=request_time,
                    inference_time_ms=result.inference_time_ms,
                    prediction=result.prediction,
                    confidence=result.confidence)
        
        return result
        
    except Exception as e:
        logger.error("Single prediction error", 
                    exception=e,
                    symbol=feature_input.symbol)
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Prediction",
    description="Make batch predictions with optimal throughput"
)
async def predict_batch(
    batch_input: BatchFeatureInput,
    config: Optional[BatchConfig] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    service: PredictionService = Depends(get_prediction_service)
) -> BatchPredictionResponse:
    """
    Make batch predictions for multiple feature sets.
    
    - **features**: List of feature dictionaries
    - **symbol**: Trading symbol for all predictions
    - **batch_id**: Optional batch identifier for tracking
    - **config**: Optional batch configuration
    
    Returns list of predictions with performance metrics.
    """
    try:
        batch_size = len(batch_input.features)
        logger.info("Batch prediction request",
                   symbol=batch_input.symbol,
                   batch_size=batch_size,
                   batch_id=batch_input.batch_id)
        
        start_time = time.perf_counter()
        
        # Validate batch size
        if batch_size > 1000:
            raise HTTPException(
                status_code=400,
                detail="Batch size exceeds maximum limit of 1000"
            )
        
        # Make batch prediction
        result = await service.predict_batch(batch_input, config)
        
        # Log performance
        request_time = (time.perf_counter() - start_time) * 1000
        logger.info("Batch prediction completed",
                   symbol=batch_input.symbol,
                   batch_size=batch_size,
                   batch_id=batch_input.batch_id,
                   request_time_ms=request_time,
                   total_inference_time_ms=result.total_inference_time_ms,
                   average_inference_time_ms=result.average_inference_time_ms)
        
        # Add background task for cleanup if needed
        background_tasks.add_task(
            _log_batch_completion,
            batch_input.batch_id,
            batch_size,
            request_time
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch prediction error",
                    exception=e,
                    symbol=batch_input.symbol,
                    batch_size=len(batch_input.features))
        
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post(
    "/async-batch",
    summary="Async Batch Prediction",
    description="Submit large batch for asynchronous processing"
)
async def predict_async_batch(
    batch_input: BatchFeatureInput,
    config: Optional[BatchConfig] = None,
    service: PredictionService = Depends(get_prediction_service)
) -> dict:
    """
    Submit large batch for asynchronous processing.
    
    For very large batches that would exceed request timeout limits.
    Returns a task ID for polling results.
    """
    try:
        batch_size = len(batch_input.features)
        
        if batch_size <= 100:
            raise HTTPException(
                status_code=400,
                detail="Use /batch endpoint for batches with <= 100 items"
            )
        
        # Generate task ID
        task_id = f"batch_{int(time.time() * 1000)}_{batch_input.symbol}"
        
        logger.info("Async batch prediction submitted",
                   task_id=task_id,
                   batch_size=batch_size,
                   symbol=batch_input.symbol)
        
        # Start async processing
        asyncio.create_task(
            _process_async_batch(task_id, batch_input, config, service)
        )
        
        return {
            "task_id": task_id,
            "status": "submitted",
            "batch_size": batch_size,
            "estimated_completion_time": batch_size * 0.01,  # seconds
            "poll_url": f"/predict/status/{task_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Async batch submission error", exception=e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit async batch: {str(e)}"
        )


@router.get(
    "/status/{task_id}",
    summary="Check Async Batch Status",
    description="Check status of asynchronous batch prediction"
)
async def get_batch_status(task_id: str) -> dict:
    """
    Check status of asynchronous batch prediction.
    
    Returns current status and results if completed.
    """
    # This would typically check a task queue or database
    # For now, return a simple response
    
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Async batch processing not fully implemented",
        "completed_predictions": 0,
        "total_predictions": 0,
        "estimated_remaining_time": 0
    }


@router.get(
    "/config",
    summary="Get Prediction Configuration",
    description="Get current prediction service configuration"
)
async def get_prediction_config(
    service: PredictionService = Depends(get_prediction_service)
) -> dict:
    """Get current prediction service configuration."""
    
    return {
        "model_version": service.model_version,
        "model_path": service.model_path,
        "is_ready": service.is_ready,
        "auto_reload_enabled": service.auto_reload_enabled,
        "performance_stats": service.get_performance_stats()
    }


@router.post(
    "/config",
    summary="Update Prediction Configuration",
    description="Update prediction service configuration"
)
async def update_prediction_config(
    config: dict,
    service: PredictionService = Depends(get_prediction_service)
) -> dict:
    """Update prediction service configuration."""
    
    try:
        # This would update the service configuration
        # For now, return the current config
        
        logger.info("Prediction config update requested", config=config)
        
        return {
            "message": "Configuration update requested",
            "current_config": {
                "model_version": service.model_version,
                "is_ready": service.is_ready
            }
        }
        
    except Exception as e:
        logger.error("Config update error", exception=e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update configuration: {str(e)}"
        )


async def _process_async_batch(
    task_id: str,
    batch_input: BatchFeatureInput,
    config: Optional[BatchConfig],
    service: PredictionService
) -> None:
    """Process asynchronous batch prediction."""
    
    try:
        logger.info("Starting async batch processing", task_id=task_id)
        
        # Process batch
        result = await service.predict_batch(batch_input, config)
        
        # Store results (would typically save to database or cache)
        logger.info("Async batch completed",
                   task_id=task_id,
                   predictions=len(result.predictions))
        
    except Exception as e:
        logger.error("Async batch processing error",
                    task_id=task_id,
                    exception=e)


async def _log_batch_completion(
    batch_id: Optional[str],
    batch_size: int,
    request_time_ms: float
) -> None:
    """Background task to log batch completion."""
    
    logger.info("Batch processing summary",
               batch_id=batch_id,
               batch_size=batch_size,
               total_time_ms=request_time_ms,
               throughput_per_second=batch_size / (request_time_ms / 1000))