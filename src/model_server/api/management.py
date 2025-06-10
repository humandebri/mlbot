"""
Model management API endpoints.

Provides model lifecycle management including:
- Model loading and reloading
- Version switching
- Configuration updates
- Cache management
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import asyncio
from datetime import datetime

from ..schemas import ModelLoadRequest, ModelLoadResponse, ErrorResponse
from ..services.prediction_service import PredictionService
from ...common.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/model", tags=["Model Management"])


async def get_prediction_service() -> Optional[PredictionService]:
    """Get prediction service instance."""
    from .prediction import prediction_service
    return prediction_service


@router.post(
    "/load",
    response_model=ModelLoadResponse,
    summary="Load Model",
    description="Load or reload a model version"
)
async def load_model(
    request: ModelLoadRequest,
    service: Optional[PredictionService] = Depends(get_prediction_service)
) -> ModelLoadResponse:
    """
    Load or reload a model version.
    
    - **model_version**: Model version to load (e.g., "v1.1")
    - **model_path**: Optional custom model path
    - **force_reload**: Force reload even if same version
    
    Returns model load status and information.
    """
    
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available"
        )
    
    try:
        logger.info("Model load request",
                   version=request.model_version,
                   path=request.model_path,
                   force_reload=request.force_reload)
        
        start_time = datetime.now()
        
        # Load model
        success = await service.load_model(
            model_version=request.model_version,
            model_path=request.model_path,
            force_reload=request.force_reload
        )
        
        load_time = (datetime.now() - start_time).total_seconds()
        
        if success:
            message = f"Model {request.model_version} loaded successfully"
            model_info = service.model_info
        else:
            message = f"Failed to load model {request.model_version}"
            model_info = None
        
        logger.info("Model load completed",
                   version=request.model_version,
                   success=success,
                   load_time=load_time)
        
        return ModelLoadResponse(
            success=success,
            message=message,
            model_info=model_info,
            load_time_seconds=load_time
        )
        
    except Exception as e:
        logger.error("Model load error",
                    exception=e,
                    version=request.model_version)
        
        raise HTTPException(
            status_code=500,
            detail=f"Model load failed: {str(e)}"
        )


@router.post(
    "/reload",
    summary="Reload Current Model",
    description="Reload the currently loaded model"
)
async def reload_current_model(
    service: Optional[PredictionService] = Depends(get_prediction_service)
) -> dict:
    """
    Reload the currently loaded model.
    
    Useful for picking up model updates without changing versions.
    """
    
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available"
        )
    
    if not service.model_version or service.model_version == "unknown":
        raise HTTPException(
            status_code=400,
            detail="No model currently loaded"
        )
    
    try:
        logger.info("Model reload request", version=service.model_version)
        
        start_time = datetime.now()
        
        success = await service.load_model(
            model_version=service.model_version,
            model_path=service.model_path,
            force_reload=True
        )
        
        load_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": success,
            "message": f"Model {service.model_version} {'reloaded' if success else 'failed to reload'}",
            "model_version": service.model_version,
            "reload_time_seconds": load_time,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error("Model reload error", exception=e)
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {str(e)}"
        )


@router.get(
    "/versions",
    summary="List Available Models",
    description="List all available model versions"
)
async def list_model_versions() -> dict:
    """
    List all available model versions.
    
    Scans the models directory for available versions.
    """
    
    try:
        from pathlib import Path
        
        models_dir = Path("models")
        available_versions = []
        
        if models_dir.exists():
            for version_dir in models_dir.iterdir():
                if version_dir.is_dir():
                    model_file = version_dir / "model.onnx"
                    metadata_file = version_dir / "metadata.json"
                    
                    if model_file.exists():
                        version_info = {
                            "version": version_dir.name,
                            "model_path": str(model_file),
                            "has_metadata": metadata_file.exists(),
                            "model_size_mb": model_file.stat().st_size / 1024 / 1024,
                            "created_at": datetime.fromtimestamp(model_file.stat().st_mtime)
                        }
                        
                        # Load metadata if available
                        if metadata_file.exists():
                            try:
                                import json
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                    version_info.update({
                                        "training_time": metadata.get("training_time"),
                                        "feature_count": len(metadata.get("feature_importance", {}))
                                    })
                            except Exception as e:
                                logger.warning("Error loading metadata", 
                                              version=version_dir.name, 
                                              exception=e)
                        
                        available_versions.append(version_info)
        
        # Sort by creation time (newest first)
        available_versions.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "available_versions": available_versions,
            "total_versions": len(available_versions),
            "models_directory": str(models_dir.absolute())
        }
        
    except Exception as e:
        logger.error("Error listing model versions", exception=e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list model versions: {str(e)}"
        )


@router.delete(
    "/cache",
    summary="Clear Prediction Cache",
    description="Clear the prediction cache"
)
async def clear_cache(
    service: Optional[PredictionService] = Depends(get_prediction_service)
) -> dict:
    """
    Clear the prediction cache.
    
    Useful for debugging or forcing fresh predictions.
    """
    
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available"
        )
    
    try:
        cache_size_before = 0
        
        if service.inference_engine:
            cache_size_before = len(service.inference_engine.prediction_cache)
            service.inference_engine.reset_cache()
        
        logger.info("Cache cleared", 
                   cache_size_before=cache_size_before)
        
        return {
            "success": True,
            "message": "Prediction cache cleared",
            "cache_size_before": cache_size_before,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error("Cache clear error", exception=e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.post(
    "/auto-reload",
    summary="Toggle Auto-Reload",
    description="Enable or disable automatic model reloading"
)
async def toggle_auto_reload(
    enabled: bool,
    service: Optional[PredictionService] = Depends(get_prediction_service)
) -> dict:
    """
    Enable or disable automatic model reloading.
    
    When enabled, the service will automatically reload the model
    when it detects file changes.
    """
    
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available"
        )
    
    try:
        previous_state = service.auto_reload_enabled
        service.auto_reload_enabled = enabled
        
        logger.info("Auto-reload toggled",
                   enabled=enabled,
                   previous_state=previous_state)
        
        return {
            "success": True,
            "message": f"Auto-reload {'enabled' if enabled else 'disabled'}",
            "auto_reload_enabled": enabled,
            "previous_state": previous_state,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error("Auto-reload toggle error", exception=e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to toggle auto-reload: {str(e)}"
        )


@router.get(
    "/config",
    summary="Get Model Configuration",
    description="Get current model and service configuration"
)
async def get_model_config(
    service: Optional[PredictionService] = Depends(get_prediction_service)
) -> dict:
    """
    Get current model and service configuration.
    
    Returns detailed configuration information.
    """
    
    if service is None:
        return {
            "service_available": False,
            "message": "Prediction service not available"
        }
    
    try:
        config = {
            "service_available": True,
            "model_version": service.model_version,
            "model_path": service.model_path,
            "is_ready": service.is_ready,
            "auto_reload_enabled": service.auto_reload_enabled,
            "model_watch_interval": service.model_watch_interval,
            "start_time": service.start_time,
            "uptime_seconds": (datetime.now() - service.start_time).total_seconds()
        }
        
        if service.inference_engine:
            engine_config = service.inference_engine.config
            config.update({
                "inference_config": {
                    "max_inference_time_ms": engine_config.max_inference_time_ms,
                    "batch_size": engine_config.batch_size,
                    "enable_batching": engine_config.enable_batching,
                    "cache_size": engine_config.cache_size,
                    "enable_thompson_sampling": engine_config.enable_thompson_sampling,
                    "confidence_threshold": engine_config.confidence_threshold
                }
            })
        
        return config
        
    except Exception as e:
        logger.error("Error getting model config", exception=e)
        return {
            "service_available": False,
            "error": str(e),
            "timestamp": datetime.now()
        }


@router.post(
    "/warmup",
    summary="Warm Up Model",
    description="Warm up the model with dummy predictions"
)
async def warmup_model(
    num_predictions: int = 10,
    service: Optional[PredictionService] = Depends(get_prediction_service)
) -> dict:
    """
    Warm up the model with dummy predictions.
    
    Useful after model loading to ensure optimal performance.
    """
    
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available"
        )
    
    if not service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not ready for warmup"
        )
    
    try:
        import numpy as np
        import pandas as pd
        
        logger.info("Model warmup started", num_predictions=num_predictions)
        
        start_time = datetime.now()
        warmup_times = []
        
        for i in range(num_predictions):
            # Create dummy features
            dummy_features = {
                f"feature_{j}": float(np.random.randn()) 
                for j in range(50)
            }
            
            feature_input = FeatureInput(
                features=dummy_features,
                symbol="WARMUP",
                timestamp=datetime.now()
            )
            
            # Make prediction
            pred_start = datetime.now()
            await service.predict(feature_input)
            pred_time = (datetime.now() - pred_start).total_seconds() * 1000
            warmup_times.append(pred_time)
        
        total_time = (datetime.now() - start_time).total_seconds()
        avg_time = sum(warmup_times) / len(warmup_times)
        
        logger.info("Model warmup completed",
                   num_predictions=num_predictions,
                   total_time=total_time,
                   average_time_ms=avg_time)
        
        return {
            "success": True,
            "message": f"Model warmed up with {num_predictions} predictions",
            "num_predictions": num_predictions,
            "total_time_seconds": total_time,
            "average_inference_time_ms": avg_time,
            "min_inference_time_ms": min(warmup_times),
            "max_inference_time_ms": max(warmup_times),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error("Model warmup error", exception=e)
        raise HTTPException(
            status_code=500,
            detail=f"Model warmup failed: {str(e)}"
        )