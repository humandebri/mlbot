"""
Pydantic schemas for Model Server API requests and responses.

Type-safe data validation for all API endpoints with comprehensive
error handling and documentation generation.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import numpy as np
from datetime import datetime


class FeatureInput(BaseModel):
    """Single feature vector for prediction."""
    
    features: Dict[str, float] = Field(
        ..., 
        description="Feature dictionary with feature names as keys and values as floats",
        example={
            "price_volatility": 0.025,
            "liquidation_pressure": 1.5,
            "volume_surge": 0.8,
            "spread_tightness": 0.12
        }
    )
    
    symbol: str = Field(
        default="BTCUSDT",
        description="Trading symbol for prediction",
        example="BTCUSDT"
    )
    
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp for the prediction request"
    )
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Feature '{key}' must be numeric")
            if np.isnan(value) or np.isinf(value):
                raise ValueError(f"Feature '{key}' contains invalid value: {value}")
        
        return v


class BatchFeatureInput(BaseModel):
    """Batch of feature vectors for batch prediction."""
    
    features: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries",
        min_items=1,
        max_items=1000
    )
    
    symbol: str = Field(
        default="BTCUSDT",
        description="Trading symbol for predictions"
    )
    
    batch_id: Optional[str] = Field(
        default=None,
        description="Optional batch identifier for tracking"
    )


class PredictionResponse(BaseModel):
    """Single prediction response."""
    
    prediction: float = Field(
        ...,
        description="Model prediction (expected PnL)",
        example=0.0025
    )
    
    confidence: float = Field(
        ...,
        description="Prediction confidence score [0, 1]",
        ge=0.0,
        le=1.0,
        example=0.85
    )
    
    symbol: str = Field(
        ...,
        description="Trading symbol",
        example="BTCUSDT"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Prediction timestamp"
    )
    
    model_version: str = Field(
        ...,
        description="Model version used for prediction",
        example="v1.0"
    )
    
    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds",
        example=0.85
    )
    
    thompson_parameters: Optional[Dict[str, float]] = Field(
        default=None,
        description="Thompson sampling parameters used"
    )
    
    risk_adjusted: bool = Field(
        default=True,
        description="Whether risk adjustments were applied"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of individual predictions"
    )
    
    batch_id: Optional[str] = Field(
        default=None,
        description="Batch identifier"
    )
    
    total_predictions: int = Field(
        ...,
        description="Total number of predictions in batch"
    )
    
    total_inference_time_ms: float = Field(
        ...,
        description="Total batch inference time in milliseconds"
    )
    
    average_inference_time_ms: float = Field(
        ...,
        description="Average per-prediction inference time"
    )


class ModelInfo(BaseModel):
    """Model information."""
    
    model_version: str = Field(..., description="Current model version")
    model_type: str = Field(..., description="Model type (e.g., CatBoost)")
    feature_count: int = Field(..., description="Number of features")
    model_size_mb: float = Field(..., description="Model size in MB")
    created_at: datetime = Field(..., description="Model creation timestamp")
    training_metrics: Dict[str, float] = Field(..., description="Training performance metrics")
    

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status", example="healthy")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_info: Optional[ModelInfo] = Field(default=None, description="Model information")
    
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    total_predictions: int = Field(..., description="Total predictions served")
    average_inference_time_ms: float = Field(..., description="Average inference time")
    cache_size: int = Field(..., description="Current cache size")
    
    system_metrics: Dict[str, Any] = Field(
        ...,
        description="System resource metrics",
        example={
            "cpu_usage_percent": 25.5,
            "memory_usage_mb": 512.0,
            "memory_usage_percent": 15.2
        }
    )


class MetricsResponse(BaseModel):
    """Performance metrics response."""
    
    total_predictions: int = Field(..., description="Total predictions served")
    predictions_per_second: float = Field(..., description="Current prediction rate")
    
    inference_times: Dict[str, float] = Field(
        ...,
        description="Inference time statistics",
        example={
            "mean_ms": 0.85,
            "p50_ms": 0.75,
            "p95_ms": 1.2,
            "p99_ms": 2.1
        }
    )
    
    error_rate: float = Field(..., description="Error rate (0-1)")
    cache_hit_rate: float = Field(..., description="Cache hit rate (0-1)")
    
    model_performance: Dict[str, Any] = Field(
        ...,
        description="Model performance metrics"
    )
    
    system_resources: Dict[str, float] = Field(
        ...,
        description="System resource usage"
    )


class ModelLoadRequest(BaseModel):
    """Request to load a new model."""
    
    model_version: str = Field(
        ...,
        description="Model version to load",
        example="v1.1"
    )
    
    model_path: Optional[str] = Field(
        default=None,
        description="Optional custom model path"
    )
    
    force_reload: bool = Field(
        default=False,
        description="Force reload even if same version"
    )


class ModelLoadResponse(BaseModel):
    """Response after loading a model."""
    
    success: bool = Field(..., description="Whether model load was successful")
    message: str = Field(..., description="Status message")
    model_info: Optional[ModelInfo] = Field(default=None, description="Loaded model information")
    load_time_seconds: float = Field(..., description="Time taken to load model")


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier for tracking")


class PredictionConfig(BaseModel):
    """Configuration for prediction requests."""
    
    use_cache: bool = Field(default=True, description="Whether to use prediction cache")
    return_confidence: bool = Field(default=True, description="Whether to return confidence scores")
    apply_risk_adjustment: bool = Field(default=True, description="Whether to apply risk adjustments")
    enable_thompson_sampling: bool = Field(default=True, description="Whether to use Thompson sampling")
    
    confidence_threshold: Optional[float] = Field(
        default=None,
        description="Minimum confidence threshold for predictions",
        ge=0.0,
        le=1.0
    )
    
    max_inference_time_ms: Optional[float] = Field(
        default=None,
        description="Maximum allowed inference time",
        gt=0.0
    )


class BatchConfig(BaseModel):
    """Configuration for batch processing."""
    
    batch_size: int = Field(
        default=100,
        description="Maximum batch size",
        ge=1,
        le=1000
    )
    
    timeout_seconds: float = Field(
        default=30.0,
        description="Batch processing timeout",
        gt=0.0
    )
    
    parallel_processing: bool = Field(
        default=True,
        description="Whether to use parallel processing"
    )