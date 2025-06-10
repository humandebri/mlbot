"""
High-performance prediction service for liquidation-driven trading models.

Integrates with ML pipeline components to provide:
- Real-time inference with sub-millisecond latency
- Batch processing with optimal throughput
- Model hot-reloading and versioning
- Performance monitoring and caching
"""

import numpy as np
import pandas as pd
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from ...ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from ...ml_pipeline.data_preprocessing import DataPreprocessor
from ...common.config import settings
from ...common.logging import get_logger
from ...common.monitoring import (
    MODEL_PREDICTIONS, MODEL_INFERENCE_TIME, MODEL_ERRORS,
    increment_counter, observe_histogram
)
from ..schemas import (
    FeatureInput, BatchFeatureInput, PredictionResponse, BatchPredictionResponse,
    ModelInfo, PredictionConfig, BatchConfig
)

logger = get_logger(__name__)


class PredictionService:
    """
    High-performance prediction service for real-time trading inference.
    
    Features:
    - Sub-millisecond inference latency
    - Intelligent caching and batching
    - Model hot-reloading
    - Comprehensive monitoring
    - Error handling and recovery
    """
    
    def __init__(self):
        """Initialize prediction service."""
        
        # Core components
        self.inference_engine: Optional[InferenceEngine] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        
        # Service state
        self.is_ready = False
        self.model_version = "unknown"
        self.model_path = None
        self.start_time = datetime.now()
        
        # Performance tracking
        self.total_predictions = 0
        self.total_errors = 0
        self.inference_times = []
        self.last_health_check = datetime.now()
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.RLock()
        
        # Model management
        self.model_info: Optional[ModelInfo] = None
        self.auto_reload_enabled = False
        self.model_watch_interval = 300  # 5 minutes
        
        logger.info("Prediction service initialized")
    
    async def initialize(
        self, 
        model_version: str = "v1.0",
        model_path: Optional[str] = None,
        enable_auto_reload: bool = False
    ) -> bool:
        """
        Initialize prediction service with model loading.
        
        Args:
            model_version: Model version to load
            model_path: Optional custom model path
            enable_auto_reload: Enable automatic model reloading
            
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing prediction service", 
                       model_version=model_version,
                       enable_auto_reload=enable_auto_reload)
            
            # Load model
            success = await self.load_model(model_version, model_path)
            
            if success:
                self.auto_reload_enabled = enable_auto_reload
                
                # Start background tasks
                if enable_auto_reload:
                    asyncio.create_task(self._model_watch_task())
                
                asyncio.create_task(self._cleanup_task())
                
                self.is_ready = True
                logger.info("Prediction service initialized successfully")
                return True
            else:
                logger.error("Failed to initialize prediction service")
                return False
                
        except Exception as e:
            logger.error("Error initializing prediction service", exception=e)
            return False
    
    async def load_model(
        self, 
        model_version: str,
        model_path: Optional[str] = None,
        force_reload: bool = False
    ) -> bool:
        """
        Load or reload ML model.
        
        Args:
            model_version: Model version to load
            model_path: Optional custom model path
            force_reload: Force reload even if same version
            
        Returns:
            True if model loaded successfully
        """
        if not force_reload and self.model_version == model_version and self.is_ready:
            logger.info("Model already loaded", version=model_version)
            return True
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                logger.info("Loading model", version=model_version, path=model_path)
                
                # Determine model path
                if model_path is None:
                    model_path = f"models/{model_version}/model.onnx"
                
                model_path_obj = Path(model_path)
                if not model_path_obj.exists():
                    logger.error("Model file not found", path=model_path)
                    return False
                
                # Initialize inference engine
                inference_config = InferenceConfig(
                    model_path=str(model_path),
                    max_inference_time_ms=1.0,
                    enable_thompson_sampling=True,
                    enable_performance_tracking=True
                )
                
                new_engine = InferenceEngine(inference_config)
                new_engine.load_model()
                
                # Initialize preprocessor
                preprocessor_path = model_path_obj.parent / "preprocessor.pkl"
                new_preprocessor = None
                if preprocessor_path.exists():
                    new_preprocessor = DataPreprocessor()
                    # Load preprocessor state if available
                
                # Test model with dummy data
                await self._test_model(new_engine)
                
                # Replace old components
                if self.inference_engine:
                    self.inference_engine.stop_batch_processing()
                
                self.inference_engine = new_engine
                self.preprocessor = new_preprocessor
                self.model_version = model_version
                self.model_path = str(model_path)
                
                # Start batch processing
                self.inference_engine.start_batch_processing()
                
                # Update model info
                await self._update_model_info()
                
                load_time = time.perf_counter() - start_time
                logger.info("Model loaded successfully", 
                           version=model_version,
                           load_time=load_time)
                
                return True
                
        except Exception as e:
            logger.error("Error loading model", exception=e, version=model_version)
            return False
    
    async def predict(
        self, 
        feature_input: FeatureInput,
        config: Optional[PredictionConfig] = None
    ) -> PredictionResponse:
        """
        Make single prediction.
        
        Args:
            feature_input: Input features
            config: Prediction configuration
            
        Returns:
            Prediction response
        """
        if not self.is_ready or not self.inference_engine:
            raise RuntimeError("Prediction service not ready")
        
        config = config or PredictionConfig()
        request_start = time.perf_counter()
        
        try:
            # Convert to DataFrame
            features_df = pd.DataFrame([feature_input.features])
            
            # Preprocess if available
            if self.preprocessor:
                features_df = await self._preprocess_features(features_df)
            
            # Make prediction
            result = self.inference_engine.predict(
                features_df,
                return_confidence=config.return_confidence,
                use_cache=config.use_cache
            )
            
            # Extract results
            prediction = float(result["predictions"][0])
            confidence = float(result["confidence_scores"][0]) if result["confidence_scores"] is not None else 0.5
            
            # Apply confidence threshold if specified
            if config.confidence_threshold and confidence < config.confidence_threshold:
                prediction *= confidence  # Reduce prediction based on low confidence
            
            # Create response
            response = PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                symbol=feature_input.symbol,
                timestamp=feature_input.timestamp or datetime.now(),
                model_version=self.model_version,
                inference_time_ms=result["inference_time_ms"],
                thompson_parameters=result.get("thompson_parameters"),
                risk_adjusted=config.apply_risk_adjustment
            )
            
            # Update metrics
            self._update_prediction_metrics(time.perf_counter() - request_start)
            
            return response
            
        except Exception as e:
            self.total_errors += 1
            increment_counter(MODEL_ERRORS, symbol=feature_input.symbol)
            logger.error("Prediction error", exception=e, symbol=feature_input.symbol)
            raise
    
    async def predict_batch(
        self,
        batch_input: BatchFeatureInput,
        config: Optional[BatchConfig] = None
    ) -> BatchPredictionResponse:
        """
        Make batch predictions.
        
        Args:
            batch_input: Batch input features
            config: Batch configuration
            
        Returns:
            Batch prediction response
        """
        if not self.is_ready or not self.inference_engine:
            raise RuntimeError("Prediction service not ready")
        
        config = config or BatchConfig()
        batch_start = time.perf_counter()
        
        try:
            # Convert to DataFrame
            features_df = pd.DataFrame(batch_input.features)
            
            # Preprocess if available
            if self.preprocessor:
                features_df = await self._preprocess_features(features_df)
            
            # Split into batches if needed
            batch_size = config.batch_size
            if len(features_df) > batch_size:
                predictions = await self._process_large_batch(features_df, batch_input, batch_size)
            else:
                predictions = await self._process_single_batch(features_df, batch_input)
            
            total_time = (time.perf_counter() - batch_start) * 1000  # ms
            avg_time = total_time / len(predictions)
            
            response = BatchPredictionResponse(
                predictions=predictions,
                batch_id=batch_input.batch_id,
                total_predictions=len(predictions),
                total_inference_time_ms=total_time,
                average_inference_time_ms=avg_time
            )
            
            # Update metrics
            for _ in predictions:
                self._update_prediction_metrics(avg_time / 1000)  # Convert to seconds
            
            return response
            
        except Exception as e:
            self.total_errors += 1
            logger.error("Batch prediction error", exception=e, batch_size=len(batch_input.features))
            raise
    
    async def _process_single_batch(
        self,
        features_df: pd.DataFrame,
        batch_input: BatchFeatureInput
    ) -> List[PredictionResponse]:
        """Process single batch of predictions."""
        
        # Make batch prediction
        result = self.inference_engine.predict(
            features_df,
            return_confidence=True,
            use_cache=False  # Disable cache for batch processing
        )
        
        predictions = []
        now = datetime.now()
        
        for i in range(len(features_df)):
            prediction = float(result["predictions"][i])
            confidence = float(result["confidence_scores"][i]) if result["confidence_scores"] is not None else 0.5
            
            response = PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                symbol=batch_input.symbol,
                timestamp=now,
                model_version=self.model_version,
                inference_time_ms=result["inference_time_ms"] / len(features_df),
                thompson_parameters=result.get("thompson_parameters"),
                risk_adjusted=True
            )
            predictions.append(response)
        
        return predictions
    
    async def _process_large_batch(
        self,
        features_df: pd.DataFrame,
        batch_input: BatchFeatureInput,
        batch_size: int
    ) -> List[PredictionResponse]:
        """Process large batch by splitting into smaller chunks."""
        
        all_predictions = []
        
        for i in range(0, len(features_df), batch_size):
            chunk = features_df.iloc[i:i + batch_size]
            
            # Create chunk input
            chunk_input = BatchFeatureInput(
                features=chunk.to_dict('records'),
                symbol=batch_input.symbol,
                batch_id=f"{batch_input.batch_id}_chunk_{i // batch_size}" if batch_input.batch_id else None
            )
            
            chunk_predictions = await self._process_single_batch(chunk, chunk_input)
            all_predictions.extend(chunk_predictions)
        
        return all_predictions
    
    async def _preprocess_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features using the fitted preprocessor."""
        
        if not self.preprocessor:
            return features_df
        
        try:
            # Run preprocessing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            processed_df = await loop.run_in_executor(
                self.executor,
                self.preprocessor.transform,
                features_df
            )
            
            return processed_df
            
        except Exception as e:
            logger.warning("Preprocessing error, using raw features", exception=e)
            return features_df
    
    async def _test_model(self, engine: InferenceEngine) -> None:
        """Test model with dummy data."""
        
        # Create dummy features
        dummy_features = pd.DataFrame({
            f"feature_{i}": [np.random.randn()] for i in range(50)
        })
        
        # Test prediction
        result = engine.predict(dummy_features, return_confidence=True)
        
        if not result or "predictions" not in result:
            raise RuntimeError("Model test failed - no predictions returned")
        
        logger.info("Model test passed", 
                   inference_time=result.get("inference_time_ms", 0))
    
    async def _update_model_info(self) -> None:
        """Update model information."""
        
        if not self.inference_engine:
            return
        
        try:
            model_path = Path(self.model_path)
            model_size_mb = model_path.stat().st_size / 1024 / 1024
            
            self.model_info = ModelInfo(
                model_version=self.model_version,
                model_type="ONNX",
                feature_count=50,  # This should be extracted from model metadata
                model_size_mb=model_size_mb,
                created_at=datetime.fromtimestamp(model_path.stat().st_mtime),
                training_metrics={}  # This should be loaded from metadata
            )
            
        except Exception as e:
            logger.warning("Error updating model info", exception=e)
    
    def _update_prediction_metrics(self, inference_time: float) -> None:
        """Update prediction performance metrics."""
        
        self.total_predictions += 1
        self.inference_times.append(inference_time * 1000)  # Convert to ms
        
        # Keep only recent inference times (last 1000)
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]
        
        # Update Prometheus metrics
        observe_histogram(MODEL_INFERENCE_TIME, inference_time)
        increment_counter(MODEL_PREDICTIONS, symbol="all")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        if not self.inference_times:
            return {
                "total_predictions": self.total_predictions,
                "total_errors": self.total_errors,
                "error_rate": 0.0,
                "average_inference_time_ms": 0.0,
                "p95_inference_time_ms": 0.0,
                "p99_inference_time_ms": 0.0
            }
        
        inference_array = np.array(self.inference_times)
        
        return {
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(self.total_predictions, 1),
            "average_inference_time_ms": float(np.mean(inference_array)),
            "p95_inference_time_ms": float(np.percentile(inference_array, 95)),
            "p99_inference_time_ms": float(np.percentile(inference_array, 99)),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "cache_size": len(self.inference_engine.prediction_cache) if self.inference_engine else 0
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        
        return {
            "status": "healthy" if self.is_ready else "unhealthy",
            "model_loaded": self.inference_engine is not None,
            "model_version": self.model_version,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            **self.get_performance_stats()
        }
    
    async def _model_watch_task(self) -> None:
        """Background task to watch for model updates."""
        
        while self.auto_reload_enabled:
            try:
                await asyncio.sleep(self.model_watch_interval)
                
                if self.model_path:
                    model_path = Path(self.model_path)
                    if model_path.exists():
                        # Check if model has been updated
                        mtime = model_path.stat().st_mtime
                        current_mtime = self.model_info.created_at.timestamp() if self.model_info else 0
                        
                        if mtime > current_mtime:
                            logger.info("Model file updated, reloading", path=self.model_path)
                            await self.load_model(self.model_version, self.model_path, force_reload=True)
                
            except Exception as e:
                logger.error("Error in model watch task", exception=e)
    
    async def _cleanup_task(self) -> None:
        """Background cleanup task."""
        
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Clean up old inference times
                if len(self.inference_times) > 10000:
                    self.inference_times = self.inference_times[-5000:]
                
                # Update last health check
                self.last_health_check = datetime.now()
                
            except Exception as e:
                logger.error("Error in cleanup task", exception=e)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the prediction service."""
        
        logger.info("Shutting down prediction service")
        
        self.is_ready = False
        self.auto_reload_enabled = False
        
        if self.inference_engine:
            self.inference_engine.stop_batch_processing()
        
        self.executor.shutdown(wait=True)
        
        logger.info("Prediction service shutdown complete")