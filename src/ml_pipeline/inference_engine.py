"""
High-performance ONNX inference engine for real-time liquidation trading.

Optimized for:
- Sub-millisecond inference latency
- High-throughput batch processing
- Memory-efficient operations
- Thompson Sampling parameter optimization
- Production-grade reliability
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
from collections import deque
import threading
import warnings
warnings.filterwarnings('ignore')

import onnxruntime as ort
from scipy import stats

from ..common.config import settings
from ..common.logging import get_logger
from ..common.monitoring import (
    MODEL_PREDICTIONS, MODEL_INFERENCE_TIME,
    increment_counter, observe_histogram
)
try:
    from .feature_adapter_26 import FeatureAdapter26
except ImportError:
    # Fallback if module not found
    class FeatureAdapter26:
        def adapt(self, features):
            if isinstance(features, dict):
                return np.array(list(features.values())[:26])
            return features[:26]

logger = get_logger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    
    # Model paths - use working regressor (26 dimensions)
    model_path: str = "models/v1.0/model.onnx"  # Alternative working regressor
    preprocessor_path: str = "models/v1.0/scaler.pkl"
    
    # Performance settings
    max_inference_time_ms: float = 1.0
    batch_size: int = 100
    enable_batching: bool = True
    cache_size: int = 1000
    
    # ONNX Runtime settings
    providers: List[str] = None
    session_options: Dict[str, Any] = None
    
    # Thompson Sampling
    enable_thompson_sampling: bool = True
    thompson_alpha: float = 1.0
    thompson_beta: float = 1.0
    exploration_rate: float = 0.1
    update_frequency: int = 100
    
    # Risk management
    confidence_threshold: float = 0.6
    max_position_size: float = 0.1
    risk_adjustment: bool = True
    
    # Monitoring
    enable_performance_tracking: bool = True
    log_predictions: bool = False
    alert_on_slow_inference: bool = True
    
    def __post_init__(self):
        if self.providers is None:
            self.providers = ['CPUExecutionProvider']
        
        if self.session_options is None:
            self.session_options = {
                'enable_cpu_mem_arena': False,
                'enable_mem_pattern': False,
                'enable_mem_reuse': False,
                'execution_mode': ort.ExecutionMode.ORT_SEQUENTIAL,
                'graph_optimization_level': ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            }


class ThompsonSampler:
    """Thompson Sampling for dynamic parameter optimization."""
    
    def __init__(self, parameters: Dict[str, Tuple[float, float]]):
        """
        Initialize Thompson Sampler.
        
        Args:
            parameters: Dict of parameter_name -> (min_value, max_value)
        """
        self.parameters = parameters
        self.alpha = {param: 1.0 for param in parameters}
        self.beta = {param: 1.0 for param in parameters}
        self.rewards = {param: deque(maxlen=1000) for param in parameters}
        self.current_values = {}
        
        # Initialize with random values
        self.sample_parameters()
    
    def sample_parameters(self) -> Dict[str, float]:
        """Sample parameters using Thompson Sampling."""
        for param, (min_val, max_val) in self.parameters.items():
            # Sample from Beta distribution
            beta_sample = np.random.beta(self.alpha[param], self.beta[param])
            
            # Scale to parameter range
            self.current_values[param] = min_val + beta_sample * (max_val - min_val)
        
        return self.current_values.copy()
    
    def update_reward(self, parameter: str, reward: float) -> None:
        """Update parameter reward and Beta distribution."""
        self.rewards[parameter].append(reward)
        
        # Update Beta parameters based on reward
        if reward > 0:
            self.alpha[parameter] += reward
        else:
            self.beta[parameter] += abs(reward)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get Thompson Sampling statistics."""
        stats = {}
        for param in self.parameters:
            stats[param] = {
                "current_value": self.current_values.get(param, 0),
                "alpha": self.alpha[param],
                "beta": self.beta[param],
                "mean_reward": np.mean(self.rewards[param]) if self.rewards[param] else 0,
                "total_rewards": len(self.rewards[param])
            }
        return stats


class InferenceEngine:
    """
    High-performance ONNX inference engine for liquidation trading models.
    
    Features:
    - Sub-millisecond inference with ONNX Runtime
    - Intelligent batching and caching
    - Thompson Sampling parameter optimization
    - Real-time performance monitoring
    - Risk-adjusted prediction scores
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize inference engine.
        
        Args:
            config: Inference configuration
        """
        self.config = config or InferenceConfig()
        
        # Model components
        self.onnx_session = None
        self.preprocessor = None
        self.input_name = None
        self.output_name = None
        
        # Performance tracking
        self.inference_times = deque(maxlen=1000)
        self.prediction_cache = {}
        self.batch_queue = deque()
        self.total_predictions = 0
        
        # Thompson Sampling
        if self.config.enable_thompson_sampling:
            self.thompson_sampler = ThompsonSampler({
                "confidence_threshold": (0.5, 0.9),
                "risk_adjustment_factor": (0.8, 1.2),
                "batch_size": (50, 200)
            })
        else:
            self.thompson_sampler = None
        
        # Threading for batch processing
        self.batch_thread = None
        self.running = False
        self.batch_lock = threading.Lock()
        
        logger.info("Inference engine initialized", config=self.config.__dict__)
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load ONNX model and preprocessor."""
        model_path = model_path or self.config.model_path
        
        try:
            # Configure ONNX Runtime session
            session_options = ort.SessionOptions()
            for key, value in self.config.session_options.items():
                setattr(session_options, key, value)
            
            # Load ONNX model
            self.onnx_session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=self.config.providers
            )
            
            # Get input/output names
            self.input_name = self.onnx_session.get_inputs()[0].name
            self.output_names = [output.name for output in self.onnx_session.get_outputs()]
            self.output_name = self.output_names[0]  # Keep for backward compatibility
            
            # Skip preprocessor for models without scaling (catboost, v1.0)
            if "catboost" in model_path.lower() or "v1.0" in model_path.lower():
                self.preprocessor = None
                logger.info(f"Skipping preprocessor for model: {model_path}")
            else:
                # Load preprocessor if available for other models
                preprocessor_path = Path(model_path).parent / "preprocessor.pkl"
                manual_scaler_path = Path(model_path).parent / "manual_scaler.json"
                
                # Try manual scaler first (preferred)
                if manual_scaler_path.exists():
                    import json
                    with open(manual_scaler_path, 'r') as f:
                        scaler_data = json.load(f)
                        self.preprocessor = {
                            'type': 'manual',
                            'means': np.array(scaler_data['means']),
                            'stds': np.array(scaler_data['stds'])
                        }
                        logger.info("Loaded manual JSON scaler")
                elif preprocessor_path.exists():
                    # Fallback to pickle
                    import pickle
                    with open(preprocessor_path, 'rb') as f:
                        self.preprocessor = pickle.load(f)
            
            # Warm up model with dummy data
            self._warmup_model()
            
            logger.info("Model loaded successfully",
                       model_path=model_path,
                       input_shape=self.onnx_session.get_inputs()[0].shape,
                       providers=self.onnx_session.get_providers())
            
        except Exception as e:
            logger.error("Failed to load model", exception=e, model_path=model_path)
            raise
    
    def start_batch_processing(self) -> None:
        """Start background batch processing thread."""
        if self.config.enable_batching and not self.running:
            self.running = True
            self.batch_thread = threading.Thread(target=self._batch_processing_loop)
            self.batch_thread.daemon = True
            self.batch_thread.start()
            logger.info("Batch processing started")
    
    def stop_batch_processing(self) -> None:
        """Stop background batch processing."""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=5.0)
        logger.info("Batch processing stopped")
    
    def predict(
        self,
        features: Union[pd.DataFrame, np.ndarray, Dict[str, float]],
        return_confidence: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction with confidence scoring.
        
        Args:
            features: Input features
            return_confidence: Whether to return confidence scores
            use_cache: Whether to use prediction cache
            
        Returns:
            Dictionary with predictions and metadata
        """
        if self.onnx_session is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.perf_counter()
        
        try:
            # Convert to numpy if needed
            if isinstance(features, pd.DataFrame):
                feature_array = features.values.astype(np.float32)
                feature_hash = hash(features.values.tobytes()) if use_cache else None
            elif isinstance(features, dict):
                # Convert dict to numpy array using FeatureAdapter26
                adapter = FeatureAdapter26()
                feature_array = adapter.adapt(features).astype(np.float32).reshape(1, -1)
                feature_hash = hash(str(sorted(features.items())).encode()) if use_cache else None
            else:
                feature_array = features.astype(np.float32)
                feature_hash = hash(feature_array.tobytes()) if use_cache else None
            
            # Check cache
            if use_cache and feature_hash in self.prediction_cache:
                cached_result = self.prediction_cache[feature_hash]
                logger.debug("Cache hit for prediction")
                return cached_result
            
            # Skip feature adaptation for v3.1_improved model (44 dimensions)
            if "v3.1_improved" not in str(self.config.model_path) and feature_array.shape[-1] != 26:
                # Use feature adapter to convert to 26 dimensions
                feature_adapter = FeatureAdapter26()
                if len(feature_array.shape) == 1:
                    # Convert single feature vector
                    feature_dict = {f"feature_{i}": feature_array[i] for i in range(len(feature_array))}
                    feature_array = feature_adapter.adapt(feature_dict).reshape(1, -1)
                else:
                    # Batch processing
                    adapted_features = []
                    for row in feature_array:
                        feature_dict = {f"feature_{i}": row[i] for i in range(len(row))}
                        adapted_features.append(feature_adapter.adapt(feature_dict))
                    feature_array = np.array(adapted_features, dtype=np.float32)
            
            # Preprocess features if preprocessor available
            if self.preprocessor:
                if isinstance(self.preprocessor, dict) and self.preprocessor.get('type') == 'manual':
                    # Use manual scaler
                    means = self.preprocessor['means']
                    stds = self.preprocessor['stds']
                    feature_array = (feature_array - means) / stds
                    feature_array = np.clip(feature_array, -5, 5).astype(np.float32)
                elif hasattr(self.preprocessor, 'transform'):
                    # Use sklearn-style preprocessor
                    feature_array = feature_array.astype(np.float32)
                    feature_array = self.preprocessor.transform(feature_array).astype(np.float32)
            
            # Make prediction - get both outputs
            outputs = self.onnx_session.run(
                None,  # Get all outputs
                {self.input_name: feature_array}
            )
            
            # Extract probability values from outputs
            # outputs[0] = class labels, outputs[1] = probabilities dict
            if len(outputs) > 1 and isinstance(outputs[1], list):
                # Extract probability of positive class (1) from each prediction
                predictions = []
                for prob_dict in outputs[1]:
                    if isinstance(prob_dict, dict):
                        # Get probability of class 1 (positive/buy signal)
                        prob_positive = prob_dict.get(1, 0.5)
                        predictions.append(prob_positive)
                predictions = np.array(predictions, dtype=np.float32)
            else:
                # Fallback to class labels if probability not available
                predictions = outputs[0].flatten().astype(np.float32)
            
            # Calculate confidence scores
            confidence_scores = None
            if return_confidence:
                confidence_scores = self._calculate_confidence_scores(
                    predictions, feature_array
                )
            
            # Apply Thompson Sampling adjustments
            adjusted_predictions = predictions
            if self.thompson_sampler:
                adjusted_predictions = self._apply_thompson_adjustments(
                    predictions, confidence_scores
                )
            
            # Apply risk adjustments
            risk_adjusted_predictions = self._apply_risk_adjustments(
                adjusted_predictions, confidence_scores
            )
            
            # Prepare result
            result = {
                "predictions": risk_adjusted_predictions,
                "raw_predictions": predictions,
                "confidence_scores": confidence_scores,
                "thompson_parameters": self.thompson_sampler.current_values if self.thompson_sampler else None,
                "inference_time_ms": (time.perf_counter() - start_time) * 1000,
                "model_info": {
                    "input_shape": feature_array.shape,
                    "prediction_count": len(predictions)
                }
            }
            
            # Cache result
            if use_cache and feature_hash:
                if len(self.prediction_cache) >= self.config.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.prediction_cache))
                    del self.prediction_cache[oldest_key]
                self.prediction_cache[feature_hash] = result
            
            # Update performance tracking
            self._update_performance_metrics(result)
            
            self.total_predictions += len(predictions)
            
            # Log slow inference
            if (result["inference_time_ms"] > self.config.max_inference_time_ms and 
                self.config.alert_on_slow_inference):
                logger.warning("Slow inference detected",
                              inference_time=result["inference_time_ms"],
                              max_allowed=self.config.max_inference_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, features_list: List[Union[pd.DataFrame, np.ndarray]]) -> List[Dict[str, Any]]:
        """Make batch predictions efficiently."""
        if not features_list:
            return []
        
        # Stack features for batch inference
        if isinstance(features_list[0], pd.DataFrame):
            batch_features = np.vstack([f.values for f in features_list]).astype(np.float32)
        else:
            batch_features = np.vstack(features_list).astype(np.float32)
        
        # Single batch prediction
        batch_result = self.predict(batch_features, return_confidence=True, use_cache=False)
        
        # Split results back to individual predictions
        predictions = batch_result["predictions"]
        confidences = batch_result["confidence_scores"]
        
        results = []
        batch_size = len(features_list)
        
        for i in range(batch_size):
            result = {
                "predictions": np.array([predictions[i]]),
                "raw_predictions": np.array([batch_result["raw_predictions"][i]]),
                "confidence_scores": np.array([confidences[i]]) if confidences is not None else None,
                "thompson_parameters": batch_result["thompson_parameters"],
                "inference_time_ms": batch_result["inference_time_ms"] / batch_size,
                "model_info": {
                    "input_shape": (1, batch_features.shape[1]),
                    "prediction_count": 1
                }
            }
            results.append(result)
        
        return results
    
    def _calculate_confidence_scores(self, predictions: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for predictions with proper handling."""
        try:
            # Ensure predictions is numpy array
            predictions = np.atleast_1d(predictions)
            
            # Method 1: Based on prediction magnitude
            # Map predictions to confidence: higher absolute predictions = higher confidence
            # But cap at reasonable values
            abs_predictions = np.abs(predictions)
            max_pred = 0.3  # Expected max prediction magnitude
            magnitude_confidence = np.clip(abs_predictions / max_pred, 0.0, 1.0)
            
            # Method 2: Based on feature quality
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Check for NaN or extreme values in features
            nan_ratio = np.isnan(features).sum(axis=1) / features.shape[1]
            extreme_ratio = ((np.abs(features) > 10).sum(axis=1) / features.shape[1])
            feature_quality = 1.0 - np.clip(nan_ratio + extreme_ratio * 0.5, 0.0, 1.0)
            
            # Method 3: Prediction reasonableness
            # Penalize extreme predictions
            extreme_penalty = np.where(abs_predictions > 0.5, 0.5, 1.0)
            
            # For single predictions, don't use consistency score
            if len(predictions) == 1:
                # Use only magnitude and feature quality
                confidence_scores = (magnitude_confidence * 0.6 + 
                                   feature_quality * 0.3 + 
                                   extreme_penalty * 0.1)
            else:
                # For batch predictions, can use consistency
                median_pred = np.median(predictions)
                std_pred = np.std(predictions) if len(predictions) > 1 else 0.1
                consistency = 1.0 - np.abs(predictions - median_pred) / (std_pred + 0.1)
                consistency = np.clip(consistency, 0.0, 1.0)
                
                confidence_scores = (magnitude_confidence * 0.4 + 
                                   feature_quality * 0.3 + 
                                   consistency * 0.2 +
                                   extreme_penalty * 0.1)
            
            # Apply sigmoid to smooth confidence scores
            # This prevents always getting 0% or 100%
            confidence_scores = 1 / (1 + np.exp(-4 * (confidence_scores - 0.5)))
            
            # Final clipping to reasonable range
            confidence_scores = np.clip(confidence_scores, 0.1, 0.9)
            
            return confidence_scores
            
        except Exception as e:
            logger.warning("Error calculating confidence scores", exception=e)
            return np.ones(len(predictions)) * 0.5  # Default medium confidence    
    def _apply_thompson_adjustments(
        self, 
        predictions: np.ndarray, 
        confidence_scores: Optional[np.ndarray]
    ) -> np.ndarray:
        """Apply Thompson Sampling parameter adjustments."""
        if not self.thompson_sampler:
            return predictions.astype(np.float32)
        
        # Ensure predictions are float32
        predictions = predictions.astype(np.float32)
        
        # Get current Thompson parameters
        params = self.thompson_sampler.current_values
        
        # Apply confidence threshold adjustment
        confidence_threshold = params.get("confidence_threshold", self.config.confidence_threshold)
        if confidence_scores is not None:
            confidence_scores = confidence_scores.astype(np.float32)
            low_confidence_mask = confidence_scores < confidence_threshold
            # Create a copy to avoid in-place modification issues
            adjusted_predictions = predictions.copy()
            adjusted_predictions[low_confidence_mask] *= 0.5  # Reduce low-confidence predictions
            predictions = adjusted_predictions
        
        # Apply risk adjustment factor
        risk_factor = float(params.get("risk_adjustment_factor", 1.0))
        adjusted_predictions = predictions * risk_factor
        
        return adjusted_predictions.astype(np.float32)
    
    def _apply_risk_adjustments(
        self, 
        predictions: np.ndarray, 
        confidence_scores: Optional[np.ndarray]
    ) -> np.ndarray:
        """Apply risk management adjustments to predictions."""
        if not self.config.risk_adjustment:
            return predictions.astype(np.float32)
        
        # Ensure predictions are float32
        predictions = predictions.astype(np.float32)
        adjusted_predictions = predictions.copy()
        
        # Position sizing based on confidence
        if confidence_scores is not None:
            confidence_scores = confidence_scores.astype(np.float32)
            position_sizes = confidence_scores * float(self.config.max_position_size)
            adjusted_predictions = adjusted_predictions * position_sizes
        
        # Cap extreme predictions
        percentile_99 = np.percentile(np.abs(adjusted_predictions), 99)
        adjusted_predictions = np.clip(adjusted_predictions, -percentile_99, percentile_99)
        
        return adjusted_predictions.astype(np.float32)
    
    def _warmup_model(self) -> None:
        """Warm up model with dummy data to improve first inference speed."""
        try:
            input_shape = self.onnx_session.get_inputs()[0].shape
            
            # Create dummy input with correct shape
            if input_shape[0] is None or input_shape[0] == 'batch_size':
                dummy_shape = (1,) + tuple(input_shape[1:])
            else:
                dummy_shape = tuple(input_shape)
            
            dummy_input = np.random.randn(*dummy_shape).astype(np.float32)
            
            # Run a few warming predictions
            for _ in range(5):
                _ = self.onnx_session.run([self.output_name], {self.input_name: dummy_input})
            
            logger.info("Model warmup completed", input_shape=dummy_shape)
            
        except Exception as e:
            logger.warning("Model warmup failed", exception=e)
    
    def _batch_processing_loop(self) -> None:
        """Background batch processing loop."""
        while self.running:
            try:
                # Collect batch from queue
                batch_features = []
                batch_callbacks = []
                
                with self.batch_lock:
                    while (len(batch_features) < self.config.batch_size and 
                           len(self.batch_queue) > 0):
                        features, callback = self.batch_queue.popleft()
                        batch_features.append(features)
                        batch_callbacks.append(callback)
                
                if batch_features:
                    # Process batch
                    try:
                        results = self.predict_batch(batch_features)
                        
                        # Execute callbacks
                        for result, callback in zip(results, batch_callbacks):
                            if callback:
                                callback(result)
                                
                    except Exception as e:
                        logger.error("Batch processing error", exception=e)
                        # Execute error callbacks
                        for callback in batch_callbacks:
                            if callback:
                                callback({"error": str(e)})
                else:
                    time.sleep(0.001)  # Short sleep when no work
                    
            except Exception as e:
                logger.error("Error in batch processing loop", exception=e)
                time.sleep(0.1)
    
    def _update_performance_metrics(self, result: Dict[str, Any]) -> None:
        """Update performance tracking metrics."""
        if not self.config.enable_performance_tracking:
            return
        
        inference_time = result["inference_time_ms"]
        prediction_count = result["model_info"]["prediction_count"]
        
        # Update local tracking
        self.inference_times.append(inference_time)
        
        # Update Prometheus metrics
        observe_histogram(MODEL_INFERENCE_TIME, inference_time / 1000.0)  # Convert to seconds
        increment_counter(MODEL_PREDICTIONS, symbol="all")
        
        # Log performance periodically
        if self.total_predictions % 1000 == 0:
            avg_inference_time = np.mean(self.inference_times)
            p95_inference_time = np.percentile(self.inference_times, 95)
            
            logger.info("Performance update",
                       total_predictions=self.total_predictions,
                       avg_inference_time_ms=avg_inference_time,
                       p95_inference_time_ms=p95_inference_time,
                       cache_size=len(self.prediction_cache))
        
        # Update Thompson Sampling rewards
        if self.thompson_sampler and result.get("confidence_scores") is not None:
            avg_confidence = np.mean(result["confidence_scores"])
            reward = avg_confidence - 0.5  # Reward above 50% confidence
            
            for param in self.thompson_sampler.parameters:
                self.thompson_sampler.update_reward(param, reward)
            
            # Resample parameters periodically
            if self.total_predictions % self.config.update_frequency == 0:
                self.thompson_sampler.sample_parameters()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "total_predictions": self.total_predictions,
            "cache_size": len(self.prediction_cache),
            "cache_hit_rate": 0.0,  # TODO: Implement cache hit tracking
            "average_inference_time_ms": np.mean(self.inference_times) if self.inference_times else 0,
            "p95_inference_time_ms": np.percentile(self.inference_times, 95) if self.inference_times else 0,
            "p99_inference_time_ms": np.percentile(self.inference_times, 99) if self.inference_times else 0,
            "batch_queue_size": len(self.batch_queue),
            "batch_processing_active": self.running
        }
        
        if self.thompson_sampler:
            stats["thompson_sampling"] = self.thompson_sampler.get_statistics()
        
        return stats
    
    def reset_cache(self) -> None:
        """Clear prediction cache."""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")