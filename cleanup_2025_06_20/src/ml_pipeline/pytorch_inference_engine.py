"""
PyTorch inference engine for real-time trading predictions.
Compatible with the existing ONNX interface.
"""

import numpy as np
import torch
import torch.nn as nn
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from ..common.config import settings
from ..common.logging import get_logger
from ..common.monitoring import (
    MODEL_PREDICTIONS, MODEL_INFERENCE_TIME,
    increment_counter, observe_histogram
)

logger = get_logger(__name__)


class FastNNModel(nn.Module):
    """Fast neural network for trading signal prediction."""
    
    def __init__(self, input_features=156, hidden_size=64, dropout_rate=0.2):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class PyTorchInferenceEngine:
    """PyTorch-based inference engine with ONNX-compatible interface."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = None
        self.device = torch.device('cpu')
        self.model_loaded = False
        self._prediction_cache = {}
        
    def load_model(self, model_path: str) -> bool:
        """Load PyTorch model and scaler."""
        try:
            # Load model
            if model_path.endswith('.onnx'):
                # For ONNX path, look for PyTorch model instead
                pytorch_path = model_path.replace('catboost_model.onnx', 'fast_nn_final.pth')
                scaler_path = model_path.replace('catboost_model.onnx', 'fast_nn_scaler.pkl')
            else:
                pytorch_path = model_path
                scaler_path = model_path.replace('.pth', '_scaler.pkl')
            
            # Check if files exist
            if not Path(pytorch_path).exists():
                logger.error(f"PyTorch model not found: {pytorch_path}")
                return False
                
            # Load model
            self.model = FastNNModel()
            state_dict = torch.load(pytorch_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Load scaler if available
            if Path(scaler_path).exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            
            self.model_loaded = True
            logger.info(f"Successfully loaded PyTorch model from {pytorch_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return False
    
    def predict(self, features: Union[np.ndarray, List[List[float]]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Returns:
            Tuple of (predictions, confidences)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Convert to numpy array if needed
            if isinstance(features, list):
                features = np.array(features, dtype=np.float32)
            
            # Ensure 2D array
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features if scaler available
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Convert to torch tensor
            with torch.no_grad():
                x = torch.FloatTensor(features).to(self.device)
                predictions = self.model(x).cpu().numpy()
            
            # Convert to binary predictions with confidence
            binary_predictions = (predictions > 0.5).astype(int).flatten()
            confidences = np.abs(predictions - 0.5) * 2  # Scale to 0-1
            confidences = confidences.flatten()
            
            # Record metrics
            inference_time = (time.time() - start_time) * 1000
            observe_histogram(MODEL_INFERENCE_TIME, inference_time / 1000)
            increment_counter(MODEL_PREDICTIONS, symbol="all")
            
            return binary_predictions, confidences
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_proba(self, features: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """Get probability predictions."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert to numpy array if needed
            if isinstance(features, list):
                features = np.array(features, dtype=np.float32)
            
            # Ensure 2D array
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features if scaler available
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Convert to torch tensor
            with torch.no_grad():
                x = torch.FloatTensor(features).to(self.device)
                probabilities = self.model(x).cpu().numpy()
            
            return probabilities.flatten()
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.model_loaded:
            return {
                "loaded": False,
                "type": "PyTorch Neural Network",
                "status": "Not loaded"
            }
        
        return {
            "loaded": True,
            "type": "PyTorch Neural Network",
            "architecture": "FastNN",
            "input_features": 156,
            "device": str(self.device),
            "scaler_loaded": self.scaler is not None,
            "status": "Ready"
        }


# Create a wrapper to match ONNX InferenceEngine interface
class InferenceEngine:
    """Wrapper to provide ONNX-compatible interface for PyTorch engine."""
    
    def __init__(self, config):
        self.config = config
        self.engine = PyTorchInferenceEngine(config)
        self.model_loaded = False
        
    def load_model(self, model_path: str) -> bool:
        """Load model with fallback to PyTorch if ONNX fails."""
        success = self.engine.load_model(model_path)
        self.model_loaded = success
        return success
    
    def predict(self, features: Union[np.ndarray, List[List[float]]]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        return self.engine.predict(features)
    
    def predict_proba(self, features: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """Get probability predictions."""
        return self.engine.predict_proba(features)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.engine.get_model_info()