"""
Feature adapter to convert between different feature dimensions.
Maps 156-dimensional feature vectors to 26-dimensional model inputs.
"""

import numpy as np
from typing import Dict, List, Union
from ..common.logging import get_logger

logger = get_logger(__name__)


class FeatureAdapter:
    """Adapts feature vectors to match model expectations."""
    
    # Define the most important features to keep (based on typical trading signals)
    FEATURE_MAPPING = {
        # Price-based features (0-5)
        'price_change_5m': 0,
        'price_change_15m': 1,
        'price_change_1h': 2,
        'volatility_5m': 3,
        'volatility_15m': 4,
        'volatility_1h': 5,
        
        # Volume features (6-11)
        'volume_ratio_5m': 6,
        'volume_ratio_15m': 7,
        'volume_ratio_1h': 8,
        'buy_sell_ratio_5m': 9,
        'buy_sell_ratio_15m': 10,
        'buy_sell_ratio_1h': 11,
        
        # Liquidation features (12-17)
        'liquidation_volume_long': 12,
        'liquidation_volume_short': 13,
        'liquidation_ratio': 14,
        'liquidation_spike_score': 15,
        'liquidation_momentum': 16,
        'liquidation_cascade_risk': 17,
        
        # Market microstructure (18-23)
        'spread': 18,
        'bid_ask_imbalance': 19,
        'order_flow_imbalance': 20,
        'trade_intensity': 21,
        'market_depth_ratio': 22,
        'price_impact': 23,
        
        # Technical indicators (24-25)
        'rsi': 24,
        'momentum': 25
    }
    
    @staticmethod
    def adapt_features(features: Union[Dict[str, float], np.ndarray, List[float]]) -> np.ndarray:
        """
        Adapt features from any format to 26-dimensional array expected by model.
        
        Args:
            features: Input features (dict, array, or list)
            
        Returns:
            26-dimensional numpy array
        """
        # Convert to numpy array
        if isinstance(features, dict):
            # If dict, try to extract known features
            output = np.zeros(26)
            
            # Map known features
            for feature_name, idx in FeatureAdapter.FEATURE_MAPPING.items():
                if feature_name in features:
                    output[idx] = features[feature_name]
            
            # If dict has numeric keys, use first 26 values
            if all(isinstance(k, (int, str)) for k in features.keys()):
                sorted_keys = sorted(features.keys())[:26]
                for i, key in enumerate(sorted_keys):
                    if i < 26:
                        output[i] = float(features[key])
            
            return output
            
        elif isinstance(features, (list, tuple)):
            features = np.array(features, dtype=np.float32)
        
        elif not isinstance(features, np.ndarray):
            raise ValueError(f"Unsupported feature type: {type(features)}")
        
        # Handle different array sizes
        if features.size >= 26:
            # Take first 26 features
            return features.flatten()[:26].astype(np.float32)
        else:
            # Pad with zeros if less than 26
            output = np.zeros(26, dtype=np.float32)
            output[:features.size] = features.flatten()
            return output
    
    @staticmethod
    def adapt_batch(batch_features: List[Union[Dict, np.ndarray, List]]) -> np.ndarray:
        """Adapt a batch of features."""
        adapted = []
        for features in batch_features:
            adapted.append(FeatureAdapter.adapt_features(features))
        return np.array(adapted, dtype=np.float32)