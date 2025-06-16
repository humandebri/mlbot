#!/usr/bin/env python3
"""
FeatureAdapter for 44-dimension high-performance model
Convert 156 features to 44 selected features for the improved model
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureAdapter44:
    """
    Adapter to convert 156-dimensional features to 44-dimensional features
    for the improved high-performance model (AUC 0.838)
    """
    
    def __init__(self):
        # 44 features used by the improved model
        self.target_features = [
            "returns", "log_returns", "hl_ratio", "oc_ratio",
            "return_1", "return_3", "return_5", "return_10", "return_20",
            "vol_5", "vol_10", "vol_20", "vol_30",
            "vol_ratio_10", "vol_ratio_20",
            "price_vs_sma_5", "price_vs_sma_10", "price_vs_sma_20", "price_vs_sma_30",
            "price_vs_ema_5", "price_vs_ema_12",
            "macd", "macd_hist",
            "rsi_14", "rsi_21",
            "bb_position_20", "bb_width_20",
            "volume_ratio_10", "volume_ratio_20",
            "log_volume", "volume_price_trend",
            "momentum_3", "momentum_5", "momentum_10",
            "price_percentile_20", "price_percentile_50",
            "trend_strength_short", "trend_strength_long",
            "high_vol_regime", "low_vol_regime", "trending_market",
            "hour_sin", "hour_cos", "is_weekend"
        ]
        
        # Mapping from 156-feature names to 44-feature names
        self.feature_mapping = {
            # Direct matches
            "returns": "returns",
            "log_returns": "log_returns", 
            "hl_ratio": "hl_ratio",
            "oc_ratio": "oc_ratio",
            "return_1": "return_1",
            "return_3": "return_3", 
            "return_5": "return_5",
            "return_10": "return_10",
            "return_20": "return_20",
            "vol_5": "vol_5",
            "vol_10": "vol_10",
            "vol_20": "vol_20", 
            "vol_30": "vol_30",
            "vol_ratio_10": "vol_ratio_10",
            "vol_ratio_20": "vol_ratio_20",
            "price_vs_sma_5": "price_vs_sma_5",
            "price_vs_sma_10": "price_vs_sma_10",
            "price_vs_sma_20": "price_vs_sma_20",
            "price_vs_sma_30": "price_vs_sma_30", 
            "price_vs_ema_5": "price_vs_ema_5",
            "price_vs_ema_12": "price_vs_ema_12",
            "macd": "macd",
            "macd_hist": "macd_hist",
            "rsi_14": "rsi_14",
            "rsi_21": "rsi_21", 
            "bb_position_20": "bb_position_20",
            "bb_width_20": "bb_width_20",
            "volume_ratio_10": "volume_ratio_10",
            "volume_ratio_20": "volume_ratio_20",
            "log_volume": "log_volume",
            "volume_price_trend": "volume_price_trend",
            "momentum_3": "momentum_3",
            "momentum_5": "momentum_5",
            "momentum_10": "momentum_10",
            "price_percentile_20": "price_percentile_20",
            "price_percentile_50": "price_percentile_50",
            "trend_strength_short": "trend_strength_short",
            "trend_strength_long": "trend_strength_long", 
            "high_vol_regime": "high_vol_regime",
            "low_vol_regime": "low_vol_regime",
            "trending_market": "trending_market",
            "hour_sin": "hour_sin",
            "hour_cos": "hour_cos",
            "is_weekend": "is_weekend",
            
            # Alternative names for compatibility
            "volume_trend": "volume_price_trend",
            "volatility_5": "vol_5",
            "volatility_10": "vol_10", 
            "volatility_20": "vol_20",
            "volatility_30": "vol_30",
            "sma_ratio_5": "price_vs_sma_5",
            "sma_ratio_10": "price_vs_sma_10",
            "sma_ratio_20": "price_vs_sma_20",
            "sma_ratio_30": "price_vs_sma_30",
            "ema_ratio_5": "price_vs_ema_5", 
            "ema_ratio_12": "price_vs_ema_12",
            "macd_line": "macd",
            "macd_histogram": "macd_hist",
            "bollinger_position": "bb_position_20",
            "bollinger_width": "bb_width_20",
            "momentum_short": "momentum_3",
            "momentum_medium": "momentum_5",
            "momentum_long": "momentum_10",
            "percentile_20": "price_percentile_20",
            "percentile_50": "price_percentile_50",
            "trend_short": "trend_strength_short",
            "trend_long": "trend_strength_long",
            "high_volatility": "high_vol_regime",
            "low_volatility": "low_vol_regime",
            "trending": "trending_market"
        }
        
        logger.info(f"FeatureAdapter44 initialized with {len(self.target_features)} target features")
    
    def adapt(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert 156-dimensional feature dict to 44-dimensional array
        
        Args:
            features: Dictionary of features from FeatureHub
            
        Returns:
            np.ndarray: 44-dimensional feature array
        """
        try:
            adapted_features = []
            missing_features = []
            
            for target_feature in self.target_features:
                value = None
                
                # Try direct mapping first
                if target_feature in features:
                    value = features[target_feature]
                else:
                    # Try alternative names
                    for alt_name in self.feature_mapping:
                        if (self.feature_mapping[alt_name] == target_feature and 
                            alt_name in features):
                            value = features[alt_name]
                            break
                
                if value is not None:
                    # Handle infinite and NaN values
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    adapted_features.append(float(value))
                else:
                    # Feature not found, use default value based on feature type
                    default_value = self._get_default_value(target_feature)
                    adapted_features.append(default_value)
                    missing_features.append(target_feature)
            
            if missing_features:
                logger.warning(f"Missing features (using defaults): {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            
            result = np.array(adapted_features, dtype=np.float32)
            
            # Validate output shape
            if result.shape[0] != 44:
                logger.error(f"Invalid output shape: {result.shape}, expected (44,)")
                return np.zeros(44, dtype=np.float32)
            
            return result
            
        except Exception as e:
            logger.error(f"Feature adaptation failed: {e}")
            return np.zeros(44, dtype=np.float32)
    
    def _get_default_value(self, feature_name: str) -> float:
        """Get default value for missing features based on feature type"""
        
        # Ratio and percentage features
        if any(x in feature_name.lower() for x in ['ratio', 'vs_', 'position', 'percentile']):
            return 0.0
        
        # Volume features  
        elif 'volume' in feature_name.lower():
            if 'log' in feature_name.lower():
                return 10.0  # Reasonable log volume
            else:
                return 1.0  # Neutral ratio
        
        # Volatility features
        elif 'vol' in feature_name.lower():
            return 0.01  # 1% volatility
        
        # Price-based features
        elif any(x in feature_name.lower() for x in ['price', 'sma', 'ema', 'bb']):
            return 0.0
        
        # Returns and momentum
        elif any(x in feature_name.lower() for x in ['return', 'momentum']):
            return 0.0
        
        # Technical indicators
        elif feature_name.lower() in ['rsi_14', 'rsi_21']:
            return 50.0  # Neutral RSI
        elif 'macd' in feature_name.lower():
            return 0.0
        
        # Regime indicators (binary)
        elif any(x in feature_name.lower() for x in ['regime', 'trending', 'weekend']):
            return 0.0
        
        # Time features
        elif any(x in feature_name.lower() for x in ['hour_sin', 'hour_cos']):
            return 0.0
        
        # Default
        else:
            return 0.0
    
    def get_feature_names(self) -> List[str]:
        """Get list of target feature names"""
        return self.target_features.copy()
    
    def validate_input(self, features: Dict[str, float]) -> bool:
        """Validate input features"""
        if not isinstance(features, dict):
            return False
        
        if len(features) == 0:
            return False
        
        # Check if we have at least some of the required features
        matched_features = 0
        for target_feature in self.target_features[:10]:  # Check first 10
            if target_feature in features:
                matched_features += 1
        
        return matched_features >= 3  # At least 3 matches
    
    def get_adaptation_stats(self, features: Dict[str, float]) -> Dict[str, any]:
        """Get statistics about feature adaptation"""
        stats = {
            'input_features': len(features),
            'target_features': len(self.target_features),
            'matched_features': 0,
            'missing_features': 0
        }
        
        for target_feature in self.target_features:
            if target_feature in features:
                stats['matched_features'] += 1
            else:
                stats['missing_features'] += 1
        
        stats['match_rate'] = stats['matched_features'] / len(self.target_features)
        
        return stats

if __name__ == "__main__":
    # Test the adapter
    adapter = FeatureAdapter44()
    
    # Mock features for testing
    test_features = {
        'returns': 0.001,
        'log_returns': 0.0009,
        'hl_ratio': 0.002,
        'vol_20': 0.015,
        'rsi_14': 55.0,
        'extra_feature_1': 1.0,
        'extra_feature_2': 2.0
    }
    
    print("Testing FeatureAdapter44...")
    print(f"Input features: {len(test_features)}")
    
    adapted = adapter.adapt(test_features)
    print(f"Output shape: {adapted.shape}")
    print(f"Output sample: {adapted[:5]}")
    
    stats = adapter.get_adaptation_stats(test_features)
    print(f"Adaptation stats: {stats}")
    
    print("✅ FeatureAdapter44 test completed")