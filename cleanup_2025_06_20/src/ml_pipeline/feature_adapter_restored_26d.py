#!/usr/bin/env python3
"""
FeatureAdapter for the restored high-performance 26-dimension model
Convert any number of features to the specific 26 features expected by balanced_restored_26d model
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureAdapterRestored26D:
    """
    Adapter to convert any-dimensional features to 26-dimensional features
    for the restored high-performance model (balanced_restored_26d)
    """
    
    def __init__(self):
        # Exact 26 features that our restored model was trained on
        self.target_features = [
            "returns",
            "log_returns", 
            "hl_ratio",
            "oc_ratio",
            "return_1",
            "return_3",
            "return_5",
            "return_10",
            "return_15",
            "return_30",
            "vol_5",
            "vol_10",
            "vol_20",
            "price_vs_sma_5",
            "price_vs_sma_10",
            "price_vs_sma_20",
            "rsi",
            "bb_position",
            "macd_hist",
            "volume_ratio",
            "log_volume",
            "volume_price_change",
            "momentum_3",
            "momentum_5",
            "trend_strength",
            "price_above_ma"
        ]
        
        # Mapping from various feature names to our target features
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
            "return_15": "return_15",
            "return_30": "return_30",
            "vol_5": "vol_5",
            "vol_10": "vol_10",
            "vol_20": "vol_20",
            "price_vs_sma_5": "price_vs_sma_5",
            "price_vs_sma_10": "price_vs_sma_10",
            "price_vs_sma_20": "price_vs_sma_20",
            "rsi": "rsi",
            "bb_position": "bb_position",
            "macd_hist": "macd_hist",
            "volume_ratio": "volume_ratio",
            "log_volume": "log_volume",
            "volume_price_change": "volume_price_change",
            "momentum_3": "momentum_3",
            "momentum_5": "momentum_5",
            "trend_strength": "trend_strength",
            "price_above_ma": "price_above_ma",
            
            # Alternative names (common variations)
            "high_low_ratio": "hl_ratio",
            "open_close_ratio": "oc_ratio",
            "volatility_5": "vol_5",
            "volatility_10": "vol_10",
            "volatility_20": "vol_20",
            "rsi_14": "rsi",
            "bb_position_20": "bb_position",
            "bollinger_position": "bb_position",
            "macd_histogram": "macd_hist",
            "volume_change": "volume_ratio",
            "log_volume_ratio": "log_volume",
            "trend_strength_short": "trend_strength",
            "price_above_sma20": "price_above_ma",
            
            # Technical indicator variations
            "close_vs_sma5": "price_vs_sma_5",
            "close_vs_sma10": "price_vs_sma_10", 
            "close_vs_sma20": "price_vs_sma_20",
            "sma5_ratio": "price_vs_sma_5",
            "sma10_ratio": "price_vs_sma_10",
            "sma20_ratio": "price_vs_sma_20",
        }
        
        logger.info(f"FeatureAdapterRestored26D initialized with {len(self.target_features)} target features")
    
    def adapt(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert any-dimensional feature dict to 26-dimensional array for restored model
        
        Args:
            features: Dictionary of features from FeatureHub
            
        Returns:
            np.ndarray: 26-dimensional feature array
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
                    
                    # Try to derive the feature from available data
                    if value is None:
                        value = self._derive_feature(target_feature, features)
                
                if value is not None:
                    # Handle infinite and NaN values
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    # Convert to float and keep raw value (model was trained on raw features)
                    adapted_features.append(float(value))
                else:
                    # Feature not found, use default value
                    default_value = self._get_default_value(target_feature)
                    adapted_features.append(default_value)
                    missing_features.append(target_feature)
            
            if missing_features:
                logger.debug(f"Missing features (using defaults): {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            
            result = np.array(adapted_features, dtype=np.float32)
            
            # Validate output shape
            if result.shape[0] != 26:
                logger.error(f"Invalid output shape: {result.shape}, expected (26,)")
                return np.zeros(26, dtype=np.float32)
            
            return result
            
        except Exception as e:
            logger.error(f"Feature adaptation failed: {e}")
            return np.zeros(26, dtype=np.float32)
    
    def _derive_feature(self, target_feature: str, features: Dict[str, float]) -> Optional[float]:
        """Try to derive missing features from available data"""
        
        # Calculate returns if price data is available
        if target_feature == "returns" and "close" in features:
            # For live data, we can't calculate returns without previous close
            return 0.0
        
        # Calculate log_returns from returns
        if target_feature == "log_returns" and "returns" in features:
            return np.log(1 + features["returns"]) if features["returns"] > -0.99 else -1.0
        
        # Calculate hl_ratio from OHLC
        if target_feature == "hl_ratio":
            if "high" in features and "low" in features and "close" in features:
                return (features["high"] - features["low"]) / features["close"]
        
        # Calculate oc_ratio from open/close
        if target_feature == "oc_ratio":
            if "open" in features and "close" in features and features["open"] > 0:
                return (features["close"] - features["open"]) / features["open"]
        
        # Calculate volume_ratio
        if target_feature == "volume_ratio":
            if "volume" in features and "volume_ma" in features and features["volume_ma"] > 0:
                return features["volume"] / features["volume_ma"]
        
        # Calculate log_volume
        if target_feature == "log_volume":
            if "volume" in features:
                return np.log(features["volume"] + 1)
        
        # Calculate volume_price_change
        if target_feature == "volume_price_change":
            if "volume_ratio" in features and "returns" in features:
                return features["volume_ratio"] * abs(features["returns"])
            elif "volume_ratio" in self._derive_feature("volume_ratio", features) and "returns" in features:
                vol_ratio = self._derive_feature("volume_ratio", features)
                if vol_ratio is not None:
                    return vol_ratio * abs(features["returns"])
        
        # Calculate RSI from rsi_14 or similar
        if target_feature == "rsi":
            for rsi_variant in ["rsi_14", "rsi_21", "RSI", "rsi_indicator"]:
                if rsi_variant in features:
                    return features[rsi_variant]
        
        # Calculate trend_strength from SMA comparison
        if target_feature == "trend_strength":
            if "sma_5" in features and "sma_20" in features and features["sma_20"] > 0:
                return (features["sma_5"] - features["sma_20"]) / features["sma_20"]
        
        # Calculate price_above_ma
        if target_feature == "price_above_ma":
            if "close" in features and "sma_20" in features:
                return 1.0 if features["close"] > features["sma_20"] else 0.0
        
        # Calculate price_vs_sma features
        for sma_period in [5, 10, 20]:
            if target_feature == f"price_vs_sma_{sma_period}":
                sma_key = f"sma_{sma_period}"
                if "close" in features and sma_key in features and features[sma_key] > 0:
                    return (features["close"] - features[sma_key]) / features[sma_key]
        
        return None
    
    def _get_default_value(self, feature_name: str) -> float:
        """Get default value for missing features based on feature type and typical ranges"""
        
        # Returns and momentum features - centered around 0
        if any(x in feature_name.lower() for x in ['return', 'momentum']):
            return 0.0
        
        # Volatility features - small positive values
        elif 'vol_' in feature_name:
            return 0.015  # 1.5% typical volatility
        
        # Ratio features that compare prices/volumes
        elif any(x in feature_name for x in ['_ratio', '_vs_']):
            return 0.0  # Neutral relative position
        
        # RSI - neutral value
        elif 'rsi' in feature_name.lower():
            return 50.0  # Neutral RSI
        
        # Bollinger band position - neutral
        elif 'bb_position' in feature_name:
            return 0.5  # Middle of Bollinger bands
        
        # MACD histogram - neutral
        elif 'macd' in feature_name.lower():
            return 0.0
        
        # Volume features
        elif 'volume' in feature_name.lower():
            if 'log' in feature_name:
                return 13.8  # log(1M) typical volume
            elif 'ratio' in feature_name:
                return 1.0   # Average volume ratio
            else:
                return 0.0
        
        # Binary indicators
        elif any(x in feature_name for x in ['above', 'trend', 'strength']):
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
        for target_feature in self.target_features:
            if target_feature in features:
                matched_features += 1
            else:
                # Check alternative names
                for alt_name in self.feature_mapping:
                    if (self.feature_mapping[alt_name] == target_feature and 
                        alt_name in features):
                        matched_features += 1
                        break
        
        return matched_features >= 3  # At least 3 matches for basic functionality
    
    def get_adaptation_stats(self, features: Dict[str, float]) -> Dict[str, any]:
        """Get statistics about feature adaptation"""
        stats = {
            'input_features': len(features),
            'target_features': len(self.target_features),
            'matched_features': 0,
            'missing_features': 0,
            'derived_features': 0
        }
        
        for target_feature in self.target_features:
            if target_feature in features:
                stats['matched_features'] += 1
            else:
                # Check alternative names
                found = False
                for alt_name in self.feature_mapping:
                    if (self.feature_mapping[alt_name] == target_feature and 
                        alt_name in features):
                        stats['matched_features'] += 1
                        found = True
                        break
                
                if not found:
                    # Check if we can derive it
                    derived = self._derive_feature(target_feature, features)
                    if derived is not None:
                        stats['derived_features'] += 1
                        stats['matched_features'] += 1
                    else:
                        stats['missing_features'] += 1
        
        stats['match_rate'] = stats['matched_features'] / len(self.target_features)
        
        return stats

if __name__ == "__main__":
    # Test the adapter
    adapter = FeatureAdapterRestored26D()
    
    # Mock features similar to what FeatureHub generates
    test_features = {
        'returns': 0.0072,
        'log_returns': 0.0071,
        'close': 70000.0,
        'open': 69500.0,
        'high': 70200.0,
        'low': 69300.0,
        'volume': 1500000.0,
        'hl_ratio': 0.0013,
        'oc_ratio': 0.0072,
        'return_1': 0.0024,
        'return_3': 0.0051,
        'vol_20': 0.0138,
        'rsi_14': 65.4,
        'volume_ratio': 1.34,
        'sma_5': 69800.0,
        'sma_20': 69000.0,
        'macd_hist': 12.5,
    }
    
    print("Testing FeatureAdapterRestored26D...")
    print(f"Input features: {len(test_features)}")
    print(f"Available features: {list(test_features.keys())}")
    
    adapted = adapter.adapt(test_features)
    print(f"Output shape: {adapted.shape}")
    print(f"Output sample: {adapted[:10]}")
    
    stats = adapter.get_adaptation_stats(test_features)
    print(f"Adaptation stats: {stats}")
    print(f"Match rate: {stats['match_rate']:.1%}")
    
    print("✅ FeatureAdapterRestored26D test completed")