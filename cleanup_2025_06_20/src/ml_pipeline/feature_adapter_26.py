#!/usr/bin/env python3
"""
FeatureAdapter for 26-dimension working model
Convert any number of features to 26 selected features for the v1.0 model
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureAdapter26:
    """
    Adapter to convert any-dimensional features to 26-dimensional features
    for the working v1.0 model
    """
    
    def __init__(self):
        # 26 core features for the working model
        self.target_features = [
            # Basic price features (8)
            "returns", "log_returns", "close", "volume",
            "price_change_pct", "high_low_ratio", "volume_ratio", "volatility_20",
            
            # Technical indicators (10)
            "rsi_14", "macd", "bb_position_20", "bb_width_20",
            "sma_5", "sma_10", "sma_20", "close_to_sma5", "close_to_sma10", "close_to_sma20",
            
            # Advanced features (5)
            "trend_strength_short", "trend_strength_long", "market_regime",
            "momentum_3", "momentum_5",
            
            # Time features (3)
            "hour_sin", "hour_cos", "is_weekend"
        ]
        
        # Mapping from various feature names to target features
        self.feature_mapping = {
            # Direct matches
            "returns": "returns",
            "log_returns": "log_returns", 
            "close": "close",
            "volume": "volume",
            "price_change_pct": "price_change_pct",
            "high_low_ratio": "high_low_ratio",
            "hl_ratio": "high_low_ratio",  # Alternative name
            "volume_ratio": "volume_ratio",
            "volatility_20": "volatility_20",
            "vol_20": "volatility_20",  # Alternative name
            
            # Technical indicators
            "rsi_14": "rsi_14",
            "macd": "macd",
            "bb_position_20": "bb_position_20",
            "bb_width_20": "bb_width_20",
            "sma_5": "sma_5",
            "sma_10": "sma_10", 
            "sma_20": "sma_20",
            "close_to_sma5": "close_to_sma5",
            "close_to_sma10": "close_to_sma10",
            "close_to_sma20": "close_to_sma20",
            
            # Advanced features
            "trend_strength_short": "trend_strength_short",
            "trend_strength_long": "trend_strength_long",
            "market_regime": "market_regime",
            "momentum_3": "momentum_3",
            "momentum_5": "momentum_5",
            
            # Time features
            "hour_sin": "hour_sin",
            "hour_cos": "hour_cos",
            "is_weekend": "is_weekend",
            
            # Alternative mappings
            "body_to_range": "price_change_pct",
            "trades_count": "volume",
            "open": "close",  # Use close as fallback for open
            "high": "close",  # Use close as fallback for high
            "low": "close",   # Use close as fallback for low
        }
        
        logger.info(f"FeatureAdapter26 initialized with {len(self.target_features)} target features")
    
    def adapt(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert any-dimensional feature dict to 26-dimensional array
        
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
                
                if value is not None:
                    # Handle infinite and NaN values
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    # Normalize the feature value
                    normalized_value = self._normalize_feature(target_feature, float(value))
                    adapted_features.append(normalized_value)
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
    
    def _get_default_value(self, feature_name: str) -> float:
        """Get default value for missing features based on feature type"""
        
        # Price features - normalize to 0-1 range
        if feature_name in ["close"]:
            return 0.5  # Normalized price (0.5 = middle range)
        elif feature_name in ["volume"]:
            return 0.5  # Normalized volume
        
        # Ratios and percentages
        elif any(x in feature_name.lower() for x in ['ratio', 'pct', 'position', 'percentile']):
            return 0.0
        
        # Volume features  
        elif 'volume' in feature_name.lower():
            if 'log' in feature_name.lower():
                return 0.5  # Normalized log volume
            else:
                return 1.0  # Neutral ratio
        
        # Volatility features
        elif 'vol' in feature_name.lower() or 'volatility' in feature_name.lower():
            return 0.01  # 1% volatility
        
        # Moving averages - normalize
        elif 'sma' in feature_name.lower():
            return 0.5  # Normalized SMA
        
        # Price-based features
        elif any(x in feature_name.lower() for x in ['price', 'close', 'sma', 'ema', 'bb']):
            return 0.0
        
        # Returns and momentum
        elif any(x in feature_name.lower() for x in ['return', 'momentum']):
            return 0.0
        
        # Technical indicators
        elif 'rsi' in feature_name.lower():
            return 0.5  # Neutral RSI (normalized from 50 to 0.5)
        elif 'macd' in feature_name.lower():
            return 0.0
        
        # Regime indicators
        elif any(x in feature_name.lower() for x in ['regime', 'trending', 'weekend']):
            return 0.0
        
        # Time features
        elif any(x in feature_name.lower() for x in ['hour_sin', 'hour_cos']):
            return 0.0
        
        # Default
        else:
            return 0.0
    
    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """Normalize feature value to expected model input range (roughly 0-1)"""
        
        # Price features - normalize using typical BTC price range (50k-150k)
        if feature_name in ["close"]:
            return np.clip((value - 50000) / 100000, 0, 1)  # 50k-150k -> 0-1
        
        # Volume features - normalize using typical volume range (100k-10M)
        elif feature_name in ["volume"]:
            return np.clip(np.log10(value + 1) / 7, 0, 1)  # log scale, cap at 10M
        
        # SMA features - normalize like close price
        elif feature_name in ["sma_5", "sma_10", "sma_20"]:
            return np.clip((value - 50000) / 100000, 0, 1)
        
        # Close to SMA features - these are already relative, just scale
        elif "close_to_sma" in feature_name:
            return np.clip(value / 0.1, -1, 1)  # ±10% -> ±1
        
        # RSI features - convert from 0-100 to 0-1
        elif "rsi" in feature_name.lower():
            return np.clip(value / 100, 0, 1)
        
        # MACD features - typically small values, scale appropriately
        elif "macd" in feature_name.lower():
            return np.clip(value / 1000, -1, 1)  # ±1000 -> ±1
        
        # Volatility features - scale by typical range (0-5%)
        elif "vol" in feature_name.lower() or "volatility" in feature_name.lower():
            return np.clip(value / 0.05, 0, 1)  # 0-5% -> 0-1
        
        # High-low ratio - typically 1.000-1.050
        elif "high_low_ratio" in feature_name:
            return np.clip((value - 1) / 0.05, 0, 1)  # 1.00-1.05 -> 0-1
        
        # Returns and momentum - typically ±5%
        elif any(x in feature_name.lower() for x in ['return', 'momentum']):
            return np.clip(value / 0.05, -1, 1)  # ±5% -> ±1
        
        # Price change percentage - typically ±5%
        elif "price_change_pct" in feature_name:
            return np.clip(value / 0.05, -1, 1)
        
        # Volume ratios - typically 0.5-2.0
        elif "volume_ratio" in feature_name:
            return np.clip((value - 0.5) / 1.5, 0, 1)  # 0.5-2.0 -> 0-1
        
        # Bollinger bands position - already -1 to 1
        elif "bb_position" in feature_name:
            return np.clip(value, -1, 1)
        
        # Bollinger bands width - typically 0-10%
        elif "bb_width" in feature_name:
            return np.clip(value / 0.1, 0, 1)  # 0-10% -> 0-1
        
        # Trend strength - typically ±1
        elif "trend_strength" in feature_name:
            return np.clip(value, -1, 1)
        
        # Market regime features - already 0 or 1
        elif any(x in feature_name.lower() for x in ['regime', 'trending', 'weekend']):
            return np.clip(value, 0, 1)
        
        # Time features (sin/cos) - already -1 to 1
        elif any(x in feature_name.lower() for x in ['hour_sin', 'hour_cos']):
            return np.clip(value, -1, 1)
        
        # Default: assume already normalized or small values
        else:
            return np.clip(value, -1, 1)
    
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
        
        return matched_features >= 2  # At least 2 matches
    
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
                # Check alternative names
                found = False
                for alt_name in self.feature_mapping:
                    if (self.feature_mapping[alt_name] == target_feature and 
                        alt_name in features):
                        stats['matched_features'] += 1
                        found = True
                        break
                if not found:
                    stats['missing_features'] += 1
        
        stats['match_rate'] = stats['matched_features'] / len(self.target_features)
        
        return stats

if __name__ == "__main__":
    # Test the adapter
    adapter = FeatureAdapter26()
    
    # Mock features for testing
    test_features = {
        'returns': 0.001,
        'close': 106000,
        'volume': 1000000,
        'rsi_14': 55.0,
        'vol_20': 0.015,
        'macd': 10.5,
        'extra_feature_1': 1.0,
        'extra_feature_2': 2.0
    }
    
    print("Testing FeatureAdapter26...")
    print(f"Input features: {len(test_features)}")
    
    adapted = adapter.adapt(test_features)
    print(f"Output shape: {adapted.shape}")
    print(f"Output sample: {adapted[:5]}")
    
    stats = adapter.get_adaptation_stats(test_features)
    print(f"Adaptation stats: {stats}")
    
    print("✅ FeatureAdapter26 test completed")