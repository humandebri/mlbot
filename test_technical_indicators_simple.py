#!/usr/bin/env python3
"""
Simple test script to verify technical indicator generation.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_hub.technical_indicators import TechnicalIndicatorEngine
from src.ml_pipeline.feature_adapter_44 import FeatureAdapter44

def test_technical_indicators():
    """Test the technical indicator generation."""
    
    print("🔧 Testing Technical Indicator Generation")
    print("=" * 50)
    
    # 1. Initialize components
    print("\n1. Initializing components...")
    technical_engine = TechnicalIndicatorEngine()
    feature_adapter = FeatureAdapter44()
    print("   ✅ Components initialized")
    
    # 2. Generate test kline data
    print("\n2. Generating test kline data...")
    test_data = {
        "symbol": "BTCUSDT",
        "open": 106000.0,
        "high": 106500.0,
        "low": 105500.0,
        "close": 106250.0,
        "volume": 1234567.89
    }
    print(f"   Test data: {test_data}")
    
    # 3. Calculate technical indicators
    print("\n3. Calculating technical indicators...")
    features = technical_engine.update_price_data(
        test_data["symbol"],
        test_data["open"],
        test_data["high"],
        test_data["low"],
        test_data["close"],
        test_data["volume"]
    )
    
    print(f"   ✅ Generated {len(features)} features")
    
    # Show all 44 features
    print("\n   All 44 technical indicators:")
    expected_features = feature_adapter.get_feature_names()
    for i, feature_name in enumerate(expected_features):
        value = features.get(feature_name, "MISSING")
        status = "✅" if feature_name in features else "❌"
        print(f"   {i+1:2d}. {status} {feature_name:25s} = {value}")
    
    # 4. Adapt features for model
    print("\n4. Adapting features for ML model...")
    feature_array = feature_adapter.adapt(features)
    print(f"   Feature array shape: {feature_array.shape}")
    
    # Check adaptation statistics
    stats = feature_adapter.get_adaptation_stats(features)
    print(f"\n   Adaptation statistics:")
    print(f"   - Input features: {stats['input_features']}")
    print(f"   - Target features: {stats['target_features']}")
    print(f"   - Matched features: {stats['matched_features']}")
    print(f"   - Missing features: {stats['missing_features']}")
    print(f"   - Match rate: {stats['match_rate']:.2%}")
    
    # 5. Build price history
    print("\n5. Testing with price history...")
    
    # Simulate 50 price updates to build history
    prices = np.linspace(105000, 107000, 50)
    for i, price in enumerate(prices):
        features = technical_engine.update_price_data(
            "BTCUSDT",
            price - 100,  # open
            price + 200,  # high
            price - 200,  # low
            price,        # close
            1000000 + i * 10000  # volume
        )
    
    print("   ✅ Built price history with 50 data points")
    
    # Check final features
    final_features = technical_engine.get_latest_features("BTCUSDT")
    final_array = feature_adapter.adapt(final_features)
    final_stats = feature_adapter.get_adaptation_stats(final_features)
    
    print(f"\n   Final feature generation:")
    print(f"   - Generated features: {len(final_features)}")
    print(f"   - Match rate: {final_stats['match_rate']:.2%}")
    print(f"   - All zeros?: {np.all(final_array == 0)}")
    
    # Show some key indicators
    print("\n   Key indicator values:")
    key_indicators = ["returns", "vol_20", "rsi_14", "macd", "bb_position_20", "trend_strength_long"]
    for indicator in key_indicators:
        value = final_features.get(indicator, "MISSING")
        print(f"   - {indicator}: {value}")
    
    print("\n" + "=" * 50)
    
    if final_stats['match_rate'] == 1.0:
        print("✅ SUCCESS: All 44 technical indicators are being generated correctly!")
    else:
        print(f"⚠️  WARNING: Only {final_stats['match_rate']:.0%} of features matched.")
        print("   Some features may still be missing.")
    
    print("\nTest completed!")


if __name__ == "__main__":
    test_technical_indicators()