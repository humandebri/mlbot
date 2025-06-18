#!/usr/bin/env python3
"""
Test script to verify technical indicator integration with ML model.
"""

import asyncio
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_hub.technical_indicators import TechnicalIndicatorEngine
from src.ml_pipeline.feature_adapter_44 import FeatureAdapter44
# from src.ml_pipeline.inference_engine import InferenceEngine
# from src.common.config import settings

async def test_technical_indicators():
    """Test the technical indicator generation and ML prediction pipeline."""
    
    print("🔧 Testing Technical Indicator Integration")
    print("=" * 50)
    
    # 1. Initialize components
    print("\n1. Initializing components...")
    technical_engine = TechnicalIndicatorEngine()
    feature_adapter = FeatureAdapter44()
    
    # Load the model directly without InferenceEngine for testing
    import onnxruntime as ort
    
    model_path = "models/v3.1_improved/model.onnx"
    scaler_path = "models/v3.1_improved/scaler.pkl"
    
    print(f"   Model path: {model_path}")
    print(f"   Scaler path: {scaler_path}")
    
    await inference_engine.load_model(model_path, scaler_path)
    print("   ✅ Model loaded successfully")
    
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
    
    print(f"   Generated {len(features)} features")
    print("\n   Sample features:")
    for i, (key, value) in enumerate(list(features.items())[:10]):
        print(f"   - {key}: {value:.6f}")
    
    # 4. Adapt features for model
    print("\n4. Adapting features for ML model...")
    feature_array = feature_adapter.adapt(features)
    print(f"   Feature array shape: {feature_array.shape}")
    print(f"   Feature array sample: {feature_array[:5]}")
    
    # Check adaptation statistics
    stats = feature_adapter.get_adaptation_stats(features)
    print(f"\n   Adaptation statistics:")
    print(f"   - Input features: {stats['input_features']}")
    print(f"   - Target features: {stats['target_features']}")
    print(f"   - Matched features: {stats['matched_features']}")
    print(f"   - Match rate: {stats['match_rate']:.2%}")
    
    # 5. Make prediction
    print("\n5. Making ML prediction...")
    prediction = await inference_engine.predict(feature_array.tolist())
    
    print(f"\n   Prediction result:")
    print(f"   - Confidence: {prediction['confidence']:.4f}")
    print(f"   - Expected PnL: {prediction['expected_pnl']:.4f}")
    print(f"   - Direction: {prediction['direction']}")
    print(f"   - Raw scores: {prediction.get('scores', 'N/A')}")
    
    # 6. Verify prediction is not all zeros
    print("\n6. Verification:")
    if prediction['confidence'] == 0.0 and prediction['expected_pnl'] == 0.0:
        print("   ❌ FAILURE: Model still returning all zeros!")
        print("   This means the technical indicators are not being used correctly.")
    else:
        print("   ✅ SUCCESS: Model is making non-zero predictions!")
        print("   Technical indicators are being used correctly.")
    
    # 7. Test with multiple price updates to build history
    print("\n7. Testing with price history...")
    
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
    
    # Now make prediction with full history
    print("   Built price history with 50 data points")
    
    final_features = technical_engine.get_latest_features("BTCUSDT")
    final_array = feature_adapter.adapt(final_features)
    final_prediction = await inference_engine.predict(final_array.tolist())
    
    print(f"\n   Final prediction with history:")
    print(f"   - Confidence: {final_prediction['confidence']:.4f}")
    print(f"   - Expected PnL: {final_prediction['expected_pnl']:.4f}")
    print(f"   - Direction: {final_prediction['direction']}")
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(test_technical_indicators())