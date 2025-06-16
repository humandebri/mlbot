#!/usr/bin/env python3
"""
Debug script to test trading loop functionality
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

from src.common.config import settings
from src.feature_hub.main import FeatureHub
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.common.discord_notifier import discord_notifier
import asyncio

async def test_trading_loop():
    """Test the trading loop components"""
    print("Testing trading loop components...")
    
    # Initialize components
    feature_hub = FeatureHub()
    
    inference_config = InferenceConfig(
        model_path=settings.model.model_path,
        confidence_threshold=0.6
    )
    inference_engine = InferenceEngine(inference_config)
    
    # Load model
    inference_engine.load_model()
    print(f"✅ Model loaded: {settings.model.model_path}")
    
    # Start feature hub
    await feature_hub.start()
    print("✅ Feature Hub started")
    
    # Test each symbol
    for symbol in settings.bybit.symbols:
        print(f"\n🔍 Testing {symbol}:")
        
        features = feature_hub.get_latest_features(symbol)
        print(f"  Features count: {len(features) if features else 0}")
        
        if features:
            print(f"  Sample features: {list(features.keys())[:5]}...")
            
            if len(features) > 10:
                try:
                    result = inference_engine.predict(features)
                    prediction = result["predictions"][0] if result["predictions"] else 0
                    confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                    
                    print(f"  ✅ Prediction: {prediction:.4f}, Confidence: {confidence:.2%}")
                    
                    if confidence > 0.6:
                        print(f"  🚨 HIGH CONFIDENCE! Sending Discord notification...")
                        discord_notifier.send_trade_signal(
                            symbol=symbol,
                            side="BUY" if prediction > 0 else "SELL",
                            price=features.get("close", 50000),
                            confidence=confidence,
                            expected_pnl=prediction
                        )
                    else:
                        print(f"  ⚠️ Confidence too low: {confidence:.2%} < 60%")
                        
                except Exception as e:
                    print(f"  ❌ Prediction error: {e}")
            else:
                print(f"  ⚠️ Not enough features: {len(features)} < 10")
        else:
            print("  ❌ No features available")
    
    await feature_hub.stop()
    print("\n✅ Test completed")

if __name__ == "__main__":
    asyncio.run(test_trading_loop())