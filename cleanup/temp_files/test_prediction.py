#!/usr/bin/env python3
"""
Simple prediction test
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

from src.common.config import settings
from src.feature_hub.main import FeatureHub
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.common.discord_notifier import discord_notifier
import redis

def test_prediction():
    """Test prediction with real data"""
    print("🔍 Testing prediction with real data...")
    
    # Get Redis features directly
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    # Initialize inference engine
    inference_config = InferenceConfig(
        model_path=settings.model.model_path,
        confidence_threshold=0.6
    )
    inference_engine = InferenceEngine(inference_config)
    inference_engine.load_model()
    print("✅ Model loaded")
    
    # Test each symbol
    for symbol in ["BTCUSDT", "ETHUSDT", "ICPUSDT"]:
        print(f"\n🔍 Testing {symbol}:")
        
        # Get features from Redis
        features_data = r.hgetall(f"features:{symbol}:latest")
        
        if features_data:
            features = {k.decode(): float(v.decode()) for k, v in features_data.items()}
            print(f"  Redis features count: {len(features)}")
            print(f"  Sample features: {list(features.keys())[:5]}...")
            
            try:
                result = inference_engine.predict(features)
                prediction = result["predictions"][0] if result["predictions"] else 0
                confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                
                print(f"  ✅ Prediction: {prediction:.4f}, Confidence: {confidence:.2%}")
                
                if confidence > 0.6:
                    print(f"  🚨 HIGH CONFIDENCE! This should trigger Discord notification")
                    
                    # Send test notification
                    discord_notifier.send_trade_signal(
                        symbol=symbol,
                        side="BUY" if prediction > 0 else "SELL",
                        price=features.get("close", 50000),
                        confidence=confidence,
                        expected_pnl=prediction
                    )
                    print(f"  📲 Discord notification sent!")
                else:
                    print(f"  ⚠️ Confidence too low: {confidence:.2%} < 60%")
                    
            except Exception as e:
                print(f"  ❌ Prediction error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ❌ No features in Redis for {symbol}")
    
    print("\n✅ Test completed")

if __name__ == "__main__":
    test_prediction()