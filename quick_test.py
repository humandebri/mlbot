#!/usr/bin/env python3
"""
Quick test to see actual prediction values
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

from src.feature_hub.main import FeatureHub
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.common.config import settings
from src.common.discord_notifier import discord_notifier

def quick_test():
    """Quick test of current prediction values"""
    print("🔍 Quick prediction test...")
    
    # Initialize Feature Hub
    feature_hub = FeatureHub()
    
    # Initialize inference engine with lower threshold
    inference_config = InferenceConfig(
        model_path=settings.model.model_path,
        confidence_threshold=0.3  # Lower threshold for testing
    )
    inference_engine = InferenceEngine(inference_config)
    inference_engine.load_model()
    
    print("✅ Components initialized")
    
    # Test each symbol
    for symbol in settings.bybit.symbols:
        print(f"\n🔍 {symbol}:")
        
        features = feature_hub.get_latest_features(symbol)
        print(f"  Features: {len(features) if features else 0}")
        
        if features and len(features) > 10:
            try:
                result = inference_engine.predict(features)
                prediction = result["predictions"][0] if result["predictions"] else 0
                confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                
                print(f"  📊 Prediction: {prediction:.6f}")
                print(f"  📊 Confidence: {confidence:.4f} ({confidence:.2%})")
                
                # Test with lower thresholds
                for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    if confidence > threshold:
                        print(f"  ✅ Would trigger at {threshold:.1%} threshold")
                        if threshold == 0.4:  # Send notification at 40%
                            print(f"  📲 Sending test notification...")
                            discord_notifier.send_trade_signal(
                                symbol=symbol,
                                side="BUY" if prediction > 0 else "SELL",
                                price=features.get("close", 50000),
                                confidence=confidence,
                                expected_pnl=prediction
                            )
                        break
                else:
                    print(f"  ⚠️ Below all thresholds (max confidence: {confidence:.2%})")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        else:
            print(f"  ❌ Insufficient features")
    
    print("\n✅ Quick test completed")

if __name__ == "__main__":
    quick_test()