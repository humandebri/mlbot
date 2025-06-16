#!/usr/bin/env python3
"""
Check if features are being generated and predictions are working
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import asyncio
from src.feature_hub.main import FeatureHub
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.common.config import settings
from src.common.discord_notifier import discord_notifier

async def check_features():
    """Check feature generation and prediction"""
    print("🔍 Checking feature generation...")
    
    # Initialize Feature Hub
    feature_hub = FeatureHub()
    await feature_hub.start()
    
    # Wait for features to accumulate
    print("⏳ Waiting 10 seconds for features to accumulate...")
    await asyncio.sleep(10)
    
    # Initialize inference engine
    inference_config = InferenceConfig(
        model_path=settings.model.model_path,
        confidence_threshold=0.6
    )
    inference_engine = InferenceEngine(inference_config)
    inference_engine.load_model()
    
    print("\n📊 Checking each symbol...")
    for symbol in settings.bybit.symbols:
        print(f"\n🔍 {symbol}:")
        
        features = feature_hub.get_latest_features(symbol)
        print(f"  Features count: {len(features) if features else 0}")
        
        if features:
            print(f"  Sample keys: {list(features.keys())[:5]}...")
            
            if len(features) > 10:
                try:
                    result = inference_engine.predict(features)
                    prediction = result["predictions"][0] if result["predictions"] else 0
                    confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                    
                    print(f"  ✅ Prediction: {prediction:.4f}")
                    print(f"  ✅ Confidence: {confidence:.2%}")
                    
                    if confidence > 0.6:
                        print(f"  🚨 HIGH CONFIDENCE! Sending notification...")
                        discord_notifier.send_trade_signal(
                            symbol=symbol,
                            side="BUY" if prediction > 0 else "SELL",
                            price=features.get("close", 50000),
                            confidence=confidence,
                            expected_pnl=prediction
                        )
                        print(f"  📲 Discord notification sent!")
                    else:
                        print(f"  ⚠️ Low confidence: {confidence:.2%}")
                        
                except Exception as e:
                    print(f"  ❌ Prediction error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  ⚠️ Not enough features")
        else:
            print(f"  ❌ No features found")
    
    await feature_hub.stop()
    print("\n✅ Check completed")

if __name__ == "__main__":
    asyncio.run(check_features())