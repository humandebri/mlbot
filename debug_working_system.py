#!/usr/bin/env python3
"""
Debug script to check if the working system is actually making predictions
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import asyncio
from src.feature_hub.main import FeatureHub
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.common.config import settings
from src.common.discord_notifier import discord_notifier

async def debug_working_system():
    """Check if the working system components are functional"""
    print("🔍 Debugging working system components...")
    
    # Test FeatureHub directly (without starting - just get cached features)
    feature_hub = FeatureHub()
    
    # Test inference engine
    inference_config = InferenceConfig(
        model_path=settings.model.model_path,
        confidence_threshold=0.5  # Lower threshold for testing
    )
    inference_engine = InferenceEngine(inference_config)
    inference_engine.load_model()
    
    print("✅ Components initialized")
    
    # Check features and make predictions
    predictions_made = 0
    high_confidence_signals = 0
    
    for symbol in settings.bybit.symbols:
        print(f"\n🔍 {symbol}:")
        
        # Get features (this should work if FeatureHub cache is populated)
        features = feature_hub.get_latest_features(symbol)
        print(f"  Features: {len(features) if features else 0}")
        
        if features and len(features) > 10:
            try:
                result = inference_engine.predict(features)
                prediction = result["predictions"][0] if result["predictions"] else 0
                confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                
                predictions_made += 1
                print(f"  📊 Prediction: {prediction:.4f}, Confidence: {confidence:.2%}")
                
                # Test at different thresholds
                for threshold in [0.3, 0.5, 0.6, 0.7]:
                    if confidence > threshold:
                        print(f"  ✅ Passes {threshold:.0%} threshold")
                        if threshold >= 0.6:
                            high_confidence_signals += 1
                            print(f"  🚨 HIGH CONFIDENCE ({confidence:.2%}) - Should trigger Discord!")
                            
                            # Send test notification
                            success = discord_notifier.send_trade_signal(
                                symbol=symbol,
                                side="BUY" if prediction > 0 else "SELL", 
                                price=features.get("close", 50000),
                                confidence=confidence,
                                expected_pnl=prediction
                            )
                            print(f"  📲 Discord notification: {'✅ Success' if success else '❌ Failed'}")
                        break
                else:
                    print(f"  ⚠️ Below all thresholds")
                    
            except Exception as e:
                print(f"  ❌ Prediction error: {e}")
        else:
            print(f"  ❌ Insufficient features")
    
    print(f"\n📈 Summary:")
    print(f"  Predictions made: {predictions_made}")
    print(f"  High confidence signals: {high_confidence_signals}")
    print(f"  System status: {'🟢 Working' if predictions_made > 0 else '🔴 Not working'}")
    
    if high_confidence_signals > 0:
        print(f"  💬 Discord notifications should have been sent!")
    else:
        print(f"  📱 No high confidence signals - Discord quiet is expected")

if __name__ == "__main__":
    asyncio.run(debug_working_system())