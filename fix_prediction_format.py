#!/usr/bin/env python3
"""
Fix prediction format error by implementing proper feature adapter
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import numpy as np
import redis
from src.common.config import settings
from src.common.logging import get_logger
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.ml_pipeline.feature_adapter_44 import FeatureAdapter44
from src.common.discord_notifier import discord_notifier

logger = get_logger(__name__)

def get_latest_features_fixed(redis_client, symbol: str) -> dict:
    """Fixed version that handles both hash and stream types"""
    
    r = redis_client
    
    # Try different key patterns
    possible_keys = [
        f"features:{symbol}:latest",
        f"features:{symbol}",
        f"{symbol}:features",
        f"features_{symbol}"
    ]
    
    for key in possible_keys:
        if r.exists(key):
            key_type = r.type(key)
            
            if key_type == 'hash':
                return r.hgetall(key)
            
            elif key_type == 'stream':
                try:
                    entries = r.xrevrange(key, count=1)
                    if entries:
                        entry_id, fields = entries[0]
                        return {str(k): float(v) for k, v in dict(fields).items()}
                    else:
                        return {}
                except Exception as e:
                    logger.error(f"Error reading stream {key}: {e}")
                    return {}
    
    return {}

async def test_prediction_pipeline():
    """Test the complete prediction pipeline with proper format conversion"""
    
    logger.info("🧪 Testing prediction pipeline with format fix...")
    
    discord_notifier.send_system_status(
        "testing_prediction",
        "🧪 予測パイプラインのテストを開始します..."
    )
    
    try:
        # Initialize components
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Test Redis connection
        if not redis_client.ping():
            raise Exception("Redis connection failed")
        
        logger.info("✅ Redis connected")
        
        # Initialize inference engine
        inference_config = InferenceConfig(
            model_path=settings.model.model_path,
            confidence_threshold=0.5
        )
        inference_engine = InferenceEngine(inference_config)
        inference_engine.load_model()
        
        logger.info("✅ Model loaded")
        
        # Initialize feature adapter
        feature_adapter = FeatureAdapter44()
        logger.info("✅ Feature adapter initialized")
        
        # Test for each symbol
        test_results = {}
        
        for symbol in settings.bybit.symbols:
            logger.info(f"🧪 Testing {symbol}...")
            
            try:
                # Get features
                features = get_latest_features_fixed(redis_client, symbol)
                feature_count = len(features)
                
                logger.info(f"  📊 Raw features: {feature_count}")
                
                if feature_count < 10:
                    test_results[symbol] = {
                        "success": False,
                        "error": f"Insufficient features: {feature_count}"
                    }
                    continue
                
                # Convert to the right format for adapter
                # The adapter expects a dictionary of feature name -> float value
                logger.info(f"  🔄 Converting features...")
                
                # Adapt features to 44 dimensions
                adapted_features = feature_adapter.adapt(features)
                logger.info(f"  ✅ Adapted to {adapted_features.shape} shape")
                
                # Make prediction
                logger.info(f"  🎯 Making prediction...")
                
                # The inference engine expects a dict, but we need to convert the numpy array back
                # Let's check what the inference engine actually expects
                
                # Method 1: Try with the adapted numpy array directly
                try:
                    # Convert numpy array to dict with numeric indices
                    feature_dict = {str(i): float(adapted_features[i]) for i in range(len(adapted_features))}
                    
                    result = inference_engine.predict(feature_dict)
                    
                    prediction = result["predictions"][0] if result["predictions"] else 0
                    confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                    
                    test_results[symbol] = {
                        "success": True,
                        "prediction": float(prediction),
                        "confidence": float(confidence),
                        "feature_count": feature_count,
                        "adapted_shape": adapted_features.shape
                    }
                    
                    logger.info(f"  ✅ Prediction successful: pred={prediction:.4f}, conf={confidence:.2%}")
                    
                except Exception as e:
                    logger.error(f"  ❌ Method 1 failed: {e}")
                    
                    # Method 2: Try with different format
                    try:
                        # Maybe the inference engine expects specific feature names
                        # Let's try with a simple dict format
                        simple_features = {}
                        for i, value in enumerate(adapted_features):
                            simple_features[f"feature_{i}"] = float(value)
                        
                        result = inference_engine.predict(simple_features)
                        
                        prediction = result["predictions"][0] if result["predictions"] else 0
                        confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                        
                        test_results[symbol] = {
                            "success": True,
                            "prediction": float(prediction),
                            "confidence": float(confidence),
                            "method": "feature_names"
                        }
                        
                        logger.info(f"  ✅ Method 2 successful: pred={prediction:.4f}, conf={confidence:.2%}")
                        
                    except Exception as e2:
                        logger.error(f"  ❌ Method 2 also failed: {e2}")
                        test_results[symbol] = {
                            "success": False,
                            "error": f"Prediction failed: {e}, {e2}"
                        }
            
            except Exception as e:
                logger.error(f"❌ {symbol} test failed: {e}")
                test_results[symbol] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Generate test report
        successful_predictions = sum(1 for result in test_results.values() if result.get("success", False))
        
        report = f"""
🧪 **予測パイプラインテスト結果** 🧪

📊 **成功率**: {successful_predictions}/3

"""
        
        for symbol, result in test_results.items():
            if result.get("success"):
                pred = result.get("prediction", 0)
                conf = result.get("confidence", 0)
                report += f"✅ {symbol}: pred={pred:.4f}, conf={conf:.2%}\n"
            else:
                error = result.get("error", "Unknown error")
                report += f"❌ {symbol}: {error}\n"
        
        logger.info(report)
        
        # Send Discord notification
        if successful_predictions == 3:
            discord_notifier.send_system_status(
                "prediction_test_success",
                f"✅ **予測テスト完全成功** ✅\n\n" +
                f"成功率: {successful_predictions}/3\n" +
                "すべてのシンボルで予測生成成功\n\n" +
                "🚀 本番取引システム準備完了"
            )
        else:
            discord_notifier.send_error(
                "prediction_test",
                f"予測テスト部分成功: {successful_predictions}/3\n" +
                "詳細確認が必要"
            )
        
        return test_results
    
    except Exception as e:
        logger.error(f"❌ Test pipeline failed: {e}")
        discord_notifier.send_error("prediction_test", f"テスト失敗: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import asyncio
    logger.info("Starting prediction format fix test")
    result = asyncio.run(test_prediction_pipeline())
    logger.info(f"Test complete: {result}")