#!/usr/bin/env python3
"""
Comprehensive investigation of feature generation and access
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import asyncio
import redis
import json
from src.common.config import settings
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier
from src.feature_hub.main import FeatureHub
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig

logger = get_logger(__name__)

async def investigate_feature_system():
    """Comprehensive investigation of feature generation and access"""
    
    logger.info("🔍 Starting comprehensive feature investigation...")
    
    # Send investigation start notification
    discord_notifier.send_system_status(
        "investigation",
        "🔍 特徴量システムの詳細調査を開始します..."
    )
    
    investigation_results = {
        "redis_status": {},
        "feature_hub_status": {},
        "feature_data": {},
        "prediction_test": {},
        "issues": []
    }
    
    # 1. Redis Investigation
    logger.info("📊 Step 1: Redis investigation")
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Check Redis connection
        redis_ping = r.ping()
        investigation_results["redis_status"]["connection"] = redis_ping
        logger.info(f"Redis connection: {'✅ Connected' if redis_ping else '❌ Failed'}")
        
        # Check existing keys
        feature_keys = r.keys("features:*")
        stream_keys = r.keys("market_data:*")
        
        investigation_results["redis_status"]["feature_keys"] = len(feature_keys)
        investigation_results["redis_status"]["stream_keys"] = len(stream_keys)
        
        logger.info(f"Redis keys - Features: {len(feature_keys)}, Streams: {len(stream_keys)}")
        
        # Check specific feature data
        for symbol in settings.bybit.symbols:
            key = f"features:{symbol}:latest"
            if r.exists(key):
                key_type = r.type(key)
                if key_type == 'hash':
                    feature_count = r.hlen(key)
                    investigation_results["feature_data"][symbol] = {
                        "exists": True,
                        "type": key_type,
                        "count": feature_count
                    }
                    logger.info(f"✅ {symbol}: {feature_count} features in Redis")
                    
                    # Sample some feature names
                    sample_fields = r.hkeys(key)[:10]
                    investigation_results["feature_data"][symbol]["sample_features"] = sample_fields
                    
                else:
                    investigation_results["feature_data"][symbol] = {
                        "exists": True,
                        "type": key_type,
                        "count": 0,
                        "issue": f"Wrong data type: {key_type}"
                    }
                    investigation_results["issues"].append(f"{symbol}: Wrong Redis data type: {key_type}")
            else:
                investigation_results["feature_data"][symbol] = {
                    "exists": False,
                    "count": 0
                }
                investigation_results["issues"].append(f"{symbol}: No Redis data found")
    
    except Exception as e:
        logger.error(f"❌ Redis investigation failed: {e}")
        investigation_results["redis_status"]["error"] = str(e)
        investigation_results["issues"].append(f"Redis error: {e}")
    
    # 2. FeatureHub Investigation
    logger.info("🏭 Step 2: FeatureHub investigation")
    try:
        feature_hub = FeatureHub()
        
        # Test feature access
        for symbol in settings.bybit.symbols:
            features = feature_hub.get_latest_features(symbol)
            feature_count = len(features) if features else 0
            
            investigation_results["feature_hub_status"][symbol] = {
                "accessible": feature_count > 0,
                "count": feature_count
            }
            
            if feature_count > 0:
                # Sample feature names and values
                sample_features = dict(list(features.items())[:5])
                investigation_results["feature_hub_status"][symbol]["sample"] = sample_features
                logger.info(f"✅ FeatureHub {symbol}: {feature_count} features accessible")
            else:
                logger.warning(f"⚠️ FeatureHub {symbol}: No features accessible")
                investigation_results["issues"].append(f"FeatureHub {symbol}: No features accessible")
    
    except Exception as e:
        logger.error(f"❌ FeatureHub investigation failed: {e}")
        investigation_results["feature_hub_status"]["error"] = str(e)
        investigation_results["issues"].append(f"FeatureHub error: {e}")
    
    # 3. Prediction Test
    logger.info("🎯 Step 3: Prediction test")
    try:
        inference_config = InferenceConfig(
            model_path=settings.model.model_path,
            confidence_threshold=0.5
        )
        inference_engine = InferenceEngine(inference_config)
        inference_engine.load_model()
        
        logger.info("✅ Model loaded for prediction test")
        
        predictions_made = 0
        high_confidence_predictions = 0
        
        for symbol in settings.bybit.symbols:
            features = feature_hub.get_latest_features(symbol)
            
            if features and len(features) > 10:
                try:
                    result = inference_engine.predict(features)
                    prediction = result["predictions"][0] if result["predictions"] else 0
                    confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                    
                    predictions_made += 1
                    if confidence > 0.6:
                        high_confidence_predictions += 1
                    
                    investigation_results["prediction_test"][symbol] = {
                        "success": True,
                        "prediction": float(prediction),
                        "confidence": float(confidence),
                        "high_confidence": confidence > 0.6
                    }
                    
                    logger.info(f"🎯 {symbol} prediction: {prediction:.4f}, confidence: {confidence:.2%}")
                    
                except Exception as e:
                    logger.error(f"❌ Prediction failed for {symbol}: {e}")
                    investigation_results["prediction_test"][symbol] = {
                        "success": False,
                        "error": str(e)
                    }
                    investigation_results["issues"].append(f"Prediction error {symbol}: {e}")
            else:
                investigation_results["prediction_test"][symbol] = {
                    "success": False,
                    "error": "Insufficient features"
                }
                investigation_results["issues"].append(f"Prediction {symbol}: Insufficient features")
        
        investigation_results["prediction_test"]["summary"] = {
            "total_predictions": predictions_made,
            "high_confidence": high_confidence_predictions
        }
        
    except Exception as e:
        logger.error(f"❌ Prediction test failed: {e}")
        investigation_results["prediction_test"]["error"] = str(e)
        investigation_results["issues"].append(f"Prediction test error: {e}")
    
    # 4. Generate Investigation Report
    logger.info("📋 Step 4: Generating investigation report")
    
    total_issues = len(investigation_results["issues"])
    redis_features = sum(
        data.get("count", 0) for data in investigation_results["feature_data"].values()
    )
    featurehub_features = sum(
        data.get("count", 0) for data in investigation_results["feature_hub_status"].values() 
        if isinstance(data, dict)
    )
    
    predictions_made = investigation_results["prediction_test"].get("summary", {}).get("total_predictions", 0)
    high_confidence = investigation_results["prediction_test"].get("summary", {}).get("high_confidence", 0)
    
    report = f"""
🔍 **特徴量システム調査結果** 🔍

📊 **Redis状況**:
• 接続: {'✅ 正常' if investigation_results["redis_status"].get("connection") else '❌ 失敗'}
• 特徴量キー数: {investigation_results["redis_status"].get("feature_keys", 0)}
• ストリームキー数: {investigation_results["redis_status"].get("stream_keys", 0)}
• Redis内特徴量総数: {redis_features}

🏭 **FeatureHub状況**:
• アクセス可能特徴量: {featurehub_features}個
• 各シンボル状況:
"""
    
    for symbol in settings.bybit.symbols:
        fh_status = investigation_results["feature_hub_status"].get(symbol, {})
        accessible = fh_status.get("accessible", False)
        count = fh_status.get("count", 0)
        report += f"  • {symbol}: {'✅' if accessible else '❌'} {count}個\n"
    
    report += f"""
🎯 **予測テスト結果**:
• 成功した予測: {predictions_made}/3
• 高信頼度予測: {high_confidence}個

🚨 **発見された問題**: {total_issues}件
"""
    
    if investigation_results["issues"]:
        for issue in investigation_results["issues"][:5]:  # Top 5 issues
            report += f"• {issue}\n"
    else:
        report += "なし - すべて正常 ✅"
    
    logger.info(report)
    
    # Send comprehensive Discord notification
    if total_issues == 0 and predictions_made == 3:
        discord_notifier.send_system_status(
            "investigation_success",
            f"✅ **特徴量調査完了 - 問題なし** ✅\n\n" +
            f"Redis特徴量: {redis_features}個\n" +
            f"FeatureHub: {featurehub_features}個\n" +
            f"予測成功: {predictions_made}/3\n" +
            f"高信頼度: {high_confidence}個\n\n" +
            "🚀 次: 統合取引システム構築"
        )
    else:
        discord_notifier.send_error(
            "feature_investigation",
            f"特徴量調査で{total_issues}件の問題発見:\n" + 
            "\n".join(investigation_results["issues"][:3])
        )
    
    return investigation_results

if __name__ == "__main__":
    logger.info("Starting feature investigation")
    result = asyncio.run(investigate_feature_system())
    logger.info("Investigation complete")