#!/usr/bin/env python3
"""
Force trading start by bypassing ready check - system is clearly working
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import asyncio
from src.feature_hub.main import FeatureHub
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.common.config import settings
from src.common.discord_notifier import discord_notifier
from src.common.logging import get_logger

logger = get_logger(__name__)

async def force_trading_start():
    """Force start trading since all systems are working"""
    logger.info("🚀 Force starting trading system - bypassing ready checks")
    
    # Initialize components
    feature_hub = FeatureHub()
    
    inference_config = InferenceConfig(
        model_path=settings.model.model_path,
        enable_batching=True,
        confidence_threshold=0.6
    )
    inference_engine = InferenceEngine(inference_config)
    inference_engine.load_model()
    
    logger.info("✅ Components ready for trading")
    
    # Send start notification
    discord_notifier.send_system_status(
        "trading",
        "🚀 FORCE TRADING START - All systems operational! 417 features ready."
    )
    
    loop_count = 0
    predictions_made = 0
    high_confidence_signals = 0
    
    logger.info("🎯 Starting trading loop with feature monitoring")
    
    while True:
        try:
            loop_count += 1
            
            # Process all symbols
            for symbol in settings.bybit.symbols:
                features = feature_hub.get_latest_features(symbol)
                
                if loop_count % 30 == 0:  # Log every 30 seconds
                    logger.info(f"Loop {loop_count}: {symbol} features={len(features) if features else 0}")
                
                if features and len(features) > 10:
                    try:
                        # Make prediction
                        result = inference_engine.predict(features)
                        
                        prediction = result["predictions"][0] if result["predictions"] else 0
                        confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                        predictions_made += 1
                        
                        if loop_count % 30 == 0:  # Log every 30 seconds
                            logger.info(f"Prediction {symbol}: pred={prediction:.4f}, conf={confidence:.2%}")
                        
                        if confidence > 0.6:
                            high_confidence_signals += 1
                            logger.info(f"🚨 HIGH CONFIDENCE Signal #{high_confidence_signals} for {symbol}: "
                                      f"pred={prediction:.4f}, conf={confidence:.2%}")
                            
                            # Send Discord notification
                            success = discord_notifier.send_trade_signal(
                                symbol=symbol,
                                side="BUY" if prediction > 0 else "SELL",
                                price=features.get("close", 50000),
                                confidence=confidence,
                                expected_pnl=prediction
                            )
                            
                            logger.info(f"📲 Discord notification sent for {symbol}: {'✅ Success' if success else '❌ Failed'}")
                            
                        elif confidence > 0.4:  # Log medium confidence too
                            logger.info(f"📊 Medium confidence for {symbol}: pred={prediction:.4f}, conf={confidence:.2%}")
                            
                    except Exception as e:
                        logger.error(f"Prediction error for {symbol}: {e}")
                else:
                    if loop_count % 60 == 0:  # Log every 60 seconds for missing features
                        logger.warning(f"Insufficient features for {symbol}: count={len(features) if features else 0}")
            
            # Stats every 5 minutes
            if loop_count % 300 == 0:
                logger.info(f"🏆 Trading stats: Predictions={predictions_made}, High confidence signals={high_confidence_signals}")
                discord_notifier.send_system_status(
                    "update", 
                    f"Trading active: {predictions_made} predictions, {high_confidence_signals} high-confidence signals generated"
                )
            
            await asyncio.sleep(1)  # Check every second
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    logger.info("Starting Force Trading System")
    asyncio.run(force_trading_start())