#!/usr/bin/env python3
"""
Production trading system with fixed feature access
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import asyncio
import signal
import redis
import numpy as np
from pathlib import Path
from src.common.config import settings
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.ml_pipeline.feature_adapter_44 import FeatureAdapter44

logger = get_logger(__name__)

# Import the fixed feature access function with numpy string parsing
import re

def parse_numpy_string(value_str):
    """Parse numpy string representation back to float"""
    
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    if not isinstance(value_str, str):
        return float(value_str)
    
    # Remove numpy type wrapper
    patterns = [
        r'np\.float\d*\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)',
        r'numpy\.float\d*\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)',
        r'float\d*\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, value_str)
        if match:
            return float(match.group(1))
    
    try:
        return float(value_str)
    except ValueError:
        number_pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
        match = re.search(number_pattern, value_str)
        if match:
            return float(match.group())
        else:
            raise ValueError(f"Cannot parse numeric value from: {value_str}")

def get_latest_features_fixed(redis_client, symbol: str) -> dict:
    """Fixed version that handles both hash and stream types with numpy parsing"""
    
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
                raw_data = r.hgetall(key)
                parsed_data = {}
                for k, v in raw_data.items():
                    try:
                        parsed_data[str(k)] = parse_numpy_string(v)
                    except ValueError:
                        continue
                return parsed_data
            
            elif key_type == 'stream':
                try:
                    entries = r.xrevrange(key, count=1)
                    if entries:
                        entry_id, fields = entries[0]
                        parsed_data = {}
                        
                        for k, v in dict(fields).items():
                            try:
                                parsed_data[str(k)] = parse_numpy_string(v)
                            except ValueError:
                                continue
                        
                        return parsed_data
                    else:
                        return {}
                except Exception as e:
                    logger.error(f"Error reading stream {key}: {e}")
                    return {}
            
            else:
                logger.warning(f"Unsupported key type {key_type} for key {key}")
                return {}
    
    return {}

class ProductionTradingSystem:
    """Production-ready trading system with fixed feature access"""
    
    def __init__(self):
        self.running = False
        self.redis_client = None
        
        # Initialize inference engine
        inference_config = InferenceConfig(
            model_path=settings.model.model_path,
            confidence_threshold=0.6
        )
        self.inference_engine = InferenceEngine(inference_config)
        
        # Initialize feature adapter for 44-dimension model
        self.feature_adapter = FeatureAdapter44()
        
        # Statistics
        self.stats = {
            "predictions_made": 0,
            "high_confidence_signals": 0,
            "discord_notifications_sent": 0,
            "errors": 0
        }
    
    async def start(self):
        """Start the production trading system"""
        if self.running:
            return
        
        logger.info("🚀 Starting Production Trading System")
        
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Test Redis connection
            ping_result = self.redis_client.ping()
            if not ping_result:
                raise Exception("Redis connection failed")
            
            logger.info("✅ Redis connected")
            
            # Load model
            self.inference_engine.load_model()
            logger.info("✅ Model loaded")
            
            # Send startup notification
            discord_notifier.send_system_status(
                "production_start",
                "🚀 **本番取引システム開始** 🚀\n\n" +
                "• 特徴量アクセス: ✅ 修正済み\n" +
                "• モデル: v3.1_improved (AUC 0.838)\n" +
                "• 対象: BTCUSDT, ETHUSDT, ICPUSDT\n" +
                "• 閾値: 60%信頼度\n\n" +
                "📊 取引予測を開始します..."
            )
            
            # Start trading loop
            self.running = True
            await self._trading_loop()
            
        except Exception as e:
            logger.error(f"Failed to start production system: {e}")
            discord_notifier.send_error("production_system", f"起動失敗: {e}")
            raise
    
    async def stop(self):
        """Stop the trading system"""
        logger.info("🛑 Stopping Production Trading System")
        self.running = False
        
        # Send final statistics
        discord_notifier.send_system_status(
            "production_stop",
            f"🛑 **本番システム停止** 🛑\n\n" +
            f"実行統計:\n" +
            f"• 予測回数: {self.stats['predictions_made']}\n" +
            f"• 高信頼度シグナル: {self.stats['high_confidence_signals']}\n" +
            f"• Discord通知: {self.stats['discord_notifications_sent']}\n" +
            f"• エラー: {self.stats['errors']}"
        )
    
    async def _trading_loop(self):
        """Main trading loop with fixed feature access"""
        
        loop_count = 0
        logger.info("🎯 Production trading loop started")
        
        while self.running:
            try:
                loop_count += 1
                
                # Process each symbol
                for symbol in settings.bybit.symbols:
                    await self._process_symbol(symbol, loop_count)
                
                # Log statistics every 5 minutes
                if loop_count % 300 == 0:
                    await self._log_statistics()
                
                # Health check every 10 minutes
                if loop_count % 600 == 0:
                    await self._health_check()
                
                await asyncio.sleep(1)  # 1 second intervals
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self.stats["errors"] += 1
                
                # Send error notification for critical errors
                if self.stats["errors"] % 10 == 0:  # Every 10th error
                    discord_notifier.send_error(
                        "trading_loop",
                        f"取引ループで{self.stats['errors']}件のエラー発生"
                    )
                
                await asyncio.sleep(5)  # Wait before retry
    
    async def _process_symbol(self, symbol: str, loop_count: int):
        """Process trading for a single symbol"""
        
        try:
            # Get features using fixed access method
            features = get_latest_features_fixed(self.redis_client, symbol)
            
            feature_count = len(features)
            
            # Log feature status every 30 seconds
            if loop_count % 30 == 0:
                logger.info(f"📊 {symbol}: {feature_count} features available")
            
            if feature_count < 10:
                if loop_count % 60 == 0:  # Log every minute
                    logger.warning(f"⚠️ {symbol}: Insufficient features ({feature_count})")
                return
            
            # Convert features using feature adapter (156 -> 44 dimensions)
            try:
                # Features are already parsed as numeric values from numpy strings
                if len(features) < 10:
                    logger.warning(f"⚠️ {symbol}: Too few raw features ({len(features)})")
                    return
                
                # Adapt features to 44 dimensions for v3.1_improved model
                adapted_features = self.feature_adapter.adapt(features)
                logger.debug(f"📊 {symbol}: Adapted to {adapted_features.shape} shape")
                
                # Adapted features is already a numpy array, use it directly
                if len(adapted_features) != 44:
                    logger.warning(f"⚠️ {symbol}: Incorrect adapted feature count ({len(adapted_features)})")
                    return
                
                # Reshape for inference (model expects 2D array: [1, 44])
                model_features = adapted_features.reshape(1, -1).astype(np.float32)
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Feature adaptation error: {e}")
                return
            
            # Make prediction
            try:
                result = self.inference_engine.predict(model_features)
                
                prediction = result["predictions"][0] if result["predictions"] else 0
                confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                
                self.stats["predictions_made"] += 1
                
                # Log prediction every 30 seconds
                if loop_count % 30 == 0:
                    logger.info(f"🎯 {symbol}: pred={prediction:.4f}, conf={confidence:.2%}")
                
                # Check for high confidence signals
                if confidence > 0.6:
                    await self._handle_high_confidence_signal(symbol, prediction, confidence, features)
                
                elif confidence > 0.4:  # Medium confidence logging
                    if loop_count % 60 == 0:  # Every minute
                        logger.info(f"📈 {symbol}: Medium confidence pred={prediction:.4f}, conf={confidence:.2%}")
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Prediction error: {e}")
                self.stats["errors"] += 1
                
        except Exception as e:
            logger.error(f"❌ {symbol}: Processing error: {e}")
            self.stats["errors"] += 1
    
    async def _handle_high_confidence_signal(self, symbol: str, prediction: float, confidence: float, features: dict):
        """Handle high confidence trading signal"""
        
        self.stats["high_confidence_signals"] += 1
        
        logger.info(f"🚨 HIGH CONFIDENCE #{self.stats['high_confidence_signals']} - {symbol}: "
                   f"pred={prediction:.4f}, conf={confidence:.2%}")
        
        # Determine trade direction
        side = "BUY" if prediction > 0 else "SELL"
        
        # Get current price from original features (before adaptation)
        current_price = features.get("close", features.get("last_price", 50000))
        if isinstance(current_price, str):
            try:
                current_price = float(current_price)
            except ValueError:
                current_price = 50000  # Fallback
        
        # Send Discord notification
        try:
            success = discord_notifier.send_trade_signal(
                symbol=symbol,
                side=side,
                price=current_price,
                confidence=confidence,
                expected_pnl=prediction
            )
            
            if success:
                self.stats["discord_notifications_sent"] += 1
                logger.info(f"📲 Discord notification sent for {symbol}")
            else:
                logger.warning(f"⚠️ Discord notification failed for {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Discord notification error for {symbol}: {e}")
    
    async def _log_statistics(self):
        """Log system statistics"""
        
        logger.info(f"📊 Production Statistics:")
        logger.info(f"  Predictions: {self.stats['predictions_made']}")
        logger.info(f"  High confidence signals: {self.stats['high_confidence_signals']}")
        logger.info(f"  Discord notifications: {self.stats['discord_notifications_sent']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        
        # Send periodic update to Discord
        discord_notifier.send_system_status(
            "production_update",
            f"📊 **本番システム稼働中** 📊\n\n" +
            f"予測回数: {self.stats['predictions_made']}\n" +
            f"高信頼度シグナル: {self.stats['high_confidence_signals']}\n" +
            f"Discord通知: {self.stats['discord_notifications_sent']}\n" +
            f"エラー: {self.stats['errors']}\n\n" +
            "🟢 システム正常運用中"
        )
    
    async def _health_check(self):
        """Perform system health check"""
        
        try:
            # Check Redis connection
            redis_ok = self.redis_client.ping()
            
            # Check feature availability
            feature_counts = {}
            for symbol in settings.bybit.symbols:
                features = get_latest_features_fixed(self.redis_client, symbol)
                feature_counts[symbol] = len(features)
            
            total_features = sum(feature_counts.values())
            
            # Check model
            model_ok = self.inference_engine.onnx_session is not None
            
            logger.info(f"🏥 Health Check:")
            logger.info(f"  Redis: {'✅' if redis_ok else '❌'}")
            logger.info(f"  Model: {'✅' if model_ok else '❌'}")
            logger.info(f"  Features: {total_features} total")
            
            # Send health report if there are issues
            if not redis_ok or not model_ok or total_features < 300:
                discord_notifier.send_error(
                    "health_check",
                    f"ヘルスチェック警告:\n" +
                    f"Redis: {'✅' if redis_ok else '❌'}\n" +
                    f"Model: {'✅' if model_ok else '❌'}\n" +
                    f"Features: {total_features}個"
                )
                
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            discord_notifier.send_error("health_check", f"ヘルスチェック失敗: {e}")

async def main():
    """Run the production trading system"""
    
    system = ProductionTradingSystem()
    
    # Setup signal handlers
    def signal_handler(sig):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(system.stop())
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler, sig)
    
    try:
        await system.start()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        await system.stop()

if __name__ == "__main__":
    logger.info("🚀 Starting Production Trading Bot")
    asyncio.run(main())