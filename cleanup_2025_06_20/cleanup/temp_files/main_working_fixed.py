#!/usr/bin/env python3
"""
Fixed working unified main entry point for the integrated trading system.
"""

import asyncio
import signal
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.config import settings
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier
from src.ingestor.main import BybitIngestor
from src.feature_hub.main import FeatureHub
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.order_router.main import OrderRouter

logger = get_logger(__name__)


class FixedTradingSystem:
    """Fixed trading system with proper FeatureHub integration."""
    
    def __init__(self):
        self.running = False
        self.tasks = []
        
        # Initialize components
        self.ingestor = BybitIngestor()
        self.feature_hub = FeatureHub()
        self.order_router = OrderRouter()
        
        # Shared feature cache accessible from trading loop
        self.feature_cache = {}
        self.feature_cache_lock = asyncio.Lock()
        
        # Initialize inference engine
        inference_config = InferenceConfig(
            model_path=settings.model.model_path,
            enable_batching=True,
            confidence_threshold=0.6
        )
        self.inference_engine = InferenceEngine(inference_config)
    
    async def start(self):
        """Start all system components."""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("Starting Fixed Trading System")
        
        try:
            # Load ML model
            self.inference_engine.load_model()
            logger.info(f"Model loaded: {settings.model.model_path}")
            
            # Start ingestor
            await self.ingestor.start()
            logger.info("Ingestor started")
            
            # Start feature hub
            await self.feature_hub.start()
            logger.info("FeatureHub started")
            
            # Wait for data to accumulate
            await asyncio.sleep(10)
            logger.info("Initial data accumulation complete")
            
            # Start order router
            await self.order_router.start()
            logger.info("Order Router started")
            
            # Start background tasks
            self.running = True
            self.tasks = [
                asyncio.create_task(self._feature_cache_updater()),
                asyncio.create_task(self._trading_loop()),
                asyncio.create_task(self._health_check_loop())
            ]
            
            # Send Discord notification
            discord_notifier.send_system_status(
                "online",
                f"Fixed Trading System started with {settings.model.model_version} model"
            )
            
            logger.info("Fixed Trading System fully started")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop all system components."""
        if not self.running:
            return
        
        logger.info("Stopping Fixed Trading System")
        self.running = False
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop components
        await self.ingestor.stop()
        await self.feature_hub.stop()
        await self.order_router.stop()
        
        logger.info("Fixed Trading System stopped")
    
    async def _feature_cache_updater(self):
        """Background task to update feature cache from FeatureHub."""
        logger.info("Starting feature cache updater")
        
        while self.running:
            try:
                # Get features for all symbols
                async with self.feature_cache_lock:
                    for symbol in settings.bybit.symbols:
                        features = self.feature_hub.get_latest_features(symbol)
                        if features:
                            self.feature_cache[symbol] = features.copy()
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in feature cache updater: {e}")
                await asyncio.sleep(5)
    
    async def _trading_loop(self):
        """Main trading loop that processes features and makes predictions."""
        loop_count = 0
        logger.info("Starting trading loop")
        
        while self.running:
            try:
                loop_count += 1
                
                # Wait for Feature Hub to accumulate data
                if loop_count < 30:
                    await asyncio.sleep(1)
                    continue
                
                # Get latest features from cache
                async with self.feature_cache_lock:
                    cached_features = self.feature_cache.copy()
                
                for symbol in settings.bybit.symbols:
                    features = cached_features.get(symbol, {})
                    
                    if loop_count % 30 == 0:  # Log every 30 seconds
                        logger.info(f"Loop {loop_count}: {symbol} features={len(features)}")
                    
                    if features and len(features) > 10:  # Ensure we have enough features
                        try:
                            # Make prediction
                            result = self.inference_engine.predict(features)
                            
                            prediction = result["predictions"][0] if result["predictions"] else 0
                            confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                            
                            if loop_count % 30 == 0:  # Log every 30 seconds
                                logger.info(f"Prediction {symbol}: pred={prediction:.4f}, conf={confidence:.2%}")
                            
                            if confidence > 0.6:
                                logger.info(f"🚨 HIGH CONFIDENCE Signal for {symbol}: pred={prediction:.4f}, conf={confidence:.2%}")
                                
                                # Send Discord notification
                                discord_notifier.send_trade_signal(
                                    symbol=symbol,
                                    side="BUY" if prediction > 0 else "SELL",
                                    price=features.get("close", 50000),
                                    confidence=confidence,
                                    expected_pnl=prediction
                                )
                                
                        except Exception as e:
                            logger.error(f"Prediction error for {symbol}: {e}")
                    else:
                        if loop_count % 60 == 0:  # Log every 60 seconds for missing features
                            logger.warning(f"Insufficient features for {symbol}: count={len(features)}")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self):
        """Monitor system health."""
        while self.running:
            try:
                # Check component health and feature cache
                async with self.feature_cache_lock:
                    cached_symbols = len(self.feature_cache)
                    total_features = sum(len(features) for features in self.feature_cache.values())
                
                health_status = {
                    "ingestor": self.ingestor.running if hasattr(self.ingestor, 'running') else False,
                    "feature_hub": self.feature_hub.running if hasattr(self.feature_hub, 'running') else False,
                    "order_router": self.order_router.running if hasattr(self.order_router, 'running') else False,
                    "model": self.inference_engine.onnx_session is not None,
                    "feature_cache": cached_symbols > 0
                }
                
                logger.info(f"System health: {health_status}, cached_symbols: {cached_symbols}, total_features: {total_features}")
                
                unhealthy = [k for k, v in health_status.items() if not v]
                if unhealthy:
                    logger.warning(f"Unhealthy components: {unhealthy}")
                
                await asyncio.sleep(60)  # Check every 60 seconds
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(60)


async def main():
    """Run the fixed trading system."""
    system = FixedTradingSystem()
    
    # Setup signal handlers
    def signal_handler(sig):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(system.stop())
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler, sig)
    
    try:
        await system.start()
        
        # Keep running until stopped
        while system.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        await system.stop()


if __name__ == "__main__":
    logger.info("Starting Fixed Trading Bot")
    asyncio.run(main())