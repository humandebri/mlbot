#!/usr/bin/env python3
"""
Complete working trading system that properly handles async service initialization.
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


class CompleteTradingSystem:
    """Complete working trading system with proper service management."""
    
    def __init__(self):
        self.running = False
        self.tasks = []
        
        # Initialize components
        self.ingestor = BybitIngestor()
        self.feature_hub = FeatureHub()
        self.order_router = OrderRouter()
        
        # Initialize inference engine
        inference_config = InferenceConfig(
            model_path=settings.model.model_path,
            enable_batching=True,
            confidence_threshold=0.6
        )
        self.inference_engine = InferenceEngine(inference_config)
        
        # Service ready flags
        self.ingestor_ready = False
        self.feature_hub_ready = False
        self.order_router_ready = False
    
    async def start(self):
        """Start all system components with proper async management."""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("Starting Complete Trading System")
        
        try:
            # Load ML model first
            self.inference_engine.load_model()
            logger.info(f"Model loaded: {settings.model.model_path}")
            
            # Start components as background tasks
            self.running = True
            self.tasks = [
                asyncio.create_task(self._start_ingestor(), name="IngestorStarter"),
                asyncio.create_task(self._start_feature_hub(), name="FeatureHubStarter"),
                asyncio.create_task(self._start_order_router(), name="OrderRouterStarter"),
                asyncio.create_task(self._wait_and_start_trading(), name="TradingStarter"),
                asyncio.create_task(self._health_monitor(), name="HealthMonitor")
            ]
            
            logger.info("All service startup tasks created")
            
            # Send Discord notification
            discord_notifier.send_system_status(
                "online",
                f"Complete Trading System starting with model at {settings.model.model_path}"
            )
            
            logger.info("Complete Trading System startup initiated")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    async def _start_ingestor(self):
        """Start the ingestor service."""
        try:
            await self.ingestor.start()
            # Wait a bit for ingestor to be fully operational
            await asyncio.sleep(5)
            self.ingestor_ready = True
            logger.info("Ingestor service ready")
        except Exception as e:
            logger.error(f"Failed to start ingestor: {e}")
            self.ingestor_ready = False
    
    async def _start_feature_hub(self):
        """Start the feature hub service."""
        try:
            # Initialize Redis connection and feature engines first
            from src.common.database import get_redis_client, RedisStreams
            self.feature_hub.redis_client = await get_redis_client()
            self.feature_hub.redis_streams = RedisStreams(self.feature_hub.redis_client)
            
            await self.feature_hub._initialize_feature_engines()
            await self.feature_hub._setup_consumer_groups()
            
            # Set ready flag
            self.feature_hub_ready = True
            self.feature_hub.running = True
            logger.info("FeatureHub initialization ready")
            
            # Start background processing tasks
            tasks = [
                asyncio.create_task(self.feature_hub._process_market_data()),
                asyncio.create_task(self.feature_hub._publish_features()),
                asyncio.create_task(self.feature_hub._cleanup_cache()),
                asyncio.create_task(self.feature_hub._log_statistics()),
            ]
            
            logger.info("FeatureHub background tasks started")
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start feature hub: {e}")
    
    async def _start_order_router(self):
        """Start the order router service."""
        try:
            # Initialize order router components without blocking
            await self.order_router.start()
            self.order_router_ready = True
            logger.info("OrderRouter service ready")
        except Exception as e:
            logger.error(f"Failed to start order router: {e}")
    
    async def _wait_and_start_trading(self):
        """Wait for services to be ready and start trading loop."""
        logger.info("Waiting for services to be ready...")
        
        # Wait for all services to be ready
        max_wait = 120  # 2 minutes max wait
        wait_count = 0
        
        while not (self.ingestor_ready and self.feature_hub_ready and self.order_router_ready):
            if wait_count >= max_wait:
                logger.error("Services did not become ready within timeout")
                return
            
            await asyncio.sleep(1)
            wait_count += 1
            
            if wait_count % 10 == 0:
                logger.info(f"Waiting for services... Ingestor: {self.ingestor_ready}, "
                           f"FeatureHub: {self.feature_hub_ready}, OrderRouter: {self.order_router_ready}")
        
        logger.info("All services ready! Starting trading loop...")
        
        # Wait additional time for feature accumulation
        await asyncio.sleep(30)
        logger.info("Feature accumulation period complete. Starting predictions...")
        
        # Send ready notification
        discord_notifier.send_system_status(
            "trading",
            "All services ready - Trading predictions starting now!"
        )
        
        # Start trading loop
        await self._trading_loop()
    
    async def _trading_loop(self):
        """Main trading loop that processes features and makes predictions."""
        loop_count = 0
        predictions_made = 0
        high_confidence_signals = 0
        
        logger.info("Trading loop started")
        
        while self.running:
            try:
                loop_count += 1
                
                # Get latest features from FeatureHub
                for symbol in settings.bybit.symbols:
                    features = self.feature_hub.get_latest_features(symbol)
                    
                    if loop_count % 30 == 0:  # Log every 30 seconds
                        logger.info(f"Loop {loop_count}: {symbol} features={len(features) if features else 0}")
                    
                    if features and len(features) > 10:  # Ensure we have enough features
                        try:
                            # Make prediction
                            result = self.inference_engine.predict(features)
                            
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
                                discord_notifier.send_trade_signal(
                                    symbol=symbol,
                                    side="BUY" if prediction > 0 else "SELL",
                                    price=features.get("close", 50000),
                                    confidence=confidence,
                                    expected_pnl=prediction
                                )
                                
                                logger.info(f"📲 Discord notification sent for {symbol}")
                                
                        except Exception as e:
                            logger.error(f"Prediction error for {symbol}: {e}")
                    else:
                        if loop_count % 60 == 0:  # Log every 60 seconds for missing features
                            logger.warning(f"Insufficient features for {symbol}: count={len(features) if features else 0}")
                
                # Log trading statistics every 5 minutes
                if loop_count % 300 == 0:
                    logger.info(f"Trading stats: Predictions={predictions_made}, High confidence signals={high_confidence_signals}")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def _health_monitor(self):
        """Monitor system health continuously."""
        while self.running:
            try:
                # Check service status
                health_status = {
                    "ingestor": self.ingestor.running if hasattr(self.ingestor, 'running') else self.ingestor_ready,
                    "feature_hub": self.feature_hub.running if hasattr(self.feature_hub, 'running') else self.feature_hub_ready,
                    "order_router": self.order_router.running if hasattr(self.order_router, 'running') else self.order_router_ready,
                    "model": self.inference_engine.onnx_session is not None
                }
                
                # Count features for each symbol
                feature_counts = {}
                total_features = 0
                for symbol in settings.bybit.symbols:
                    features = self.feature_hub.get_latest_features(symbol) if self.feature_hub_ready else {}
                    count = len(features) if features else 0
                    feature_counts[symbol] = count
                    total_features += count
                
                logger.info(f"System health: {health_status}")
                logger.info(f"Feature counts: {feature_counts} (total: {total_features})")
                
                # Check for failed components
                unhealthy = [k for k, v in health_status.items() if not v]
                if unhealthy:
                    logger.warning(f"Unhealthy components: {unhealthy}")
                
                # Check for failed tasks
                failed_tasks = []
                for task in self.tasks:
                    if task.done() and not task.cancelled():
                        exception = task.exception()
                        if exception:
                            failed_tasks.append(f"{task.get_name()}: {exception}")
                
                if failed_tasks:
                    logger.error(f"Failed tasks: {failed_tasks}")
                
                await asyncio.sleep(60)  # Check every 60 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)
    
    async def stop(self):
        """Stop all system components gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping Complete Trading System")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete with timeout
        try:
            await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("Some tasks did not complete within timeout")
        
        # Stop components
        try:
            await self.ingestor.stop()
            await self.feature_hub.stop()
            await self.order_router.stop()
        except Exception as e:
            logger.error(f"Error stopping components: {e}")
        
        # Send final notification
        discord_notifier.send_system_status(
            "offline",
            "Complete Trading System stopped"
        )
        
        logger.info("Complete Trading System stopped")


async def main():
    """Run the complete trading system."""
    system = CompleteTradingSystem()
    
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
    logger.info("Starting Complete Trading Bot")
    asyncio.run(main())