#!/usr/bin/env python3
"""
Fix excessive signal generation and Discord spam.

Issues:
1. Model always returns same prediction (0.12) with 100% confidence
2. No cooldown between signals
3. Discord rate limiting from spam
"""

import asyncio
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
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
    """Trading system with fixed signal generation and cooldown."""
    
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
            confidence_threshold=0.8  # Increased from 0.6
        )
        self.inference_engine = InferenceEngine(inference_config)
        
        # Service ready flags
        self.ingestor_ready = False
        self.feature_hub_ready = False
        self.order_router_ready = False
        
        # Signal tracking for cooldown
        self.last_signal_time = defaultdict(lambda: datetime.min)
        self.signal_cooldown = timedelta(minutes=15)  # 15 minute cooldown per symbol
        self.confidence_threshold = 0.8  # Only signal on high confidence
        self.min_prediction_change = 0.02  # 2% minimum prediction change
        self.last_predictions = {}
    
    async def start(self):
        """Start all system components with proper async management."""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("Starting Fixed Trading System")
        
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
                f"Trading System starting with signal cooldown ({self.signal_cooldown.total_seconds()/60:.0f} min) "
                f"and confidence threshold ({self.confidence_threshold:.0%})"
            )
            
            logger.info("Fixed Trading System startup initiated")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    async def _start_ingestor(self):
        """Start the ingestor service."""
        try:
            await self.ingestor.start()
            await asyncio.sleep(5)
            self.ingestor_ready = True
            logger.info("Ingestor service ready")
        except Exception as e:
            logger.error(f"Failed to start ingestor: {e}")
            self.ingestor_ready = False
    
    async def _start_feature_hub(self):
        """Start the feature hub service."""
        try:
            from src.common.database import get_redis_client, RedisStreams
            self.feature_hub.redis_client = await get_redis_client()
            self.feature_hub.redis_streams = RedisStreams(self.feature_hub.redis_client)
            
            await self.feature_hub._initialize_feature_engines()
            await self.feature_hub._setup_consumer_groups()
            
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
            "All services ready - Trading predictions starting with signal filtering"
        )
        
        # Start trading loop
        await self._trading_loop()
    
    async def _trading_loop(self):
        """Main trading loop with signal filtering and cooldown."""
        loop_count = 0
        predictions_made = 0
        high_confidence_signals = 0
        signals_sent = 0
        
        logger.info("Trading loop started with signal filtering")
        
        while self.running:
            try:
                loop_count += 1
                
                # Get latest features from FeatureHub
                for symbol in settings.bybit.symbols:
                    features = self.feature_hub.get_latest_features(symbol)
                    
                    if loop_count % 300 == 0:  # Log every 5 minutes
                        logger.info(f"Loop {loop_count}: {symbol} features={len(features) if features else 0}")
                    
                    if features and len(features) > 10:  # Ensure we have enough features
                        try:
                            # Make prediction
                            result = self.inference_engine.predict(features)
                            
                            if result and "predictions" in result and len(result["predictions"]) > 0:
                                prediction = float(result["predictions"][0])
                                confidence = float(result["confidence_scores"][0]) if "confidence_scores" in result else 0.5
                            else:
                                prediction = 0.0
                                confidence = 0.0
                            
                            predictions_made += 1
                            
                            # Check if this is a significant signal
                            should_signal = self._should_generate_signal(symbol, prediction, confidence)
                            
                            if should_signal:
                                high_confidence_signals += 1
                                signals_sent += 1
                                
                                logger.info(f"🎯 VALID Signal #{high_confidence_signals} for {symbol}: "
                                          f"pred={prediction:.4f}, conf={confidence:.2%}")
                                
                                # Send Discord notification
                                current_price = features.get("close", 0)
                                if current_price > 0:  # Only send if we have valid price
                                    discord_notifier.send_trade_signal(
                                        symbol=symbol,
                                        side="BUY" if prediction > 0 else "SELL",
                                        price=current_price,
                                        confidence=confidence,
                                        expected_pnl=prediction
                                    )
                                    
                                    # Update tracking
                                    self.last_signal_time[symbol] = datetime.now()
                                    self.last_predictions[symbol] = prediction
                                    
                                    logger.info(f"✅ Signal sent for {symbol} (#{signals_sent} total)")
                                
                        except Exception as e:
                            logger.error(f"Prediction error for {symbol}: {e}")
                    else:
                        if loop_count % 600 == 0:  # Log every 10 minutes
                            logger.warning(f"Insufficient features for {symbol}: count={len(features) if features else 0}")
                
                # Log trading statistics every 5 minutes
                if loop_count % 300 == 0:
                    logger.info(f"Trading stats: Predictions={predictions_made}, "
                              f"High confidence={high_confidence_signals}, Signals sent={signals_sent}")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    def _should_generate_signal(self, symbol: str, prediction: float, confidence: float) -> bool:
        """Determine if a signal should be generated based on filtering criteria."""
        now = datetime.now()
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False
        
        # Check cooldown
        time_since_last = now - self.last_signal_time[symbol]
        if time_since_last < self.signal_cooldown:
            return False
        
        # Check prediction significance
        if abs(prediction) < self.min_prediction_change:
            return False
        
        # Check if prediction has changed significantly from last signal
        if symbol in self.last_predictions:
            pred_change = abs(prediction - self.last_predictions[symbol])
            if pred_change < self.min_prediction_change:
                return False
        
        # All checks passed
        return True
    
    async def _health_monitor(self):
        """Monitor system health continuously."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every 60 seconds
                
                health_status = {
                    "ingestor": self.ingestor.running if hasattr(self.ingestor, 'running') else self.ingestor_ready,
                    "feature_hub": self.feature_hub.running if hasattr(self.feature_hub, 'running') else self.feature_hub_ready,
                    "order_router": self.order_router.running if hasattr(self.order_router, 'running') else self.order_router_ready,
                    "model": self.inference_engine.onnx_session is not None
                }
                
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
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)
    
    async def stop(self):
        """Stop all system components gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping Fixed Trading System")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
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
            "Fixed Trading System stopped"
        )
        
        logger.info("Fixed Trading System stopped")


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