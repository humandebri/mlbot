#!/usr/bin/env python3
"""
Unified main entry point for the integrated trading system.
Starts all components in a single process.
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


class UnifiedTradingSystem:
    """Unified trading system that runs all components in one process."""
    
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
    
    async def start(self):
        """Start all system components."""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("Starting Unified Trading System")
        
        try:
            # Load ML model
            self.inference_engine.load_model()
            logger.info(f"Model loaded: {settings.model.model_path}")
            
            # Start components
            await self.ingestor.start()
            logger.info("Ingestor started")
            
            await self.feature_hub.start()
            logger.info("Feature Hub started")
            
            await self.order_router.start()
            logger.info("Order Router started")
            
            # Start background tasks
            self.running = True
            self.tasks = [
                asyncio.create_task(self._trading_loop()),
                asyncio.create_task(self._health_check_loop())
            ]
            
            # Send Discord notification
            discord_notifier.send_system_status(
                "online",
                f"Unified Trading System started with {settings.model.model_version} model"
            )
            
            logger.info("Unified Trading System fully started")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop all system components."""
        if not self.running:
            return
        
        logger.info("Stopping Unified Trading System")
        self.running = False
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop components
        await self.ingestor.stop()
        await self.feature_hub.stop()
        await self.order_router.stop()
        
        # Send Discord notification
        discord_notifier.send_system_status(
            "offline",
            "Unified Trading System stopped"
        )
        
        logger.info("Unified Trading System stopped")
    
    async def _trading_loop(self):
        """Main trading loop that processes features and makes predictions."""
        while self.running:
            try:
                # Get latest features from Feature Hub
                for symbol in settings.bybit.symbols:
                    features = await self.feature_hub.get_latest_features(symbol)
                    
                    if features and len(features) > 10:  # Ensure we have enough features
                        # Make prediction
                        result = self.inference_engine.predict(features)
                        
                        if result["confidence_scores"] and result["confidence_scores"][0] > 0.6:
                            prediction = result["predictions"][0]
                            confidence = result["confidence_scores"][0]
                            
                            logger.info(f"Signal for {symbol}: pred={prediction:.4f}, conf={confidence:.2%}")
                            
                            # Send Discord notification
                            discord_notifier.send_trade_signal(
                                symbol=symbol,
                                side="BUY" if prediction > 0 else "SELL",
                                price=features.get("close", 0),
                                confidence=confidence,
                                expected_pnl=prediction
                            )
                            
                            # Create order (if enabled)
                            if settings.trading.auto_start:
                                # OrderRouter will handle the actual order placement
                                pass
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self):
        """Monitor system health."""
        while self.running:
            try:
                # Check component health
                health_status = {
                    "ingestor": self.ingestor.running if hasattr(self.ingestor, 'running') else False,
                    "feature_hub": self.feature_hub.running if hasattr(self.feature_hub, 'running') else False,
                    "order_router": self.order_router.running if hasattr(self.order_router, 'running') else False,
                    "model": self.inference_engine.onnx_session is not None
                }
                
                unhealthy = [k for k, v in health_status.items() if not v]
                if unhealthy:
                    logger.warning(f"Unhealthy components: {unhealthy}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(30)


async def main():
    """Run the unified trading system."""
    system = UnifiedTradingSystem()
    
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
    logger.info("Starting Unified Trading Bot")
    asyncio.run(main())