"""
Fixed version of SimpleServiceManager with proper FeatureHub initialization.
"""

import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger
from src.common.database import get_redis_client, RedisStreams
from src.ingestor.main import BybitIngestor
from src.feature_hub.main import FeatureHub
from src.order_router.main import OrderRouter
from src.common.account_monitor import AccountMonitor

logger = get_logger(__name__)


class SimpleServiceManager:
    """Fixed service manager that properly initializes all components."""
    
    def __init__(self):
        self.ingestor: Optional[BybitIngestor] = None
        self.feature_hub: Optional[FeatureHub] = None
        self.order_router: Optional[OrderRouter] = None
        self.account_monitor: Optional[AccountMonitor] = None
        self.tasks = []
        self.running = False
        
    async def start_ingestor(self) -> None:
        """Start the data ingestor service."""
        try:
            logger.info("Starting Ingestor...")
            self.ingestor = BybitIngestor()
            await self.ingestor.start()
            logger.info("Ingestor started successfully")
        except Exception as e:
            logger.error(f"Failed to start Ingestor: {e}")
            raise
    
    async def start_feature_hub(self) -> None:
        """Start the feature hub service with proper initialization."""
        try:
            logger.info("Starting FeatureHub...")
            self.feature_hub = FeatureHub()
            
            # CRITICAL: Initialize Redis connection
            self.feature_hub.redis_client = await get_redis_client()
            self.feature_hub.redis_streams = RedisStreams(self.feature_hub.redis_client)
            
            # CRITICAL: Initialize all feature engines
            await self.feature_hub._initialize_feature_engines()
            logger.info("Feature engines initialized")
            
            # CRITICAL: Setup consumer groups
            await self.feature_hub._setup_consumer_groups()
            logger.info("Consumer groups set up")
            
            # Set running flag
            self.feature_hub.running = True
            
            # CRITICAL: Start all background tasks
            feature_hub_tasks = [
                asyncio.create_task(self.feature_hub._process_market_data(), name="FeatureHub-ProcessData"),
                asyncio.create_task(self.feature_hub._publish_features(), name="FeatureHub-PublishFeatures"),
                asyncio.create_task(self.feature_hub._cleanup_cache(), name="FeatureHub-CleanupCache"),
                asyncio.create_task(self.feature_hub._log_statistics(), name="FeatureHub-LogStats")
            ]
            self.tasks.extend(feature_hub_tasks)
            
            logger.info("FeatureHub started successfully with all background tasks")
            
        except Exception as e:
            logger.error(f"Failed to start FeatureHub: {e}")
            raise
    
    async def start_order_router(self) -> None:
        """Start the order router service."""
        try:
            logger.info("Starting OrderRouter...")
            self.order_router = OrderRouter()
            
            # Initialize components
            await self.order_router.initialize()
            
            # Set running flag
            self.order_router.running = True
            
            logger.info("OrderRouter started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start OrderRouter: {e}")
            raise
    
    async def start_account_monitor(self) -> None:
        """Start the account monitor service."""
        try:
            logger.info("Starting AccountMonitor...")
            self.account_monitor = AccountMonitor()
            
            # Start monitoring loop
            monitor_task = asyncio.create_task(
                self.account_monitor._monitor_loop(), 
                name="AccountMonitor-Loop"
            )
            self.tasks.append(monitor_task)
            
            logger.info("AccountMonitor started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start AccountMonitor: {e}")
            raise
    
    async def start_all(self) -> None:
        """Start all services in proper order."""
        self.running = True
        
        # Start services
        await self.start_ingestor()
        await asyncio.sleep(2)  # Let ingestor establish connections
        
        await self.start_feature_hub()
        await asyncio.sleep(5)  # Let feature hub accumulate initial data
        
        await self.start_order_router()
        await self.start_account_monitor()
        
        logger.info("All services started successfully")
    
    async def stop_all(self) -> None:
        """Stop all services gracefully."""
        logger.info("Stopping all services...")
        self.running = False
        
        # Cancel background tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop services
        if self.ingestor:
            await self.ingestor.stop()
        
        if self.feature_hub:
            self.feature_hub.running = False
            
        if self.order_router:
            self.order_router.running = False
            
        if self.account_monitor:
            self.account_monitor._running = False
        
        logger.info("All services stopped")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        status = {
            "ingestor": {
                "running": self.ingestor.running if self.ingestor else False,
                "connected": self.ingestor.ws_manager.connected if self.ingestor and hasattr(self.ingestor, 'ws_manager') else False
            },
            "feature_hub": {
                "running": self.feature_hub.running if self.feature_hub else False,
                "feature_counts": {}
            },
            "order_router": {
                "running": self.order_router.running if self.order_router else False,
                "initialized": self.order_router.initialized if self.order_router else False
            },
            "account_monitor": {
                "running": self.account_monitor._running if self.account_monitor else False,
                "balance": self.account_monitor.current_balance.total_equity if self.account_monitor and self.account_monitor.current_balance else 0
            }
        }
        
        # Get feature counts if available
        if self.feature_hub:
            try:
                from src.common.config import settings
                for symbol in settings.bybit.symbols:
                    features = self.feature_hub.get_latest_features(symbol)
                    status["feature_hub"]["feature_counts"][symbol] = len(features) if features else 0
            except:
                pass
        
        # Count running tasks
        status["background_tasks"] = {
            "total": len(self.tasks),
            "running": len([t for t in self.tasks if not t.done()])
        }
        
        return status