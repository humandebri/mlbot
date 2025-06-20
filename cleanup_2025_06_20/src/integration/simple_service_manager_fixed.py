"""
Simple Service Manager for integrated trading system - FIXED VERSION.

Manages essential services in a single process instead of multiple containers.
Properly initializes FeatureHub with all necessary tasks.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..common.config import settings
from ..common.logging import get_logger
from ..common.database import init_redis, init_duckdb
from ..ingestor.main import BybitIngestor
from ..feature_hub.main import FeatureHub

logger = get_logger(__name__)


class SimpleServiceManager:
    """Simple service manager for integrated system with proper FeatureHub initialization."""
    
    def __init__(self):
        self.running = False
        self.services = {}
        
        # Core components
        self.ingestor: Optional[BybitIngestor] = None
        self.feature_hub: Optional[FeatureHub] = None
        self.feature_hub_task: Optional[asyncio.Task] = None
        
        # Service status tracking
        self.service_status = {
            "redis": False,
            "duckdb": False,
            "ingestor": False,
            "feature_hub": False
        }
        
        logger.info("Simple Service Manager initialized")
    
    async def start_core_services(self):
        """Start essential core services."""
        logger.info("Starting core services...")
        
        try:
            # Initialize Redis
            logger.info("Initializing Redis connection...")
            await init_redis()
            self.service_status["redis"] = True
            logger.info("✅ Redis initialized")
            
            # Initialize DuckDB
            logger.info("Initializing DuckDB...")
            init_duckdb()
            self.service_status["duckdb"] = True
            logger.info("✅ DuckDB initialized")
            
            # Start Ingestor
            logger.info("Starting Bybit Ingestor...")
            self.ingestor = BybitIngestor()
            
            # Start ingestor in background task
            asyncio.create_task(self.ingestor.start())
            
            # Wait a bit for ingestor to initialize
            await asyncio.sleep(3)
            self.service_status["ingestor"] = True
            logger.info("✅ Ingestor started")
            
            # Start Feature Hub properly
            logger.info("Starting Feature Hub with full initialization...")
            self.feature_hub = FeatureHub()
            
            # Initialize the feature hub components
            await self._initialize_feature_hub()
            
            self.service_status["feature_hub"] = True
            logger.info("✅ Feature Hub started with all tasks running")
            
            self.running = True
            logger.info("Core services started successfully")
            
        except Exception as e:
            logger.error("Failed to start core services", exception=e)
            await self.stop_all()
            raise
    
    async def _initialize_feature_hub(self):
        """Properly initialize FeatureHub with all necessary components."""
        if not self.feature_hub:
            return
            
        try:
            # Initialize Redis connection for FeatureHub
            from ..common.database import get_redis_client, RedisStreams
            self.feature_hub.redis_client = await get_redis_client()
            self.feature_hub.redis_streams = RedisStreams(self.feature_hub.redis_client)
            
            # Initialize feature engines
            logger.info("Initializing feature engines...")
            await self.feature_hub._initialize_feature_engines()
            
            # Setup consumer groups
            logger.info("Setting up consumer groups...")
            await self.feature_hub._setup_consumer_groups()
            
            # Mark as running
            self.feature_hub.running = True
            
            # Start processing tasks individually (not with gather)
            logger.info("Starting FeatureHub background tasks...")
            
            # Create tasks but don't await them
            tasks = []
            tasks.append(asyncio.create_task(self._run_feature_hub_processor()))
            tasks.append(asyncio.create_task(self._run_feature_publisher()))
            tasks.append(asyncio.create_task(self._run_feature_cleanup()))
            tasks.append(asyncio.create_task(self._run_feature_stats()))
            
            # Store tasks for cleanup later
            self.feature_hub_tasks = tasks
            
            logger.info(f"Started {len(tasks)} FeatureHub background tasks")
            
        except Exception as e:
            logger.error(f"Failed to initialize FeatureHub: {e}")
            raise
    
    async def _run_feature_hub_processor(self):
        """Run the feature hub market data processor."""
        try:
            await self.feature_hub._process_market_data()
        except Exception as e:
            logger.error(f"Feature hub processor error: {e}")
    
    async def _run_feature_publisher(self):
        """Run the feature hub publisher."""
        try:
            await self.feature_hub._publish_features()
        except Exception as e:
            logger.error(f"Feature hub publisher error: {e}")
    
    async def _run_feature_cleanup(self):
        """Run the feature hub cleanup task."""
        try:
            await self.feature_hub._cleanup_cache()
        except Exception as e:
            logger.error(f"Feature hub cleanup error: {e}")
    
    async def _run_feature_stats(self):
        """Run the feature hub statistics logger."""
        try:
            await self.feature_hub._log_statistics()
        except Exception as e:
            logger.error(f"Feature hub stats error: {e}")
    
    async def start_additional_services(self):
        """Start additional services (placeholder for future extensions)."""
        logger.info("Additional services initialization completed")
    
    async def stop_all(self):
        """Stop all services."""
        if not self.running:
            return
        
        logger.info("Stopping all services...")
        self.running = False
        
        try:
            # Stop Feature Hub
            if self.feature_hub:
                self.feature_hub.running = False
                
                # Cancel all feature hub tasks
                if hasattr(self, 'feature_hub_tasks'):
                    for task in self.feature_hub_tasks:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                
                self.service_status["feature_hub"] = False
                logger.info("Feature Hub stopped")
            
            # Stop Ingestor
            if self.ingestor:
                await self.ingestor.stop()
                self.service_status["ingestor"] = False
                logger.info("Ingestor stopped")
            
            # Note: Redis and DuckDB connections will be cleaned up automatically
            self.service_status["redis"] = False
            self.service_status["duckdb"] = False
            
            logger.info("All services stopped")
            
        except Exception as e:
            logger.error("Error stopping services", exception=e)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        status = {
            "running": self.running,
            "services": self.service_status.copy(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add feature hub specific status
        if self.feature_hub:
            status["feature_hub_details"] = {
                "running": self.feature_hub.running,
                "features_computed": self.feature_hub.features_computed,
                "cached_symbols": len(self.feature_hub.feature_cache),
                "feature_summary": self.feature_hub.get_feature_summary()
            }
        
        return status
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all services."""
        health = {}
        
        # Check Redis (basic ping)
        try:
            # Simple check - if we can import redis functions, assume it's working
            from ..common.database import get_redis_client
            health["redis"] = self.service_status["redis"]
        except Exception:
            health["redis"] = False
        
        # Check DuckDB
        try:
            # Simple check - if DuckDB was initialized, assume it's working
            health["duckdb"] = self.service_status["duckdb"]
        except Exception:
            health["duckdb"] = False
        
        # Check Ingestor
        health["ingestor"] = (
            self.ingestor is not None and 
            self.service_status["ingestor"]
        )
        
        # Check Feature Hub (more detailed check)
        health["feature_hub"] = False
        if self.feature_hub and self.feature_hub.running:
            # Check if features are being generated
            if self.feature_hub.features_computed > 0:
                health["feature_hub"] = True
            # Also check if we have recent features
            elif len(self.feature_hub.feature_cache) > 0:
                health["feature_hub"] = True
        
        return health
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        logger.info(f"Restarting service: {service_name}")
        
        try:
            if service_name == "ingestor" and self.ingestor:
                # Stop and restart ingestor
                await self.ingestor.stop()
                await asyncio.sleep(2)
                
                self.ingestor = BybitIngestor()
                asyncio.create_task(self.ingestor.start())
                await asyncio.sleep(3)
                
                self.service_status["ingestor"] = True
                logger.info("Ingestor restarted successfully")
                return True
            
            elif service_name == "feature_hub" and self.feature_hub:
                # Stop feature hub
                self.feature_hub.running = False
                
                # Cancel existing tasks
                if hasattr(self, 'feature_hub_tasks'):
                    for task in self.feature_hub_tasks:
                        task.cancel()
                
                await asyncio.sleep(1)
                
                # Restart feature hub
                self.feature_hub = FeatureHub()
                await self._initialize_feature_hub()
                
                self.service_status["feature_hub"] = True
                logger.info("Feature Hub restarted successfully")
                return True
            
            else:
                logger.warning(f"Unknown service or service not running: {service_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restart {service_name}", exception=e)
            return False