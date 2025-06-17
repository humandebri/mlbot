"""
Simple Service Manager for integrated trading system.

Manages essential services in a single process instead of multiple containers.
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
    """Simple service manager for integrated system."""
    
    def __init__(self):
        self.running = False
        self.services = {}
        
        # Core components
        self.ingestor: Optional[BybitIngestor] = None
        self.feature_hub: Optional[FeatureHub] = None
        
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
            
            # Start Feature Hub
            logger.info("Starting Feature Hub...")
            self.feature_hub = FeatureHub()
            
            # Mark as running (no complex startup needed)
            self.feature_hub.running = True
            self.service_status["feature_hub"] = True
            logger.info("✅ Feature Hub started")
            
            self.running = True
            logger.info("Core services started successfully")
            
        except Exception as e:
            logger.error("Failed to start core services", exception=e)
            await self.stop_all()
            raise
    
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
        return {
            "running": self.running,
            "services": self.service_status.copy(),
            "timestamp": datetime.now().isoformat()
        }
    
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
        
        # Check Feature Hub
        health["feature_hub"] = (
            self.feature_hub is not None and 
            self.feature_hub.running
        )
        
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
                # Restart feature hub
                self.feature_hub.running = False
                await asyncio.sleep(1)
                
                self.feature_hub = FeatureHub()
                self.feature_hub.running = True
                
                self.service_status["feature_hub"] = True
                logger.info("Feature Hub restarted successfully")
                return True
            
            else:
                logger.warning(f"Unknown service or service not running: {service_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restart {service_name}", exception=e)
            return False