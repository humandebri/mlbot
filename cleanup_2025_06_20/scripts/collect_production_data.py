#!/usr/bin/env python3
"""
Production data collection script with enhanced safety measures.

This script collects data from Bybit production environment with:
- Conservative rate limiting to avoid bans
- Error handling and automatic recovery
- Monitoring and alerting
- Safe connection management
"""

import asyncio
import argparse
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging
from src.common.config import settings
from src.ingestor.main import BybitIngestor
from src.feature_hub.main import FeatureHub
from src.storage.duckdb_manager import DuckDBManager

setup_logging()
logger = get_logger(__name__)


class ProductionDataCollector:
    """
    Production-safe data collector with enhanced monitoring and safety measures.
    """
    
    def __init__(self, duration_hours: int = 24):
        self.duration_hours = duration_hours
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=duration_hours)
        self.error_count = 0
        self.max_errors = 10
        self.last_error_time = 0
        self.error_cooldown = 300  # 5 minutes
        
    async def collect_data(self):
        """
        Main data collection loop with safety measures.
        """
        logger.info("Starting PRODUCTION data collection")
        logger.info(f"Duration: {self.duration_hours} hours")
        logger.info(f"Will end at: {self.end_time}")
        logger.warning("⚠️  PRODUCTION MODE - Using conservative rate limits")
        
        # Override settings for production safety
        self._apply_production_safety_settings()
        
        try:
            # Initialize storage (production database with separate file)
            from src.storage.duckdb_manager import DuckDBManager
            storage_manager = DuckDBManager(db_path="data/production_market_data.duckdb")
            
            # Start services with safety monitoring
            await self._run_with_monitoring()
            
        except Exception as e:
            logger.error("Critical error in production data collection", exception=e)
            raise
    
    def _apply_production_safety_settings(self):
        """Apply conservative settings for production environment."""
        # Override for production safety
        settings.bybit.testnet = False
        settings.bybit.symbols = ["ETHUSDT", "ICPUSDT"]  # Remove BTCUSDT
        settings.bybit.requests_per_second = 3  # Very conservative
        settings.bybit.requests_per_minute = 150  # Half of normal limit
        settings.bybit.max_reconnect_attempts = 5
        settings.bybit.reconnect_delay = 10.0  # Longer delays
        settings.bybit.ping_interval = 30  # Less frequent pings
        
        logger.info("Applied production safety settings")
        logger.info(f"Rate limits: {settings.bybit.requests_per_second}/s, {settings.bybit.requests_per_minute}/min")
    
    async def _run_with_monitoring(self):
        """Run data collection with monitoring and error handling."""
        ingestor = BybitIngestor()
        feature_hub = FeatureHub()
        
        try:
            # Start ingestor
            logger.info("Starting production ingestor...")
            ingestor_task = asyncio.create_task(ingestor.start())
            
            # Wait a bit for ingestor to stabilize
            await asyncio.sleep(10)
            
            # Start feature hub
            logger.info("Starting production feature hub...")
            feature_task = asyncio.create_task(feature_hub.start())
            
            # Monitor collection
            monitor_task = asyncio.create_task(self._monitor_collection())
            
            # Wait until collection time is up or error
            while datetime.now() < self.end_time:
                # Check if any task failed
                done_tasks = [t for t in [ingestor_task, feature_task, monitor_task] if t.done()]
                if done_tasks:
                    for task in done_tasks:
                        if task.exception():
                            logger.error(f"Task failed: {task.exception()}")
                            self._handle_error(task.exception())
                            if self.error_count > self.max_errors:
                                logger.error("Too many errors, stopping collection")
                                break
                
                await asyncio.sleep(60)  # Check every minute
            
            logger.info("Collection duration reached, stopping...")
            
        except Exception as e:
            logger.error("Error in monitoring loop", exception=e)
            self._handle_error(e)
        
        finally:
            # Cleanup
            await ingestor.stop()
            await feature_hub.stop()
            
            # Cancel remaining tasks
            for task in [ingestor_task, feature_task, monitor_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
    
    async def _monitor_collection(self):
        """Monitor collection health and performance."""
        while datetime.now() < self.end_time:
            try:
                # Log status every 10 minutes
                elapsed = datetime.now() - self.start_time
                remaining = self.end_time - datetime.now()
                
                logger.info(
                    "Production collection status",
                    elapsed=str(elapsed).split('.')[0],
                    remaining=str(remaining).split('.')[0],
                    errors=self.error_count
                )
                
                # Check Redis health
                from src.common.database import get_redis_client
                redis_client = await get_redis_client()
                await redis_client.ping()
                
                await asyncio.sleep(600)  # 10 minutes
                
            except Exception as e:
                logger.warning("Monitor check failed", exception=e)
                await asyncio.sleep(60)
    
    def _handle_error(self, error: Exception):
        """Handle errors with backoff and counting."""
        current_time = time.time()
        
        # Check if we're in error cooldown
        if current_time - self.last_error_time < self.error_cooldown:
            self.error_count += 1
        else:
            self.error_count = 1  # Reset if enough time passed
        
        self.last_error_time = current_time
        
        logger.error(
            "Production collection error",
            error_count=self.error_count,
            max_errors=self.max_errors,
            exception=error
        )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Production data collection")
    parser.add_argument(
        "--duration", 
        type=int, 
        default=24, 
        help="Collection duration in hours"
    )
    
    args = parser.parse_args()
    
    # Safety check
    logger.warning("⚠️  Starting PRODUCTION data collection")
    logger.warning("⚠️  This will use real Bybit production APIs")
    
    collector = ProductionDataCollector(duration_hours=args.duration)
    await collector.collect_data()


if __name__ == "__main__":
    asyncio.run(main())