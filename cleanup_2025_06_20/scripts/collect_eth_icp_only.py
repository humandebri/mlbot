#!/usr/bin/env python3
"""
Focused data collection for ETHUSDT and ICPUSDT only.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging
from src.common.config import settings
from src.ingestor.main import start_ingestor
from src.feature_hub.main import start_feature_hub
from src.storage.duckdb_manager import DuckDBManager

setup_logging()
logger = get_logger(__name__)

# Focus on just ETHUSDT and ICPUSDT
TARGET_SYMBOLS = ["ETHUSDT", "ICPUSDT"]


class FocusedDataCollector:
    """Focused data collector for ETHUSDT and ICPUSDT only."""
    
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        self.symbols = TARGET_SYMBOLS
        
    async def collect_data(self, duration_hours: int = 24):
        """
        Collect market data for ETHUSDT and ICPUSDT only.
        
        Args:
            duration_hours: How many hours to collect data
        """
        # Apply settings
        self._apply_settings()
        
        env_name = "TESTNET" if self.testnet else "MAINNET"
        logger.info(f"Starting {env_name} focused data collection")
        logger.info(f"Target symbols: {self.symbols}")
        
        # Initialize storage
        db_suffix = "_testnet_focused" if self.testnet else "_production_focused"
        db_path = f"data/market_data{db_suffix}.duckdb"
        storage_manager = DuckDBManager(db_path=db_path)
        
        # Start time tracking
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        logger.info(f"Collection duration: {duration_hours} hours")
        logger.info(f"Will end at: {end_time}")
        
        # Start services
        tasks = []
        
        try:
            # Start ingestor
            logger.info("Starting WebSocket Ingestor...")
            ingestor_task = asyncio.create_task(self._run_ingestor())
            tasks.append(ingestor_task)
            
            # Wait for ingestor to stabilize
            await asyncio.sleep(8)
            
            # Start feature hub
            logger.info("Starting Feature Hub...")
            feature_hub_task = asyncio.create_task(self._run_feature_hub())
            tasks.append(feature_hub_task)
            
            # Monitor and wait for completion
            while datetime.now() < end_time:
                # Check if any task completed
                done = [task for task in tasks if task.done()]
                if done:
                    for task in done:
                        if task.exception():
                            logger.error(f"Task failed: {task.exception()}")
                            raise task.exception()
                    break
                
                # Log progress every 30 minutes
                elapsed = datetime.now() - start_time
                remaining = end_time - datetime.now()
                
                if int(elapsed.total_seconds()) % 1800 == 0 and int(elapsed.total_seconds()) > 0:
                    logger.info(
                        f"Collection progress - Elapsed: {str(elapsed).split('.')[0]}, "
                        f"Remaining: {str(remaining).split('.')[0]}"
                    )
                
                await asyncio.sleep(60)  # Check every minute
            
            logger.info("Focused data collection completed successfully")
            
        except Exception as e:
            logger.error("Error during focused data collection", exception=e)
            raise
        
        finally:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
    
    def _apply_settings(self):
        """Apply collection settings."""
        # Set symbols to only ETHUSDT and ICPUSDT
        settings.bybit.symbols = self.symbols
        
        # Set environment
        settings.bybit.testnet = self.testnet
        
        # Conservative settings for reliability
        if self.testnet:
            settings.bybit.requests_per_second = 5
            settings.bybit.requests_per_minute = 200
        else:
            # Extra conservative for production
            settings.bybit.requests_per_second = 3
            settings.bybit.requests_per_minute = 120
        
        settings.bybit.max_reconnect_attempts = 5
        settings.bybit.reconnect_delay = 8.0
        settings.bybit.ping_interval = 25
        
        logger.info(f"Applied settings: {len(self.symbols)} symbols, "
                   f"{settings.bybit.requests_per_second}req/s, "
                   f"testnet={self.testnet}")
    
    async def _run_ingestor(self):
        """Run the WebSocket ingestor."""
        from src.ingestor.main import BybitIngestor
        
        ingestor = BybitIngestor()
        try:
            await ingestor.start()
        finally:
            await ingestor.stop()
    
    async def _run_feature_hub(self):
        """Run the feature hub."""
        from src.feature_hub.main import FeatureHub
        
        feature_hub = FeatureHub()
        try:
            await feature_hub.start()
        finally:
            await feature_hub.stop()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Focused data collection for ETHUSDT and ICPUSDT")
    parser.add_argument(
        "--duration", 
        type=int, 
        default=24, 
        help="Collection duration in hours"
    )
    parser.add_argument(
        "--testnet", 
        action="store_true", 
        help="Use testnet (default)"
    )
    parser.add_argument(
        "--mainnet", 
        action="store_true", 
        help="Use mainnet"
    )
    
    args = parser.parse_args()
    
    # Determine environment
    testnet = not args.mainnet
    
    if args.mainnet:
        print("⚠️  Using mainnet for ETHUSDT and ICPUSDT data collection")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            return
    
    collector = FocusedDataCollector(testnet=testnet)
    await collector.collect_data(duration_hours=args.duration)


if __name__ == "__main__":
    asyncio.run(main())