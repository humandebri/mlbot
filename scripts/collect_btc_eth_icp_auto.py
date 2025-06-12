#!/usr/bin/env python3
"""
Auto-run version of BTC/ETH/ICP collector (no confirmation needed).
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

# Target symbols for ML training
TARGET_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ICPUSDT"]


class OptimizedDataCollector:
    """Optimized data collector for BTC, ETH, and ICP."""
    
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        self.symbols = TARGET_SYMBOLS
        
    async def collect_data(self, duration_hours: int = 24):
        """
        Collect market data for BTCUSDT, ETHUSDT, and ICPUSDT.
        
        Args:
            duration_hours: How many hours to collect data
        """
        # Apply settings
        self._apply_settings()
        
        env_name = "TESTNET" if self.testnet else "MAINNET"
        logger.info(f"Starting {env_name} optimized data collection")
        logger.info(f"Target symbols: {self.symbols}")
        logger.info("✅ Including BTCUSDT for feature engineering")
        
        # Initialize storage
        db_suffix = "_testnet_optimized" if self.testnet else "_production_optimized"
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
            await asyncio.sleep(10)
            
            # Start feature hub
            logger.info("Starting Feature Hub...")
            feature_hub_task = asyncio.create_task(self._run_feature_hub())
            tasks.append(feature_hub_task)
            
            # Start monitoring task
            monitor_task = asyncio.create_task(self._monitor_collection(start_time, end_time))
            tasks.append(monitor_task)
            
            # Wait for completion or error
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Check for errors
            for task in done:
                if task.exception():
                    logger.error(f"Task failed: {task.exception()}")
                    raise task.exception()
            
            logger.info("Optimized data collection completed successfully")
            
        except Exception as e:
            logger.error("Error during optimized data collection", exception=e)
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
    
    async def _monitor_collection(self, start_time: datetime, end_time: datetime):
        """Monitor collection progress and health."""
        while datetime.now() < end_time:
            elapsed = datetime.now() - start_time
            remaining = end_time - datetime.now()
            
            # Log progress every 30 minutes
            if int(elapsed.total_seconds()) % 1800 == 0 and int(elapsed.total_seconds()) > 0:
                progress_pct = (elapsed.total_seconds() / (end_time - start_time).total_seconds()) * 100
                logger.info(
                    f"📊 Collection progress: {progress_pct:.1f}% - "
                    f"Elapsed: {str(elapsed).split('.')[0]}, "
                    f"Remaining: {str(remaining).split('.')[0]}"
                )
            
            await asyncio.sleep(60)  # Check every minute
    
    def _apply_settings(self):
        """Apply collection settings."""
        # Set symbols to BTC, ETH, ICP
        settings.bybit.symbols = self.symbols
        
        # Set environment
        settings.bybit.testnet = self.testnet
        
        # Balanced settings for 3 symbols
        if self.testnet:
            settings.bybit.requests_per_second = 6
            settings.bybit.requests_per_minute = 250
            settings.bybit.ping_interval = 20
        else:
            # Conservative for production with 3 symbols
            settings.bybit.requests_per_second = 4
            settings.bybit.requests_per_minute = 180
            settings.bybit.ping_interval = 30
        
        settings.bybit.max_reconnect_attempts = 5
        settings.bybit.reconnect_delay = 8.0
        
        logger.info(f"Applied settings: {len(self.symbols)} symbols, "
                   f"{settings.bybit.requests_per_second}req/s, "
                   f"{settings.bybit.requests_per_minute}req/min, "
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
    parser = argparse.ArgumentParser(description="Optimized data collection for BTCUSDT, ETHUSDT, and ICPUSDT (auto-run)")
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
        print("⚠️  Using mainnet for BTC, ETH, ICP data collection (auto-run mode)")
        print("   This will collect BTCUSDT data for feature engineering")
    
    collector = OptimizedDataCollector(testnet=testnet)
    await collector.collect_data(duration_hours=args.duration)


if __name__ == "__main__":
    asyncio.run(main())