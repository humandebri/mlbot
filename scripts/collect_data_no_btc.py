#!/usr/bin/env python3
"""
Script to collect market data without BTCUSDT (ETHUSDT and ICPUSDT only).
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


async def collect_data(duration_hours: int = 24, testnet: bool = True):
    """
    Collect market data for specified duration without BTCUSDT.
    
    Args:
        duration_hours: How many hours to collect data
        testnet: Whether to use testnet
    """
    # Override symbols to exclude BTCUSDT
    settings.bybit.symbols = ["ETHUSDT", "ICPUSDT"]
    
    # Set environment
    if testnet:
        os.environ["USE_TESTNET"] = "true"
        settings.bybit.testnet = True
        logger.info("Using TESTNET for data collection")
    else:
        settings.bybit.testnet = False
        logger.info("Using MAINNET for data collection")
    
    logger.info(f"Collecting data for symbols: {settings.bybit.symbols}")
    
    # Initialize storage manager
    storage_manager = DuckDBManager()
    
    # Start time tracking
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)
    
    logger.info(f"Starting data collection for {duration_hours} hours")
    logger.info(f"Collection will end at: {end_time}")
    
    # Start services
    tasks = []
    
    # Start ingestor
    logger.info("Starting WebSocket Ingestor...")
    ingestor_task = asyncio.create_task(run_ingestor())
    tasks.append(ingestor_task)
    
    # Wait for ingestor to stabilize
    await asyncio.sleep(5)
    
    # Start feature hub
    logger.info("Starting Feature Hub...")
    feature_hub_task = asyncio.create_task(run_feature_hub())
    tasks.append(feature_hub_task)
    
    # Monitor and wait for completion
    try:
        # Wait until end time or tasks complete
        while datetime.now() < end_time:
            # Check if any task completed
            done = [task for task in tasks if task.done()]
            if done:
                for task in done:
                    if task.exception():
                        logger.error(f"Task failed: {task.exception()}")
                        raise task.exception()
                break
            
            await asyncio.sleep(60)  # Check every minute
        
        logger.info("Data collection completed successfully")
        
    except Exception as e:
        logger.error("Error during data collection", exception=e)
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


async def run_ingestor():
    """Run the WebSocket ingestor."""
    from src.ingestor.main import BybitIngestor
    
    ingestor = BybitIngestor()
    try:
        await ingestor.start()
    finally:
        await ingestor.stop()


async def run_feature_hub():
    """Run the feature hub."""
    from src.feature_hub.main import FeatureHub
    
    feature_hub = FeatureHub()
    try:
        await feature_hub.start()
    finally:
        await feature_hub.stop()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect market data without BTCUSDT")
    parser.add_argument(
        "--duration", 
        type=int, 
        default=24, 
        help="Collection duration in hours"
    )
    parser.add_argument(
        "--testnet", 
        action="store_true", 
        help="Use testnet (default: True)"
    )
    parser.add_argument(
        "--mainnet", 
        action="store_true", 
        help="Use mainnet"
    )
    
    args = parser.parse_args()
    
    # Determine environment
    testnet = not args.mainnet if args.mainnet else True
    
    await collect_data(duration_hours=args.duration, testnet=testnet)


if __name__ == "__main__":
    asyncio.run(main())