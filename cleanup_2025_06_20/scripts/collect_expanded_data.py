#!/usr/bin/env python3
"""
Expanded data collection with more symbols for better ML training data diversity.
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

# Expanded symbol list for better ML training diversity
EXPANDED_SYMBOLS = [
    # Current symbols
    "ETHUSDT", "ICPUSDT",
    
    # Major cryptocurrencies
    "BNBUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT",
    "AVAXUSDT", "LINKUSDT", "MATICUSDT", "LTCUSDT", "BCHUSDT",
    
    # Additional popular pairs for diversity
    "ATOMUSDT", "FILUSDT", "ETCUSDT", "XLMUSDT", "VETUSDT"
]

# Conservative symbol list for production (to avoid rate limits)
PRODUCTION_SYMBOLS = [
    "ETHUSDT", "ICPUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
    "XRPUSDT", "DOTUSDT", "AVAXUSDT"
]


class ExpandedDataCollector:
    """Enhanced data collector with more symbols and safety features."""
    
    def __init__(self, testnet: bool = True, production_mode: bool = False):
        self.testnet = testnet
        self.production_mode = production_mode
        self.symbols = PRODUCTION_SYMBOLS if production_mode else EXPANDED_SYMBOLS
        
    async def collect_data(self, duration_hours: int = 24):
        """
        Collect market data with expanded symbol coverage.
        
        Args:
            duration_hours: How many hours to collect data
        """
        # Apply environment and safety settings
        self._apply_settings()
        
        logger.info(f"Starting {'TESTNET' if self.testnet else 'MAINNET'} expanded data collection")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Total symbols: {len(self.symbols)}")
        
        if self.production_mode:
            logger.warning("🚨 PRODUCTION MODE: Using conservative settings")
        
        # Initialize storage
        db_suffix = "_expanded" if not self.production_mode else "_production_expanded"
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
                
                # Log progress
                elapsed = datetime.now() - start_time
                remaining = end_time - datetime.now()
                if elapsed.total_seconds() % 1800 == 0:  # Every 30 minutes
                    logger.info(
                        f"Collection progress - Elapsed: {elapsed}, Remaining: {remaining}"
                    )
                
                await asyncio.sleep(60)  # Check every minute
            
            logger.info("Expanded data collection completed successfully")
            
        except Exception as e:
            logger.error("Error during expanded data collection", exception=e)
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
        """Apply collection settings based on environment."""
        # Set symbols
        settings.bybit.symbols = self.symbols
        
        # Set environment
        settings.bybit.testnet = self.testnet
        
        if self.production_mode:
            # Extra conservative settings for production
            settings.bybit.requests_per_second = 2  # Very conservative
            settings.bybit.requests_per_minute = 100
            settings.bybit.max_reconnect_attempts = 3
            settings.bybit.reconnect_delay = 15.0
            settings.bybit.ping_interval = 45
        else:
            # Testnet can be more aggressive
            settings.bybit.requests_per_second = 5
            settings.bybit.requests_per_minute = 200
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
    parser = argparse.ArgumentParser(description="Expanded data collection")
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
    parser.add_argument(
        "--production", 
        action="store_true", 
        help="Use production-safe conservative settings"
    )
    
    args = parser.parse_args()
    
    # Determine environment
    testnet = not args.mainnet
    
    if args.mainnet and not args.production:
        print("⚠️  Warning: Using mainnet without --production flag")
        print("   Consider using --production for safer rate limits")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            return
    
    collector = ExpandedDataCollector(
        testnet=testnet, 
        production_mode=args.production
    )
    
    await collector.collect_data(duration_hours=args.duration)


if __name__ == "__main__":
    asyncio.run(main())