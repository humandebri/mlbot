#!/usr/bin/env python3
"""
Script to collect market data for ML training.

This script runs only the data ingestion components without trading.
Usage:
    python scripts/collect_data.py [--testnet] [--duration HOURS]
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
from src.ingestor.main import start_ingestor
from src.feature_hub.main import start_feature_hub
from src.storage.duckdb_manager import DuckDBManager

setup_logging()
logger = get_logger(__name__)


async def collect_data(duration_hours: int = 24, testnet: bool = True):
    """
    Collect market data for specified duration.
    
    Args:
        duration_hours: How many hours to collect data
        testnet: Whether to use testnet
    """
    # Set environment
    if testnet:
        os.environ["USE_TESTNET"] = "true"
        logger.info("Using TESTNET for data collection")
    else:
        logger.info("Using MAINNET for data collection")
    
    # Initialize storage manager
    storage_manager = DuckDBManager()
    
    # Start time tracking
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)
    
    logger.info(f"Starting data collection for {duration_hours} hours")
    logger.info(f"Collection will end at: {end_time}")
    
    # Create tasks for data collection components
    tasks = []
    
    # Task 1: WebSocket Ingestor
    async def run_ingestor():
        try:
            logger.info("Starting WebSocket Ingestor...")
            await start_ingestor()
        except Exception as e:
            logger.error(f"Ingestor error: {e}")
            raise
    
    # Task 2: Feature Hub
    async def run_feature_hub():
        try:
            # Wait a bit for ingestor to start
            await asyncio.sleep(5)
            logger.info("Starting Feature Hub...")
            await start_feature_hub()
        except Exception as e:
            logger.error(f"Feature Hub error: {e}")
            raise
    
    # Task 3: Storage task (periodic data persistence)
    async def run_storage():
        try:
            while datetime.now() < end_time:
                await asyncio.sleep(300)  # Save every 5 minutes
                
                # Persist data from Redis to DuckDB
                logger.info("Persisting data to DuckDB...")
                stats = await storage_manager.persist_from_redis()
                
                logger.info(
                    f"Data persistence stats: {stats}",
                    elapsed_time=(datetime.now() - start_time).total_seconds() / 3600,
                    remaining_hours=(end_time - datetime.now()).total_seconds() / 3600
                )
        except Exception as e:
            logger.error(f"Storage error: {e}")
            raise
    
    # Task 4: Progress monitoring
    async def monitor_progress():
        try:
            while datetime.now() < end_time:
                await asyncio.sleep(60)  # Update every minute
                
                elapsed = datetime.now() - start_time
                remaining = end_time - datetime.now()
                progress = (elapsed.total_seconds() / (duration_hours * 3600)) * 100
                
                logger.info(
                    f"Collection progress: {progress:.1f}%",
                    elapsed=str(elapsed).split('.')[0],
                    remaining=str(remaining).split('.')[0]
                )
        except Exception as e:
            logger.error(f"Monitor error: {e}")
    
    # Start all tasks
    tasks = [
        asyncio.create_task(run_ingestor()),
        asyncio.create_task(run_feature_hub()),
        asyncio.create_task(run_storage()),
        asyncio.create_task(monitor_progress())
    ]
    
    try:
        # Wait for duration or until interrupted
        await asyncio.sleep(duration_hours * 3600)
        
        logger.info("Data collection period completed!")
        
        # Final data persistence
        logger.info("Performing final data persistence...")
        final_stats = await storage_manager.persist_from_redis()
        
        # Get collection summary
        summary = await storage_manager.get_data_summary()
        
        logger.info(
            "Data collection summary:",
            total_records=summary.get("total_records", 0),
            symbols=summary.get("symbols", []),
            time_range=summary.get("time_range", {}),
            data_types=summary.get("data_types", {})
        )
        
        # Save summary to file
        summary_path = project_root / "data" / "collection_summary.json"
        summary_path.parent.mkdir(exist_ok=True)
        
        import json
        with open(summary_path, "w") as f:
            json.dump({
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_hours": duration_hours,
                "testnet": testnet,
                "summary": summary
            }, f, indent=2)
        
        logger.info(f"Collection summary saved to: {summary_path}")
        
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
    except Exception as e:
        logger.error(f"Data collection error: {e}")
        raise
    finally:
        # Cancel all tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Close storage connection
        await storage_manager.close()
        
        logger.info("Data collection stopped")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect market data for ML training")
    parser.add_argument(
        "--duration", 
        type=int, 
        default=24, 
        help="Duration in hours to collect data (default: 24)"
    )
    parser.add_argument(
        "--testnet", 
        action="store_true", 
        help="Use testnet (recommended for initial collection)"
    )
    parser.add_argument(
        "--mainnet",
        action="store_true",
        help="Use mainnet (real data)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mainnet and args.testnet:
        logger.error("Cannot use both --testnet and --mainnet")
        sys.exit(1)
    
    # Default to testnet if neither specified
    use_testnet = not args.mainnet
    
    # Run data collection
    await collect_data(
        duration_hours=args.duration,
        testnet=use_testnet
    )


if __name__ == "__main__":
    asyncio.run(main())