#!/usr/bin/env python3
"""
Automatic DuckDB updater - runs periodically to keep database current
"""

import schedule
import time
import logging
from datetime import datetime
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_update_duckdb.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def update_duckdb():
    """Run the DuckDB update script."""
    try:
        logger.info("Starting scheduled DuckDB update...")
        
        # Run the update script
        result = subprocess.run(
            [sys.executable, 'update_duckdb_enhanced.py', '--lookback-hours', '24'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("DuckDB update completed successfully")
            logger.info(f"Output: {result.stdout[-500:]}")  # Last 500 chars
        else:
            logger.error(f"DuckDB update failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Error running update: {e}")


def main():
    """Main loop for automatic updates."""
    logger.info("Starting automatic DuckDB updater...")
    
    # Schedule updates
    schedule.every(1).hours.do(update_duckdb)  # Every hour
    schedule.every().day.at("00:00").do(update_duckdb)  # Daily at midnight
    
    # Run initial update
    update_duckdb()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user")