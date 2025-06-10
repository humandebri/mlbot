#!/usr/bin/env python3
"""
Script to stop the trading system gracefully.

Usage:
    python scripts/stop_system.py
"""

import asyncio
import aiohttp
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger

logger = get_logger(__name__)


async def stop_system(api_url: str = "http://localhost:8080") -> bool:
    """Stop the trading system via API."""
    try:
        async with aiohttp.ClientSession() as session:
            logger.info("Stopping trading system...")
            
            async with session.post(f"{api_url}/system/stop", timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"System stop response: {data['message']}")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Failed to stop system: {error}")
                    return False
                    
    except Exception as e:
        logger.error(f"Error stopping system: {e}")
        return False


async def main():
    """Main entry point."""
    success = await stop_system()
    
    if success:
        logger.info("Trading system stopped successfully")
        sys.exit(0)
    else:
        logger.error("Failed to stop trading system")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())