#!/usr/bin/env python3
"""Check data in Redis streams."""

import asyncio
import redis.asyncio as redis
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


async def check_redis_data():
    """Check what data is available in Redis."""
    try:
        # Connect to Redis
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Check if Redis is running
        await client.ping()
        logger.info("Redis connection successful")
        
        # Get all keys
        keys = await client.keys('*')
        logger.info(f"Found {len(keys)} keys in Redis")
        
        # Check market data streams
        market_streams = [k for k in keys if k.startswith('market_data:')]
        logger.info(f"Market data streams: {market_streams}")
        
        # Check stream lengths
        for stream in market_streams:
            length = await client.xlen(stream)
            logger.info(f"  {stream}: {length} messages")
            
            # Show last few messages
            if length > 0:
                messages = await client.xrevrange(stream, count=3)
                logger.info(f"  Last 3 messages from {stream}:")
                for msg_id, data in messages:
                    logger.info(f"    {msg_id}: {list(data.keys())}")
        
        # Check feature streams
        feature_keys = [k for k in keys if 'features' in k]
        logger.info(f"\nFeature keys: {feature_keys}")
        
        await client.close()
        
    except Exception as e:
        logger.error(f"Error checking Redis: {e}")
        logger.info("Make sure Redis is running: docker-compose up -d redis")


if __name__ == "__main__":
    asyncio.run(check_redis_data())