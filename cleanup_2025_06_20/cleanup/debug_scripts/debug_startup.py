#!/usr/bin/env python3
"""
Debug startup script to identify where the system is hanging
"""
import asyncio
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier

logger = get_logger(__name__)

async def test_startup_components():
    """Test each component individually"""
    
    print("🔍 Testing individual components...")
    
    # Test 1: Discord notification
    print("\n1. Testing Discord notification...")
    try:
        result = discord_notifier.send_notification(
            title="🧪 Startup Debug Test",
            description="Testing Discord connectivity during debugging",
            color="00ff00",
            fields={
                "Status": "Testing",
                "Time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )
        print(f"   Discord test result: {result}")
    except Exception as e:
        print(f"   Discord test failed: {e}")
    
    # Test 2: Feature Hub import
    print("\n2. Testing FeatureHub import...")
    try:
        from src.feature_hub.main import FeatureHub
        feature_hub = FeatureHub()
        print(f"   FeatureHub import: OK")
        print(f"   FeatureHub has _start_feature_hub_background: {hasattr(feature_hub, '_start_feature_hub_background')}")
    except Exception as e:
        print(f"   FeatureHub import failed: {e}")
    
    # Test 3: Order Router import
    print("\n3. Testing OrderRouter import...")
    try:
        from src.order_router.main import OrderRouter
        order_router = OrderRouter()
        print(f"   OrderRouter import: OK")
        print(f"   OrderRouter running attribute: {hasattr(order_router, 'running')}")
    except Exception as e:
        print(f"   OrderRouter import failed: {e}")
    
    # Test 4: Redis connection
    print("\n4. Testing Redis connection...")
    try:
        from src.common.database import get_redis_client
        redis_client = await get_redis_client()
        await redis_client.ping()
        print(f"   Redis connection: OK")
    except Exception as e:
        print(f"   Redis connection failed: {e}")
    
    print("\n✅ Component testing completed")

async def main():
    print("🚀 Starting startup debug...")
    await test_startup_components()
    print("🏁 Debug completed")

if __name__ == "__main__":
    asyncio.run(main())