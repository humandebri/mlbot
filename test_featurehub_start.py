#!/usr/bin/env python3
"""
Test FeatureHub startup independently to identify the issue
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import asyncio
from src.feature_hub.main import FeatureHub
from src.common.logging import get_logger

logger = get_logger(__name__)

async def test_featurehub_startup():
    """Test FeatureHub startup step by step"""
    print("🔍 Testing FeatureHub startup...")
    
    try:
        # Initialize FeatureHub
        print("Step 1: Initializing FeatureHub...")
        feature_hub = FeatureHub()
        print("✅ FeatureHub instance created")
        
        # Try to start it
        print("Step 2: Starting FeatureHub...")
        await feature_hub.start()
        print("✅ FeatureHub started successfully")
        
        # Wait a bit and check for features
        print("Step 3: Waiting 10 seconds for feature generation...")
        await asyncio.sleep(10)
        
        print("Step 4: Checking feature cache...")
        for symbol in ["BTCUSDT", "ETHUSDT", "ICPUSDT"]:
            features = feature_hub.get_latest_features(symbol)
            print(f"  {symbol}: {len(features) if features else 0} features")
        
        print("Step 5: Stopping FeatureHub...")
        await feature_hub.stop()
        print("✅ FeatureHub stopped successfully")
        
    except Exception as e:
        print(f"❌ Error during FeatureHub startup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_featurehub_startup())