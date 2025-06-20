#!/usr/bin/env python3
"""Test script to verify ML bot initialization fix."""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))

async def test_initialization():
    """Test that the ML bot initializes correctly."""
    print("🧪 Testing ML bot initialization fix...")
    
    try:
        # Import after path setup
        from src.common.config import settings
        from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
        
        # Test creating InferenceConfig
        print("✅ Creating InferenceConfig...")
        inference_config = InferenceConfig(
            model_path=settings.model.model_path,
            enable_batching=False,
            enable_thompson_sampling=False,
            confidence_threshold=0.65
        )
        print(f"   Model path: {inference_config.model_path}")
        print(f"   Thompson Sampling: {inference_config.enable_thompson_sampling}")
        
        # Test creating InferenceEngine
        print("✅ Creating InferenceEngine...")
        inference_engine = InferenceEngine(inference_config)
        print("   InferenceEngine initialized successfully!")
        
        # Test the bot initialization
        print("\n✅ Testing full bot initialization...")
        from working_ml_production_bot import WorkingMLProductionBot
        
        bot = WorkingMLProductionBot()
        await bot.initialize()
        print("   Bot initialized successfully!")
        
        # Clean up
        await bot.cleanup()
        
        print("\n🎉 All tests passed! The fix is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_initialization())