#!/usr/bin/env python3
"""
Quick integration test: 特徴量生成が正しく動作することを確認
短時間版（10秒）
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.config import settings
from src.common.logging import get_logger
from src.integration.simple_service_manager import SimpleServiceManager

logger = get_logger(__name__)


async def quick_test():
    """Quick feature generation test (10 seconds)"""
    service_manager = SimpleServiceManager()
    test_passed = False
    
    try:
        # Start services
        logger.info("Starting services...")
        await service_manager.start_all()
        
        # Wait for initial data
        logger.info("Waiting 8 seconds for data accumulation...")
        await asyncio.sleep(8)
        
        # Check feature counts
        status = service_manager.get_service_status()
        feature_counts = status["feature_hub"]["feature_counts"]
        
        logger.info("\n=== FEATURE COUNTS ===")
        all_good = True
        for symbol in settings.bybit.symbols:
            count = feature_counts.get(symbol, 0)
            if count > 0:
                logger.info(f"✅ {symbol}: {count} features")
            else:
                logger.error(f"❌ {symbol}: 0 features")
                all_good = False
        
        if all_good:
            logger.info("\n🎉 TEST PASSED: All symbols have features!")
            test_passed = True
        else:
            logger.error("\n❌ TEST FAILED: Some symbols have no features")
            
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
    finally:
        # Stop services
        await service_manager.stop_all()
    
    return test_passed


def main():
    print("Quick Feature Generation Test")
    print("="*40)
    
    # Run test
    passed = asyncio.run(quick_test())
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()