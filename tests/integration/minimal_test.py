#!/usr/bin/env python3
"""
最小テスト: 特徴量生成の確認のみ（5秒）
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.config import settings
from src.common.logging import get_logger
from src.integration.simple_service_manager import SimpleServiceManager

logger = get_logger(__name__)


async def minimal_test():
    """5秒間の最小テスト"""
    service_manager = SimpleServiceManager()
    test_passed = False
    
    try:
        # Start only essential services
        logger.info("Starting Ingestor and FeatureHub...")
        await service_manager.start_ingestor()
        await asyncio.sleep(2)
        await service_manager.start_feature_hub()
        
        # Wait briefly for features
        logger.info("Waiting 3 seconds for feature generation...")
        await asyncio.sleep(3)
        
        # Check only feature counts
        status = service_manager.get_service_status()
        feature_counts = status["feature_hub"]["feature_counts"]
        
        print("\n=== FEATURE COUNTS ===")
        all_good = True
        for symbol in settings.bybit.symbols:
            count = feature_counts.get(symbol, 0)
            if count > 0:
                print(f"✅ {symbol}: {count} features")
            else:
                print(f"❌ {symbol}: 0 features")
                all_good = False
        
        if all_good:
            print("\n🎉 MINIMAL TEST PASSED!")
            test_passed = True
        else:
            print("\n❌ MINIMAL TEST FAILED")
            
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        # Stop services
        await service_manager.stop_all()
    
    return test_passed


def main():
    print("Minimal Feature Test (5 seconds)")
    print("="*40)
    
    passed = asyncio.run(minimal_test())
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()