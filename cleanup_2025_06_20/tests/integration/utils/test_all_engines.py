#!/usr/bin/env python3
"""Test importing all feature engines"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    print("Testing imports...")
    
    from src.feature_hub.price_features import PriceFeatureEngine
    print("✅ PriceFeatureEngine imported")
    
    from src.feature_hub.micro_liquidity import MicroLiquidityEngine
    print("✅ MicroLiquidityEngine imported")
    
    from src.feature_hub.volatility_momentum import VolatilityMomentumEngine
    print("✅ VolatilityMomentumEngine imported")
    
    from src.feature_hub.liquidation_features import LiquidationFeatureEngine
    print("✅ LiquidationFeatureEngine imported")
    
    from src.feature_hub.time_context import TimeContextEngine
    print("✅ TimeContextEngine imported")
    
    from src.feature_hub.advanced_features import AdvancedFeatureAggregator
    print("✅ AdvancedFeatureAggregator imported")
    
    print("\nTesting instantiation...")
    
    e1 = PriceFeatureEngine()
    print("✅ PriceFeatureEngine instantiated")
    
    e2 = MicroLiquidityEngine()
    print("✅ MicroLiquidityEngine instantiated")
    
    e3 = VolatilityMomentumEngine()
    print("✅ VolatilityMomentumEngine instantiated")
    
    e4 = LiquidationFeatureEngine()
    print("✅ LiquidationFeatureEngine instantiated")
    
    e5 = TimeContextEngine()
    print("✅ TimeContextEngine instantiated")
    
    e6 = AdvancedFeatureAggregator()
    print("✅ AdvancedFeatureAggregator instantiated")
    
    print("\nAll tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()