#!/usr/bin/env python3
"""Test importing feature engines directly"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.feature_hub.price_features import PriceFeatureEngine
    print("✅ PriceFeatureEngine imported successfully")
    
    # Try to instantiate
    engine = PriceFeatureEngine()
    print("✅ PriceFeatureEngine instantiated successfully")
    
    # Check if latest_features exists
    if hasattr(engine, 'latest_features'):
        print("✅ latest_features attribute exists")
    else:
        print("❌ latest_features attribute missing")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()