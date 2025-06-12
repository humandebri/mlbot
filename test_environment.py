#!/usr/bin/env python3
"""Test environment and dependencies."""

import sys
print(f"Python: {sys.version}")
print(f"Python path: {sys.executable}")

try:
    import structlog
    print("✓ structlog imported")
except ImportError as e:
    print(f"✗ structlog import failed: {e}")

try:
    import redis
    print("✓ redis imported")
except ImportError as e:
    print(f"✗ redis import failed: {e}")

try:
    import duckdb
    print("✓ duckdb imported")
except ImportError as e:
    print(f"✗ duckdb import failed: {e}")

try:
    import websockets
    print("✓ websockets imported")
except ImportError as e:
    print(f"✗ websockets import failed: {e}")

try:
    import lightgbm
    print("✓ lightgbm imported")
except ImportError as e:
    print(f"✗ lightgbm import failed: {e}")

try:
    from src.common.logging import get_logger
    print("✓ src.common.logging imported")
except ImportError as e:
    print(f"✗ src.common.logging import failed: {e}")

try:
    from src.common.config import settings
    print("✓ src.common.config imported")
    print(f"  TESTNET: {settings.bybit.testnet}")
except ImportError as e:
    print(f"✗ src.common.config import failed: {e}")

print("\nEnvironment test complete!")