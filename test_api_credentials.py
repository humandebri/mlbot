#!/usr/bin/env python3
"""Test API credentials are loaded correctly."""

import os
import hmac
import hashlib
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BYBIT__API_KEY")
api_secret = os.getenv("BYBIT__API_SECRET")

print(f"API Key loaded: {'Yes' if api_key else 'No'}")
print(f"API Secret loaded: {'Yes' if api_secret else 'No'}")

if api_key:
    print(f"API Key (first 10 chars): {api_key[:10]}...")
    print(f"API Key length: {len(api_key)}")
    
if api_secret:
    print(f"API Secret (first 10 chars): {api_secret[:10]}...")
    print(f"API Secret length: {len(api_secret)}")
    
    # Test signature generation with known values
    test_string = "1750297177103KgMS2YHiCPG49hmWuV5000{\"category\":\"linear\",\"symbol\":\"BTCUSDT\",\"side\":\"Buy\",\"orderType\":\"Limit\",\"qty\":\"0.001\",\"price\":\"52279\",\"timeInForce\":\"PostOnly\"}"
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        test_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"\nTest signature: {signature}")
    print(f"Test string: {test_string}")