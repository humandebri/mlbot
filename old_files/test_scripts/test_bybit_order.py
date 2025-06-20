#!/usr/bin/env python3
"""Test Bybit order placement directly."""

import asyncio
import json
import hmac
import hashlib
import time
import os
from dotenv import load_dotenv
import aiohttp

load_dotenv()

API_KEY = os.getenv("BYBIT__API_KEY")
API_SECRET = os.getenv("BYBIT__API_SECRET")

async def test_order():
    """Test order placement with correct signature."""
    url = "https://api.bybit.com/v5/order/create"
    
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    
    # Order parameters - IMPORTANT: Order matters for signature!
    params = {
        "category": "linear",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderType": "Limit",
        "qty": "0.001",
        "price": "50000",
        "timeInForce": "PostOnly"
    }
    
    # Create param string for signature - DO NOT sort keys for POST requests
    param_str = json.dumps(params, separators=(',', ':'))
    
    # Create signature
    sign_str = f"{timestamp}{API_KEY}{recv_window}{param_str}"
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        sign_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }
    
    print(f"Timestamp: {timestamp}")
    print(f"Sign string: {sign_str}")
    print(f"Signature: {signature}")
    print(f"Params: {param_str}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=params, headers=headers) as response:
            result = await response.json()
            print(f"\nResponse: {json.dumps(result, indent=2)}")
            
            if result.get("retCode") != 0:
                print(f"\nError: {result.get('retMsg')}")
            else:
                print("\nSuccess!")

if __name__ == "__main__":
    asyncio.run(test_order())