#!/usr/bin/env python3
"""Test simple order placement with minimal code."""

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

async def test_simple_order():
    """Test order with minimal implementation."""
    # First, let's get account info to verify API works
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    
    # Test with account info endpoint first
    endpoint = "/v5/account/wallet-balance"
    params_str = "accountType=UNIFIED&coin=USDT"
    sign_str = f"{timestamp}{API_KEY}{recv_window}{params_str}"
    
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        sign_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature,
        "X-BAPI-RECV-WINDOW": recv_window
    }
    
    async with aiohttp.ClientSession() as session:
        # Test wallet balance first
        url = f"https://api.bybit.com{endpoint}?{params_str}"
        print(f"Testing wallet balance endpoint...")
        print(f"URL: {url}")
        
        async with session.get(url, headers=headers) as response:
            result = await response.json()
            print(f"Wallet response: {json.dumps(result, indent=2)}")
            
            if result.get("retCode") != 0:
                print(f"\nError: {result.get('retMsg')}")
                return
        
        # Now test order placement
        print("\n" + "="*50)
        print("Testing order placement...")
        
        timestamp = str(int(time.time() * 1000))
        params = {
            "category": "linear",
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderType": "Limit",
            "qty": "0.001",
            "price": "50000",
            "timeInForce": "PostOnly"
        }
        
        param_str = json.dumps(params, separators=(',', ':'))
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
        
        print(f"Params: {param_str}")
        print(f"Sign string: {sign_str}")
        print(f"Signature: {signature}")
        
        url = "https://api.bybit.com/v5/order/create"
        async with session.post(url, json=params, headers=headers) as response:
            result = await response.json()
            print(f"\nOrder response: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_simple_order())