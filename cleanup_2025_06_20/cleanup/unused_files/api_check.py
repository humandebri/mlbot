import os
from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.getenv("BYBIT__API_KEY", "NOT_SET")
api_secret = os.getenv("BYBIT__API_SECRET", "NOT_SET")
testnet = os.getenv("BYBIT__TESTNET", "NOT_SET")
env = os.getenv("ENVIRONMENT", "NOT_SET")

print("=== API設定確認 ===")
print(f"API_KEY: {api_key[:10]}..." if api_key \!= "NOT_SET" else "API_KEY: NOT_SET")
print(f"API_SECRET: {api_secret[:10]}..." if api_secret \!= "NOT_SET" else "API_SECRET: NOT_SET")
print(f"TESTNET: {testnet}")
print(f"ENVIRONMENT: {env}")

# 実際のBybit APIテスト
print("\n=== Bybit API接続テスト ===")
import aiohttp
import asyncio
import time
import hmac
import hashlib
import json

async def test_bybit_api():
    if api_key == "NOT_SET" or api_secret == "NOT_SET":
        print("APIキーが設定されていません")
        return
    
    # Test endpoint
    url = "https://api.bybit.com/v5/account/wallet-balance"
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    params = {"accountType": "UNIFIED"}
    param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    
    sign_str = f"{timestamp}{api_key}{recv_window}{param_str}"
    signature = hmac.new(
        api_secret.encode('utf-8'),
        sign_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}?{param_str}", headers=headers) as response:
                print(f"Response Status: {response.status}")
                data = await response.json()
                print(f"Response: {data}")
                
                if response.status == 200 and data.get("retCode") == 0:
                    print("✅ APIキー認証成功")
                else:
                    print("❌ APIキー認証失敗")
                    
    except Exception as e:
        print(f"❌ API接続エラー: {e}")

asyncio.run(test_bybit_api())
