#!/usr/bin/env python3
"""
APIキー認証をテスト
"""
import asyncio
import hmac
import hashlib
import time
import aiohttp
import json
import os

async def test_api_key():
    """APIキーの動作確認"""
    
    # 環境変数から読み込み
    api_key = os.getenv('BYBIT__API_KEY', 'KgMS2YHiCPG49hmWuV')
    api_secret = os.getenv('BYBIT__API_SECRET', 'LUkW6d5SHHIrbu2SvEDPjFwGYkyUGRun8JF3')
    
    print(f"APIキー長: {len(api_key)}")
    print(f"APIシークレット長: {len(api_secret)}")
    print(f"APIキー（最初の5文字）: {api_key[:5]}...")
    
    # 本番環境のエンドポイント
    base_url = "https://api.bybit.com"
    endpoint = "/v5/account/wallet-balance"
    
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    
    # パラメータ
    params = {
        "accountType": "UNIFIED"
    }
    
    # クエリ文字列作成
    query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    
    # 署名作成
    param_str = f"{timestamp}{api_key}{recv_window}{query_string}"
    signature = hmac.new(
        api_secret.encode('utf-8'),
        param_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # ヘッダー
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-SIGN": signature,
        "X-BAPI-SIGN-TYPE": "2",
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }
    
    # リクエスト
    url = f"{base_url}{endpoint}?{query_string}"
    
    print(f"\n🔍 APIキー認証テスト")
    print(f"URL: {url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                print(f"ステータスコード: {response.status}")
                
                data = await response.json()
                print(f"レスポンス: {json.dumps(data, indent=2)}")
                
                if response.status == 200 and data.get("retCode") == 0:
                    result = data.get("result", {})
                    if "list" in result and result["list"]:
                        balance = result["list"][0]
                        total_equity = float(balance.get("totalEquity", 0))
                        print(f"\n✅ 認証成功！")
                        print(f"💰 残高: ${total_equity:.8f}")
                    else:
                        print(f"⚠️ 残高データなし")
                else:
                    print(f"❌ エラー: {data.get('retMsg', 'Unknown error')}")
                    
    except Exception as e:
        print(f"❌ 接続エラー: {e}")
    
    # Testnet接続もテスト
    print(f"\n🔍 Testnet接続テスト")
    testnet_url = "https://api-testnet.bybit.com"
    url = f"{testnet_url}{endpoint}?{query_string}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                print(f"Testnetステータス: {response.status}")
                if response.status == 200:
                    print("✅ Testnet接続OK")
                else:
                    print("❌ Testnet接続失敗")
    except:
        pass

if __name__ == "__main__":
    asyncio.run(test_api_key())