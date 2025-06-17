#!/usr/bin/env python3
"""
利用可能残高の詳細を確認
"""
import asyncio
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.common.account_monitor import AccountMonitor
from src.common.discord_notifier import discord_notifier
import json

async def check_balance_details():
    """残高の詳細を確認"""
    
    print("💰 残高詳細を確認中...")
    
    # AccountMonitorを使って残高取得
    monitor = AccountMonitor(check_interval=5)
    
    # 生のAPIレスポンスを取得するため、一時的に修正
    from src.common.config import settings
    import aiohttp
    import hmac
    import hashlib
    import time
    
    api_key = settings.bybit.api_key
    api_secret = settings.bybit.api_secret
    base_url = "https://api.bybit.com"
    
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    
    params = {"accountType": "UNIFIED"}
    query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    
    param_str = f"{timestamp}{api_key}{recv_window}{query_string}"
    signature = hmac.new(
        api_secret.encode('utf-8'),
        param_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-SIGN": signature,
        "X-BAPI-SIGN-TYPE": "2",
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }
    
    url = f"{base_url}/v5/account/wallet-balance?{query_string}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                data = await response.json()
                
                if response.status == 200 and data.get("retCode") == 0:
                    result = data.get("result", {})
                    if "list" in result and result["list"]:
                        account = result["list"][0]
                        
                        print("\n📊 アカウント概要:")
                        print(f"  - totalEquity: ${account.get('totalEquity')}")
                        print(f"  - totalWalletBalance: ${account.get('totalWalletBalance')}")
                        print(f"  - totalAvailableBalance: ${account.get('totalAvailableBalance')}")
                        print(f"  - totalMarginBalance: ${account.get('totalMarginBalance')}")
                        print(f"  - totalInitialMargin: ${account.get('totalInitialMargin')}")
                        print(f"  - totalMaintenanceMargin: ${account.get('totalMaintenanceMargin')}")
                        
                        # USDTの詳細を探す
                        coins = account.get("coin", [])
                        for coin in coins:
                            if coin.get("coin") == "USDT":
                                print(f"\n💵 USDT詳細:")
                                print(f"  - walletBalance: ${coin.get('walletBalance')}")
                                print(f"  - equity: ${coin.get('equity')}")
                                print(f"  - availableToWithdraw: ${coin.get('availableToWithdraw')}")
                                print(f"  - availableToBorrow: {coin.get('availableToBorrow')}")
                                print(f"  - totalOrderIM: ${coin.get('totalOrderIM')}")
                                print(f"  - totalPositionIM: ${coin.get('totalPositionIM')}")
                                print(f"  - unrealisedPnl: ${coin.get('unrealisedPnl')}")
                                print(f"  - locked: ${coin.get('locked')}")
                                
                                # Discord通知
                                fields = {
                                    "Total Equity": f"${account.get('totalEquity')}",
                                    "Total Available": f"${account.get('totalAvailableBalance')}",
                                    "USDT Wallet": f"${coin.get('walletBalance')}",
                                    "USDT Available": f"${coin.get('availableToWithdraw')}",
                                    "Order Margin": f"${coin.get('totalOrderIM')}",
                                    "Position Margin": f"${coin.get('totalPositionIM')}",
                                    "問題": "availableToWithdraw=出金可能額（通常0）"
                                }
                                
                                discord_notifier.send_notification(
                                    title="💰 利用可能残高の詳細分析",
                                    description="Available $0の原因を特定しました",
                                    color="ffff00",
                                    fields=fields
                                )
                                
                                print("\n❗ 問題の原因:")
                                print("  availableToWithdraw は出金可能額です")
                                print("  取引に使える残高は totalAvailableBalance を使用すべきです")
                                break
                        
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    asyncio.run(check_balance_details())