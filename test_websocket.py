#!/usr/bin/env python3
"""Test WebSocket connection to Bybit."""

import asyncio
import websockets
import json


async def test_websocket():
    """Test WebSocket connection and subscriptions."""
    # Test both testnet and mainnet
    urls = {
        "testnet": "wss://stream-testnet.bybit.com/v5/public/linear",
        "mainnet": "wss://stream.bybit.com/v5/public/linear"
    }
    
    for env, url in urls.items():
        print(f"\n{'='*60}")
        print(f"Testing {env}: {url}")
        print('='*60)
        
        try:
            async with websockets.connect(url) as ws:
                print(f"✓ Connected to {env}")
                
                # Subscribe to multiple data feeds
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [
                        "kline.1.BTCUSDT",      # 1 minute klines
                        "trades.BTCUSDT",        # Trades
                        "liquidation.BTCUSDT",   # Liquidations
                        "orderbook.50.BTCUSDT"   # Orderbook
                    ]
                }
                
                await ws.send(json.dumps(subscribe_msg))
                print(f"→ Sent subscription request")
                
                # Listen for messages
                message_count = 0
                start_time = asyncio.get_event_loop().time()
                
                while message_count < 10 and (asyncio.get_event_loop().time() - start_time) < 30:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        data = json.loads(msg)
                        
                        if 'topic' in data:
                            message_count += 1
                            print(f"← Received: {data['topic']} - {data.get('type', 'update')}")
                            
                            # Show sample data
                            if message_count == 1:
                                print(f"  Sample data keys: {list(data.keys())}")
                                if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                                    print(f"  Data item keys: {list(data['data'][0].keys())}")
                        else:
                            print(f"← Status: {data}")
                            
                    except asyncio.TimeoutError:
                        print("⏱️  No messages received in 5 seconds")
                        
                print(f"\n✓ {env} test completed - received {message_count} messages")
                
        except Exception as e:
            print(f"✗ Error connecting to {env}: {e}")


async def test_rest_api():
    """Test REST API endpoints."""
    import aiohttp
    
    endpoints = {
        "testnet": "https://api-testnet.bybit.com",
        "mainnet": "https://api.bybit.com"
    }
    
    print(f"\n{'='*60}")
    print("Testing REST API")
    print('='*60)
    
    async with aiohttp.ClientSession() as session:
        for env, base_url in endpoints.items():
            print(f"\nTesting {env} REST API:")
            
            # Test server time
            try:
                async with session.get(f"{base_url}/v5/market/time") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"✓ Server time: {data['result']['timeSecond']}")
                    else:
                        print(f"✗ Server time failed: {resp.status}")
            except Exception as e:
                print(f"✗ Error: {e}")
            
            # Test recent trades
            try:
                async with session.get(
                    f"{base_url}/v5/market/recent-trade",
                    params={"category": "linear", "symbol": "BTCUSDT", "limit": 5}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        trades = data['result']['list']
                        print(f"✓ Recent trades: {len(trades)} trades")
                        if trades:
                            print(f"  Latest: ${trades[0]['price']} @ {trades[0]['time']}")
                    else:
                        print(f"✗ Recent trades failed: {resp.status}")
            except Exception as e:
                print(f"✗ Error: {e}")


async def main():
    """Run all tests."""
    print("Bybit Connection Test")
    print("====================")
    
    # Test WebSocket
    await test_websocket()
    
    # Test REST API
    await test_rest_api()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())