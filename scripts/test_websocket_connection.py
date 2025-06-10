#!/usr/bin/env python3
"""
Simple WebSocket connection test to diagnose connectivity issues.
"""

import asyncio
import websockets
import json
import sys

async def test_bybit_websocket():
    """Test direct connection to Bybit WebSocket."""
    testnet_url = "wss://stream-testnet.bybit.com/v5/public/linear"
    mainnet_url = "wss://stream.bybit.com/v5/public/linear"
    
    print("Testing Bybit WebSocket connectivity...")
    
    for name, url in [("Testnet", testnet_url), ("Mainnet", mainnet_url)]:
        print(f"\nTesting {name}: {url}")
        
        try:
            print("Attempting connection...")
            websocket = await asyncio.wait_for(
                websockets.connect(url, ping_timeout=30, open_timeout=30),
                timeout=60
            )
            
            print(f"✅ {name} connected successfully!")
            
            # Try to send a simple subscription
            subscription = {
                "op": "subscribe",
                "args": ["kline.1s.BTCUSDT"]
            }
            
            print("Sending subscription...")
            await websocket.send(json.dumps(subscription))
            print("✅ Subscription sent successfully!")
            
            # Try to receive a few messages
            print("Waiting for messages...")
            message_count = 0
            
            async for message in websocket:
                if message_count >= 3:  # Stop after 3 messages
                    break
                    
                try:
                    data = json.loads(message)
                    print(f"✅ Received message {message_count + 1}: {data.get('topic', 'unknown')}")
                    message_count += 1
                except json.JSONDecodeError:
                    print(f"⚠️ Received non-JSON message: {message[:100]}...")
                    message_count += 1
            
            await websocket.close()
            print(f"✅ {name} test completed successfully!")
            
        except asyncio.TimeoutError:
            print(f"❌ {name} connection timeout")
        except ConnectionRefusedError:
            print(f"❌ {name} connection refused")
        except Exception as e:
            print(f"❌ {name} connection failed: {e}")

async def test_network_connectivity():
    """Test basic network connectivity."""
    import aiohttp
    
    print("\nTesting basic network connectivity...")
    
    # Test DNS resolution and HTTP connectivity
    test_urls = [
        "https://api-testnet.bybit.com/v5/market/recent-trade?symbol=BTCUSDT&limit=1",
        "https://api.bybit.com/v5/market/recent-trade?symbol=BTCUSDT&limit=1",
        "https://httpbin.org/get"
    ]
    
    async with aiohttp.ClientSession() as session:
        for url in test_urls:
            try:
                print(f"Testing HTTP GET: {url}")
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        print(f"✅ HTTP GET successful: {response.status}")
                    else:
                        print(f"⚠️ HTTP GET returned: {response.status}")
            except Exception as e:
                print(f"❌ HTTP GET failed: {e}")

async def main():
    """Main test function."""
    print("=== WebSocket Connectivity Diagnosis ===\n")
    
    # Test basic network connectivity first
    await test_network_connectivity()
    
    # Test WebSocket connectivity
    await test_bybit_websocket()
    
    print("\n=== Diagnosis Complete ===")
    print("\nIf WebSocket connections fail but HTTP works:")
    print("1. Check firewall settings")
    print("2. Check corporate proxy/VPN settings")
    print("3. Try different network (mobile hotspot)")
    print("4. Check if WebSocket traffic is blocked")

if __name__ == "__main__":
    asyncio.run(main())