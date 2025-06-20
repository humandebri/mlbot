#!/usr/bin/env python3
"""
Debug WebSocket to see what messages we're actually receiving.
"""

import asyncio
import json
import websockets

async def debug_websocket():
    """Connect to Bybit and see what messages we get."""
    # Try mainnet instead of testnet for more activity
    url = "wss://stream.bybit.com/v5/public/linear"
    
    print(f"Connecting to: {url}")
    
    try:
        async with websockets.connect(url, ping_timeout=30, open_timeout=30) as websocket:
            print("✅ Connected successfully!")
            
            # Subscribe to feeds
            subscription = {
                "op": "subscribe",
                "args": ["kline.1s.BTCUSDT", "orderbook.25.BTCUSDT", "trades.BTCUSDT"]
            }
            
            print("Sending subscription...")
            await websocket.send(json.dumps(subscription))
            print(f"Subscription sent: {subscription}")
            
            # Listen for messages with timeout
            print("Listening for messages (max 20, 30 second timeout)...")
            message_count = 0
            max_messages = 20
            start_time = asyncio.get_event_loop().time()
            timeout = 30  # 30 second timeout
            
            try:
                async for message in websocket:
                    # Check timeout
                    if asyncio.get_event_loop().time() - start_time > timeout:
                        print("⏰ Timeout reached, stopping...")
                        break
                        
                    if message_count >= max_messages:
                        break
                    
                    try:
                        data = json.loads(message)
                        print(f"\n--- Message {message_count + 1} ---")
                        print(f"Type: {type(data)}")
                        print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                        
                        if isinstance(data, dict):
                            if 'op' in data:
                                print(f"Operation: {data['op']}")
                            if 'topic' in data:
                                print(f"Topic: {data['topic']}")
                            if 'data' in data:
                                print(f"Data type: {type(data['data'])}")
                                if isinstance(data['data'], list) and len(data['data']) > 0:
                                    print(f"First data item: {data['data'][0]}")
                            if 'success' in data:
                                print(f"Success: {data['success']}")
                            if 'ret_msg' in data:
                                print(f"Message: {data['ret_msg']}")
                        
                        # Print full message for first few
                        if message_count < 3:
                            print(f"Full message: {json.dumps(data, indent=2)}")
                        
                        message_count += 1
                        
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error: {e}")
                        print(f"Raw message: {message[:200]}...")
                    except Exception as e:
                        print(f"❌ Error processing message: {e}")
                        print(f"Raw message: {message[:200]}...")
                    
                    # Small delay between messages
                    await asyncio.sleep(0.1)
            
            except asyncio.TimeoutError:
                print("⏰ Timeout waiting for messages")
            
            print(f"\n✅ Received {message_count} messages total")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_websocket())