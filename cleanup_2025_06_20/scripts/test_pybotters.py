#!/usr/bin/env python3
"""
Test pybotters with Bybit to see if we can get market data.
"""

import asyncio
import pybotters

async def test_pybotters_bybit():
    """Test pybotters with Bybit."""
    print("Testing pybotters with Bybit...")
    
    async with pybotters.Client() as client:
        # Create WebSocket connection to Bybit
        ws = await client.ws_connect('wss://stream.bybit.com/v5/public/linear')
        
        # Send subscription
        await ws.send_json({
            'op': 'subscribe',
            'args': ['kline.1s.BTCUSDT', 'orderbook.25.BTCUSDT', 'trades.BTCUSDT']
        })
        
        print("✅ Connected with pybotters")
        print("Subscription sent, listening for messages...")
        
        message_count = 0
        max_messages = 10
        
        async for msg in ws:
            print(f"\n--- Message {message_count + 1} ---")
            print(f"Message type: {type(msg)}")
            print(f"Content: {msg}")
            
            message_count += 1
            if message_count >= max_messages:
                break
        
        print(f"\n✅ Received {message_count} messages via pybotters")

async def test_pybotters_store():
    """Test pybotters with data store functionality."""
    print("\nTesting pybotters with data store...")
    
    async with pybotters.Client() as client:
        store = pybotters.BybitDataStore()
        
        # Subscribe to Bybit WebSocket
        ws = await client.ws_connect('wss://stream.bybit.com/v5/public/linear')
        
        # Send subscription
        await ws.send_json({
            'op': 'subscribe', 
            'args': ['kline.1s.BTCUSDT', 'orderbook.25.BTCUSDT']
        })
        
        # Process messages through the store
        msg_count = 0
        async for msg in ws:
            store.onmessage(msg)
            msg_count += 1
            if msg_count >= 20:  # Process 20 messages then check store
                break
        
        print("✅ Connected with pybotters DataStore")
        
        # Wait a bit for data to accumulate
        await asyncio.sleep(10)
        
        # Check what data we have
        print(f"Kline data count: {len(store.kline)}")
        print(f"Orderbook data count: {len(store.orderbook)}")
        
        if store.kline:
            print("Sample kline data:")
            for i, (key, value) in enumerate(store.kline.items()):
                if i >= 3:  # Show first 3 items
                    break
                print(f"  {key}: {value}")
        
        if store.orderbook:
            print("Sample orderbook data:")
            for i, (key, value) in enumerate(store.orderbook.items()):
                if i >= 3:  # Show first 3 items
                    break
                print(f"  {key}: {value}")

if __name__ == "__main__":
    print("=== Testing pybotters with Bybit ===")
    
    try:
        # Test basic WebSocket
        asyncio.run(test_pybotters_bybit())
        
        # Test with DataStore
        asyncio.run(test_pybotters_store())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()