#!/usr/bin/env python3
import asyncio
import websockets

async def quick_test():
    url = "wss://stream-testnet.bybit.com/v5/public/linear"
    print(f"Testing: {url}")
    
    try:
        ws = await asyncio.wait_for(
            websockets.connect(url, open_timeout=10, ping_timeout=10), 
            timeout=15
        )
        print("✅ Connected!")
        await ws.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(quick_test())