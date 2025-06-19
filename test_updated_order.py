#!/usr/bin/env python3
"""Test the updated Bybit order placement."""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))

from src.common.bybit_client import BybitRESTClient

load_dotenv()

async def test_order():
    """Test order placement with the updated client."""
    
    async with BybitRESTClient(testnet=False) as client:
        # Get current BTC price first
        ticker = await client.get_ticker("BTCUSDT")
        if ticker:
            current_price = float(ticker.get("lastPrice", 0))
            print(f"Current BTC price: ${current_price:,.2f}")
            
            # Place a small test order well below market price
            test_price = round(current_price * 0.5, 0)  # 50% below market
            test_qty = 0.001  # Minimum for BTC
            
            print(f"\nPlacing test order:")
            print(f"Symbol: BTCUSDT")
            print(f"Side: Buy")
            print(f"Type: post_only (Limit with PostOnly)")
            print(f"Quantity: {test_qty}")
            print(f"Price: ${test_price:,.2f}")
            
            result = await client.create_order(
                symbol="BTCUSDT",
                side="buy",
                order_type="post_only",
                qty=test_qty,
                price=test_price
            )
            
            if result:
                print(f"\nSuccess! Order ID: {result.get('orderId')}")
                print(f"Full result: {result}")
                
                # Cancel the order immediately
                order_id = result.get('orderId')
                if order_id:
                    await asyncio.sleep(1)
                    cancelled = await client.cancel_order("BTCUSDT", order_id)
                    print(f"\nOrder cancelled: {cancelled}")
            else:
                print("\nFailed to place order - check logs for details")
        else:
            print("Failed to get ticker price")

if __name__ == "__main__":
    asyncio.run(test_order())