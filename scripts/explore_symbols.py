#!/usr/bin/env python3
"""
Explore available trading symbols on Bybit to expand data collection.
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any

async def get_available_symbols(testnet: bool = False) -> List[Dict[str, Any]]:
    """Get all available linear perpetual symbols from Bybit."""
    base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
    url = f"{base_url}/v5/market/instruments-info"
    
    params = {
        "category": "linear",
        "status": "Trading"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("retCode") == 0:
                    return data.get("result", {}).get("list", [])
    return []

async def analyze_symbols():
    """Analyze available symbols and recommend additions."""
    print("🔍 Analyzing available symbols on Bybit...")
    
    # Get symbols from both testnet and mainnet
    testnet_symbols = await get_available_symbols(testnet=True)
    mainnet_symbols = await get_available_symbols(testnet=False)
    
    print(f"📊 Testnet symbols: {len(testnet_symbols)}")
    print(f"📊 Mainnet symbols: {len(mainnet_symbols)}")
    
    # Focus on major cryptocurrencies with high volume
    major_symbols = []
    for symbol_info in mainnet_symbols:
        symbol = symbol_info.get("symbol", "")
        base_coin = symbol_info.get("baseCoin", "")
        quote_coin = symbol_info.get("quoteCoin", "")
        
        # Filter for USDT perpetuals of major coins
        if (quote_coin == "USDT" and 
            symbol.endswith("USDT") and 
            base_coin in ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "MATIC", "DOT", 
                         "AVAX", "LINK", "UNI", "LTC", "BCH", "ATOM", "FIL", "TRX",
                         "ETC", "XLM", "VET", "ICP", "THETA", "FTT", "ALGO", "AAVE",
                         "MKR", "COMP", "YFI", "SUSHI", "CRV", "1INCH", "BAL"]):
            major_symbols.append({
                "symbol": symbol,
                "baseCoin": base_coin,
                "quoteCoin": quote_coin,
                "priceScale": symbol_info.get("priceScale", ""),
                "lotSizeFilter": symbol_info.get("lotSizeFilter", {})
            })
    
    print(f"\n🎯 Recommended major symbols ({len(major_symbols)}):")
    for i, symbol_info in enumerate(major_symbols[:20], 1):  # Show first 20
        print(f"{i:2d}. {symbol_info['symbol']} ({symbol_info['baseCoin']})")
    
    # Current symbols
    current_symbols = ["ETHUSDT", "ICPUSDT"]
    recommended_additions = [s["symbol"] for s in major_symbols if s["symbol"] not in current_symbols]
    
    print(f"\n✅ Current symbols: {current_symbols}")
    print(f"➕ Recommended additions: {recommended_additions[:15]}")  # Top 15 additions
    
    return recommended_additions[:15]

if __name__ == "__main__":
    asyncio.run(analyze_symbols())