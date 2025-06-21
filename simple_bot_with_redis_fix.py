#!/usr/bin/env python3
"""
Simple bot with minimal Redis integration fix
Based on working simple_improved_bot_with_trading_fixed.py
Just adds Redis data refresh capability
"""

import os
import sys
import redis
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the working bot
from simple_improved_bot_with_trading_fixed import SimpleImprovedTradingBot

class SimpleImprovedTradingBotWithRedis(SimpleImprovedTradingBot):
    """Extend the working bot with Redis data refresh capability."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize Redis connection
        self.redis_client = None
        self._connect_redis()
        
        # Track last Redis update time
        self.last_redis_update = {}
        
    def _connect_redis(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                decode_responses=True
            )
            self.redis_client.ping()
            print("✅ Redis connected successfully")
        except Exception as e:
            print(f"⚠️  Redis connection failed: {e}")
            self.redis_client = None
    
    def refresh_historical_data(self):
        """Periodically refresh historical data cache from Redis."""
        if not self.redis_client:
            return
        
        try:
            # Get latest kline data from Redis
            kline_data = self.redis_client.xrevrange('market_data:kline', count=1000)
            
            if kline_data:
                print(f"📊 Found {len(kline_data)} recent kline entries in Redis")
                
                # Process and update feature generator's cache
                for symbol in self.symbols:
                    symbol_data = []
                    
                    for entry_id, data in kline_data:
                        try:
                            parsed = json.loads(data.get('data', '{}'))
                            if symbol in parsed.get('topic', ''):
                                symbol_data.append({
                                    'timestamp': parsed.get('timestamp', 0),
                                    'open': float(parsed.get('open', 0)),
                                    'high': float(parsed.get('high', 0)),
                                    'low': float(parsed.get('low', 0)),
                                    'close': float(parsed.get('close', 0)),
                                    'volume': float(parsed.get('volume', 0))
                                })
                        except:
                            continue
                    
                    if symbol_data:
                        # Refresh the feature generator's cache
                        self.feature_generator.update_historical_cache(symbol)
                        self.last_redis_update[symbol] = len(symbol_data)
                        print(f"✅ Updated {symbol} with {len(symbol_data)} Redis records")
                        
        except Exception as e:
            print(f"⚠️  Error refreshing from Redis: {e}")
    
    async def initialize(self):
        """Initialize with Redis refresh."""
        result = await super().initialize()
        
        if result:
            # Do initial Redis refresh
            self.refresh_historical_data()
            
        return result
    
    async def trading_loop(self):
        """Enhanced trading loop with periodic Redis refresh."""
        self.running = True
        refresh_counter = 0
        
        while self.running:
            try:
                # Refresh from Redis every 60 iterations (10 minutes)
                refresh_counter += 1
                if refresh_counter >= 60:
                    self.refresh_historical_data()
                    refresh_counter = 0
                
                # Run normal trading logic
                await super().trading_loop()
                
            except Exception as e:
                print(f"Error in enhanced trading loop: {e}")
                await asyncio.sleep(30)


if __name__ == "__main__":
    import asyncio
    
    async def main():
        bot = SimpleImprovedTradingBotWithRedis()
        
        # Use the same signal handlers as parent
        import signal
        
        def signal_handler(sig):
            print(f"Received signal {sig}")
            asyncio.create_task(bot.stop())
        
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler, sig)
        
        try:
            if await bot.start():
                while bot.running:
                    await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
        finally:
            await bot.stop()
    
    print("⚠️  警告: 実際の資金で取引を行います")
    print("🔴 LIVE TRADING MODE - Redis Data Refresh Enabled")
    asyncio.run(main())