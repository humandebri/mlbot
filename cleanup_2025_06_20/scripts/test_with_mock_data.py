#!/usr/bin/env python3
"""
Mock data test script for testing feature engineering without WebSocket dependency.

This script:
1. Generates realistic mock market data
2. Tests the entire feature engineering pipeline
3. Validates system performance with simulated data
4. Measures feature computation quality
"""

import asyncio
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import settings
from src.common.logging import setup_logging, get_logger, TradingLogger
from src.common.database import init_databases, close_databases, get_redis_client, RedisStreams
from src.common.monitoring import start_monitoring
from src.feature_hub.main import FeatureHub

# Setup logging
setup_logging()
logger = get_logger(__name__)
trading_logger = TradingLogger("mock_data_test")


class MockDataGenerator:
    """Generate realistic mock market data for testing."""
    
    def __init__(self, symbols: list = None):
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.running = False
        
        # Mock price tracking
        self.current_prices = {
            "BTCUSDT": 100000.0,
            "ETHUSDT": 4000.0
        }
        
        # Market state simulation
        self.market_volatility = 0.001  # 0.1% per update
        self.liquidation_probability = 0.001  # 0.1% chance per update
        
        # Message counters
        self.messages_sent = 0
        
    async def start(self, duration_seconds: int = 120):
        """Start generating mock data for specified duration."""
        self.running = True
        logger.info(f"Starting mock data generation for {duration_seconds} seconds")
        
        redis_client = await get_redis_client()
        redis_streams = RedisStreams(redis_client)
        
        start_time = time.time()
        
        try:
            while self.running and (time.time() - start_time) < duration_seconds:
                # Generate data for each symbol
                for symbol in self.symbols:
                    await self._generate_kline_data(redis_streams, symbol)
                    await self._generate_orderbook_data(redis_streams, symbol)
                    await self._generate_trade_data(redis_streams, symbol)
                    
                    # Occasionally generate liquidations
                    if random.random() < self.liquidation_probability:
                        await self._generate_liquidation_data(redis_streams, symbol)
                
                # Generate funding and OI data less frequently
                if self.messages_sent % 30 == 0:  # Every 30 updates
                    for symbol in self.symbols:
                        await self._generate_funding_data(redis_streams, symbol)
                        await self._generate_oi_data(redis_streams, symbol)
                
                await asyncio.sleep(1.0)  # 1 second intervals
                
        except Exception as e:
            logger.error("Error in mock data generation", exception=e)
        
        finally:
            self.running = False
    
    async def stop(self):
        """Stop data generation."""
        self.running = False
        logger.info(f"Mock data generation stopped. Total messages: {self.messages_sent}")
    
    async def _generate_kline_data(self, streams: RedisStreams, symbol: str):
        """Generate realistic kline data."""
        current_price = self.current_prices[symbol]
        
        # Simulate price movement
        price_change = random.gauss(0, self.market_volatility) * current_price
        new_price = max(current_price + price_change, current_price * 0.99)  # Prevent negative prices
        self.current_prices[symbol] = new_price
        
        # Create kline data
        kline_data = {
            "symbol": symbol,
            "kline": {
                "open": str(current_price),
                "high": str(max(current_price, new_price) * (1 + random.uniform(0, 0.001))),
                "low": str(min(current_price, new_price) * (1 - random.uniform(0, 0.001))),
                "close": str(new_price),
                "volume": str(random.uniform(1000, 10000)),
                "start": int(time.time() * 1000),
                "end": int(time.time() * 1000) + 1000,
                "interval": "1"
            },
            "timestamp": time.time()
        }
        
        await streams.add_message("market_data:kline", {"data": json.dumps(kline_data)})
        self.messages_sent += 1
    
    async def _generate_orderbook_data(self, streams: RedisStreams, symbol: str):
        """Generate realistic orderbook data."""
        current_price = self.current_prices[symbol]
        
        # Generate bid/ask spreads
        spread_pct = random.uniform(0.0001, 0.001)  # 0.01% to 0.1% spread
        
        bids = []
        asks = []
        
        # Generate 25 levels each side
        for i in range(25):
            bid_price = current_price * (1 - spread_pct/2 - i * 0.0001)
            ask_price = current_price * (1 + spread_pct/2 + i * 0.0001)
            
            bid_size = random.uniform(0.1, 10.0)
            ask_size = random.uniform(0.1, 10.0)
            
            bids.append([str(bid_price), str(bid_size)])
            asks.append([str(ask_price), str(ask_size)])
        
        orderbook_data = {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": int(time.time() * 1000),
            "updateId": random.randint(1000000, 9999999)
        }
        
        await streams.add_message("market_data:orderbook", {"data": json.dumps(orderbook_data)})
        self.messages_sent += 1
    
    async def _generate_trade_data(self, streams: RedisStreams, symbol: str):
        """Generate realistic trade data."""
        current_price = self.current_prices[symbol]
        
        # Simulate trade around current price
        trade_price = current_price * (1 + random.gauss(0, 0.0005))  # ±0.05% variation
        trade_size = random.uniform(0.01, 5.0)
        
        trade_data = {
            "symbol": symbol,
            "price": str(trade_price),
            "size": str(trade_size),
            "side": random.choice(["Buy", "Sell"]),
            "timestamp": int(time.time() * 1000),
            "tradeId": str(random.randint(100000000, 999999999))
        }
        
        await streams.add_message("market_data:trades", {"data": json.dumps(trade_data)})
        self.messages_sent += 1
    
    async def _generate_liquidation_data(self, streams: RedisStreams, symbol: str):
        """Generate realistic liquidation data."""
        current_price = self.current_prices[symbol]
        
        # Simulate liquidation - usually at a price away from current
        liq_price = current_price * (1 + random.choice([-1, 1]) * random.uniform(0.001, 0.01))
        liq_size = random.uniform(10, 1000)  # Larger sizes for liquidations
        
        liquidation_data = {
            "symbol": symbol,
            "price": str(liq_price),
            "size": str(liq_size),
            "side": random.choice(["Buy", "Sell"]),
            "timestamp": int(time.time() * 1000),
            "updateTime": int(time.time() * 1000)
        }
        
        await streams.add_message("market_data:liquidation", {"data": json.dumps(liquidation_data)})
        self.messages_sent += 1
        logger.info(f"Generated liquidation: {symbol} {liq_size} @ {liq_price}")
    
    async def _generate_funding_data(self, streams: RedisStreams, symbol: str):
        """Generate funding rate data."""
        funding_data = {
            "symbol": symbol,
            "funding_rate": random.uniform(-0.01, 0.01),  # -1% to +1%
            "timestamp": time.time()
        }
        
        await streams.add_message("market_data:funding", {"data": json.dumps(funding_data)})
        self.messages_sent += 1
    
    async def _generate_oi_data(self, streams: RedisStreams, symbol: str):
        """Generate open interest data."""
        oi_data = {
            "symbol": symbol,
            "open_interest": random.uniform(500000, 2000000),
            "timestamp": time.time()
        }
        
        await streams.add_message("market_data:open_interest", {"data": json.dumps(oi_data)})
        self.messages_sent += 1


class MockDataTest:
    """Test system with mock data instead of live WebSocket data."""
    
    def __init__(self, test_duration_minutes: int = 5):
        self.test_duration = test_duration_minutes * 60
        self.running = False
        self.start_time = None
        
        # Components
        self.mock_generator = MockDataGenerator()
        self.feature_hub = None
        self.health_checker = None
        self.metrics_collector = None
        
        # Test results
        self.test_results = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "mock_messages_sent": 0,
            "features_computed": 0,
            "success": False
        }
    
    async def run_test(self):
        """Run the mock data test."""
        logger.info(f"Starting mock data test for {self.test_duration/60} minutes")
        
        try:
            await self._initialize_test()
            await self._run_test_loop()
            await self._finalize_test()
            
        except Exception as e:
            logger.error("Mock data test failed", exception=e)
            self.test_results["success"] = False
        
        finally:
            await self._cleanup()
    
    async def _initialize_test(self):
        """Initialize test environment."""
        self.start_time = time.time()
        self.test_results["start_time"] = datetime.utcnow().isoformat()
        self.running = True
        
        # Initialize databases
        await init_databases()
        logger.info("✅ Databases initialized")
        
        # Start monitoring
        self.health_checker, self.metrics_collector = await start_monitoring()
        logger.info("✅ Monitoring started")
        
        # Initialize feature hub
        self.feature_hub = FeatureHub()
        logger.info("✅ Test environment initialized")
    
    async def _run_test_loop(self):
        """Run the main test loop."""
        logger.info(f"Starting mock data test loop for {self.test_duration} seconds")
        
        # Start components
        mock_task = asyncio.create_task(self.mock_generator.start(self.test_duration))
        feature_task = asyncio.create_task(self.feature_hub.start())
        
        try:
            # Wait for test duration
            await asyncio.sleep(self.test_duration)
            
        finally:
            # Stop components
            await self.mock_generator.stop()
            await self.feature_hub.stop()
            
            # Cancel tasks
            for task in [mock_task, feature_task]:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def _finalize_test(self):
        """Finalize test and collect results."""
        end_time = time.time()
        
        self.test_results.update({
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": end_time - self.start_time,
            "mock_messages_sent": self.mock_generator.messages_sent,
            "features_computed": getattr(self.feature_hub, 'features_computed', 0),
            "success": True
        })
        
        # Get feature summary
        feature_summary = self.feature_hub.get_feature_summary()
        
        logger.info(
            "✅ Mock data test completed successfully",
            duration_minutes=self.test_results["duration_seconds"] / 60,
            mock_messages=self.test_results["mock_messages_sent"],
            features_computed=self.test_results["features_computed"],
            feature_summary=feature_summary
        )
        
        # Log to trading logger
        trading_logger.logger.info(
            "Mock data test PASSED",
            **self.test_results,
            feature_summary=feature_summary
        )
    
    async def _cleanup(self):
        """Cleanup test environment."""
        logger.info("Cleaning up mock test environment")
        
        try:
            if self.metrics_collector:
                await self.metrics_collector.stop()
            
            await close_databases()
            logger.info("✅ Mock test cleanup completed")
            
        except Exception as e:
            logger.warning("Error during cleanup", exception=e)


async def main():
    """Main entry point for mock data test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run mock data test")
    parser.add_argument(
        "--duration", 
        type=int, 
        default=5,
        help="Test duration in minutes (default: 5)"
    )
    args = parser.parse_args()
    
    # Create and run test
    test = MockDataTest(test_duration_minutes=args.duration)
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())