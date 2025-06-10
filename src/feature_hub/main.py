"""
FeatureHub: Real-time feature engineering service optimized for performance and cost.

Core Design Principles:
- Ultra-low latency feature computation (<5ms)
- Memory-efficient rolling calculations
- Cost-optimized Redis operations
- Scalable for multiple symbols
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from ..common.config import settings
from ..common.database import get_redis_client, RedisStreams
from ..common.logging import get_logger, setup_logging, TradingLogger
from ..common.monitoring import (
    MESSAGES_PROCESSED,
    increment_counter,
    measure_time,
    observe_histogram,
    set_gauge,
    FEATURE_VECTOR_SIZE,
    FEATURE_COMPUTATION_LATENCY,
    FEATURE_HUB_ERRORS,
)

# Setup logging
setup_logging()
logger = get_logger(__name__)
trading_logger = TradingLogger("feature_hub")


class FeatureHub:
    """
    High-performance real-time feature engineering hub.
    
    Optimized for:
    - Sub-5ms feature computation
    - Memory-efficient rolling windows
    - Cost-effective Redis operations
    - ML-ready feature delivery
    """
    
    def __init__(self):
        self.running = False
        self.symbols = settings.bybit.symbols
        
        # Redis connections
        self.redis_client: Optional[object] = None
        self.redis_streams: Optional[RedisStreams] = None
        
        # Feature processors (imported below to avoid circular imports)
        self.micro_liquidity_engine = None
        self.volatility_engine = None
        self.liquidation_engine = None
        self.time_context_engine = None
        
        # Feature cache for ML consumption
        self.feature_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.feature_timestamps: Dict[str, float] = defaultdict(float)
        
        # Performance optimization
        self.feature_update_interval = 1.0  # Update features every second
        self.cache_ttl = 300  # Feature cache TTL: 5 minutes
        self.batch_size = 100
        
        # Statistics
        self.features_computed = 0
        self.last_stats_log = time.time()
        
        # Consumer group configuration
        self.consumer_group = "feature_hub_group"
        self.consumer_name = f"feature_hub_{int(time.time())}"
    
    async def start(self):
        """Start the FeatureHub service with all processors."""
        self.running = True
        logger.info(
            "Starting FeatureHub",
            symbols=self.symbols,
            update_interval=self.feature_update_interval,
            cache_ttl=self.cache_ttl
        )
        
        try:
            # Initialize Redis connection
            self.redis_client = await get_redis_client()
            self.redis_streams = RedisStreams(self.redis_client)
            
            # Initialize feature processors
            await self._initialize_feature_engines()
            
            # Setup consumer groups for data streams
            await self._setup_consumer_groups()
            
            # Start processing tasks
            tasks = [
                asyncio.create_task(self._process_market_data()),
                asyncio.create_task(self._publish_features()),
                asyncio.create_task(self._cleanup_cache()),
                asyncio.create_task(self._log_statistics()),
            ]
            
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
        except Exception as e:
            logger.error("Critical error in FeatureHub", exception=e)
            raise
    
    async def stop(self):
        """Stop the FeatureHub service gracefully."""
        logger.info("Stopping FeatureHub")
        self.running = False
        
        # Final feature publish
        await self._publish_all_features()
        
        logger.info("FeatureHub stopped successfully")
    
    async def _initialize_feature_engines(self):
        """Initialize all feature processing engines."""
        from .micro_liquidity import MicroLiquidityEngine
        from .volatility_momentum import VolatilityMomentumEngine
        from .liquidation_features import LiquidationFeatureEngine
        from .time_context import TimeContextEngine
        
        # Initialize engines with optimized parameters
        self.micro_liquidity_engine = MicroLiquidityEngine()
        self.volatility_engine = VolatilityMomentumEngine()
        self.liquidation_engine = LiquidationFeatureEngine()
        self.time_context_engine = TimeContextEngine()
        
        logger.info("Feature engines initialized successfully")
    
    async def _setup_consumer_groups(self):
        """Setup Redis consumer groups for reliable data processing."""
        data_streams = [
            "market_data:kline",
            "market_data:orderbook", 
            "market_data:trades",
            "market_data:liquidation",
            "market_data:open_interest",
            "market_data:funding"
        ]
        
        for stream in data_streams:
            try:
                await self.redis_streams.create_consumer_group(
                    stream, self.consumer_group, "0"
                )
            except Exception as e:
                logger.debug(f"Consumer group setup for {stream}: {e}")
    
    async def _process_market_data(self):
        """Main data processing loop with consumer groups."""
        while self.running:
            try:
                # Read from all market data streams
                streams = {
                    "market_data:kline": ">",
                    "market_data:orderbook": ">", 
                    "market_data:trades": ">",
                    "market_data:liquidation": ">",
                    "market_data:open_interest": ">",
                    "market_data:funding": ">"
                }
                
                # Read messages with consumer group
                messages = await self.redis_streams.read_group_messages(
                    self.consumer_group,
                    self.consumer_name,
                    streams,
                    count=self.batch_size,
                    block=1000  # 1 second timeout
                )
                
                if messages:
                    await self._process_message_batch(messages)
                
            except Exception as e:
                logger.error("Error in market data processing", exception=e)
                await asyncio.sleep(1)
    
    async def _process_message_batch(self, messages: List[tuple]):
        """Process a batch of messages efficiently."""
        processed_count = 0
        
        for stream_name, stream_messages in messages:
            for message_id, fields in stream_messages:
                try:
                    async with measure_time("feature_computation"):
                        # Handle both bytes and string types
                        stream_name_str = stream_name.decode() if isinstance(stream_name, bytes) else stream_name
                        message_id_str = message_id.decode() if isinstance(message_id, bytes) else message_id
                        
                        await self._process_single_message(
                            stream_name_str, 
                            message_id_str, 
                            fields
                        )
                        processed_count += 1
                
                except Exception as e:
                    logger.warning(f"Error processing message {message_id}", exception=e)
                    continue
                
                # Acknowledge message processing
                try:
                    await self.redis_streams.ack_message(
                        stream_name_str, 
                        self.consumer_group, 
                        message_id_str
                    )
                except Exception as e:
                    logger.warning(f"Error acknowledging message {message_id}", exception=e)
        
        if processed_count > 0:
            self.features_computed += processed_count
            increment_counter(MESSAGES_PROCESSED, component="feature_hub", symbol="all")
    
    async def _process_single_message(
        self, 
        stream_name: str, 
        message_id: str, 
        fields: Dict[str, Any]
    ):
        """Process a single market data message and update features."""
        try:
            # Parse message data
            data = json.loads(fields.get("data", "{}"))
            symbol = data.get("symbol", "")
            
            if not symbol or symbol not in self.symbols:
                return
            
            # Route to appropriate feature engines based on stream type
            if "kline" in stream_name:
                await self._update_kline_features(symbol, data)
            elif "orderbook" in stream_name:
                await self._update_orderbook_features(symbol, data)
            elif "trades" in stream_name:
                await self._update_trade_features(symbol, data)
            elif "liquidation" in stream_name:
                await self._update_liquidation_features(symbol, data)
            elif "open_interest" in stream_name:
                await self._update_oi_features(symbol, data)
            elif "funding" in stream_name:
                await self._update_funding_features(symbol, data)
        
        except Exception as e:
            logger.warning(f"Error processing message from {stream_name}", exception=e)
    
    async def _update_kline_features(self, symbol: str, data: Dict[str, Any]):
        """Update features based on kline data."""
        if self.volatility_engine:
            features = self.volatility_engine.process_kline(symbol, data)
            self._merge_features(symbol, features)
    
    async def _update_orderbook_features(self, symbol: str, data: Dict[str, Any]):
        """Update features based on orderbook data."""
        if self.micro_liquidity_engine:
            features = self.micro_liquidity_engine.process_orderbook(symbol, data)
            self._merge_features(symbol, features)
    
    async def _update_trade_features(self, symbol: str, data: Dict[str, Any]):
        """Update features based on trade data."""
        if self.volatility_engine:
            features = self.volatility_engine.process_trade(symbol, data)
            self._merge_features(symbol, features)
    
    async def _update_liquidation_features(self, symbol: str, data: Dict[str, Any]):
        """Update features based on liquidation data."""
        if self.liquidation_engine:
            features = self.liquidation_engine.process_liquidation(symbol, data)
            self._merge_features(symbol, features)
    
    async def _update_oi_features(self, symbol: str, data: Dict[str, Any]):
        """Update features based on open interest data."""
        # Add open interest to feature cache
        self._merge_features(symbol, {
            "open_interest": data.get("open_interest", 0),
            "oi_timestamp": data.get("timestamp", 0)
        })
    
    async def _update_funding_features(self, symbol: str, data: Dict[str, Any]):
        """Update features based on funding rate data."""
        # Add funding rate to feature cache
        self._merge_features(symbol, {
            "funding_rate": data.get("funding_rate", 0),
            "funding_timestamp": data.get("timestamp", 0)
        })
    
    def _merge_features(self, symbol: str, new_features: Dict[str, float]):
        """Merge new features into the cache efficiently."""
        if not new_features:
            return
        
        current_time = time.time()
        
        # Update feature cache
        self.feature_cache[symbol].update(new_features)
        self.feature_timestamps[symbol] = current_time
        
        # Add time context features
        if self.time_context_engine:
            time_features = self.time_context_engine.get_time_features(current_time)
            self.feature_cache[symbol].update(time_features)
    
    async def _publish_features(self):
        """Periodically publish computed features to Redis."""
        while self.running:
            try:
                await asyncio.sleep(self.feature_update_interval)
                await self._publish_all_features()
            
            except Exception as e:
                logger.error("Error publishing features", exception=e)
                await asyncio.sleep(1)
    
    async def _publish_all_features(self):
        """Publish all cached features to Redis."""
        current_time = time.time()
        
        for symbol in self.symbols:
            if symbol in self.feature_cache:
                features = self.feature_cache[symbol].copy()
                
                if features:
                    # Add metadata
                    features.update({
                        "symbol": symbol,
                        "timestamp": current_time,
                        "feature_count": len(features),
                        "computation_latency": current_time - self.feature_timestamps.get(symbol, current_time)
                    })
                    
                    # Publish to Redis for ML consumption
                    await self.redis_streams.add_message(
                        f"features:{symbol}:latest",
                        features
                    )
                    
                    # Update monitoring metrics
                    FEATURE_VECTOR_SIZE.labels(symbol=symbol).set(len(features))
                    FEATURE_COMPUTATION_LATENCY.labels(symbol=symbol).observe(
                        features["computation_latency"]
                    )
    
    async def _cleanup_cache(self):
        """Periodically cleanup stale feature cache entries."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                
                current_time = time.time()
                stale_symbols = []
                
                for symbol, timestamp in self.feature_timestamps.items():
                    if current_time - timestamp > self.cache_ttl:
                        stale_symbols.append(symbol)
                
                for symbol in stale_symbols:
                    if symbol in self.feature_cache:
                        del self.feature_cache[symbol]
                    if symbol in self.feature_timestamps:
                        del self.feature_timestamps[symbol]
                
                if stale_symbols:
                    logger.debug(f"Cleaned up stale features for symbols: {stale_symbols}")
            
            except Exception as e:
                logger.error("Error in cache cleanup", exception=e)
                await asyncio.sleep(10)
    
    async def _log_statistics(self):
        """Log performance statistics periodically."""
        while self.running:
            await asyncio.sleep(60)  # Log every minute
            
            now = time.time()
            elapsed = now - self.last_stats_log
            
            if elapsed > 0:
                feature_rate = self.features_computed / elapsed
                
                logger.info(
                    "FeatureHub performance stats",
                    feature_rate=f"{feature_rate:.1f} features/s",
                    total_features=self.features_computed,
                    cached_symbols=len(self.feature_cache),
                    cache_sizes={symbol: len(features) for symbol, features in self.feature_cache.items()},
                    elapsed_seconds=elapsed
                )
                
                # Reset counters
                self.features_computed = 0
                self.last_stats_log = now
    
    def get_latest_features(self, symbol: str) -> Dict[str, float]:
        """Get latest features for a symbol (API endpoint)."""
        if symbol in self.feature_cache:
            features = self.feature_cache[symbol].copy()
            features["cache_age"] = time.time() - self.feature_timestamps.get(symbol, time.time())
            return features
        return {}
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of feature computation status."""
        return {
            "symbols": list(self.feature_cache.keys()),
            "total_symbols": len(self.feature_cache),
            "feature_counts": {
                symbol: len(features) 
                for symbol, features in self.feature_cache.items()
            },
            "cache_ages": {
                symbol: time.time() - self.feature_timestamps.get(symbol, time.time())
                for symbol in self.feature_cache.keys()
            },
            "running": self.running,
            "update_interval": self.feature_update_interval
        }


async def start_feature_hub():
    """Start the FeatureHub service."""
    feature_hub = FeatureHub()
    try:
        await feature_hub.start()
    except Exception as e:
        logger.error("FeatureHub service failed", exception=e)
        raise
    finally:
        await feature_hub.stop()


async def main():
    """Main entry point for standalone FeatureHub service."""
    await start_feature_hub()


if __name__ == "__main__":
    asyncio.run(main())