"""Main entry point for the WebSocket data ingestor service."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from ..common.bybit_client import get_bybit_clients
from ..common.config import settings
from ..common.database import get_redis_client, RedisStreams
from ..common.logging import get_logger, setup_logging, TradingLogger
from ..common.monitoring import (
    MESSAGES_PROCESSED,
    increment_counter,
    measure_time,
    observe_histogram,
    BATCH_FLUSH_SIZE,
    REDIS_WRITE_ERRORS,
    INGESTOR_ERRORS,
)
from .liquidation_processor import LiquidationEvent, LiquidationSpikeDetector
from .data_archiver import DataArchiver

# Setup logging
setup_logging()
logger = get_logger(__name__)
trading_logger = TradingLogger("ingestor")


class BybitIngestor:
    """
    High-performance Bybit WebSocket data ingestor.
    
    Optimized for:
    - Low-latency data streaming
    - Efficient Redis Streams publishing
    - Memory-conscious data processing
    - Cost-effective operation
    """
    
    def __init__(self):
        self.running = False
        self.symbols = settings.bybit.symbols
        self.testnet = settings.bybit.testnet
        
        # Data processors
        self.liquidation_detector = LiquidationSpikeDetector(
            window_seconds=300,  # 5-minute rolling window
            spike_threshold=2.0  # 2-sigma threshold
        )
        
        # Data archiver
        self.data_archiver = DataArchiver()
        
        # Redis client
        self.redis_client: Optional[object] = None
        self.redis_streams: Optional[RedisStreams] = None
        
        # Performance optimization
        self.batch_buffer: Dict[str, List[Dict[str, Any]]] = {
            "kline": [],
            "orderbook": [],
            "trades": [],
            "liquidation": [],
        }
        self.batch_size = 50
        self.batch_timeout = 0.1  # 100ms
        self.last_flush = time.time()
        
        # Statistics
        self.message_count = 0
        self.last_stats_log = time.time()
        self.stats_interval = 60  # Log stats every 60 seconds
    
    async def start(self):
        """Start the ingestor service with all optimizations."""
        self.running = True
        logger.info(
            "Starting Bybit Ingestor",
            symbols=self.symbols,
            testnet=self.testnet,
            batch_size=self.batch_size
        )
        
        try:
            # Initialize Redis connection
            self.redis_client = await get_redis_client()
            self.redis_streams = RedisStreams(self.redis_client)
            
            # Start Bybit clients and message processing
            async with get_bybit_clients(
                symbols=self.symbols,
                on_message=self._handle_message,
                testnet=self.testnet
            ) as (ws_client, rest_client):
                
                # Start batch flusher task
                flush_task = asyncio.create_task(self._batch_flusher())
                
                # Start periodic REST data collection
                rest_task = asyncio.create_task(self._collect_rest_data(rest_client))
                
                # Start statistics logging
                stats_task = asyncio.create_task(self._log_statistics())
                
                # Start data archiver
                archiver_task = asyncio.create_task(self.data_archiver.start())
                
                try:
                    # Keep running until stopped
                    while self.running:
                        await asyncio.sleep(1)
                
                finally:
                    # Cleanup tasks
                    for task in [flush_task, rest_task, stats_task, archiver_task]:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
        
        except Exception as e:
            logger.error("Critical error in ingestor", exception=e)
            raise
    
    async def stop(self):
        """Stop the ingestor service gracefully."""
        logger.info("Stopping Bybit Ingestor")
        self.running = False
        
        # Stop data archiver
        await self.data_archiver.stop()
        
        # Flush remaining batches
        await self._flush_all_batches()
        
        logger.info("Bybit Ingestor stopped successfully")
    
    def _handle_message(self, topic: str, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message with efficient routing."""
        try:
            with measure_time("message_processing"):
                self._route_message(topic, data)
                self.message_count += 1
                increment_counter(MESSAGES_PROCESSED, component="ingestor", symbol="all")
        
        except Exception as e:
            logger.error("Error handling message", exception=e, topic=topic)
            INGESTOR_ERRORS.labels(error_type=type(e).__name__).inc()
    
    def _route_message(self, topic: str, data: Dict[str, Any]) -> None:
        """Route message to appropriate processor based on topic."""
        # Extract message type from topic
        if "kline" in topic:
            self._process_kline_message(topic, data)
        elif "orderbook" in topic:
            self._process_orderbook_message(topic, data)
        elif "trades" in topic:
            self._process_trades_message(topic, data)
        elif "allLiquidation" in topic:
            self._process_liquidation_message(topic, data)
        else:
            logger.debug(f"Unhandled topic: {topic}")
    
    def _process_kline_message(self, topic: str, data: Dict[str, Any]) -> None:
        """Process kline (candlestick) data efficiently."""
        try:
            kline_data = data.get("data", [])
            if not kline_data:
                return
            
            # Process each kline in the message
            for kline in kline_data:
                processed_data = {
                    "timestamp": time.time(),
                    "topic": topic,
                    "symbol": kline.get("symbol", ""),
                    "interval": kline.get("interval", ""),
                    "open_time": kline.get("start", 0),
                    "close_time": kline.get("end", 0),
                    "open": float(kline.get("open", 0)),
                    "high": float(kline.get("high", 0)),
                    "low": float(kline.get("low", 0)),
                    "close": float(kline.get("close", 0)),
                    "volume": float(kline.get("volume", 0)),
                    "turnover": float(kline.get("turnover", 0)),
                    "confirm": kline.get("confirm", False),
                }
                
                self.batch_buffer["kline"].append(processed_data)
        
        except Exception as e:
            logger.warning("Error processing kline message", exception=e)
    
    def _process_orderbook_message(self, topic: str, data: Dict[str, Any]) -> None:
        """Process orderbook data with depth analysis."""
        try:
            ob_data = data.get("data", {})
            if not ob_data:
                return
            
            symbol = ob_data.get("s", "")
            bids = ob_data.get("b", [])
            asks = ob_data.get("a", [])
            
            # Calculate depth metrics efficiently
            bid_depth = sum(float(bid[1]) for bid in bids[:5])  # Top 5 levels
            ask_depth = sum(float(ask[1]) for ask in asks[:5])
            
            spread = 0.0
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                spread = best_ask - best_bid
            
            processed_data = {
                "timestamp": time.time(),
                "topic": topic,
                "symbol": symbol,
                "update_id": ob_data.get("u", 0),
                "bids_top5": bids[:5],
                "asks_top5": asks[:5],
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "depth_ratio": bid_depth / (ask_depth + bid_depth) if (bid_depth + ask_depth) > 0 else 0.5,
                "spread": spread,
                "mid_price": (float(bids[0][0]) + float(asks[0][0])) / 2 if bids and asks else 0,
            }
            
            self.batch_buffer["orderbook"].append(processed_data)
        
        except Exception as e:
            logger.warning("Error processing orderbook message", exception=e)
    
    def _process_trades_message(self, topic: str, data: Dict[str, Any]) -> None:
        """Process trade data for flow analysis."""
        try:
            trades_data = data.get("data", [])
            if not trades_data:
                return
            
            for trade in trades_data:
                processed_data = {
                    "timestamp": time.time(),
                    "topic": topic,
                    "symbol": trade.get("s", ""),
                    "trade_id": trade.get("i", ""),
                    "price": float(trade.get("p", 0)),
                    "size": float(trade.get("v", 0)),
                    "side": trade.get("S", ""),
                    "trade_time": trade.get("T", 0),
                    "is_block_trade": trade.get("BT", False),
                }
                
                self.batch_buffer["trades"].append(processed_data)
        
        except Exception as e:
            logger.warning("Error processing trades message", exception=e)
    
    def _process_liquidation_message(self, topic: str, data: Dict[str, Any]) -> None:
        """Process liquidation data with spike detection."""
        try:
            liq_data = data.get("data", {})
            if not liq_data:
                return
            
            # Create liquidation event
            liq_event = LiquidationEvent.from_bybit_data(liq_data)
            if not liq_event:
                return
            
            # Run spike detection
            spike_analysis = self.liquidation_detector.process_liquidation(liq_event)
            
            processed_data = {
                "timestamp": time.time(),
                "topic": topic,
                "symbol": liq_event.symbol,
                "side": liq_event.side,
                "size": liq_event.size,
                "price": liq_event.price,
                "liquidation_time": liq_event.timestamp,
                
                # Spike analysis results
                "spike_detected": spike_analysis["spike_detected"],
                "spike_severity": spike_analysis["spike_severity"],
                "spike_type": spike_analysis["spike_info"]["type"],
                "liq_metrics": spike_analysis["metrics"],
            }
            
            self.batch_buffer["liquidation"].append(processed_data)
            
            # Log significant spikes
            if spike_analysis["spike_detected"] and spike_analysis["spike_severity"] > 3.0:
                trading_logger.logger.warning(
                    "Significant liquidation spike detected",
                    symbol=liq_event.symbol,
                    severity=spike_analysis["spike_severity"],
                    spike_type=spike_analysis["spike_info"]["type"],
                    size=liq_event.size,
                    side=liq_event.side
                )
        
        except Exception as e:
            logger.warning("Error processing liquidation message", exception=e)
    
    async def _batch_flusher(self) -> None:
        """Periodically flush batched data to Redis Streams."""
        while self.running:
            try:
                await asyncio.sleep(self.batch_timeout)
                
                # Check if it's time to flush
                now = time.time()
                should_flush = (
                    now - self.last_flush >= self.batch_timeout or
                    any(len(batch) >= self.batch_size for batch in self.batch_buffer.values())
                )
                
                if should_flush:
                    await self._flush_all_batches()
                    self.last_flush = now
            
            except Exception as e:
                logger.error("Error in batch flusher", exception=e)
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def _flush_all_batches(self) -> None:
        """Flush all batched data to Redis Streams efficiently."""
        for data_type, batch in self.batch_buffer.items():
            if batch:
                await self._flush_batch(data_type, batch)
                batch.clear()
    
    async def _flush_batch(self, data_type: str, batch: List[Dict[str, Any]]) -> None:
        """Flush a single batch to Redis Stream."""
        if not batch:
            return
        
        try:
            stream_name = f"market_data:{data_type}"
            
            # Use pipeline for efficient batch writes
            pipe = self.redis_client.pipeline()
            
            for item in batch:
                # Convert to JSON for Redis
                message_data = {"data": json.dumps(item)}
                pipe.xadd(stream_name, message_data, maxlen=settings.redis.stream_maxlen)
            
            await pipe.execute()
            
            BATCH_FLUSH_SIZE.labels(data_type=data_type).observe(len(batch))
            logger.debug(f"Flushed {len(batch)} {data_type} messages to Redis")
        
        except Exception as e:
            logger.error(f"Error flushing {data_type} batch", exception=e)
            REDIS_WRITE_ERRORS.labels(data_type=data_type).inc()
    
    async def _collect_rest_data(self, rest_client) -> None:
        """Periodically collect data via REST API."""
        while self.running:
            try:
                # Collect open interest data every 5 minutes
                oi_data = await rest_client.get_open_interest(self.symbols)
                if oi_data:
                    for symbol, oi_value in oi_data.items():
                        await self.redis_streams.add_message(
                            f"market_data:open_interest",
                            {
                                "timestamp": time.time(),
                                "symbol": symbol,
                                "open_interest": oi_value,
                                "data_type": "open_interest"
                            }
                        )
                
                # Collect funding rate data every hour
                funding_data = await rest_client.get_funding_rate(self.symbols)
                if funding_data:
                    for symbol, funding_rate in funding_data.items():
                        await self.redis_streams.add_message(
                            f"market_data:funding",
                            {
                                "timestamp": time.time(),
                                "symbol": symbol,
                                "funding_rate": funding_rate,
                                "data_type": "funding_rate"
                            }
                        )
                
                # Wait 5 minutes before next collection
                await asyncio.sleep(300)
            
            except Exception as e:
                logger.error("Error collecting REST data", exception=e)
                await asyncio.sleep(60)  # Shorter retry interval
    
    async def _log_statistics(self) -> None:
        """Log performance statistics periodically."""
        while self.running:
            await asyncio.sleep(self.stats_interval)
            
            now = time.time()
            elapsed = now - self.last_stats_log
            
            if elapsed > 0:
                msg_rate = self.message_count / elapsed
                
                logger.info(
                    "Ingestor performance stats",
                    message_rate=f"{msg_rate:.1f} msg/s",
                    total_messages=self.message_count,
                    batch_sizes={k: len(v) for k, v in self.batch_buffer.items()},
                    elapsed_seconds=elapsed
                )
                
                # Reset counters
                self.message_count = 0
                self.last_stats_log = now


async def start_ingestor():
    """Start the ingestor service."""
    ingestor = BybitIngestor()
    try:
        await ingestor.start()
    except Exception as e:
        logger.error("Ingestor service failed", exception=e)
        raise
    finally:
        await ingestor.stop()


async def main():
    """Main entry point for standalone ingestor service."""
    await start_ingestor()


if __name__ == "__main__":
    asyncio.run(main())