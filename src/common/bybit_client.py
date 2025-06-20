"""Bybit WebSocket and REST API client with optimized performance."""

import asyncio
import hashlib
import hmac
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set
from urllib.parse import urljoin

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI

from .config import settings
from .config_manager import ConfigManager
from .decorators import with_error_handling, retry_with_backoff
from .performance import profile_performance
from .error_handler import error_context, error_handler
from .exceptions import APIError, ConnectionError, TradingBotError
from .logging import get_logger
from .monitoring import (
    ERRORS_TOTAL,
    MESSAGES_RECEIVED,
    WEBSOCKET_CONNECTIONS,
    WEBSOCKET_RECONNECTS,
    increment_counter,
    set_gauge,
)
from .performance import performance_context
from .types import Symbol
from .utils import safe_float, safe_int, get_utc_timestamp, normalize_symbol

logger = get_logger(__name__)


class BybitWebSocketClient:
    """
    High-performance Bybit WebSocket client with automatic reconnection.
    
    Optimized for:
    - Low latency data processing
    - Efficient memory usage
    - Automatic error recovery
    - Rate limit compliance
    """
    
    def __init__(
        self,
        symbols: List[Symbol],
        on_message: Callable[[Symbol, Dict[str, Any]], None],
        testnet: Optional[bool] = None,
    ):
        # Use settings directly instead of ConfigManager
        from .config import settings
        self.config = settings
        self.symbols = [normalize_symbol(s) for s in symbols]
        self.on_message = on_message
        self.testnet = testnet if testnet is not None else self.config.bybit.testnet
        
        # Connection management
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.running = False
        self.reconnect_count = 0
        self.last_ping = 0.0
        
        # Subscription tracking
        self.subscribed_topics: Set[str] = set()
        self.subscription_queue: List[Dict[str, Any]] = []
        
        # Performance optimization
        self.message_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 100
        self.flush_interval = 0.1  # 100ms
        
        # URLs
        self.ws_url = (
            self.config.bybit.testnet_ws_url if self.testnet 
            else self.config.bybit.ws_url
        )
    
    @profile_performance()
    async def start(self) -> None:
        """Start the WebSocket connection with retry logic."""
        with performance_context("websocket_start"):
            self.running = True
            logger.info(
                "Starting Bybit WebSocket client",
                symbols=self.symbols,
                testnet=self.testnet,
                url=self.ws_url
            )
            
            while self.running:
                with error_context({"symbols": self.symbols, "attempt": self.reconnect_count}):
                    try:
                        await self._connect_and_run()
                    except (ConnectionClosed, InvalidURI) as e:
                        logger.warning(f"WebSocket connection error: {e}")
                        error_handler.handle_error(ConnectionError(str(e)), {
                            "url": self.ws_url,
                            "symbols": self.symbols
                        })
                    except Exception as e:
                        logger.error("WebSocket connection failed", exception=e)
                        increment_counter(ERRORS_TOTAL, component="websocket", error_type=type(e).__name__)
                        
                        error_handler.handle_error(e, {
                            "operation": "websocket_connection",
                            "url": self.ws_url,
                            "symbols": self.symbols
                        })
                    
                    if self.running:
                        self.reconnect_count += 1
                        increment_counter(WEBSOCKET_RECONNECTS, symbol="all")
                        
                        max_attempts = self.config.bybit.max_reconnect_attempts
                        if self.reconnect_count > max_attempts:
                            logger.error(f"Max reconnect attempts ({max_attempts}) reached, stopping")
                            break
                        
                        reconnect_delay = self.config.bybit.reconnect_delay
                        wait_time = min(reconnect_delay * self.reconnect_count, 60)
                        logger.info(f"Reconnecting in {wait_time}s (attempt {self.reconnect_count})")
                        await asyncio.sleep(wait_time)
    
    async def stop(self) -> None:
        """Stop the WebSocket connection gracefully."""
        logger.info("Stopping Bybit WebSocket client")
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        # Update metrics
        for symbol in self.symbols:
            set_gauge(WEBSOCKET_CONNECTIONS, 0, symbol=symbol)
    
    @retry_with_backoff(max_attempts=3)
    async def _connect_and_run(self) -> None:
        """Establish connection and run message processing loop."""
        with performance_context("websocket_connect_and_run"):
            try:
                # Connect with optimized settings
                ping_interval = self.config.bybit.ping_interval
                connection_timeout = self.config.bybit.connection_timeout
                
                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        self.ws_url,
                        ping_interval=ping_interval,
                        ping_timeout=30,
                        max_size=1024 * 1024 * 10,  # 10MB max message size
                        compression=None,  # Disable compression for speed
                        close_timeout=10,
                        open_timeout=30,
                    ),
                    timeout=connection_timeout
                )
                
                logger.info("WebSocket connected successfully")
                self.reconnect_count = 0
                
                # Update connection metrics
                for symbol in self.symbols:
                    set_gauge(WEBSOCKET_CONNECTIONS, 1, symbol=symbol)
                
                # Subscribe to feeds
                await self._subscribe_to_feeds()
                
                # Start message processing tasks
                tasks = [
                    asyncio.create_task(self._message_receiver()),
                    asyncio.create_task(self._ping_sender()),
                    asyncio.create_task(self._buffer_flusher()),
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
            
            except (ConnectionClosed, InvalidURI) as e:
                logger.warning(f"WebSocket connection error: {e}")
                raise
            except Exception as e:
                logger.error("Unexpected error in WebSocket connection", exception=e)
                raise
    
    async def _subscribe_to_feeds(self) -> None:
        """Subscribe to all required data feeds efficiently."""
        # Build subscription requests
        subscriptions = []
        
        for symbol in self.symbols:
            # High-frequency feeds
            subscriptions.extend([
                f"kline.1.{symbol}",         # 1 minute klines (1s not available on v5)
                f"orderbook.50.{symbol}",    # 50-level orderbook  
                f"publicTrade.{symbol}",     # Public trades
            ])
            
            # Liquidation feeds (critical for our strategy)
            subscriptions.append(f"liquidation.{symbol}")
        
        # Send subscription in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(subscriptions), batch_size):
            batch = subscriptions[i:i + batch_size]
            
            subscription_msg = {
                "op": "subscribe",
                "args": batch
            }
            
            await self.websocket.send(json.dumps(subscription_msg))
            logger.info(f"Subscribed to feeds", topics=batch)
            
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(subscriptions):
                await asyncio.sleep(0.1)
        
        self.subscribed_topics.update(subscriptions)
    
    async def _message_receiver(self) -> None:
        """Receive and buffer messages for efficient processing."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON received: {e}")
                    increment_counter(ERRORS_TOTAL, component="websocket", error_type="json_decode")
                
                except Exception as e:
                    logger.error("Error processing message", exception=e, message=message[:200])
                    increment_counter(ERRORS_TOTAL, component="websocket", error_type="message_processing")
        
        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
            raise
        except Exception as e:
            logger.error("Error in message receiver", exception=e)
            raise
    
    async def _process_message(self, data: Dict[str, Any]) -> None:
        """Process incoming WebSocket message efficiently."""
        # Handle system messages
        if "op" in data:
            if data["op"] == "pong":
                return
            elif data["op"] == "subscribe":
                logger.debug("Subscription confirmed", topics=data.get("args", []))
                return
        
        # Handle data messages
        topic = data.get("topic", "")
        if not topic:
            return
        
        # Extract symbol from topic
        symbol = self._extract_symbol_from_topic(topic)
        if not symbol:
            return
        
        # Buffer message for batch processing
        self.message_buffer.append({
            "topic": topic,
            "symbol": symbol,
            "data": data,
            "timestamp": time.time()
        })
        
        # Increment metrics
        increment_counter(MESSAGES_RECEIVED, source="websocket", symbol=symbol)
        
        # Flush buffer if it's getting full
        if len(self.message_buffer) >= self.buffer_size:
            await self._flush_buffer()
    
    async def _buffer_flusher(self) -> None:
        """Periodically flush message buffer for consistent latency."""
        while self.running:
            await asyncio.sleep(self.flush_interval)
            if self.message_buffer:
                await self._flush_buffer()
    
    async def _flush_buffer(self) -> None:
        """Flush buffered messages to the message handler."""
        if not self.message_buffer:
            return
        
        # Process all buffered messages
        buffer_copy = self.message_buffer.copy()
        self.message_buffer.clear()
        
        for msg in buffer_copy:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.on_message, msg["topic"], msg["data"]
                )
            except Exception as e:
                logger.error("Error in message callback", exception=e)
                increment_counter(ERRORS_TOTAL, component="callback", error_type=type(e).__name__)
    
    async def _ping_sender(self) -> None:
        """Send periodic ping messages to keep connection alive."""
        while self.running:
            try:
                await asyncio.sleep(settings.bybit.ping_interval)
                
                if self.websocket and hasattr(self.websocket, 'closed') and not self.websocket.closed:
                    ping_msg = {"op": "ping"}
                    await self.websocket.send(json.dumps(ping_msg))
                    self.last_ping = time.time()
            
            except Exception as e:
                logger.error("Error sending ping", exception=e)
                break
    
    def _extract_symbol_from_topic(self, topic: str) -> Optional[str]:
        """Extract symbol from topic string efficiently."""
        try:
            # Topic formats:
            # kline.1.BTCUSDT
            # orderbook.50.BTCUSDT
            # publicTrade.BTCUSDT
            # liquidation.BTCUSDT
            parts = topic.split(".")
            if len(parts) >= 2:
                symbol = parts[-1]  # Last part is always symbol
                return symbol if symbol in self.symbols else None
        except Exception:
            pass
        return None


class BybitRESTClient:
    """
    Efficient Bybit REST API client with rate limiting and caching.
    
    Optimized for:
    - Rate limit compliance
    - Request caching
    - Automatic retries
    - Cost-effective API usage
    """
    
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        self.base_url = (
            settings.bybit.testnet_url if testnet 
            else settings.bybit.base_url
        )
        
        # API credentials
        self.api_key = settings.bybit.api_key
        self.api_secret = settings.bybit.api_secret
        
        # Debug log API key status
        logger.info(
            f"BybitRESTClient initialized - API Key: {'Set' if self.api_key else 'Not Set'}, "
            f"Secret: {'Set' if self.api_secret else 'Not Set'}, "
            f"Testnet: {testnet}, Base URL: {self.base_url}"
        )
        
        # API key permissions cache
        self._api_permissions_verified = False
        self._api_permissions = None
        
        # Rate limiting
        self.request_semaphore = asyncio.Semaphore(settings.bybit.requests_per_second)
        self.request_times: List[float] = []
        
        # Caching for expensive calls
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 60  # 60 seconds cache TTL
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(
                limit=20,  # Connection pool limit
                limit_per_host=10,
                keepalive_timeout=60,
            )
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def verify_api_permissions(self) -> Dict[str, Any]:
        """Verify API key permissions for trading."""
        if self._api_permissions_verified:
            return self._api_permissions
        
        try:
            url = urljoin(self.base_url, "/v5/user/query-api")
            headers = self._get_auth_headers("GET", "/v5/user/query-api", {})
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("retCode") == 0:
                        result = data.get("result", {})
                        permissions = {
                            "can_trade": "Trade" in result.get("permissions", []),
                            "can_read": "ReadOnly" in result.get("permissions", []),
                            "can_transfer": "Transfer" in result.get("permissions", []),
                            "rate_limit": result.get("ips", [""])[0] if result.get("ips") else None,
                            "expires": result.get("expiredAt", ""),
                            "permissions": result.get("permissions", [])
                        }
                        
                        self._api_permissions = permissions
                        self._api_permissions_verified = True
                        
                        if not permissions["can_trade"]:
                            logger.error("API key does not have trading permissions!")
                        else:
                            logger.info("API key permissions verified", permissions=permissions)
                        
                        return permissions
                    else:
                        logger.error(f"API error verifying permissions: {data.get('retMsg')}")
                else:
                    logger.error(f"HTTP error {response.status} verifying permissions")
        
        except Exception as e:
            logger.error("Error verifying API permissions", exception=e)
        
        return {"can_trade": False, "error": "Failed to verify permissions"}
    
    async def get_open_interest(self, symbols: List[str]) -> Dict[str, float]:
        """Get open interest data with caching."""
        cache_key = f"oi:{','.join(sorted(symbols))}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]
        
        try:
            oi_data = {}
            
            for symbol in symbols:
                async with self.request_semaphore:
                    await self._wait_for_rate_limit()
                    
                    url = urljoin(self.base_url, "/v5/market/open-interest")
                    params = {
                        "category": "linear",
                        "symbol": symbol,
                        "intervalTime": "1h"
                    }
                    
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("retCode") == 0:
                                result = data.get("result", {})
                                oi_list = result.get("list", [])
                                if oi_list:
                                    oi_data[symbol] = float(oi_list[0].get("openInterest", 0))
                            else:
                                logger.warning(f"API error for {symbol}: {data.get('retMsg')}")
                        else:
                            logger.warning(f"HTTP error {response.status} for {symbol}")
            
            # Cache the result
            self.cache[cache_key] = {
                "data": oi_data,
                "timestamp": time.time()
            }
            
            return oi_data
        
        except Exception as e:
            logger.error("Error fetching open interest", exception=e)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
            return {}
    
    async def get_funding_rate(self, symbols: List[str]) -> Dict[str, float]:
        """Get funding rate data with caching."""
        cache_key = f"funding:{','.join(sorted(symbols))}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]
        
        try:
            funding_data = {}
            
            for symbol in symbols:
                async with self.request_semaphore:
                    await self._wait_for_rate_limit()
                    
                    url = urljoin(self.base_url, "/v5/market/funding/history")
                    params = {
                        "category": "linear",
                        "symbol": symbol,
                        "limit": 1
                    }
                    
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("retCode") == 0:
                                result = data.get("result", {})
                                funding_list = result.get("list", [])
                                if funding_list:
                                    funding_data[symbol] = float(funding_list[0].get("fundingRate", 0))
                            else:
                                logger.warning(f"API error for {symbol}: {data.get('retMsg')}")
                        else:
                            logger.warning(f"HTTP error {response.status} for {symbol}")
            
            # Cache the result
            self.cache[cache_key] = {
                "data": funding_data,
                "timestamp": time.time()
            }
            
            return funding_data
        
        except Exception as e:
            logger.error("Error fetching funding rate", exception=e)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
            return {}
    
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker data for a symbol."""
        cache_key = f"ticker:{symbol}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]
        
        try:
            async with self.request_semaphore:
                await self._wait_for_rate_limit()
                
                url = urljoin(self.base_url, "/v5/market/tickers")
                params = {
                    "category": "linear",
                    "symbol": symbol
                }
                
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                            ticker_data = data["result"]["list"][0]
                            
                            # Cache the result
                            self.cache[cache_key] = {
                                "data": ticker_data,
                                "timestamp": time.time()
                            }
                            
                            return ticker_data
                        else:
                            logger.warning(f"No ticker data for {symbol}")
                            return None
                    else:
                        logger.error(f"Ticker API error: {response.status}")
                        return None
        
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}", exception=e)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
            return None
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self.cache:
            return False
        
        entry = self.cache[key]
        return (time.time() - entry["timestamp"]) < self.cache_ttl
    
    async def _wait_for_rate_limit(self) -> None:
        """Ensure we don't exceed rate limits."""
        now = time.time()
        
        # Clean old request times
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Check if we need to wait
        if len(self.request_times) >= settings.bybit.requests_per_minute:
            oldest_time = min(self.request_times)
            wait_time = 60 - (now - oldest_time) + 0.1  # Small buffer
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self.request_times.append(now)
    
    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open positions with proper error handling."""
        try:
            # CRITICAL: Ensure session exists
            if not self.session:
                self.session = aiohttp.ClientSession()
                logger.info("Created new aiohttp session for positions")
            
            async with self.request_semaphore:
                await self._wait_for_rate_limit()
                
                url = urljoin(self.base_url, "/v5/position/list")
                params = {
                    "category": "linear",
                    "settleCoin": "USDT"
                }
                if symbol:
                    params["symbol"] = symbol
                
                # Get auth headers
                headers = self._get_auth_headers("GET", "/v5/position/list", params)
                
                # CRITICAL: Add None check for headers
                if not headers:
                    logger.error("Failed to generate auth headers for positions")
                    return []
                
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # CRITICAL: Check if data is None
                        if data is None:
                            logger.error("Received None response from positions API")
                            return []
                        
                        if data.get("retCode") == 0:
                            result = data.get("result", {})
                            # CRITICAL: Safe navigation for nested dict
                            if isinstance(result, dict):
                                return result.get("list", [])
                            else:
                                logger.error(f"Unexpected result format: {type(result)}")
                                return []
                        else:
                            logger.error(f"API error getting positions: {data.get('retMsg')}")
                    else:
                        logger.error(f"HTTP error {response.status} getting positions")
                        # Log response body for debugging
                        try:
                            error_body = await response.text()
                            logger.error(f"Error response: {error_body}")
                            # Try to parse as JSON for more details
                            try:
                                error_json = json.loads(error_body)
                                logger.error(f"Error details - retCode: {error_json.get('retCode')}, retMsg: {error_json.get('retMsg')}")
                            except:
                                pass
                        except Exception as e:
                            logger.error(f"Failed to read error response: {e}")
                        
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching positions: {e}")
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type="network_error")
        except Exception as e:
            logger.error(f"Unexpected error fetching positions: {e}", exc_info=True)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
        
        return []
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        try:
            async with self.request_semaphore:
                await self._wait_for_rate_limit()
                
                url = urljoin(self.base_url, "/v5/order/realtime")
                params = {
                    "category": "linear",
                    "settleCoin": "USDT"
                }
                if symbol:
                    params["symbol"] = symbol
                
                async with self.session.get(url, params=params, headers=self._get_auth_headers("GET", "/v5/order/realtime", params)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0:
                            return data.get("result", {}).get("list", [])
                        else:
                            logger.error(f"API error getting orders: {data.get('retMsg')}")
                    else:
                        logger.error(f"HTTP error {response.status} getting orders")
                        
        except Exception as e:
            logger.error("Error fetching orders", exception=e)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
        
        return []
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reduce_only: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Create a new order."""
        try:
            async with self.request_semaphore:
                await self._wait_for_rate_limit()
                
                url = urljoin(self.base_url, "/v5/order/create")
                # Map order types to Bybit API format
                order_type_map = {
                    "limit": "Limit",
                    "market": "Market",
                    "limit_maker": "Limit",  # LimitMaker is now just Limit with PostOnly
                    "post_only": "Limit"     # PostOnly is a timeInForce option, not orderType
                }
                
                # Build params in specific order required by Bybit API v5
                # Order matters for signature generation!
                params = {}
                params["category"] = "linear"
                params["symbol"] = symbol
                params["side"] = side.capitalize()  # Buy/Sell
                params["orderType"] = order_type_map.get(order_type.lower(), order_type.capitalize())
                params["qty"] = str(qty)
                
                # Add price for limit orders
                if price and order_type.lower() in ["limit", "post_only", "limit_maker"]:
                    # Format price to remove unnecessary decimal points
                    if price == int(price):
                        params["price"] = str(int(price))
                    else:
                        params["price"] = str(price)
                
                # For post_only and limit_maker orders, set timeInForce
                if order_type.lower() in ["post_only", "limit_maker"]:
                    params["timeInForce"] = "PostOnly"
                
                # Add positionIdx - 0 for one-way mode, 1 for Buy hedge, 2 for Sell hedge
                # Try hedge mode: 1 for Buy side, 2 for Sell side
                if side.lower() == "buy":
                    params["positionIdx"] = 1
                else:
                    params["positionIdx"] = 2
                
                # Add optional parameters in order
                if reduce_only:
                    params["reduceOnly"] = reduce_only
                    
                if stop_loss:
                    params["stopLoss"] = str(stop_loss)
                    
                if take_profit:
                    params["takeProfit"] = str(take_profit)
                
                headers = self._get_auth_headers("POST", "/v5/order/create", params)
                
                # Log for debugging
                logger.info(f"Placing order: {params}")
                
                # Ensure session exists
                if not self.session:
                    self.session = aiohttp.ClientSession()
                    logger.info("Created new aiohttp session for order placement")
                
                async with self.session.post(url, json=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0:
                            logger.info(f"Order created: {data.get('result')}")
                            return data.get("result")
                        else:
                            logger.error(f"API error creating order: {data.get('retMsg')}")
                            # Log the full error response for debugging
                            logger.error(f"Full error response: {data}")
                    else:
                        logger.error(f"HTTP error {response.status} creating order")
                        error_text = await response.text()
                        logger.error(f"Error response: {error_text}")
                        
        except Exception as e:
            logger.error("Error creating order", exception=e)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
        
        return None
    
    async def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status."""
        try:
            async with self.request_semaphore:
                await self._wait_for_rate_limit()
                
                url = urljoin(self.base_url, "/v5/order/realtime")
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "orderId": order_id
                }
                
                headers = self._get_auth_headers("GET", "/v5/order/realtime", params)
                
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                            return data["result"]["list"][0]
                        else:
                            logger.warning(f"No order found: {order_id}")
                    else:
                        logger.error(f"HTTP error {response.status} getting order status")
                        
        except Exception as e:
            logger.error("Error getting order status", exception=e)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
        
        return None
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        try:
            async with self.request_semaphore:
                await self._wait_for_rate_limit()
                
                url = urljoin(self.base_url, "/v5/order/cancel")
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "orderId": order_id
                }
                
                headers = self._get_auth_headers("POST", "/v5/order/cancel", params)
                
                async with self.session.post(url, json=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0:
                            logger.info(f"Order cancelled: {order_id}")
                            return True
                        else:
                            logger.error(f"API error cancelling order: {data.get('retMsg')}")
                    else:
                        logger.error(f"HTTP error {response.status} cancelling order")
                        
        except Exception as e:
            logger.error("Error cancelling order", exception=e)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
        
        return False
    
    async def close_position(self, symbol: str, side: str = None) -> bool:
        """Close a position by creating a reduce-only market order."""
        try:
            # First get the open position
            positions = await self.get_open_positions(symbol)
            if not positions:
                logger.warning(f"No open position found for {symbol}")
                return False
            
            position = positions[0]
            position_side = position.get("side")  # Buy or Sell
            position_size = float(position.get("size", 0))
            
            if position_size == 0:
                logger.warning(f"Position size is 0 for {symbol}")
                return False
            
            # Determine closing side (opposite of position side)
            close_side = "Sell" if position_side == "Buy" else "Buy"
            
            # Create reduce-only market order to close
            result = await self.create_order(
                symbol=symbol,
                side=close_side,
                order_type="Market",
                qty=position_size,
                reduce_only=True
            )
            
            return result is not None
            
        except Exception as e:
            logger.error("Error closing position", exception=e)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
        
        return False
    
    async def set_stop_loss(self, symbol: str, stop_loss: float) -> bool:
        """Set or update stop loss for a position."""
        try:
            async with self.request_semaphore:
                await self._wait_for_rate_limit()
                
                url = urljoin(self.base_url, "/v5/position/trading-stop")
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "stopLoss": str(stop_loss)
                }
                
                headers = self._get_auth_headers("POST", "/v5/position/trading-stop", params)
                
                async with self.session.post(url, json=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0:
                            logger.info(f"Stop loss set for {symbol}: {stop_loss}")
                            return True
                        else:
                            logger.error(f"API error setting stop loss: {data.get('retMsg')}")
                    else:
                        logger.error(f"HTTP error {response.status} setting stop loss")
                        
        except Exception as e:
            logger.error("Error setting stop loss", exception=e)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
        
        return False
    
    async def set_take_profit(self, symbol: str, take_profit: float) -> bool:
        """Set or update take profit for a position."""
        try:
            async with self.request_semaphore:
                await self._wait_for_rate_limit()
                
                url = urljoin(self.base_url, "/v5/position/trading-stop")
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "takeProfit": str(take_profit)
                }
                
                headers = self._get_auth_headers("POST", "/v5/position/trading-stop", params)
                
                async with self.session.post(url, json=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0:
                            logger.info(f"Take profit set for {symbol}: {take_profit}")
                            return True
                        else:
                            logger.error(f"API error setting take profit: {data.get('retMsg')}")
                    else:
                        logger.error(f"HTTP error {response.status} setting take profit")
                        
        except Exception as e:
            logger.error("Error setting take profit", exception=e)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
        
        return False
    
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker information for a symbol."""
        try:
            async with self.request_semaphore:
                await self._wait_for_rate_limit()
                
                url = urljoin(self.base_url, "/v5/market/tickers")
                params = {
                    "category": "linear",
                    "symbol": symbol
                }
                
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0:
                            result = data.get("result", {})
                            if result.get("list"):
                                return result["list"][0]
                        else:
                            logger.error(f"API error getting ticker: {data.get('retMsg')}")
                    else:
                        logger.error(f"HTTP error {response.status} getting ticker")
                        
        except Exception as e:
            logger.error("Error getting ticker", exception=e)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
        
        return None
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        try:
            async with self.request_semaphore:
                await self._wait_for_rate_limit()
                
                url = urljoin(self.base_url, "/v5/position/set-leverage")
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "buyLeverage": str(leverage),
                    "sellLeverage": str(leverage)
                }
                
                headers = self._get_auth_headers("POST", "/v5/position/set-leverage", params)
                
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.post(url, json=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0:
                            logger.info(f"Leverage set for {symbol}: {leverage}x")
                            return True
                        else:
                            logger.error(f"API error setting leverage: {data.get('retMsg')}")
                    else:
                        logger.error(f"HTTP error {response.status} setting leverage")
                        
        except Exception as e:
            logger.error("Error setting leverage", exception=e)
            increment_counter(ERRORS_TOTAL, component="rest_api", error_type=type(e).__name__)
        
        return False
    
    def _get_auth_headers(self, method: str, endpoint: str, params: Dict[str, Any]) -> Dict[str, str]:
        """Generate authentication headers for Bybit API."""
        if not self.api_key or not self.api_secret:
            logger.error(f"Missing API credentials - Key: {bool(self.api_key)}, Secret: {bool(self.api_secret)}")
            return {}
        
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        logger.debug(f"Generating auth headers - Method: {method}, Endpoint: {endpoint}, Timestamp: {timestamp}")
        
        # Create param string based on method
        if method == "GET":
            # For GET requests, create query string
            param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            # The signature payload for GET requests includes the query string
            sign_payload = f"{timestamp}{self.api_key}{recv_window}{param_str}"
        else:
            # For POST requests, use JSON body - order is preserved in Python 3.7+
            # IMPORTANT: Bybit expects spaces after colons in JSON!
            param_str = json.dumps(params, separators=(', ', ': '))
            # The signature payload for POST requests includes the JSON body
            sign_payload = f"{timestamp}{self.api_key}{recv_window}{param_str}"
        
        # Debug logging for all requests
        logger.debug(f"Auth headers for {method} {endpoint}")
        logger.debug(f"Timestamp: {timestamp}")
        logger.debug(f"API Key: {self.api_key[:10] if self.api_key else 'None'}...")
        logger.debug(f"API Secret length: {len(self.api_secret) if self.api_secret else 0}")
        logger.debug(f"Recv Window: {recv_window}")
        logger.debug(f"Param String: {param_str}")
        logger.debug(f"Sign Payload: {sign_payload}")
        
        # Create signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            sign_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        logger.debug(f"Generated signature: {signature[:20]}...")
        
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json"
        }


@asynccontextmanager
async def get_bybit_clients(
    symbols: List[str],
    on_message: Callable[[str, Dict[str, Any]], None],
    testnet: bool = True
) -> AsyncGenerator[tuple[BybitWebSocketClient, BybitRESTClient], None]:
    """Context manager for Bybit clients."""
    ws_client = BybitWebSocketClient(symbols, on_message, testnet)
    
    async with BybitRESTClient(testnet) as rest_client:
        try:
            # Start WebSocket client
            ws_task = asyncio.create_task(ws_client.start())
            
            yield ws_client, rest_client
            
        finally:
            # Cleanup
            await ws_client.stop()
            if not ws_task.done():
                ws_task.cancel()
                try:
                    await ws_task
                except asyncio.CancelledError:
                    pass