"""
Order execution engine for liquidation-driven trading.

High-performance order management with:
- Smart order routing
- Limit order optimization
- Fill monitoring and management
- Retry logic and error handling
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import deque
import numpy as np

from ..common.config import settings
from ..common.logging import get_logger
from ..common.monitoring import (
    ORDERS_PLACED, ORDERS_FILLED, ORDER_LATENCY,
    increment_counter, observe_histogram
)
from ..common.bybit_client import BybitRESTClient

logger = get_logger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PLACED = "placed"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration."""
    LIMIT = "limit"
    MARKET = "market"
    LIMIT_MAKER = "limit_maker"
    POST_ONLY = "post_only"


@dataclass
class Order:
    """Order information."""
    
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    quantity: float
    price: float
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None
    
    # Execution details
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fees: float = 0.0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    placed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Strategy info
    position_id: Optional[str] = None
    strategy: str = "liquidation_cascade"
    
    # Risk parameters
    time_in_force: str = "GTC"  # Good Till Cancelled
    reduce_only: bool = False
    close_on_trigger: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if order is active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.PLACED, OrderStatus.PARTIAL]
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    @property
    def fill_rate(self) -> float:
        """Calculate fill rate."""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0
    
    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity."""
        return self.quantity - self.filled_quantity


@dataclass
class ExecutionConfig:
    """Order execution configuration."""
    
    # Execution parameters
    max_slippage_pct: float = 0.001  # 0.1% max slippage
    price_buffer_pct: float = 0.0005  # 0.05% price buffer for limit orders
    
    # Order management
    max_order_age_seconds: int = 300  # Cancel after 5 minutes
    partial_fill_threshold: float = 0.1  # Cancel if less than 10% filled
    retry_count: int = 3  # Number of retries for failed orders
    retry_delay_seconds: int = 1
    
    # Smart routing
    use_post_only: bool = True  # Use post-only orders when possible
    split_large_orders: bool = True  # Split large orders
    max_order_size_usd: float = 50000.0  # Maximum single order size
    
    # Fill improvement
    price_improvement_check: bool = True  # Check for better prices
    aggressive_fill_timeout: int = 30  # Switch to aggressive after 30s
    
    # Rate limiting
    max_orders_per_second: float = 10.0
    max_orders_per_symbol: int = 5  # Max concurrent orders per symbol
    
    # Monitoring
    latency_threshold_ms: float = 100.0  # Alert on high latency


class OrderExecutor:
    """
    High-performance order execution engine.
    
    Features:
    - Smart order routing and optimization
    - Limit order placement with minimal market impact
    - Fill monitoring and order management
    - Retry logic and error recovery
    """
    
    def __init__(
        self,
        bybit_client: BybitRESTClient,
        config: Optional[ExecutionConfig] = None
    ):
        """
        Initialize order executor.
        
        Args:
            bybit_client: Bybit client instance
            config: Execution configuration
        """
        self.client = bybit_client
        self.config = config or ExecutionConfig()
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: List[Order] = []
        self.order_by_exchange_id: Dict[str, str] = {}  # Exchange ID -> Order ID
        
        # Rate limiting
        self.order_timestamps: deque = deque(maxlen=100)
        self.symbol_order_count: Dict[str, int] = {}
        
        # Performance tracking
        self.execution_times: deque = deque(maxlen=1000)
        self.fill_rates: deque = deque(maxlen=1000)
        
        # Callbacks
        self.on_order_filled: Optional[Callable] = None
        self.on_order_cancelled: Optional[Callable] = None
        self.on_order_rejected: Optional[Callable] = None
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("Order executor initialized", config=self.config.__dict__)
    
    async def start(self) -> None:
        """Start order executor background tasks."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_orders())
        logger.info("Order executor started")
    
    async def stop(self) -> None:
        """Stop order executor."""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active orders
        await self.cancel_all_orders()
        
        logger.info("Order executor stopped")
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.LIMIT,
        position_id: Optional[str] = None,
        reduce_only: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        Place a new order.
        
        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Order price
            order_type: Order type
            position_id: Associated position ID
            reduce_only: Reduce-only order
            metadata: Additional metadata
            
        Returns:
            Order object
        """
        # Check rate limits
        if not await self._check_rate_limits():
            raise RuntimeError("Rate limit exceeded")
        
        # Check symbol order limit
        symbol_count = self.symbol_order_count.get(symbol, 0)
        if symbol_count >= self.config.max_orders_per_symbol:
            raise RuntimeError(f"Max orders per symbol ({self.config.max_orders_per_symbol}) exceeded")
        
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            position_id=position_id,
            reduce_only=reduce_only,
            metadata=metadata or {}
        )
        
        # Split large orders if configured
        if self.config.split_large_orders:
            order_value = quantity * price
            if order_value > self.config.max_order_size_usd:
                return await self._place_split_order(order)
        
        # Place single order
        return await self._place_single_order(order)
    
    async def _place_single_order(self, order: Order) -> Order:
        """Place a single order."""
        start_time = time.perf_counter()
        
        try:
            # Prepare order parameters
            params = self._prepare_order_params(order)
            
            # Place order on exchange
            response = await self._execute_with_retry(
                lambda: self.client.create_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type.value,
                    qty=order.quantity,
                    price=order.price if order.order_type != OrderType.MARKET else None,
                    reduce_only=order.reduce_only
                )
            )
            
            # Update order with exchange response
            order.exchange_order_id = response.get("orderId")
            order.status = OrderStatus.PLACED
            order.placed_at = datetime.now()
            
            # Track order
            self.active_orders[order.order_id] = order
            self.order_by_exchange_id[order.exchange_order_id] = order.order_id
            self.symbol_order_count[order.symbol] = self.symbol_order_count.get(order.symbol, 0) + 1
            
            # Update metrics
            execution_time = (time.perf_counter() - start_time) * 1000  # ms
            self.execution_times.append(execution_time)
            
            observe_histogram(ORDER_LATENCY, execution_time / 1000)  # seconds
            increment_counter(ORDERS_PLACED, symbol=order.symbol)
            
            logger.info("Order placed",
                       order_id=order.order_id,
                       exchange_order_id=order.exchange_order_id,
                       symbol=order.symbol,
                       side=order.side,
                       quantity=order.quantity,
                       price=order.price,
                       execution_time_ms=execution_time)
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.completed_at = datetime.now()
            order.metadata["rejection_reason"] = str(e)
            
            logger.error("Order placement failed",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        exception=e)
            
            if self.on_order_rejected:
                await self.on_order_rejected(order)
            
            raise
    
    async def _place_split_order(self, original_order: Order) -> Order:
        """Split and place large order."""
        max_quantity = self.config.max_order_size_usd / original_order.price
        num_splits = int(np.ceil(original_order.quantity / max_quantity))
        
        logger.info("Splitting large order",
                   order_id=original_order.order_id,
                   original_quantity=original_order.quantity,
                   num_splits=num_splits)
        
        # Create parent order
        parent_order = original_order
        parent_order.metadata["is_parent"] = True
        parent_order.metadata["child_orders"] = []
        
        # Place child orders
        remaining_quantity = original_order.quantity
        child_orders = []
        
        for i in range(num_splits):
            child_quantity = min(max_quantity, remaining_quantity)
            
            # Create child order
            child_order = Order(
                order_id=f"{original_order.order_id}_child_{i}",
                symbol=original_order.symbol,
                side=original_order.side,
                order_type=original_order.order_type,
                quantity=child_quantity,
                price=original_order.price,
                position_id=original_order.position_id,
                reduce_only=original_order.reduce_only,
                strategy=original_order.strategy,
                metadata={
                    "parent_order_id": original_order.order_id,
                    "child_index": i
                }
            )
            
            # Place child order
            try:
                placed_child = await self._place_single_order(child_order)
                child_orders.append(placed_child)
                parent_order.metadata["child_orders"].append(placed_child.order_id)
                
                remaining_quantity -= child_quantity
                
                # Small delay between orders
                if i < num_splits - 1:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error("Failed to place child order",
                            child_order_id=child_order.order_id,
                            exception=e)
        
        # Update parent order status
        if child_orders:
            parent_order.status = OrderStatus.PLACED
            parent_order.placed_at = datetime.now()
        else:
            parent_order.status = OrderStatus.REJECTED
            parent_order.completed_at = datetime.now()
        
        return parent_order
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        if order_id not in self.active_orders:
            logger.warning("Order not found for cancellation", order_id=order_id)
            return False
        
        order = self.active_orders[order_id]
        
        if not order.is_active:
            logger.warning("Order not active", order_id=order_id, status=order.status)
            return False
        
        try:
            # Cancel on exchange
            if order.exchange_order_id:
                await self._execute_with_retry(
                    lambda: self.client.cancel_order(
                        symbol=order.symbol,
                        orderId=order.exchange_order_id
                    )
                )
            
            # Update order status
            order.status = OrderStatus.CANCELLED
            order.completed_at = datetime.now()
            
            # Move to completed
            self._complete_order(order)
            
            logger.info("Order cancelled",
                       order_id=order_id,
                       symbol=order.symbol,
                       filled_quantity=order.filled_quantity,
                       fill_rate=order.fill_rate)
            
            if self.on_order_cancelled:
                await self.on_order_cancelled(order)
            
            return True
            
        except Exception as e:
            logger.error("Order cancellation failed",
                        order_id=order_id,
                        exception=e)
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all active orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Number of orders cancelled
        """
        orders_to_cancel = []
        
        for order_id, order in self.active_orders.items():
            if symbol is None or order.symbol == symbol:
                orders_to_cancel.append(order_id)
        
        cancelled_count = 0
        
        for order_id in orders_to_cancel:
            if await self.cancel_order(order_id):
                cancelled_count += 1
        
        logger.info("Bulk order cancellation",
                   requested=len(orders_to_cancel),
                   cancelled=cancelled_count,
                   symbol=symbol)
        
        return cancelled_count
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status."""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            await self._update_order_status(order)
            return order
        
        # Check completed orders
        for order in self.completed_orders[-100:]:  # Last 100 orders
            if order.order_id == order_id:
                return order
        
        return None
    
    async def _monitor_orders(self) -> None:
        """Background task to monitor active orders."""
        while self._running:
            try:
                await self._check_active_orders()
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error("Error in order monitor", exception=e)
                await asyncio.sleep(5)
    
    async def _check_active_orders(self) -> None:
        """Check and update active orders."""
        current_time = datetime.now()
        orders_to_check = list(self.active_orders.values())
        
        for order in orders_to_check:
            try:
                # Update order status
                await self._update_order_status(order)
                
                # Check for timeout
                if order.placed_at:
                    age_seconds = (current_time - order.placed_at).total_seconds()
                    
                    if age_seconds > self.config.max_order_age_seconds:
                        logger.warning("Order timeout",
                                     order_id=order.order_id,
                                     age_seconds=age_seconds)
                        await self.cancel_order(order.order_id)
                        continue
                    
                    # Check for aggressive fill
                    if (self.config.aggressive_fill_timeout and 
                        age_seconds > self.config.aggressive_fill_timeout and
                        order.fill_rate < self.config.partial_fill_threshold):
                        
                        await self._convert_to_aggressive(order)
                
            except Exception as e:
                logger.error("Error checking order",
                            order_id=order.order_id,
                            exception=e)
    
    async def _update_order_status(self, order: Order) -> None:
        """Update order status from exchange."""
        if not order.exchange_order_id:
            return
        
        try:
            # Get order status from exchange
            status_response = await self.client.get_order_status(
                symbol=order.symbol,
                orderId=order.exchange_order_id
            )
            
            if not status_response:
                return
            
            # Update order fields
            order.updated_at = datetime.now()
            
            # Update fill information
            filled_qty = float(status_response.get("cumExecQty", 0))
            if filled_qty > order.filled_quantity:
                order.filled_quantity = filled_qty
                order.avg_fill_price = float(status_response.get("avgPrice", order.price))
                order.fees = float(status_response.get("cumExecFee", 0))
            
            # Update status
            exchange_status = status_response.get("orderStatus", "").lower()
            
            if exchange_status == "filled":
                order.status = OrderStatus.FILLED
                order.completed_at = datetime.now()
                self._complete_order(order)
                
                if self.on_order_filled:
                    await self.on_order_filled(order)
                    
            elif exchange_status == "partiallyFilled":
                order.status = OrderStatus.PARTIAL
                
            elif exchange_status in ["cancelled", "canceled"]:
                order.status = OrderStatus.CANCELLED
                order.completed_at = datetime.now()
                self._complete_order(order)
                
            elif exchange_status == "rejected":
                order.status = OrderStatus.REJECTED
                order.completed_at = datetime.now()
                self._complete_order(order)
                
        except Exception as e:
            logger.error("Error updating order status",
                        order_id=order.order_id,
                        exception=e)
    
    async def _convert_to_aggressive(self, order: Order) -> None:
        """Convert order to more aggressive pricing."""
        logger.info("Converting to aggressive order",
                   order_id=order.order_id,
                   current_fill_rate=order.fill_rate)
        
        # Cancel current order
        await self.cancel_order(order.order_id)
        
        # Place new order with better price
        if order.side == "buy":
            new_price = order.price * (1 + self.config.price_buffer_pct * 2)
        else:
            new_price = order.price * (1 - self.config.price_buffer_pct * 2)
        
        # Place new order
        await self.place_order(
            symbol=order.symbol,
            side=order.side,
            quantity=order.remaining_quantity,
            price=new_price,
            order_type=OrderType.LIMIT,
            position_id=order.position_id,
            reduce_only=order.reduce_only,
            metadata={
                **order.metadata,
                "original_order_id": order.order_id,
                "is_aggressive": True
            }
        )
    
    def _prepare_order_params(self, order: Order) -> Dict[str, Any]:
        """Prepare order parameters for exchange."""
        # Apply price buffer for limit orders
        execution_price = order.price
        
        if order.order_type == OrderType.LIMIT and self.config.price_buffer_pct > 0:
            if order.side == "buy":
                execution_price = order.price * (1 - self.config.price_buffer_pct)
            else:
                execution_price = order.price * (1 + self.config.price_buffer_pct)
        
        params = {
            "symbol": order.symbol,
            "side": "Buy" if order.side == "buy" else "Sell",
            "orderType": "Limit" if order.order_type in [OrderType.LIMIT, OrderType.POST_ONLY] else "Market",
            "qty": str(order.quantity),
            "price": str(execution_price),
            "timeInForce": order.time_in_force,
            "reduceOnly": order.reduce_only,
            "closeOnTrigger": order.close_on_trigger,
            "orderLinkId": order.order_id
        }
        
        # Add post-only flag
        if order.order_type == OrderType.POST_ONLY or (
            self.config.use_post_only and order.order_type == OrderType.LIMIT
        ):
            params["postOnly"] = True
        
        return params
    
    def _complete_order(self, order: Order) -> None:
        """Move order to completed list."""
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
        
        if order.exchange_order_id in self.order_by_exchange_id:
            del self.order_by_exchange_id[order.exchange_order_id]
        
        if order.symbol in self.symbol_order_count:
            self.symbol_order_count[order.symbol] = max(0, self.symbol_order_count[order.symbol] - 1)
        
        self.completed_orders.append(order)
        
        # Track metrics
        if order.status == OrderStatus.FILLED:
            increment_counter(ORDERS_FILLED, symbol=order.symbol)
            self.fill_rates.append(1.0)
        elif order.status == OrderStatus.PARTIAL and order.fill_rate > 0:
            self.fill_rates.append(order.fill_rate)
        
        # Keep only recent completed orders
        if len(self.completed_orders) > 1000:
            self.completed_orders = self.completed_orders[-500:]
    
    async def _check_rate_limits(self) -> bool:
        """Check if we can place another order."""
        current_time = time.time()
        
        # Clean old timestamps
        cutoff_time = current_time - 1.0  # 1 second window
        while self.order_timestamps and self.order_timestamps[0] < cutoff_time:
            self.order_timestamps.popleft()
        
        # Check rate
        if len(self.order_timestamps) >= self.config.max_orders_per_second:
            return False
        
        # Record new order
        self.order_timestamps.append(current_time)
        return True
    
    async def _execute_with_retry(self, func: Callable, max_retries: Optional[int] = None) -> Any:
        """Execute function with retry logic."""
        max_retries = max_retries or self.config.retry_count
        
        for attempt in range(max_retries):
            try:
                return await func()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                logger.warning(f"Execution failed, retrying",
                             attempt=attempt + 1,
                             max_retries=max_retries,
                             exception=e)
                
                await asyncio.sleep(self.config.retry_delay_seconds * (attempt + 1))
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0
        avg_fill_rate = np.mean(self.fill_rates) if self.fill_rates else 0
        
        return {
            "active_orders": len(self.active_orders),
            "completed_orders": len(self.completed_orders),
            "avg_execution_time_ms": avg_execution_time,
            "avg_fill_rate": avg_fill_rate,
            "orders_per_symbol": dict(self.symbol_order_count)
        }