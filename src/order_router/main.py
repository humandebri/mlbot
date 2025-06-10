"""
Order Router main entry point.

Coordinates all order routing components:
- Risk management
- Position tracking
- Order execution
- Smart routing
"""

import asyncio
import signal
import sys
from typing import Optional, Dict, Any
from datetime import datetime

from .risk_manager import RiskManager, RiskConfig
from .position_manager import PositionManager
from .order_executor import OrderExecutor, ExecutionConfig
from .smart_router import SmartRouter, TradingSignal, RoutingConfig
from ..common.config import settings
from ..common.logging import get_logger
from ..common.bybit_client import BybitClient
from ..common.monitoring import setup_metrics_server

logger = get_logger(__name__)


class OrderRouter:
    """
    Main order routing system coordinating all components.
    
    Integrates:
    - Risk management
    - Position tracking
    - Order execution
    - Smart routing logic
    """
    
    def __init__(self):
        """Initialize order router."""
        
        # Initialize Bybit client
        self.bybit_client = BybitClient(
            api_key=settings.BYBIT_API_KEY,
            api_secret=settings.BYBIT_API_SECRET,
            testnet=settings.USE_TESTNET
        )
        
        # Initialize components
        self.risk_manager = RiskManager(RiskConfig())
        self.position_manager = PositionManager()
        self.order_executor = OrderExecutor(
            self.bybit_client,
            ExecutionConfig()
        )
        self.smart_router = SmartRouter(
            self.order_executor,
            self.risk_manager,
            self.position_manager,
            RoutingConfig()
        )
        
        # Setup callbacks
        self._setup_callbacks()
        
        # State
        self.running = False
        self._tasks = []
        
        logger.info("Order router initialized")
    
    def _setup_callbacks(self) -> None:
        """Setup component callbacks."""
        
        # Order executor callbacks
        self.order_executor.on_order_filled = self._on_order_filled
        self.order_executor.on_order_cancelled = self._on_order_cancelled
        self.order_executor.on_order_rejected = self._on_order_rejected
    
    async def _on_order_filled(self, order) -> None:
        """Handle filled order."""
        logger.info("Order filled callback",
                   order_id=order.order_id,
                   symbol=order.symbol,
                   quantity=order.filled_quantity,
                   price=order.avg_fill_price)
        
        # Update position with fill
        if order.position_id:
            await self.position_manager.update_position(
                position_id=order.position_id,
                current_price=order.avg_fill_price,
                fees=order.fees
            )
        
        # Update risk manager trade counts
        self.risk_manager.update_trade_counts()
    
    async def _on_order_cancelled(self, order) -> None:
        """Handle cancelled order."""
        logger.info("Order cancelled callback",
                   order_id=order.order_id,
                   symbol=order.symbol,
                   filled_quantity=order.filled_quantity)
        
        # Handle partial fills
        if order.filled_quantity > 0 and order.position_id:
            await self.position_manager.update_position(
                position_id=order.position_id,
                current_price=order.avg_fill_price,
                fees=order.fees
            )
    
    async def _on_order_rejected(self, order) -> None:
        """Handle rejected order."""
        logger.warning("Order rejected callback",
                      order_id=order.order_id,
                      symbol=order.symbol,
                      reason=order.metadata.get("rejection_reason"))
        
        # Close position if order was rejected
        if order.position_id:
            positions = await self.position_manager.get_all_positions()
            position = next((p for p in positions if p["position_id"] == order.position_id), None)
            
            if position and position["quantity"] == 0:
                await self.position_manager.close_position(
                    position_id=order.position_id,
                    exit_price=order.price,
                    reason="order_rejected"
                )
    
    async def start(self) -> None:
        """Start order router."""
        if self.running:
            logger.warning("Order router already running")
            return
        
        logger.info("Starting order router")
        self.running = True
        
        # Start components
        await self.order_executor.start()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._monitor_positions()),
            asyncio.create_task(self._check_risk_limits()),
            asyncio.create_task(self._report_stats())
        ]
        
        logger.info("Order router started")
    
    async def stop(self) -> None:
        """Stop order router."""
        if not self.running:
            return
        
        logger.info("Stopping order router")
        self.running = False
        
        # Cancel all active orders
        cancelled_count = await self.order_executor.cancel_all_orders()
        logger.info(f"Cancelled {cancelled_count} active orders")
        
        # Stop components
        await self.order_executor.stop()
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close all positions (optional - depends on strategy)
        # await self._close_all_positions()
        
        logger.info("Order router stopped")
    
    async def process_signal(self, signal: TradingSignal) -> Optional[str]:
        """
        Process trading signal.
        
        Args:
            signal: Trading signal to process
            
        Returns:
            Position ID if executed
        """
        if not self.running:
            logger.warning("Order router not running")
            return None
        
        try:
            # Route signal through smart router
            position_id = await self.smart_router.process_signal(signal)
            
            if position_id:
                logger.info("Signal processed successfully",
                           position_id=position_id,
                           symbol=signal.symbol,
                           prediction=signal.prediction,
                           confidence=signal.confidence)
            
            return position_id
            
        except Exception as e:
            logger.error("Error processing signal",
                        symbol=signal.symbol,
                        exception=e)
            return None
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        # Get component stats
        risk_metrics = self.risk_manager.get_risk_metrics()
        performance_stats = await self.position_manager.get_performance_stats()
        execution_stats = self.order_executor.get_execution_stats()
        routing_stats = self.smart_router.get_routing_stats()
        
        # Get positions
        positions = await self.position_manager.get_all_positions()
        total_pnl = await self.position_manager.calculate_total_pnl()
        
        return {
            "status": "running" if self.running else "stopped",
            "timestamp": datetime.now(),
            "risk": risk_metrics,
            "performance": {
                "total_trades": performance_stats.total_trades,
                "win_rate": performance_stats.win_rate,
                "profit_factor": performance_stats.profit_factor,
                "sharpe_ratio": performance_stats.sharpe_ratio,
                "total_pnl": total_pnl["total_pnl"],
                "realized_pnl": total_pnl["realized_pnl"],
                "unrealized_pnl": total_pnl["unrealized_pnl"]
            },
            "execution": execution_stats,
            "routing": routing_stats,
            "active_positions": len(positions),
            "positions": [
                {
                    "symbol": p["symbol"],
                    "side": p["side"],
                    "quantity": p["quantity"],
                    "entry_price": p["entry_price"],
                    "current_price": p["current_price"],
                    "unrealized_pnl": p["unrealized_pnl"],
                    "pnl_pct": (p["unrealized_pnl"] / (p["quantity"] * p["entry_price"]) * 100)
                    if p["quantity"] * p["entry_price"] > 0 else 0
                }
                for p in positions
            ]
        }
    
    async def _monitor_positions(self) -> None:
        """Monitor and update positions."""
        while self.running:
            try:
                positions = await self.position_manager.get_all_positions()
                
                for position in positions:
                    # Get current price (would typically fetch from market data)
                    current_price = await self._get_current_price(position["symbol"])
                    
                    if current_price:
                        # Update position
                        await self.position_manager.update_position(
                            position_id=position["position_id"],
                            current_price=current_price
                        )
                        
                        # Update in risk manager
                        await self.risk_manager.update_position(
                            position_id=position["position_id"],
                            current_price=current_price
                        )
                        
                        # Check for exit conditions
                        if position["unrealized_pnl"] < -1000:  # Example stop loss
                            logger.warning("Stop loss triggered",
                                         position_id=position["position_id"],
                                         unrealized_pnl=position["unrealized_pnl"])
                            
                            # Close position
                            await self._close_position(position["position_id"], current_price)
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error("Error monitoring positions", exception=e)
                await asyncio.sleep(5)
    
    async def _check_risk_limits(self) -> None:
        """Monitor risk limits and circuit breakers."""
        while self.running:
            try:
                risk_metrics = self.risk_manager.get_risk_metrics()
                
                # Log risk status
                if risk_metrics["trading_halted"]:
                    logger.critical("Trading halted by risk manager",
                                   reason=risk_metrics["halt_reason"])
                
                # Check drawdown
                if risk_metrics["current_drawdown"] > 0.15:  # 15% drawdown warning
                    logger.warning("High drawdown detected",
                                  current_drawdown=risk_metrics["current_drawdown"],
                                  max_drawdown=risk_metrics["max_drawdown"])
                
                # Check daily loss
                if risk_metrics["daily_pnl"] < -5000:  # $5k daily loss warning
                    logger.warning("Significant daily loss",
                                  daily_pnl=risk_metrics["daily_pnl"])
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error("Error checking risk limits", exception=e)
                await asyncio.sleep(10)
    
    async def _report_stats(self) -> None:
        """Periodically report system statistics."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                
                status = await self.get_status()
                
                logger.info("System status report",
                           active_positions=status["active_positions"],
                           total_pnl=status["performance"]["total_pnl"],
                           win_rate=status["performance"]["win_rate"],
                           execution_rate=status["routing"]["execution_rate"],
                           avg_fill_time_ms=status["execution"]["avg_execution_time_ms"])
                
            except Exception as e:
                logger.error("Error reporting stats", exception=e)
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        try:
            # This would typically fetch from market data service
            # For now, return mock price
            return 50000.0  # Mock BTC price
        except Exception as e:
            logger.error("Error getting current price",
                        symbol=symbol,
                        exception=e)
            return None
    
    async def _close_position(self, position_id: str, exit_price: float) -> None:
        """Close a position."""
        try:
            # Cancel any open orders for this position
            orders_cancelled = 0
            for order_id, order in self.order_executor.active_orders.items():
                if order.position_id == position_id:
                    if await self.order_executor.cancel_order(order_id):
                        orders_cancelled += 1
            
            if orders_cancelled > 0:
                logger.info(f"Cancelled {orders_cancelled} orders for position",
                           position_id=position_id)
            
            # Close position
            close_result = await self.position_manager.close_position(
                position_id=position_id,
                exit_price=exit_price,
                reason="risk_management"
            )
            
            if close_result:
                # Close in risk manager
                await self.risk_manager.close_position(
                    position_id=position_id,
                    exit_price=exit_price
                )
                
                logger.info("Position closed",
                           position_id=position_id,
                           exit_price=exit_price,
                           pnl=close_result["total_pnl"])
                
        except Exception as e:
            logger.error("Error closing position",
                        position_id=position_id,
                        exception=e)


async def main():
    """Main entry point."""
    
    # Setup metrics server
    setup_metrics_server(port=9093)
    
    # Create order router
    router = OrderRouter()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(shutdown(router))
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start router
        await router.start()
        
        logger.info("Order router running. Press Ctrl+C to stop.")
        
        # Keep running
        while router.running:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error("Fatal error in order router", exception=e)
    finally:
        await shutdown(router)


async def shutdown(router: OrderRouter):
    """Graceful shutdown."""
    logger.info("Shutting down order router...")
    await router.stop()
    logger.info("Order router shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())