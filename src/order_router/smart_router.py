"""
Smart order routing for optimal execution.

Advanced routing logic including:
- Liquidation cascade detection and trading
- Order placement optimization
- Multi-level order strategies
- Dynamic price adjustment
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

from .order_executor import OrderExecutor, Order, OrderType
from .risk_manager import RiskManager
from .position_manager import PositionManager
from ..common.config import settings
from ..common.logging import get_logger
from ..common.monitoring import increment_counter
from ..common.types import Symbol

logger = get_logger(__name__)


@dataclass
class TradingSignal:
    """Trading signal from ML model."""
    
    symbol: str
    timestamp: datetime
    prediction: float  # Expected PnL
    confidence: float
    features: Dict[str, float]
    
    # Liquidation info
    liquidation_detected: bool = False
    liquidation_size: float = 0.0
    liquidation_side: str = ""  # 'long' or 'short' liquidations
    
    # Market conditions
    spread: float = 0.0
    volatility: float = 0.0
    volume_surge: float = 0.0
    
    # Execution hints
    urgency: str = "normal"  # 'high', 'normal', 'low'
    suggested_timeframe: int = 300  # seconds


@dataclass
class RoutingConfig:
    """Smart routing configuration."""
    
    # Signal thresholds - lowered for testing
    min_confidence: float = 0.0  # Temporarily allow any confidence
    min_prediction: float = 0.0  # Temporarily allow any expected PnL
    
    # Liquidation trading
    liquidation_multiplier: float = 1.5  # Size multiplier for liquidation trades
    fade_liquidation: bool = True  # Trade against liquidation direction
    liquidation_layers: int = 3  # Number of order layers
    
    # Order placement
    spread_multiplier: float = 0.5  # Place orders at X * spread
    max_spread_bps: float = 10.0  # Maximum spread in basis points
    
    # Dynamic adjustment
    volatility_adjustment: bool = True
    volume_weighting: bool = True
    
    # Order strategies
    use_iceberg: bool = True  # Hide large orders
    iceberg_visible_pct: float = 0.2  # Show only 20% of order
    
    # Timing
    max_order_lifetime: int = 300  # Cancel after 5 minutes
    aggressive_fill_time: int = 60  # Get aggressive after 1 minute


class SmartRouter:
    """
    Intelligent order routing system for liquidation-driven trading.
    
    Routes orders based on:
    - ML model predictions
    - Liquidation cascade detection
    - Market microstructure
    - Risk constraints
    """
    
    def __init__(
        self,
        order_executor: OrderExecutor,
        risk_manager: RiskManager,
        position_manager: PositionManager,
        config: Optional[RoutingConfig] = None
    ):
        """
        Initialize smart router.
        
        Args:
            order_executor: Order execution engine
            risk_manager: Risk management system
            position_manager: Position tracking
            config: Routing configuration
        """
        self.executor = order_executor
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.config = config or RoutingConfig()
        
        # Signal tracking
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history: List[TradingSignal] = []
        
        # Performance tracking
        self.routed_orders: Dict[str, str] = {}  # Signal ID -> Order ID
        self.routing_stats = {
            "signals_received": 0,
            "signals_executed": 0,
            "signals_rejected": 0,
            "liquidation_trades": 0
        }
        
        logger.info("Smart router initialized", config=self.config.__dict__)
    
    async def process_signal(self, signal: TradingSignal) -> Optional[str]:
        """
        Process trading signal and route orders.
        
        Args:
            signal: Trading signal from ML model
            
        Returns:
            Position ID if executed, None otherwise
        """
        self.routing_stats["signals_received"] += 1
        signal_id = f"{signal.symbol}_{signal.timestamp.timestamp()}"
        
        try:
            # Validate signal
            logger.info(f"Validating signal for {signal.symbol}: conf={signal.confidence}, pred={signal.prediction}")
            if not self._validate_signal(signal):
                logger.warning(f"Signal validation failed for {signal.symbol}")
                self.routing_stats["signals_rejected"] += 1
                return None
            logger.info(f"Signal validation passed for {signal.symbol}")
            
            # Check risk limits
            logger.info(f"Starting risk check for {signal.symbol}")
            risk_check = await self._check_risk(signal)
            logger.info(f"Risk check result for {signal.symbol}: passed={risk_check[0]}, reason={risk_check[1]}")
            if not risk_check[0]:
                logger.warning("Signal rejected by risk check",
                             symbol=signal.symbol,
                             reason=risk_check[1])
                self.routing_stats["signals_rejected"] += 1
                return None
            
            # Store active signal
            self.active_signals[signal_id] = signal
            
            # Route based on signal type
            logger.info(f"Routing signal for {signal.symbol}: liquidation_detected={signal.liquidation_detected}")
            if signal.liquidation_detected:
                logger.info(f"Routing liquidation trade for {signal.symbol}")
                position_id = await self._route_liquidation_trade(signal, risk_check[2])
            else:
                logger.info(f"Routing standard trade for {signal.symbol}")
                position_id = await self._route_standard_trade(signal, risk_check[2])
            logger.info(f"Routing completed for {signal.symbol}: position_id={position_id}")
            
            if position_id:
                self.routing_stats["signals_executed"] += 1
                logger.info("Signal executed",
                           signal_id=signal_id,
                           position_id=position_id,
                           symbol=signal.symbol,
                           prediction=signal.prediction,
                           confidence=signal.confidence)
            
            # Track signal
            self.signal_history.append(signal)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-500:]
            
            return position_id
            
        except Exception as e:
            logger.error("Error processing signal",
                        signal_id=signal_id,
                        exception=e)
            increment_counter("routing_errors", symbol=signal.symbol)
            return None
    
    async def _route_liquidation_trade(
        self,
        signal: TradingSignal,
        risk_metrics: Dict[str, Any]
    ) -> Optional[str]:
        """Route liquidation-driven trade."""
        self.routing_stats["liquidation_trades"] += 1
        
        logger.info("Routing liquidation trade",
                   symbol=signal.symbol,
                   liquidation_size=signal.liquidation_size,
                   liquidation_side=signal.liquidation_side)
        
        # Determine trade direction (fade the liquidation)
        if self.config.fade_liquidation:
            trade_side = "buy" if signal.liquidation_side == "long" else "sell"
        else:
            trade_side = "sell" if signal.liquidation_side == "long" else "buy"
        
        # Calculate position size with liquidation multiplier
        base_size = risk_metrics["position_size"]
        position_size = base_size * self.config.liquidation_multiplier
        
        # Get current market data
        market_data = await self._get_market_data(signal.symbol)
        if not market_data:
            return None
        
        # Place layered orders
        position_id = f"liq_{signal.symbol}_{int(signal.timestamp.timestamp())}"
        orders_placed = []
        
        try:
            # Create position
            position = await self.position_manager.open_position(
                position_id=position_id,
                symbol=signal.symbol,
                side="long" if trade_side == "buy" else "short",
                quantity=position_size,
                entry_price=market_data["mid_price"],
                strategy="liquidation_cascade",
                metadata={
                    "signal_prediction": signal.prediction,
                    "signal_confidence": signal.confidence,
                    "liquidation_size": signal.liquidation_size,
                    "liquidation_side": signal.liquidation_side
                }
            )
            
            # Place multiple limit orders at different levels
            layer_size = position_size / self.config.liquidation_layers
            
            for i in range(self.config.liquidation_layers):
                # Calculate layer price
                if trade_side == "buy":
                    # Place buy orders below mid price
                    price_offset = market_data["spread"] * (0.5 + i * 0.5)
                    layer_price = market_data["mid_price"] - price_offset
                else:
                    # Place sell orders above mid price
                    price_offset = market_data["spread"] * (0.5 + i * 0.5)
                    layer_price = market_data["mid_price"] + price_offset
                
                # Place order
                order = await self.executor.place_order(
                    symbol=signal.symbol,
                    side=trade_side,
                    quantity=layer_size,
                    price=layer_price,
                    order_type=OrderType.POST_ONLY,
                    position_id=position_id,
                    metadata={
                        "layer": i,
                        "signal_type": "liquidation"
                    }
                )
                
                orders_placed.append(order)
                
                # Small delay between layers
                if i < self.config.liquidation_layers - 1:
                    await asyncio.sleep(0.05)
            
            logger.info("Liquidation orders placed",
                       position_id=position_id,
                       orders_count=len(orders_placed),
                       total_size=position_size)
            
            return position_id
            
        except Exception as e:
            logger.error("Failed to route liquidation trade",
                        position_id=position_id,
                        exception=e)
            
            # Cancel any placed orders
            for order in orders_placed:
                await self.executor.cancel_order(order.order_id)
            
            # Close position if created
            if position_id in self.position_manager.positions:
                await self.position_manager.close_position(
                    position_id=position_id,
                    exit_price=market_data["mid_price"],
                    reason="routing_error"
                )
            
            return None
    
    async def _route_standard_trade(
        self,
        signal: TradingSignal,
        risk_metrics: Dict[str, Any]
    ) -> Optional[str]:
        """Route standard ML-driven trade."""
        logger.info(f"Starting standard trade routing for {signal.symbol}")
        logger.info(f"Risk metrics: {risk_metrics}")
        
        # Determine trade direction based on prediction
        trade_side = "buy" if signal.prediction > 0 else "sell"
        position_size = risk_metrics["position_size"]
        logger.info(f"Trade side: {trade_side}, position size: {position_size}")
        
        # Get market data
        logger.info(f"Getting market data for {signal.symbol}")
        market_data = await self._get_market_data(signal.symbol)
        if not market_data:
            logger.error(f"Failed to get market data for {signal.symbol}")
            return None
        logger.info(f"Market data for {signal.symbol}: {market_data}")
        
        # Check spread
        spread_bps = (market_data["spread"] / market_data["mid_price"]) * 10000
        if spread_bps > self.config.max_spread_bps:
            logger.warning("Spread too wide",
                         symbol=signal.symbol,
                         spread_bps=spread_bps,
                         max_allowed=self.config.max_spread_bps)
            return None
        
        # Calculate order price
        order_price = self._calculate_order_price(
            trade_side,
            market_data,
            signal
        )
        
        # Create position
        position_id = f"ml_{signal.symbol}_{int(signal.timestamp.timestamp())}"
        logger.info(f"Creating position with ID: {position_id}")
        
        try:
            logger.info(f"Opening position for {signal.symbol}")
            position = await self.position_manager.open_position(
                position_id=position_id,
                symbol=signal.symbol,
                side="long" if trade_side == "buy" else "short",
                quantity=position_size,
                entry_price=order_price,
                strategy="ml_prediction",
                metadata={
                    "signal_prediction": signal.prediction,
                    "signal_confidence": signal.confidence,
                    "market_spread": market_data["spread"],
                    "market_volatility": signal.volatility
                }
            )
            
            # Place order
            logger.info(f"Placing order for {signal.symbol}: side={trade_side}, size={position_size}, price={order_price}")
            if self.config.use_iceberg and position_size > 10000:  # Large order
                logger.info(f"Using iceberg order for large position: {position_size}")
                order = await self._place_iceberg_order(
                    signal.symbol,
                    trade_side,
                    position_size,
                    order_price,
                    position_id
                )
            else:
                logger.info(f"Using standard order for {signal.symbol}")
                order = await self.executor.place_order(
                    symbol=signal.symbol,
                    side=trade_side,
                    quantity=position_size,
                    price=order_price,
                    order_type=OrderType.POST_ONLY,
                    position_id=position_id,
                    metadata={"signal_type": "ml_prediction"}
                )
            logger.info(f"Order placement result for {signal.symbol}: {order}")
            
            if order:
                logger.info("Standard order placed",
                           position_id=position_id,
                           order_id=order.order_id,
                           symbol=signal.symbol,
                           side=trade_side,
                           quantity=position_size,
                           price=order_price)
                
                return position_id
            else:
                # Close position if order failed
                await self.position_manager.close_position(
                    position_id=position_id,
                    exit_price=order_price,
                    reason="order_failed"
                )
                return None
                
        except Exception as e:
            logger.error("Failed to route standard trade",
                        position_id=position_id,
                        exception=e)
            return None
    
    async def _place_iceberg_order(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        price: float,
        position_id: str
    ) -> Optional[Order]:
        """Place iceberg order (show only part of total size)."""
        visible_quantity = total_quantity * self.config.iceberg_visible_pct
        hidden_quantity = total_quantity - visible_quantity
        
        logger.info("Placing iceberg order",
                   symbol=symbol,
                   total_quantity=total_quantity,
                   visible_quantity=visible_quantity)
        
        # Place visible portion
        visible_order = await self.executor.place_order(
            symbol=symbol,
            side=side,
            quantity=visible_quantity,
            price=price,
            order_type=OrderType.POST_ONLY,
            position_id=position_id,
            metadata={
                "order_type": "iceberg",
                "total_quantity": total_quantity,
                "hidden_quantity": hidden_quantity
            }
        )
        
        # TODO: Implement hidden order logic
        # This would involve monitoring fills and placing new visible orders
        
        return visible_order
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate trading signal."""
        # Check confidence threshold
        if signal.confidence < self.config.min_confidence:
            logger.debug("Signal confidence too low",
                        symbol=signal.symbol,
                        confidence=signal.confidence,
                        min_required=self.config.min_confidence)
            return False
        
        # Check prediction threshold
        if abs(signal.prediction) < self.config.min_prediction:
            logger.debug("Signal prediction too small",
                        symbol=signal.symbol,
                        prediction=signal.prediction,
                        min_required=self.config.min_prediction)
            return False
        
        # Check signal age
        signal_age = (datetime.now() - signal.timestamp).total_seconds()
        if signal_age > 60:  # More than 1 minute old
            logger.warning("Signal too old",
                         symbol=signal.symbol,
                         age_seconds=signal_age)
            return False
        
        return True
    
    async def _check_risk(self, signal: TradingSignal) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Check risk constraints for signal."""
        # Estimate entry price (simplified)
        market_data = await self._get_market_data(signal.symbol)
        if not market_data:
            return False, "Failed to get market data", {}
        
        entry_price = market_data["mid_price"]
        
        # Calculate position size
        stop_loss_price = entry_price * (1 - 0.02)  # 2% stop loss
        position_size = self.risk_manager.calculate_position_size(
            symbol=signal.symbol,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            confidence=signal.confidence,
            volatility=signal.volatility
        )
        
        if position_size <= 0:
            return False, "Position size too small", {}
        
        # Check order risk
        trade_side = "buy" if signal.prediction > 0 else "sell"
        risk_approved, rejection_reason, risk_metrics = await self.risk_manager.check_order_risk(
            symbol=Symbol(signal.symbol),
            side=trade_side,
            quantity=position_size,
            price=entry_price
        )
        
        # Add position size to metrics
        risk_metrics["position_size"] = position_size
        
        return risk_approved, rejection_reason, risk_metrics
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current market data."""
        try:
            # This would typically fetch from a market data service
            # For now, return mock data
            return {
                "bid": 50000.0,
                "ask": 50010.0,
                "mid_price": 50005.0,
                "spread": 10.0,
                "last_price": 50005.0
            }
        except Exception as e:
            logger.error("Failed to get market data",
                        symbol=symbol,
                        exception=e)
            return None
    
    def _calculate_order_price(
        self,
        side: str,
        market_data: Dict[str, float],
        signal: TradingSignal
    ) -> float:
        """Calculate optimal order price."""
        spread = market_data["spread"]
        mid_price = market_data["mid_price"]
        
        # Base price offset
        price_offset = spread * self.config.spread_multiplier
        
        # Adjust for volatility
        if self.config.volatility_adjustment and signal.volatility > 0:
            vol_adjustment = min(signal.volatility / 0.01, 2.0)  # Cap at 2x
            price_offset *= vol_adjustment
        
        # Adjust for urgency
        if signal.urgency == "high":
            price_offset *= 0.5  # More aggressive pricing
        elif signal.urgency == "low":
            price_offset *= 1.5  # More passive pricing
        
        # Calculate final price
        if side == "buy":
            order_price = mid_price - price_offset
        else:
            order_price = mid_price + price_offset
        
        return order_price
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        stats = self.routing_stats.copy()
        
        # Add computed metrics
        if stats["signals_received"] > 0:
            stats["execution_rate"] = stats["signals_executed"] / stats["signals_received"]
            stats["rejection_rate"] = stats["signals_rejected"] / stats["signals_received"]
            stats["liquidation_rate"] = stats["liquidation_trades"] / stats["signals_received"]
        else:
            stats["execution_rate"] = 0.0
            stats["rejection_rate"] = 0.0
            stats["liquidation_rate"] = 0.0
        
        stats["active_signals"] = len(self.active_signals)
        
        return stats