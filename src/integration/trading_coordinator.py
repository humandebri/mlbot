"""
Main trading system coordinator.

Orchestrates all components for end-to-end trading operations:
- Real-time data processing
- Feature computation
- ML predictions
- Order execution
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import numpy as np

from ..common.config import settings
from ..common.logging import get_logger
from ..common.monitoring import (
    SIGNALS_GENERATED, PREDICTIONS_MADE, SYSTEM_HEALTH,
    increment_counter, set_gauge
)
from ..common.database import RedisManager
from ..order_router.smart_router import TradingSignal
from ..order_router.main import OrderRouter

logger = get_logger(__name__)


@dataclass
class SystemConfig:
    """System integration configuration."""
    
    # Service endpoints
    model_server_url: str = "http://localhost:8000"
    feature_hub_url: str = "http://localhost:8002"
    
    # Trading parameters
    symbols: List[str] = None
    min_prediction_confidence: float = 0.6
    min_expected_pnl: float = 0.001  # 0.1%
    
    # Processing
    feature_window_seconds: int = 60
    prediction_interval_seconds: int = 1
    max_concurrent_predictions: int = 10
    
    # Monitoring
    health_check_interval: int = 30
    metrics_update_interval: int = 60
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTCUSDT", "ETHUSDT"]


class TradingCoordinator:
    """
    Main coordinator for the trading system.
    
    Integrates all components:
    - Connects to data sources
    - Requests features from FeatureHub
    - Gets predictions from Model Server
    - Routes orders through Order Router
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize trading coordinator.
        
        Args:
            config: System configuration
        """
        self.config = config or SystemConfig()
        
        # Components
        self.order_router = OrderRouter()
        self.redis_manager = RedisManager()
        
        # HTTP session for API calls
        self.session: Optional[aiohttp.ClientSession] = None
        
        # State tracking
        self.running = False
        self.active_symbols: Set[str] = set(self.config.symbols)
        self.last_predictions: Dict[str, datetime] = {}
        self.signal_history: List[TradingSignal] = []
        
        # Performance tracking
        self.prediction_count = 0
        self.signal_count = 0
        self.error_count = 0
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        logger.info("Trading coordinator initialized", 
                   symbols=self.config.symbols,
                   model_server=self.config.model_server_url)
    
    async def start(self) -> None:
        """Start the trading system."""
        if self.running:
            logger.warning("Trading system already running")
            return
        
        logger.info("Starting trading system")
        
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession()
            
            # Connect to Redis
            await self.redis_manager.connect()
            
            # Start order router
            await self.order_router.start()
            
            # Check service health
            await self._check_services_health()
            
            # Start background tasks
            self.running = True
            self._tasks = [
                asyncio.create_task(self._prediction_loop()),
                asyncio.create_task(self._health_monitor()),
                asyncio.create_task(self._metrics_reporter()),
                asyncio.create_task(self._signal_processor())
            ]
            
            logger.info("Trading system started successfully")
            
        except Exception as e:
            logger.error("Failed to start trading system", exception=e)
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the trading system."""
        if not self.running:
            return
        
        logger.info("Stopping trading system")
        self.running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Stop components
        await self.order_router.stop()
        
        # Close connections
        if self.session:
            await self.session.close()
        
        await self.redis_manager.close()
        
        logger.info("Trading system stopped")
    
    async def _prediction_loop(self) -> None:
        """Main prediction loop - requests features and makes predictions."""
        while self.running:
            try:
                # Process each symbol
                tasks = []
                for symbol in self.active_symbols:
                    # Check if we should make a prediction
                    last_pred = self.last_predictions.get(symbol)
                    if last_pred:
                        time_since = (datetime.now() - last_pred).total_seconds()
                        if time_since < self.config.prediction_interval_seconds:
                            continue
                    
                    # Create prediction task
                    task = self._process_symbol_prediction(symbol)
                    tasks.append(task)
                
                # Limit concurrent predictions
                if tasks:
                    # Process in batches
                    for i in range(0, len(tasks), self.config.max_concurrent_predictions):
                        batch = tasks[i:i + self.config.max_concurrent_predictions]
                        await asyncio.gather(*batch, return_exceptions=True)
                
                # Short sleep to prevent tight loop
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Error in prediction loop", exception=e)
                self.error_count += 1
                await asyncio.sleep(1)
    
    async def _process_symbol_prediction(self, symbol: str) -> None:
        """Process prediction for a single symbol."""
        try:
            # 1. Get features from FeatureHub
            features = await self._get_features(symbol)
            if not features:
                logger.warning("No features available", symbol=symbol)
                return
            
            # 2. Get prediction from Model Server
            prediction_result = await self._get_prediction(symbol, features)
            if not prediction_result:
                logger.warning("Prediction failed", symbol=symbol)
                return
            
            # 3. Check if we should generate a trading signal
            if self._should_trade(prediction_result):
                signal = self._create_trading_signal(symbol, prediction_result, features)
                
                # 4. Send signal to order router
                position_id = await self.order_router.process_signal(signal)
                
                if position_id:
                    logger.info("Trading signal executed",
                               symbol=symbol,
                               position_id=position_id,
                               prediction=prediction_result["prediction"],
                               confidence=prediction_result["confidence"])
                    
                    increment_counter(SIGNALS_GENERATED, symbol=symbol)
                    self.signal_count += 1
            
            # Update tracking
            self.last_predictions[symbol] = datetime.now()
            self.prediction_count += 1
            increment_counter(PREDICTIONS_MADE, symbol=symbol)
            
        except Exception as e:
            logger.error("Error processing symbol prediction",
                        symbol=symbol,
                        exception=e)
            self.error_count += 1
    
    async def _get_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get features from FeatureHub."""
        if not self.session:
            return None
        
        try:
            url = f"{self.config.feature_hub_url}/features/{symbol}/latest"
            
            async with self.session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract feature values
                    features = {}
                    
                    # Add all feature categories
                    for category in ["micro_liquidity", "volatility_momentum", 
                                   "liquidation", "time_context"]:
                        if category in data:
                            features.update(data[category])
                    
                    return features
                else:
                    logger.warning("Failed to get features",
                                 symbol=symbol,
                                 status=response.status)
                    return None
                    
        except Exception as e:
            logger.error("Error getting features",
                        symbol=symbol,
                        exception=e)
            return None
    
    async def _get_prediction(
        self, 
        symbol: str, 
        features: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Get prediction from Model Server."""
        if not self.session:
            return None
        
        try:
            url = f"{self.config.model_server_url}/predict/single"
            
            payload = {
                "features": features,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
            
            async with self.session.post(url, json=payload, timeout=5) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning("Prediction request failed",
                                 symbol=symbol,
                                 status=response.status)
                    return None
                    
        except Exception as e:
            logger.error("Error getting prediction",
                        symbol=symbol,
                        exception=e)
            return None
    
    def _should_trade(self, prediction_result: Dict[str, Any]) -> bool:
        """Determine if prediction warrants a trade."""
        prediction = prediction_result.get("prediction", 0)
        confidence = prediction_result.get("confidence", 0)
        
        # Check confidence threshold
        if confidence < self.config.min_prediction_confidence:
            return False
        
        # Check expected PnL threshold
        if abs(prediction) < self.config.min_expected_pnl:
            return False
        
        return True
    
    def _create_trading_signal(
        self,
        symbol: str,
        prediction_result: Dict[str, Any],
        features: Dict[str, float]
    ) -> TradingSignal:
        """Create trading signal from prediction."""
        
        # Check for liquidation signals
        liquidation_detected = False
        liquidation_size = 0.0
        liquidation_side = ""
        
        if "liquidation_volume_long" in features and "liquidation_volume_short" in features:
            long_liq = features.get("liquidation_volume_long", 0)
            short_liq = features.get("liquidation_volume_short", 0)
            
            if long_liq > 1000000:  # $1M threshold
                liquidation_detected = True
                liquidation_size = long_liq
                liquidation_side = "long"
            elif short_liq > 1000000:
                liquidation_detected = True
                liquidation_size = short_liq
                liquidation_side = "short"
        
        # Extract market conditions
        spread = features.get("spread", 0)
        volatility = features.get("volatility_20", 0.01)
        volume_surge = features.get("volume_surge_ratio", 1.0)
        
        # Determine urgency
        urgency = "normal"
        if liquidation_detected or volume_surge > 3.0:
            urgency = "high"
        elif volatility < 0.005:  # Low volatility
            urgency = "low"
        
        signal = TradingSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            prediction=prediction_result["prediction"],
            confidence=prediction_result["confidence"],
            features=features,
            liquidation_detected=liquidation_detected,
            liquidation_size=liquidation_size,
            liquidation_side=liquidation_side,
            spread=spread,
            volatility=volatility,
            volume_surge=volume_surge,
            urgency=urgency,
            suggested_timeframe=300 if urgency == "normal" else 60
        )
        
        # Track signal
        self.signal_history.append(signal)
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-500:]
        
        return signal
    
    async def _signal_processor(self) -> None:
        """Process signals from Redis stream."""
        while self.running:
            try:
                # Read from liquidation events stream
                messages = await self.redis_manager.xread(
                    {"liquidation_events": "$"},
                    block=1000  # 1 second timeout
                )
                
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        await self._process_liquidation_event(data)
                
            except Exception as e:
                logger.error("Error in signal processor", exception=e)
                await asyncio.sleep(1)
    
    async def _process_liquidation_event(self, event_data: Dict[str, Any]) -> None:
        """Process liquidation event from stream."""
        try:
            symbol = event_data.get("symbol")
            side = event_data.get("side")
            size_usd = float(event_data.get("size_usd", 0))
            
            if symbol in self.active_symbols and size_usd > 500000:  # $500k threshold
                logger.info("Processing liquidation event",
                           symbol=symbol,
                           side=side,
                           size_usd=size_usd)
                
                # Trigger immediate prediction
                await self._process_symbol_prediction(symbol)
                
        except Exception as e:
            logger.error("Error processing liquidation event", exception=e)
    
    async def _health_monitor(self) -> None:
        """Monitor system health."""
        while self.running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check services
                health_status = await self._check_services_health()
                
                # Update health metric
                health_score = 1.0 if all(health_status.values()) else 0.0
                set_gauge(SYSTEM_HEALTH, health_score)
                
                # Log unhealthy services
                for service, healthy in health_status.items():
                    if not healthy:
                        logger.warning(f"{service} is unhealthy")
                
            except Exception as e:
                logger.error("Error in health monitor", exception=e)
    
    async def _check_services_health(self) -> Dict[str, bool]:
        """Check health of all services."""
        health_status = {}
        
        # Check Model Server
        try:
            async with self.session.get(
                f"{self.config.model_server_url}/health/ready",
                timeout=5
            ) as response:
                health_status["model_server"] = response.status == 200
        except:
            health_status["model_server"] = False
        
        # Check FeatureHub
        try:
            async with self.session.get(
                f"{self.config.feature_hub_url}/health",
                timeout=5
            ) as response:
                health_status["feature_hub"] = response.status == 200
        except:
            health_status["feature_hub"] = False
        
        # Check Redis
        try:
            await self.redis_manager.ping()
            health_status["redis"] = True
        except:
            health_status["redis"] = False
        
        # Check Order Router
        health_status["order_router"] = self.order_router.running
        
        return health_status
    
    async def _metrics_reporter(self) -> None:
        """Report system metrics periodically."""
        while self.running:
            try:
                await asyncio.sleep(self.config.metrics_update_interval)
                
                # Get system status
                router_status = await self.order_router.get_status()
                
                # Log metrics
                logger.info("System metrics",
                           predictions_total=self.prediction_count,
                           signals_generated=self.signal_count,
                           errors_total=self.error_count,
                           active_positions=router_status["active_positions"],
                           total_pnl=router_status["performance"]["total_pnl"],
                           win_rate=router_status["performance"]["win_rate"])
                
            except Exception as e:
                logger.error("Error reporting metrics", exception=e)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        # Get health status
        health_status = await self._check_services_health()
        
        # Get router status
        router_status = await self.order_router.get_status()
        
        # Recent signals
        recent_signals = []
        for signal in self.signal_history[-10:]:
            recent_signals.append({
                "symbol": signal.symbol,
                "timestamp": signal.timestamp,
                "prediction": signal.prediction,
                "confidence": signal.confidence,
                "liquidation_detected": signal.liquidation_detected,
                "urgency": signal.urgency
            })
        
        return {
            "status": "running" if self.running else "stopped",
            "timestamp": datetime.now(),
            "health": health_status,
            "statistics": {
                "predictions_total": self.prediction_count,
                "signals_generated": self.signal_count,
                "errors_total": self.error_count,
                "active_symbols": list(self.active_symbols)
            },
            "recent_signals": recent_signals,
            "trading": router_status,
            "config": {
                "symbols": self.config.symbols,
                "min_confidence": self.config.min_prediction_confidence,
                "min_expected_pnl": self.config.min_expected_pnl
            }
        }