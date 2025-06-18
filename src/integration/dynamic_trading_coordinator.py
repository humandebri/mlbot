"""
Dynamic Trading System Coordinator with account balance-based parameter adjustment.

Enhanced version of trading_coordinator.py with:
- Dynamic risk management based on actual account balance
- Real-time market price fetching
- Account monitoring integration
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
    SIGNALS_GENERATED, MODEL_PREDICTIONS, ORDERS_PLACED, SYSTEM_HEALTH,
    increment_counter, set_gauge
)
from ..common.database import RedisManager
from ..common.account_monitor import AccountMonitor
from ..common.discord_notifier import discord_notifier
from ..order_router.smart_router import TradingSignal
from ..order_router.main import OrderRouter
from ..order_router.risk_manager import RiskConfig
from ..order_router.order_executor import ExecutionConfig

logger = get_logger(__name__)


@dataclass
class DynamicSystemConfig:
    """Enhanced system configuration with dynamic parameters."""
    
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
    
    # Dynamic parameter settings (% of account balance)
    max_position_pct: float = 0.3  # 30% per position
    max_total_exposure_pct: float = 0.6  # 60% total exposure
    max_daily_loss_pct: float = 0.1  # 10% daily loss limit
    leverage: float = 3.0  # 3x leverage max
    
    # Balance update settings
    balance_update_interval: int = 300  # 5 minutes
    parameter_update_threshold: float = 0.1  # 10% balance change triggers update
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = settings.bybit.symbols


class DynamicTradingCoordinator:
    """Enhanced trading coordinator with dynamic parameter management."""
    
    def __init__(self, config: DynamicSystemConfig):
        self.config = config
        self.running = False
        self.active_symbols: Set[str] = set(config.symbols)
        
        # Account monitoring
        self.account_monitor = AccountMonitor(check_interval=60)
        self.last_balance = 0.0
        self.last_parameter_update = datetime.now()
        
        # Component initialization
        self.redis_manager = None
        self.order_router = None
        
        # Tracking
        self.signal_count = 0
        self.prediction_count = 0
        self.last_predictions: Dict[str, Dict] = {}
        self.recent_signals: List[Dict] = []
        
        # Signal cooldown to prevent spam
        self.last_signal_time = {}
        self.signal_cooldown = 300  # 5 minutes
        
        logger.info("Dynamic Trading Coordinator initialized")
    
    def _create_dynamic_risk_config(self, account_balance: float) -> RiskConfig:
        """Create risk configuration based on account balance."""
        return RiskConfig(
            # Position limits (% of account)
            max_position_size_usd=account_balance * self.config.max_position_pct,
            max_total_exposure_usd=account_balance * self.config.max_total_exposure_pct,
            max_leverage=self.config.leverage,
            max_positions=3,  # Max 3 concurrent positions
            
            # Loss limits (% of account)
            max_daily_loss_usd=account_balance * self.config.max_daily_loss_pct,
            max_drawdown_pct=0.2,  # 20% max drawdown
            stop_loss_pct=0.02,  # 2% stop loss
            trailing_stop_pct=0.015,  # 1.5% trailing stop
            
            # Risk per trade
            risk_per_trade_pct=0.01,  # Risk 1% of capital per trade
            kelly_fraction=0.25,  # 25% fractional Kelly
            
            # Other settings remain constant
            max_correlated_positions=5,
            correlation_threshold=0.7,
            cooldown_period_seconds=300,
            max_trades_per_hour=50,
            max_trades_per_day=200,
            volatility_scaling=True,
            volatility_lookback=20,
            target_volatility=0.15,
            var_confidence=0.95,
            cvar_confidence=0.95,
            circuit_breaker_loss_pct=0.05,
            emergency_liquidation=True
        )
    
    def _create_dynamic_execution_config(self, account_balance: float) -> ExecutionConfig:
        """Create execution configuration based on account balance."""
        return ExecutionConfig(
            # Order management
            max_slippage_pct=0.001,  # 0.1% max slippage
            price_buffer_pct=0.0005,  # 0.05% price buffer
            max_order_age_seconds=300,  # Cancel after 5 minutes
            partial_fill_threshold=0.1,  # Cancel if less than 10% filled
            retry_count=3,
            retry_delay_seconds=1,
            
            # Smart routing
            use_post_only=True,
            split_large_orders=True,
            max_order_size_usd=account_balance * self.config.max_position_pct,
            
            # Performance
            price_improvement_check=True,
            aggressive_fill_timeout=30,
            max_orders_per_second=10.0,
            max_orders_per_symbol=5,
            latency_threshold_ms=100.0
        )
    
    async def _update_dynamic_parameters(self) -> None:
        """Update risk and execution parameters based on current balance."""
        try:
            if not self.account_monitor.current_balance:
                logger.warning("Account balance not available yet")
                return
            current_balance = self.account_monitor.current_balance.total_equity
            
            # Check if update is needed
            balance_change = abs(current_balance - self.last_balance) / max(self.last_balance, 1)
            time_since_update = (datetime.now() - self.last_parameter_update).total_seconds()
            
            should_update = (
                balance_change >= self.config.parameter_update_threshold or
                time_since_update >= self.config.balance_update_interval
            )
            
            if should_update and self.order_router:
                # Create new configurations
                new_risk_config = self._create_dynamic_risk_config(current_balance)
                new_execution_config = self._create_dynamic_execution_config(current_balance)
                
                # Update configurations
                self.order_router.risk_manager.config = new_risk_config
                self.order_router.order_executor.config = new_execution_config
                
                # Log updates
                logger.info(f"Dynamic parameters updated for ${current_balance:.2f} account:")
                logger.info(f"  - Max position size: ${new_risk_config.max_position_size_usd:.2f}")
                logger.info(f"  - Max total exposure: ${new_risk_config.max_total_exposure_usd:.2f}")
                logger.info(f"  - Max daily loss: ${new_risk_config.max_daily_loss_usd:.2f}")
                
                self.last_balance = current_balance
                self.last_parameter_update = datetime.now()
                
                # Send Discord notification if significant change
                if balance_change >= 0.05:  # 5% change
                    discord_notifier.send_notification(
                        title="💰 Dynamic Parameters Updated",
                        description=f"Account balance: ${current_balance:.2f}\n"
                                  f"Max position: ${new_risk_config.max_position_size_usd:.2f}\n"
                                  f"Max exposure: ${new_risk_config.max_total_exposure_usd:.2f}",
                        color="0066ff"
                    )
                
        except Exception as e:
            logger.error("Error updating dynamic parameters", exception=e)
    
    async def start(self) -> None:
        """Start the enhanced trading coordinator."""
        if self.running:
            logger.warning("Trading coordinator already running")
            return
        
        self.running = True
        logger.info("Starting Dynamic Trading Coordinator")
        
        try:
            # Initialize Redis connection
            self.redis_manager = RedisManager()
            await self.redis_manager.connect()
            
            # Start account monitoring
            await self.account_monitor.start()
            await asyncio.sleep(2)  # Wait for initial balance
            
            # Get initial balance and create dynamic configurations
            if not self.account_monitor.current_balance:
                logger.error("Failed to get initial account balance")
                raise RuntimeError("Account balance not available")
            initial_balance = self.account_monitor.current_balance.total_equity
            self.last_balance = initial_balance
            
            logger.info(f"Initial account balance: ${initial_balance:.2f}")
            
            # Initialize order router with dynamic configurations
            dynamic_risk_config = self._create_dynamic_risk_config(initial_balance)
            dynamic_execution_config = self._create_dynamic_execution_config(initial_balance)
            
            self.order_router = OrderRouter()
            
            # Update configurations
            self.order_router.risk_manager.config = dynamic_risk_config
            self.order_router.order_executor.config = dynamic_execution_config
            
            # Start order router
            await self.order_router.start()
            
            # Start background tasks
            asyncio.create_task(self._prediction_loop())
            asyncio.create_task(self._parameter_update_loop())
            asyncio.create_task(self._health_monitor_loop())
            
            logger.info("Dynamic Trading Coordinator started successfully")
            
            # Send startup notification
            discord_notifier.send_notification(
                title="🚀 Dynamic Trading System Started",
                description=f"Account balance: ${initial_balance:.2f}\n"
                          f"Max position: ${dynamic_risk_config.max_position_size_usd:.2f}\n"
                          f"Max exposure: ${dynamic_risk_config.max_total_exposure_usd:.2f}\n"
                          f"Trading symbols: {', '.join(self.config.symbols)}",
                color="00ff00"
            )
            
        except Exception as e:
            logger.error("Failed to start dynamic trading coordinator", exception=e)
            await self.stop()
            raise
    
    async def _parameter_update_loop(self) -> None:
        """Background task to periodically update parameters."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._update_dynamic_parameters()
            except Exception as e:
                logger.error("Error in parameter update loop", exception=e)
                await asyncio.sleep(60)
    
    async def _prediction_loop(self) -> None:
        """Main prediction and trading loop."""
        while self.running:
            try:
                # Get features for all symbols
                for symbol in self.active_symbols:
                    features = await self._get_features(symbol)
                    if features:
                        prediction = await self._get_prediction(symbol, features)
                        if prediction:
                            await self._process_prediction(symbol, prediction, features)
                
                await asyncio.sleep(self.config.prediction_interval_seconds)
                
            except Exception as e:
                logger.error("Error in prediction loop", exception=e)
                await asyncio.sleep(5)
    
    async def _health_monitor_loop(self) -> None:
        """Monitor system health."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                health_status = await self._check_services_health()
                overall_health = all(health_status.values())
                
                set_gauge(SYSTEM_HEALTH, 1.0 if overall_health else 0.0)
                
                if not overall_health:
                    logger.warning("System health degraded", health=health_status)
                
            except Exception as e:
                logger.error("Error in health monitor", exception=e)
    
    async def _get_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get features for a symbol from FeatureHub."""
        try:
            # Features are stored as Redis streams, not strings
            feature_key = f"features:{symbol}:latest"
            
            # Read the latest entry from the stream
            if self.redis_manager:
                stream_data = await self.redis_manager.xrevrange(feature_key, count=1)
                
                if stream_data:
                    entry_id, fields = stream_data[0]
                    # Convert stream fields to dict
                    features = dict(fields)
                    
                    # Convert string values to appropriate types
                    processed_features = {}
                    for key, value in features.items():
                        try:
                            # Try to convert to float for numeric features
                            processed_features[key] = float(value)
                        except (ValueError, TypeError):
                            # Keep as string if not numeric
                            processed_features[key] = value
                    
                    return processed_features
                else:
                    # No stream data available yet
                    return None
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting features for {symbol}", exception=e)
            return None
    
    async def _get_prediction(self, symbol: str, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get prediction from integrated model service."""
        try:
            # Import and use the ONNX inference engine directly in integrated mode
            from ..ml_pipeline.inference_engine import InferenceEngine
            
            # Initialize inference engine if not already done
            if not hasattr(self, '_inference_engine'):
                from ..common.config import settings
                self._inference_engine = InferenceEngine(settings)
                # Load the model with correct path
                model_path = "models/v3.1_improved/model.onnx"
                if not self._inference_engine.load_model(model_path):
                    logger.error(f"Failed to load model from {model_path}")
                    return None
                logger.info(f"Model loaded successfully from {model_path}")
            
            # Convert features to the format expected by the model
            feature_array = self._prepare_features_for_model(features)
            
            if feature_array is not None:
                # Get prediction
                prediction_result = self._inference_engine.predict(feature_array)
                
                self.prediction_count += 1
                increment_counter(MODEL_PREDICTIONS, symbol=symbol)
                
                # Format prediction result
                return {
                    "probability": float(prediction_result.get("probability", 0.5)),
                    "confidence": float(prediction_result.get("confidence", 0.0)),
                    "direction": "long" if prediction_result.get("probability", 0.5) > 0.5 else "short",
                    "expected_pnl": float(prediction_result.get("expected_return", 0.0)),
                    "model_version": "v3.1_improved",
                    "symbol": symbol
                }
            else:
                logger.warning(f"Could not prepare features for prediction: {symbol}")
                return None
                        
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}", exception=e)
            return None
    
    def _prepare_features_for_model(self, features: Dict[str, Any]) -> Optional[List[float]]:
        """Prepare features for model input."""
        try:
            # Extract the 44 key features needed for the v3.1_improved model
            feature_names = [
                'returns_1', 'returns_5', 'returns_15', 'returns_30', 'returns_60',
                'volatility_1h', 'volatility_4h', 'volatility_24h',
                'volume_ratio_1h', 'volume_ratio_4h', 'volume_ratio_24h',
                'price_ma_ratio_5', 'price_ma_ratio_20', 'price_ma_ratio_50',
                'rsi_14', 'bb_position', 'bb_width',
                'spread_bps', 'bid_ask_imbalance', 'order_flow_imbalance',
                'liquidation_pressure', 'funding_rate', 'open_interest_change',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'close', 'volume', 'trades_count',
                'vwap_ratio', 'price_momentum', 'volume_momentum',
                'support_distance', 'resistance_distance',
                'trend_strength', 'volatility_rank', 'volume_rank',
                'price_acceleration', 'volume_acceleration',
                'market_regime', 'liquidity_score', 'momentum_score', 'mean_reversion_score'
            ]
            
            feature_vector = []
            for name in feature_names:
                value = features.get(name, 0.0)
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                else:
                    feature_vector.append(0.0)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    async def _process_prediction(self, symbol: str, prediction: Dict[str, Any], features: Dict[str, Any]) -> None:
        """Process prediction and potentially generate trading signal."""
        try:
            confidence = prediction.get("confidence", 0.0)
            expected_pnl = prediction.get("expected_pnl", 0.0)
            direction = prediction.get("direction", "hold")
            
            # Store for monitoring
            self.last_predictions[symbol] = {
                "prediction": prediction,
                "features": features,
                "timestamp": datetime.now().isoformat()
            }
            
            # Check signal cooldown
            now = datetime.now().timestamp()
            last_signal = self.last_signal_time.get(symbol, 0)
            if now - last_signal < self.signal_cooldown:
                return
            
            # Check thresholds
            if (confidence >= self.config.min_prediction_confidence and 
                abs(expected_pnl) >= self.config.min_expected_pnl and
                direction != "hold"):
                
                # Create trading signal
                signal = TradingSignal(
                    symbol=symbol,
                    side="buy" if direction == "long" else "sell",
                    prediction=prediction.get("probability", 0.5),
                    confidence=confidence,
                    features=features,
                    expected_pnl=expected_pnl,
                    metadata={
                        "signal_time": datetime.now().isoformat(),
                        "model_version": prediction.get("model_version", "unknown"),
                        "feature_count": len(features)
                    }
                )
                
                # Process signal through order router
                position_id = await self.order_router.process_signal(signal)
                
                if position_id:
                    self.signal_count += 1
                    increment_counter(SIGNALS_GENERATED, symbol=symbol)
                    increment_counter(ORDERS_PLACED, symbol=symbol)
                    
                    # Update cooldown
                    self.last_signal_time[symbol] = now
                    
                    # Store signal
                    signal_record = {
                        "symbol": symbol,
                        "direction": direction,
                        "confidence": confidence,
                        "expected_pnl": expected_pnl,
                        "position_id": position_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    self.recent_signals.append(signal_record)
                    if len(self.recent_signals) > 100:
                        self.recent_signals = self.recent_signals[-100:]
                    
                    # Send Discord notification
                    discord_notifier.send_notification(
                        title="📈 Trading Signal Generated",
                        description=f"Symbol: {symbol}\n"
                                  f"Direction: {direction.upper()}\n"
                                  f"Confidence: {confidence:.1%}\n"
                                  f"Expected PnL: {expected_pnl:.2%}\n"
                                  f"Position ID: {position_id}",
                        color="0066ff"
                    )
                    
                    logger.info(f"Trading signal processed for {symbol}", 
                              direction=direction, confidence=confidence, position_id=position_id)
                
        except Exception as e:
            logger.error(f"Error processing prediction for {symbol}", exception=e)
    
    async def _check_services_health(self) -> Dict[str, bool]:
        """Check health of all services."""
        health = {
            "redis": False,
            "model_server": False,
            "feature_hub": False,
            "order_router": False,
            "account_monitor": False
        }
        
        try:
            # Check Redis
            if self.redis_manager:
                await self.redis_manager.ping()
                health["redis"] = True
        except:
            pass
        
        # For integrated system, check if model and feature services are available internally
        try:
            # Check if model exists by attempting to load inference engine
            from ..ml_pipeline.pytorch_inference_engine import InferenceEngine
            health["model_server"] = True  # Model is available as a module
        except Exception as e:
            logger.debug(f"Model server check failed: {e}")
            health["model_server"] = False
        
        # Check feature hub by testing if we can access features
        try:
            if self.redis_manager:
                # Test if feature generation is working by checking Redis streams for recent features
                test_key = "features:BTCUSDT:latest"
                stream_data = await self.redis_manager.xrevrange(test_key, count=1)
                health["feature_hub"] = len(stream_data) > 0
            else:
                health["feature_hub"] = False
        except Exception as e:
            logger.debug(f"Feature hub check failed: {e}")
            health["feature_hub"] = False
        
        # Check order router
        health["order_router"] = self.order_router is not None and self.order_router.running
        
        # Check account monitor
        health["account_monitor"] = self.account_monitor._running
        
        return health
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            health = await self._check_services_health()
            current_balance = self.account_monitor.current_balance
            
            # Get order router status if available
            order_status = {}
            if self.order_router:
                order_status = await self.order_router.get_status()
            
            return {
                "running": self.running,
                "health": health,
                "overall_health": all(health.values()),
                "account": current_balance,
                "trading": {
                    "signal_count": self.signal_count,
                    "prediction_count": self.prediction_count,
                    "active_symbols": list(self.active_symbols),
                    "recent_signals": self.recent_signals[-10:],  # Last 10 signals
                },
                "order_router": order_status,
                "dynamic_config": {
                    "max_position_pct": self.config.max_position_pct,
                    "max_total_exposure_pct": self.config.max_total_exposure_pct,
                    "max_daily_loss_pct": self.config.max_daily_loss_pct,
                    "leverage": self.config.leverage,
                    "last_parameter_update": self.last_parameter_update.isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error getting system status", exception=e)
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def stop(self) -> None:
        """Stop the trading coordinator."""
        if not self.running:
            return
        
        logger.info("Stopping Dynamic Trading Coordinator")
        self.running = False
        
        try:
            # Stop order router
            if self.order_router:
                await self.order_router.stop()
            
            # Stop account monitor
            await self.account_monitor.stop()
            
            # Close Redis connection
            if self.redis_manager:
                await self.redis_manager.close()
            
            # Send shutdown notification
            discord_notifier.send_notification(
                title="🛑 Dynamic Trading System Stopped",
                description="Trading system has been shut down",
                color="ff0000"
            )
            
            logger.info("Dynamic Trading Coordinator stopped")
            
        except Exception as e:
            logger.error("Error stopping dynamic trading coordinator", exception=e)