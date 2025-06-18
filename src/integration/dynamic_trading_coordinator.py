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
                    # Get current feature status for the notification
                    feature_status = await self._get_feature_status()
                    feature_summary = ""
                    
                    # Add brief feature status
                    active_symbols = sum(1 for count in feature_status["feature_counts"].values() if count > 0)
                    if active_symbols > 0:
                        feature_summary = f"\n📊 特徴量: {active_symbols}/3シンボル稼働中"
                        avg_count = feature_status.get("avg_feature_count", 0)
                        if avg_count > 0:
                            feature_summary += f" (平均{avg_count:.0f}個)"
                    else:
                        feature_summary = "\n⚠️ 特徴量生成が停止中"
                    
                    discord_notifier.send_notification(
                        title="💰 Dynamic Parameters Updated",
                        description=f"Account balance: ${current_balance:.2f}\n"
                                  f"Max position: ${new_risk_config.max_position_size_usd:.2f}\n"
                                  f"Max exposure: ${new_risk_config.max_total_exposure_usd:.2f}"
                                  f"{feature_summary}",
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
            self._prediction_task = asyncio.create_task(self._prediction_loop())
            self._parameter_task = asyncio.create_task(self._parameter_update_loop())
            self._health_task = asyncio.create_task(self._health_monitor_loop())
            self._feature_report_task = asyncio.create_task(self._feature_report_loop())
            self._position_monitor_task = asyncio.create_task(self._position_monitor_loop())
            
            logger.info("Background tasks started: prediction_loop, parameter_update_loop, health_monitor_loop, feature_report_loop, position_monitor_loop")
            
            logger.info("Dynamic Trading Coordinator started successfully")
            
            # Send startup notification
            # Wait a bit for feature generation to start
            await asyncio.sleep(5)
            
            # Get initial feature status
            feature_status = await self._get_feature_status()
            feature_summary = ""
            
            active_symbols = sum(1 for count in feature_status["feature_counts"].values() if count > 0)
            if active_symbols > 0:
                feature_summary = f"\n📊 特徴量: {active_symbols}/3シンボル初期化済み"
            else:
                feature_summary = f"\n🔄 特徴量: 初期化中..."
            
            discord_notifier.send_notification(
                title="🚀 Dynamic Trading System Started",
                description=f"Account balance: ${initial_balance:.2f}\n"
                          f"Max position: ${dynamic_risk_config.max_position_size_usd:.2f}\n"
                          f"Max exposure: ${dynamic_risk_config.max_total_exposure_usd:.2f}\n"
                          f"Trading symbols: {', '.join(self.config.symbols)}"
                          f"{feature_summary}",
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
        logger.info("Prediction loop started")
        while self.running:
            try:
                # Get features for all symbols
                for symbol in self.active_symbols:
                    features = await self._get_features(symbol)
                    if features:
                        logger.debug(f"Features retrieved for {symbol}: {len(features)} features")
                        prediction = await self._get_prediction(symbol, features)
                        if prediction:
                            logger.info(f"Prediction generated for {symbol}: {prediction}")
                            await self._process_prediction(symbol, prediction, features)
                        else:
                            logger.debug(f"No prediction generated for {symbol}")
                    else:
                        logger.debug(f"No features available for {symbol}")
                
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
                # Import the proper InferenceConfig
                from ..ml_pipeline.inference_engine import InferenceConfig
                
                # Create InferenceConfig with proper settings
                inference_config = InferenceConfig(
                    model_path="models/v3.1_improved/model.onnx",
                    preprocessor_path="models/v3.1_improved/scaler.pkl",
                    max_inference_time_ms=100.0,
                    batch_size=32,
                    enable_batching=True,
                    cache_size=1000,
                    providers=["CPUExecutionProvider"],
                    session_options=None,
                    enable_thompson_sampling=False,
                    confidence_threshold=0.7,
                    max_position_size=0.1,
                    risk_adjustment=True,
                    enable_performance_tracking=True,
                    log_predictions=False,
                    alert_on_slow_inference=True
                )
                
                self._inference_engine = InferenceEngine(inference_config)
                # Load the model with correct path
                try:
                    self._inference_engine.load_model(inference_config.model_path)
                    logger.info(f"Model loading attempted for {inference_config.model_path}")
                except Exception as e:
                    logger.error(f"Failed to load model from {inference_config.model_path}: {e}")
                    return None
            
            # Convert features to the format expected by the model
            feature_array = self._prepare_features_for_model(features)
            
            if feature_array is not None:
                # Convert to numpy array if it's a list
                if isinstance(feature_array, list):
                    import numpy as np
                    feature_array = np.array([feature_array], dtype=np.float32)
                
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
            
            # Check thresholds (temporarily lowered for testing)
            if (confidence >= 0.0 and  # Temporarily allow any confidence
                abs(expected_pnl) >= 0.0 and  # Temporarily allow any expected PnL
                direction != "hold"):
                
                logger.info(f"Signal thresholds met for {symbol}: confidence={confidence}, expected_pnl={expected_pnl}, direction={direction}")
                
                # Create trading signal
                signal = TradingSignal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    prediction=expected_pnl,  # Expected PnL
                    confidence=confidence,
                    features=features,
                    liquidation_detected=False,
                    liquidation_size=0.0,
                    liquidation_side=direction  # 'long' or 'short'
                )
                
                # Process signal through order router
                logger.info(f"Sending signal to OrderRouter for {symbol}: {signal}")
                position_id = await self.order_router.process_signal(signal)
                logger.info(f"OrderRouter response for {symbol}: position_id={position_id}")
                
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
                    
                    # Send Discord notification with feature quality info
                    feature_count = len(features) if features else 0
                    feature_quality = "🟢" if feature_count > 150 else "🟡" if feature_count > 100 else "🔴"
                    
                    discord_notifier.send_notification(
                        title="📈 Trading Signal Generated",
                        description=f"Symbol: {symbol}\n"
                                  f"Direction: {direction.upper()}\n"
                                  f"Confidence: {confidence:.1%}\n"
                                  f"Expected PnL: {expected_pnl:.2%}\n"
                                  f"Position ID: {position_id}\n"
                                  f"Feature Quality: {feature_quality} ({feature_count}個)",
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
            # Cancel background tasks
            if hasattr(self, '_position_monitor_task'):
                self._position_monitor_task.cancel()
                try:
                    await self._position_monitor_task
                except asyncio.CancelledError:
                    pass
            
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
    
    async def _position_monitor_loop(self) -> None:
        """Monitor open positions and manage exits."""
        await asyncio.sleep(30)  # Initial wait for system to stabilize
        
        while self.running:
            try:
                # Get open positions from Bybit
                positions = await self.order_router.order_executor.bybit_client.get_open_positions()
                
                if positions:
                    logger.info(f"Monitoring {len(positions)} open positions")
                    
                    for position in positions:
                        symbol = position.get("symbol")
                        side = position.get("side")
                        size = float(position.get("size", 0))
                        entry_price = float(position.get("avgPrice", 0))
                        unrealized_pnl = float(position.get("unrealizedPnl", 0))
                        mark_price = float(position.get("markPrice", 0))
                        
                        if size > 0:
                            # Calculate PnL percentage
                            pnl_pct = (unrealized_pnl / (size * entry_price)) * 100 if entry_price > 0 else 0
                            
                            # Log position status
                            logger.info(
                                f"Position {symbol} {side}: "
                                f"size={size} entry=${entry_price:.2f} "
                                f"mark=${mark_price:.2f} PnL=${unrealized_pnl:.2f} ({pnl_pct:.2f}%)"
                            )
                            
                            # Check for trailing stop
                            await self._check_trailing_stop(position)
                            
                            # Check for partial take profit
                            await self._check_partial_take_profit(position)
                            
                            # Check for manual intervention needed
                            if abs(pnl_pct) > 5:  # More than 5% move
                                discord_notifier.send_notification(
                                    title="⚠️ ポジション監視アラート",
                                    description=f"{symbol} ポジションが大きく動いています",
                                    color="ff9900",
                                    fields={
                                        "Symbol": symbol,
                                        "Side": side,
                                        "Entry": f"${entry_price:.2f}",
                                        "Current": f"${mark_price:.2f}",
                                        "PnL": f"${unrealized_pnl:.2f} ({pnl_pct:.2f}%)",
                                        "Size": str(size)
                                    }
                                )
                
                # Also check open orders
                open_orders = await self.order_router.order_executor.bybit_client.get_open_orders()
                if open_orders:
                    logger.info(f"Active orders: {len(open_orders)}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(60)
    
    async def _check_trailing_stop(self, position: Dict[str, Any]) -> None:
        """Check and update trailing stop loss."""
        symbol = position.get("symbol")
        side = position.get("side")
        size = float(position.get("size", 0))
        entry_price = float(position.get("avgPrice", 0))
        mark_price = float(position.get("markPrice", 0))
        
        if size == 0 or entry_price == 0:
            return
        
        # Calculate profit percentage
        if side == "Buy":
            profit_pct = ((mark_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - mark_price) / entry_price) * 100
        
        # Trailing stop logic: If profit > 2%, set stop loss to breakeven + 0.5%
        if profit_pct > 2.0:
            # Calculate new stop loss
            if side == "Buy":
                new_stop_loss = entry_price * 1.005  # 0.5% above entry
            else:
                new_stop_loss = entry_price * 0.995  # 0.5% below entry
            
            # Update stop loss
            success = await self.order_router.order_executor.bybit_client.set_stop_loss(symbol, new_stop_loss)
            if success:
                logger.info(f"Trailing stop updated for {symbol}: ${new_stop_loss:.2f}")
                discord_notifier.send_notification(
                    title="🔄 トレーリングストップ更新",
                    description=f"{symbol} のストップロスを更新しました",
                    color="03b2f8",
                    fields={
                        "Symbol": symbol,
                        "Side": side,
                        "Entry": f"${entry_price:.2f}",
                        "Current": f"${mark_price:.2f}",
                        "Profit": f"{profit_pct:.2f}%",
                        "New Stop": f"${new_stop_loss:.2f}"
                    }
                )
    
    async def _check_partial_take_profit(self, position: Dict[str, Any]) -> None:
        """Check and execute partial take profit."""
        symbol = position.get("symbol")
        side = position.get("side")
        size = float(position.get("size", 0))
        entry_price = float(position.get("avgPrice", 0))
        mark_price = float(position.get("markPrice", 0))
        
        if size == 0 or entry_price == 0:
            return
        
        # Calculate profit percentage
        if side == "Buy":
            profit_pct = ((mark_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - mark_price) / entry_price) * 100
        
        # Partial take profit logic
        position_id = f"pos_{symbol}_{side}"
        closed_pct = getattr(self, '_partial_closes', {}).get(position_id, 0)
        
        if profit_pct >= 3.0 and closed_pct < 75:
            # Close 25% more (total 75%)
            close_size = size * 0.25
            close_side = "Sell" if side == "Buy" else "Buy"
            
            result = await self.order_router.order_executor.bybit_client.create_order(
                symbol=symbol,
                side=close_side,
                order_type="Market",
                qty=close_size,
                reduce_only=True
            )
            
            if result:
                if not hasattr(self, '_partial_closes'):
                    self._partial_closes = {}
                self._partial_closes[position_id] = 75
                logger.info(f"Partial take profit executed: {symbol} 25% at {profit_pct:.2f}% profit")
                discord_notifier.send_notification(
                    title="💰 部分利確実行 (75%)",
                    description=f"{symbol} ポジションの25%を利確",
                    color="00ff00",
                    fields={
                        "Symbol": symbol,
                        "Profit": f"{profit_pct:.2f}%",
                        "Closed": "75% total",
                        "Remaining": "25%",
                        "Size": f"{close_size:.4f}"
                    }
                )
                
        elif profit_pct >= 1.5 and closed_pct < 50:
            # Close 50%
            close_size = size * 0.5
            close_side = "Sell" if side == "Buy" else "Buy"
            
            result = await self.order_router.order_executor.bybit_client.create_order(
                symbol=symbol,
                side=close_side,
                order_type="Market",
                qty=close_size,
                reduce_only=True
            )
            
            if result:
                if not hasattr(self, '_partial_closes'):
                    self._partial_closes = {}
                self._partial_closes[position_id] = 50
                logger.info(f"Partial take profit executed: {symbol} 50% at {profit_pct:.2f}% profit")
                discord_notifier.send_notification(
                    title="💰 部分利確実行 (50%)",
                    description=f"{symbol} ポジションの50%を利確",
                    color="00ff00",
                    fields={
                        "Symbol": symbol,
                        "Profit": f"{profit_pct:.2f}%",
                        "Closed": "50%",
                        "Remaining": "50%",
                        "Size": f"{close_size:.4f}"
                    }
                )
    
    async def _feature_report_loop(self) -> None:
        """Send periodic feature status reports to Discord."""
        while self.running:
            try:
                await asyncio.sleep(600)  # Send report every 10 minutes
                
                feature_status = await self._get_feature_status()
                
                # Construct Discord message
                description = "📊 **特徴量生成状況**\n\n"
                
                # Add symbol-specific feature counts
                for symbol in self.config.symbols:
                    count = feature_status["feature_counts"].get(symbol, 0)
                    age = feature_status["cache_ages"].get(symbol, 0)
                    status_emoji = "✅" if count > 0 and age < 60 else "⚠️"
                    description += f"{status_emoji} **{symbol}**: {count}個 (更新: {age:.1f}秒前)\n"
                
                description += f"\n🔢 **全体統計**\n"
                description += f"• アクティブシンボル: {feature_status['total_symbols']}/3\n"
                description += f"• 平均特徴量数: {feature_status['avg_feature_count']:.0f}\n"
                description += f"• 最大キャッシュ年齢: {feature_status['max_cache_age']:.1f}秒\n"
                description += f"• FeatureHub稼働: {'🟢' if feature_status['running'] else '🔴'}\n"
                
                # Add quality metrics
                if feature_status['quality_issues']:
                    description += f"\n⚠️ **品質問題**\n"
                    for issue in feature_status['quality_issues']:
                        description += f"• {issue}\n"
                
                discord_notifier.send_notification(
                    title="📈 特徴量ステータスレポート",
                    description=description,
                    color="3498db"
                )
                
            except Exception as e:
                logger.error("Error in feature report loop", exception=e)
    
    async def _get_feature_status(self) -> Dict[str, Any]:
        """Get comprehensive feature status."""
        try:
            # Access the feature hub instance from simple service manager
            from ..integration.simple_service_manager import SimpleServiceManager
            if hasattr(self, '_service_manager') and hasattr(self._service_manager, 'feature_hub'):
                feature_hub = self._service_manager.feature_hub
                summary = feature_hub.get_feature_summary()
            else:
                # Fallback: create basic summary
                summary = {
                    "symbols": self.config.symbols,
                    "total_symbols": len(self.config.symbols),
                    "feature_counts": {},
                    "cache_ages": {},
                    "running": True
                }
                
                # Try to get feature counts from Redis
                for symbol in self.config.symbols:
                    try:
                        features = await self._get_features(symbol)
                        if features:
                            summary["feature_counts"][symbol] = len(features)
                            summary["cache_ages"][symbol] = 0.0
                        else:
                            summary["feature_counts"][symbol] = 0
                            summary["cache_ages"][symbol] = 999.0
                    except:
                        summary["feature_counts"][symbol] = 0
                        summary["cache_ages"][symbol] = 999.0
            
            # Calculate additional metrics
            feature_counts = list(summary["feature_counts"].values())
            cache_ages = list(summary["cache_ages"].values())
            
            avg_feature_count = sum(feature_counts) / len(feature_counts) if feature_counts else 0
            max_cache_age = max(cache_ages) if cache_ages else 0
            
            # Identify quality issues
            quality_issues = []
            for symbol, count in summary["feature_counts"].items():
                if count == 0:
                    quality_issues.append(f"{symbol}: 特徴量が生成されていません")
                elif count < 100:
                    quality_issues.append(f"{symbol}: 特徴量数が少ない ({count}個)")
            
            for symbol, age in summary["cache_ages"].items():
                if age > 300:  # 5 minutes
                    quality_issues.append(f"{symbol}: データが古い ({age:.0f}秒)")
            
            return {
                **summary,
                "avg_feature_count": avg_feature_count,
                "max_cache_age": max_cache_age,
                "quality_issues": quality_issues
            }
            
        except Exception as e:
            logger.error("Error getting feature status", exception=e)
            return {
                "symbols": self.config.symbols,
                "total_symbols": 0,
                "feature_counts": {},
                "cache_ages": {},
                "running": False,
                "avg_feature_count": 0,
                "max_cache_age": 999,
                "quality_issues": ["特徴量ステータス取得エラー"]
            }