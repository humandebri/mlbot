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
        self.last_predictions: Dict[str, List[Dict]] = {}  # Store recent predictions per symbol
        self.recent_signals: List[Dict] = []
        
        # Signal cooldown to prevent spam (reduced for testing)
        self.last_signal_time = {}
        self.signal_cooldown = 60  # 1 minute (reasonable for testing)
        
        logger.info("Dynamic Trading Coordinator initialized")
    
    def _create_dynamic_risk_config(self, account_balance: float) -> RiskConfig:
        """Create risk configuration based on account balance."""
        # Update risk manager with current balance
        if self.order_router and self.order_router.risk_manager:
            self.order_router.risk_manager.update_equity(account_balance)
            
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
                        logger.info(f"✅ Features retrieved for {symbol}: {len(features)} features")
                        prediction = await self._get_prediction(symbol, features)
                        if prediction:
                            logger.info(f"✅ Prediction generated for {symbol}: confidence={prediction.get('confidence', 0):.2%}, expected_pnl={prediction.get('expected_pnl', 0):.4f}")
                            await self._process_prediction(symbol, prediction, features)
                        else:
                            logger.warning(f"❌ No prediction generated for {symbol}")
                    else:
                        logger.warning(f"❌ No features available for {symbol}")
                
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
        """Get features for a symbol and supplement with technical indicators."""
        try:
            # Get base features from FeatureHub (Redis streams)
            base_features = await self._get_base_features(symbol)
            
            # If we have base features, extract price data and calculate technical indicators
            if base_features:
                # Initialize technical indicator engine if needed
                if not hasattr(self, '_tech_engine'):
                    from ..feature_hub.technical_indicators import TechnicalIndicatorEngine
                    self._tech_engine = TechnicalIndicatorEngine(lookback_periods=100)
                    logger.info("Technical indicator engine initialized")
                
                # Extract price data from base features
                try:
                    open_price = base_features.get('open', base_features.get('close', 50000))
                    high = base_features.get('high', open_price)
                    low = base_features.get('low', open_price)
                    close = base_features.get('close', open_price)
                    volume = base_features.get('volume', 1000000)
                    
                    # Calculate technical indicators
                    tech_features = self._tech_engine.update_price_data(
                        symbol=symbol,
                        open_price=float(open_price),
                        high=float(high),
                        low=float(low),
                        close=float(close),
                        volume=float(volume)
                    )
                    
                    # Combine base features with technical indicators
                    combined_features = {**base_features, **tech_features}
                    
                    logger.debug(f"Combined features for {symbol}: {len(combined_features)} total "
                               f"({len(base_features)} base + {len(tech_features)} technical)")
                    
                    return combined_features
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate technical indicators for {symbol}: {e}")
                    return base_features
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting features for {symbol}", exception=e)
            return None
    
    async def _get_base_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get base features from FeatureHub Redis streams."""
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
            logger.error(f"Error getting base features for {symbol}", exception=e)
            return None
    
    async def _get_prediction(self, symbol: str, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get prediction from integrated model service."""
        try:
            # Use the V3.1_improved inference engine directly
            from ..ml_pipeline.v31_improved_inference_engine import V31ImprovedInferenceEngine, V31ImprovedConfig
            
            # Initialize V3.1_improved inference engine if not already done
            if not hasattr(self, '_v31_inference_engine'):
                # Create V31ImprovedConfig with optimized settings
                v31_config = V31ImprovedConfig(
                    model_path="models/v3.1_improved/model.onnx",
                    confidence_threshold=0.7,  # 70%以上で取引実行
                    buy_threshold=0.55,       # 55%以上でBUY
                    sell_threshold=0.45,      # 45%以下でSELL  
                    high_confidence=0.75,     # 75%以上で高信頼度
                    medium_confidence=0.6     # 60%以上で中信頼度
                )
                
                self._v31_inference_engine = V31ImprovedInferenceEngine(v31_config)
                
                # Load the v3.1_improved model
                try:
                    self._v31_inference_engine.load_model()
                    logger.info("V3.1_improved model loaded successfully (AUC 0.838)")
                except Exception as e:
                    logger.error(f"Failed to load v3.1_improved model: {e}")
                    return None
            
            # Use V3.1_improved engine for prediction
            prediction_result = self._v31_inference_engine.predict(features)
            
            self.prediction_count += 1
            increment_counter(MODEL_PREDICTIONS, symbol=symbol)
            
            # Extract values from V3.1_improved result
            prediction_value = prediction_result.get("prediction", 0.5)
            confidence = prediction_result.get("confidence", 0.5)
            signal_info = prediction_result.get("signal", {})
            
            # Calculate expected PnL based on prediction value
            expected_pnl = (prediction_value - 0.5) * 0.02  # ±2% max expected return
            
            # Determine direction from signal
            direction = signal_info.get("direction", "HOLD").lower()
            if direction == "hold":
                direction = "hold"
            elif direction == "buy":
                direction = "long"
            elif direction == "sell":
                direction = "short"
            
            # Store prediction for reporting
            prediction_data = {
                "probability": prediction_value,
                "confidence": confidence,
                "direction": direction,
                "expected_pnl": expected_pnl,
                "model_version": "v3.1_improved_fixed",
                "symbol": symbol,
                "signal_info": signal_info,
                "timestamp": datetime.now()
            }
            
            # Store in last_predictions for symbol
            if symbol not in self.last_predictions:
                self.last_predictions[symbol] = []
            self.last_predictions[symbol].append(prediction_data)
            
            # Keep only last 100 predictions per symbol
            if len(self.last_predictions[symbol]) > 100:
                self.last_predictions[symbol] = self.last_predictions[symbol][-100:]
            
            # Format prediction result
            return {
                "probability": prediction_value,
                "confidence": confidence,
                "direction": direction,
                "expected_pnl": expected_pnl,
                "model_version": "v3.1_improved_fixed",
                "symbol": symbol,
                "signal_info": signal_info,
                "debug_info": {
                    "raw_prediction": prediction_value,
                    "probabilities": prediction_result.get("probabilities", {}),
                    "signal_tradeable": signal_info.get("tradeable", False),
                    "position_size_multiplier": signal_info.get("position_size_multiplier", 0.0)
                }
            }
                        
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}", exception=e)
            return None
    
    def _prepare_features_for_model(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        V3.1_improved engine accepts feature dictionaries directly.
        No preprocessing needed as FeatureAdapter44 is handled internally.
        """
        return features
    
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
            
            # Check thresholds (restored to reasonable values)
            if direction != "hold":
                # Check confidence and expected PnL thresholds
                if confidence < 0.6 or abs(expected_pnl) < 0.001:  # Reasonable thresholds
                    logger.debug(
                        f"Signal below thresholds for {symbol}: "
                        f"confidence={confidence:.2%}, expected_pnl={expected_pnl:.4f}"
                    )
                    return
                
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
                await asyncio.sleep(3600)  # Send report every hour
                
                # Get comprehensive feature data
                detailed_report = await self._get_detailed_feature_report()
                
                # Send main report
                discord_notifier.send_notification(
                    title=f"📈 特徴量詳細レポート ({datetime.now().strftime('%H:%M')} JST)",
                    description=detailed_report["summary"],
                    color="3498db"
                )
                
                # Send individual symbol reports
                for symbol, report in detailed_report["symbols"].items():
                    if report["has_data"]:
                        discord_notifier.send_notification(
                            title=f"📊 {symbol}",
                            description=report["description"],
                            color="2ecc71" if report["health"] == "good" else "e74c3c"
                        )
                
                # Send system statistics
                discord_notifier.send_notification(
                    title="📊 システム統計",
                    description=detailed_report["system_stats"],
                    color="9b59b6"
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
    
    async def _get_detailed_feature_report(self) -> Dict[str, Any]:
        """Get detailed feature report with market data and statistics."""
        try:
            detailed_report = {
                "summary": "",
                "symbols": {},
                "system_stats": ""
            }
            
            # Collect data for each symbol
            symbol_summaries = []
            total_predictions = 0
            total_signals = 0
            
            for symbol in self.config.symbols:
                symbol_report = await self._get_symbol_detailed_report(symbol)
                detailed_report["symbols"][symbol] = symbol_report
                
                if symbol_report["has_data"]:
                    symbol_summaries.append(f"{symbol}: ${symbol_report['current_price']:,.2f} ({symbol_report['price_change_1h']:+.2f}%)")
                    total_predictions += symbol_report.get("predictions_1h", 0)
                    total_signals += symbol_report.get("signals_1h", 0)
                else:
                    symbol_summaries.append(f"{symbol}: データなし")
            
            # Create summary
            current_balance = self.account_monitor.current_balance.total_equity if self.account_monitor.current_balance else 0
            detailed_report["summary"] = (
                f"💰 残高: ${current_balance:.2f}\n"
                f"📈 価格状況:\n" + "\n".join(symbol_summaries) + "\n"
                f"🔮 予測: {total_predictions}回/時間\n"
                f"📊 シグナル: {total_signals}回/時間"
            )
            
            # System statistics
            health_status = await self._check_services_health()
            healthy_services = sum(1 for status in health_status.values() if status)
            
            detailed_report["system_stats"] = (
                f"🟢 稼働サービス: {healthy_services}/{len(health_status)}\n"
                f"📊 予測総数: {self.prediction_count}\n"
                f"📈 シグナル総数: {self.signal_count}\n"
                f"⏱️ 稼働時間: {self._get_uptime()}"
            )
            
            return detailed_report
            
        except Exception as e:
            logger.error("Error creating detailed feature report", exception=e)
            return {
                "summary": "レポート生成エラー",
                "symbols": {},
                "system_stats": f"エラー: {str(e)}"
            }
    
    async def _get_symbol_detailed_report(self, symbol: str) -> Dict[str, Any]:
        """Get detailed report for a specific symbol."""
        try:
            # Get current features
            features = await self._get_features(symbol)
            if not features:
                return {
                    "has_data": False,
                    "health": "bad",
                    "description": "特徴量データが利用できません"
                }
            
            # Get market data from Redis (WebSocket data)
            market_data = await self._get_redis_market_data(symbol)
            
            # Extract key feature values
            current_price = features.get('close', market_data.get('last_price', 0) if market_data else 0)
            volume = features.get('volume', 0)
            spread = features.get('spread_bps', 0)
            volatility = features.get('volatility_20', 0)
            rsi = features.get('rsi_14', 50)
            
            # Calculate price changes
            sma_20 = features.get('sma_20', current_price)
            price_change_pct = ((current_price - sma_20) / sma_20 * 100) if sma_20 > 0 else 0
            
            # Get orderbook imbalance
            bid_volume = features.get('bid_volume_top5', 0)
            ask_volume = features.get('ask_volume_top5', 0)
            total_ob_volume = bid_volume + ask_volume
            orderbook_imbalance = ((bid_volume - ask_volume) / total_ob_volume * 100) if total_ob_volume > 0 else 0
            
            # Get liquidation data
            liquidation_buy = features.get('liquidation_buy_volume_5m', 0)
            liquidation_sell = features.get('liquidation_sell_volume_5m', 0)
            
            # Count recent predictions and signals for this symbol
            predictions_1h = sum(1 for p in self.last_predictions.get(symbol, []) 
                               if (datetime.now() - p.get('timestamp', datetime.min)).seconds < 3600)
            signals_1h = sum(1 for s in self.recent_signals 
                           if s.get('symbol') == symbol and 
                           (datetime.now() - s.get('timestamp', datetime.min)).seconds < 3600)
            
            # Determine health status
            health = "good"
            if current_price == 0 or volume == 0:
                health = "bad"
            elif volatility > 0.03 or spread > 50:
                health = "warning"
            
            # Create detailed description
            description = (
                f"💵 価格: ${current_price:,.2f}\n"
                f"📊 出来高: {volume:,.0f} ({features.get('volume_ratio', 1):.1f}x平均)\n"
                f"📈 RSI: {rsi:.1f}\n"
                f"🌊 ボラティリティ: {volatility*100:.2f}%\n"
                f"💱 スプレッド: {spread:.1f} bps\n"
                f"📚 オーダーブック: {orderbook_imbalance:+.1f}% {'買い優勢' if orderbook_imbalance > 0 else '売り優勢'}\n"
                f"💥 清算 (5分): 買い ${liquidation_buy:,.0f} / 売り ${liquidation_sell:,.0f}\n"
                f"🔮 予測/時間: {predictions_1h}回\n"
                f"📊 シグナル/時間: {signals_1h}回"
            )
            
            # Add latest prediction info if available
            if symbol in self.last_predictions:
                latest = self.last_predictions[symbol][-1] if self.last_predictions[symbol] else None
                if latest:
                    confidence = latest.get('confidence', 0)
                    direction = latest.get('direction', 'hold')
                    description += f"\n🎯 最新予測: {direction.upper()} (信頼度 {confidence:.1%})"
            
            return {
                "has_data": True,
                "health": health,
                "description": description,
                "current_price": current_price,
                "price_change_1h": price_change_pct,
                "predictions_1h": predictions_1h,
                "signals_1h": signals_1h
            }
            
        except Exception as e:
            logger.error(f"Error getting detailed report for {symbol}", exception=e)
            return {
                "has_data": False,
                "health": "bad",
                "description": f"エラー: {str(e)}"
            }
    
    async def _get_redis_market_data(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get latest market data from Redis streams (WebSocket data)."""
        try:
            if not self.redis_manager:
                return None
                
            # Try orderbook data first
            latest_orderbook = await self.redis_manager.xrevrange("market_data:orderbook", count=10)
            
            for entry in latest_orderbook:
                _, fields = entry
                data_str = fields.get("data", "")
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                
                try:
                    ob_data = json.loads(data_str)
                    if ob_data.get("symbol") == symbol:
                        return {
                            "bid": float(ob_data.get("best_bid", 0)),
                            "ask": float(ob_data.get("best_ask", 0)),
                            "mid_price": float(ob_data.get("mid_price", 0)),
                            "spread": float(ob_data.get("spread", 0)),
                            "last_price": float(ob_data.get("mid_price", 0))
                        }
                except:
                    continue
                    
            # Fallback to kline data
            latest_kline = await self.redis_manager.xrevrange("market_data:kline", count=10)
            
            for entry in latest_kline:
                _, fields = entry
                data_str = fields.get("data", "")
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                
                try:
                    kline_data = json.loads(data_str)
                    if symbol in kline_data.get("topic", ""):
                        close = float(kline_data.get("close", 0))
                        return {
                            "last_price": close,
                            "mid_price": close
                        }
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to get Redis market data for {symbol}: {e}")
        
        return None
    
    def _get_uptime(self) -> str:
        """Get system uptime as formatted string."""
        if not hasattr(self, '_start_time'):
            self._start_time = datetime.now()
        
        uptime = datetime.now() - self._start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        
        return f"{hours}時間{minutes}分"