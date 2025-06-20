#!/usr/bin/env python3
"""
修正版：FeatureHub問題を解決したトレーディングシステム
"""

import asyncio
import signal
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.common.config import settings
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier
from src.common.account_monitor import AccountMonitor
from src.ingestor.main import BybitIngestor
from src.feature_hub.main import FeatureHub
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.order_router.main import OrderRouter
from src.order_router.order_executor import OrderExecutor
from src.common.bybit_client import BybitRESTClient

logger = get_logger(__name__)


class FixedTradingSystem:
    """修正版：FeatureHub起動問題を解決"""
    
    def __init__(self):
        self.running = False
        self.tasks = []
        
        # Initialize components
        self.ingestor = BybitIngestor()
        self.feature_hub = FeatureHub()
        self.order_router = OrderRouter()
        
        # Initialize Bybit REST client
        self.bybit_client = BybitRESTClient()
        
        # Initialize Account Monitor with 60 second interval
        self.account_monitor = AccountMonitor(check_interval=60)
        
        # Initialize inference engine
        inference_config = InferenceConfig(
            model_path=settings.model.model_path,
            enable_batching=True,
            confidence_threshold=0.6
        )
        self.inference_engine = InferenceEngine(inference_config)
        
        # Track last balance notification time
        self.last_balance_notification = None
    
    async def start(self):
        """Start all system components."""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("Starting Fixed Trading System")
        
        try:
            # Load ML model
            self.inference_engine.load_model()
            logger.info(f"Model loaded: {settings.model.model_path}")
            
            # Start account monitor first to get initial balance
            await self.account_monitor.start()
            logger.info("Account monitor started - retrieving real balance")
            
            # Wait for initial balance
            await asyncio.sleep(2)
            
            # Get initial balance and send notification
            if self.account_monitor.current_balance:
                balance = self.account_monitor.current_balance
                stats = self.account_monitor.get_performance_stats()
                
                # Send initial balance notification
                fields = {
                    "Balance": f"${balance.total_equity:,.2f}",
                    "Available": f"${balance.available_balance:,.2f}",
                    "Unrealized PnL": f"${balance.unrealized_pnl:,.2f}",
                    "Free Margin": f"{balance.free_margin_pct:.1f}%",
                    "API Status": "Connected to Real Bybit API",
                    "FeatureHub": "✅ 修正版で起動"
                }
                discord_notifier.send_notification(
                    title="💰 修正版システム起動 - 残高確認",
                    description=f"FeatureHub問題を修正した新システム",
                    color="03b2f8",
                    fields=fields
                )
                logger.info(f"Initial balance retrieved: ${balance.total_equity:.2f}")
            else:
                logger.warning("Failed to retrieve initial balance")
            
            # Start other components
            await self.ingestor.start()
            logger.info("Ingestor started")
            
            await self.order_router.start()
            logger.info("Order Router started")
            
            # Wait for initial data accumulation
            await asyncio.sleep(10)
            logger.info("Initial data accumulation complete")
            
            # Start FeatureHub in a different way
            self.running = True
            
            # FeatureHubの内部タスクを直接起動
            logger.info("Starting FeatureHub with fixed approach")
            await self._start_feature_hub_fixed()
            
            # Start other background tasks
            self.tasks = [
                asyncio.create_task(self._trading_loop(), name="TradingLoop"),
                asyncio.create_task(self._balance_notification_loop(), name="BalanceNotification"),
                asyncio.create_task(self._health_check_loop(), name="HealthCheck")
            ]
            
            logger.info("All background tasks created")
            
            # Send Discord notification
            discord_notifier.send_system_status(
                "online",
                f"修正版Trading System started\n" +
                f"Account Balance: ${self.account_monitor.current_balance.total_equity:.2f}\n" +
                f"FeatureHub: ✅ 修正済み"
            )
            
            logger.info("Fixed Trading System fully started")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    async def _start_feature_hub_fixed(self):
        """修正版：FeatureHubを正しく起動"""
        try:
            # FeatureHubの初期化処理を直接実行
            self.feature_hub.running = True
            
            # Redis接続
            from src.common.database import get_redis_client
            from src.common.redis_manager import RedisStreams
            
            self.feature_hub.redis_client = await get_redis_client()
            self.feature_hub.redis_streams = RedisStreams(self.feature_hub.redis_client)
            
            # Feature engines初期化
            await self.feature_hub._initialize_feature_engines()
            
            # Consumer groups設定
            await self.feature_hub._setup_consumer_groups()
            
            # 各処理タスクを個別に起動（gather()を使わない）
            self.tasks.extend([
                asyncio.create_task(self.feature_hub._process_market_data(), name="ProcessMarketData"),
                asyncio.create_task(self.feature_hub._publish_features(), name="PublishFeatures"),
                asyncio.create_task(self.feature_hub._cleanup_cache(), name="CleanupCache"),
                asyncio.create_task(self.feature_hub._log_statistics(), name="LogStatistics")
            ])
            
            logger.info("FeatureHub tasks started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start FeatureHub: {e}")
            raise
    
    async def stop(self):
        """Stop all system components."""
        if not self.running:
            return
        
        logger.info("Stopping Fixed Trading System")
        self.running = False
        
        # Cancel tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete with timeout
        try:
            await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Some tasks did not complete within timeout")
        
        # Stop components
        await self.account_monitor.stop()
        await self.ingestor.stop()
        await self.feature_hub.stop()
        await self.order_router.stop()
        
        # Send final balance notification
        if self.account_monitor.current_balance:
            balance = self.account_monitor.current_balance
            stats = self.account_monitor.get_performance_stats()
            
            fields = {
                "Final Balance": f"${balance.total_equity:,.2f}",
                "Total Return": f"{stats.get('total_return_pct', 0):.2f}%",
                "Max Drawdown": f"{stats.get('max_drawdown_pct', 0):.2f}%",
                "Unrealized PnL": f"${balance.unrealized_pnl:,.2f}"
            }
            discord_notifier.send_notification(
                title="🛑 System Stopped",
                description="Trading system shutdown - Final balance report",
                color="ff9900",
                fields=fields
            )
        
        logger.info("Fixed Trading System stopped")
    
    async def _trading_loop(self):
        """Main trading loop that processes features and makes predictions."""
        loop_count = 0
        logger.info("Starting trading loop")
        
        # Wait for FeatureHub to start processing data
        await asyncio.sleep(30)
        logger.info("FeatureHub warm-up complete, starting predictions")
        
        while self.running:
            try:
                loop_count += 1
                
                # Get latest features from FeatureHub
                for symbol in settings.bybit.symbols:
                    features = self.feature_hub.get_latest_features(symbol)
                    
                    if loop_count % 30 == 0:  # Log every 30 seconds
                        logger.info(f"Loop {loop_count}: {symbol} features={len(features) if features else 0}")
                    
                    if features and len(features) > 10:  # Ensure we have enough features
                        try:
                            # Make prediction
                            result = self.inference_engine.predict(features)
                            
                            prediction = result["predictions"][0] if result["predictions"] else 0
                            confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                            
                            if loop_count % 30 == 0:  # Log every 30 seconds
                                logger.info(f"Prediction {symbol}: pred={prediction:.4f}, conf={confidence:.2%}")
                            
                            if confidence > 0.6:
                                # Get current balance for position sizing
                                position_size = 0
                                if self.account_monitor.current_balance:
                                    # Use Kelly criterion with current balance
                                    base_position_pct = float(settings.trading.position_size_pct)
                                    position_size = self.account_monitor.get_current_position_size(base_position_pct)
                                    
                                    logger.info(
                                        f"🚨 HIGH CONFIDENCE Signal for {symbol}: pred={prediction:.4f}, conf={confidence:.2%}, " +
                                        f"position_size=${position_size:.2f}"
                                    )
                                    
                                    # Send Discord notification with real balance info
                                    fields = {
                                        "Symbol": symbol,
                                        "Side": "BUY" if prediction > 0 else "SELL",
                                        "Price": f"${features.get('close', 50000):,.2f}",
                                        "Confidence": f"{confidence:.2%}",
                                        "Expected PnL": f"{prediction:.2%}",
                                        "Account Balance": f"${self.account_monitor.current_balance.total_equity:,.2f}",
                                        "Position Size": f"${position_size:.2f}"
                                    }
                                    
                                    discord_notifier.send_notification(
                                        title="🚨 Trade Signal with Real Balance",
                                        description=f"High confidence signal for {symbol}",
                                        color="00ff00" if prediction > 0 else "ff0000",
                                        fields=fields
                                    )
                                else:
                                    logger.warning("No balance information available for position sizing")
                                
                        except Exception as e:
                            logger.error(f"Prediction error for {symbol}: {e}")
                    else:
                        if loop_count % 60 == 0:  # Log every 60 seconds for missing features
                            logger.warning(f"Insufficient features for {symbol}: count={len(features) if features else 0}")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def _balance_notification_loop(self):
        """Send periodic balance notifications."""
        notification_interval = 3600  # 1 hour
        
        while self.running:
            try:
                # Wait for the interval
                await asyncio.sleep(notification_interval)
                
                # Get current balance and stats
                if self.account_monitor.current_balance:
                    balance = self.account_monitor.current_balance
                    stats = self.account_monitor.get_performance_stats()
                    
                    # Send balance update notification
                    fields = {
                        "Balance": f"${balance.total_equity:,.2f}",
                        "Available": f"${balance.available_balance:,.2f}",
                        "Unrealized PnL": f"${balance.unrealized_pnl:,.2f}",
                        "Total Return": f"{stats.get('total_return_pct', 0):.2f}%",
                        "Max Drawdown": f"{stats.get('max_drawdown_pct', 0):.2f}%",
                        "Peak Balance": f"${stats.get('peak_balance', 0):,.2f}"
                    }
                    
                    discord_notifier.send_notification(
                        title="📊 Hourly Balance Update",
                        description="Real-time account status from Bybit API",
                        color="03b2f8",
                        fields=fields
                    )
                    
                    logger.info(
                        f"Balance notification sent: ${balance.total_equity:.2f} " +
                        f"(return: {stats.get('total_return_pct', 0):.2f}%)"
                    )
                
            except Exception as e:
                logger.error(f"Error in balance notification loop: {e}")
    
    async def _health_check_loop(self):
        """Monitor system health including API connectivity."""
        while self.running:
            try:
                # Check component health and feature availability
                health_status = {
                    "ingestor": self.ingestor.running if hasattr(self.ingestor, 'running') else False,
                    "feature_hub": self.feature_hub.running if hasattr(self.feature_hub, 'running') else False,
                    "order_router": self.order_router.running if hasattr(self.order_router, 'running') else False,
                    "model": self.inference_engine.onnx_session is not None,
                    "account_monitor": self.account_monitor._running,
                    "api_connected": self.account_monitor.current_balance is not None
                }
                
                # Count features for each symbol
                feature_counts = {}
                for symbol in settings.bybit.symbols:
                    features = self.feature_hub.get_latest_features(symbol)
                    feature_counts[symbol] = len(features) if features else 0
                
                # Get account info
                account_info = {}
                if self.account_monitor.current_balance:
                    balance = self.account_monitor.current_balance
                    account_info = {
                        "equity": balance.total_equity,
                        "available": balance.available_balance,
                        "unrealized_pnl": balance.unrealized_pnl,
                        "free_margin_pct": balance.free_margin_pct
                    }
                
                logger.info(f"System health: {health_status}")
                logger.info(f"Feature counts: {feature_counts}")
                logger.info(f"Account info: {account_info}")
                
                unhealthy = [k for k, v in health_status.items() if not v]
                if unhealthy:
                    logger.warning(f"Unhealthy components: {unhealthy}")
                    
                    # Send alert if API disconnected
                    if "api_connected" in unhealthy:
                        discord_notifier.send_system_status(
                            "warning",
                            "⚠️ Bybit API connection lost - unable to retrieve account balance"
                        )
                
                # Check for task failures
                for task in self.tasks:
                    if task.done() and not task.cancelled():
                        exception = task.exception()
                        if exception:
                            logger.error(f"Task {task.get_name()} failed with exception: {exception}")
                
                await asyncio.sleep(60)  # Check every 60 seconds
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(60)


async def main():
    """Run the fixed trading system."""
    system = FixedTradingSystem()
    
    # Setup signal handlers
    def signal_handler(sig):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(system.stop())
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler, sig)
    
    try:
        await system.start()
        
        # Keep running until stopped
        while system.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        await system.stop()


if __name__ == "__main__":
    logger.info("Starting Fixed Trading Bot")
    asyncio.run(main())