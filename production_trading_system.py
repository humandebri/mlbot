#!/usr/bin/env python3
"""
本番環境用トレーディングシステム（設定を明示的にオーバーライド）
"""
import os
# 環境変数を最初に設定
os.environ['BYBIT__TESTNET'] = 'false'
os.environ['ENVIRONMENT'] = 'production'

# Load .env file
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

import asyncio
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 設定を読み込む前に環境変数を設定済み
from src.common.config import settings
# 強制的に本番設定
settings.bybit.testnet = False

from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier
from src.common.account_monitor import AccountMonitor
from src.ingestor.main import BybitIngestor
from src.feature_hub.main import FeatureHub
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.order_router.main import OrderRouter
from src.order_router.order_executor import OrderExecutor
from src.order_router.position_manager import PositionManager
from src.common.bybit_client import BybitRESTClient
from src.common.database import create_trading_tables, save_trade, save_position

logger = get_logger(__name__)


class ProductionTradingSystem:
    """本番環境用トレーディングシステム"""
    
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
        
        # Initialize Order Executor for actual trading
        self.order_executor = OrderExecutor(self.bybit_client)
        
        # Initialize Position Manager for tracking positions
        self.position_manager = PositionManager()
        
        # Initialize inference engine
        inference_config = InferenceConfig(
            model_path=settings.model.model_path,
            enable_batching=True,
            confidence_threshold=0.6
        )
        self.inference_engine = InferenceEngine(inference_config)
        
        # Track last balance notification time
        self.last_balance_notification = None
        
        # Track open positions
        self.open_positions = {}
    
    async def start(self):
        """Start all system components."""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("Starting Production Trading System")
        logger.info(f"Testnet mode: {settings.bybit.testnet} (should be False)")
        
        try:
            # Create trading tables
            create_trading_tables()
            logger.info("Trading tables initialized")
            
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
                    "API Status": "✅ 本番環境接続",
                    "Trading Mode": "🔴 LIVE TRADING"
                }
                discord_notifier.send_notification(
                    title="🚀 本番取引システム起動",
                    description=f"実際の資金で取引を開始します",
                    color="ff0000",  # 赤色で警告
                    fields=fields
                )
                logger.info(f"Initial balance retrieved: ${balance.total_equity:.2f}")
            else:
                logger.warning("Failed to retrieve initial balance")
            
            # Start other components as background tasks
            self.tasks.append(
                asyncio.create_task(self.ingestor.start(), name="Ingestor")
            )
            logger.info("Ingestor background task created")
            
            # Start Order Executor as background task
            self.tasks.append(
                asyncio.create_task(self.order_executor.start(), name="OrderExecutor")
            )
            logger.info("Order Executor background task created")
            
            # Start Position Manager (non-blocking)
            try:
                # Position Manager doesn't have a start method that hangs
                logger.info("Position Manager initialized for tracking positions")
            except Exception as e:
                logger.warning(f"Position Manager start issue: {e}")
            
            # Wait for initial data accumulation
            await asyncio.sleep(10)
            logger.info("Initial data accumulation complete")
            
            # Start FeatureHub and OrderRouter as background tasks
            self.running = True
            
            # FeatureHubとOrderRouterをバックグラウンドタスクとして起動
            logger.info("Starting FeatureHub and OrderRouter as background tasks")
            
            # FeatureHubをバックグラウンドで起動（実際のメソッドを使用）
            try:
                # FeatureHubのバックグラウンドタスクを起動
                self.tasks.append(
                    asyncio.create_task(self.feature_hub.start(), name="FeatureHub")
                )
                logger.info("FeatureHub background task created")
            except Exception as e:
                logger.warning(f"FeatureHub initialization issue: {e}")
            
            # OrderRouterをバックグラウンドで起動
            try:
                # OrderRouterのバックグラウンドタスクを起動
                self.tasks.append(
                    asyncio.create_task(self.order_router.start(), name="OrderRouter")
                )
                logger.info("OrderRouter background task created")
            except Exception as e:
                logger.warning(f"OrderRouter initialization issue: {e}")
            
            # Wait a bit for background components to initialize
            await asyncio.sleep(5)
            
            # Start other background tasks
            self.tasks.extend([
                asyncio.create_task(self._trading_loop(), name="TradingLoop"),
                asyncio.create_task(self._balance_notification_loop(), name="BalanceNotification"),
                asyncio.create_task(self._health_check_loop(), name="HealthCheck"),
                asyncio.create_task(self._feature_monitor_loop(), name="FeatureMonitor"),
                asyncio.create_task(self._position_monitor_loop(), name="PositionMonitor"),
                asyncio.create_task(self._daily_report_loop(), name="DailyReport")
            ])
            
            logger.info("All background tasks created")
            
            # Wait a bit for balance to be available
            await asyncio.sleep(5)
            
            # Send Discord notification
            try:
                balance_text = "取得中..."
                if self.account_monitor.current_balance:
                    balance_text = f"${self.account_monitor.current_balance.total_equity:.2f}"
                
                discord_notifier.send_notification(
                    title="🚀 本番取引システム起動完了",
                    description=f"24時間取引システムが正常に起動しました",
                    color="00ff00",
                    fields={
                        "💰 残高": balance_text,
                        "🔴 モード": "実際の資金で取引",
                        "✅ 状態": "稼働中",
                        "📊 機能": "日次レポート・部分利確・トレーリングストップ有効",
                        "🕰️ 時刻": "9:00 AM JST日次レポート、毎時残高更新"
                    }
                )
                logger.info("Startup notification sent to Discord")
            except Exception as e:
                logger.error(f"Failed to send startup notification: {e}")
            
            logger.info("🎉 Production Trading System fully started and operational!")
            
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    
    async def _feature_monitor_loop(self):
        """特徴量生成を監視"""
        await asyncio.sleep(30)  # 初期待機
        
        while self.running:
            try:
                feature_status = {}
                for symbol in settings.bybit.symbols:
                    features = self.feature_hub.get_latest_features(symbol)
                    feature_status[symbol] = len(features) if features else 0
                
                # 特徴量が生成されているかチェック
                if any(count > 0 for count in feature_status.values()):
                    logger.info(f"✅ Features generating: {feature_status}")
                    
                    # 一度だけ通知
                    if not hasattr(self, '_feature_notification_sent'):
                        self._feature_notification_sent = True
                        discord_notifier.send_notification(
                            title="✅ 特徴量生成開始",
                            description="システムが正常に動作しています",
                            color="00ff00",
                            fields={
                                "BTCUSDT": f"{feature_status.get('BTCUSDT', 0)} features",
                                "ETHUSDT": f"{feature_status.get('ETHUSDT', 0)} features",
                                "Status": "取引シグナル待機中"
                            }
                        )
                else:
                    logger.warning(f"⚠️ No features yet: {feature_status}")
                
                await asyncio.sleep(60)  # 1分ごとにチェック
                
            except Exception as e:
                logger.error(f"Feature monitor error: {e}")
                await asyncio.sleep(60)
    
    async def stop(self):
        """Stop all system components."""
        if not self.running:
            return
        
        logger.info("Stopping Production Trading System")
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
        
        # Stop components safely
        try:
            await self.account_monitor.stop()
        except Exception as e:
            logger.warning(f"Error stopping account monitor: {e}")
        
        try:
            await self.ingestor.stop()
        except Exception as e:
            logger.warning(f"Error stopping ingestor: {e}")
        
        try:
            if hasattr(self.feature_hub, 'stop'):
                await self.feature_hub.stop()
        except Exception as e:
            logger.warning(f"Error stopping feature hub: {e}")
        
        try:
            if hasattr(self.order_router, 'stop'):
                await self.order_router.stop()
        except Exception as e:
            logger.warning(f"Error stopping order router: {e}")
        
        try:
            await self.order_executor.stop()
        except Exception as e:
            logger.warning(f"Error stopping order executor: {e}")
        
        try:
            if hasattr(self.position_manager, 'stop'):
                await self.position_manager.stop()
        except Exception as e:
            logger.warning(f"Error stopping position manager: {e}")
        
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
        
        logger.info("Production Trading System stopped")
    
    async def _trading_loop(self):
        """Main trading loop that processes features and makes predictions."""
        loop_count = 0
        logger.info("Starting trading loop")
        
        # Wait for FeatureHub to start processing data
        await asyncio.sleep(60)  # 1分待機
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
                                    base_position_pct = 0.2  # 20% of equity
                                    position_size = self.account_monitor.current_balance.total_equity * base_position_pct
                                    
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
                                        "Position Size": f"${position_size:.2f}",
                                        "⚠️ Mode": "🔴 LIVE TRADING"
                                    }
                                    
                                    discord_notifier.send_notification(
                                        title="🚨 本番取引シグナル",
                                        description=f"高信頼度シグナル検出 - {symbol}",
                                        color="00ff00" if prediction > 0 else "ff0000",
                                        fields=fields
                                    )
                                    
                                    # 実際の取引実行
                                    try:
                                        # リスク管理チェック
                                        if self.order_router.risk_manager.can_trade(
                                            symbol=symbol,
                                            side="buy" if prediction > 0 else "sell",
                                            size=position_size
                                        ):
                                            # 取引価格の決定（スリッページ考慮）
                                            current_price = features.get('close', 50000)
                                            slippage = 0.001  # 0.1%
                                            
                                            if prediction > 0:  # Buy
                                                order_price = current_price * (1 + slippage)
                                                order_side = "buy"
                                            else:  # Sell
                                                order_price = current_price * (1 - slippage)
                                                order_side = "sell"
                                            
                                            # 損切り・利確価格の計算
                                            stop_loss_pct = 0.02  # 2% 損切り
                                            take_profit_pct = 0.03  # 3% 利確
                                            
                                            if order_side == "buy":
                                                stop_loss_price = current_price * (1 - stop_loss_pct)
                                                take_profit_price = current_price * (1 + take_profit_pct)
                                            else:
                                                stop_loss_price = current_price * (1 + stop_loss_pct)
                                                take_profit_price = current_price * (1 - take_profit_pct)
                                            
                                            # 注文実行（新しいcreate_orderメソッドを使用）
                                            order_result = await self.bybit_client.create_order(
                                                symbol=symbol,
                                                side=order_side,
                                                order_type="limit",
                                                qty=position_size / current_price,  # USD to quantity
                                                price=order_price,
                                                stop_loss=stop_loss_price,
                                                take_profit=take_profit_price
                                            )
                                            
                                            if order_result:
                                                order_id = order_result.get("orderId")
                                                position_id = f"pos_{order_id}"
                                                
                                                # Save to database
                                                save_position(
                                                    position_id=position_id,
                                                    symbol=symbol,
                                                    side=order_side,
                                                    entry_price=order_price,
                                                    quantity=position_size / current_price,
                                                    stop_loss=stop_loss_price,
                                                    take_profit=take_profit_price,
                                                    metadata={
                                                        "signal_confidence": confidence,
                                                        "expected_pnl": prediction,
                                                        "signal_time": datetime.now().isoformat()
                                                    }
                                                )
                                                
                                                save_trade(
                                                    trade_id=order_id,
                                                    position_id=position_id,
                                                    symbol=symbol,
                                                    side=order_side,
                                                    order_type="limit",
                                                    quantity=position_size / current_price,
                                                    price=order_price,
                                                    metadata={
                                                        "signal_confidence": confidence,
                                                        "expected_pnl": prediction
                                                    }
                                                )
                                                
                                                # PositionManagerに通知
                                                await self.position_manager.open_position(
                                                    position_id=position_id,
                                                    symbol=symbol,
                                                    side=order_side,
                                                    entry_price=order_price,
                                                    quantity=position_size / current_price,
                                                    stop_loss=stop_loss_price,
                                                    take_profit=take_profit_price,
                                                    metadata={
                                                        "signal_confidence": confidence,
                                                        "expected_pnl": prediction,
                                                        "signal_time": datetime.now().isoformat()
                                                    }
                                                )
                                                
                                                logger.info(
                                                    f"🎯 Order placed: {symbol} {order_side} "
                                                    f"qty={position_size/current_price:.4f} @ ${order_price:.2f} "
                                                    f"order_id={order_id} "
                                                    f"SL=${stop_loss_price:.2f} TP=${take_profit_price:.2f}"
                                                )
                                            
                                                # 取引実行通知
                                                discord_notifier.send_notification(
                                                    title="✅ 取引実行",
                                                    description=f"{symbol} {order_side.upper()} 注文送信",
                                                    color="00ff00",
                                                    fields={
                                                        "注文ID": order_id,
                                                        "数量": f"{position_size/current_price:.4f}",
                                                        "価格": f"${order_price:.2f}",
                                                        "損切り": f"${stop_loss_price:.2f}",
                                                        "利確": f"${take_profit_price:.2f}",
                                                        "ポジションサイズ": f"${position_size:.2f}",
                                                        "Status": "⚡ LIVE ORDER"
                                                    }
                                                )
                                            
                                        else:
                                            logger.warning(
                                                f"Risk check failed for {symbol} - trade blocked"
                                            )
                                            discord_notifier.send_notification(
                                                title="⚠️ 取引ブロック",
                                                description="リスク管理により取引がブロックされました",
                                                color="ff9900",
                                                fields={
                                                    "Symbol": symbol,
                                                    "Reason": "リスク上限到達または取引条件不適合"
                                                }
                                            )
                                            
                                    except Exception as e:
                                        logger.error(f"Order execution error for {symbol}: {e}")
                                        discord_notifier.send_notification(
                                            title="❌ 取引エラー",
                                            description=f"注文実行エラー: {str(e)}",
                                            color="ff0000",
                                            fields={
                                                "Symbol": symbol,
                                                "Error": str(e)
                                            }
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
                        "Peak Balance": f"${stats.get('peak_balance', 0):,.2f}",
                        "Mode": "🔴 LIVE TRADING"
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
                        discord_notifier.send_notification(
                            title="⚠️ システム警告",
                            description="Bybit API接続が失われました - アカウント残高を取得できません",
                            color="ff9900"
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
    
    async def _position_monitor_loop(self):
        """Monitor open positions and manage exits."""
        await asyncio.sleep(30)  # Initial wait
        
        while self.running:
            try:
                # Get open positions from Bybit
                positions = await self.bybit_client.get_open_positions()
                
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
                            
                            # Update position in PositionManager
                            position_id = f"pos_{symbol}_{side}"
                            await self.position_manager.update_position(
                                position_id=position_id,
                                current_price=mark_price,
                                pnl=unrealized_pnl
                            )
                            
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
                open_orders = await self.bybit_client.get_open_orders()
                if open_orders:
                    logger.info(f"Active orders: {len(open_orders)}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(60)
    
    async def _daily_report_loop(self):
        """Send daily report at 9:00 AM JST."""
        while self.running:
            try:
                # Get current time in JST (UTC+9)
                from datetime import timezone, timedelta
                jst = timezone(timedelta(hours=9))
                now = datetime.now(jst)
                
                # Calculate next 9:00 AM JST
                next_report = now.replace(hour=9, minute=0, second=0, microsecond=0)
                if now.hour >= 9:
                    # If past 9 AM, schedule for tomorrow
                    next_report += timedelta(days=1)
                
                # Wait until next report time
                wait_seconds = (next_report - now).total_seconds()
                logger.info(f"Next daily report scheduled for {next_report} JST ({wait_seconds/3600:.1f} hours)")
                await asyncio.sleep(wait_seconds)
                
                # Generate daily report
                await self._send_daily_report()
                
                # Wait a bit to avoid duplicate sends
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in daily report loop: {e}")
                await asyncio.sleep(3600)  # Try again in an hour
    
    async def _send_daily_report(self):
        """Send comprehensive daily report."""
        try:
            # Get account balance
            balance = self.account_monitor.current_balance
            if not balance:
                logger.warning("No balance information for daily report")
                return
            
            # Get trading statistics from database
            from src.common.database import get_duckdb_connection
            conn = get_duckdb_connection()
            
            # Today's trades
            today_trades = conn.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    SUM(fees) as total_fees
                FROM positions
                WHERE DATE(closed_at) = CURRENT_DATE
                AND status = 'closed'
            """).fetchone()
            
            # Calculate win rate
            total = today_trades[0] or 0
            wins = today_trades[1] or 0
            win_rate = (wins / total * 100) if total > 0 else 0
            
            # Get open positions
            open_positions = await self.bybit_client.get_open_positions()
            
            # Create report
            fields = {
                "📊 残高": f"${balance.total_equity:,.2f}",
                "💵 利用可能": f"${balance.available_balance:,.2f}",
                "📈 未実現損益": f"${balance.unrealized_pnl:,.2f}",
                "📉 ポジション数": str(len(open_positions)),
                "🎯 今日の取引": str(total),
                "✅ 勝率": f"{win_rate:.1f}%",
                "💰 今日の損益": f"${today_trades[3] or 0:,.2f}",
                "💸 手数料": f"${today_trades[4] or 0:,.2f}",
                "📅 日付": datetime.now().strftime("%Y-%m-%d")
            }
            
            discord_notifier.send_notification(
                title="📊 日次レポート (9:00 AM JST)",
                description="24時間の取引結果サマリー",
                color="00ff00" if (today_trades[3] or 0) >= 0 else "ff0000",
                fields=fields
            )
            
            logger.info("Daily report sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send daily report: {e}")
    
    async def _check_trailing_stop(self, position: Dict[str, Any]) -> None:
        """Check and update trailing stop for a position."""
        symbol = position.get("symbol")
        side = position.get("side")
        size = float(position.get("size", 0))
        entry_price = float(position.get("avgPrice", 0))
        mark_price = float(position.get("markPrice", 0))
        unrealized_pnl = float(position.get("unrealizedPnl", 0))
        
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
            success = await self.bybit_client.set_stop_loss(symbol, new_stop_loss)
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
        
        # Partial take profit logic: 
        # At 1.5% profit, close 50% of position
        # At 3% profit, close another 25% (total 75%)
        position_id = f"pos_{symbol}_{side}"
        closed_pct = self.open_positions.get(position_id, {}).get("closed_pct", 0)
        
        if profit_pct >= 3.0 and closed_pct < 75:
            # Close 25% more (total 75%)
            close_size = size * 0.25
            close_side = "Sell" if side == "Buy" else "Buy"
            
            result = await self.bybit_client.create_order(
                symbol=symbol,
                side=close_side,
                order_type="Market",
                qty=close_size,
                reduce_only=True
            )
            
            if result:
                self.open_positions[position_id] = {"closed_pct": 75}
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
            
            result = await self.bybit_client.create_order(
                symbol=symbol,
                side=close_side,
                order_type="Market",
                qty=close_size,
                reduce_only=True
            )
            
            if result:
                self.open_positions[position_id] = {"closed_pct": 50}
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


async def main():
    """Run the production trading system."""
    system = ProductionTradingSystem()
    
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
    logger.info("Starting Production Trading Bot")
    print("⚠️  警告: 本番環境で実際の資金を使用します")
    print("💰 残高: 約$100 (USDT)")
    print("🔴 LIVE TRADING MODE")
    asyncio.run(main())