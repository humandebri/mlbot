#!/usr/bin/env python3
"""
本番環境用トレーディングシステム（動的パラメータ版）
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
import time
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
from src.order_router.order_executor import OrderExecutor, ExecutionConfig
from src.order_router.position_manager import PositionManager
from src.order_router.risk_manager import RiskConfig
from src.common.bybit_client import BybitRESTClient
from src.common.database import create_trading_tables, save_trade, save_position

logger = get_logger(__name__)


class ProductionTradingSystem:
    """本番環境用トレーディングシステム（動的パラメータ版）"""
    
    def __init__(self):
        self.running = False
        self.tasks = []
        
        # Initialize components
        self.ingestor = BybitIngestor()
        self.feature_hub = FeatureHub()
        
        # Initialize Bybit REST client
        self.bybit_client = BybitRESTClient()
        
        # Initialize Account Monitor with 60 second interval
        self.account_monitor = AccountMonitor(check_interval=60)
        
        # Signal cooldown tracking to prevent spam
        self.last_signal_time = {}  # {symbol: timestamp}
        self.signal_cooldown = 300  # 5 minutes between signals for same symbol
        
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
        
        # Will be initialized after account balance is retrieved
        self.order_router = None
        self.order_executor = None
        self.position_manager = None
    
    def _create_dynamic_risk_config(self, account_balance: float) -> RiskConfig:
        """アカウント残高に基づいて動的にリスク設定を作成"""
        return RiskConfig(
            # Position limits (% of account)
            max_position_size_usd=account_balance * 0.3,  # 30% of account per position
            max_total_exposure_usd=account_balance * 0.6,  # 60% total exposure
            max_leverage=3.0,  # 3x leverage
            max_positions=3,  # Max 3 concurrent positions
            
            # Loss limits (% of account)
            max_daily_loss_usd=account_balance * 0.1,  # 10% daily loss limit
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
        """アカウント残高に基づいて動的に実行設定を作成"""
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
            max_order_size_usd=account_balance * 0.3,  # Max single order = 30% of account
            
            # Fill improvement
            price_improvement_check=True,
            aggressive_fill_timeout=30,
            
            # Rate limiting
            max_orders_per_second=10.0,
            max_orders_per_symbol=5,
            
            # Monitoring
            latency_threshold_ms=100.0
        )
    
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
            await asyncio.sleep(5)  # Wait longer for balance to be retrieved
            
            # Get initial balance and send notification
            if self.account_monitor.current_balance:
                balance = self.account_monitor.current_balance
                account_balance = balance.total_equity
                
                logger.info(f"Account balance retrieved: ${account_balance:.2f}")
                
                # Initialize dynamic components based on account balance
                risk_config = self._create_dynamic_risk_config(account_balance)
                execution_config = self._create_dynamic_execution_config(account_balance)
                
                # Initialize order router with dynamic risk config
                self.order_router = OrderRouter(risk_config=risk_config)
                
                # Initialize order executor with dynamic execution config
                self.order_executor = OrderExecutor(self.bybit_client, execution_config)
                
                # Initialize position manager
                self.position_manager = PositionManager()
                
                logger.info(f"Dynamic risk parameters initialized for ${account_balance:.2f} account:")
                logger.info(f"  - Max position size: ${risk_config.max_position_size_usd:.2f}")
                logger.info(f"  - Max total exposure: ${risk_config.max_total_exposure_usd:.2f}")
                logger.info(f"  - Max daily loss: ${risk_config.max_daily_loss_usd:.2f}")
                logger.info(f"  - Max order size: ${execution_config.max_order_size_usd:.2f}")
                
                # Send initial balance notification
                fields = {
                    "Balance": f"${balance.total_equity:,.2f}",
                    "Available": f"${balance.available_balance:,.2f}",
                    "Unrealized PnL": f"${balance.unrealized_pnl:,.2f}",
                    "Free Margin": f"{balance.free_margin_pct:.1f}%",
                    "Max Position": f"${risk_config.max_position_size_usd:.2f}",
                    "Max Exposure": f"${risk_config.max_total_exposure_usd:.2f}",
                    "API Status": "✅ 本番環境接続",
                    "Trading Mode": "🔴 LIVE TRADING"
                }
                discord_notifier.send_notification(
                    title="🚀 本番取引システム起動（動的パラメータ）",
                    description=f"アカウント残高に基づいてパラメータを自動調整しました",
                    color="ff0000",  # 赤色で警告
                    fields=fields
                )
            else:
                logger.error("Failed to retrieve initial balance - using default parameters")
                # Use minimal safe defaults if balance unavailable
                self.order_router = OrderRouter()
                self.order_executor = OrderExecutor(self.bybit_client)
                self.position_manager = PositionManager()
            
            # Set leverage for all trading symbols
            leverage = 3  # 3x leverage as configured
            for symbol in settings.bybit.symbols:
                try:
                    leverage_set = await self.bybit_client.set_leverage(symbol, leverage)
                    if leverage_set:
                        logger.info(f"Leverage set to {leverage}x for {symbol}")
                    else:
                        logger.error(f"Failed to set leverage for {symbol}")
                except Exception as e:
                    logger.error(f"Error setting leverage for {symbol}: {e}")
            
            # Start components
            self.running = True
            
            # Start order router (includes risk manager)
            self.order_router.running = True
            self.order_router.risk_manager._circuit_breaker_active = False
            self.order_router.risk_manager._daily_loss = 0
            self.order_router.risk_manager._current_drawdown = 0
            self.order_router.risk_manager._active_positions = {}
            self.order_router.risk_manager._last_trade_time = 0
            logger.info("Order router started")
            
            # Start position manager
            await self.position_manager.start()
            logger.info("Position manager started")
            
            # Start ingestor
            ingestor_task = asyncio.create_task(self.ingestor.start())
            self.tasks.append(ingestor_task)
            logger.info("Ingestor started")
            
            # Start feature hub
            self.feature_hub.running = True
            logger.info("Feature hub marked as running")
            
            # Start background tasks
            self.tasks.extend([
                asyncio.create_task(self._trading_loop()),
                asyncio.create_task(self._balance_notification_loop()),
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._position_monitor_loop()),
                asyncio.create_task(self._daily_report_loop()),
                asyncio.create_task(self._update_risk_parameters_loop()),  # New task
            ])
            
            logger.info("All trading system components started successfully")
            
            # Send startup notification
            await self._send_startup_notification()
            
            # Wait for all tasks
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    async def _update_risk_parameters_loop(self):
        """定期的にアカウント残高をチェックしてリスクパラメータを更新"""
        await asyncio.sleep(300)  # 最初の5分待機
        
        while self.running:
            try:
                if self.account_monitor.current_balance:
                    current_balance = self.account_monitor.current_balance.total_equity
                    
                    # 現在の設定値と比較
                    current_max_position = self.order_router.risk_manager.config.max_position_size_usd
                    expected_max_position = current_balance * 0.3
                    
                    # 10%以上の差があれば更新
                    if abs(current_max_position - expected_max_position) / current_max_position > 0.1:
                        logger.info(f"Updating risk parameters for new balance: ${current_balance:.2f}")
                        
                        # 新しい設定を作成
                        new_risk_config = self._create_dynamic_risk_config(current_balance)
                        new_execution_config = self._create_dynamic_execution_config(current_balance)
                        
                        # RiskManagerの設定を更新
                        self.order_router.risk_manager.config = new_risk_config
                        
                        # OrderExecutorの設定を更新
                        self.order_executor.config = new_execution_config
                        
                        # 通知を送信
                        discord_notifier.send_notification(
                            title="⚙️ リスクパラメータ更新",
                            description="アカウント残高の変化に基づいてパラメータを調整しました",
                            color="03b2f8",
                            fields={
                                "Account Balance": f"${current_balance:,.2f}",
                                "Max Position Size": f"${new_risk_config.max_position_size_usd:.2f}",
                                "Max Total Exposure": f"${new_risk_config.max_total_exposure_usd:.2f}",
                                "Max Daily Loss": f"${new_risk_config.max_daily_loss_usd:.2f}"
                            }
                        )
                
                await asyncio.sleep(300)  # 5分ごとにチェック
                
            except Exception as e:
                logger.error(f"Error updating risk parameters: {e}")
                await asyncio.sleep(300)
    
    async def _trading_loop(self):
        """Main trading loop with improved price fetching."""
        await asyncio.sleep(10)  # Initial wait
        
        loop_count = 0
        
        while self.running:
            try:
                # Check each symbol
                for symbol in settings.bybit.symbols:
                    # Get latest features
                    features = self.feature_hub.get_latest_features(symbol)
                    
                    if features and len(features) > 10:  # Ensure we have enough features
                        try:
                            # Make prediction
                            result = self.inference_engine.predict(features)
                            
                            prediction = result["predictions"][0] if result["predictions"] else 0
                            confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                            
                            if loop_count % 30 == 0:  # Log every 30 seconds
                                logger.info(f"Prediction {symbol}: pred={prediction:.4f}, conf={confidence:.2%}")
                            
                            if confidence > 0.6:
                                # Check signal cooldown to prevent spam
                                current_time = time.time()
                                if symbol in self.last_signal_time:
                                    time_since_last = current_time - self.last_signal_time[symbol]
                                    if time_since_last < self.signal_cooldown:
                                        if loop_count % 30 == 0:  # Log cooldown status occasionally
                                            logger.info(f"⏳ Signal cooldown for {symbol}: {self.signal_cooldown - time_since_last:.0f}s remaining")
                                        continue
                                
                                # Update last signal time
                                self.last_signal_time[symbol] = current_time
                                
                                # Get current balance for position sizing
                                position_size = 0
                                if self.account_monitor.current_balance:
                                    # Dynamic position sizing based on confidence and Kelly criterion
                                    account_equity = self.account_monitor.current_balance.total_equity
                                    
                                    # Base position: 10-30% of equity based on confidence
                                    confidence_multiplier = (confidence - 0.6) / 0.4  # 0 at 60%, 1 at 100%
                                    base_position_pct = 0.1 + (0.2 * confidence_multiplier)  # 10% to 30%
                                    
                                    # Apply Kelly fraction
                                    kelly_fraction = 0.25  # Conservative Kelly
                                    position_pct = base_position_pct * kelly_fraction
                                    
                                    position_size = account_equity * position_pct
                                    
                                    # Ensure within configured limits
                                    max_position = self.order_router.risk_manager.config.max_position_size_usd
                                    position_size = min(position_size, max_position)
                                    
                                    logger.info(
                                        f"🚨 HIGH CONFIDENCE Signal for {symbol}: pred={prediction:.4f}, conf={confidence:.2%}, " +
                                        f"position_pct={position_pct:.1%}, position_size=${position_size:.2f}"
                                    )
                                    
                                    # Get real-time market price from Bybit API
                                    ticker = await self.bybit_client.get_ticker(symbol)
                                    if not ticker or 'lastPrice' not in ticker:
                                        logger.error(f"Failed to get ticker data for {symbol}")
                                        continue
                                    
                                    current_price = float(ticker['lastPrice'])
                                    bid_price = float(ticker.get('bidPrice', current_price))
                                    ask_price = float(ticker.get('askPrice', current_price))
                                    
                                    # Log actual prices
                                    logger.info(f"{symbol} Market Prices - Last: ${current_price:.2f}, Bid: ${bid_price:.2f}, Ask: ${ask_price:.2f}")
                                    
                                    # Send Discord notification
                                    fields = {
                                        "Symbol": symbol,
                                        "Side": "BUY" if prediction > 0 else "SELL",
                                        "Market Price": f"${current_price:,.2f}",
                                        "Bid/Ask": f"${bid_price:,.2f} / ${ask_price:,.2f}",
                                        "Confidence": f"{confidence:.2%}",
                                        "Expected PnL": f"{prediction:.2%}",
                                        "Account Balance": f"${account_equity:,.2f}",
                                        "Position Size": f"${position_size:.2f} ({position_pct:.1%})",
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
                                            # Use actual bid/ask for order placement
                                            if prediction > 0:  # Buy
                                                order_price = ask_price * 1.0001  # Slightly above ask
                                                order_side = "buy"
                                            else:  # Sell
                                                order_price = bid_price * 0.9999  # Slightly below bid
                                                order_side = "sell"
                                            
                                            # 損切り・利確価格の計算（現在価格ベース）
                                            stop_loss_pct = 0.02  # 2% 損切り
                                            take_profit_pct = 0.03  # 3% 利確
                                            
                                            if order_side == "buy":
                                                stop_loss_price = current_price * (1 - stop_loss_pct)
                                                take_profit_price = current_price * (1 + take_profit_pct)
                                            else:
                                                stop_loss_price = current_price * (1 + stop_loss_pct)
                                                take_profit_price = current_price * (1 - take_profit_pct)
                                            
                                            # 最小注文サイズチェック
                                            min_order_size_usd = 10.0  # Bybit minimum
                                            if position_size < min_order_size_usd:
                                                logger.warning(f"Position size ${position_size:.2f} is below minimum ${min_order_size_usd}")
                                                discord_notifier.send_notification(
                                                    title="⚠️ 注文サイズ警告",
                                                    description=f"{symbol}: ポジションサイズが最小注文サイズ未満です",
                                                    color="ff9900",
                                                    fields={
                                                        "Position Size": f"${position_size:.2f}",
                                                        "Minimum Required": f"${min_order_size_usd:.2f}",
                                                        "Action": "注文をスキップしました"
                                                    }
                                                )
                                                continue
                                            
                                            # 注文数量の計算
                                            order_qty = position_size / current_price
                                            
                                            # 通貨ペアごとの精度調整
                                            if "BTC" in symbol:
                                                order_qty = round(order_qty, 3)  # 0.001 BTC
                                            elif "ETH" in symbol:
                                                order_qty = round(order_qty, 2)  # 0.01 ETH
                                            else:
                                                order_qty = round(order_qty, 1)  # 0.1 units
                                            
                                            # 注文実行
                                            logger.info(f"Placing {order_side} order: {symbol} qty={order_qty} @ ${order_price:.2f}")
                                            
                                            order_result = await self.bybit_client.create_order(
                                                symbol=symbol,
                                                side=order_side,
                                                order_type="limit",
                                                qty=order_qty,
                                                price=order_price,
                                                stop_loss=stop_loss_price,
                                                take_profit=take_profit_price
                                            )
                                            
                                            if order_result:
                                                order_id = order_result.get("orderId")
                                                logger.info(f"✅ Order placed successfully: {order_id}")
                                                
                                                # Save to database
                                                position_id = f"pos_{order_id}"
                                                save_position(
                                                    position_id=position_id,
                                                    symbol=symbol,
                                                    side=order_side,
                                                    entry_price=order_price,
                                                    quantity=order_qty,
                                                    stop_loss=stop_loss_price,
                                                    take_profit=take_profit_price,
                                                    metadata={
                                                        "signal_confidence": confidence,
                                                        "expected_pnl": prediction,
                                                        "signal_time": datetime.now().isoformat(),
                                                        "account_balance": account_equity,
                                                        "position_pct": position_pct
                                                    }
                                                )
                                                
                                                save_trade(
                                                    trade_id=order_id,
                                                    position_id=position_id,
                                                    symbol=symbol,
                                                    side=order_side,
                                                    order_type="limit",
                                                    quantity=order_qty,
                                                    price=order_price,
                                                    metadata={
                                                        "signal_confidence": confidence,
                                                        "expected_pnl": prediction,
                                                        "market_price": current_price,
                                                        "bid_price": bid_price,
                                                        "ask_price": ask_price
                                                    }
                                                )
                                                
                                                # Send execution notification
                                                discord_notifier.send_notification(
                                                    title="✅ 注文実行完了",
                                                    description=f"{symbol} {order_side.upper()} 注文が正常に実行されました",
                                                    color="00ff00",
                                                    fields={
                                                        "Order ID": order_id,
                                                        "Symbol": symbol,
                                                        "Side": order_side.upper(),
                                                        "Quantity": f"{order_qty}",
                                                        "Order Price": f"${order_price:.2f}",
                                                        "Stop Loss": f"${stop_loss_price:.2f}",
                                                        "Take Profit": f"${take_profit_price:.2f}"
                                                    }
                                                )
                                            else:
                                                logger.error(f"Failed to place order for {symbol}")
                                                discord_notifier.send_notification(
                                                    title="❌ 注文実行失敗",
                                                    description=f"{symbol}の注文実行に失敗しました",
                                                    color="ff0000"
                                                )
                                        else:
                                            logger.warning(f"Risk check failed for {symbol}")
                                            discord_notifier.send_notification(
                                                title="⚠️ リスクチェック",
                                                description=f"{symbol}の取引はリスク管理により拒否されました",
                                                color="ff9900"
                                            )
                                    
                                    except Exception as e:
                                        logger.error(f"Error executing trade for {symbol}: {e}")
                                        discord_notifier.send_notification(
                                            title="❌ 取引エラー",
                                            description=f"注文実行エラー: {str(e)}",
                                            color="ff0000",
                                            fields={
                                                "Symbol": symbol,
                                                "Error": str(e)
                                            }
                                        )
                        
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}")
                
                # Increment loop count
                loop_count += 1
                
                # Sleep for next iteration
                await asyncio.sleep(3)  # Check every 3 seconds
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def _balance_notification_loop(self):
        """Send periodic balance notifications."""
        await asyncio.sleep(30)  # Initial wait
        
        while self.running:
            try:
                # Wait for the next hour
                current_time = datetime.now()
                minutes_until_hour = 60 - current_time.minute
                await asyncio.sleep(minutes_until_hour * 60)
                
                # Get account info
                balance = self.account_monitor.current_balance
                if balance:
                    stats = self.account_monitor.get_performance_stats()
                    
                    fields = {
                        "Balance": f"${balance.total_equity:,.2f}",
                        "Available": f"${balance.available_balance:,.2f}",
                        "Unrealized PnL": f"${balance.unrealized_pnl:,.2f}",
                        "Total Return": f"{stats.get('total_return_pct', 0):.2f}%",
                        "Daily PnL": f"${stats.get('daily_pnl', 0):,.2f}",
                        "Win Rate": f"{stats.get('win_rate', 0):.1f}%",
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
                    "order_router": self.order_router is not None and hasattr(self.order_router, 'running') and self.order_router.running,
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
                            position_value = size * entry_price
                            pnl_pct = (unrealized_pnl / position_value * 100) if position_value > 0 else 0
                            
                            # Log position status
                            logger.info(
                                f"Position {symbol} {side}: size={size}, entry=${entry_price:.2f}, " +
                                f"mark=${mark_price:.2f}, PnL=${unrealized_pnl:.2f} ({pnl_pct:.2f}%)"
                            )
                            
                            # Check for partial take profit (1.5% and 3%)
                            await self._check_partial_take_profit(position, pnl_pct)
                            
                            # Check for trailing stop (2%以上の利益で発動)
                            await self._check_trailing_stop(position, pnl_pct)
                            
                            # Alert on significant PnL changes (positive or negative)
                            if abs(pnl_pct) > 5:
                                discord_notifier.send_notification(
                                    title="💰 ポジション更新" if pnl_pct > 0 else "⚠️ ポジション警告",
                                    description=f"{symbol} ポジションが大きく変動しています",
                                    color="00ff00" if pnl_pct > 0 else "ff0000",
                                    fields={
                                        "Symbol": symbol,
                                        "Side": side,
                                        "Size": f"{size}",
                                        "Entry Price": f"${entry_price:.2f}",
                                        "Current Price": f"${mark_price:.2f}",
                                        "PnL": f"${unrealized_pnl:.2f} ({pnl_pct:.2f}%)"
                                    }
                                )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(60)
    
    async def _check_partial_take_profit(self, position: Dict[str, Any], pnl_pct: float):
        """部分利確のチェックと実行"""
        symbol = position.get("symbol")
        size = float(position.get("size", 0))
        
        # 1.5%で50%利確
        if pnl_pct >= 1.5 and size > 0:
            partial_size = size * 0.5
            logger.info(f"Executing partial take profit (50%) for {symbol} at {pnl_pct:.2f}% profit")
            
            # 部分クローズ実行
            result = await self.bybit_client.close_position(symbol, side="partial")
            if result:
                discord_notifier.send_notification(
                    title="💰 部分利確実行",
                    description=f"{symbol}: 50%のポジションを利確しました",
                    color="00ff00",
                    fields={
                        "Profit": f"{pnl_pct:.2f}%",
                        "Closed Size": f"{partial_size:.3f}"
                    }
                )
        
        # 3%で追加25%利確（合計75%）
        elif pnl_pct >= 3.0 and size > 0:
            partial_size = size * 0.25
            logger.info(f"Executing additional partial take profit (25%) for {symbol} at {pnl_pct:.2f}% profit")
            
            result = await self.bybit_client.close_position(symbol, side="partial")
            if result:
                discord_notifier.send_notification(
                    title="💰 追加部分利確実行",
                    description=f"{symbol}: 追加25%のポジションを利確しました",
                    color="00ff00",
                    fields={
                        "Profit": f"{pnl_pct:.2f}%",
                        "Closed Size": f"{partial_size:.3f}"
                    }
                )
    
    async def _check_trailing_stop(self, position: Dict[str, Any], pnl_pct: float):
        """トレーリングストップのチェックと更新"""
        if pnl_pct >= 2.0:  # 2%以上の利益でトレーリングストップ発動
            symbol = position.get("symbol")
            entry_price = float(position.get("avgPrice", 0))
            mark_price = float(position.get("markPrice", 0))
            side = position.get("side")
            
            # エントリー価格の0.5%上（ロング）または下（ショート）にストップロスを移動
            if side == "Buy":
                new_stop_loss = entry_price * 1.005
            else:
                new_stop_loss = entry_price * 0.995
            
            logger.info(f"Moving stop loss to breakeven + 0.5% for {symbol}")
            
            result = await self.bybit_client.set_stop_loss(symbol, new_stop_loss)
            if result:
                discord_notifier.send_notification(
                    title="🛡️ トレーリングストップ更新",
                    description=f"{symbol}: ストップロスをブレークイーブン+0.5%に移動しました",
                    color="03b2f8",
                    fields={
                        "Current Profit": f"{pnl_pct:.2f}%",
                        "New Stop Loss": f"${new_stop_loss:.2f}"
                    }
                )
    
    async def _daily_report_loop(self):
        """日次レポートを毎日午前9時（JST）に送信"""
        while self.running:
            try:
                # 次の午前9時（JST）まで待機
                now = datetime.now()
                # UTC+9 for JST
                target_hour = 0  # 9 AM JST = 0 AM UTC
                
                if now.hour >= target_hour:
                    # 今日はもう過ぎているので明日まで待つ
                    hours_until_target = 24 - now.hour + target_hour
                else:
                    hours_until_target = target_hour - now.hour
                
                minutes_until_target = hours_until_target * 60 - now.minute
                await asyncio.sleep(minutes_until_target * 60)
                
                # 日次レポート生成
                from src.common.database import get_duckdb_connection
                
                conn = get_duckdb_connection()
                today = datetime.now().strftime("%Y-%m-%d")
                
                # 今日の取引統計を取得
                daily_stats = conn.execute(f"""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as filled_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl
                    FROM trades
                    WHERE DATE(timestamp) = '{today}'
                """).fetchone()
                
                if daily_stats and daily_stats[0] > 0:  # 取引があった場合
                    total_trades = daily_stats[0]
                    filled_trades = daily_stats[1] or 0
                    winning_trades = daily_stats[2] or 0
                    total_pnl = daily_stats[3] or 0
                    avg_pnl = daily_stats[4] or 0
                    win_rate = (winning_trades / filled_trades * 100) if filled_trades > 0 else 0
                    
                    # アカウント情報
                    balance = self.account_monitor.current_balance
                    
                    fields = {
                        "📅 Date": today,
                        "💰 Current Balance": f"${balance.total_equity:,.2f}" if balance else "N/A",
                        "📊 Total Trades": str(total_trades),
                        "✅ Filled Trades": str(filled_trades),
                        "🎯 Win Rate": f"{win_rate:.1f}%",
                        "💵 Total PnL": f"${total_pnl:,.2f}",
                        "📈 Average PnL": f"${avg_pnl:,.2f}"
                    }
                    
                    discord_notifier.send_notification(
                        title="📊 日次取引レポート",
                        description="本日の取引結果サマリー",
                        color="03b2f8",
                        fields=fields
                    )
                else:
                    discord_notifier.send_notification(
                        title="📊 日次取引レポート",
                        description="本日は取引がありませんでした",
                        color="808080"
                    )
                
                conn.close()
                
            except Exception as e:
                logger.error(f"Error in daily report: {e}")
                await asyncio.sleep(3600)  # エラー時は1時間待機
    
    async def _send_startup_notification(self):
        """Send detailed startup notification."""
        if self.account_monitor.current_balance:
            balance = self.account_monitor.current_balance
            risk_config = self.order_router.risk_manager.config if self.order_router else None
            
            description = "本番環境で実際の資金を使用した自動取引を開始しました。\n" + \
                         "アカウント残高に基づいてリスクパラメータが自動調整されています。"
            
            fields = {
                "💰 Account Balance": f"${balance.total_equity:,.2f}",
                "📊 Max Position Size": f"${risk_config.max_position_size_usd:,.2f}" if risk_config else "N/A",
                "🎯 Max Total Exposure": f"${risk_config.max_total_exposure_usd:,.2f}" if risk_config else "N/A",
                "⚠️ Max Daily Loss": f"${risk_config.max_daily_loss_usd:,.2f}" if risk_config else "N/A",
                "🔧 Leverage": "3x",
                "🤖 Model": settings.model.model_path,
                "⏱️ Signal Cooldown": f"{self.signal_cooldown}s",
                "🔴 Mode": "LIVE TRADING - 実際の資金"
            }
            
            discord_notifier.send_notification(
                title="🚀 Production Trading System Started (Dynamic Parameters)",
                description=description,
                color="ff0000",  # Red for warning
                fields=fields
            )
    
    async def stop(self):
        """Stop all components gracefully."""
        logger.info("Stopping Production Trading System")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop components
        await self.ingestor.stop()
        await self.account_monitor.stop()
        if self.position_manager:
            await self.position_manager.stop()
        
        # Send shutdown notification
        discord_notifier.send_notification(
            title="🛑 Trading System Stopped",
            description="Production trading system has been shut down",
            color="ff0000"
        )
        
        logger.info("All components stopped")


async def main():
    """Main entry point."""
    system = ProductionTradingSystem()
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await system.start()
    except Exception as e:
        logger.error(f"System error: {e}")
        await system.stop()
        raise


if __name__ == "__main__":
    # Send initial notification
    discord_notifier.send_notification(
        title="🔧 Starting Production Trading Bot",
        description="本番環境での自動取引を開始しています...",
        color="ff9900"
    )
    
    logger.info("Starting Production Trading Bot")
    print("⚠️  警告: 本番環境で実際の資金を使用します")
    print("💰 残高: 約$100 (USDT)")
    print("🔴 LIVE TRADING MODE")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        discord_notifier.send_notification(
            title="❌ System Crash",
            description=f"Production system crashed: {str(e)}",
            color="ff0000"
        )
        raise