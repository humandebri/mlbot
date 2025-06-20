#!/bin/bash

echo "🚀 Deploying real-time data fix to EC2..."

# EC2 details
EC2_HOST="ubuntu@13.212.91.54"
EC2_KEY="~/.ssh/mlbot-key-1749802416.pem"

# Copy all necessary files
echo "📤 Copying files to EC2..."
scp -i $EC2_KEY update_duckdb_complete.py $EC2_HOST:~/mlbot/
scp -i $EC2_KEY test_redis_ec2.py $EC2_HOST:~/mlbot/
scp -i $EC2_KEY improved_feature_generator_redis.py $EC2_HOST:~/mlbot/

# Create updated bot with Redis integration
cat > simple_improved_bot_redis.py << 'BOTEOF'
#!/usr/bin/env python3
"""
Simple Improved ML Bot with Real-time Redis Integration
Fixes static data issue by continuously updating from Redis
"""

import asyncio
import os
import sys
import signal
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import numpy as np
import logging

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the new Redis-enabled feature generator
from improved_feature_generator_redis import ImprovedFeatureGeneratorRedis

# Import other components
from src.common.discord_notifier import discord_notifier
from src.common.database import create_trading_tables, save_trade, save_position, get_duckdb_connection, update_position_close
from src.common.bybit_client import BybitRESTClient
from src.common.account_monitor import AccountMonitor
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.order_router.risk_manager import RiskManager, RiskConfig

class SimpleImprovedTradingBotRedis:
    """ML Bot with real-time data updates from Redis."""
    
    def __init__(self):
        self.running = False
        self.tasks = []
        
        # Initialize components with Redis-enabled feature generator
        self.feature_generator = ImprovedFeatureGeneratorRedis()
        self.bybit_client = BybitRESTClient(testnet=False)
        self.account_monitor = AccountMonitor()
        self.risk_manager = RiskManager(RiskConfig())
        
        # Trading parameters
        self.symbols = ["BTCUSDT", "ETHUSDT", "ICPUSDT"]
        self.min_confidence = 0.45  # 45% threshold
        self.signal_cooldown = 300  # 5 minutes
        self.base_position_pct = 0.2  # 20% of equity per trade
        self.min_order_size_usd = 10.0
        
        # Tracking
        self.last_signal_time = {}
        self.open_positions = {}
        self.signal_count = 0
        self.prediction_count = 0
        self.last_hourly_report = datetime.now()
        self.prediction_history = []
        
        # Current account balance
        self.current_balance = None
        
        # Initialize inference engine
        model_path = os.getenv("MODEL__MODEL_PATH", "models/v3.1_improved/model.onnx")
        self.inference_config = InferenceConfig(
            model_path=model_path,
            enable_batching=False,
            enable_thompson_sampling=False,
            confidence_threshold=0.45,
            risk_adjustment=False
        )
        self.inference_engine = InferenceEngine(self.inference_config)
        
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Simple Improved Trading Bot with Redis...")
        
        try:
            # Create trading tables
            create_trading_tables()
            
            # Initialize Bybit client
            await self.bybit_client.__aenter__()
            
            # Initialize Account Monitor
            await self.account_monitor.start()
            
            # Load ML model
            self.inference_engine.load_model()
            
            # Load initial historical data
            logger.info("Loading initial data...")
            for symbol in self.symbols:
                self.feature_generator.update_historical_cache(symbol)
                logger.info(f"Loaded data for {symbol}")
            
            # Get initial balance
            await self.update_balance()
            
            # Send startup notification
            balance_text = f"${self.current_balance:.2f}" if self.current_balance else "取得中..."
            
            discord_notifier.send_notification(
                title="🚀 リアルタイム取引ボット起動",
                description="Redis統合版MLボットが起動しました",
                color="00ff00",
                fields={
                    "💰 残高": balance_text,
                    "🔴 モード": "実際の資金で取引",
                    "📊 機能": "リアルタイムデータ更新・Redis統合",
                    "🧠 特徴量": "履歴データ + Redis リアルタイムデータ",
                    "⚙️ 信頼度閾値": f"{self.min_confidence * 100:.0f}%"
                }
            )
            
            logger.info("Bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            return False
    
    async def update_balance(self):
        """Update current account balance."""
        try:
            balance = await self.account_monitor.update_balance()
            if balance:
                self.current_balance = balance.total_equity
                logger.info(f"Account balance updated: ${self.current_balance:.2f}")
        except Exception as e:
            logger.error(f"Failed to update balance: {e}")
    
    def predict(self, features):
        """Make prediction with ML model."""
        result = self.inference_engine.predict(features.reshape(1, -1))
        prediction = float(result["predictions"][0])
        return prediction
    
    async def execute_trade(self, symbol: str, prediction: float, confidence: float, features: Dict[str, float]):
        """Execute actual trade with risk management."""
        try:
            # Update balance before trading
            await self.update_balance()
            if not self.current_balance:
                logger.error("Cannot trade without balance information")
                return
            
            # Calculate position size
            position_size = self.current_balance * self.base_position_pct
            position_size = min(position_size, self.current_balance * 0.3)
            
            if position_size < self.min_order_size_usd:
                logger.warning(f"Position size ${position_size:.2f} below minimum")
                return
            
            # Get current price
            ticker = await self.bybit_client.get_ticker(symbol)
            if not ticker or "lastPrice" not in ticker:
                logger.error(f"Failed to get ticker for {symbol}")
                return
            
            current_price = float(ticker["lastPrice"])
            
            # Determine order side
            order_side = "buy" if prediction > 0.5 else "sell"
            
            # Risk management check
            if not self.risk_manager.can_trade(symbol=symbol, side=order_side, size=position_size):
                logger.warning(f"Risk manager blocked trade for {symbol}")
                discord_notifier.send_notification(
                    title="🛑 リスク管理ブロック",
                    description=f"{symbol} の取引がリスク管理によりブロックされました",
                    color="ff0000"
                )
                return
            
            # Calculate order parameters
            slippage = 0.001
            if order_side == "buy":
                order_price = current_price * (1 + slippage)
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.03
            else:
                order_price = current_price * (1 - slippage)
                stop_loss = current_price * 1.02
                take_profit = current_price * 0.97
            
            # Calculate order quantity
            order_qty = position_size / current_price
            if "BTC" in symbol:
                order_qty = round(order_qty, 3)
            else:
                order_qty = round(order_qty, 2)
            
            # Execute order
            logger.info(f"Executing {order_side} order for {symbol}: {order_qty} @ ${order_price:.2f}")
            
            order_result = await self.bybit_client.create_order(
                symbol=symbol,
                side=order_side,
                order_type="limit",
                qty=order_qty,
                price=order_price,
                stop_loss=stop_loss,
                take_profit=take_profit
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
                    quantity=order_qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        "signal_confidence": confidence,
                        "ml_prediction": prediction,
                        "signal_time": datetime.now().isoformat()
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
                        "ml_prediction": prediction
                    }
                )
                
                # Send success notification
                discord_notifier.send_notification(
                    title="✅ 注文実行成功",
                    description=f"{symbol} の注文が正常に実行されました",
                    color="00ff00",
                    fields={
                        "Side": order_side.upper(),
                        "Quantity": f"{order_qty}",
                        "Price": f"${order_price:.2f}",
                        "Stop Loss": f"${stop_loss:.2f}",
                        "Take Profit": f"${take_profit:.2f}",
                        "Position Size": f"${position_size:.2f}",
                        "Confidence": f"{confidence*100:.1f}%",
                        "Order ID": order_id
                    }
                )
                
                logger.info(f"Order executed successfully: {order_id}")
                
            else:
                logger.error(f"Failed to execute order for {symbol}")
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            discord_notifier.send_notification(
                title="❌ 取引エラー",
                description=f"{symbol} の取引中にエラーが発生しました: {str(e)}",
                color="ff0000"
            )
    
    async def trading_loop(self):
        """Main trading loop."""
        self.running = True
        
        while self.running:
            try:
                for symbol in self.symbols:
                    # Generate features with real-time data
                    ticker = await self.bybit_client.get_ticker(symbol)
                    if not ticker:
                        continue
                    
                    features = self.feature_generator.generate_features(ticker, symbol)
                    normalized = self.feature_generator.normalize_features(features)
                    
                    # Get prediction
                    prediction = self.predict(normalized)
                    self.prediction_count += 1
                    
                    # Calculate confidence
                    confidence = abs(prediction - 0.5) * 2
                    direction = "BUY" if prediction > 0.5 else "SELL"
                    price = float(ticker.get("lastPrice", 0))
                    
                    # Store prediction
                    self.prediction_history.append({
                        "symbol": symbol,
                        "prediction": prediction,
                        "confidence": confidence,
                        "direction": direction,
                        "price": price,
                        "timestamp": datetime.now()
                    })
                    
                    # Keep only last 2000 predictions
                    if len(self.prediction_history) > 2000:
                        self.prediction_history = self.prediction_history[-2000:]
                    
                    logger.info(
                        f"📊 {symbol}: pred={prediction:.4f}, "
                        f"conf={confidence*100:.2f}%, dir={direction}, "
                        f"price=${price:,.2f}"
                    )
                    
                    # Check if we should send signal and trade
                    now = datetime.now()
                    last_signal = self.last_signal_time.get(symbol)
                    
                    if confidence >= self.min_confidence:
                        if not last_signal or (now - last_signal).total_seconds() >= self.signal_cooldown:
                            self.signal_count += 1
                            self.last_signal_time[symbol] = now
                            
                            # Get data status
                            combined_data = self.feature_generator.get_combined_data(symbol)
                            data_status = "リアルタイム"
                            if not combined_data.empty:
                                latest_time = combined_data.index[-1]
                                age_minutes = (datetime.now() - latest_time).total_seconds() / 60
                                data_status = f"データ鮮度: {age_minutes:.1f}分前"
                            
                            # Send signal notification
                            discord_notifier.send_notification(
                                title=f"🎯 ML Signal #{self.signal_count} - {symbol}",
                                description=(
                                    f"**Direction:** {direction}\n"
                                    f"**Confidence:** {confidence*100:.1f}%\n"
                                    f"**ML Score:** {prediction:.4f}\n"
                                    f"**Price:** ${price:,.2f}\n"
                                    f"**{data_status}**\n"
                                    f"**Status:** 取引実行中...\n"
                                ),
                                color="00ff00" if direction == "BUY" else "ff0000"
                            )
                            
                            # Execute actual trade
                            await self.execute_trade(symbol, prediction, confidence, features)
                
                # Check if hourly report is due
                if (datetime.now() - self.last_hourly_report).total_seconds() >= 3600:
                    await self.send_hourly_report()
                    self.last_hourly_report = datetime.now()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)
    
    async def position_monitor_loop(self):
        """Monitor open positions."""
        await asyncio.sleep(30)  # Initial wait
        
        while self.running:
            try:
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
                            pnl_pct = (unrealized_pnl / (size * entry_price)) * 100 if entry_price > 0 else 0
                            
                            # Check for trailing stop and partial take profit
                            await self.check_trailing_stop(position)
                            await self.check_partial_take_profit(position)
                            
                            # Alert if large move
                            if abs(pnl_pct) > 5:
                                discord_notifier.send_notification(
                                    title="⚠️ ポジション監視アラート",
                                    description=f"{symbol} ポジションが大きく動いています",
                                    color="ff9900",
                                    fields={
                                        "Symbol": symbol,
                                        "PnL": f"${unrealized_pnl:.2f} ({pnl_pct:.2f}%)"
                                    }
                                )
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(60)
    
    async def check_trailing_stop(self, position: Dict[str, Any]):
        """Check and update trailing stop."""
        symbol = position.get("symbol")
        side = position.get("side")
        entry_price = float(position.get("avgPrice", 0))
        mark_price = float(position.get("markPrice", 0))
        
        if entry_price == 0:
            return
        
        # Calculate profit percentage
        if side == "Buy":
            profit_pct = ((mark_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - mark_price) / entry_price) * 100
        
        # If profit > 2%, move stop loss to breakeven + 0.5%
        if profit_pct > 2.0:
            if side == "Buy":
                new_stop_loss = entry_price * 1.005
            else:
                new_stop_loss = entry_price * 0.995
            
            success = await self.bybit_client.set_stop_loss(symbol, new_stop_loss)
            if success:
                logger.info(f"Trailing stop updated for {symbol}: ${new_stop_loss:.2f}")
    
    async def check_partial_take_profit(self, position: Dict[str, Any]):
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
        
        position_id = f"pos_{symbol}_{side}"
        closed_pct = self.open_positions.get(position_id, {}).get("closed_pct", 0)
        
        # At 3% profit, close 25% more (total 75%)
        if profit_pct >= 3.0 and closed_pct < 75:
            close_size = size * 0.25
            close_side = "sell" if side == "Buy" else "buy"
            
            result = await self.bybit_client.create_order(
                symbol=symbol,
                side=close_side,
                order_type="market",
                qty=round(close_size, 3 if "BTC" in symbol else 2),
                reduce_only=True
            )
            
            if result:
                self.open_positions[position_id] = {"closed_pct": 75}
                discord_notifier.send_notification(
                    title="💰 部分利確実行 (75%)",
                    description=f"{symbol} ポジションの25%を利確",
                    color="00ff00"
                )
        
        # At 1.5% profit, close 50%
        elif profit_pct >= 1.5 and closed_pct < 50:
            close_size = size * 0.5
            close_side = "sell" if side == "Buy" else "buy"
            
            result = await self.bybit_client.create_order(
                symbol=symbol,
                side=close_side,
                order_type="market",
                qty=round(close_size, 3 if "BTC" in symbol else 2),
                reduce_only=True
            )
            
            if result:
                self.open_positions[position_id] = {"closed_pct": 50}
                discord_notifier.send_notification(
                    title="💰 部分利確実行 (50%)",
                    description=f"{symbol} ポジションの50%を利確",
                    color="00ff00"
                )
    
    async def send_hourly_report(self):
        """Send hourly report."""
        try:
            # Update balance
            await self.update_balance()
            
            # Get recent predictions
            recent_preds = [p for p in self.prediction_history 
                          if (datetime.now() - p["timestamp"]).total_seconds() < 3600]
            
            # Get open positions
            positions = await self.bybit_client.get_open_positions()
            total_unrealized_pnl = sum(float(p.get("unrealizedPnl", 0)) for p in positions) if positions else 0
            
            # Get data freshness for each symbol
            data_status = []
            for symbol in self.symbols:
                combined_data = self.feature_generator.get_combined_data(symbol)
                if not combined_data.empty:
                    latest_time = combined_data.index[-1]
                    age_minutes = (datetime.now() - latest_time).total_seconds() / 60
                    records = len(combined_data)
                    data_status.append(f"{symbol}: {records}件 ({age_minutes:.1f}分前)")
            
            # Build report
            report_lines = [
                "**📊 時間レポート（Redis統合版）**",
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
                "**💰 アカウント情報:**",
                f"• 残高: ${self.current_balance:.2f}" if self.current_balance else "• 残高: 取得失敗",
                f"• オープンポジション: {len(positions) if positions else 0}",
                f"• 未実現損益: ${total_unrealized_pnl:.2f}",
                "",
                "**📡 データステータス:**"
            ]
            
            report_lines.extend([f"• {status}" for status in data_status])
            
            report_lines.extend([
                "",
                "**📈 取引統計:**",
                f"• 予測回数: {len(recent_preds)}",
                f"• シグナル数: {sum(1 for p in recent_preds if p['confidence'] >= self.min_confidence)}"
            ])
            
            discord_notifier.send_notification(
                title="📈 時間レポート",
                description="\n".join(report_lines),
                color="3498db"
            )
            
        except Exception as e:
            logger.error(f"Error sending hourly report: {e}")
    
    async def periodic_data_update(self):
        """Periodically update data from Redis."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                for symbol in self.symbols:
                    records = self.feature_generator.update_from_redis(symbol)
                    if records > 0:
                        logger.info(f"Updated {symbol} with {records} new records from Redis")
                
            except Exception as e:
                logger.error(f"Error in periodic data update: {e}")
                await asyncio.sleep(60)
    
    async def daily_report_loop(self):
        """Send daily report at 9:00 AM JST."""
        while self.running:
            try:
                # Calculate next 9:00 AM JST
                jst = timezone(timedelta(hours=9))
                now = datetime.now(jst)
                next_report = now.replace(hour=9, minute=0, second=0, microsecond=0)
                if now.hour >= 9:
                    next_report += timedelta(days=1)
                
                wait_seconds = (next_report - now).total_seconds()
                logger.info(f"Next daily report in {wait_seconds/3600:.1f} hours")
                await asyncio.sleep(wait_seconds)
                
                await self.send_daily_report()
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in daily report loop: {e}")
                await asyncio.sleep(3600)
    
    async def send_daily_report(self):
        """Send comprehensive daily report."""
        try:
            await self.update_balance()
            
            # Get today's trades from database
            conn = get_duckdb_connection()
            
            today_trades = conn.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl
                FROM trades 
                WHERE DATE(created_at) = CURRENT_DATE
            """).fetchone()
            
            discord_notifier.send_notification(
                title="📊 日次レポート (9:00 AM JST)",
                description=datetime.now().strftime('%Y-%m-%d'),
                color="00ff00",
                fields={
                    "💰 残高": f"${self.current_balance:.2f}" if self.current_balance else "N/A",
                    "📈 本日の取引": f"{today_trades[0] if today_trades else 0}回",
                    "✅ 勝率": f"{(today_trades[1]/today_trades[0]*100) if today_trades and today_trades[0] > 0 else 0:.1f}%",
                    "💵 本日の損益": f"${today_trades[2] if today_trades else 0:.2f}"
                }
            )
            
        except Exception as e:
            logger.error(f"Error sending daily report: {e}")
    
    async def balance_notification_loop(self):
        """Send balance updates every hour."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour
                await self.update_balance()
                
                if self.current_balance:
                    positions = await self.bybit_client.get_open_positions()
                    open_count = len(positions) if positions else 0
                    
                    discord_notifier.send_notification(
                        title="💰 残高更新",
                        description="1時間ごとの残高通知",
                        color="3498db",
                        fields={
                            "残高": f"${self.current_balance:.2f}",
                            "オープンポジション": str(open_count),
                            "時刻": datetime.now().strftime('%H:%M:%S')
                        }
                    )
                
            except Exception as e:
                logger.error(f"Error in balance notification: {e}")
    
    async def start(self):
        """Start all bot components."""
        if not await self.initialize():
            return False
        
        self.running = True
        
        # Create background tasks
        self.tasks = [
            asyncio.create_task(self.trading_loop()),
            asyncio.create_task(self.position_monitor_loop()),
            asyncio.create_task(self.balance_notification_loop()),
            asyncio.create_task(self.daily_report_loop()),
            asyncio.create_task(self.periodic_data_update())  # New Redis update task
        ]
        
        logger.info("All tasks started")
        return True
    
    async def stop(self):
        """Stop all bot components."""
        logger.info("Stopping bot...")
        self.running = False
        
        # Cancel tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Cleanup
        await self.bybit_client.__aexit__(None, None, None)
        await self.account_monitor.stop()
        self.feature_generator.close()
        
        discord_notifier.send_notification(
            title="🛑 取引ボット停止",
            description="Redis統合版MLボットが停止しました",
            color="ff0000"
        )


async def main():
    """Main entry point."""
    bot = SimpleImprovedTradingBotRedis()
    
    # Setup signal handlers
    def signal_handler(sig):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(bot.stop())
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler, sig)
    
    try:
        if await bot.start():
            # Keep running
            while bot.running:
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    finally:
        await bot.stop()


if __name__ == "__main__":
    print("⚠️  警告: 実際の資金で取引を行います")
    print("🔴 LIVE TRADING MODE - Redis統合版")
    asyncio.run(main())
BOTEOF

# Copy the new bot
scp -i $EC2_KEY simple_improved_bot_redis.py $EC2_HOST:~/mlbot/

# SSH and execute everything
ssh -i $EC2_KEY $EC2_HOST << 'EOF'
cd ~/mlbot

echo "🧪 Testing Redis connectivity..."
python3 test_redis_ec2.py

echo ""
echo "⏸️  Stopping current bot..."
pkill -f "simple_improved_bot_with_trading_fixed.py" || true
sleep 2

echo ""
echo "💾 Backing up database..."
cp data/historical_data.duckdb data/historical_data.duckdb.backup.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

echo ""
echo "🔄 Updating DuckDB with latest data..."
python3 update_duckdb_complete.py

echo ""
echo "🚀 Starting new Redis-integrated bot..."
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq"
nohup python3 simple_improved_bot_redis.py > logs/bot_redis.log 2>&1 &

sleep 5

# Check if running
if pgrep -f "simple_improved_bot_redis.py" > /dev/null; then
    echo "✅ Bot is running with PID: $(pgrep -f simple_improved_bot_redis.py)"
    echo ""
    echo "📋 Recent log entries:"
    tail -20 logs/bot_redis.log
else
    echo "❌ Bot failed to start! Check logs:"
    tail -50 logs/bot_redis.log
fi

echo ""
echo "✅ Deployment complete!"
EOF