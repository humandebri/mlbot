#!/usr/bin/env python3
"""
Trading system with real API integration and daily reporting.
"""
import sys
sys.path.insert(0, '/Users/0xhude/Desktop/mlbot')

import asyncio
import signal
from typing import Optional, Dict, List, Any
from datetime import datetime
import traceback

# Core components
from src.ingestor.main import BybitIngestor
from src.feature_hub.main import FeatureHub
from src.ml_pipeline.inference_engine import InferenceEngine
from src.common.config import settings
from src.common.logging import get_logger
from src.common.database import RedisManager
from src.common.discord_notifier import discord_notifier
from src.common.account_monitor import AccountMonitor
from src.common.daily_report import DailyReportManager

logger = get_logger(__name__)


class TradingSystemWithDailyReport:
    """
    Complete trading system with:
    - Real Bybit API integration
    - Daily performance reports
    - Hourly balance updates
    - Kelly criterion position sizing
    """
    
    def __init__(self):
        # Components
        self.ingestor = BybitIngestor()
        self.feature_hub = FeatureHub()
        self.redis_manager = RedisManager()
        self.account_monitor = AccountMonitor(check_interval=60)
        
        # Daily report manager - Report at 9:00 AM JST (00:00 UTC)
        self.daily_report = DailyReportManager(
            account_monitor=self.account_monitor,
            report_time="00:00",  # UTC
            timezone="Asia/Tokyo"  # JST timezone
        )
        
        # ML inference
        self.inference_engine = InferenceEngine()
        
        # Control flags
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Trading symbols
        self.symbols = settings.ingestor.symbols
        
        logger.info("Trading system with daily report initialized")
    
    async def start(self):
        """Start all components."""
        logger.info("Starting Trading System with Daily Reports")
        
        # Load ML model
        model_path = settings.model.model_path
        self.inference_engine.load_model(model_path)
        logger.info(f"Model loaded: {model_path}")
        
        # Start account monitor
        await self.account_monitor.start()
        logger.info("Account monitor started - retrieving real balance")
        
        # Start daily report manager
        await self.daily_report.start()
        logger.info("Daily report manager started")
        
        # Get initial balance
        await asyncio.sleep(2)  # Wait for initial balance update
        
        if self.account_monitor.current_balance:
            balance = self.account_monitor.current_balance
            initial_stats = self.account_monitor.get_performance_stats()
            
            # Send startup notification
            fields = {
                "💰 総資産": f"${balance.total_equity:,.2f}",
                "💵 利用可能残高": f"${balance.available_balance:,.2f}",
                "📊 未実現損益": f"${balance.unrealized_pnl:,.2f}",
                "⚙️ レバレッジ": "3倍",
                "📈 ケリー基準": "25%",
                "⏰ 日次レポート": "毎日 09:00 JST"
            }
            
            discord_notifier.send_notification(
                title="🚀 取引システム起動（日次レポート機能付き）",
                description="実際のBybit APIと連携し、毎日の残高推移をレポートします",
                color="00ff00",
                fields=fields
            )
        else:
            logger.warning("No initial balance retrieved")
        
        # Start components
        await self.ingestor.start()
        await self.feature_hub.start()
        
        # Start background tasks
        self.running = True
        self.tasks = [
            asyncio.create_task(self._trading_loop()),
            asyncio.create_task(self._hourly_balance_notification()),
            asyncio.create_task(self._health_check_loop()),
        ]
        
        logger.info("All components started successfully")
    
    async def stop(self):
        """Stop all components."""
        logger.info("Stopping trading system...")
        self.running = False
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop components
        await self.daily_report.stop()
        await self.account_monitor.stop()
        await self.feature_hub.stop()
        await self.ingestor.stop()
        
        # Send shutdown notification with final balance
        if self.account_monitor.current_balance:
            balance = self.account_monitor.current_balance
            stats = self.account_monitor.get_performance_stats()
            
            fields = {
                "最終残高": f"${balance.total_equity:,.2f}",
                "総リターン": f"{stats.get('total_return_pct', 0):+.2f}%",
                "最大ドローダウン": f"{stats.get('max_drawdown_pct', 0):.2f}%"
            }
            
            discord_notifier.send_notification(
                title="🛑 取引システム停止",
                description="システムを安全に停止しました",
                color="ff0000",
                fields=fields
            )
        
        logger.info("Trading system stopped")
    
    async def _trading_loop(self):
        """Main trading loop with signal recording for daily stats."""
        logger.info("Starting trading loop")
        loop_count = 0
        
        while self.running:
            try:
                loop_count += 1
                
                for symbol in self.symbols:
                    # Get latest features
                    features = await self._get_latest_features(symbol)
                    
                    if features and len(features) > 10:
                        try:
                            # Make prediction
                            result = self.inference_engine.predict(features)
                            
                            prediction = result["predictions"][0] if result["predictions"] else 0
                            confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                            
                            # Record signal for daily stats
                            self.daily_report.record_signal({
                                'symbol': symbol,
                                'prediction': prediction,
                                'confidence': confidence,
                                'price': features.get('close', 0)
                            })
                            
                            if confidence > 0.6:
                                # High confidence signal
                                position_size = 0
                                if self.account_monitor.current_balance:
                                    base_position_pct = float(settings.trading.position_size_pct)
                                    position_size = self.account_monitor.get_current_position_size(base_position_pct)
                                    
                                    logger.info(
                                        f"🚨 HIGH CONFIDENCE Signal for {symbol}: "
                                        f"pred={prediction:.4f}, conf={confidence:.2%}, "
                                        f"position_size=${position_size:.2f}"
                                    )
                                    
                                    # Send Discord notification
                                    fields = {
                                        "Symbol": symbol,
                                        "Side": "BUY" if prediction > 0 else "SELL",
                                        "Price": f"${features.get('close', 0):,.2f}",
                                        "Confidence": f"{confidence:.2%}",
                                        "Expected PnL": f"{prediction:.2%}",
                                        "Account Balance": f"${self.account_monitor.current_balance.total_equity:,.2f}",
                                        "Position Size": f"${position_size:.2f}"
                                    }
                                    
                                    discord_notifier.send_notification(
                                        title="🚨 Trade Signal",
                                        description=f"High confidence signal for {symbol}",
                                        color="00ff00" if prediction > 0 else "ff0000",
                                        fields=fields
                                    )
                                
                        except Exception as e:
                            logger.error(f"Prediction error for {symbol}: {e}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def _hourly_balance_notification(self):
        """Send hourly balance updates."""
        interval = 3600  # 1 hour
        
        while self.running:
            try:
                await asyncio.sleep(interval)
                
                if self.account_monitor.current_balance:
                    balance = self.account_monitor.current_balance
                    stats = self.account_monitor.get_performance_stats()
                    
                    fields = {
                        "残高": f"${balance.total_equity:,.2f}",
                        "利用可能": f"${balance.available_balance:,.2f}",
                        "未実現損益": f"${balance.unrealized_pnl:,.2f}",
                        "総リターン": f"{stats.get('total_return_pct', 0):+.2f}%",
                        "最大DD": f"{stats.get('max_drawdown_pct', 0):.2f}%"
                    }
                    
                    discord_notifier.send_notification(
                        title="📊 1時間ごとの残高更新",
                        description=datetime.now().strftime("%Y-%m-%d %H:%M JST"),
                        color="03b2f8",
                        fields=fields
                    )
                
            except Exception as e:
                logger.error(f"Error in hourly notification: {e}")
    
    async def _health_check_loop(self):
        """Health check loop."""
        while self.running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Check component health
                redis_ok = await self.redis_manager.ping()
                
                if not redis_ok:
                    logger.error("Redis connection lost")
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _get_latest_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest features for a symbol."""
        try:
            messages = await self.redis_manager.xread(
                {f"features:{symbol}": "$"},
                count=1,
                block=100
            )
            
            if messages and f"features:{symbol}" in messages:
                entries = messages[f"features:{symbol}"]
                if entries:
                    features = entries[0][1]
                    # Convert bytes to proper types
                    return {k.decode(): float(v) for k, v in features.items()}
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {e}")
            return None


# Signal handlers
def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}")
    asyncio.create_task(shutdown())


async def shutdown():
    """Graceful shutdown."""
    if trading_system:
        await trading_system.stop()


# Global instance
trading_system: Optional[TradingSystemWithDailyReport] = None


async def main():
    """Main entry point."""
    global trading_system
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create and start system
        trading_system = TradingSystemWithDailyReport()
        await trading_system.start()
        
        # Keep running
        while trading_system.running:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        if trading_system:
            await trading_system.stop()


if __name__ == "__main__":
    asyncio.run(main())