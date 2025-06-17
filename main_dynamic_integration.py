#!/usr/bin/env python3
"""
Dynamic Parameter Integrated Trading System

Combines the best of both worlds:
- Proven microservice architecture from integration system
- Dynamic parameter functionality from the standalone system
- Real market price fetching
- Account balance-based risk management
"""

import os
import signal
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables first
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Set production environment
os.environ['BYBIT__TESTNET'] = 'false'
os.environ['ENVIRONMENT'] = 'production'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.common.config import settings
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier
from src.integration.dynamic_trading_coordinator import DynamicTradingCoordinator, DynamicSystemConfig
from src.integration.simple_service_manager import SimpleServiceManager

# Force production settings
settings.bybit.testnet = False

logger = get_logger(__name__)


class DynamicIntegratedTradingSystem:
    """Integrated trading system with dynamic parameters."""
    
    def __init__(self):
        self.running = False
        self.service_manager = None
        self.trading_coordinator = None
        
        # Create dynamic configuration
        self.config = DynamicSystemConfig(
            symbols=settings.bybit.symbols,
            min_prediction_confidence=0.6,
            min_expected_pnl=0.001,
            
            # Dynamic parameter settings (% of account balance)
            max_position_pct=0.3,  # 30% per position
            max_total_exposure_pct=0.6,  # 60% total exposure
            max_daily_loss_pct=0.1,  # 10% daily loss limit
            leverage=3.0,  # 3x leverage max
            
            # Update settings
            balance_update_interval=300,  # 5 minutes
            parameter_update_threshold=0.1  # 10% balance change triggers update
        )
        
        logger.info("Dynamic Integrated Trading System initialized")
    
    async def start(self):
        """Start the integrated system."""
        if self.running:
            logger.warning("System already running")
            return
        
        self.running = True
        logger.info("Starting Dynamic Integrated Trading System")
        logger.info(f"Testnet mode: {settings.bybit.testnet} (should be False)")
        
        try:
            # Send startup notification
            discord_notifier.send_notification(
                title="🚀 Starting Dynamic Trading System",
                description=f"Environment: {'TESTNET' if settings.bybit.testnet else 'PRODUCTION'}\n"
                          f"Symbols: {', '.join(self.config.symbols)}\n"
                          f"Dynamic parameters enabled",
                color="0066ff"  # Blue color in hex
            )
            
            # Initialize service manager
            logger.info("Initializing service manager...")
            self.service_manager = SimpleServiceManager()
            
            # Start core services (Redis, etc.)
            logger.info("Starting core services...")
            await self.service_manager.start_core_services()
            
            # Wait for services to be ready
            await asyncio.sleep(5)
            
            # Initialize and start dynamic trading coordinator
            logger.info("Starting dynamic trading coordinator...")
            self.trading_coordinator = DynamicTradingCoordinator(self.config)
            await self.trading_coordinator.start()
            
            # Start additional services if needed
            logger.info("Starting additional services...")
            await self.service_manager.start_additional_services()
            
            logger.info("Dynamic Integrated Trading System started successfully")
            
            # Send success notification with dynamic parameters
            balance = self.trading_coordinator.account_monitor.current_balance.total_equity
            discord_notifier.send_notification(
                title="✅ Dynamic Trading System Online",
                description=f"🏦 Account Balance: ${balance:.2f}\n"
                          f"📊 Max Position: ${balance * self.config.max_position_pct:.2f} ({self.config.max_position_pct:.0%})\n"
                          f"💼 Max Exposure: ${balance * self.config.max_total_exposure_pct:.2f} ({self.config.max_total_exposure_pct:.0%})\n"
                          f"🔴 Daily Loss Limit: ${balance * self.config.max_daily_loss_pct:.2f} ({self.config.max_daily_loss_pct:.0%})\n"
                          f"⚡ Leverage: {self.config.leverage}x\n"
                          f"🎯 Trading: {', '.join(self.config.symbols)}",
                color="00ff00"  # Green color in hex
            )
            
            # Keep running
            while self.running:
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the integrated system."""
        if not self.running:
            return
        
        logger.info("Stopping Dynamic Integrated Trading System")
        self.running = False
        
        try:
            # Stop trading coordinator
            if self.trading_coordinator:
                await self.trading_coordinator.stop()
            
            # Stop services
            if self.service_manager:
                await self.service_manager.stop_all()
            
            # Send shutdown notification
            discord_notifier.send_notification(
                title="🛑 Dynamic Trading System Stopped",
                description="Trading system has been shut down gracefully",
                color="ff0000"  # Red color in hex
            )
            
            logger.info("Dynamic Integrated Trading System stopped")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    async def get_status(self):
        """Get comprehensive system status."""
        try:
            status = {
                "running": self.running,
                "timestamp": datetime.now().isoformat(),
                "services": {},
                "trading": {}
            }
            
            # Get service status
            if self.service_manager:
                service_status = self.service_manager.get_status()
                status["services"] = service_status
            
            # Get trading status
            if self.trading_coordinator:
                trading_status = await self.trading_coordinator.get_system_status()
                status["trading"] = trading_status
            
            return status
            
        except Exception as e:
            logger.error("Error getting system status", exception=e)
            return {"error": str(e), "timestamp": datetime.now().isoformat()}


async def main():
    """Main entry point."""
    print("🚀 Dynamic Parameter Integrated Trading System")
    print("💰 Real account balance monitoring")
    print("📊 Dynamic risk management")
    print("🔄 Microservice architecture")
    print(f"🌍 Environment: {'TESTNET' if settings.bybit.testnet else 'PRODUCTION'}")
    print(f"🎯 Symbols: {', '.join(settings.bybit.symbols)}")
    print("=" * 60)
    
    system = DynamicIntegratedTradingSystem()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await system.stop()
    except Exception as e:
        logger.error(f"System error: {e}")
        await system.stop()
        raise


if __name__ == "__main__":
    print("Starting Dynamic Parameter Integrated Trading System...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)