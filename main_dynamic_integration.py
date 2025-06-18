#!/usr/bin/env python3
"""
Dynamic Parameter Integrated Trading System

Combines the best of both worlds:
- Proven microservice architecture from integration system
- Dynamic parameter functionality from the standalone system
- Real market price fetching
- Account balance-based risk management
"""

import asyncio
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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
# from src.common.config_manager import ConfigManager
from src.common.decorators import with_error_handling
from src.common.performance import profile_performance
from src.common.discord_notifier import discord_notifier
from src.common.error_handler import error_context, error_handler, setup_exception_hooks
from src.common.exceptions import TradingBotError
from src.common.logging import get_logger
from src.common.performance import performance_context, optimize_performance
from src.common.types import SystemStatus
from src.integration.dynamic_trading_coordinator import DynamicSystemConfig, DynamicTradingCoordinator
from src.integration.simple_service_manager import SimpleServiceManager

# Setup global exception handling
setup_exception_hooks()

# Force production settings
settings.bybit.testnet = False

logger = get_logger(__name__)


class DynamicIntegratedTradingSystem:
    """Integrated trading system with dynamic parameters and enhanced error handling."""
    
    def __init__(self):
        # self.config_manager = ConfigManager()
        self.running = False
        self.service_manager: Optional[SimpleServiceManager] = None
        self.trading_coordinator: Optional[DynamicTradingCoordinator] = None
        self.status = SystemStatus.STOPPED
        
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
    
    @profile_performance(include_memory=True)
    @with_error_handling(TradingBotError)
    async def start(self) -> None:
        """Start the integrated system with enhanced error handling."""
        if self.running:
            logger.warning("System already running")
            return
        
        with performance_context("system_startup"):
            self.running = True
            self.status = SystemStatus.STARTING
            
            logger.info("Starting Dynamic Integrated Trading System")
            logger.info(f"Testnet mode: {settings.bybit.testnet} (should be False)")
            
            with error_context({"operation": "system_startup", "config": str(self.config)}):
                try:
                    # Send startup notification
                    await self._send_startup_notification()
                    
                    # Initialize service manager
                    await self._initialize_services()
                    
                    # Initialize and start dynamic trading coordinator
                    await self._start_trading_coordinator()
                    
                    # Send success notification
                    await self._send_success_notification()
                    
                    self.status = SystemStatus.RUNNING
                    logger.info("Dynamic Integrated Trading System started successfully")
                    
                    # Keep running with periodic optimization
                    await self._main_loop()
                    
                except Exception as e:
                    self.status = SystemStatus.ERROR
                    error_handler.handle_error(e, {
                        "operation": "system_startup",
                        "status": self.status.value
                    })
                    await self.stop()
                    raise
    
    @with_error_handling(TradingBotError)
    async def stop(self) -> None:
        """Stop the integrated system gracefully."""
        if not self.running:
            return
        
        with performance_context("system_shutdown"):
            logger.info("Stopping Dynamic Integrated Trading System")
            self.running = False
            self.status = SystemStatus.STOPPING
            
            with error_context({"operation": "system_shutdown"}):
                try:
                    # Stop trading coordinator
                    if self.trading_coordinator:
                        logger.info("Stopping trading coordinator...")
                        await self.trading_coordinator.stop()
                    
                    # Stop services
                    if self.service_manager:
                        logger.info("Stopping service manager...")
                        await self.service_manager.stop_all()
                    
                    # Send shutdown notification
                    await self._send_shutdown_notification()
                    
                    self.status = SystemStatus.STOPPED
                    logger.info("Dynamic Integrated Trading System stopped successfully")
                
                except Exception as e:
                    self.status = SystemStatus.ERROR
                    logger.error(f"Error during shutdown: {e}")
                    raise
    
    async def _send_startup_notification(self) -> None:
        """Send system startup notification."""
        discord_notifier.send_notification(
            title="🚀 Starting Dynamic Trading System",
            description=f"Environment: {'TESTNET' if settings.bybit.testnet else 'PRODUCTION'}\n"
                      f"Symbols: {', '.join(self.config.symbols)}\n"
                      f"Dynamic parameters enabled",
            color="0066ff"  # Blue color
        )
    
    async def _initialize_services(self) -> None:
        """Initialize and start core services."""
        logger.info("Initializing service manager...")
        self.service_manager = SimpleServiceManager()
        
        logger.info("Starting core services...")
        await self.service_manager.start_all()
        
        # Wait for services to be ready
        await asyncio.sleep(5)
        
        logger.info("All core services started successfully")
        # All necessary services are already started by start_all()
    
    async def _start_trading_coordinator(self) -> None:
        """Initialize and start the trading coordinator."""
        logger.info("Starting dynamic trading coordinator...")
        self.trading_coordinator = DynamicTradingCoordinator(self.config)
        await self.trading_coordinator.start()
    
    async def _send_success_notification(self) -> None:
        """Send system startup success notification with dynamic parameters."""
        if not self.trading_coordinator:
            return
        
        balance = self.trading_coordinator.account_monitor.current_balance.total_equity
        discord_notifier.send_notification(
            title="✅ Dynamic Trading System Online",
            description=f"🏦 Account Balance: ${balance:.2f}\n"
                      f"📊 Max Position: ${balance * self.config.max_position_pct:.2f} ({self.config.max_position_pct:.0%})\n"
                      f"💼 Max Exposure: ${balance * self.config.max_total_exposure_pct:.2f} ({self.config.max_total_exposure_pct:.0%})\n"
                      f"🔴 Daily Loss Limit: ${balance * self.config.max_daily_loss_pct:.2f} ({self.config.max_daily_loss_pct:.0%})\n"
                      f"⚡ Leverage: {self.config.leverage}x\n"
                      f"🎯 Trading: {', '.join(self.config.symbols)}",
            color="00ff00"  # Green color
        )
    
    async def _send_shutdown_notification(self) -> None:
        """Send system shutdown notification."""
        discord_notifier.send_notification(
            title="🛑 Dynamic Trading System Stopped",
            description="Trading system has been shut down gracefully",
            color="ff0000"  # Red color
        )
    
    async def _main_loop(self) -> None:
        """Main system loop with periodic optimization."""
        loop_count = 0
        
        while self.running:
            await asyncio.sleep(1)
            loop_count += 1
            
            # Periodic performance optimization (every 10 minutes)
            if loop_count % 600 == 0:
                with error_context({"operation": "periodic_optimization"}):
                    try:
                        optimize_performance()
                        logger.debug("Periodic performance optimization completed")
                    except Exception as e:
                        logger.warning(f"Performance optimization failed: {e}")
    
    def get_system_status(self) -> dict:
        """Get current system status."""
        return {
            "status": self.status.value,
            "running": self.running,
            "service_manager_active": self.service_manager is not None,
            "trading_coordinator_active": self.trading_coordinator is not None,
            "config": {
                "symbols": self.config.symbols,
                "max_position_pct": self.config.max_position_pct,
                "max_total_exposure_pct": self.config.max_total_exposure_pct,
                "leverage": self.config.leverage,
            }
        }
    
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


@with_error_handling(TradingBotError)
async def main() -> None:
    """Main entry point with enhanced error handling."""
    with performance_context("application_main"):
        # Display startup banner
        _display_startup_banner()
        
        system = DynamicIntegratedTradingSystem()
        
        # Setup signal handlers with proper error handling
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            # Use asyncio.create_task for proper async handling
            task = asyncio.create_task(system.stop())
            # Don't wait here to avoid blocking signal handler
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        with error_context({"operation": "main_execution"}):
            try:
                await system.start()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                await system.stop()
            except Exception as e:
                logger.error(f"System error: {e}")
                await system.stop()
                raise


def _display_startup_banner() -> None:
    """Display system startup banner."""
    print("🚀 Dynamic Parameter Integrated Trading System")
    print("💰 Real account balance monitoring")
    print("📊 Dynamic risk management")
    print("🔄 Microservice architecture")
    print(f"🌍 Environment: {'TESTNET' if settings.bybit.testnet else 'PRODUCTION'}")
    print(f"🎯 Symbols: {', '.join(settings.bybit.symbols)}")
    print("=" * 60)
    


if __name__ == "__main__":
    print("Starting Dynamic Parameter Integrated Trading System...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        sys.exit(1)