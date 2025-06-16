#!/usr/bin/env python3
"""Test integration system startup."""

import asyncio
import sys
import logging
from src.integration.api_gateway import app
from src.integration.service_manager import ServiceManager
from src.integration.trading_coordinator import TradingCoordinator, SystemConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

async def test_startup():
    """Test startup of all services."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize service manager
        logger.info("Initializing service manager...")
        service_manager = ServiceManager()
        
        # Start all services
        logger.info("Starting all services...")
        await service_manager.start_all()
        
        # Wait a bit for services to stabilize
        await asyncio.sleep(5)
        
        # Check service status
        status = service_manager.get_status()
        logger.info(f"Service status: {status}")
        
        # Initialize trading coordinator
        logger.info("Initializing trading coordinator...")
        config = SystemConfig()
        coordinator = TradingCoordinator(config)
        
        # Initialize coordinator
        await coordinator.initialize(service_manager)
        
        logger.info("All services started successfully!")
        
        # Keep running for testing
        await asyncio.sleep(30)
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
    finally:
        # Cleanup
        if 'service_manager' in locals():
            await service_manager.stop_all()
        if 'coordinator' in locals():
            await coordinator.shutdown()

if __name__ == "__main__":
    asyncio.run(test_startup())