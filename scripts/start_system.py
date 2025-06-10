#!/usr/bin/env python3
"""
Script to start the trading system.

Usage:
    python scripts/start_system.py [--testnet] [--dashboard]
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import aiohttp
from src.common.logging import get_logger

logger = get_logger(__name__)


async def check_system_health(api_url: str) -> bool:
    """Check if system is healthy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url}/system/health", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "healthy"
    except:
        pass
    return False


async def start_system(api_url: str) -> bool:
    """Start the trading system via API."""
    try:
        async with aiohttp.ClientSession() as session:
            logger.info("Starting trading system...")
            
            async with session.post(f"{api_url}/system/start", timeout=60) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"System start response: {data['message']}")
                    
                    # Wait for system to be healthy
                    for i in range(30):
                        await asyncio.sleep(2)
                        if await check_system_health(api_url):
                            logger.info("System is healthy and ready!")
                            return True
                        logger.info(f"Waiting for system to be ready... ({i+1}/30)")
                    
                    logger.warning("System started but health check failed")
                    return False
                else:
                    error = await response.text()
                    logger.error(f"Failed to start system: {error}")
                    return False
                    
    except Exception as e:
        logger.error(f"Error starting system: {e}")
        return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start the liquidation trading bot system")
    parser.add_argument("--testnet", action="store_true", help="Use testnet mode")
    parser.add_argument("--dashboard", action="store_true", help="Launch monitoring dashboard")
    parser.add_argument("--api-url", default="http://localhost:8080", help="API gateway URL")
    
    args = parser.parse_args()
    
    # Set environment variables
    if args.testnet:
        os.environ["USE_TESTNET"] = "true"
        logger.info("Running in TESTNET mode")
    
    # First, start the API gateway in a subprocess
    import subprocess
    
    logger.info("Starting API Gateway...")
    gateway_process = subprocess.Popen(
        [sys.executable, "-m", "src.integration.main"],
        cwd=project_root
    )
    
    # Wait for gateway to be ready
    await asyncio.sleep(5)
    
    try:
        # Start the trading system
        success = await start_system(args.api_url)
        
        if success:
            logger.info("Trading system started successfully!")
            
            if args.dashboard:
                # Launch dashboard in another subprocess
                logger.info("Launching monitoring dashboard...")
                dashboard_process = subprocess.Popen(
                    [sys.executable, str(project_root / "monitoring" / "dashboard.py")],
                    cwd=project_root
                )
                
                # Wait for processes
                try:
                    gateway_process.wait()
                except KeyboardInterrupt:
                    logger.info("Shutting down...")
                    
                    # Stop system via API
                    async with aiohttp.ClientSession() as session:
                        await session.post(f"{args.api_url}/system/stop")
                    
                    gateway_process.terminate()
                    if args.dashboard:
                        dashboard_process.terminate()
            else:
                # Just wait for gateway
                gateway_process.wait()
        else:
            logger.error("Failed to start trading system")
            gateway_process.terminate()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        gateway_process.terminate()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())