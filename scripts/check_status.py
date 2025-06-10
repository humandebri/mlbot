#!/usr/bin/env python3
"""
Script to check system status.

Usage:
    python scripts/check_status.py [--json]
"""

import asyncio
import aiohttp
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger

logger = get_logger(__name__)


async def get_system_status(api_url: str = "http://localhost:8080") -> dict:
    """Get system status from API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url}/system/status", timeout=5) as response:
                if response.status == 200:
                    return await response.json()
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
    
    return None


async def get_health_status(api_url: str = "http://localhost:8080") -> dict:
    """Get health status from API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url}/system/health", timeout=5) as response:
                if response.status == 200:
                    return await response.json()
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
    
    return None


def print_status_summary(status: dict, health: dict):
    """Print human-readable status summary."""
    print("\n" + "="*60)
    print("LIQUIDATION TRADING BOT STATUS")
    print("="*60)
    
    # System Health
    print("\n🏥 SYSTEM HEALTH")
    print("-" * 30)
    
    if health:
        overall = health.get("status", "unknown")
        health_emoji = "✅" if overall == "healthy" else "⚠️" if overall == "degraded" else "❌"
        print(f"Overall Status: {health_emoji} {overall.upper()}")
        
        components = health.get("components", {})
        for name, is_healthy in components.items():
            status_text = "✅ Running" if is_healthy else "❌ Down"
            print(f"  {name}: {status_text}")
    else:
        print("❌ Unable to retrieve health status")
    
    # Trading Status
    print("\n📊 TRADING STATUS")
    print("-" * 30)
    
    if status and "trading_system" in status and isinstance(status["trading_system"], dict):
        trading = status["trading_system"]
        
        if "statistics" in trading:
            stats = trading["statistics"]
            print(f"Predictions Made: {stats.get('predictions_total', 0)}")
            print(f"Signals Generated: {stats.get('signals_generated', 0)}")
            print(f"Active Symbols: {', '.join(stats.get('active_symbols', []))}")
        
        if "trading" in trading:
            router = trading["trading"]
            
            # Performance
            if "performance" in router:
                perf = router["performance"]
                print(f"\n💰 Performance:")
                print(f"  Total P&L: ${perf.get('total_pnl', 0):,.2f}")
                print(f"  Win Rate: {perf.get('win_rate', 0):.1%}")
                print(f"  Total Trades: {perf.get('total_trades', 0)}")
            
            # Risk
            if "risk" in router:
                risk = router["risk"]
                print(f"\n⚠️  Risk Metrics:")
                print(f"  Current Equity: ${risk.get('current_equity', 0):,.2f}")
                print(f"  Total Exposure: ${risk.get('total_exposure', 0):,.2f}")
                print(f"  Current Drawdown: {risk.get('current_drawdown', 0):.1%}")
                
                if risk.get('trading_halted'):
                    print(f"  🛑 TRADING HALTED: {risk.get('halt_reason', 'Unknown')}")
            
            # Positions
            print(f"\n📈 Active Positions: {router.get('active_positions', 0)}")
    else:
        print("❌ Trading system not running or not initialized")
    
    # Services Status
    print("\n⚙️  SERVICES")
    print("-" * 30)
    
    if status and "services" in status:
        services = status["services"]
        for name, service in services.items():
            status_emoji = "✅" if service["status"] == "running" else "❌"
            print(f"{status_emoji} {name}: {service['status']}")
            if service.get("pid"):
                print(f"   PID: {service['pid']}")
            if service.get("cpu_percent") is not None:
                print(f"   CPU: {service['cpu_percent']:.1f}%")
            if service.get("memory_mb") is not None:
                print(f"   Memory: {service['memory_mb']:.1f} MB")
    
    print("\n" + "="*60)
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check trading system status")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--api-url", default="http://localhost:8080", help="API gateway URL")
    
    args = parser.parse_args()
    
    # Get status
    status = await get_system_status(args.api_url)
    health = await get_health_status(args.api_url)
    
    if args.json:
        # JSON output
        output = {
            "status": status,
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        if status or health:
            print_status_summary(status, health)
        else:
            print("❌ Unable to connect to trading system API")
            print("   Make sure the system is running with: python scripts/start_system.py")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())