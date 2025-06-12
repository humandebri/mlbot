#!/usr/bin/env python3
"""
Real-time data collection dashboard and monitoring interface.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging
from src.common.database import get_redis_client

setup_logging()
logger = get_logger(__name__)


class DataDashboard:
    """Real-time data collection dashboard."""
    
    def __init__(self):
        self.redis_client = None
        self.stats_history = []
        self.start_time = datetime.now()
        
    async def initialize(self):
        """Initialize Redis connection."""
        self.redis_client = await get_redis_client()
        
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive data collection statistics."""
        try:
            # Get all keys
            keys = await self.redis_client.keys("*")
            
            # Categorize keys
            market_streams = []
            feature_keys = []
            other_keys = []
            
            for k in keys:
                key_str = k.decode() if hasattr(k, 'decode') else str(k)
                if key_str.startswith("market_data:"):
                    market_streams.append(key_str)
                elif key_str.startswith("features:"):
                    feature_keys.append(key_str)
                else:
                    other_keys.append(key_str)
            
            # Analyze market data streams
            stream_stats = {}
            total_messages = 0
            latest_activity = None
            
            for stream in market_streams:
                length = await self.redis_client.xlen(stream)
                total_messages += length
                
                stream_info = {
                    "length": length,
                    "rate_per_minute": 0,
                    "latest_timestamp": None,
                    "age_seconds": 0
                }
                
                # Get latest message for activity analysis
                if length > 0:
                    messages = await self.redis_client.xrevrange(stream, count=1)
                    if messages:
                        msg_id, _ = messages[0]
                        msg_id_str = msg_id.decode() if hasattr(msg_id, 'decode') else str(msg_id)
                        timestamp_ms = int(msg_id_str.split('-')[0])
                        timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                        
                        stream_info["latest_timestamp"] = timestamp
                        stream_info["age_seconds"] = (datetime.now() - timestamp).total_seconds()
                        
                        if latest_activity is None or timestamp > latest_activity:
                            latest_activity = timestamp
                
                # Calculate message rate (last 5 minutes)
                five_min_ago = int((datetime.now() - timedelta(minutes=5)).timestamp() * 1000)
                try:
                    recent_messages = await self.redis_client.xrevrange(
                        stream, 
                        min=f"{five_min_ago}-0", 
                        count=10000
                    )
                    stream_info["rate_per_minute"] = len(recent_messages) / 5.0
                except:
                    pass
                
                stream_stats[stream] = stream_info
            
            # Analyze features
            feature_stats = {}
            active_symbols = set()
            
            for key in feature_keys:
                if ":latest" in key:
                    symbol = key.split(":")[1]
                    active_symbols.add(symbol)
                    
                    try:
                        feature_data = await self.redis_client.get(key)
                        if feature_data:
                            feature_str = feature_data.decode() if hasattr(feature_data, 'decode') else str(feature_data)
                            parsed_data = json.loads(feature_str)
                            
                            feature_stats[symbol] = {
                                "features_count": len(parsed_data.get("features", {})),
                                "timestamp": parsed_data.get("timestamp"),
                                "last_update": None
                            }
                            
                            if parsed_data.get("timestamp"):
                                feature_stats[symbol]["last_update"] = datetime.fromtimestamp(
                                    parsed_data["timestamp"]
                                )
                    except Exception as e:
                        feature_stats[symbol] = {"error": str(e)}
            
            return {
                "timestamp": datetime.now(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "total_keys": len(keys),
                "market_streams": {
                    "count": len(market_streams),
                    "total_messages": total_messages,
                    "latest_activity": latest_activity,
                    "streams": stream_stats
                },
                "features": {
                    "active_symbols": sorted(list(active_symbols)),
                    "symbol_count": len(active_symbols),
                    "stats": feature_stats
                }
            }
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now()}
    
    def format_dashboard(self, stats: Dict[str, Any]) -> str:
        """Format comprehensive dashboard output."""
        if "error" in stats:
            return f"❌ Error: {stats['error']}"
        
        output = []
        now = stats["timestamp"]
        uptime = timedelta(seconds=stats["uptime_seconds"])
        
        # Header
        output.append("╔" + "═" * 78 + "╗")
        output.append(f"║{'🚀 BYBIT ML BOT - DATA COLLECTION DASHBOARD':^78}║")
        output.append(f"║{now.strftime('%Y-%m-%d %H:%M:%S'):^78}║")
        output.append(f"║{'Uptime: ' + str(uptime).split('.')[0]:^78}║")
        output.append("╠" + "═" * 78 + "╣")
        
        # Market Data Overview
        market = stats["market_streams"]
        latest_activity = market.get("latest_activity")
        if latest_activity:
            activity_age = (now - latest_activity).total_seconds()
            if activity_age < 60:
                activity_str = f"{activity_age:.1f}s ago"
            else:
                activity_str = f"{activity_age/60:.1f}m ago"
        else:
            activity_str = "No activity"
        
        output.append(f"║ 📊 MARKET DATA: {market['count']} streams | {market['total_messages']:,} total messages")
        output.append(f"║    Latest Activity: {activity_str}")
        output.append("╠" + "─" * 78 + "╣")
        
        # Stream Details
        output.append("║ 📈 STREAM STATUS:")
        
        # Group streams by type
        stream_types = defaultdict(list)
        for stream_name, info in market["streams"].items():
            stream_type = stream_name.split(":")[-1]
            stream_types[stream_type].append((stream_name, info))
        
        for stream_type, streams in sorted(stream_types.items()):
            total_msgs = sum(info["length"] for _, info in streams)
            avg_rate = sum(info["rate_per_minute"] for _, info in streams) / len(streams) if streams else 0
            
            output.append(f"║   {stream_type:12} | {total_msgs:8,} msgs | {avg_rate:5.1f} msg/min avg")
        
        output.append("╠" + "─" * 78 + "╣")
        
        # Feature Processing
        features = stats["features"]
        output.append(f"║ 🧠 FEATURE PROCESSING: {features['symbol_count']} symbols")
        output.append(f"║    Active Symbols: {', '.join(features['active_symbols'][:10])}")
        if len(features['active_symbols']) > 10:
            output.append(f"║                   + {len(features['active_symbols']) - 10} more...")
        
        output.append("╠" + "─" * 78 + "╣")
        
        # Symbol Details
        output.append("║ 💎 SYMBOL DETAILS:")
        for symbol, info in sorted(features["stats"].items()):
            if "error" not in info:
                feature_count = info.get("features_count", 0)
                last_update = info.get("last_update")
                if last_update:
                    age = (now - last_update).total_seconds()
                    if age < 60:
                        age_str = f"{age:.0f}s"
                    else:
                        age_str = f"{age/60:.1f}m"
                else:
                    age_str = "N/A"
                
                output.append(f"║   {symbol:10} | {feature_count:3d} features | Updated {age_str} ago")
            else:
                output.append(f"║   {symbol:10} | ❌ Error: {info['error'][:35]}")
        
        output.append("╚" + "═" * 78 + "╝")
        
        return "\n".join(output)
    
    async def run_dashboard(self, refresh_interval: int = 10):
        """Run the real-time dashboard."""
        print("🚀 Starting Data Collection Dashboard...")
        print("Press Ctrl+C to exit\n")
        
        try:
            while True:
                # Get latest stats
                stats = await self.get_comprehensive_stats()
                
                # Clear screen and display dashboard
                print("\033[2J\033[H")  # Clear screen and move cursor to top
                print(self.format_dashboard(stats))
                
                # Add footer with refresh info
                print(f"\n⏱️  Auto-refresh every {refresh_interval}s | Press Ctrl+C to exit")
                
                # Wait for next refresh
                await asyncio.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard stopped. Data collection continues in background.")
    
    async def show_current_status(self):
        """Show current status once and exit."""
        stats = await self.get_comprehensive_stats()
        print(self.format_dashboard(stats))


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Collection Dashboard")
    parser.add_argument(
        "--refresh", 
        type=int, 
        default=10, 
        help="Dashboard refresh interval in seconds"
    )
    parser.add_argument(
        "--once", 
        action="store_true", 
        help="Show status once and exit"
    )
    
    args = parser.parse_args()
    
    dashboard = DataDashboard()
    await dashboard.initialize()
    
    if args.once:
        await dashboard.show_current_status()
    else:
        await dashboard.run_dashboard(args.refresh)


if __name__ == "__main__":
    asyncio.run(main())