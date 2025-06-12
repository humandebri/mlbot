#!/usr/bin/env python3
"""
Redis-only data monitoring tool (no DuckDB conflicts).
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging
from src.common.database import get_redis_client

setup_logging()
logger = get_logger(__name__)


class RedisMonitor:
    """Redis-only data monitoring tool."""
    
    def __init__(self):
        self.redis_client = None
        
    async def initialize(self):
        """Initialize Redis connection."""
        self.redis_client = await get_redis_client()
        
    async def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connection and data streams."""
        try:
            # Ping Redis
            await self.redis_client.ping()
            
            # Get all keys
            keys = await self.redis_client.keys("*")
            
            # Analyze market data streams
            stream_stats = {}
            market_streams = []
            for k in keys:
                key_str = k.decode() if hasattr(k, 'decode') else str(k)
                if key_str.startswith("market_data:"):
                    market_streams.append(key_str)
            
            for stream in market_streams:
                length = await self.redis_client.xlen(stream)
                stream_stats[stream] = {
                    "length": length,
                    "latest_messages": []
                }
                
                # Get latest messages
                if length > 0:
                    messages = await self.redis_client.xrevrange(stream, count=3)
                    for msg_id, fields in messages:
                        msg_id_str = msg_id.decode() if hasattr(msg_id, 'decode') else str(msg_id)
                        timestamp_ms = int(msg_id_str.split('-')[0])
                        stream_stats[stream]["latest_messages"].append({
                            "id": msg_id_str,
                            "timestamp": datetime.fromtimestamp(timestamp_ms / 1000),
                            "fields": len(fields),
                            "age_seconds": (datetime.now().timestamp() * 1000 - timestamp_ms) / 1000
                        })
            
            # Check feature keys
            feature_keys = []
            for k in keys:
                key_str = k.decode() if hasattr(k, 'decode') else str(k)
                if key_str.startswith("features:"):
                    feature_keys.append(key_str)
            feature_symbols = []
            feature_stats = {}
            
            for key in feature_keys:
                if ":latest" in key:
                    symbol = key.split(":")[1]
                    feature_symbols.append(symbol)
                    
                    # Get feature data
                    try:
                        feature_data = await self.redis_client.get(key)
                        if feature_data:
                            feature_str = feature_data.decode() if hasattr(feature_data, 'decode') else str(feature_data)
                            parsed_data = json.loads(feature_str)
                            feature_stats[symbol] = {
                                "features_count": len(parsed_data.get("features", {})),
                                "timestamp": parsed_data.get("timestamp"),
                                "symbol": parsed_data.get("symbol")
                            }
                    except Exception as e:
                        feature_stats[symbol] = {"error": str(e)}
            
            return {
                "status": "healthy",
                "total_keys": len(keys),
                "stream_count": len(market_streams),
                "feature_keys": len(feature_keys),
                "streams": stream_stats,
                "feature_symbols": feature_symbols,
                "feature_stats": feature_stats
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def get_recent_data_sample(self, stream_name: str, count: int = 3) -> List[Dict]:
        """Get recent data samples from a Redis stream."""
        try:
            messages = await self.redis_client.xrevrange(f"market_data:{stream_name}", count=count)
            samples = []
            
            for msg_id, fields in messages:
                msg_id_str = msg_id.decode() if hasattr(msg_id, 'decode') else str(msg_id)
                timestamp_ms = int(msg_id_str.split('-')[0])
                sample = {
                    "id": msg_id_str,
                    "timestamp": datetime.fromtimestamp(timestamp_ms / 1000),
                    "age_seconds": (datetime.now().timestamp() * 1000 - timestamp_ms) / 1000
                }
                
                # Parse fields
                for key, value in fields.items():
                    key_str = key.decode() if hasattr(key, 'decode') else str(key)
                    value_str = value.decode() if hasattr(value, 'decode') else str(value)
                    
                    if key_str == "data":
                        try:
                            sample["data"] = json.loads(value_str)
                        except:
                            sample["data"] = value_str
                    else:
                        sample[key_str] = value_str
                
                samples.append(sample)
            
            return samples
            
        except Exception as e:
            logger.error(f"Error getting data sample for {stream_name}: {e}")
            return []
    
    def format_stats_output(self, stats: Dict[str, Any]) -> str:
        """Format statistics for console output."""
        output = []
        output.append("=" * 80)
        output.append(f"📊 REDIS DATA MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80)
        
        # Redis stats
        if stats.get("status") == "healthy":
            output.append(f"\n🔴 REDIS STATUS: ✅ Healthy")
            output.append(f"   Total Keys: {stats['total_keys']}")
            output.append(f"   Market Data Streams: {stats['stream_count']}")
            output.append(f"   Feature Keys: {stats['feature_keys']}")
            output.append(f"   Active Symbols: {', '.join(stats['feature_symbols'])}")
            
            # Stream details
            output.append(f"\n📈 MARKET DATA STREAMS:")
            for stream, info in stats.get("streams", {}).items():
                stream_name = stream.replace("market_data:", "")
                output.append(f"   {stream_name:15} | {info['length']:6,} messages")
                
                if info['latest_messages']:
                    latest = info['latest_messages'][0]
                    age = latest['age_seconds']
                    if age < 60:
                        age_str = f"{age:.1f}s ago"
                    elif age < 3600:
                        age_str = f"{age/60:.1f}m ago"
                    else:
                        age_str = f"{age/3600:.1f}h ago"
                    output.append(f"                    | Latest: {age_str}")
            
            # Feature stats
            if stats['feature_stats']:
                output.append(f"\n🧠 FEATURE PROCESSING:")
                for symbol, info in stats['feature_stats'].items():
                    if "error" not in info:
                        output.append(f"   {symbol:10} | {info['features_count']:3d} features")
                        if info.get('timestamp'):
                            feature_time = datetime.fromtimestamp(info['timestamp'])
                            age = (datetime.now() - feature_time).total_seconds()
                            if age < 60:
                                age_str = f"{age:.1f}s ago"
                            else:
                                age_str = f"{age/60:.1f}m ago"
                            output.append(f"              | Updated: {age_str}")
                    else:
                        output.append(f"   {symbol:10} | ❌ Error: {info['error']}")
        else:
            output.append(f"\n🔴 REDIS STATUS: ❌ Error - {stats.get('error')}")
        
        output.append("\n" + "=" * 80)
        return "\n".join(output)
    
    async def show_stream_samples(self, stream_names: List[str] = None):
        """Show sample data from streams."""
        if not stream_names:
            stream_names = ["kline", "orderbook", "trades", "liquidation"]
        
        print(f"🔍 Sample data from streams:")
        
        for stream_name in stream_names:
            samples = await self.get_recent_data_sample(stream_name, count=2)
            print(f"\n📊 {stream_name.upper()} (latest 2 messages):")
            
            if samples:
                for i, sample in enumerate(samples, 1):
                    print(f"   {i}. {sample['timestamp']} ({sample['age_seconds']:.1f}s ago)")
                    if 'data' in sample and isinstance(sample['data'], dict):
                        # Show key fields
                        data = sample['data']
                        if stream_name == "kline":
                            print(f"      Symbol: {data.get('symbol')}, Close: {data.get('close')}, Volume: {data.get('volume')}")
                        elif stream_name == "orderbook":
                            print(f"      Symbol: {data.get('symbol')}, Spread: {data.get('spread')}, Mid: {data.get('mid_price')}")
                        elif stream_name == "trades":
                            print(f"      Symbol: {data.get('symbol')}, Price: {data.get('price')}, Size: {data.get('size')}, Side: {data.get('side')}")
                        elif stream_name == "liquidation":
                            print(f"      Symbol: {data.get('symbol')}, Price: {data.get('price')}, Size: {data.get('size')}, Side: {data.get('side')}")
            else:
                print(f"   No recent data")
    
    async def monitor_loop(self, interval: int = 30):
        """Continuous monitoring loop."""
        print("🚀 Starting Redis data monitoring...")
        
        while True:
            try:
                stats = await self.check_redis_health()
                
                # Clear screen and show stats
                print("\033[2J\033[H")  # Clear screen
                print(self.format_stats_output(stats))
                
                # Wait for next check
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n👋 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(interval)


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Redis data collection monitor")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--samples", action="store_true", help="Show sample data")
    parser.add_argument("--stream", type=str, help="Show specific stream data")
    
    args = parser.parse_args()
    
    monitor = RedisMonitor()
    await monitor.initialize()
    
    if args.samples:
        # Show sample data
        await monitor.show_stream_samples()
    
    elif args.stream:
        # Show specific stream
        samples = await monitor.get_recent_data_sample(args.stream)
        print(f"🔍 Recent samples from {args.stream}:")
        for i, sample in enumerate(samples, 1):
            print(f"\n{i}. {sample['timestamp']} ({sample['age_seconds']:.1f}s ago)")
            if 'data' in sample:
                print(json.dumps(sample['data'], indent=2))
    
    elif args.once:
        # Run once
        stats = await monitor.check_redis_health()
        print(monitor.format_stats_output(stats))
    
    else:
        # Continuous monitoring
        await monitor.monitor_loop(args.interval)


if __name__ == "__main__":
    asyncio.run(main())