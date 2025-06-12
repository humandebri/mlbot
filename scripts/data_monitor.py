#!/usr/bin/env python3
"""
Comprehensive data monitoring and visualization tool.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging
from src.common.database import get_redis_client
from src.storage.duckdb_manager import DuckDBManager

setup_logging()
logger = get_logger(__name__)


class DataMonitor:
    """Comprehensive data monitoring and analysis tool."""
    
    def __init__(self):
        self.redis_client = None
        self.duckdb_manager = None
        self.production_duckdb_manager = None
        
    async def initialize(self):
        """Initialize connections."""
        self.redis_client = await get_redis_client()
        self.duckdb_manager = DuckDBManager(db_path="data/market_data.duckdb")
        self.production_duckdb_manager = DuckDBManager(db_path="data/production_market_data.duckdb")
        
    async def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connection and data streams."""
        try:
            # Ping Redis
            await self.redis_client.ping()
            
            # Get all keys
            keys = await self.redis_client.keys("*")
            
            # Analyze market data streams
            stream_stats = {}
            market_streams = [k.decode() for k in keys if k.decode().startswith("market_data:")]
            
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
                        stream_stats[stream]["latest_messages"].append({
                            "id": msg_id.decode(),
                            "timestamp": datetime.fromtimestamp(int(msg_id.decode().split('-')[0]) / 1000),
                            "fields": len(fields)
                        })
            
            # Check feature keys
            feature_keys = [k.decode() for k in keys if k.decode().startswith("features:")]
            
            return {
                "status": "healthy",
                "total_keys": len(keys),
                "stream_count": len(market_streams),
                "feature_keys": len(feature_keys),
                "streams": stream_stats,
                "feature_symbols": [k.split(":")[1] for k in feature_keys if ":latest" in k]
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_duckdb_data(self, manager: DuckDBManager, db_name: str) -> Dict[str, Any]:
        """Check DuckDB data quality and statistics."""
        try:
            # Get table list
            tables_df = manager.conn.execute("SHOW TABLES").df()
            tables = tables_df["name"].tolist() if not tables_df.empty else []
            
            table_stats = {}
            
            for table in tables:
                try:
                    # Get row count
                    count_result = manager.conn.execute(f"SELECT COUNT(*) as count FROM {table}").df()
                    row_count = count_result["count"].iloc[0] if not count_result.empty else 0
                    
                    # Get column info
                    columns_df = manager.conn.execute(f"DESCRIBE {table}").df()
                    columns = columns_df["column_name"].tolist() if not columns_df.empty else []
                    
                    # Get latest timestamp if available
                    latest_timestamp = None
                    if "timestamp" in columns:
                        try:
                            latest_df = manager.conn.execute(
                                f"SELECT MAX(timestamp) as latest FROM {table}"
                            ).df()
                            if not latest_df.empty and latest_df["latest"].iloc[0] is not None:
                                latest_timestamp = latest_df["latest"].iloc[0]
                        except:
                            pass
                    
                    table_stats[table] = {
                        "row_count": row_count,
                        "columns": len(columns),
                        "column_names": columns,
                        "latest_timestamp": latest_timestamp
                    }
                
                except Exception as e:
                    table_stats[table] = {"error": str(e)}
            
            return {
                "status": "healthy",
                "database": db_name,
                "table_count": len(tables),
                "tables": table_stats
            }
            
        except Exception as e:
            return {"status": "error", "database": db_name, "error": str(e)}
    
    async def get_collection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics."""
        redis_stats = await self.check_redis_health()
        testnet_stats = self.check_duckdb_data(self.duckdb_manager, "testnet")
        production_stats = self.check_duckdb_data(self.production_duckdb_manager, "production")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "redis": redis_stats,
            "testnet_db": testnet_stats,
            "production_db": production_stats
        }
    
    async def get_recent_data_sample(self, stream_name: str, count: int = 5) -> List[Dict]:
        """Get recent data samples from a Redis stream."""
        try:
            messages = await self.redis_client.xrevrange(f"market_data:{stream_name}", count=count)
            samples = []
            
            for msg_id, fields in messages:
                sample = {
                    "id": msg_id.decode(),
                    "timestamp": datetime.fromtimestamp(int(msg_id.decode().split('-')[0]) / 1000),
                }
                
                # Parse fields
                for key, value in fields.items():
                    key_str = key.decode()
                    value_str = value.decode()
                    
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
        output.append(f"📊 DATA COLLECTION MONITOR - {stats['timestamp']}")
        output.append("=" * 80)
        
        # Redis stats
        redis = stats.get("redis", {})
        if redis.get("status") == "healthy":
            output.append(f"\n🔴 REDIS STATUS: ✅ Healthy")
            output.append(f"   Total Keys: {redis['total_keys']}")
            output.append(f"   Streams: {redis['stream_count']}")
            output.append(f"   Feature Symbols: {redis['feature_symbols']}")
            
            output.append(f"\n   📈 Stream Details:")
            for stream, info in redis.get("streams", {}).items():
                output.append(f"      {stream}: {info['length']} messages")
                if info['latest_messages']:
                    latest = info['latest_messages'][0]
                    output.append(f"         Latest: {latest['timestamp']}")
        else:
            output.append(f"\n🔴 REDIS STATUS: ❌ Error - {redis.get('error')}")
        
        # Database stats
        for db_key, db_name in [("testnet_db", "Testnet"), ("production_db", "Production")]:
            db_stats = stats.get(db_key, {})
            if db_stats.get("status") == "healthy":
                output.append(f"\n💾 {db_name.upper()} DB: ✅ Healthy")
                output.append(f"   Tables: {db_stats['table_count']}")
                
                for table, info in db_stats.get("tables", {}).items():
                    if "error" not in info:
                        output.append(f"      {table}: {info['row_count']:,} rows")
                        if info.get('latest_timestamp'):
                            output.append(f"         Latest: {info['latest_timestamp']}")
                    else:
                        output.append(f"      {table}: ❌ {info['error']}")
            else:
                output.append(f"\n💾 {db_name.upper()} DB: ❌ Error - {db_stats.get('error')}")
        
        output.append("\n" + "=" * 80)
        return "\n".join(output)
    
    async def monitor_loop(self, interval: int = 30):
        """Continuous monitoring loop."""
        print("🚀 Starting data monitoring...")
        
        while True:
            try:
                stats = await self.get_collection_statistics()
                
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
    
    parser = argparse.ArgumentParser(description="Data collection monitor")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--sample", type=str, help="Show sample data from stream (e.g., 'kline')")
    
    args = parser.parse_args()
    
    monitor = DataMonitor()
    await monitor.initialize()
    
    if args.sample:
        # Show sample data
        samples = await monitor.get_recent_data_sample(args.sample)
        print(f"🔍 Recent samples from {args.sample}:")
        for i, sample in enumerate(samples, 1):
            print(f"\n{i}. {sample['timestamp']}")
            if 'data' in sample:
                print(json.dumps(sample['data'], indent=2))
    
    elif args.once:
        # Run once
        stats = await monitor.get_collection_statistics()
        print(monitor.format_stats_output(stats))
    
    else:
        # Continuous monitoring
        await monitor.monitor_loop(args.interval)


if __name__ == "__main__":
    asyncio.run(main())