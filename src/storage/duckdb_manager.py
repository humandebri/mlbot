"""
DuckDB storage manager for time series data persistence.

Handles:
- Redis to DuckDB data transfer
- Efficient columnar storage with Parquet
- Time series queries and aggregations
- Data quality checks
"""

import duckdb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path

from ..common.config import settings
from ..common.logging import get_logger
from ..common.database import get_redis_client

logger = get_logger(__name__)


class DuckDBManager:
    """
    Manages DuckDB storage for market data and features.
    
    Optimized for:
    - Time series data storage
    - Fast analytical queries
    - Efficient compression with Parquet
    - Data versioning
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize DuckDB manager.
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path or str(Path("data/market_data.duckdb"))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self._connect()
        self._init_schema()
        
    def _connect(self):
        """Connect to DuckDB."""
        self.conn = duckdb.connect(self.db_path)
        
        # Configure for performance
        self.conn.execute("SET memory_limit='2GB'")
        self.conn.execute("SET threads TO 4")
        
    def _init_schema(self):
        """Initialize database schema."""
        # Market data tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS klines (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                trades INTEGER,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                side VARCHAR,
                price DOUBLE,
                size DOUBLE,
                is_liquidation BOOLEAN,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS liquidations (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                side VARCHAR,
                price DOUBLE,
                size DOUBLE,
                size_usd DOUBLE,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                bid_price_1 DOUBLE,
                bid_size_1 DOUBLE,
                ask_price_1 DOUBLE,
                ask_size_1 DOUBLE,
                bid_price_5 DOUBLE,
                bid_size_5 DOUBLE,
                ask_price_5 DOUBLE,
                ask_size_5 DOUBLE,
                total_bid_depth DOUBLE,
                total_ask_depth DOUBLE,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                feature_data JSON,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        # Create indexes for performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_klines_time ON klines(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_liquidations_time ON liquidations(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_liquidations_size ON liquidations(size_usd)")
        
    async def persist_from_redis(self, batch_size: int = 1000) -> Dict[str, int]:
        """
        Persist data from Redis to DuckDB.
        
        Args:
            batch_size: Number of records to process at once
            
        Returns:
            Statistics about persisted data
        """
        redis_client = await get_redis_client()
        stats = {}
        
        try:
            # Get all stream keys
            stream_keys = await redis_client.keys("market_data:*")
            
            for key in stream_keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                
                # Read messages from stream
                messages = await redis_client.xread({key_str: '0'}, count=batch_size)
                
                if messages:
                    stream_name, stream_messages = messages[0]
                    data_type = key_str.split(':')[1]
                    
                    # Process based on data type
                    if data_type == "kline":
                        count = await self._persist_klines(stream_messages)
                        stats["klines"] = count
                    elif data_type == "liquidation":
                        count = await self._persist_liquidations(stream_messages)
                        stats["liquidations"] = count
                    elif data_type == "trades":
                        count = await self._persist_trades(stream_messages)
                        stats["trades"] = count
                    elif data_type == "orderbook":
                        count = await self._persist_orderbook(stream_messages)
                        stats["orderbook"] = count
                        
            return stats
            
        except Exception as e:
            logger.error(f"Error persisting from Redis: {e}")
            return stats
            
    async def _persist_klines(self, messages: List[Tuple]) -> int:
        """Persist kline data."""
        records = []
        
        for msg_id, fields in messages:
            try:
                data = json.loads(fields.get(b"data", b"{}"))
                records.append({
                    "symbol": data["symbol"],
                    "timestamp": pd.Timestamp(data["timestamp"], unit="ms"),
                    "open": data["open"],
                    "high": data["high"],
                    "low": data["low"],
                    "close": data["close"],
                    "volume": data["volume"],
                    "trades": data.get("trades", 0)
                })
            except Exception as e:
                logger.warning(f"Error parsing kline message: {e}")
                
        if records:
            df = pd.DataFrame(records)
            self.conn.execute("INSERT OR REPLACE INTO klines SELECT * FROM df")
            
        return len(records)
        
    async def _persist_liquidations(self, messages: List[Tuple]) -> int:
        """Persist liquidation data."""
        records = []
        
        for msg_id, fields in messages:
            try:
                data = json.loads(fields.get(b"data", b"{}"))
                records.append({
                    "symbol": data["symbol"],
                    "timestamp": pd.Timestamp(data["timestamp"], unit="ms"),
                    "side": data["side"],
                    "price": data["price"],
                    "size": data["size"],
                    "size_usd": data["price"] * data["size"]
                })
            except Exception as e:
                logger.warning(f"Error parsing liquidation message: {e}")
                
        if records:
            df = pd.DataFrame(records)
            self.conn.execute("INSERT OR REPLACE INTO liquidations SELECT * FROM df")
            
        return len(records)
        
    async def _persist_trades(self, messages: List[Tuple]) -> int:
        """Persist trade data."""
        records = []
        
        for msg_id, fields in messages:
            try:
                data = json.loads(fields.get(b"data", b"{}"))
                records.append({
                    "symbol": data["symbol"],
                    "timestamp": pd.Timestamp(data["timestamp"], unit="ms"),
                    "side": data["side"],
                    "price": data["price"],
                    "size": data["size"],
                    "is_liquidation": data.get("is_liquidation", False)
                })
            except Exception as e:
                logger.warning(f"Error parsing trade message: {e}")
                
        if records:
            df = pd.DataFrame(records)
            self.conn.execute("INSERT OR REPLACE INTO trades SELECT * FROM df")
            
        return len(records)
        
    async def _persist_orderbook(self, messages: List[Tuple]) -> int:
        """Persist orderbook snapshot data."""
        records = []
        
        for msg_id, fields in messages:
            try:
                data = json.loads(fields.get(b"data", b"{}"))
                
                # Extract top 5 levels
                bids = data.get("bids", {})
                asks = data.get("asks", {})
                
                bid_prices = sorted([float(p) for p in bids.keys()], reverse=True)[:5]
                ask_prices = sorted([float(p) for p in asks.keys()])[:5]
                
                record = {
                    "symbol": data["symbol"],
                    "timestamp": pd.Timestamp(data["timestamp"], unit="ms"),
                    "total_bid_depth": sum(bids.values()),
                    "total_ask_depth": sum(asks.values())
                }
                
                # Add top levels
                for i, price in enumerate(bid_prices[:5], 1):
                    record[f"bid_price_{i}"] = price
                    record[f"bid_size_{i}"] = bids[str(price)]
                    
                for i, price in enumerate(ask_prices[:5], 1):
                    record[f"ask_price_{i}"] = price
                    record[f"ask_size_{i}"] = asks[str(price)]
                    
                records.append(record)
                
            except Exception as e:
                logger.warning(f"Error parsing orderbook message: {e}")
                
        if records:
            df = pd.DataFrame(records)
            # Fill missing levels with NaN
            for i in range(1, 6):
                for side in ["bid", "ask"]:
                    for field in ["price", "size"]:
                        col = f"{side}_{field}_{i}"
                        if col not in df.columns:
                            df[col] = np.nan
                            
            self.conn.execute("INSERT OR REPLACE INTO orderbook_snapshots SELECT * FROM df")
            
        return len(records)
        
    async def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of stored data."""
        summary = {}
        
        try:
            # Total records
            tables = ["klines", "liquidations", "trades", "orderbook_snapshots"]
            total_records = 0
            data_types = {}
            
            for table in tables:
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                data_types[table] = count
                total_records += count
                
            summary["total_records"] = total_records
            summary["data_types"] = data_types
            
            # Get symbols
            symbols = self.conn.execute(
                "SELECT DISTINCT symbol FROM klines"
            ).fetchall()
            summary["symbols"] = [s[0] for s in symbols]
            
            # Time range
            time_range = self.conn.execute("""
                SELECT MIN(timestamp) as start, MAX(timestamp) as end
                FROM klines
            """).fetchone()
            
            if time_range[0]:
                summary["time_range"] = {
                    "start": time_range[0].isoformat(),
                    "end": time_range[1].isoformat()
                }
            else:
                summary["time_range"] = {"start": None, "end": None}
                
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            
        return summary
        
    async def get_symbol_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistics for a specific symbol."""
        stats = {}
        
        try:
            # Record counts
            for table in ["klines", "liquidations", "trades"]:
                count = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE symbol = ?",
                    [symbol]
                ).fetchone()[0]
                stats[table] = count
                
            # Orderbook snapshots
            ob_count = self.conn.execute(
                "SELECT COUNT(*) FROM orderbook_snapshots WHERE symbol = ?",
                [symbol]
            ).fetchone()[0]
            stats["orderbook"] = ob_count
            
        except Exception as e:
            logger.error(f"Error getting symbol statistics: {e}")
            
        return stats
        
    async def get_liquidation_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get liquidation statistics for a symbol."""
        stats = {}
        
        try:
            result = self.conn.execute("""
                SELECT 
                    COUNT(*) as count,
                    SUM(size_usd) as total_volume,
                    AVG(size_usd) as avg_size,
                    MAX(size_usd) as max_size,
                    SUM(CASE WHEN side = 'Buy' THEN 1 ELSE 0 END) as long_count,
                    SUM(CASE WHEN side = 'Sell' THEN 1 ELSE 0 END) as short_count
                FROM liquidations
                WHERE symbol = ?
            """, [symbol]).fetchone()
            
            if result[0] > 0:
                stats = {
                    "count": result[0],
                    "total_volume": result[1],
                    "avg_size": result[2],
                    "max_size": result[3],
                    "long_ratio": result[4] / result[0],
                    "short_ratio": result[5] / result[0]
                }
                
        except Exception as e:
            logger.error(f"Error getting liquidation statistics: {e}")
            
        return stats
        
    async def check_time_gaps(self, symbol: str, expected_interval: int = 1) -> List[Dict]:
        """Check for gaps in time series data."""
        gaps = []
        
        try:
            # Get consecutive timestamps
            df = self.conn.execute("""
                SELECT timestamp FROM klines 
                WHERE symbol = ? 
                ORDER BY timestamp
            """, [symbol]).df()
            
            if len(df) > 1:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['diff'] = df['timestamp'].diff()
                
                # Find gaps larger than expected
                gap_threshold = pd.Timedelta(seconds=expected_interval * 2)
                gap_mask = df['diff'] > gap_threshold
                
                for idx in df[gap_mask].index:
                    gaps.append({
                        "timestamp": df.loc[idx, "timestamp"].isoformat(),
                        "duration": df.loc[idx, "diff"].total_seconds()
                    })
                    
        except Exception as e:
            logger.error(f"Error checking time gaps: {e}")
            
        return gaps
        
    async def check_data_anomalies(self) -> List[str]:
        """Check for data quality issues."""
        issues = []
        
        try:
            # Check for negative prices
            neg_prices = self.conn.execute("""
                SELECT COUNT(*) FROM klines 
                WHERE open < 0 OR high < 0 OR low < 0 OR close < 0
            """).fetchone()[0]
            
            if neg_prices > 0:
                issues.append(f"Found {neg_prices} records with negative prices")
                
            # Check for zero volumes
            zero_volumes = self.conn.execute("""
                SELECT COUNT(*) FROM klines WHERE volume = 0
            """).fetchone()[0]
            
            if zero_volumes > 0:
                issues.append(f"Found {zero_volumes} klines with zero volume")
                
            # Check for price consistency
            price_issues = self.conn.execute("""
                SELECT COUNT(*) FROM klines 
                WHERE low > high OR open > high OR close > high
                   OR low > open OR low > close
            """).fetchone()[0]
            
            if price_issues > 0:
                issues.append(f"Found {price_issues} klines with inconsistent prices")
                
        except Exception as e:
            logger.error(f"Error checking data anomalies: {e}")
            
        return issues
        
    async def load_features_for_training(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load feature data for model training."""
        try:
            query = "SELECT * FROM features"
            conditions = []
            
            if start_date:
                conditions.append(f"timestamp >= '{start_date}'")
            if end_date:
                conditions.append(f"timestamp <= '{end_date}'")
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY timestamp"
            
            df = self.conn.execute(query).df()
            
            if len(df) > 0:
                # Expand JSON feature data
                features = pd.json_normalize(df['feature_data'].apply(json.loads))
                df = pd.concat([df[['symbol', 'timestamp']], features], axis=1)
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return pd.DataFrame()
            
    async def create_training_labels(
        self,
        lookahead_seconds: int = 60,
        barrier_pct: float = 0.002
    ) -> pd.Series:
        """Create training labels based on future price movements."""
        try:
            # Get price data
            df = self.conn.execute("""
                SELECT symbol, timestamp, close as price
                FROM klines
                ORDER BY symbol, timestamp
            """).df()
            
            if len(df) == 0:
                return pd.Series()
                
            # Calculate future returns for each symbol
            labels = []
            
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data = symbol_data.set_index('timestamp').sort_index()
                
                # Calculate future price at lookahead horizon
                future_price = symbol_data['price'].shift(-lookahead_seconds)
                
                # Calculate expected PnL
                returns = (future_price / symbol_data['price'] - 1)
                
                # Apply barriers
                returns = returns.clip(-barrier_pct, barrier_pct)
                
                # Account for fees
                fee_rate = 0.00055
                returns = returns - fee_rate
                
                labels.extend(returns.values)
                
            return pd.Series(labels)
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return pd.Series()
            
    async def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()