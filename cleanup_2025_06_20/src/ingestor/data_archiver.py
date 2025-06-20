"""Cost-efficient data archiving to Parquet with DuckDB optimization."""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..common.config import settings
from ..common.database import get_duckdb_connection, get_redis_client
from ..common.logging import get_logger
from ..common.monitoring import (
    increment_counter, 
    observe_histogram,
    ARCHIVE_BATCH_SIZE,
    ARCHIVER_ERRORS,
)

logger = get_logger(__name__)


class DataArchiver:
    """
    Cost-optimized data archiver for market data.
    
    Features:
    - Efficient Parquet compression
    - Automated data lifecycle management
    - Background processing to minimize latency impact
    - Storage cost optimization
    """
    
    def __init__(self):
        self.running = False
        self.archive_interval = 3600  # Archive every hour
        self.retention_days = 30  # Keep data for 30 days
        self.batch_size = 10000  # Process in batches for memory efficiency
        
        # Storage paths
        self.data_dir = Path("data")
        self.parquet_dir = self.data_dir / "parquet"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        # Data types to archive
        self.data_types = ["kline", "orderbook", "trades", "liquidation", "open_interest", "funding"]
        
        # Compression settings for cost optimization
        self.compression_config = {
            "compression": settings.duckdb.parquet_compression,
            "row_group_size": settings.duckdb.row_group_size,
        }
    
    async def start(self) -> None:
        """Start the background archiving process."""
        self.running = True
        logger.info(
            "Starting data archiver",
            archive_interval=self.archive_interval,
            retention_days=self.retention_days,
            compression=self.compression_config["compression"]
        )
        
        while self.running:
            try:
                await self._archive_cycle()
                await asyncio.sleep(self.archive_interval)
            except Exception as e:
                logger.error("Error in archiving cycle", exception=e)
                await asyncio.sleep(60)  # Retry after 1 minute on error
    
    async def stop(self) -> None:
        """Stop the archiving process gracefully."""
        logger.info("Stopping data archiver")
        self.running = False
    
    async def _archive_cycle(self) -> None:
        """Perform one complete archiving cycle."""
        try:
            # Get current hour boundary for consistent archiving
            now = datetime.utcnow()
            archive_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            
            logger.info(f"Starting archive cycle for hour: {archive_hour}")
            
            # Archive each data type
            for data_type in self.data_types:
                await self._archive_data_type(data_type, archive_hour)
            
            # Cleanup old files
            await self._cleanup_old_files()
            
            logger.info("Archive cycle completed successfully")
            
        except Exception as e:
            logger.error("Error in archive cycle", exception=e)
            ARCHIVER_ERRORS.labels(error_type=type(e).__name__).inc()
    
    async def _archive_data_type(self, data_type: str, archive_hour: datetime) -> None:
        """Archive specific data type for the given hour."""
        try:
            # Get data from Redis Streams
            data_records = await self._fetch_stream_data(data_type, archive_hour)
            
            if not data_records:
                logger.debug(f"No data to archive for {data_type} at {archive_hour}")
                return
            
            # Convert to DataFrame for efficient processing
            df = await self._prepare_dataframe(data_records, data_type)
            
            if df.empty:
                return
            
            # Generate filename with hour-based partitioning
            filename = self._generate_filename(data_type, archive_hour)
            
            # Save to Parquet with optimization
            await self._save_to_parquet(df, filename)
            
            # Also save to DuckDB for fast querying
            await self._save_to_duckdb(df, data_type, archive_hour)
            
            logger.info(
                f"Archived {len(df)} {data_type} records",
                hour=archive_hour.isoformat(),
                file_size_mb=self._get_file_size_mb(filename)
            )
            
            ARCHIVE_BATCH_SIZE.labels(data_type=data_type).observe(len(df))
            
        except Exception as e:
            logger.error(f"Error archiving {data_type}", exception=e)
            ARCHIVER_ERRORS.labels(error_type=type(e).__name__).inc()
    
    async def _fetch_stream_data(self, data_type: str, archive_hour: datetime) -> List[Dict[str, Any]]:
        """Fetch data from Redis Streams for archiving."""
        try:
            redis_client = await get_redis_client()
            stream_name = f"market_data:{data_type}"
            
            # Calculate time range for the hour
            start_time = int(archive_hour.timestamp() * 1000)
            end_time = int((archive_hour + timedelta(hours=1)).timestamp() * 1000)
            
            # Read all messages in the time range
            # Note: This is a simplified approach. In production, you'd want
            # to use consumer groups and acknowledge messages after archiving
            try:
                stream_info = await redis_client.xinfo_stream(stream_name)
                if stream_info.get("length", 0) == 0:
                    return []
            except Exception:
                # Stream doesn't exist yet
                return []
            
            # For this implementation, we'll read recent messages
            # In production, you'd want to track message IDs and ranges
            messages = await redis_client.xrange(stream_name, count=self.batch_size)
            
            records = []
            for message_id, fields in messages:
                try:
                    import json
                    data = json.loads(fields.get("data", "{}"))
                    
                    # Filter by timestamp if available
                    msg_timestamp = data.get("timestamp", 0)
                    if start_time <= msg_timestamp * 1000 <= end_time:
                        records.append(data)
                    
                except json.JSONDecodeError:
                    continue
            
            return records
            
        except Exception as e:
            logger.error(f"Error fetching stream data for {data_type}", exception=e)
            return []
    
    async def _prepare_dataframe(
        self, 
        records: List[Dict[str, Any]], 
        data_type: str
    ) -> pd.DataFrame:
        """Prepare and optimize DataFrame for archiving."""
        if not records:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(records)
            
            # Data type specific optimizations
            if data_type == "kline":
                df = self._optimize_kline_dataframe(df)
            elif data_type == "orderbook":
                df = self._optimize_orderbook_dataframe(df)
            elif data_type == "trades":
                df = self._optimize_trades_dataframe(df)
            elif data_type == "liquidation":
                df = self._optimize_liquidation_dataframe(df)
            
            # Common optimizations
            df = self._apply_common_optimizations(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing DataFrame for {data_type}", exception=e)
            return pd.DataFrame()
    
    def _optimize_kline_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize kline DataFrame for storage efficiency."""
        try:
            # Convert timestamps
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Optimize numeric types for storage
            numeric_columns = ["open", "high", "low", "close", "volume", "turnover"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            
            # Categorical columns for compression
            categorical_columns = ["symbol", "interval"]
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype("category")
            
            return df
            
        except Exception as e:
            logger.warning(f"Error optimizing kline DataFrame: {e}")
            return df
    
    def _optimize_orderbook_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize orderbook DataFrame for storage efficiency."""
        try:
            # Convert timestamps
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Optimize numeric types
            numeric_columns = ["bid_depth", "ask_depth", "depth_ratio", "spread", "mid_price"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            
            # Categorical columns
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].astype("category")
            
            # Remove large JSON columns for archiving (keep only aggregated metrics)
            columns_to_drop = ["bids_top5", "asks_top5"]
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            return df
            
        except Exception as e:
            logger.warning(f"Error optimizing orderbook DataFrame: {e}")
            return df
    
    def _optimize_trades_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize trades DataFrame for storage efficiency."""
        try:
            # Convert timestamps
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Optimize numeric types
            numeric_columns = ["price", "size"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            
            # Categorical columns
            categorical_columns = ["symbol", "side"]
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype("category")
            
            return df
            
        except Exception as e:
            logger.warning(f"Error optimizing trades DataFrame: {e}")
            return df
    
    def _optimize_liquidation_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize liquidation DataFrame for storage efficiency."""
        try:
            # Convert timestamps
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Optimize numeric types
            numeric_columns = ["size", "price", "spike_severity"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            
            # Categorical columns
            categorical_columns = ["symbol", "side", "spike_type"]
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype("category")
            
            # Boolean optimization
            if "spike_detected" in df.columns:
                df["spike_detected"] = df["spike_detected"].astype("bool")
            
            return df
            
        except Exception as e:
            logger.warning(f"Error optimizing liquidation DataFrame: {e}")
            return df
    
    def _apply_common_optimizations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply common DataFrame optimizations."""
        try:
            # Remove completely null columns
            df = df.dropna(axis=1, how="all")
            
            # Sort by timestamp for better compression
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            
            # Reset index for cleaner storage
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error applying common optimizations: {e}")
            return df
    
    def _generate_filename(self, data_type: str, archive_hour: datetime) -> Path:
        """Generate optimized filename for Parquet storage."""
        # Use date-based partitioning for efficient querying
        date_str = archive_hour.strftime("%Y-%m-%d")
        hour_str = archive_hour.strftime("%H")
        
        # Create directory structure
        type_dir = self.parquet_dir / data_type / date_str
        type_dir.mkdir(parents=True, exist_ok=True)
        
        filename = type_dir / f"{data_type}_{hour_str}.parquet"
        return filename
    
    async def _save_to_parquet(self, df: pd.DataFrame, filename: Path) -> None:
        """Save DataFrame to Parquet with optimal compression."""
        try:
            # Use asyncio thread pool for I/O
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: df.to_parquet(
                    filename,
                    engine="pyarrow",
                    compression=self.compression_config["compression"],
                    index=False,
                    row_group_size=self.compression_config["row_group_size"],
                )
            )
            
        except Exception as e:
            logger.error(f"Error saving to Parquet: {filename}", exception=e)
            raise
    
    async def _save_to_duckdb(
        self, 
        df: pd.DataFrame, 
        data_type: str, 
        archive_hour: datetime
    ) -> None:
        """Save data to DuckDB for fast querying."""
        try:
            conn = get_duckdb_connection()
            
            # Create table name
            table_name = f"archived_{data_type}"
            
            # Create table if it doesn't exist
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} AS 
                    SELECT * FROM df WHERE 1=0
                """)
            )
            
            # Insert data
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            )
            
        except Exception as e:
            logger.warning(f"Error saving to DuckDB for {data_type}", exception=e)
            # Don't raise - DuckDB is optional for archiving
    
    async def _cleanup_old_files(self) -> None:
        """Remove old archived files to manage storage costs."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            deleted_count = 0
            
            for data_type in self.data_types:
                type_dir = self.parquet_dir / data_type
                if not type_dir.exists():
                    continue
                
                # Walk through date directories
                for date_dir in type_dir.iterdir():
                    if not date_dir.is_dir():
                        continue
                    
                    try:
                        dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                        if dir_date < cutoff_date:
                            # Remove entire day directory
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self._remove_directory(date_dir)
                            )
                            deleted_count += 1
                    except ValueError:
                        # Invalid date format, skip
                        continue
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old archive directories")
            
        except Exception as e:
            logger.error("Error cleaning up old files", exception=e)
    
    def _remove_directory(self, directory: Path) -> None:
        """Safely remove a directory and its contents."""
        import shutil
        try:
            shutil.rmtree(directory)
        except Exception as e:
            logger.warning(f"Error removing directory {directory}: {e}")
    
    def _get_file_size_mb(self, filename: Path) -> float:
        """Get file size in MB."""
        try:
            if filename.exists():
                return filename.stat().st_size / (1024 * 1024)
        except Exception:
            pass
        return 0.0
    
    async def get_archive_stats(self) -> Dict[str, Any]:
        """Get archiving statistics for monitoring."""
        try:
            stats = {
                "total_files": 0,
                "total_size_mb": 0.0,
                "data_types": {},
                "retention_days": self.retention_days,
            }
            
            for data_type in self.data_types:
                type_dir = self.parquet_dir / data_type
                if not type_dir.exists():
                    continue
                
                type_stats = {"files": 0, "size_mb": 0.0}
                
                for file_path in type_dir.rglob("*.parquet"):
                    type_stats["files"] += 1
                    file_size = self._get_file_size_mb(file_path)
                    type_stats["size_mb"] += file_size
                
                stats["data_types"][data_type] = type_stats
                stats["total_files"] += type_stats["files"]
                stats["total_size_mb"] += type_stats["size_mb"]
            
            return stats
            
        except Exception as e:
            logger.error("Error getting archive stats", exception=e)
            return {"error": str(e)}