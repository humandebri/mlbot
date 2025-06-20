#!/usr/bin/env python3
"""
Complete DuckDB update from both Parquet files and Redis Streams
Improved version with better performance and error handling
"""

import duckdb
import redis
import json
import pandas as pd
import os
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DuckDBUpdater:
    def __init__(self, db_path='data/historical_data.duckdb'):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']
        
    def create_optimized_tables(self):
        """Create tables with proper indexes for performance."""
        for symbol in self.symbols:
            table_name = f"kline_{symbol}"
            
            # Create table with primary key
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    open_time BIGINT PRIMARY KEY,
                    timestamp_ms BIGINT,
                    topic VARCHAR,
                    symbol VARCHAR,
                    interval VARCHAR,
                    close_time BIGINT,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    turnover DOUBLE,
                    confirm BOOLEAN
                )
            """)
            logger.info(f"Created/verified table {table_name}")
    
    def update_from_parquet(self):
        """Update from Parquet files with efficient duplicate handling."""
        kline_dir = "data/parquet/kline"
        
        if not os.path.exists(kline_dir):
            logger.warning(f"Parquet directory not found: {kline_dir}")
            return
        
        # Get all parquet files
        parquet_files = []
        for date_dir in sorted(os.listdir(kline_dir)):
            if date_dir.startswith("2025-06"):
                full_path = os.path.join(kline_dir, date_dir)
                if os.path.isdir(full_path):
                    for file in os.listdir(full_path):
                        if file.endswith('.parquet'):
                            parquet_files.append(os.path.join(full_path, file))
        
        logger.info(f"Found {len(parquet_files)} parquet files to process")
        
        for file_path in parquet_files:
            try:
                logger.info(f"Processing {file_path}...")
                
                # Read parquet and process each symbol
                df = self.conn.execute(f"""
                    SELECT 
                        CAST(open_time AS BIGINT) as open_time,
                        CAST(timestamp * 1000 AS BIGINT) as timestamp_ms,
                        topic,
                        symbol,
                        interval,
                        CAST(close_time AS BIGINT) as close_time,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        turnover,
                        confirm
                    FROM read_parquet('{file_path}')
                    WHERE symbol IN ('BTCUSDT', 'ETHUSDT', 'ICPUSDT')
                """).df()
                
                if df.empty:
                    continue
                
                # Insert data for each symbol using INSERT OR IGNORE
                for symbol in df['symbol'].unique():
                    symbol_df = df[df['symbol'] == symbol]
                    table_name = f"kline_{symbol}"
                    
                    # Register DataFrame and insert
                    self.conn.register('temp_df', symbol_df)
                    result = self.conn.execute(f"""
                        INSERT OR IGNORE INTO {table_name}
                        SELECT * FROM temp_df
                    """)
                    
                    logger.info(f"  Processed {len(symbol_df)} records for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
    
    def update_from_redis(self):
        """Update from Redis Streams with proper timestamp handling."""
        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Test Redis connection
            r.ping()
            logger.info("Connected to Redis")
            
            # Get latest entries from Redis (using xrevrange for most recent first)
            stream_data = r.xrevrange('market_data:kline', count=50000)
            
            if not stream_data:
                logger.warning("No data found in Redis stream")
                return
            
            logger.info(f"Found {len(stream_data)} entries in Redis")
            
            # Parse Redis data
            records = []
            for entry_id, data in stream_data:
                try:
                    parsed = json.loads(data.get('data', '{}'))
                    
                    # Extract symbol from topic
                    topic = parsed.get('topic', '')
                    symbol = None
                    for s in self.symbols:
                        if s in topic:
                            symbol = s
                            break
                    
                    if not symbol:
                        continue
                    
                    # Handle timestamp - ensure it's in milliseconds
                    timestamp = parsed.get('timestamp', 0)
                    if timestamp < 1e10:  # Likely in seconds
                        timestamp_ms = int(timestamp * 1000)
                    else:
                        timestamp_ms = int(timestamp)
                    
                    record = {
                        'open_time': int(parsed.get('open_time', 0)),
                        'timestamp_ms': timestamp_ms,
                        'topic': topic,
                        'symbol': symbol,
                        'interval': str(parsed.get('interval', '1')),
                        'close_time': int(parsed.get('close_time', 0)),
                        'open': float(parsed.get('open', 0)),
                        'high': float(parsed.get('high', 0)),
                        'low': float(parsed.get('low', 0)),
                        'close': float(parsed.get('close', 0)),
                        'volume': float(parsed.get('volume', 0)),
                        'turnover': float(parsed.get('turnover', 0)),
                        'confirm': bool(parsed.get('confirm', False))
                    }
                    
                    records.append(record)
                    
                except Exception as e:
                    logger.debug(f"Error parsing entry: {e}")
                    continue
            
            if not records:
                logger.warning("No valid records parsed from Redis")
                return
            
            # Convert to DataFrame and insert
            df = pd.DataFrame(records)
            logger.info(f"Parsed {len(df)} valid records from Redis")
            
            # Insert for each symbol
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol]
                table_name = f"kline_{symbol}"
                
                self.conn.register('redis_df', symbol_df)
                result = self.conn.execute(f"""
                    INSERT OR IGNORE INTO {table_name}
                    SELECT * FROM redis_df
                """)
                
                logger.info(f"  Inserted records for {symbol}")
                
        except redis.ConnectionError:
            logger.error("Failed to connect to Redis. Make sure Redis is running.")
        except Exception as e:
            logger.error(f"Error updating from Redis: {e}")
    
    def create_views_and_indexes(self):
        """Create optimized views and additional indexes."""
        # Create unified view
        self.conn.execute("""
            CREATE OR REPLACE VIEW all_klines AS
            SELECT * FROM kline_BTCUSDT
            UNION ALL
            SELECT * FROM kline_ETHUSDT
            UNION ALL
            SELECT * FROM kline_ICPUSDT
        """)
        
        # Create indexes for common queries
        for symbol in self.symbols:
            table_name = f"kline_{symbol}"
            try:
                # Index on timestamp for time-based queries
                self.conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp 
                    ON {table_name}(timestamp_ms)
                """)
            except:
                pass  # Index might already exist
        
        logger.info("Created views and indexes")
    
    def show_statistics(self):
        """Display final statistics."""
        logger.info("\n" + "="*50)
        logger.info("FINAL DATABASE STATISTICS")
        logger.info("="*50)
        
        for symbol in self.symbols:
            try:
                result = self.conn.execute(f"""
                    SELECT 
                        COUNT(*) as total_records,
                        datetime(MIN(open_time/1000), 'unixepoch') as earliest,
                        datetime(MAX(open_time/1000), 'unixepoch') as latest,
                        ROUND((MAX(open_time) - MIN(open_time)) / 3600000.0, 1) as hours_covered
                    FROM kline_{symbol}
                """).fetchone()
                
                logger.info(f"\n{symbol}:")
                logger.info(f"  Records: {result[0]:,}")
                logger.info(f"  Period: {result[1]} to {result[2]}")
                logger.info(f"  Coverage: {result[3]} hours")
                
                # Check for recent data
                recent = self.conn.execute(f"""
                    SELECT COUNT(*) as recent_count
                    FROM kline_{symbol}
                    WHERE open_time > {int((time.time() - 3600) * 1000)}
                """).fetchone()
                
                logger.info(f"  Recent (last hour): {recent[0]} records")
                
            except Exception as e:
                logger.error(f"Error getting stats for {symbol}: {e}")
    
    def run_full_update(self):
        """Run complete update process."""
        logger.info("Starting complete DuckDB update...")
        
        # Create optimized tables
        self.create_optimized_tables()
        
        # Update from Parquet files
        logger.info("\n--- Updating from Parquet files ---")
        self.update_from_parquet()
        
        # Update from Redis
        logger.info("\n--- Updating from Redis streams ---")
        self.update_from_redis()
        
        # Create views and indexes
        self.create_views_and_indexes()
        
        # Show final statistics
        self.show_statistics()
        
        logger.info("\n✅ Database update completed successfully!")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    updater = DuckDBUpdater()
    try:
        updater.run_full_update()
    finally:
        updater.close()