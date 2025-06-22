#!/usr/bin/env python3
"""
Fixed version of DuckDB updater with proper SQL syntax
"""

import duckdb
import redis
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedDuckDBUpdaterFixed:
    def __init__(self, db_path="data/historical_data.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
    def check_database_state(self):
        """Check current state of the database."""
        logger.info("=== Checking Database State ===")
        
        # List all tables
        tables = self.conn.execute("SHOW TABLES").fetchall()
        logger.info(f"Tables found: {[t[0] for t in tables]}")
        
        # Check if all_klines exists
        if 'all_klines' in [t[0] for t in tables]:
            try:
                result = self.conn.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as records,
                        to_timestamp(MIN(open_time/1000)) as min_date,
                        to_timestamp(MAX(open_time/1000)) as max_date
                    FROM all_klines
                    GROUP BY symbol
                """).fetchall()
                
                for row in result:
                    logger.info(f"  all_klines - {row[0]}: {row[1]:,} records, {row[2]} to {row[3]}")
            except Exception as e:
                logger.warning(f"  Error checking all_klines: {e}")
    
    def create_unified_table(self):
        """Create a unified all_klines table if it doesn't exist."""
        logger.info("\n=== Creating Unified Table ===")
        
        # Drop old view if exists
        self.conn.execute("DROP VIEW IF EXISTS all_klines")
        
        # Create all_klines table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS all_klines (
                symbol VARCHAR,
                open_time BIGINT,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                turnover DOUBLE
            )
        """)
        logger.info("Created/verified all_klines table")
    
    def merge_historical_tables(self):
        """Merge all historical kline tables into unified all_klines."""
        logger.info("\n=== Merging Historical Tables ===")
        
        tables = self.conn.execute("SHOW TABLES").fetchall()
        kline_tables = [t[0] for t in tables if ('kline' in t[0] and t[0] != 'all_klines')]
        
        total_merged = 0
        
        for table in kline_tables:
            try:
                # Extract symbol from table name
                if table.startswith('klines_'):
                    symbol = table.replace('klines_', '').upper()
                elif table.startswith('kline_'):
                    symbol = table.replace('kline_', '')
                else:
                    continue
                
                logger.info(f"Merging {table} as {symbol}...")
                
                # Count existing records
                existing = self.conn.execute(f"""
                    SELECT COUNT(*) FROM all_klines WHERE symbol = '{symbol}'
                """).fetchone()[0]
                
                # Check table structure
                columns = self.conn.execute(f"DESCRIBE {table}").fetchall()
                col_names = [col[0].lower() for col in columns]
                
                # Insert new records
                if 'symbol' in col_names:
                    result = self.conn.execute(f"""
                        INSERT INTO all_klines
                        SELECT 
                            symbol,
                            open_time,
                            open,
                            high,
                            low,
                            close,
                            volume,
                            COALESCE(turnover, volume * close) as turnover
                        FROM {table}
                        WHERE open_time IS NOT NULL
                        AND NOT EXISTS (
                            SELECT 1 FROM all_klines a 
                            WHERE a.symbol = {table}.symbol 
                            AND a.open_time = {table}.open_time
                        )
                    """)
                else:
                    result = self.conn.execute(f"""
                        INSERT INTO all_klines
                        SELECT 
                            '{symbol}' as symbol,
                            open_time,
                            open,
                            high,
                            low,
                            close,
                            volume,
                            COALESCE(turnover, volume * close) as turnover
                        FROM {table}
                        WHERE open_time IS NOT NULL
                        AND NOT EXISTS (
                            SELECT 1 FROM all_klines a 
                            WHERE a.symbol = '{symbol}' 
                            AND a.open_time = {table}.open_time
                        )
                    """)
                
                # Count new records
                new_count = self.conn.execute(f"""
                    SELECT COUNT(*) FROM all_klines WHERE symbol = '{symbol}'
                """).fetchone()[0]
                
                added = new_count - existing
                total_merged += added
                logger.info(f"  Added {added:,} records from {table}")
                
            except Exception as e:
                logger.error(f"Error merging {table}: {e}")
        
        logger.info(f"Total merged: {total_merged:,} records")
        return total_merged
    
    def update_from_redis(self, lookback_hours=24):
        """Update with latest data from Redis."""
        logger.info(f"\n=== Updating from Redis (last {lookback_hours} hours) ===")
        
        # Get latest timestamps from database
        latest_times = {}
        try:
            result = self.conn.execute("""
                SELECT symbol, MAX(open_time) as max_time
                FROM all_klines
                GROUP BY symbol
            """).fetchall()
            
            for symbol, max_time in result:
                if max_time:
                    latest_times[symbol] = datetime.fromtimestamp(max_time / 1000)
                    logger.info(f"Latest {symbol} data: {latest_times[symbol]}")
        except Exception as e:
            logger.warning(f"Error getting latest times: {e}")
        
        # Fetch from Redis
        stream_key = 'market_data:kline'
        records_by_symbol = {'BTCUSDT': [], 'ETHUSDT': [], 'ICPUSDT': []}
        
        try:
            # Get entries from Redis
            entries = self.redis_client.xrevrange(stream_key, count=20000)
            logger.info(f"Found {len(entries)} entries in Redis")
            
            parsed_count = 0
            for entry_id, data in entries:
                try:
                    parsed = json.loads(data.get('data', '{}'))
                    topic = parsed.get('topic', '')
                    
                    # Determine symbol
                    symbol = None
                    for sym in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
                        if sym in topic:
                            symbol = sym
                            break
                    
                    if not symbol:
                        continue
                    
                    # Parse timestamp
                    timestamp = datetime.fromtimestamp(parsed.get('timestamp', 0))
                    
                    # Skip if older than latest in DB
                    if symbol in latest_times and timestamp <= latest_times[symbol]:
                        continue
                    
                    # Skip if too old
                    if timestamp < datetime.utcnow() - timedelta(hours=lookback_hours):
                        continue
                    
                    # Extract kline data from the nested structure
                    kline_data = parsed.get('data', [])
                    if isinstance(kline_data, list) and len(kline_data) > 0:
                        kline = kline_data[0]
                    else:
                        continue
                    
                    record = {
                        'symbol': symbol,
                        'open_time': int(kline.get('timestamp', timestamp.timestamp() * 1000)),
                        'open': float(kline.get('open', 0)),
                        'high': float(kline.get('high', 0)),
                        'low': float(kline.get('low', 0)),
                        'close': float(kline.get('close', 0)),
                        'volume': float(kline.get('volume', 0)),
                        'turnover': float(kline.get('turnover', 0))
                    }
                    
                    records_by_symbol[symbol].append(record)
                    parsed_count += 1
                    
                except Exception as e:
                    logger.debug(f"Error parsing Redis entry: {e}")
            
            logger.info(f"Successfully parsed {parsed_count} records")
            
        except Exception as e:
            logger.error(f"Error reading from Redis: {e}")
        
        # Insert new records
        total_added = 0
        for symbol, records in records_by_symbol.items():
            if records:
                df = pd.DataFrame(records)
                df = df.drop_duplicates(subset=['open_time'], keep='last')
                df = df.sort_values('open_time')
                
                logger.info(f"Inserting {len(df)} records for {symbol}")
                
                # Register with DuckDB and insert
                self.conn.register('new_data', df)
                
                try:
                    self.conn.execute(f"""
                        INSERT INTO all_klines
                        SELECT * FROM new_data
                        WHERE NOT EXISTS (
                            SELECT 1 FROM all_klines a
                            WHERE a.symbol = new_data.symbol
                            AND a.open_time = new_data.open_time
                        )
                    """)
                    
                    total_added += len(df)
                    logger.info(f"Successfully added records for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error inserting {symbol} data: {e}")
        
        logger.info(f"Total added from Redis: {total_added} records")
        return total_added
    
    def verify_update(self):
        """Verify the update and show statistics."""
        logger.info("\n=== Verification ===")
        
        try:
            # Overall statistics
            total = self.conn.execute("SELECT COUNT(*) FROM all_klines").fetchone()[0]
            logger.info(f"\nTotal records in all_klines: {total:,}")
            
            # Per symbol statistics
            result = self.conn.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as records,
                    to_timestamp(MIN(open_time/1000)) as min_date,
                    to_timestamp(MAX(open_time/1000)) as max_date
                FROM all_klines
                GROUP BY symbol
                ORDER BY symbol
            """).fetchall()
            
            logger.info("\nPer symbol statistics:")
            for row in result:
                logger.info(f"  {row[0]}: {row[1]:,} records, {row[2]} to {row[3]}")
                
        except Exception as e:
            logger.error(f"Error during verification: {e}")
    
    def run_full_update(self):
        """Run the complete update process."""
        logger.info("Starting enhanced DuckDB update process...\n")
        
        try:
            # Check initial state
            self.check_database_state()
            
            # Create unified table
            self.create_unified_table()
            
            # Merge historical tables
            self.merge_historical_tables()
            
            # Update from Redis
            self.update_from_redis(lookback_hours=72)  # Get last 3 days
            
            # Verify results
            self.verify_update()
            
            logger.info("\n✅ Update completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during update: {e}")
            raise
        finally:
            self.conn.close()
            self.redis_client.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fixed DuckDB updater')
    parser.add_argument('--lookback-hours', type=int, default=72, 
                        help='Hours of Redis data to fetch (default: 72)')
    args = parser.parse_args()
    
    updater = EnhancedDuckDBUpdaterFixed()
    updater.run_full_update()