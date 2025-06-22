#!/usr/bin/env python3
"""
Enhanced DuckDB updater that:
1. Updates from Redis to fill data gaps
2. Merges historical tables for extended history
3. Optimizes for higher confidence predictions
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


class EnhancedDuckDBUpdater:
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
        
        # Check data ranges for each relevant table
        for table_name in [t[0] for t in tables]:
            if 'kline' in table_name or table_name == 'all_klines':
                try:
                    # Determine if table has symbol column
                    columns = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
                    has_symbol = any('symbol' in str(col).lower() for col in columns)
                    
                    if has_symbol:
                        result = self.conn.execute(f"""
                            SELECT 
                                symbol,
                                COUNT(*) as records,
                                datetime(MIN(open_time/1000), 'unixepoch') as min_date,
                                datetime(MAX(open_time/1000), 'unixepoch') as max_date
                            FROM {table_name}
                            GROUP BY symbol
                        """).fetchall()
                        
                        for row in result:
                            logger.info(f"  {table_name} - {row[0]}: {row[1]:,} records, {row[2]} to {row[3]}")
                    else:
                        # For tables without symbol column
                        result = self.conn.execute(f"""
                            SELECT 
                                COUNT(*) as records,
                                datetime(MIN(open_time/1000), 'unixepoch') as min_date,
                                datetime(MAX(open_time/1000), 'unixepoch') as max_date
                            FROM {table_name}
                        """).fetchone()
                        
                        symbol = table_name.replace('klines_', '').replace('kline_', '').upper()
                        logger.info(f"  {table_name} - {symbol}: {result[0]:,} records, {result[1]} to {result[2]}")
                        
                except Exception as e:
                    logger.warning(f"  Error checking {table_name}: {e}")
    
    def create_unified_table(self):
        """Create a unified all_klines table if it doesn't exist."""
        logger.info("\n=== Creating Unified Table ===")
        
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
                turnover DOUBLE,
                PRIMARY KEY (symbol, open_time)
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
                    if symbol.lower() == symbol:
                        symbol = symbol.upper()
                else:
                    continue
                
                logger.info(f"Merging {table} as {symbol}...")
                
                # Check table structure
                columns = self.conn.execute(f"DESCRIBE {table}").fetchall()
                col_names = [col[0].lower() for col in columns]
                
                # Build column mapping
                if 'symbol' in col_names:
                    # Table already has symbol column
                    merge_query = f"""
                        INSERT INTO all_klines (symbol, open_time, open, high, low, close, volume, turnover)
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
                        ON CONFLICT (symbol, open_time) DO NOTHING
                    """
                else:
                    # Add symbol to query
                    merge_query = f"""
                        INSERT INTO all_klines (symbol, open_time, open, high, low, close, volume, turnover)
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
                        ON CONFLICT (symbol, open_time) DO NOTHING
                    """
                
                # Execute merge
                before_count = self.conn.execute("SELECT COUNT(*) FROM all_klines").fetchone()[0]
                self.conn.execute(merge_query)
                after_count = self.conn.execute("SELECT COUNT(*) FROM all_klines").fetchone()[0]
                
                added = after_count - before_count
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
                    
                    # Extract kline data
                    kline_data = parsed.get('data', [{}])
                    if isinstance(kline_data, list) and kline_data:
                        kline = kline_data[0]
                    else:
                        kline = kline_data
                    
                    record = {
                        'symbol': symbol,
                        'open_time': int(timestamp.timestamp() * 1000),
                        'open': float(kline.get('open', 0)),
                        'high': float(kline.get('high', 0)),
                        'low': float(kline.get('low', 0)),
                        'close': float(kline.get('close', 0)),
                        'volume': float(kline.get('volume', 0)),
                        'turnover': float(kline.get('turnover', 0))
                    }
                    
                    records_by_symbol[symbol].append(record)
                    
                except Exception as e:
                    logger.debug(f"Error parsing Redis entry: {e}")
        
        except Exception as e:
            logger.error(f"Error reading from Redis: {e}")
        
        # Insert new records
        total_added = 0
        for symbol, records in records_by_symbol.items():
            if records:
                df = pd.DataFrame(records)
                df = df.drop_duplicates(subset=['open_time'], keep='last')
                df = df.sort_values('open_time')
                
                # Register with DuckDB and insert
                self.conn.register('new_data', df)
                
                before = self.conn.execute(f"SELECT COUNT(*) FROM all_klines WHERE symbol = '{symbol}'").fetchone()[0]
                
                self.conn.execute("""
                    INSERT INTO all_klines
                    SELECT * FROM new_data
                    ON CONFLICT (symbol, open_time) DO NOTHING
                """)
                
                after = self.conn.execute(f"SELECT COUNT(*) FROM all_klines WHERE symbol = '{symbol}'").fetchone()[0]
                added = after - before
                
                total_added += added
                logger.info(f"Added {added} new records for {symbol}")
        
        logger.info(f"Total added from Redis: {total_added} records")
        return total_added
    
    def extend_lookback_config(self):
        """Show how to configure extended lookback period."""
        logger.info("\n=== Extended Lookback Configuration ===")
        
        # Check data availability
        result = self.conn.execute("""
            SELECT 
                symbol,
                COUNT(*) as records,
                MIN(datetime(open_time/1000, 'unixepoch')) as min_date,
                MAX(datetime(open_time/1000, 'unixepoch')) as max_date,
                (MAX(open_time) - MIN(open_time)) / (1000 * 60 * 60 * 24) as days_span
            FROM all_klines
            GROUP BY symbol
        """).fetchall()
        
        logger.info("\nData availability for extended lookback:")
        for row in result:
            logger.info(f"  {row[0]}: {row[1]:,} records, {row[4]:.1f} days ({row[2]} to {row[3]})")
        
        logger.info("\nTo use extended lookback period, update improved_feature_generator_enhanced.py:")
        logger.info("  1. Change lookback_days parameter: load_historical_data(symbol, lookback_days=120)")
        logger.info("  2. Or modify the enhanced generator initialization:")
        logger.info("     gen = ImprovedFeatureGeneratorEnhanced()")
        logger.info("     gen.load_historical_data('BTCUSDT', lookback_days=180)")
    
    def verify_update(self):
        """Verify the update and show statistics."""
        logger.info("\n=== Verification ===")
        
        # Overall statistics
        total = self.conn.execute("SELECT COUNT(*) FROM all_klines").fetchone()[0]
        logger.info(f"\nTotal records in all_klines: {total:,}")
        
        # Per symbol statistics
        result = self.conn.execute("""
            SELECT 
                symbol,
                COUNT(*) as records,
                datetime(MIN(open_time/1000), 'unixepoch') as min_date,
                datetime(MAX(open_time/1000), 'unixepoch') as max_date
            FROM all_klines
            GROUP BY symbol
            ORDER BY symbol
        """).fetchall()
        
        logger.info("\nPer symbol statistics:")
        for row in result:
            logger.info(f"  {row[0]}: {row[1]:,} records, {row[2]} to {row[3]}")
        
        # Check for data gaps
        logger.info("\nChecking for data gaps...")
        for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
            try:
                # Get time differences between consecutive records
                gaps = self.conn.execute(f"""
                    WITH ordered_data AS (
                        SELECT 
                            open_time,
                            LAG(open_time) OVER (ORDER BY open_time) as prev_time
                        FROM all_klines
                        WHERE symbol = '{symbol}'
                        ORDER BY open_time
                    )
                    SELECT 
                        COUNT(*) as gap_count
                    FROM ordered_data
                    WHERE open_time - prev_time > 120000  -- More than 2 minutes gap
                """).fetchone()[0]
                
                if gaps > 0:
                    logger.warning(f"  {symbol}: Found {gaps} gaps > 2 minutes")
                else:
                    logger.info(f"  {symbol}: No significant gaps found")
                    
            except Exception as e:
                logger.error(f"  Error checking gaps for {symbol}: {e}")
    
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
            
            # Show extended lookback config
            self.extend_lookback_config()
            
            # Verify results
            self.verify_update()
            
            logger.info("\n✅ Update completed successfully!")
            logger.info("\nNext steps to achieve 50%+ confidence:")
            logger.info("1. The database now has extended historical data")
            logger.info("2. Update the bot's lookback_days parameter to use more history")
            logger.info("3. Restart the bot to use the updated data")
            
        except Exception as e:
            logger.error(f"Error during update: {e}")
            raise
        finally:
            self.conn.close()
            self.redis_client.close()


if __name__ == "__main__":
    # Run with optional lookback parameter
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced DuckDB updater')
    parser.add_argument('--lookback-hours', type=int, default=72, 
                        help='Hours of Redis data to fetch (default: 72)')
    args = parser.parse_args()
    
    updater = EnhancedDuckDBUpdater()
    updater.run_full_update()