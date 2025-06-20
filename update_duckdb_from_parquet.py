#!/usr/bin/env python3
"""
Update DuckDB historical data from Parquet files
This will merge the latest kline data from Redis/Parquet into the historical database
"""

import duckdb
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_historical_data():
    """Update historical_data.duckdb with latest data from parquet files."""
    
    conn = duckdb.connect('data/historical_data.duckdb')
    
    try:
        # First check what data we currently have
        result = conn.execute("""
            SELECT 
                symbol,
                MIN(timestamp) as min_date,
                MAX(timestamp) as max_date,
                COUNT(*) as record_count
            FROM kline_BTCUSDT
            GROUP BY symbol
        """).fetchall()
        
        logger.info("Current data in DuckDB:")
        for row in result:
            logger.info(f"  {row[0]}: {row[1]} to {row[2]} ({row[3]} records)")
        
        # Get list of parquet files
        parquet_dirs = []
        kline_dir = "data/parquet/kline"
        
        if os.path.exists(kline_dir):
            for date_dir in sorted(os.listdir(kline_dir)):
                if date_dir.startswith("2025-06"):
                    full_path = os.path.join(kline_dir, date_dir)
                    if os.path.isdir(full_path):
                        parquet_dirs.append(full_path)
        
        logger.info(f"\nFound parquet directories: {parquet_dirs}")
        
        # Import each parquet file
        for dir_path in parquet_dirs:
            date_str = os.path.basename(dir_path)
            
            for parquet_file in sorted(os.listdir(dir_path)):
                if parquet_file.endswith('.parquet'):
                    file_path = os.path.join(dir_path, parquet_file)
                    logger.info(f"\nProcessing {file_path}...")
                    
                    # Read parquet file
                    df = conn.execute(f"""
                        SELECT * FROM read_parquet('{file_path}')
                    """).df()
                    
                    if not df.empty:
                        logger.info(f"  Found {len(df)} records")
                        
                        # Get unique symbols
                        symbols = df['symbol'].unique()
                        
                        for symbol in symbols:
                            symbol_df = df[df['symbol'] == symbol]
                            table_name = f"kline_{symbol}"
                            
                            # Create table if not exists
                            conn.execute(f"""
                                CREATE TABLE IF NOT EXISTS {table_name} (
                                    timestamp BIGINT,
                                    topic VARCHAR,
                                    symbol VARCHAR,
                                    interval VARCHAR,
                                    open_time BIGINT,
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
                            
                            # Insert data (ignore duplicates based on open_time)
                            conn.execute(f"""
                                INSERT INTO {table_name}
                                SELECT * FROM symbol_df
                                WHERE open_time NOT IN (
                                    SELECT open_time FROM {table_name}
                                )
                            """)
                            
                            logger.info(f"    Inserted data for {symbol}")
        
        # Check final state
        logger.info("\nFinal data state:")
        for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
            try:
                result = conn.execute(f"""
                    SELECT 
                        COUNT(*) as count,
                        MIN(timestamp) as min_date,
                        MAX(timestamp) as max_date
                    FROM kline_{symbol}
                """).fetchone()
                
                logger.info(f"  {symbol}: {result[0]} records, {result[1]} to {result[2]}")
            except:
                logger.warning(f"  {symbol}: No data found")
        
        # Create unified view with all symbols
        conn.execute("""
            CREATE OR REPLACE VIEW all_klines AS
            SELECT * FROM kline_BTCUSDT
            UNION ALL
            SELECT * FROM kline_ETHUSDT
            UNION ALL
            SELECT * FROM kline_ICPUSDT
        """)
        
        logger.info("\nCreated unified view 'all_klines'")
        
    finally:
        conn.close()
    
    logger.info("\nDatabase update completed!")

if __name__ == "__main__":
    update_historical_data()