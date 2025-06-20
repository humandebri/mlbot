#!/usr/bin/env python3
"""
Update DuckDB historical data from Redis Streams
This will add the most recent kline data from Redis to the historical database
"""

import duckdb
import redis
import json
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_from_redis():
    """Update historical_data.duckdb with latest data from Redis streams."""
    
    # Connect to Redis
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    # Connect to DuckDB
    conn = duckdb.connect('data/historical_data.duckdb')
    
    try:
        # Get latest kline data from Redis
        logger.info("Fetching data from Redis streams...")
        
        # Read last 10000 entries from kline stream
        kline_data = r.xrevrange('market_data:kline', count=10000)
        
        if not kline_data:
            logger.warning("No kline data found in Redis")
            return
        
        logger.info(f"Found {len(kline_data)} kline entries in Redis")
        
        # Convert to DataFrame
        records = []
        for entry_id, data in kline_data:
            try:
                # Parse the JSON data
                parsed_data = json.loads(data.get('data', '{}'))
                
                # Extract symbol from topic
                topic = parsed_data.get('topic', '')
                if 'BTCUSDT' in topic:
                    symbol = 'BTCUSDT'
                elif 'ETHUSDT' in topic:
                    symbol = 'ETHUSDT'
                elif 'ICPUSDT' in topic:
                    symbol = 'ICPUSDT'
                else:
                    continue
                
                record = {
                    'timestamp': int(parsed_data.get('timestamp', 0) * 1000),
                    'topic': topic,
                    'symbol': symbol,
                    'interval': parsed_data.get('interval', '1'),
                    'open_time': parsed_data.get('open_time', 0),
                    'close_time': parsed_data.get('close_time', 0),
                    'open': float(parsed_data.get('open', 0)),
                    'high': float(parsed_data.get('high', 0)),
                    'low': float(parsed_data.get('low', 0)),
                    'close': float(parsed_data.get('close', 0)),
                    'volume': float(parsed_data.get('volume', 0)),
                    'turnover': float(parsed_data.get('turnover', 0)),
                    'confirm': parsed_data.get('confirm', False)
                }
                records.append(record)
            except Exception as e:
                logger.error(f"Error parsing entry {entry_id}: {e}")
                continue
        
        if not records:
            logger.warning("No valid records found")
            return
        
        df = pd.DataFrame(records)
        logger.info(f"Converted {len(df)} records to DataFrame")
        
        # Group by symbol and insert
        for symbol in df['symbol'].unique():
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
            
            # Get existing open_times to avoid duplicates
            existing = conn.execute(f"""
                SELECT DISTINCT open_time 
                FROM {table_name}
                WHERE open_time >= {symbol_df['open_time'].min()}
            """).df()
            
            existing_times = set(existing['open_time'].tolist()) if not existing.empty else set()
            
            # Filter out duplicates
            new_df = symbol_df[~symbol_df['open_time'].isin(existing_times)]
            
            if not new_df.empty:
                # Register the DataFrame with DuckDB
                conn.register('new_data', new_df)
                
                # Insert new data
                conn.execute(f"""
                    INSERT INTO {table_name}
                    SELECT * FROM new_data
                """)
                
                logger.info(f"Inserted {len(new_df)} new records for {symbol}")
            else:
                logger.info(f"No new records for {symbol}")
        
        # Create or update the unified view
        conn.execute("""
            CREATE OR REPLACE VIEW all_klines AS
            SELECT * FROM kline_BTCUSDT
            UNION ALL
            SELECT * FROM kline_ETHUSDT  
            UNION ALL
            SELECT * FROM kline_ICPUSDT
        """)
        
        # Show final statistics
        logger.info("\nFinal database state:")
        for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
            try:
                result = conn.execute(f"""
                    SELECT 
                        COUNT(*) as count,
                        datetime(MIN(open_time/1000), 'unixepoch') as min_date,
                        datetime(MAX(open_time/1000), 'unixepoch') as max_date
                    FROM kline_{symbol}
                """).fetchone()
                
                logger.info(f"  {symbol}: {result[0]} records, {result[1]} to {result[2]}")
            except Exception as e:
                logger.warning(f"  {symbol}: Error getting stats - {e}")
        
    finally:
        conn.close()
    
    logger.info("\nRedis to DuckDB update completed!")

if __name__ == "__main__":
    update_from_redis()