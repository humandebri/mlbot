#!/usr/bin/env python3
"""
Check historical data availability in DuckDB
"""

import duckdb
from datetime import datetime, timedelta

def check_data():
    """Check data availability for each symbol."""
    conn = duckdb.connect("data/historical_data.duckdb", read_only=True)
    
    # Check date ranges for each symbol
    symbols = ["BTCUSDT", "ETHUSDT", "ICPUSDT"]
    
    print("Data availability check:")
    print("="*60)
    
    # Check all_klines table
    print("\nall_klines table:")
    for symbol in symbols:
        query = f"""
        SELECT 
            MIN(timestamp) as start_date,
            MAX(timestamp) as end_date,
            COUNT(*) as record_count
        FROM all_klines
        WHERE symbol = '{symbol}'
        """
        result = conn.execute(query).fetchone()
        if result and result[2] > 0:
            print(f"{symbol}: {result[0]} to {result[1]} ({result[2]:,} records)")
        else:
            print(f"{symbol}: No data")
    
    # Check individual tables
    print("\nIndividual tables:")
    for symbol in symbols:
        table_name = f"klines_{symbol.lower()}"
        try:
            query = f"""
            SELECT 
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(*) as record_count
            FROM {table_name}
            """
            result = conn.execute(query).fetchone()
            if result and result[2] > 0:
                print(f"{table_name}: {result[0]} to {result[1]} ({result[2]:,} records)")
            else:
                print(f"{table_name}: No data")
        except:
            print(f"{table_name}: Table doesn't exist")
    
    # Check recent data
    print("\nRecent data check (last 30 days):")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for symbol in symbols:
        query = f"""
        SELECT COUNT(*) as count
        FROM all_klines
        WHERE symbol = '{symbol}'
            AND timestamp >= '{start_date.isoformat()}'
        """
        result = conn.execute(query).fetchone()
        print(f"{symbol}: {result[0]:,} records in last 30 days")
    
    conn.close()

if __name__ == "__main__":
    check_data()