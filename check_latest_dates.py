#!/usr/bin/env python3
"""
Check latest dates in each table
"""

import duckdb
from datetime import datetime

def check_latest():
    """Check latest dates for each symbol."""
    conn = duckdb.connect("data/historical_data.duckdb", read_only=True)
    
    tables = [
        ("all_klines", "BTCUSDT"),
        ("all_klines", "ETHUSDT"), 
        ("all_klines", "ICPUSDT"),
        ("klines_btcusdt", None),
        ("klines_ethusdt", None),
        ("klines_icpusdt", None)
    ]
    
    print("Latest data check:")
    print("="*60)
    
    for table, symbol in tables:
        try:
            if symbol:
                query = f"""
                SELECT MAX(timestamp) as latest, 
                       MIN(timestamp) as earliest,
                       COUNT(*) as count
                FROM {table}
                WHERE symbol = '{symbol}'
                """
            else:
                query = f"""
                SELECT MAX(timestamp) as latest,
                       MIN(timestamp) as earliest, 
                       COUNT(*) as count
                FROM {table}
                """
            
            result = conn.execute(query).fetchone()
            if result and result[2] > 0:
                print(f"\n{table}" + (f" ({symbol})" if symbol else ""))
                print(f"  Latest:   {result[0]}")
                print(f"  Earliest: {result[1]}")
                print(f"  Count:    {result[2]:,}")
                
                # Check gap from today
                if result[0]:
                    latest = datetime.fromisoformat(str(result[0]))
                    gap_days = (datetime.now() - latest).days
                    print(f"  Gap from today: {gap_days} days")
        except Exception as e:
            print(f"\n{table}: Error - {e}")
    
    # Check for recent ETHUSDT and ICPUSDT data in individual tables
    print("\n\nRecent data in individual tables (last 100 days):")
    print("-"*60)
    
    for symbol in ["ethusdt", "icpusdt"]:
        table = f"klines_{symbol}"
        query = f"""
        SELECT 
            DATE_TRUNC('month', timestamp) as month,
            COUNT(*) as count
        FROM {table}
        WHERE timestamp >= CURRENT_DATE - INTERVAL '100 days'
        GROUP BY month
        ORDER BY month DESC
        """
        
        print(f"\n{table}:")
        results = conn.execute(query).fetchall()
        for month, count in results:
            print(f"  {month}: {count:,} records")
    
    conn.close()

if __name__ == "__main__":
    check_latest()