#!/usr/bin/env python3
"""Check data collection status."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.duckdb_manager import DuckDBManager

def main():
    print("=== Data Collection Summary ===\n")
    
    # Check historical data
    print("Historical Data Downloaded:")
    manager = DuckDBManager(db_path="data/historical_data.duckdb")
    
    result = manager.conn.execute("""
        SELECT 
            symbol, 
            COUNT(*) as records,
            MIN(timestamp)::DATE as earliest_date,
            MAX(timestamp)::DATE as latest_date,
            ROUND(COUNT(*) / 365.0 / 1440.0, 1) as approx_years
        FROM all_klines 
        GROUP BY symbol 
        ORDER BY symbol
    """).df()
    
    print(result.to_string(index=False))
    
    total_records = result['records'].sum()
    print(f"\nTotal records: {total_records:,}")
    
    manager.conn.close()

if __name__ == "__main__":
    main()