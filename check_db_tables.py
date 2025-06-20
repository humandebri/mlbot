#!/usr/bin/env python3
"""
Check DuckDB table structure
"""

import duckdb
import pandas as pd

def check_tables():
    """Check available tables and their structure."""
    conn = duckdb.connect("data/historical_data.duckdb", read_only=True)
    
    print("Available tables:")
    tables = conn.execute("SHOW TABLES").fetchall()
    for table in tables:
        print(f"  - {table[0]}")
    
    print("\n" + "="*60)
    
    # Check each table structure
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        print("-"*40)
        
        # Get column info
        columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
        print("Columns:")
        for col in columns:
            print(f"  {col[0]:<20} {col[1]}")
        
        # Get row count
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"\nRow count: {count:,}")
        
        # Get sample data
        if count > 0:
            print("\nSample data (first 3 rows):")
            sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").df()
            print(sample)
    
    conn.close()

if __name__ == "__main__":
    check_tables()