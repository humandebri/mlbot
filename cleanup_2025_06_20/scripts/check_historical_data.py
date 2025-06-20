#!/usr/bin/env python3
"""Check the contents of historical_data.duckdb"""

import duckdb
import pandas as pd
from pathlib import Path

# Connect to database
db_path = "data/historical_data.duckdb"
conn = duckdb.connect(db_path, read_only=True)

print(f"Checking {db_path}...")
print("="*60)

# Get list of tables
tables = conn.execute("SHOW TABLES").fetchall()
print(f"Tables found: {[t[0] for t in tables]}")
print()

# Check each table
for table_name in [t[0] for t in tables]:
    try:
        # Get row count
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        # Get date range
        date_query = f"""
        SELECT 
            MIN(timestamp) as min_date,
            MAX(timestamp) as max_date
        FROM {table_name}
        """
        dates = conn.execute(date_query).fetchone()
        
        # Get sample data
        sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchdf()
        
        print(f"\nTable: {table_name}")
        print(f"Rows: {count:,}")
        if dates[0]:
            print(f"Date range: {dates[0]} to {dates[1]}")
        print(f"Columns: {list(sample.columns)}")
        
        # Check for symbols if column exists
        if 'symbol' in sample.columns:
            symbols = conn.execute(f"SELECT DISTINCT symbol FROM {table_name}").fetchall()
            print(f"Symbols: {[s[0] for s in symbols]}")
        
    except Exception as e:
        print(f"Error checking {table_name}: {e}")

conn.close()