#!/usr/bin/env python3
"""
Fix database schema issues for positions and trades tables
"""

import duckdb
import os

def fix_database_schema():
    """Fix the database schema to match what the bot expects."""
    
    db_path = "data/trading_bot.db"
    
    # Connect to database
    conn = duckdb.connect(db_path)
    
    try:
        # Check current schema
        print("Checking current schema...")
        
        # Get positions table schema
        try:
            positions_schema = conn.execute("DESCRIBE positions").df()
            print("\nCurrent positions table schema:")
            print(positions_schema)
            
            # Check if position_id column exists
            columns = positions_schema['column_name'].tolist()
            if 'position_id' not in columns:
                print("\n⚠️  position_id column missing! Need to fix schema.")
                
                # Create new positions table with correct schema
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS positions_new (
                        position_id VARCHAR PRIMARY KEY,
                        symbol VARCHAR NOT NULL,
                        side VARCHAR NOT NULL,
                        entry_price DOUBLE NOT NULL,
                        quantity DOUBLE NOT NULL,
                        stop_loss DOUBLE,
                        take_profit DOUBLE,
                        status VARCHAR DEFAULT 'open',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        closed_at TIMESTAMP,
                        exit_price DOUBLE,
                        realized_pnl DOUBLE,
                        metadata JSON
                    )
                """)
                
                # Copy data from old table if it exists
                try:
                    conn.execute("""
                        INSERT INTO positions_new (symbol, side, entry_price, quantity, stop_loss, take_profit, status, created_at, closed_at, exit_price, realized_pnl, metadata)
                        SELECT symbol, side, entry_price, quantity, stop_loss, take_profit, status, created_at, closed_at, exit_price, realized_pnl, metadata
                        FROM positions
                    """)
                    print("✅ Copied existing data to new table")
                except:
                    print("ℹ️  No data to migrate or migration failed")
                
                # Drop old table and rename new one
                conn.execute("DROP TABLE IF EXISTS positions")
                conn.execute("ALTER TABLE positions_new RENAME TO positions")
                print("✅ Fixed positions table schema")
                
        except Exception as e:
            print(f"Positions table doesn't exist or error: {e}")
            # Create positions table with correct schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    position_id VARCHAR PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    side VARCHAR NOT NULL,
                    entry_price DOUBLE NOT NULL,
                    quantity DOUBLE NOT NULL,
                    stop_loss DOUBLE,
                    take_profit DOUBLE,
                    status VARCHAR DEFAULT 'open',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP,
                    exit_price DOUBLE,
                    realized_pnl DOUBLE,
                    metadata JSON
                )
            """)
            print("✅ Created positions table with correct schema")
        
        # Check trades table
        try:
            trades_schema = conn.execute("DESCRIBE trades").df()
            print("\nCurrent trades table schema:")
            print(trades_schema)
        except:
            print("Trades table doesn't exist, creating...")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id VARCHAR PRIMARY KEY,
                    position_id VARCHAR,
                    symbol VARCHAR NOT NULL,
                    side VARCHAR NOT NULL,
                    order_type VARCHAR NOT NULL,
                    quantity DOUBLE NOT NULL,
                    price DOUBLE NOT NULL,
                    status VARCHAR DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    executed_at TIMESTAMP,
                    commission DOUBLE,
                    pnl DOUBLE,
                    metadata JSON
                )
            """)
            print("✅ Created trades table")
        
        # Verify final schema
        print("\n=== Final Schema Verification ===")
        positions_final = conn.execute("DESCRIBE positions").df()
        print("\nPositions table:")
        print(positions_final[['column_name', 'column_type']])
        
        trades_final = conn.execute("DESCRIBE trades").df()
        print("\nTrades table:")
        print(trades_final[['column_name', 'column_type']])
        
        conn.commit()
        print("\n✅ Database schema fixed successfully!")
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    fix_database_schema()