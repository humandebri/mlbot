#!/usr/bin/env python3
"""
Persistent version of improved feature generator
Automatically saves Redis data to DuckDB for persistence
"""

import threading
import time
from datetime import datetime, timedelta
import logging
import pandas as pd
from improved_feature_generator_enhanced import ImprovedFeatureGeneratorEnhanced

logger = logging.getLogger(__name__)


class ImprovedFeatureGeneratorPersistent(ImprovedFeatureGeneratorEnhanced):
    """Enhanced feature generator with automatic DuckDB persistence."""
    
    def __init__(self, db_path: str = "data/historical_data.duckdb", 
                 enable_redis: bool = True,
                 enable_persistence: bool = True,
                 persistence_interval: int = 1800):  # 30 minutes
        super().__init__(db_path, enable_redis)
        
        self.enable_persistence = enable_persistence
        self.persistence_interval = persistence_interval
        self.last_persistence_time = {}
        self.persistence_thread = None
        self.persistence_running = False
        
        # Buffer for new records to persist
        self.persistence_buffer = {}
        self.buffer_lock = threading.Lock()
        
        if self.enable_persistence:
            self._start_persistence_thread()
    
    def _start_persistence_thread(self):
        """Start background thread for persistence."""
        self.persistence_running = True
        self.persistence_thread = threading.Thread(
            target=self._persistence_loop,
            daemon=True
        )
        self.persistence_thread.start()
        logger.info("Started persistence thread")
    
    def _persistence_loop(self):
        """Background loop to persist data to DuckDB."""
        while self.persistence_running:
            try:
                # Check each symbol
                for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
                    self._persist_symbol_data(symbol)
                
                # Sleep for 1 minute between checks
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in persistence loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _persist_symbol_data(self, symbol: str):
        """Persist buffered data for a symbol to DuckDB."""
        now = datetime.utcnow()
        last_persist = self.last_persistence_time.get(symbol, datetime.min)
        
        # Check if it's time to persist
        if (now - last_persist).total_seconds() < self.persistence_interval:
            return
        
        with self.buffer_lock:
            buffer_data = self.persistence_buffer.get(symbol, [])
            if not buffer_data:
                return
            
            # Clear buffer
            self.persistence_buffer[symbol] = []
        
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(buffer_data)
            if df.empty:
                return
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['open_time'], keep='last')
            df = df.sort_values('open_time')
            
            # Connect to DuckDB with write access
            import duckdb
            write_conn = duckdb.connect(self.db_path)
            
            try:
                # Register DataFrame
                write_conn.register('new_data', df)
                
                # Insert into all_klines
                write_conn.execute("""
                    INSERT INTO all_klines (symbol, open_time, open, high, low, close, volume, turnover)
                    SELECT symbol, open_time, open, high, low, close, volume, turnover
                    FROM new_data
                    WHERE NOT EXISTS (
                        SELECT 1 FROM all_klines 
                        WHERE all_klines.symbol = new_data.symbol 
                        AND all_klines.open_time = new_data.open_time
                    )
                """)
                
                count = write_conn.execute(
                    f"SELECT COUNT(*) FROM all_klines WHERE symbol = '{symbol}'"
                ).fetchone()[0]
                
                logger.info(f"Persisted {len(df)} records for {symbol}, total: {count}")
                self.last_persistence_time[symbol] = now
                
            finally:
                write_conn.close()
                
        except Exception as e:
            logger.error(f"Error persisting {symbol} data: {e}")
    
    def update_from_redis(self, symbol: str, max_records: int = 100) -> int:
        """Override to also buffer data for persistence."""
        # Call parent method
        count = super().update_from_redis(symbol, max_records)
        
        if count > 0 and self.enable_persistence:
            # Add new records to persistence buffer
            with self.buffer_lock:
                if symbol not in self.persistence_buffer:
                    self.persistence_buffer[symbol] = []
                
                # Get the latest data from cache
                hist_data = self.historical_data.get(symbol)
                if hist_data is not None and not hist_data.empty:
                    # Get last N records
                    recent_data = hist_data.tail(count)
                    
                    # Convert to buffer format
                    for idx, row in recent_data.iterrows():
                        self.persistence_buffer[symbol].append({
                            'symbol': symbol,
                            'open_time': int(idx.timestamp() * 1000),
                            'open': row['open'],
                            'high': row['high'],
                            'low': row['low'],
                            'close': row['close'],
                            'volume': row['volume'],
                            'turnover': row['turnover']
                        })
                    
                    # Keep buffer size reasonable (max 1000 per symbol)
                    if len(self.persistence_buffer[symbol]) > 1000:
                        self.persistence_buffer[symbol] = self.persistence_buffer[symbol][-1000:]
        
        return count
    
    def close(self):
        """Close connections and stop persistence thread."""
        # Stop persistence thread
        if self.persistence_thread:
            logger.info("Stopping persistence thread...")
            self.persistence_running = False
            
            # Final persistence
            for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
                self._persist_symbol_data(symbol)
        
        # Call parent close
        super().close()


# Convenience function to update bot imports
def update_bot_to_persistent():
    """Update the bot to use persistent feature generator."""
    import fileinput
    import sys
    
    bot_file = 'simple_improved_bot_with_trading_fixed.py'
    
    # Replace import
    for line in fileinput.input(bot_file, inplace=True):
        if 'from improved_feature_generator import' in line:
            print('from improved_feature_generator_persistent import ImprovedFeatureGeneratorPersistent as ImprovedFeatureGenerator')
        else:
            print(line, end='')
    
    print(f"Updated {bot_file} to use persistent feature generator")


if __name__ == "__main__":
    # Test persistence
    logger.info("Testing persistent feature generator...")
    
    gen = ImprovedFeatureGeneratorPersistent(
        enable_persistence=True,
        persistence_interval=60  # 1 minute for testing
    )
    
    try:
        # Load initial data
        for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
            gen.update_historical_cache(symbol)
            logger.info(f"{symbol}: {len(gen.historical_data.get(symbol, []))} records loaded")
        
        # Wait for persistence
        logger.info("Waiting for automatic persistence...")
        time.sleep(120)
        
    finally:
        gen.close()