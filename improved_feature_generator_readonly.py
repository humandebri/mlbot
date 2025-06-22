#!/usr/bin/env python3
"""
Read-only version of enhanced feature generator for bot
Prevents DuckDB lock conflicts
"""

from improved_feature_generator_enhanced import ImprovedFeatureGeneratorEnhanced
import duckdb
import logging

logger = logging.getLogger(__name__)


class ImprovedFeatureGeneratorReadOnly(ImprovedFeatureGeneratorEnhanced):
    """Read-only version to avoid DuckDB locks."""
    
    def _connect_db(self):
        """Connect to DuckDB database in read-only mode."""
        try:
            # Connect in read-only mode to avoid locks
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to historical database in READ-ONLY mode: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.conn = None
    
    def update_from_redis(self, symbol: str, max_records: int = 100) -> int:
        """Override to only update memory cache, not DuckDB."""
        if not self.redis_client:
            return 0
        
        try:
            # Get current data
            current_data = self.historical_data.get(symbol, pd.DataFrame())
            
            # Get latest entries from Redis
            stream_key = 'market_data:kline'
            entries = self.redis_client.xrevrange(stream_key, count=max_records)
            
            if not entries:
                return 0
            
            # Parse entries for this symbol (same as parent)
            new_records = []
            for entry_id, data in entries:
                try:
                    parsed = json.loads(data.get('data', '{}'))
                    
                    # Check if this is for our symbol
                    if symbol not in parsed.get('topic', ''):
                        continue
                    
                    # Extract timestamp
                    timestamp = datetime.fromtimestamp(parsed.get('timestamp', 0))
                    
                    # Skip if we already have this timestamp
                    if not current_data.empty and timestamp in current_data.index:
                        continue
                    
                    new_records.append({
                        'timestamp': timestamp,
                        'open': float(parsed.get('open', 0)),
                        'high': float(parsed.get('high', 0)),
                        'low': float(parsed.get('low', 0)),
                        'close': float(parsed.get('close', 0)),
                        'volume': float(parsed.get('volume', 0)),
                        'turnover': float(parsed.get('turnover', 0))
                    })
                    
                except Exception as e:
                    logger.debug(f"Error parsing Redis entry: {e}")
                    continue
            
            if new_records:
                # Update only in-memory cache, not DuckDB
                import pandas as pd
                new_df = pd.DataFrame(new_records)
                new_df.set_index('timestamp', inplace=True)
                new_df = new_df.sort_index()
                
                # Calculate derived fields
                new_df['returns'] = new_df['close'].pct_change().fillna(0)
                new_df['log_returns'] = np.log(new_df['close'] / new_df['close'].shift(1)).fillna(0)
                new_df['hl_ratio'] = ((new_df['high'] - new_df['low']) / new_df['close']).fillna(0.02)
                new_df['oc_ratio'] = ((new_df['close'] - new_df['open']) / new_df['close']).fillna(0)
                
                # Combine with existing data
                if not current_data.empty:
                    combined_df = pd.concat([current_data, new_df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df = combined_df.sort_index()
                    
                    # Keep only recent data
                    if len(combined_df) > self.preload_window * 2:
                        combined_df = combined_df.tail(self.preload_window * 2)
                    
                    self.historical_data[symbol] = combined_df
                else:
                    self.historical_data[symbol] = new_df
                
                logger.info(f"Added {len(new_records)} new records for {symbol} from Redis (memory only)")
                return len(new_records)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error updating from Redis for {symbol}: {e}")
            return 0


import json
import pandas as pd
import numpy as np
from datetime import datetime