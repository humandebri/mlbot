#!/usr/bin/env python3
"""
Fix for improved_feature_generator_redis.py to work with actual DuckDB schema
"""

fix_content = '''
    def load_historical_data(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """Load historical kline data for a symbol from DuckDB."""
        if not self.conn:
            logger.warning("No database connection available")
            return pd.DataFrame()
        
        try:
            # Use current time as end date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Query historical kline data - FIXED for actual schema
            query = f"""
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                turnover
            FROM kline_{symbol}
            WHERE open_time >= {int(start_date.timestamp() * 1000)}
            ORDER BY open_time DESC
            LIMIT 1000
            """
            
            df = self.conn.execute(query).df()
            
            if len(df) > 0:
                # Convert to proper format
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp')
                df.set_index('timestamp', inplace=True)
                
                # Calculate additional fields
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                df['hl_ratio'] = (df['high'] - df['low']) / df['close']
                df['oc_ratio'] = (df['close'] - df['open']) / df['close']
                
                logger.info(f"Loaded {len(df)} historical records for {symbol} from DuckDB")
            else:
                logger.warning(f"No historical data found for {symbol} in DuckDB")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return pd.DataFrame()
'''

print("Fix created - apply this to improved_feature_generator_redis.py")
print("\nReplace the load_historical_data method with the fixed version above")