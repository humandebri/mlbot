#!/usr/bin/env python3
"""
Improved feature generator with Redis integration and real-time updates
Fixes the static data issue by continuously updating from Redis Streams
"""

import numpy as np
import pandas as pd
import duckdb
import redis
import json
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import deque
import logging
import asyncio
import time

logger = logging.getLogger(__name__)


class ImprovedFeatureGeneratorRedis:
    """Generate 44 features using both historical DuckDB data and real-time Redis data."""
    
    def __init__(self, db_path: str = "data/historical_data.duckdb", redis_host: str = "localhost", redis_port: int = 6379):
        # Feature names expected by the model
        self.feature_names = [
            "returns", "log_returns", "hl_ratio", "oc_ratio", 
            "return_1", "return_3", "return_5", "return_10", "return_20",
            "vol_5", "vol_10", "vol_20", "vol_30",
            "vol_ratio_10", "vol_ratio_20",
            "price_vs_sma_5", "price_vs_sma_10", "price_vs_sma_20", "price_vs_sma_30",
            "price_vs_ema_5", "price_vs_ema_12",
            "macd", "macd_hist",
            "rsi_14", "rsi_21",
            "bb_position_20", "bb_width_20",
            "volume_ratio_10", "volume_ratio_20",
            "log_volume",
            "volume_price_trend",
            "momentum_3", "momentum_5", "momentum_10",
            "price_percentile_20", "price_percentile_50",
            "trend_strength_short", "trend_strength_long",
            "high_vol_regime", "low_vol_regime", "trending_market",
            "hour_sin", "hour_cos", "is_weekend"
        ]
        
        # Load manual scaler (same as original)
        self.scaler_means = np.array([
            0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.015, 0.018, 0.02, 0.022, 1.1, 1.15, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            0.0, 0.0, 50.0, 50.0, 0.0, 0.04, 1.0, 1.0, 10.0, 
            0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.1, 0.08, 0.2, 0.8, 0.3, 
            0.0, 0.0, 0.28
        ])
        
        self.scaler_stds = np.array([
            0.01, 0.01, 0.01, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 
            0.01, 0.012, 0.015, 0.018, 0.2, 0.25, 0.02, 0.03, 0.04, 0.05, 0.02, 0.03, 
            0.1, 0.05, 15.0, 15.0, 1.0, 0.02, 0.5, 0.5, 2.0, 
            0.1, 0.02, 0.025, 0.03, 0.3, 0.3, 0.1, 0.08, 0.4, 0.4, 0.45, 
            0.7, 0.7, 0.45
        ])
        
        # Historical data cache for each symbol
        self.historical_data = {}
        self.db_path = db_path
        self.conn = None
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Real-time data buffer (keeps last 100 candles per symbol)
        self.realtime_buffer = {
            "BTCUSDT": deque(maxlen=100),
            "ETHUSDT": deque(maxlen=100), 
            "ICPUSDT": deque(maxlen=100)
        }
        
        # Last update time for each symbol
        self.last_update_time = {}
        
        # Initialize connections
        self._connect_db()
        self._connect_redis()
        
        # Load initial historical data
        self.window_size = 100  # Keep last 100 candles for calculations
        self.update_interval = 60  # Update from Redis every 60 seconds
        
    def _connect_db(self):
        """Connect to DuckDB database."""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to historical database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.conn = None
    
    def _connect_redis(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host, 
                port=self.redis_port, 
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def load_historical_data(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """Load historical kline data for a symbol from DuckDB."""
        if not self.conn:
            logger.warning("No database connection available")
            return pd.DataFrame()
        
        try:
            # Use current time as end date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Query historical kline data
            query = f"""
            SELECT 
                CASE 
                    WHEN timestamp_ms IS NOT NULL THEN timestamp_ms
                    ELSE CAST(timestamp * 1000 AS BIGINT)
                END as timestamp,
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
    
    def update_from_redis(self, symbol: str) -> int:
        """Update realtime buffer with latest data from Redis Streams."""
        if not self.redis_client:
            return 0
        
        try:
            # Get latest entries from Redis
            stream_key = 'market_data:kline'
            
            # Get last ID we processed (or start from beginning)
            last_id = self.last_update_time.get(symbol, '0')
            
            # Read new entries
            entries = self.redis_client.xread(
                {stream_key: last_id}, 
                count=1000, 
                block=0
            )
            
            if not entries:
                return 0
            
            new_records = 0
            for stream_name, messages in entries:
                for message_id, data in messages:
                    try:
                        parsed = json.loads(data.get('data', '{}'))
                        
                        # Check if this is for our symbol
                        topic = parsed.get('topic', '')
                        if symbol not in topic:
                            continue
                        
                        # Extract kline data
                        kline = {
                            'timestamp': datetime.fromtimestamp(parsed.get('timestamp', 0)),
                            'open': float(parsed.get('open', 0)),
                            'high': float(parsed.get('high', 0)),
                            'low': float(parsed.get('low', 0)),
                            'close': float(parsed.get('close', 0)),
                            'volume': float(parsed.get('volume', 0)),
                            'turnover': float(parsed.get('turnover', 0))
                        }
                        
                        # Add to buffer
                        self.realtime_buffer[symbol].append(kline)
                        new_records += 1
                        
                        # Update last processed ID
                        self.last_update_time[symbol] = message_id
                        
                    except Exception as e:
                        logger.debug(f"Error parsing Redis entry: {e}")
                        continue
            
            if new_records > 0:
                logger.info(f"Added {new_records} new records for {symbol} from Redis")
            
            return new_records
            
        except Exception as e:
            logger.error(f"Error updating from Redis for {symbol}: {e}")
            return 0
    
    def get_combined_data(self, symbol: str) -> pd.DataFrame:
        """Get combined historical and realtime data for feature calculation."""
        # Start with historical data
        hist_data = self.historical_data.get(symbol)
        
        if hist_data is None or len(hist_data) == 0:
            # Load from database if not cached
            hist_data = self.load_historical_data(symbol)
            if not hist_data.empty:
                self.historical_data[symbol] = hist_data
        
        # Convert realtime buffer to DataFrame
        if self.realtime_buffer[symbol]:
            realtime_df = pd.DataFrame(list(self.realtime_buffer[symbol]))
            if not realtime_df.empty:
                realtime_df.set_index('timestamp', inplace=True)
                
                # Calculate derived fields
                realtime_df['returns'] = realtime_df['close'].pct_change()
                realtime_df['log_returns'] = np.log(realtime_df['close'] / realtime_df['close'].shift(1))
                realtime_df['hl_ratio'] = (realtime_df['high'] - realtime_df['low']) / realtime_df['close']
                realtime_df['oc_ratio'] = (realtime_df['close'] - realtime_df['open']) / realtime_df['close']
                
                # Combine with historical data
                if hist_data is not None and not hist_data.empty:
                    # Remove overlapping timestamps
                    latest_hist_time = hist_data.index[-1]
                    realtime_df = realtime_df[realtime_df.index > latest_hist_time]
                    
                    if not realtime_df.empty:
                        combined_df = pd.concat([hist_data, realtime_df])
                        combined_df = combined_df.sort_index()
                        
                        # Keep only recent data for efficiency
                        if len(combined_df) > self.window_size * 2:
                            combined_df = combined_df.tail(self.window_size * 2)
                        
                        return combined_df
                
                return hist_data if hist_data is not None else pd.DataFrame()
        
        return hist_data if hist_data is not None else pd.DataFrame()
    
    def update_historical_cache(self, symbol: str):
        """Update historical data cache with latest data from both DuckDB and Redis."""
        # Load fresh data from DuckDB
        hist_data = self.load_historical_data(symbol)
        if not hist_data.empty:
            self.historical_data[symbol] = hist_data
        
        # Update from Redis
        self.update_from_redis(symbol)
        
        # Log current data status
        combined_data = self.get_combined_data(symbol)
        if not combined_data.empty:
            logger.info(
                f"{symbol} data updated: {len(combined_data)} records, "
                f"latest: {combined_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')}"
            )
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate various moving averages."""
        ma_dict = {}
        
        # Simple Moving Averages
        for period in [5, 10, 20, 30]:
            ma_dict[f'sma_{period}'] = prices.rolling(window=period, min_periods=1).mean()
        
        # Exponential Moving Averages
        for period in [5, 12]:
            ma_dict[f'ema_{period}'] = prices.ewm(span=period, adjust=False).mean()
        
        return ma_dict
    
    def calculate_volatility(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate volatility over different periods."""
        vol_dict = {}
        
        for period in [5, 10, 20, 30]:
            if len(returns) >= period:
                vol_dict[f'vol_{period}'] = returns.rolling(window=period).std().iloc[-1]
            else:
                vol_dict[f'vol_{period}'] = returns.std() if len(returns) > 1 else 0.015
        
        return vol_dict
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Calculate RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, num_std: float = 2) -> Tuple[float, float]:
        """Calculate Bollinger Bands position and width."""
        if len(prices) < period:
            return 0.0, 0.04
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        current_price = prices.iloc[-1]
        current_sma = sma.iloc[-1]
        band_width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / current_sma
        
        if upper_band.iloc[-1] != lower_band.iloc[-1]:
            position = 2 * (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1]) - 1
        else:
            position = 0.0
        
        return position, band_width
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[float, float]:
        """Calculate MACD and MACD histogram."""
        if len(prices) < 26:
            return 0.0, 0.0
        
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        return macd_line.iloc[-1], macd_hist.iloc[-1]
    
    def generate_features(self, ticker_data: Dict[str, float], symbol: str = None) -> Dict[str, float]:
        """Generate all 44 features from ticker data and up-to-date historical data."""
        # Update data from Redis before generating features
        self.update_from_redis(symbol)
        
        price = float(ticker_data.get("lastPrice", 0))
        volume = float(ticker_data.get("volume24h", 0))
        high = float(ticker_data.get("highPrice24h", price * 1.001))
        low = float(ticker_data.get("lowPrice24h", price * 0.999))
        prev_close = float(ticker_data.get("prevPrice24h", price))
        
        features = {}
        
        # Basic returns
        returns = (price - prev_close) / prev_close if prev_close > 0 else 0
        features["returns"] = returns
        features["log_returns"] = np.log(price / prev_close) if prev_close > 0 else 0
        
        # Price ratios
        features["hl_ratio"] = (high - low) / price if price > 0 else 0.02
        features["oc_ratio"] = (price - prev_close) / price if price > 0 else 0
        
        # Get combined historical and realtime data
        combined_data = self.get_combined_data(symbol)
        
        if not combined_data.empty and len(combined_data) > 1:
            # Use real combined data for calculations
            prices = combined_data['close']
            returns_series = combined_data['returns']
            volumes = combined_data['volume']
            
            # Multi-period returns
            for period in [1, 3, 5, 10, 20]:
                if len(prices) > period:
                    features[f"return_{period}"] = (prices.iloc[-1] - prices.iloc[-period-1]) / prices.iloc[-period-1]
                else:
                    features[f"return_{period}"] = returns * (period / 20)
            
            # Volatility
            vol_dict = self.calculate_volatility(returns_series)
            for key, value in vol_dict.items():
                features[key] = value
            
            # Volatility ratios
            if 'vol_10' in vol_dict and 'vol_20' in vol_dict:
                features["vol_ratio_10"] = vol_dict['vol_10'] / vol_dict['vol_20'] if vol_dict['vol_20'] > 0 else 1.1
                features["vol_ratio_20"] = vol_dict['vol_20'] / vol_dict['vol_30'] if 'vol_30' in vol_dict and vol_dict['vol_30'] > 0 else 1.15
            else:
                features["vol_ratio_10"] = 1.1
                features["vol_ratio_20"] = 1.15
            
            # Moving averages
            ma_dict = self.calculate_moving_averages(prices)
            for period in [5, 10, 20, 30]:
                if f'sma_{period}' in ma_dict:
                    features[f"price_vs_sma_{period}"] = price / ma_dict[f'sma_{period}'].iloc[-1]
                else:
                    features[f"price_vs_sma_{period}"] = 1.0
            
            for period in [5, 12]:
                if f'ema_{period}' in ma_dict:
                    features[f"price_vs_ema_{period}"] = price / ma_dict[f'ema_{period}'].iloc[-1]
                else:
                    features[f"price_vs_ema_{period}"] = 1.0
            
            # MACD
            macd, macd_hist = self.calculate_macd(prices)
            features["macd"] = macd
            features["macd_hist"] = macd_hist
            
            # RSI
            features["rsi_14"] = self.calculate_rsi(prices, 14)
            features["rsi_21"] = self.calculate_rsi(prices, 21)
            
            # Bollinger Bands
            bb_position, bb_width = self.calculate_bollinger_bands(prices)
            features["bb_position_20"] = bb_position
            features["bb_width_20"] = bb_width
            
            # Volume ratios
            if len(volumes) >= 20:
                vol_ma_10 = volumes.rolling(window=10).mean().iloc[-1]
                vol_ma_20 = volumes.rolling(window=20).mean().iloc[-1]
                features["volume_ratio_10"] = volume / vol_ma_10 if vol_ma_10 > 0 else 1.0
                features["volume_ratio_20"] = volume / vol_ma_20 if vol_ma_20 > 0 else 1.0
            else:
                features["volume_ratio_10"] = 1.0
                features["volume_ratio_20"] = 1.0
            
            # Volume-price trend
            if len(prices) > 1 and len(volumes) > 1:
                price_change = prices.pct_change()
                vpt = (volumes * price_change).cumsum()
                features["volume_price_trend"] = vpt.iloc[-1] / 1000000 if not np.isnan(vpt.iloc[-1]) else 0
            else:
                features["volume_price_trend"] = 0
            
            # Momentum
            for period in [3, 5, 10]:
                if len(prices) > period:
                    features[f"momentum_{period}"] = (prices.iloc[-1] - prices.iloc[-period]) / prices.iloc[-period]
                else:
                    features[f"momentum_{period}"] = returns * (period / 10)
            
            # Price percentiles
            if len(prices) >= 50:
                features["price_percentile_20"] = (prices.tail(20) < price).sum() / 20
                features["price_percentile_50"] = (prices.tail(50) < price).sum() / 50
            else:
                features["price_percentile_20"] = 0.5 + returns * 2
                features["price_percentile_50"] = 0.5 + returns
            
            # Trend strength
            if len(returns_series) >= 10:
                features["trend_strength_short"] = abs(returns_series.tail(5).mean()) * 100
                features["trend_strength_long"] = abs(returns_series.tail(20).mean()) * 50 if len(returns_series) >= 20 else abs(returns_series.mean()) * 50
            else:
                features["trend_strength_short"] = abs(returns) * 5
                features["trend_strength_long"] = abs(returns) * 3
            
            # Market regimes
            recent_vol = returns_series.tail(20).std() if len(returns_series) >= 20 else abs(returns)
            features["high_vol_regime"] = 1.0 if recent_vol > 0.02 else 0.0
            features["low_vol_regime"] = 1.0 if recent_vol < 0.01 else 0.0
            features["trending_market"] = 1.0 if abs(returns_series.tail(10).mean()) > 0.001 else 0.0 if len(returns_series) >= 10 else 0.0
            
            # Log data freshness
            latest_time = combined_data.index[-1]
            age_minutes = (datetime.now() - latest_time).total_seconds() / 60
            if age_minutes > 5:
                logger.warning(f"{symbol} data is {age_minutes:.1f} minutes old")
            
        else:
            # Fallback to approximations if no data
            logger.warning(f"No combined data for {symbol}, using approximations")
            
            # Use original approximations (same as before)
            features["return_1"] = returns
            features["return_3"] = returns * 0.8
            features["return_5"] = returns * 0.6
            features["return_10"] = returns * 0.4
            features["return_20"] = returns * 0.2
            
            vol_proxy = abs(returns) * 2
            features["vol_5"] = vol_proxy * 0.8
            features["vol_10"] = vol_proxy * 0.9
            features["vol_20"] = vol_proxy
            features["vol_30"] = vol_proxy * 1.1
            
            features["vol_ratio_10"] = 1.1
            features["vol_ratio_20"] = 1.15
            
            for period in [5, 10, 20, 30]:
                features[f"price_vs_sma_{period}"] = 1.0 + returns * (0.3 - period * 0.01)
            
            features["price_vs_ema_5"] = 1.0 + returns * 0.25
            features["price_vs_ema_12"] = 1.0 + returns * 0.12
            
            features["macd"] = returns * 0.5
            features["macd_hist"] = returns * 0.3
            
            rsi_base = 50 + (returns * 500)
            features["rsi_14"] = max(10, min(90, rsi_base))
            features["rsi_21"] = max(15, min(85, rsi_base * 0.9))
            
            features["bb_position_20"] = returns * 2
            features["bb_width_20"] = 0.04 + abs(returns) * 0.5
            
            features["volume_ratio_10"] = 1.0
            features["volume_ratio_20"] = 1.0
            
            features["volume_price_trend"] = returns * volume / 1000000
            
            features["momentum_3"] = returns * 0.8
            features["momentum_5"] = returns * 0.6
            features["momentum_10"] = returns * 0.4
            
            features["price_percentile_20"] = 0.5 + returns * 2
            features["price_percentile_50"] = 0.5 + returns
            
            features["trend_strength_short"] = abs(returns) * 5
            features["trend_strength_long"] = abs(returns) * 3
            
            features["high_vol_regime"] = 1.0 if abs(returns) > 0.01 else 0.0
            features["low_vol_regime"] = 1.0 if abs(returns) < 0.005 else 0.0
            features["trending_market"] = 1.0 if abs(returns) > 0.008 else 0.0
        
        # Log volume (always calculated)
        features["log_volume"] = np.log(volume) if volume > 0 else 10.0
        
        # Time features (always calculated)
        now = datetime.now()
        hour_angle = (now.hour / 24) * 2 * np.pi
        features["hour_sin"] = np.sin(hour_angle)
        features["hour_cos"] = np.cos(hour_angle)
        features["is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0
        
        return features
    
    def normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        """Normalize features using the manual scaler."""
        # Convert to array in correct order
        feature_array = np.array([features[name] for name in self.feature_names])
        
        # Normalize
        normalized = (feature_array - self.scaler_means) / (self.scaler_stds + 1e-8)
        normalized = np.clip(normalized, -5, 5)
        
        return normalized.astype(np.float32)
    
    async def periodic_update_loop(self, interval: int = 60):
        """Periodically update data from Redis."""
        while True:
            try:
                for symbol in ["BTCUSDT", "ETHUSDT", "ICPUSDT"]:
                    records = self.update_from_redis(symbol)
                    if records > 0:
                        logger.info(f"Updated {symbol} with {records} new records from Redis")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in periodic update: {e}")
                await asyncio.sleep(interval)
    
    def close(self):
        """Close database and Redis connections."""
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.redis_client:
            self.redis_client.close()
            self.redis_client = None


# Test function
if __name__ == "__main__":
    import asyncio
    from src.common.bybit_client import BybitRESTClient
    
    async def test():
        # Initialize generator
        generator = ImprovedFeatureGeneratorRedis()
        
        # Test symbols
        symbols = ["BTCUSDT", "ETHUSDT", "ICPUSDT"]
        
        # Load initial data
        for symbol in symbols:
            generator.update_historical_cache(symbol)
        
        # Initialize Bybit client
        client = BybitRESTClient(testnet=False)
        await client.__aenter__()
        
        try:
            # Test feature generation
            for symbol in symbols:
                print(f"\n{'='*60}")
                print(f"Testing {symbol}")
                print('='*60)
                
                # Get current ticker
                ticker = await client.get_ticker(symbol)
                
                # Generate features
                features = generator.generate_features(ticker, symbol)
                
                print(f"\nGenerated features (sample):")
                for key in ["returns", "vol_20", "rsi_14", "macd", "volume_ratio_10"]:
                    print(f"  {key}: {features.get(key, 'N/A'):.6f}")
                
                # Check data freshness
                combined_data = generator.get_combined_data(symbol)
                if not combined_data.empty:
                    latest_time = combined_data.index[-1]
                    age_minutes = (datetime.now() - latest_time).total_seconds() / 60
                    print(f"\nData freshness: {age_minutes:.1f} minutes old")
                    print(f"Total records: {len(combined_data)}")
                
                # Normalize
                normalized = generator.normalize_features(features)
                print(f"\nNormalized shape: {normalized.shape}")
                print(f"Stats - Min: {normalized.min():.3f}, Max: {normalized.max():.3f}, Mean: {normalized.mean():.3f}")
        
        finally:
            await client.__aexit__(None, None, None)
            generator.close()
    
    asyncio.run(test())