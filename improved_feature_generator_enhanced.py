#!/usr/bin/env python3
"""
Enhanced version of improved feature generator
Combines high confidence from original with real-time data capabilities
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
import threading

logger = logging.getLogger(__name__)


class ImprovedFeatureGeneratorEnhanced:
    """Generate 44 features using real historical data with real-time updates."""
    
    def __init__(self, db_path: str = "data/historical_data.duckdb", enable_redis: bool = True):
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
        self.enable_redis = enable_redis
        
        # Cache management
        self.cache_lock = threading.Lock()
        self.last_cache_update = {}
        self.cache_update_interval = 300  # 5 minutes
        
        # Initialize connections
        self._connect_db()
        if self.enable_redis:
            self._connect_redis()
        
        # Preload recent historical data
        self.preload_window = 100  # Keep last 100 candles for calculations
    
    def _connect_db(self):
        """Connect to DuckDB database."""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to historical database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.conn = None
    
    def _connect_redis(self):
        """Connect to Redis for real-time updates."""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis for real-time updates")
        except Exception as e:
            logger.warning(f"Redis connection failed, will use database only: {e}")
            self.redis_client = None
    
    def _get_latest_db_timestamp(self, symbol: str) -> datetime:
        """Get the latest timestamp from database."""
        if not self.conn:
            return datetime.utcnow()
        
        try:
            # Try different table patterns
            for table_pattern in [f"kline_{symbol}", f"klines_{symbol.lower()}", "all_klines"]:
                try:
                    if table_pattern == "all_klines":
                        query = f"SELECT MAX(open_time) as max_ts FROM {table_pattern} WHERE symbol = '{symbol}'"
                    else:
                        query = f"SELECT MAX(open_time) as max_ts FROM {table_pattern}"
                    
                    result = self.conn.execute(query).fetchone()
                    if result and result[0]:
                        # Convert milliseconds to datetime
                        if result[0] > 1e12:  # Milliseconds
                            return datetime.fromtimestamp(result[0] / 1000)
                        else:  # Seconds
                            return datetime.fromtimestamp(result[0])
                except:
                    continue
            
            # If no data found, use current time minus 30 days
            return datetime.utcnow() - timedelta(days=30)
            
        except Exception as e:
            logger.error(f"Error getting latest timestamp: {e}")
            return datetime.utcnow() - timedelta(days=30)
    
    def load_historical_data(self, symbol: str, lookback_days: int = 60) -> pd.DataFrame:
        """Load historical kline data for a symbol with dynamic end date."""
        if not self.conn:
            logger.warning("No database connection available")
            return pd.DataFrame()
        
        try:
            # Get latest available timestamp from database
            end_date = self._get_latest_db_timestamp(symbol)
            start_date = end_date - timedelta(days=lookback_days)
            
            # Convert to milliseconds for query
            start_ms = int(start_date.timestamp() * 1000)
            end_ms = int(end_date.timestamp() * 1000)
            
            # Try different table patterns
            for table_pattern in [f"kline_{symbol}", f"klines_{symbol.lower()}", "all_klines"]:
                try:
                    if table_pattern == "all_klines":
                        query = f"""
                        SELECT 
                            CASE 
                                WHEN open_time > 1e12 THEN open_time / 1000
                                ELSE open_time 
                            END as timestamp_sec,
                            open,
                            high,
                            low,
                            close,
                            volume,
                            COALESCE(turnover, volume * close) as turnover
                        FROM {table_pattern}
                        WHERE symbol = '{symbol}'
                            AND open_time >= {start_ms}
                            AND open_time <= {end_ms}
                        ORDER BY open_time ASC
                        """
                    else:
                        query = f"""
                        SELECT 
                            CASE 
                                WHEN open_time > 1e12 THEN open_time / 1000
                                ELSE open_time 
                            END as timestamp_sec,
                            open,
                            high,
                            low,
                            close,
                            volume,
                            COALESCE(turnover, volume * close) as turnover
                        FROM {table_pattern}
                        WHERE open_time >= {start_ms}
                            AND open_time <= {end_ms}
                        ORDER BY open_time ASC
                        """
                    
                    df = self.conn.execute(query).df()
                    
                    if len(df) > 0:
                        # Convert timestamp to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp_sec'], unit='s')
                        df = df.drop('timestamp_sec', axis=1)
                        df.set_index('timestamp', inplace=True)
                        
                        # Remove duplicates
                        df = df[~df.index.duplicated(keep='last')]
                        
                        # Calculate additional fields
                        df['returns'] = df['close'].pct_change().fillna(0)
                        df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
                        df['hl_ratio'] = ((df['high'] - df['low']) / df['close']).fillna(0.02)
                        df['oc_ratio'] = ((df['close'] - df['open']) / df['close']).fillna(0)
                        
                        logger.info(f"Loaded {len(df)} records for {symbol} from {table_pattern}, latest: {df.index[-1]}")
                        return df
                        
                except Exception as e:
                    logger.debug(f"Failed to query {table_pattern}: {e}")
                    continue
            
            logger.warning(f"No historical data found for {symbol} in any table")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def update_from_redis(self, symbol: str, max_records: int = 100) -> int:
        """Update historical data with latest records from Redis."""
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
            
            # Parse entries for this symbol
            new_records = []
            for entry_id, data in entries:
                try:
                    parsed = json.loads(data.get('data', '{}'))
                    
                    # Check if this is for our symbol
                    if symbol not in parsed.get('topic', ''):
                        continue
                    
                    # Extract kline data
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
                # Create DataFrame from new records
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
                
                logger.info(f"Added {len(new_records)} new records for {symbol} from Redis")
                return len(new_records)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error updating from Redis for {symbol}: {e}")
            return 0
    
    def update_historical_cache(self, symbol: str, force: bool = False):
        """Update historical data cache for a symbol."""
        with self.cache_lock:
            now = datetime.utcnow()
            last_update = self.last_cache_update.get(symbol)
            
            # Check if update is needed
            if not force and last_update and (now - last_update).total_seconds() < self.cache_update_interval:
                return
            
            # Load from database
            df = self.load_historical_data(symbol)
            if not df.empty:
                self.historical_data[symbol] = df
                self.last_cache_update[symbol] = now
                
                # Try to update from Redis if enabled
                if self.enable_redis and self.redis_client:
                    self.update_from_redis(symbol)
    
    def append_current_price(self, symbol: str, ticker_data: Dict[str, float]) -> pd.DataFrame:
        """Append current price to historical data for accurate calculations."""
        hist_data = self.historical_data.get(symbol, pd.DataFrame()).copy()
        
        if hist_data.empty:
            return hist_data
        
        # Create current candle
        current_time = datetime.utcnow()
        price = float(ticker_data.get("lastPrice", 0))
        
        # Check if we should add this as a new candle
        if len(hist_data) > 0:
            last_time = hist_data.index[-1]
            time_diff = (current_time - last_time).total_seconds()
            
            # Only add if more than 30 seconds have passed
            if time_diff > 30:
                new_row = pd.DataFrame({
                    'open': [price],
                    'high': [float(ticker_data.get("highPrice24h", price))],
                    'low': [float(ticker_data.get("lowPrice24h", price))],
                    'close': [price],
                    'volume': [float(ticker_data.get("volume24h", 0))],
                    'turnover': [float(ticker_data.get("turnover24h", 0))],
                }, index=[current_time])
                
                # Append to historical data
                hist_data = pd.concat([hist_data, new_row])
                
                # Recalculate derived fields for the new row
                if len(hist_data) > 1:
                    hist_data.loc[current_time, 'returns'] = (price - hist_data['close'].iloc[-2]) / hist_data['close'].iloc[-2]
                    hist_data.loc[current_time, 'log_returns'] = np.log(price / hist_data['close'].iloc[-2])
                else:
                    hist_data.loc[current_time, 'returns'] = 0
                    hist_data.loc[current_time, 'log_returns'] = 0
                
                hist_data.loc[current_time, 'hl_ratio'] = (new_row['high'].iloc[0] - new_row['low'].iloc[0]) / price if price > 0 else 0.02
                hist_data.loc[current_time, 'oc_ratio'] = (price - new_row['open'].iloc[0]) / price if price > 0 else 0
        
        return hist_data
    
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
        
        # Clean returns
        returns_clean = returns.dropna()
        
        for period in [5, 10, 20, 30]:
            if len(returns_clean) >= period:
                vol = returns_clean.rolling(window=period).std().iloc[-1]
                vol_dict[f'vol_{period}'] = vol if not np.isnan(vol) else 0.015
            else:
                # Use all available data if less than period
                vol_dict[f'vol_{period}'] = returns_clean.std() if len(returns_clean) > 1 else 0.015
        
        return vol_dict
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Calculate RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        result = rsi.iloc[-1]
        return result if not np.isnan(result) else 50.0
    
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
        
        if current_sma > 0:
            band_width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / current_sma
        else:
            band_width = 0.04
        
        # Position within bands (-1 to 1)
        band_range = upper_band.iloc[-1] - lower_band.iloc[-1]
        if band_range > 0:
            position = 2 * (current_price - lower_band.iloc[-1]) / band_range - 1
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
        
        macd_val = macd_line.iloc[-1]
        hist_val = macd_hist.iloc[-1]
        
        return (macd_val if not np.isnan(macd_val) else 0.0, 
                hist_val if not np.isnan(hist_val) else 0.0)
    
    def generate_features(self, ticker_data: Dict[str, float], symbol: str = None) -> Dict[str, float]:
        """Generate all 44 features from ticker data and historical data."""
        # Update cache if needed
        self.update_historical_cache(symbol)
        
        price = float(ticker_data.get("lastPrice", 0))
        volume = float(ticker_data.get("volume24h", 0))
        high = float(ticker_data.get("highPrice24h", price * 1.001))
        low = float(ticker_data.get("lowPrice24h", price * 0.999))
        prev_close = float(ticker_data.get("prevPrice24h", price))
        
        features = {}
        
        # Basic returns
        returns = (price - prev_close) / prev_close if prev_close > 0 else 0
        features["returns"] = returns
        features["log_returns"] = np.log(price / prev_close) if prev_close > 0 and price > 0 else 0
        
        # Price ratios
        features["hl_ratio"] = (high - low) / price if price > 0 else 0.02
        features["oc_ratio"] = (price - prev_close) / price if price > 0 else 0
        
        # Get historical data with current price appended
        with self.cache_lock:
            hist_data = self.append_current_price(symbol, ticker_data)
        
        if not hist_data.empty and len(hist_data) > 1:
            # Use real historical data for calculations
            prices = hist_data['close']
            returns_series = hist_data['returns']
            volumes = hist_data['volume']
            
            # Multi-period returns using current price
            for period in [1, 3, 5, 10, 20]:
                if len(prices) > period:
                    features[f"return_{period}"] = (price - prices.iloc[-period-1]) / prices.iloc[-period-1]
                else:
                    features[f"return_{period}"] = returns * (period / 20)  # Scale by period
            
            # Volatility (actual calculations)
            vol_dict = self.calculate_volatility(returns_series)
            for key, value in vol_dict.items():
                features[key] = value
            
            # Volatility ratios with safety checks
            features["vol_ratio_10"] = 1.1  # Default
            features["vol_ratio_20"] = 1.15  # Default
            
            if 'vol_10' in vol_dict and 'vol_20' in vol_dict and vol_dict['vol_20'] > 0:
                features["vol_ratio_10"] = vol_dict['vol_10'] / vol_dict['vol_20']
            
            if 'vol_20' in vol_dict and 'vol_30' in vol_dict and vol_dict['vol_30'] > 0:
                features["vol_ratio_20"] = vol_dict['vol_20'] / vol_dict['vol_30']
            
            # Moving averages - compare current price
            ma_dict = self.calculate_moving_averages(prices)
            for period in [5, 10, 20, 30]:
                if f'sma_{period}' in ma_dict:
                    sma_val = ma_dict[f'sma_{period}'].iloc[-1]
                    features[f"price_vs_sma_{period}"] = price / sma_val if sma_val > 0 else 1.0
                else:
                    features[f"price_vs_sma_{period}"] = 1.0
            
            for period in [5, 12]:
                if f'ema_{period}' in ma_dict:
                    ema_val = ma_dict[f'ema_{period}'].iloc[-1]
                    features[f"price_vs_ema_{period}"] = price / ema_val if ema_val > 0 else 1.0
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
            
            # Volume ratios using current volume
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
                price_change = prices.pct_change().fillna(0)
                vpt = (volumes * price_change).cumsum()
                features["volume_price_trend"] = vpt.iloc[-1] / 1000000 if not np.isnan(vpt.iloc[-1]) else 0
            else:
                features["volume_price_trend"] = 0
            
            # Momentum using current price
            for period in [3, 5, 10]:
                if len(prices) > period:
                    features[f"momentum_{period}"] = (price - prices.iloc[-period]) / prices.iloc[-period]
                else:
                    features[f"momentum_{period}"] = returns * (period / 10)
            
            # Price percentiles including current price
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
            features["trending_market"] = 1.0 if len(returns_series) >= 10 and abs(returns_series.tail(10).mean()) > 0.001 else 0.0
            
        else:
            # Fallback to approximations if no historical data
            logger.warning(f"No historical data for {symbol}, using approximations")
            
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
        now = datetime.utcnow()  # Use UTC
        hour_angle = (now.hour / 24) * 2 * np.pi
        features["hour_sin"] = np.sin(hour_angle)
        features["hour_cos"] = np.cos(hour_angle)
        features["is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0
        
        return features
    
    def normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        """Normalize features using the manual scaler."""
        # Convert to array in correct order
        feature_array = np.array([features[name] for name in self.feature_names])
        
        # Handle NaN/Inf
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Normalize
        normalized = (feature_array - self.scaler_means) / (self.scaler_stds + 1e-8)
        normalized = np.clip(normalized, -5, 5)
        
        return normalized.astype(np.float32)
    
    def close(self):
        """Close connections."""
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.redis_client:
            self.redis_client.close()
            self.redis_client = None