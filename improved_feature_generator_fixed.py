#!/usr/bin/env python3
"""
Fixed version of improved feature generator
Addresses all critical bugs mentioned by the user
"""

import numpy as np
import pandas as pd
import duckdb
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import json
from collections import deque
import logging
import threading

logger = logging.getLogger(__name__)


class ImprovedFeatureGeneratorFixed:
    """Generate 44 features using real historical data with bug fixes."""
    
    def __init__(self, db_path: str = "data/historical_data.duckdb"):
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
        
        # Verify scaler dimensions match
        assert len(self.feature_names) == 44, f"Expected 44 features, got {len(self.feature_names)}"
        
        # Load manual scaler
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
        
        assert len(self.scaler_means) == 44, f"Scaler means dimension mismatch"
        assert len(self.scaler_stds) == 44, f"Scaler stds dimension mismatch"
        
        # Historical data cache for each symbol (using deque for dynamic window)
        self.historical_data = {}
        self.db_path = db_path
        self.conn = None
        self.cache_lock = threading.Lock()  # Thread safety for cache updates
        
        # Initialize database connection
        self._connect_db()
        
        # Sliding window configuration
        self.window_size = 100  # Keep last 100 candles
        self.last_update_time = {}  # Track last update time per symbol
        
    def _connect_db(self):
        """Connect to DuckDB database."""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to historical database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.conn = None
    
    def _get_latest_timestamp(self, symbol: str) -> datetime:
        """Get the latest timestamp from database for a symbol."""
        if not self.conn:
            return datetime.utcnow()
        
        try:
            # Try multiple table naming conventions
            for table_pattern in [f"kline_{symbol}", f"klines_{symbol.lower()}", "all_klines"]:
                try:
                    if table_pattern == "all_klines":
                        query = f"""
                        SELECT MAX(CASE 
                            WHEN timestamp > 1e12 THEN timestamp / 1000
                            ELSE timestamp 
                        END) as max_ts
                        FROM {table_pattern}
                        WHERE symbol = '{symbol}'
                        """
                    else:
                        query = f"""
                        SELECT MAX(CASE 
                            WHEN open_time > 1e12 THEN open_time / 1000
                            ELSE open_time 
                        END) as max_ts
                        FROM {table_pattern}
                        """
                    
                    result = self.conn.execute(query).fetchone()
                    if result and result[0]:
                        # Convert from seconds to datetime
                        return datetime.fromtimestamp(result[0])
                except:
                    continue
            
            logger.warning(f"Could not find latest timestamp for {symbol}, using current time")
            return datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error getting latest timestamp: {e}")
            return datetime.utcnow()
    
    def load_historical_data(self, symbol: str, lookback_days: int = 60) -> pd.DataFrame:
        """Load historical kline data for a symbol with dynamic end date."""
        if not self.conn:
            logger.warning("No database connection available")
            return pd.DataFrame()
        
        try:
            # Get the latest available timestamp dynamically
            end_date = self._get_latest_timestamp(symbol)
            start_date = end_date - timedelta(days=lookback_days)
            
            # Convert to milliseconds for comparison
            start_ms = int(start_date.timestamp() * 1000)
            end_ms = int(end_date.timestamp() * 1000)
            
            # Try different table naming patterns
            for table_pattern in [f"kline_{symbol}", f"klines_{symbol.lower()}", "all_klines"]:
                try:
                    if table_pattern == "all_klines":
                        query = f"""
                        SELECT 
                            CASE 
                                WHEN timestamp > 1e12 THEN timestamp / 1000
                                ELSE timestamp 
                            END as timestamp_sec,
                            open,
                            high,
                            low,
                            close,
                            volume,
                            turnover
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
                            turnover
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
                        
                        # Remove any NaN rows
                        df = df.dropna()
                        
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
    
    def update_historical_cache(self, symbol: str):
        """Update historical data cache for a symbol with thread safety."""
        with self.cache_lock:
            df = self.load_historical_data(symbol)
            if not df.empty:
                self.historical_data[symbol] = df
                self.last_update_time[symbol] = datetime.utcnow()
    
    def append_realtime_data(self, symbol: str, ticker_data: Dict[str, float]) -> pd.DataFrame:
        """Append current ticker data to historical data for accurate calculations."""
        hist_data = self.historical_data.get(symbol, pd.DataFrame()).copy()
        
        if hist_data.empty:
            return hist_data
        
        # Create a new row with current ticker data
        current_time = datetime.utcnow()
        price = float(ticker_data.get("lastPrice", 0))
        
        # Check if we need to add this as a new candle
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
                
                # Keep only last window_size candles
                if len(hist_data) > self.window_size:
                    hist_data = hist_data.tail(self.window_size)
                
                # Recalculate derived fields
                hist_data['returns'] = hist_data['close'].pct_change().fillna(0)
                hist_data['log_returns'] = np.log(hist_data['close'] / hist_data['close'].shift(1)).fillna(0)
                hist_data['hl_ratio'] = ((hist_data['high'] - hist_data['low']) / hist_data['close']).fillna(0.02)
                hist_data['oc_ratio'] = ((hist_data['close'] - hist_data['open']) / hist_data['close']).fillna(0)
        
        return hist_data
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate various moving averages with safety checks."""
        ma_dict = {}
        
        # Simple Moving Averages
        for period in [5, 10, 20, 30]:
            ma = prices.rolling(window=period, min_periods=1).mean()
            ma_dict[f'sma_{period}'] = ma.fillna(prices.mean())
        
        # Exponential Moving Averages
        for period in [5, 12]:
            ema = prices.ewm(span=period, adjust=False).mean()
            ma_dict[f'ema_{period}'] = ema.fillna(prices.mean())
        
        return ma_dict
    
    def calculate_volatility(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate volatility over different periods with safety checks."""
        vol_dict = {}
        
        # Remove NaN values
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
        """Calculate RSI with improved error handling."""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        result = rsi.iloc[-1]
        return result if not np.isnan(result) else 50.0
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, num_std: float = 2) -> Tuple[float, float]:
        """Calculate Bollinger Bands with safety checks."""
        if len(prices) < period:
            return 0.0, 0.04
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        current_price = prices.iloc[-1]
        current_sma = sma.iloc[-1]
        
        # Safety check for band width calculation
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
        """Calculate MACD with safety checks."""
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
        """Generate all 44 features with real-time data integration."""
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
        
        # Get historical data and append current price
        with self.cache_lock:
            hist_data = self.append_realtime_data(symbol, ticker_data)
        
        if not hist_data.empty and len(hist_data) > 1:
            # Use combined historical + realtime data
            prices = hist_data['close']
            returns_series = hist_data['returns'].dropna()
            volumes = hist_data['volume']
            
            # Multi-period returns using actual latest price
            for period in [1, 3, 5, 10, 20]:
                if len(prices) > period:
                    # Use current price vs historical price
                    features[f"return_{period}"] = (price - prices.iloc[-period-1]) / prices.iloc[-period-1]
                else:
                    features[f"return_{period}"] = returns * (period / 20)
            
            # Volatility
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
            
            # Moving averages - compare current price to MAs
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
            
            # Use approximations for all features
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
        now = datetime.utcnow()  # Use UTC for consistency
        hour_angle = (now.hour / 24) * 2 * np.pi
        features["hour_sin"] = np.sin(hour_angle)
        features["hour_cos"] = np.cos(hour_angle)
        features["is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0
        
        return features
    
    def normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        """Normalize features with NaN/Inf handling."""
        # Convert to array in correct order
        feature_array = np.array([features[name] for name in self.feature_names])
        
        # Handle NaN and Inf values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Normalize
        normalized = (feature_array - self.scaler_means) / (self.scaler_stds + 1e-8)
        normalized = np.clip(normalized, -5, 5)
        
        return normalized.astype(np.float32)
    
    def refresh_cache(self, symbol: str, force: bool = False):
        """Refresh cache if stale or forced."""
        last_update = self.last_update_time.get(symbol)
        
        # Update if no previous update, forced, or data is older than 5 minutes
        if force or not last_update or (datetime.utcnow() - last_update).total_seconds() > 300:
            self.update_historical_cache(symbol)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


# Unit test helper
def validate_features(features: Dict[str, float], feature_names: List[str]) -> bool:
    """Validate that all features are present and within reasonable bounds."""
    if len(features) != len(feature_names):
        return False
    
    for name in feature_names:
        if name not in features:
            return False
        
        value = features[name]
        if np.isnan(value) or np.isinf(value):
            return False
    
    return True