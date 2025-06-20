"""
Fixed version of PriceFeatureEngine with proper attribute initialization.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from datetime import datetime, timedelta

from ..common.logging import get_logger

logger = get_logger(__name__)


class PriceFeatureEngine:
    """
    Fixed engine for calculating basic price-based features.
    Essential for providing foundational market data features.
    """
    
    def __init__(self):
        """Initialize price feature engine with proper attributes."""
        # CRITICAL: Initialize latest_features to avoid AttributeError
        self.latest_features = defaultdict(dict)
        
        # Price history for calculations
        self.price_history = defaultdict(lambda: deque(maxlen=100))
        self.volume_history = defaultdict(lambda: deque(maxlen=100))
        
        # Cache for computed features
        self.feature_cache = {}
        self.cache_ttl = timedelta(seconds=1)
        self.last_cache_time = defaultdict(lambda: datetime.min)
        
        logger.info("PriceFeatureEngine initialized with latest_features attribute")
    
    def process_kline(self, symbol: str, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Process kline (candlestick) data to generate price features.
        CRITICAL: This provides the basic OHLCV features that the model needs.
        """
        try:
            # Extract basic price data
            close = float(data.get("close", 0))
            open_price = float(data.get("open", 0))
            high = float(data.get("high", 0))
            low = float(data.get("low", 0))
            volume = float(data.get("volume", 0))
            trades_count = int(data.get("count", 0))
            
            # CRITICAL: Store current close price for other calculations
            if close > 0:
                self.price_history[symbol].append(close)
                self.volume_history[symbol].append(volume)
            
            # Basic OHLCV features (REQUIRED by model)
            features = {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "trades_count": trades_count,
            }
            
            # Price-based derived features
            if close > 0 and open_price > 0:
                features.update({
                    "price_change": close - open_price,
                    "price_change_pct": (close - open_price) / open_price,
                    "high_low_ratio": high / low if low > 0 else 1.0,
                    "close_to_high": (high - close) / high if high > 0 else 0,
                    "close_to_low": (close - low) / low if low > 0 else 0,
                })
            
            # Candlestick patterns
            body_size = abs(close - open_price)
            total_range = high - low if high > low else 0.01
            
            features.update({
                "body_size": body_size,
                "upper_shadow": high - max(close, open_price) if high > 0 else 0,
                "lower_shadow": min(close, open_price) - low if low > 0 else 0,
                "body_to_range": body_size / total_range if total_range > 0 else 0,
            })
            
            # Volume features
            if len(self.volume_history[symbol]) > 10:
                recent_volumes = list(self.volume_history[symbol])[-10:]
                avg_volume = np.mean(recent_volumes)
                features["volume_ratio"] = volume / avg_volume if avg_volume > 0 else 1.0
            else:
                features["volume_ratio"] = 1.0
            
            # Rolling price features
            if len(self.price_history[symbol]) >= 20:
                prices = list(self.price_history[symbol])
                
                # Moving averages
                features["sma_5"] = np.mean(prices[-5:])
                features["sma_10"] = np.mean(prices[-10:])
                features["sma_20"] = np.mean(prices[-20:])
                
                # Price relative to MAs
                features["close_to_sma5"] = (close - features["sma_5"]) / features["sma_5"] if features["sma_5"] > 0 else 0
                features["close_to_sma10"] = (close - features["sma_10"]) / features["sma_10"] if features["sma_10"] > 0 else 0
                features["close_to_sma20"] = (close - features["sma_20"]) / features["sma_20"] if features["sma_20"] > 0 else 0
                
                # Volatility
                returns = np.diff(prices[-20:]) / prices[-20:-1]
                features["volatility_20"] = np.std(returns) if len(returns) > 0 else 0
            else:
                # Default values for insufficient history
                features.update({
                    "sma_5": close,
                    "sma_10": close,
                    "sma_20": close,
                    "close_to_sma5": 0,
                    "close_to_sma10": 0,
                    "close_to_sma20": 0,
                    "volatility_20": 0,
                })
            
            # Market regime (trend detection)
            if len(self.price_history[symbol]) >= 50:
                prices = list(self.price_history[symbol])
                sma_fast = np.mean(prices[-10:])
                sma_slow = np.mean(prices[-30:])
                
                if sma_fast > sma_slow * 1.02:
                    market_regime = 1.0  # Uptrend
                elif sma_fast < sma_slow * 0.98:
                    market_regime = -1.0  # Downtrend
                else:
                    market_regime = 0.0  # Sideways
                
                features["market_regime"] = market_regime
            else:
                features["market_regime"] = 0.0
            
            # CRITICAL: Store generated features for later use
            self.latest_features[symbol].update(features)
            
            # Cache the result
            self.feature_cache[symbol] = features
            self.last_cache_time[symbol] = datetime.now()
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing kline for {symbol}: {e}", exc_info=True)
            # Return minimal features on error
            return {
                "open": 0,
                "high": 0,
                "low": 0,
                "close": 0,
                "volume": 0,
                "trades_count": 0,
            }
    
    def get_cached_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get cached features if still valid."""
        if symbol in self.feature_cache:
            if datetime.now() - self.last_cache_time[symbol] < self.cache_ttl:
                return self.feature_cache[symbol]
        return None
    
    def get_latest_prices(self, symbol: str, count: int = 10) -> List[float]:
        """Get latest price history."""
        if symbol in self.price_history:
            return list(self.price_history[symbol])[-count:]
        return []
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)."""
        prices = self.get_latest_prices(symbol, period + 1)
        
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        # Calculate price changes
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def calculate_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        prices = self.get_latest_prices(symbol, period)
        
        if len(prices) < period:
            current_price = prices[-1] if prices else 0
            return {
                "bb_upper": current_price,
                "bb_middle": current_price,
                "bb_lower": current_price,
                "bb_width": 0,
                "bb_position": 0.5,
            }
        
        middle = np.mean(prices)
        std = np.std(prices)
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        current_price = prices[-1]
        bb_width = (upper - lower) / middle if middle > 0 else 0
        bb_position = (current_price - lower) / (upper - lower) if upper > lower else 0.5
        
        return {
            "bb_upper": float(upper),
            "bb_middle": float(middle),
            "bb_lower": float(lower),
            "bb_width": float(bb_width),
            "bb_position": float(np.clip(bb_position, 0, 1)),
        }
    
    def update_trade_features(self, symbol: str, buy_volume: float, sell_volume: float) -> Dict[str, float]:
        """Update features based on trade data."""
        # Calculate trade flow features
        total_volume = buy_volume + sell_volume
        
        features = {}
        if total_volume > 0:
            features['trade_buy_ratio'] = buy_volume / total_volume
            features['trade_sell_ratio'] = sell_volume / total_volume
            features['trade_flow_imbalance'] = (buy_volume - sell_volume) / total_volume
        else:
            features['trade_buy_ratio'] = 0.5
            features['trade_sell_ratio'] = 0.5
            features['trade_flow_imbalance'] = 0.0
            
        # Store in latest features
        self.latest_features[symbol].update(features)
        
        return features