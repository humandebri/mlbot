"""
Volatility and momentum feature engine optimized for high-frequency signals.

Features computed:
- Multi-timeframe volatility (Garman-Klass, Parkinson, etc.)
- Price momentum and acceleration
- Volume-weighted metrics
- Realized vs implied volatility indicators
"""

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


class VolatilityMomentumEngine:
    """
    High-performance volatility and momentum feature engine.
    
    Optimized for:
    - Multi-timeframe analysis
    - Memory-efficient rolling calculations
    - Real-time volatility estimation
    - Cost-effective computation
    """
    
    def __init__(self, max_history: int = 3600):
        """
        Initialize volatility and momentum engine.
        
        Args:
            max_history: Maximum samples to keep in memory
        """
        self.max_history = max_history
        
        # Per-symbol price and volume data
        self.price_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_history))
        self.volume_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_history))
        self.trade_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # OHLCV data for kline-based features
        self.kline_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
        
        # Feature cache
        self.latest_features: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.last_update: Dict[str, float] = defaultdict(float)
        
        # Performance optimization
        self.min_update_interval = 0.5  # Update every 500ms max
        self.min_samples = 20  # Minimum samples for volatility calculation
        
        # Timeframes for analysis (in seconds)
        self.timeframes = {
            "5s": 5,
            "30s": 30, 
            "1m": 60,
            "5m": 300,
            "15m": 900
        }
    
    def process_kline(self, symbol: str, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Process kline data and compute volatility/momentum features.
        
        Args:
            symbol: Trading symbol
            data: Kline data from ingestor
            
        Returns:
            Dictionary of computed features
        """
        try:
            current_time = time.time()
            
            # Rate limiting
            if current_time - self.last_update[symbol] < self.min_update_interval:
                return self.latest_features[symbol]
            
            # Extract OHLCV data
            ohlcv = {
                "timestamp": data.get("timestamp", current_time),
                "open": float(data.get("open", 0)),
                "high": float(data.get("high", 0)),
                "low": float(data.get("low", 0)),
                "close": float(data.get("close", 0)),
                "volume": float(data.get("volume", 0)),
                "turnover": float(data.get("turnover", 0)),
                "confirm": data.get("confirm", False)
            }
            
            # Only process confirmed klines for accuracy
            if not ohlcv["confirm"]:
                return self.latest_features[symbol]
            
            # Store kline data
            self.kline_data[symbol].append(ohlcv)
            self.price_data[symbol].append(ohlcv["close"])
            self.volume_data[symbol].append(ohlcv["volume"])
            
            # Compute features
            features = {}
            
            # 1. Volatility features
            features.update(self._compute_volatility_features(symbol))
            
            # 2. Momentum features
            features.update(self._compute_momentum_features(symbol))
            
            # 3. Volume features
            features.update(self._compute_volume_features(symbol))
            
            # 4. Price action features
            features.update(self._compute_price_action_features(symbol))
            
            # Cache results
            self.latest_features[symbol] = features
            self.last_update[symbol] = current_time
            
            return features
            
        except Exception as e:
            logger.warning(f"Error computing volatility/momentum features for {symbol}", exception=e)
            return {}
    
    def process_trade(self, symbol: str, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Process trade data for high-frequency momentum signals.
        
        Args:
            symbol: Trading symbol
            data: Trade data from ingestor
            
        Returns:
            Dictionary of computed features
        """
        try:
            # Extract trade data
            trade = {
                "timestamp": data.get("timestamp", time.time()),
                "price": float(data.get("price", 0)),
                "size": float(data.get("size", 0)),
                "side": data.get("side", ""),
                "trade_time": data.get("trade_time", 0)
            }
            
            if trade["price"] <= 0 or trade["size"] <= 0:
                return {}
            
            # Store trade data
            self.trade_data[symbol].append(trade)
            
            # Compute trade-based features
            features = {}
            
            # 1. Trade flow features
            features.update(self._compute_trade_flow_features(symbol))
            
            # 2. Microstructure features
            features.update(self._compute_microstructure_features(symbol))
            
            return features
            
        except Exception as e:
            logger.warning(f"Error processing trade data for {symbol}", exception=e)
            return {}
    
    def _compute_volatility_features(self, symbol: str) -> Dict[str, float]:
        """Compute various volatility estimators."""
        features = {}
        
        try:
            klines = list(self.kline_data[symbol])
            if len(klines) < self.min_samples:
                return features
            
            # Extract OHLC arrays for different timeframes
            for tf_name, tf_seconds in self.timeframes.items():
                # Calculate how many samples to use for this timeframe
                samples_needed = min(tf_seconds // 60, len(klines))  # Assuming 1m klines
                if samples_needed < 5:
                    continue
                
                recent_klines = klines[-samples_needed:]
                
                # Extract price arrays
                opens = np.array([k["open"] for k in recent_klines], dtype=np.float32)
                highs = np.array([k["high"] for k in recent_klines], dtype=np.float32)
                lows = np.array([k["low"] for k in recent_klines], dtype=np.float32)
                closes = np.array([k["close"] for k in recent_klines], dtype=np.float32)
                
                # 1. Close-to-close volatility (simple)
                if len(closes) > 1:
                    returns = np.diff(closes) / closes[:-1]
                    features[f"vol_close_{tf_name}"] = float(np.std(returns) * np.sqrt(len(returns)))
                
                # 2. Garman-Klass volatility estimator
                if len(recent_klines) > 1:
                    gk_vol = self._garman_klass_volatility(opens, highs, lows, closes)
                    features[f"vol_gk_{tf_name}"] = float(gk_vol)
                
                # 3. Parkinson volatility estimator
                if len(recent_klines) > 1:
                    park_vol = self._parkinson_volatility(highs, lows)
                    features[f"vol_park_{tf_name}"] = float(park_vol)
                
                # 4. True Range based volatility
                if len(recent_klines) > 1:
                    tr_vol = self._true_range_volatility(highs, lows, closes)
                    features[f"vol_tr_{tf_name}"] = float(tr_vol)
                
                # 5. Volatility ratios and regime indicators
                if f"vol_close_{tf_name}" in features and f"vol_gk_{tf_name}" in features:
                    close_vol = features[f"vol_close_{tf_name}"]
                    gk_vol = features[f"vol_gk_{tf_name}"]
                    if gk_vol > 0:
                        features[f"vol_ratio_{tf_name}"] = close_vol / gk_vol
            
            # Rolling volatility statistics
            if len(klines) >= 60:  # At least 1 hour of data
                recent_closes = np.array([k["close"] for k in klines[-60:]], dtype=np.float32)
                returns_1h = np.diff(recent_closes) / recent_closes[:-1]
                
                # Volatility percentiles
                vol_1h = np.std(returns_1h)
                
                if len(klines) >= 300:  # 5 hours for comparison
                    longer_closes = np.array([k["close"] for k in klines[-300:]], dtype=np.float32)
                    longer_returns = np.diff(longer_closes) / longer_closes[:-1]
                    vol_5h = np.std(longer_returns)
                    
                    features["vol_regime"] = float(vol_1h / vol_5h) if vol_5h > 0 else 1.0
        
        except Exception as e:
            logger.warning(f"Error computing volatility features for {symbol}", exception=e)
        
        return features
    
    def _compute_momentum_features(self, symbol: str) -> Dict[str, float]:
        """Compute momentum and trend features."""
        features = {}
        
        try:
            prices = list(self.price_data[symbol])
            if len(prices) < self.min_samples:
                return features
            
            price_array = np.array(prices, dtype=np.float32)
            
            # Multi-timeframe returns
            for tf_name, tf_seconds in self.timeframes.items():
                lookback = min(tf_seconds // 60, len(prices) - 1)  # Assuming 1m granularity
                if lookback < 1:
                    continue
                
                current_price = price_array[-1]
                past_price = price_array[-lookback-1]
                
                if past_price > 0:
                    features[f"return_{tf_name}"] = float((current_price - past_price) / past_price)
            
            # Price acceleration (second derivative)
            if len(prices) >= 3:
                recent_prices = price_array[-10:]  # Last 10 samples
                if len(recent_prices) >= 3:
                    # Calculate price momentum acceleration
                    returns = np.diff(recent_prices) / recent_prices[:-1]
                    if len(returns) >= 2:
                        acceleration = np.diff(returns)
                        features["price_acceleration"] = float(np.mean(acceleration))
            
            # Momentum indicators
            if len(prices) >= 20:
                # RSI-like momentum
                gains = []
                losses = []
                
                for i in range(1, min(14, len(prices))):
                    change = price_array[-i] - price_array[-i-1]
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(-change)
                
                if gains and losses:
                    avg_gain = np.mean(gains)
                    avg_loss = np.mean(losses)
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        features["momentum_rsi"] = float(100 - (100 / (1 + rs)))
            
            # Price distance from moving averages
            for window in [5, 10, 20, 50]:
                if len(prices) >= window:
                    ma = np.mean(price_array[-window:])
                    current_price = price_array[-1]
                    features[f"price_vs_ma{window}"] = float((current_price - ma) / ma) if ma > 0 else 0
            
            # Trend strength
            if len(prices) >= 30:
                x = np.arange(30)
                y = price_array[-30:]
                correlation = np.corrcoef(x, y)[0, 1]
                features["trend_strength_30"] = float(correlation)
        
        except Exception as e:
            logger.warning(f"Error computing momentum features for {symbol}", exception=e)
        
        return features
    
    def _compute_volume_features(self, symbol: str) -> Dict[str, float]:
        """Compute volume-weighted features."""
        features = {}
        
        try:
            klines = list(self.kline_data[symbol])
            if len(klines) < self.min_samples:
                return features
            
            # Extract volume and price data
            volumes = np.array([k["volume"] for k in klines], dtype=np.float32)
            closes = np.array([k["close"] for k in klines], dtype=np.float32)
            turnovers = np.array([k["turnover"] for k in klines], dtype=np.float32)
            
            # Volume-weighted moving averages
            for window in [5, 10, 20]:
                if len(klines) >= window:
                    recent_volumes = volumes[-window:]
                    recent_closes = closes[-window:]
                    recent_turnovers = turnovers[-window:]
                    
                    # VWAP calculation
                    if np.sum(recent_volumes) > 0:
                        vwap = np.sum(recent_turnovers) / np.sum(recent_volumes)
                        current_price = closes[-1]
                        features[f"vwap_deviation_{window}"] = float((current_price - vwap) / vwap) if vwap > 0 else 0
                    
                    # Volume ratios
                    current_volume = volumes[-1]
                    avg_volume = np.mean(recent_volumes[:-1]) if len(recent_volumes) > 1 else 1
                    features[f"volume_ratio_{window}"] = float(current_volume / avg_volume) if avg_volume > 0 else 0
            
            # Volume momentum
            if len(volumes) >= 10:
                recent_volumes = volumes[-10:]
                if len(recent_volumes) >= 2:
                    volume_changes = np.diff(recent_volumes)
                    features["volume_momentum"] = float(np.mean(volume_changes))
            
            # On-Balance Volume (OBV) approximation
            if len(klines) >= 20:
                obv_values = []
                obv = 0
                
                for i in range(1, min(20, len(klines))):
                    current = klines[-i]
                    previous = klines[-i-1]
                    
                    if current["close"] > previous["close"]:
                        obv += current["volume"]
                    elif current["close"] < previous["close"]:
                        obv -= current["volume"]
                    
                    obv_values.append(obv)
                
                if len(obv_values) >= 2:
                    # OBV trend
                    x = np.arange(len(obv_values))
                    correlation = np.corrcoef(x, obv_values)[0, 1]
                    features["obv_trend"] = float(correlation)
        
        except Exception as e:
            logger.warning(f"Error computing volume features for {symbol}", exception=e)
        
        return features
    
    def _compute_price_action_features(self, symbol: str) -> Dict[str, float]:
        """Compute price action pattern features."""
        features = {}
        
        try:
            klines = list(self.kline_data[symbol])
            if len(klines) < 10:
                return features
            
            recent_klines = klines[-20:]  # Last 20 klines
            
            # Candle patterns
            for i, kline in enumerate(recent_klines[-5:]):  # Last 5 candles
                open_price = kline["open"]
                close_price = kline["close"]
                high_price = kline["high"]
                low_price = kline["low"]
                
                # Body size
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                
                if total_range > 0:
                    features[f"body_ratio_{i}"] = body_size / total_range
                
                # Upper and lower shadows
                if open_price != 0 and close_price != 0:
                    upper_shadow = high_price - max(open_price, close_price)
                    lower_shadow = min(open_price, close_price) - low_price
                    
                    if total_range > 0:
                        features[f"upper_shadow_ratio_{i}"] = upper_shadow / total_range
                        features[f"lower_shadow_ratio_{i}"] = lower_shadow / total_range
            
            # High/Low analysis
            highs = np.array([k["high"] for k in recent_klines], dtype=np.float32)
            lows = np.array([k["low"] for k in recent_klines], dtype=np.float32)
            
            # Support/Resistance levels
            current_price = recent_klines[-1]["close"]
            
            # Distance to recent high/low
            recent_high = np.max(highs)
            recent_low = np.min(lows)
            
            if recent_high > 0:
                features["distance_to_high"] = (recent_high - current_price) / recent_high
            if recent_low > 0:
                features["distance_to_low"] = (current_price - recent_low) / current_price
            
            # Range expansion/contraction
            if len(recent_klines) >= 10:
                recent_ranges = highs[-5:] - lows[-5:]
                older_ranges = highs[-10:-5] - lows[-10:-5]
                
                avg_recent_range = np.mean(recent_ranges)
                avg_older_range = np.mean(older_ranges)
                
                if avg_older_range > 0:
                    features["range_expansion"] = avg_recent_range / avg_older_range
        
        except Exception as e:
            logger.warning(f"Error computing price action features for {symbol}", exception=e)
        
        return features
    
    def _compute_trade_flow_features(self, symbol: str) -> Dict[str, float]:
        """Compute trade flow and aggressor analysis features."""
        features = {}
        
        try:
            trades = list(self.trade_data[symbol])
            if len(trades) < 10:
                return features
            
            # Analyze recent trades (last 100)
            recent_trades = trades[-100:]
            current_time = time.time()
            
            # Trade flow by time windows
            for window_name, window_seconds in [("5s", 5), ("30s", 30), ("1m", 60)]:
                cutoff_time = current_time - window_seconds
                window_trades = [t for t in recent_trades if t["timestamp"] >= cutoff_time]
                
                if window_trades:
                    # Buy vs sell volume
                    buy_volume = sum(t["size"] for t in window_trades if t["side"].lower() == "buy")
                    sell_volume = sum(t["size"] for t in window_trades if t["side"].lower() == "sell")
                    total_volume = buy_volume + sell_volume
                    
                    if total_volume > 0:
                        features[f"buy_ratio_{window_name}"] = buy_volume / total_volume
                        features[f"trade_imbalance_{window_name}"] = (buy_volume - sell_volume) / total_volume
                    
                    # Trade intensity
                    features[f"trade_count_{window_name}"] = len(window_trades)
                    features[f"avg_trade_size_{window_name}"] = total_volume / len(window_trades) if window_trades else 0
            
            # Price impact of recent trades
            if len(recent_trades) >= 20:
                # Calculate weighted average trade price vs current price
                total_value = sum(t["price"] * t["size"] for t in recent_trades[-10:])
                total_volume = sum(t["size"] for t in recent_trades[-10:])
                
                if total_volume > 0:
                    avg_trade_price = total_value / total_volume
                    current_price = recent_trades[-1]["price"]
                    features["trade_price_impact"] = (current_price - avg_trade_price) / avg_trade_price if avg_trade_price > 0 else 0
        
        except Exception as e:
            logger.warning(f"Error computing trade flow features for {symbol}", exception=e)
        
        return features
    
    def _compute_microstructure_features(self, symbol: str) -> Dict[str, float]:
        """Compute microstructure features from trade data."""
        features = {}
        
        try:
            trades = list(self.trade_data[symbol])
            if len(trades) < 20:
                return features
            
            recent_trades = trades[-50:]  # Last 50 trades
            
            # Trade size distribution
            sizes = [t["size"] for t in recent_trades]
            if sizes:
                size_array = np.array(sizes, dtype=np.float32)
                features["trade_size_mean"] = float(np.mean(size_array))
                features["trade_size_std"] = float(np.std(size_array))
                features["trade_size_skew"] = float(self._calculate_skewness(size_array))
            
            # Inter-trade time analysis
            if len(recent_trades) >= 10:
                timestamps = [t["timestamp"] for t in recent_trades[-10:]]
                inter_times = np.diff(timestamps)
                
                if len(inter_times) > 0:
                    features["avg_inter_trade_time"] = float(np.mean(inter_times))
                    features["inter_trade_time_std"] = float(np.std(inter_times))
            
            # Price clustering (tendency to trade at round numbers)
            prices = [t["price"] for t in recent_trades]
            if prices:
                # Count trades at round prices (ending in 0)
                round_prices = sum(1 for p in prices if abs(p - round(p, 1)) < 0.001)
                features["price_clustering"] = round_prices / len(prices)
        
        except Exception as e:
            logger.warning(f"Error computing microstructure features for {symbol}", exception=e)
        
        return features
    
    def _garman_klass_volatility(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Calculate Garman-Klass volatility estimator."""
        try:
            if len(highs) != len(lows) or len(highs) < 2:
                return 0.0
            
            # GK estimator: 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
            ln_hl = np.log(highs / lows)
            ln_co = np.log(closes / opens)
            
            gk_values = 0.5 * ln_hl ** 2 - (2 * np.log(2) - 1) * ln_co ** 2
            return np.sqrt(np.mean(gk_values))
            
        except Exception:
            return 0.0
    
    def _parkinson_volatility(self, highs: np.ndarray, lows: np.ndarray) -> float:
        """Calculate Parkinson volatility estimator."""
        try:
            if len(highs) != len(lows) or len(highs) < 2:
                return 0.0
            
            # Parkinson estimator: sqrt(1/(4*ln(2)) * mean(ln(H/L)^2))
            ln_hl = np.log(highs / lows)
            parkinson_values = ln_hl ** 2
            return np.sqrt(np.mean(parkinson_values) / (4 * np.log(2)))
            
        except Exception:
            return 0.0
    
    def _true_range_volatility(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Calculate True Range based volatility."""
        try:
            if len(highs) < 2:
                return 0.0
            
            # True Range: max(H-L, |H-C_prev|, |L-C_prev|)
            tr_values = []
            
            for i in range(1, len(highs)):
                hl = highs[i] - lows[i]
                hc = abs(highs[i] - closes[i-1])
                lc = abs(lows[i] - closes[i-1])
                tr = max(hl, hc, lc)
                tr_values.append(tr / closes[i-1] if closes[i-1] > 0 else 0)  # Normalize
            
            return np.sqrt(np.mean(tr_values)) if tr_values else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            if len(data) < 3:
                return 0.0
            
            mean = np.mean(data)
            std = np.std(data)
            
            if std == 0:
                return 0.0
            
            skew = np.mean(((data - mean) / std) ** 3)
            return skew
            
        except Exception:
            return 0.0
    
    def get_feature_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of computed features for monitoring."""
        return {
            "symbol": symbol,
            "feature_count": len(self.latest_features.get(symbol, {})),
            "last_update": self.last_update.get(symbol, 0),
            "price_samples": len(self.price_data.get(symbol, [])),
            "kline_samples": len(self.kline_data.get(symbol, [])),
            "trade_samples": len(self.trade_data.get(symbol, [])),
            "latest_features": list(self.latest_features.get(symbol, {}).keys())
        }