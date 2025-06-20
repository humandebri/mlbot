"""
Technical indicators calculator for the 44-feature model.
Calculates all features required by the v3.1_improved model.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Deque
from collections import defaultdict, deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicatorEngine:
    """
    Engine for calculating the 44 technical indicators required by the ML model.
    These are the EXACT features that the v3.1_improved model expects.
    """
    
    def __init__(self, lookback_periods: int = 200):
        """Initialize with price history storage."""
        self.lookback_periods = lookback_periods
        
        # Price history storage
        self.price_history = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.high_history = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.low_history = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.close_history = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.volume_history = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.returns_history = defaultdict(lambda: deque(maxlen=lookback_periods))
        
        # Technical indicator storage
        self.sma_5 = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.sma_10 = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.sma_20 = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.sma_30 = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.sma_50 = defaultdict(lambda: deque(maxlen=lookback_periods))
        
        self.ema_5 = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.ema_12 = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.ema_26 = defaultdict(lambda: deque(maxlen=lookback_periods))
        
        # Latest features cache
        self.latest_features = defaultdict(dict)
        
        logger.info("TechnicalIndicatorEngine initialized")
    
    def update_price_data(self, symbol: str, open_price: float, high: float, 
                         low: float, close: float, volume: float) -> Dict[str, float]:
        """
        Update price history and calculate all 44 technical indicators.
        Returns a dictionary with all features the model expects.
        """
        try:
            # Store price data
            self.close_history[symbol].append(close)
            self.high_history[symbol].append(high)
            self.low_history[symbol].append(low)
            self.volume_history[symbol].append(volume)
            
            # Calculate returns
            if len(self.close_history[symbol]) > 1:
                prev_close = self.close_history[symbol][-2]
                returns = (close - prev_close) / prev_close if prev_close > 0 else 0
                self.returns_history[symbol].append(returns)
            else:
                returns = 0
                self.returns_history[symbol].append(returns)
            
            # Initialize features dict with all 44 expected features
            features = {}
            
            # 1. Basic returns
            features["returns"] = returns
            prev_close = self.close_history[symbol][-2] if len(self.close_history[symbol]) > 1 else close
            features["log_returns"] = np.log(close / prev_close) if prev_close > 0 else 0
            
            # 2. Price ratios
            features["hl_ratio"] = high / low if low > 0 else 1.0
            features["oc_ratio"] = abs(open_price - close) / close if close > 0 else 0
            
            # 3. Multi-period returns
            for period in [1, 3, 5, 10, 20]:
                features[f"return_{period}"] = self._calculate_return(symbol, period)
            
            # 4. Volatility measures
            for period in [5, 10, 20, 30]:
                features[f"vol_{period}"] = self._calculate_volatility(symbol, period)
            
            # 5. Volatility ratios
            features["vol_ratio_10"] = self._calculate_vol_ratio(symbol, 10, 5)
            features["vol_ratio_20"] = self._calculate_vol_ratio(symbol, 20, 10)
            
            # 6. Price vs SMA
            for period in [5, 10, 20, 30]:
                features[f"price_vs_sma_{period}"] = self._calculate_price_vs_sma(symbol, close, period)
            
            # 7. Price vs EMA
            features["price_vs_ema_5"] = self._calculate_price_vs_ema(symbol, close, 5)
            features["price_vs_ema_12"] = self._calculate_price_vs_ema(symbol, close, 12)
            
            # 8. MACD
            macd, macd_hist = self._calculate_macd(symbol)
            features["macd"] = macd
            features["macd_hist"] = macd_hist
            
            # 9. RSI (with better calculation)
            features["rsi_14"] = self._calculate_rsi(symbol, 14)
            features["rsi_21"] = self._calculate_rsi(symbol, 21)
            
            # Debug: Log calculation attempts
            if len(self.close_history[symbol]) < 15:
                logger.warning(f"Insufficient data for RSI calculation: {len(self.close_history[symbol])} prices")
            
            # 10. Bollinger Bands
            bb_position, bb_width = self._calculate_bollinger_bands(symbol, close, 20)
            features["bb_position_20"] = bb_position
            features["bb_width_20"] = bb_width
            
            # 11. Volume ratios
            features["volume_ratio_10"] = self._calculate_volume_ratio(symbol, volume, 10)
            features["volume_ratio_20"] = self._calculate_volume_ratio(symbol, volume, 20)
            
            # 12. Volume features
            features["log_volume"] = np.log(volume + 1)
            features["volume_price_trend"] = self._calculate_volume_price_trend(symbol)
            
            # 13. Momentum
            for period in [3, 5, 10]:
                features[f"momentum_{period}"] = self._calculate_momentum(symbol, period)
            
            # 14. Price percentiles
            features["price_percentile_20"] = self._calculate_price_percentile(symbol, close, 20)
            features["price_percentile_50"] = self._calculate_price_percentile(symbol, close, 50)
            
            # 15. Trend strength
            features["trend_strength_short"] = self._calculate_trend_strength(symbol, 10)
            features["trend_strength_long"] = self._calculate_trend_strength(symbol, 30)
            
            # 16. Market regime
            features["high_vol_regime"] = 1.0 if features.get("vol_20", 0) > 0.02 else 0.0
            features["low_vol_regime"] = 1.0 if features.get("vol_20", 0) < 0.01 else 0.0
            features["trending_market"] = 1.0 if abs(features.get("trend_strength_long", 0)) > 0.5 else 0.0
            
            # 17. Time features
            # Calculate time-based features
            current_time = datetime.now()
            hour = current_time.hour
            day_of_week = current_time.weekday()
            
            # Hour features (sin/cos for circular encoding)
            hour_angle = 2 * np.pi * hour / 24
            features["hour_sin"] = np.sin(hour_angle)
            features["hour_cos"] = np.cos(hour_angle)
            
            # Weekend feature (Saturday=5, Sunday=6)
            features["is_weekend"] = 1.0 if day_of_week >= 5 else 0.0
            
            # Store latest features
            self.latest_features[symbol] = features
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            # Return zero features on error
            return self._get_zero_features()
    
    def _calculate_return(self, symbol: str, period: int) -> float:
        """Calculate return over specified period."""
        prices = list(self.close_history[symbol])
        if len(prices) > period:
            return (prices[-1] - prices[-period-1]) / prices[-period-1]
        return 0.0
    
    def _calculate_volatility(self, symbol: str, period: int) -> float:
        """Calculate volatility (standard deviation of returns) over period."""
        returns = list(self.returns_history[symbol])
        if len(returns) >= period:
            return np.std(returns[-period:])
        return 0.01  # Default volatility
    
    def _calculate_vol_ratio(self, symbol: str, period1: int, period2: int) -> float:
        """Calculate ratio of volatilities."""
        vol1 = self._calculate_volatility(symbol, period1)
        vol2 = self._calculate_volatility(symbol, period2)
        return vol1 / vol2 if vol2 > 0 else 1.0
    
    def _calculate_price_vs_sma(self, symbol: str, price: float, period: int) -> float:
        """Calculate price relative to simple moving average."""
        prices = list(self.close_history[symbol])
        if len(prices) >= period:
            sma = np.mean(prices[-period:])
            return (price - sma) / sma if sma > 0 else 0
        return 0.0
    
    def _calculate_price_vs_ema(self, symbol: str, price: float, period: int) -> float:
        """Calculate price relative to exponential moving average."""
        prices = list(self.close_history[symbol])
        if len(prices) >= period:
            ema = self._ema(prices, period)
            return (price - ema) / ema if ema > 0 else 0
        return 0.0
    
    def _ema(self, prices: List[float], period: int) -> float:
        """Calculate exponential moving average."""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])  # Initial SMA
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_macd(self, symbol: str) -> tuple:
        """Calculate MACD and MACD histogram."""
        prices = list(self.close_history[symbol])
        
        # Use available data even if less than ideal
        if len(prices) >= 6:  # Minimum 6 periods
            # Use shorter periods for insufficient data
            ema1_period = min(12, len(prices) // 2)
            ema2_period = min(26, len(prices) - 1)
            
            if ema1_period != ema2_period:
                ema_short = self._ema(prices, ema1_period)
                ema_long = self._ema(prices, ema2_period)
                macd = ema_short - ema_long
                
                # Simplified signal line calculation
                signal_period = min(9, len(prices) // 3)
                signal = macd * 0.9  # Simplified
                macd_hist = macd - signal
                
                return macd, macd_hist
        
        return 0.0, 0.0
    
    def _calculate_rsi(self, symbol: str, period: int) -> float:
        """Calculate Relative Strength Index."""
        prices = list(self.close_history[symbol])
        
        # Use minimum 3 periods for basic RSI calculation
        min_periods = max(3, period // 3)
        if len(prices) < min_periods:
            return 50.0  # Neutral RSI
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) >= period:
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        return 50.0
    
    def _calculate_bollinger_bands(self, symbol: str, price: float, period: int) -> tuple:
        """Calculate Bollinger Bands position and width."""
        prices = list(self.close_history[symbol])
        if len(prices) >= period:
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            # Position: -1 to 1 (-1 = at lower band, 1 = at upper band)
            bb_position = (price - lower_band) / (upper_band - lower_band) * 2 - 1 if upper_band != lower_band else 0
            
            # Width: normalized by SMA
            bb_width = (upper_band - lower_band) / sma if sma > 0 else 0
            
            return bb_position, bb_width
        
        return 0.0, 0.0
    
    def _calculate_volume_ratio(self, symbol: str, volume: float, period: int) -> float:
        """Calculate volume relative to average."""
        volumes = list(self.volume_history[symbol])
        if len(volumes) >= period:
            avg_volume = np.mean(volumes[-period:])
            return volume / avg_volume if avg_volume > 0 else 1.0
        return 1.0
    
    def _calculate_volume_price_trend(self, symbol: str) -> float:
        """Calculate volume-price trend indicator."""
        prices = list(self.close_history[symbol])
        volumes = list(self.volume_history[symbol])
        
        if len(prices) >= 2 and len(volumes) >= 2:
            price_change = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] > 0 else 0
            volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] > 0 else 0
            return price_change * volume_change
        
        return 0.0
    
    def _calculate_momentum(self, symbol: str, period: int) -> float:
        """Calculate price momentum."""
        prices = list(self.close_history[symbol])
        if len(prices) > period:
            return (prices[-1] - prices[-period]) / prices[-period] if prices[-period] > 0 else 0
        return 0.0
    
    def _calculate_price_percentile(self, symbol: str, price: float, period: int) -> float:
        """Calculate where current price sits in historical range."""
        prices = list(self.close_history[symbol])
        if len(prices) >= period:
            period_prices = prices[-period:]
            below_count = sum(1 for p in period_prices if p < price)
            return below_count / period
        return 0.5
    
    def _calculate_trend_strength(self, symbol: str, period: int) -> float:
        """Calculate trend strength using linear regression slope."""
        prices = list(self.close_history[symbol])
        if len(prices) >= period:
            x = np.arange(period)
            y = prices[-period:]
            
            # Simple linear regression
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(period))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(period))
            
            if denominator > 0:
                slope = numerator / denominator
                # Normalize by price level
                return slope / y_mean if y_mean > 0 else 0
        
        return 0.0
    
    def _get_zero_features(self) -> Dict[str, float]:
        """Return dictionary with all 44 features set to reasonable defaults."""
        return {
            "returns": 0.0, "log_returns": 0.0, "hl_ratio": 1.0, "oc_ratio": 0.0,
            "return_1": 0.0, "return_3": 0.0, "return_5": 0.0, "return_10": 0.0, "return_20": 0.0,
            "vol_5": 0.01, "vol_10": 0.01, "vol_20": 0.01, "vol_30": 0.01,
            "vol_ratio_10": 1.0, "vol_ratio_20": 1.0,
            "price_vs_sma_5": 0.0, "price_vs_sma_10": 0.0, "price_vs_sma_20": 0.0, "price_vs_sma_30": 0.0,
            "price_vs_ema_5": 0.0, "price_vs_ema_12": 0.0,
            "macd": 0.0, "macd_hist": 0.0,
            "rsi_14": 50.0, "rsi_21": 50.0,
            "bb_position_20": 0.0, "bb_width_20": 0.02,
            "volume_ratio_10": 1.0, "volume_ratio_20": 1.0,
            "log_volume": 10.0, "volume_price_trend": 0.0,
            "momentum_3": 0.0, "momentum_5": 0.0, "momentum_10": 0.0,
            "price_percentile_20": 0.5, "price_percentile_50": 0.5,
            "trend_strength_short": 0.0, "trend_strength_long": 0.0,
            "high_vol_regime": 0.0, "low_vol_regime": 0.0, "trending_market": 0.0,
            "hour_sin": 0.0, "hour_cos": 1.0, "is_weekend": 0.0
        }
    
    def get_latest_features(self, symbol: str) -> Dict[str, float]:
        """Get the latest calculated features for a symbol."""
        return self.latest_features.get(symbol, self._get_zero_features())