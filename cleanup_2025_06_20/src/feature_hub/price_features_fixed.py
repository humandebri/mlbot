"""
Price feature engine for generating basic price features and derivatives - FIXED VERSION.

Fixed the issue with self.latest_features not existing.
"""

import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


class PriceFeatureEngine:
    """
    Engine for generating basic price features needed by the 44-dimension model.
    
    This engine generates:
    - Raw price data (open, high, low, close, volume)
    - Price returns over multiple timeframes
    - Moving averages and price-to-MA ratios
    - Technical indicators
    - Price levels and trends
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize price feature engine.
        
        Args:
            max_history: Maximum price history to keep
        """
        self.max_history = max_history
        
        # Per-symbol price history
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_history))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_history))
        self.trades_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_history))
        
        # Latest values for quick access
        self.latest_price: Dict[str, float] = {}
        self.latest_volume: Dict[str, float] = {}
        self.latest_trades_count: Dict[str, int] = {}
        
        # Store latest features for each symbol (FIX for the missing attribute)
        self.latest_features: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Technical indicator parameters
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2
        
        # Moving average periods
        self.ma_periods = [5, 10, 20, 50, 100]
    
    def process_kline(self, symbol: str, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Process kline data and compute price features.
        
        Args:
            symbol: Trading symbol
            data: Kline data with OHLCV
            
        Returns:
            Dictionary of price features
        """
        try:
            # Extract basic price data
            close = float(data.get("close", 0))
            open_price = float(data.get("open", 0))
            high = float(data.get("high", 0))
            low = float(data.get("low", 0))
            volume = float(data.get("volume", 0))
            trades_count = int(data.get("count", 0))
            timestamp = data.get("timestamp", time.time())
            
            # Skip invalid data
            if close <= 0 or volume < 0:
                return {}
            
            # Update history
            self.price_history[symbol].append({
                "timestamp": timestamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close
            })
            self.volume_history[symbol].append(volume)
            self.trades_history[symbol].append(trades_count)
            
            # Update latest values
            self.latest_price[symbol] = close
            self.latest_volume[symbol] = volume
            self.latest_trades_count[symbol] = trades_count
            
            # Generate features
            features = {
                # Basic price data
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "trades_count": trades_count,
                
                # Price range
                "price_range": high - low if high > low else 0,
                "price_range_pct": (high - low) / close if close > 0 else 0,
                
                # VWAP approximation (using close as proxy)
                "vwap": close,  # Simplified for now
                "vwap_ratio": 1.0  # close / vwap
            }
            
            # Only compute derived features if we have enough history
            if len(self.price_history[symbol]) >= 2:
                # Add returns
                returns_features = self._calculate_returns(symbol)
                features.update(returns_features)
                
                # Add moving averages
                ma_features = self._calculate_moving_averages(symbol)
                features.update(ma_features)
                
                # Add technical indicators
                if len(self.price_history[symbol]) >= self.rsi_period + 1:
                    tech_features = self._calculate_technical_indicators(symbol)
                    features.update(tech_features)
                
                # Add price levels
                level_features = self._calculate_price_levels(symbol)
                features.update(level_features)
                
                # Add momentum features
                momentum_features = self._calculate_momentum(symbol)
                features.update(momentum_features)
                
                # Add volume features
                volume_features = self._calculate_volume_features(symbol)
                features.update(volume_features)
                
                # Add multi-timeframe features
                mtf_features = self._calculate_multi_timeframe_features(symbol)
                features.update(mtf_features)
            
            # Add time-based features
            time_features = self._calculate_time_features(timestamp)
            features.update(time_features)
            
            # Add market regime
            features["market_regime"] = self._calculate_market_regime(symbol)
            
            # Add placeholders for features from other sources
            features["spread_bps"] = 0.0  # Will be updated from orderbook
            features["bid_ask_imbalance"] = 0.0  # Will be updated from orderbook
            features["order_flow_imbalance"] = 0.0  # Will be updated from trades
            features["liquidation_pressure"] = 0.0  # Will be updated from liquidation data
            features["funding_rate"] = 0.0  # Will be updated from funding data
            features["open_interest_change"] = 0.0  # Will be updated from OI data
            
            # Store latest features for the symbol
            self.latest_features[symbol] = features
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing kline for {symbol}: {e}")
            return {}
    
    def _calculate_returns(self, symbol: str) -> Dict[str, float]:
        """Calculate returns over multiple timeframes."""
        features = {}
        prices = [p["close"] for p in self.price_history[symbol]]
        
        # Define return periods (in candles)
        periods = {
            "returns_1": 1,
            "returns_5": 5,
            "returns_15": 15,
            "returns_30": 30,
            "returns_60": 60
        }
        
        current_price = prices[-1]
        
        for name, period in periods.items():
            if len(prices) > period:
                past_price = prices[-period-1]
                if past_price > 0:
                    returns = (current_price - past_price) / past_price
                    features[name] = returns
                else:
                    features[name] = 0.0
            else:
                features[name] = 0.0
        
        # Log returns
        if len(prices) > 1 and prices[-2] > 0:
            features["log_returns"] = np.log(current_price / prices[-2])
        else:
            features["log_returns"] = 0.0
        
        return features
    
    def _calculate_moving_averages(self, symbol: str) -> Dict[str, float]:
        """Calculate moving averages and price-to-MA ratios."""
        features = {}
        prices = [p["close"] for p in self.price_history[symbol]]
        current_price = prices[-1]
        
        # Calculate MAs and ratios
        for period in self.ma_periods:
            if len(prices) >= period:
                ma = np.mean(prices[-period:])
                features[f"ma_{period}"] = ma
                
                if ma > 0:
                    ratio = current_price / ma
                    features[f"price_ma_ratio_{period}"] = ratio
                else:
                    features[f"price_ma_ratio_{period}"] = 1.0
            else:
                features[f"ma_{period}"] = current_price
                features[f"price_ma_ratio_{period}"] = 1.0
        
        return features
    
    def _calculate_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """Calculate RSI and Bollinger Bands."""
        features = {}
        prices = [p["close"] for p in self.price_history[symbol]]
        
        # RSI
        if len(prices) >= self.rsi_period + 1:
            rsi = self._calculate_rsi(prices, self.rsi_period)
            features["rsi_14"] = rsi
        else:
            features["rsi_14"] = 50.0  # Neutral RSI
        
        # Bollinger Bands
        if len(prices) >= self.bb_period:
            middle_band = np.mean(prices[-self.bb_period:])
            std_dev = np.std(prices[-self.bb_period:])
            upper_band = middle_band + (self.bb_std * std_dev)
            lower_band = middle_band - (self.bb_std * std_dev)
            
            current_price = prices[-1]
            band_width = upper_band - lower_band
            
            # BB position (0 = lower band, 1 = upper band)
            if band_width > 0:
                bb_position = (current_price - lower_band) / band_width
                features["bb_position"] = np.clip(bb_position, 0, 1)
            else:
                features["bb_position"] = 0.5
            
            # BB width as percentage of price
            features["bb_width"] = band_width / middle_band if middle_band > 0 else 0
            
            features["bb_upper"] = upper_band
            features["bb_lower"] = lower_band
            features["bb_middle"] = middle_band
        else:
            features["bb_position"] = 0.5
            features["bb_width"] = 0.0
        
        return features
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        # Calculate price changes
        deltas = np.diff(prices[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_price_levels(self, symbol: str) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        features = {}
        prices = [p for p in self.price_history[symbol]]
        
        if len(prices) < 20:
            features["support_distance"] = 0.0
            features["resistance_distance"] = 0.0
            features["trend_strength"] = 0.0
            return features
        
        # Get recent highs and lows
        recent_highs = [p["high"] for p in prices[-50:] if "high" in p]
        recent_lows = [p["low"] for p in prices[-50:] if "low" in p]
        current_price = prices[-1]["close"]
        
        # Simple support/resistance (recent min/max)
        if recent_highs and recent_lows:
            resistance = max(recent_highs)
            support = min(recent_lows)
            
            # Distance to levels as percentage
            features["resistance_distance"] = (resistance - current_price) / current_price if current_price > 0 else 0
            features["support_distance"] = (current_price - support) / current_price if current_price > 0 else 0
        else:
            features["resistance_distance"] = 0.0
            features["support_distance"] = 0.0
        
        # Trend strength (price position in recent range)
        price_range = max(recent_highs) - min(recent_lows) if recent_highs and recent_lows else 0
        if price_range > 0:
            trend_position = (current_price - min(recent_lows)) / price_range
            features["trend_strength"] = np.clip(trend_position, 0, 1)
        else:
            features["trend_strength"] = 0.5
        
        return features
    
    def _calculate_momentum(self, symbol: str) -> Dict[str, float]:
        """Calculate price momentum and acceleration."""
        features = {}
        prices = [p["close"] for p in self.price_history[symbol]]
        
        if len(prices) < 10:
            features["price_momentum"] = 0.0
            features["price_acceleration"] = 0.0
            features["momentum_score"] = 0.5
            features["mean_reversion_score"] = 0.5
            return features
        
        # Price momentum (rate of change)
        momentum_period = 10
        if len(prices) > momentum_period:
            price_change = prices[-1] - prices[-momentum_period-1]
            price_momentum = price_change / prices[-momentum_period-1] if prices[-momentum_period-1] > 0 else 0
            features["price_momentum"] = price_momentum
            
            # Acceleration (momentum change)
            if len(prices) > momentum_period * 2:
                prev_change = prices[-momentum_period-1] - prices[-momentum_period*2-1]
                prev_momentum = prev_change / prices[-momentum_period*2-1] if prices[-momentum_period*2-1] > 0 else 0
                features["price_acceleration"] = price_momentum - prev_momentum
            else:
                features["price_acceleration"] = 0.0
        else:
            features["price_momentum"] = 0.0
            features["price_acceleration"] = 0.0
        
        # Momentum score (0-1, higher = stronger momentum)
        if len(prices) >= 20:
            short_ma = np.mean(prices[-5:])
            long_ma = np.mean(prices[-20:])
            if long_ma > 0:
                momentum_ratio = short_ma / long_ma
                features["momentum_score"] = np.clip((momentum_ratio - 0.95) / 0.1, 0, 1)
            else:
                features["momentum_score"] = 0.5
        else:
            features["momentum_score"] = 0.5
        
        # Mean reversion score (0-1, higher = more likely to revert)
        if len(prices) >= 20:
            mean_price = np.mean(prices[-20:])
            std_price = np.std(prices[-20:])
            if std_price > 0:
                z_score = (prices[-1] - mean_price) / std_price
                features["mean_reversion_score"] = np.clip(abs(z_score) / 2, 0, 1)
            else:
                features["mean_reversion_score"] = 0.0
        else:
            features["mean_reversion_score"] = 0.5
        
        return features
    
    def _calculate_volume_features(self, symbol: str) -> Dict[str, float]:
        """Calculate volume-based features."""
        features = {}
        volumes = list(self.volume_history[symbol])
        
        if len(volumes) < 2:
            features["volume_momentum"] = 0.0
            features["volume_acceleration"] = 0.0
            features["volume_rank"] = 0.5
            features["liquidity_score"] = 0.5
            return features
        
        # Volume momentum
        if len(volumes) >= 10:
            recent_volume = np.mean(volumes[-5:])
            past_volume = np.mean(volumes[-10:-5])
            if past_volume > 0:
                features["volume_momentum"] = (recent_volume - past_volume) / past_volume
            else:
                features["volume_momentum"] = 0.0
            
            # Volume acceleration
            if len(volumes) >= 20:
                older_volume = np.mean(volumes[-20:-10])
                if older_volume > 0:
                    past_momentum = (past_volume - older_volume) / older_volume
                    features["volume_acceleration"] = features["volume_momentum"] - past_momentum
                else:
                    features["volume_acceleration"] = 0.0
            else:
                features["volume_acceleration"] = 0.0
        else:
            features["volume_momentum"] = 0.0
            features["volume_acceleration"] = 0.0
        
        # Volume rank (percentile in recent history)
        if len(volumes) >= 50:
            current_volume = volumes[-1]
            volume_percentile = np.sum(np.array(volumes[-50:]) <= current_volume) / 50.0
            features["volume_rank"] = volume_percentile
        else:
            features["volume_rank"] = 0.5
        
        # Liquidity score (normalized volume)
        if len(volumes) >= 20:
            avg_volume = np.mean(volumes[-20:])
            std_volume = np.std(volumes[-20:])
            if avg_volume > 0:
                # Higher score for consistent high volume
                volume_consistency = 1.0 - (std_volume / avg_volume) if std_volume < avg_volume else 0.0
                features["liquidity_score"] = volume_consistency
            else:
                features["liquidity_score"] = 0.0
        else:
            features["liquidity_score"] = 0.5
        
        return features
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol."""
        return self.latest_price.get(symbol)
    
    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict[str, float]]:
        """Get recent price history."""
        history = list(self.price_history[symbol])
        return history[-limit:] if len(history) > limit else history
    
    def _calculate_multi_timeframe_features(self, symbol: str) -> Dict[str, float]:
        """Calculate multi-timeframe volatility and volume ratios."""
        features = {}
        
        prices = [p["close"] for p in self.price_history[symbol]]
        volumes = list(self.volume_history[symbol])
        
        # Assume 1-minute candles
        timeframes = {
            "1h": 60,    # 60 candles = 1 hour
            "4h": 240,   # 240 candles = 4 hours
            "24h": 1440  # 1440 candles = 24 hours
        }
        
        # Calculate volatility for each timeframe
        for tf_name, tf_candles in timeframes.items():
            if len(prices) >= tf_candles:
                # Calculate returns
                tf_prices = prices[-tf_candles:]
                tf_returns = np.diff(tf_prices) / tf_prices[:-1]
                
                # Annualized volatility (assuming 525600 minutes per year)
                volatility = np.std(tf_returns) * np.sqrt(525600 / tf_candles)
                features[f"volatility_{tf_name}"] = volatility
            else:
                features[f"volatility_{tf_name}"] = 0.0
        
        # Calculate volume ratios
        if len(volumes) >= 60:  # At least 1 hour of data
            current_volume = volumes[-1]
            
            for tf_name, tf_candles in timeframes.items():
                if len(volumes) >= tf_candles:
                    avg_volume = np.mean(volumes[-tf_candles:])
                    if avg_volume > 0:
                        features[f"volume_ratio_{tf_name}"] = current_volume / avg_volume
                    else:
                        features[f"volume_ratio_{tf_name}"] = 1.0
                else:
                    features[f"volume_ratio_{tf_name}"] = 1.0
        else:
            # Default ratios
            for tf_name in timeframes:
                features[f"volume_ratio_{tf_name}"] = 1.0
        
        # Volatility rank (percentile in recent history)
        if len(prices) >= 100:
            recent_volatilities = []
            for i in range(20, len(prices)):
                window_returns = np.diff(prices[i-20:i]) / prices[i-20:i-1]
                vol = np.std(window_returns)
                recent_volatilities.append(vol)
            
            if recent_volatilities:
                current_vol = features.get("volatility_1h", 0.0)
                vol_percentile = np.sum(np.array(recent_volatilities) <= current_vol) / len(recent_volatilities)
                features["volatility_rank"] = vol_percentile
            else:
                features["volatility_rank"] = 0.5
        else:
            features["volatility_rank"] = 0.5
        
        return features
    
    def _calculate_time_features(self, timestamp: float) -> Dict[str, float]:
        """Calculate time-based features using sine/cosine encoding."""
        dt = datetime.fromtimestamp(timestamp)
        
        # Hour of day (0-23)
        hour = dt.hour
        hour_angle = 2 * np.pi * hour / 24
        features = {
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle)
        }
        
        # Day of week (0-6, Monday=0)
        day = dt.weekday()
        day_angle = 2 * np.pi * day / 7
        features["day_sin"] = np.sin(day_angle)
        features["day_cos"] = np.cos(day_angle)
        
        return features
    
    def _calculate_market_regime(self, symbol: str) -> float:
        """
        Calculate market regime (0-1 scale).
        0 = Bear/Ranging, 0.5 = Neutral, 1 = Bull/Trending
        """
        prices = [p["close"] for p in self.price_history[symbol]]
        
        if len(prices) < 50:
            return 0.5  # Neutral if not enough data
        
        # Multiple indicators for regime detection
        regime_scores = []
        
        # 1. Trend strength (MA alignment)
        if len(prices) >= 50:
            ma_5 = np.mean(prices[-5:])
            ma_20 = np.mean(prices[-20:])
            ma_50 = np.mean(prices[-50:])
            
            # Check if MAs are aligned (bullish or bearish)
            if ma_5 > ma_20 > ma_50:
                regime_scores.append(0.8)  # Bullish trend
            elif ma_5 < ma_20 < ma_50:
                regime_scores.append(0.2)  # Bearish trend
            else:
                regime_scores.append(0.5)  # Mixed/ranging
        
        # 2. Volatility regime
        if "volatility_1h" in self.latest_features.get(symbol, {}):
            vol = self.latest_features[symbol]["volatility_1h"]
            # Lower volatility = trending, higher volatility = ranging
            vol_score = 1.0 - min(vol / 0.5, 1.0)  # Normalize to 0-1
            regime_scores.append(vol_score)
        
        # 3. Price momentum
        if len(prices) >= 20:
            returns_20 = (prices[-1] - prices[-20]) / prices[-20]
            momentum_score = 0.5 + np.clip(returns_20 * 5, -0.5, 0.5)  # Map to 0-1
            regime_scores.append(momentum_score)
        
        # Average all regime scores
        if regime_scores:
            market_regime = np.mean(regime_scores)
        else:
            market_regime = 0.5
        
        return market_regime
    
    def update_orderbook_features(self, symbol: str, bid: float, ask: float, bid_size: float, ask_size: float) -> Dict[str, float]:
        """Update features from orderbook data."""
        features = {}
        
        # Spread in basis points
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
            spread = ask - bid
            features["spread_bps"] = (spread / mid_price) * 10000  # Convert to basis points
            
            # Bid-ask imbalance
            total_size = bid_size + ask_size
            if total_size > 0:
                features["bid_ask_imbalance"] = (bid_size - ask_size) / total_size
            else:
                features["bid_ask_imbalance"] = 0.0
        else:
            features["spread_bps"] = 0.0
            features["bid_ask_imbalance"] = 0.0
        
        return features
    
    def update_trade_features(self, symbol: str, buy_volume: float, sell_volume: float) -> Dict[str, float]:
        """Update features from trade data."""
        features = {}
        
        # Order flow imbalance
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            features["order_flow_imbalance"] = (buy_volume - sell_volume) / total_volume
        else:
            features["order_flow_imbalance"] = 0.0
        
        return features
    
    def update_liquidation_features(self, symbol: str, liquidation_volume: float, long_liquidations: float, short_liquidations: float) -> Dict[str, float]:
        """Update features from liquidation data."""
        features = {}
        
        # Liquidation pressure (normalized by recent volume)
        if symbol in self.latest_volume and self.latest_volume[symbol] > 0:
            features["liquidation_pressure"] = liquidation_volume / self.latest_volume[symbol]
        else:
            features["liquidation_pressure"] = 0.0
        
        # Long/short liquidation imbalance
        total_liq = long_liquidations + short_liquidations
        if total_liq > 0:
            features["liquidation_imbalance"] = (long_liquidations - short_liquidations) / total_liq
        else:
            features["liquidation_imbalance"] = 0.0
        
        return features
    
    def update_funding_features(self, symbol: str, funding_rate: float) -> Dict[str, float]:
        """Update features from funding data."""
        return {
            "funding_rate": funding_rate
        }
    
    def update_oi_features(self, symbol: str, open_interest: float, prev_open_interest: float) -> Dict[str, float]:
        """Update features from open interest data."""
        features = {}
        
        # Open interest change
        if prev_open_interest > 0:
            features["open_interest_change"] = (open_interest - prev_open_interest) / prev_open_interest
        else:
            features["open_interest_change"] = 0.0
        
        features["open_interest"] = open_interest
        
        return features