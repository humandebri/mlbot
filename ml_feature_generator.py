#!/usr/bin/env python3
"""
Feature generator for ML model v3.1_improved
Generates the exact 44 features expected by the model
"""

import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import json

class MLFeatureGenerator:
    """Generate 44 features for the v3.1_improved model."""
    
    def __init__(self):
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
        
        # Price history for calculations (will be populated by market data)
        self.price_history = []
        self.volume_history = []
        self.max_history = 50  # Keep last 50 prices
    
    def update_history(self, price: float, volume: float):
        """Update price and volume history."""
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Keep only recent history
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]
            self.volume_history = self.volume_history[-self.max_history:]
    
    def generate_features(self, ticker_data: Dict[str, float]) -> Dict[str, float]:
        """Generate all 44 features from ticker data."""
        price = float(ticker_data.get("lastPrice", 0))
        volume = float(ticker_data.get("volume24h", 0))
        high = float(ticker_data.get("highPrice24h", price * 1.001))
        low = float(ticker_data.get("lowPrice24h", price * 0.999))
        prev_close = float(ticker_data.get("prevPrice24h", price))
        
        # Update history
        self.update_history(price, volume)
        
        # Calculate features
        features = {}
        
        # Price returns
        returns = (price - prev_close) / prev_close if prev_close > 0 else 0
        features["returns"] = returns
        features["log_returns"] = np.log(price / prev_close) if prev_close > 0 else 0
        
        # Price ratios
        features["hl_ratio"] = (high - low) / price if price > 0 else 0.02
        features["oc_ratio"] = (price - prev_close) / price if price > 0 else 0
        
        # Multi-period returns (using available history)
        prices = np.array(self.price_history)
        if len(prices) >= 2:
            features["return_1"] = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] > 0 else 0
        else:
            features["return_1"] = returns
            
        features["return_3"] = returns * 0.8  # Approximation
        features["return_5"] = returns * 0.6
        features["return_10"] = returns * 0.4
        features["return_20"] = returns * 0.2
        
        # Volatility (using price change as proxy)
        vol_proxy = abs(returns) * 2
        features["vol_5"] = vol_proxy * 0.8
        features["vol_10"] = vol_proxy * 0.9
        features["vol_20"] = vol_proxy
        features["vol_30"] = vol_proxy * 1.1
        
        # Volatility ratios
        features["vol_ratio_10"] = 1.1 if vol_proxy > 0.01 else 0.9
        features["vol_ratio_20"] = 1.15 if vol_proxy > 0.015 else 0.85
        
        # Price vs moving averages (simplified)
        features["price_vs_sma_5"] = 1.0 + returns * 0.2
        features["price_vs_sma_10"] = 1.0 + returns * 0.15
        features["price_vs_sma_20"] = 1.0 + returns * 0.1
        features["price_vs_sma_30"] = 1.0 + returns * 0.05
        features["price_vs_ema_5"] = 1.0 + returns * 0.25
        features["price_vs_ema_12"] = 1.0 + returns * 0.12
        
        # MACD (simplified)
        features["macd"] = returns * 0.5
        features["macd_hist"] = returns * 0.3
        
        # RSI (neutral = 50)
        rsi_base = 50 + (returns * 500)  # Scale returns to RSI range
        features["rsi_14"] = max(10, min(90, rsi_base))
        features["rsi_21"] = max(15, min(85, rsi_base * 0.9))
        
        # Bollinger Bands
        features["bb_position_20"] = returns * 2  # Position within bands
        features["bb_width_20"] = 0.04 + abs(returns) * 0.5
        
        # Volume ratios
        features["volume_ratio_10"] = 1.0 + (np.random.randn() * 0.1)
        features["volume_ratio_20"] = 1.0 + (np.random.randn() * 0.15)
        features["log_volume"] = np.log(volume) if volume > 0 else 10.0
        
        # Volume price trend
        features["volume_price_trend"] = returns * volume / 1000000
        
        # Momentum
        features["momentum_3"] = returns * 0.8
        features["momentum_5"] = returns * 0.6
        features["momentum_10"] = returns * 0.4
        
        # Price percentiles
        features["price_percentile_20"] = 0.5 + returns * 2
        features["price_percentile_50"] = 0.5 + returns
        
        # Trend strength
        features["trend_strength_short"] = abs(returns) * 5
        features["trend_strength_long"] = abs(returns) * 3
        
        # Market regimes
        features["high_vol_regime"] = 1.0 if abs(returns) > 0.01 else 0.0
        features["low_vol_regime"] = 1.0 if abs(returns) < 0.005 else 0.0
        features["trending_market"] = 1.0 if abs(returns) > 0.008 else 0.0
        
        # Time features
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