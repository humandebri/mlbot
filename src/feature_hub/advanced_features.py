"""
Advanced market microstructure features.

This module implements sophisticated features including:
- Open Interest (OI) dynamics
- Taker/Maker flow analysis
- Order flow imbalance
- Market impact estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque
import asyncio
from datetime import datetime, timedelta

from ..common.logging import get_logger

logger = get_logger(__name__)


class OpenInterestFeatures:
    """
    Open Interest (OI) based features for capturing market positioning.
    
    OI represents the total number of outstanding derivative contracts
    and is a key indicator of market sentiment and potential squeeze scenarios.
    """
    
    def __init__(self, windows: List[int] = [60, 300, 900, 3600]):
        self.windows = windows
        self.oi_history = {}  # Symbol -> deque of (timestamp, oi)
        self.max_history = max(windows) * 2
        
    def update(self, symbol: str, oi: float, timestamp: float) -> None:
        """Update OI history for a symbol."""
        if symbol not in self.oi_history:
            self.oi_history[symbol] = deque(maxlen=self.max_history)
            
        self.oi_history[symbol].append((timestamp, oi))
        
    def calculate_features(self, symbol: str) -> Dict[str, float]:
        """Calculate OI-based features."""
        features = {}
        
        if symbol not in self.oi_history or len(self.oi_history[symbol]) < 2:
            return self._get_default_features()
            
        history = list(self.oi_history[symbol])
        current_time = history[-1][0]
        current_oi = history[-1][1]
        
        # OI change rates over different windows
        for window in self.windows:
            cutoff_time = current_time - window
            
            # Find OI at window start
            window_start_oi = None
            for ts, oi in history:
                if ts >= cutoff_time:
                    window_start_oi = oi
                    break
                    
            if window_start_oi is not None and window_start_oi > 0:
                # Percentage change
                oi_change_pct = (current_oi - window_start_oi) / window_start_oi
                features[f'oi_change_pct_{window}s'] = oi_change_pct
                
                # Absolute change normalized by average
                oi_change_abs = current_oi - window_start_oi
                avg_oi = (current_oi + window_start_oi) / 2
                features[f'oi_change_norm_{window}s'] = oi_change_abs / avg_oi
                
                # Rate of change (per minute)
                time_diff = current_time - cutoff_time
                if time_diff > 0:
                    features[f'oi_velocity_{window}s'] = oi_change_pct / (time_diff / 60)
            else:
                features[f'oi_change_pct_{window}s'] = 0.0
                features[f'oi_change_norm_{window}s'] = 0.0
                features[f'oi_velocity_{window}s'] = 0.0
        
        # OI acceleration (change in velocity)
        if len(history) >= 3:
            # Calculate recent velocities
            velocities = []
            for i in range(1, min(4, len(history))):
                if history[i][1] > 0 and history[i-1][1] > 0:
                    time_diff = history[i][0] - history[i-1][0]
                    if time_diff > 0:
                        oi_diff = (history[i][1] - history[i-1][1]) / history[i-1][1]
                        velocity = oi_diff / (time_diff / 60)
                        velocities.append(velocity)
                        
            if len(velocities) >= 2:
                features['oi_acceleration'] = velocities[-1] - velocities[-2]
            else:
                features['oi_acceleration'] = 0.0
        else:
            features['oi_acceleration'] = 0.0
            
        # OI relative to recent average
        recent_ois = [oi for _, oi in history[-20:]]  # Last 20 data points
        if recent_ois:
            avg_recent_oi = np.mean(recent_ois)
            std_recent_oi = np.std(recent_ois) if len(recent_ois) > 1 else 1.0
            
            features['oi_zscore'] = (current_oi - avg_recent_oi) / (std_recent_oi + 1e-8)
            features['oi_relative'] = current_oi / (avg_recent_oi + 1e-8)
        else:
            features['oi_zscore'] = 0.0
            features['oi_relative'] = 1.0
            
        return features
        
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when insufficient data."""
        features = {}
        for window in self.windows:
            features[f'oi_change_pct_{window}s'] = 0.0
            features[f'oi_change_norm_{window}s'] = 0.0
            features[f'oi_velocity_{window}s'] = 0.0
        features['oi_acceleration'] = 0.0
        features['oi_zscore'] = 0.0
        features['oi_relative'] = 1.0
        return features


class TakerMakerFlowFeatures:
    """
    Taker/Maker flow analysis for understanding aggressive vs passive trading.
    
    Taker orders consume liquidity (market orders), while maker orders
    provide liquidity (limit orders). The ratio indicates market aggression.
    """
    
    def __init__(self, windows: List[int] = [60, 300, 900]):
        self.windows = windows
        self.trade_history = {}  # Symbol -> deque of (timestamp, side, is_taker, size, price)
        self.max_history = max(windows) * 2 * 100  # Assuming ~100 trades per second max
        
    def update_trade(
        self, 
        symbol: str, 
        side: str,  # 'buy' or 'sell'
        is_taker: bool,
        size: float,
        price: float,
        timestamp: float
    ) -> None:
        """Update trade history."""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=self.max_history)
            
        self.trade_history[symbol].append((timestamp, side, is_taker, size, price))
        
    def calculate_features(self, symbol: str) -> Dict[str, float]:
        """Calculate taker/maker flow features."""
        features = {}
        
        if symbol not in self.trade_history or len(self.trade_history[symbol]) == 0:
            return self._get_default_features()
            
        history = list(self.trade_history[symbol])
        current_time = history[-1][0]
        
        for window in self.windows:
            cutoff_time = current_time - window
            
            # Filter trades in window
            window_trades = [(side, is_taker, size, price) 
                            for ts, side, is_taker, size, price in history 
                            if ts >= cutoff_time]
            
            if not window_trades:
                self._set_window_features(features, window, {})
                continue
                
            # Calculate taker/maker volumes
            taker_buy_volume = sum(size for side, is_taker, size, _ in window_trades 
                                  if side == 'buy' and is_taker)
            taker_sell_volume = sum(size for side, is_taker, size, _ in window_trades 
                                   if side == 'sell' and is_taker)
            maker_buy_volume = sum(size for side, is_taker, size, _ in window_trades 
                                  if side == 'buy' and not is_taker)
            maker_sell_volume = sum(size for side, is_taker, size, _ in window_trades 
                                   if side == 'sell' and not is_taker)
            
            total_taker = taker_buy_volume + taker_sell_volume
            total_maker = maker_buy_volume + maker_sell_volume
            total_volume = total_taker + total_maker
            
            # Taker/Maker ratio
            features[f'taker_maker_ratio_{window}s'] = (
                total_taker / (total_maker + 1e-8)
            )
            
            # Taker buy/sell imbalance
            taker_imbalance = (taker_buy_volume - taker_sell_volume) / (
                taker_buy_volume + taker_sell_volume + 1e-8
            )
            features[f'taker_buy_sell_imbalance_{window}s'] = taker_imbalance
            
            # Maker buy/sell imbalance
            maker_imbalance = (maker_buy_volume - maker_sell_volume) / (
                maker_buy_volume + maker_sell_volume + 1e-8
            )
            features[f'maker_buy_sell_imbalance_{window}s'] = maker_imbalance
            
            # Aggressive buying/selling pressure
            features[f'aggressive_buy_ratio_{window}s'] = (
                taker_buy_volume / (total_volume + 1e-8)
            )
            features[f'aggressive_sell_ratio_{window}s'] = (
                taker_sell_volume / (total_volume + 1e-8)
            )
            
            # Volume-weighted average trade size
            avg_taker_size = total_taker / (len([1 for _, is_taker, _, _ in window_trades if is_taker]) + 1e-8)
            avg_maker_size = total_maker / (len([1 for _, is_taker, _, _ in window_trades if not is_taker]) + 1e-8)
            features[f'avg_taker_trade_size_{window}s'] = avg_taker_size
            features[f'avg_maker_trade_size_{window}s'] = avg_maker_size
            
            # Large trade detection (trades > 2x average)
            large_threshold = 2 * (total_volume / len(window_trades))
            large_taker_volume = sum(size for _, is_taker, size, _ in window_trades 
                                    if is_taker and size > large_threshold)
            features[f'large_taker_volume_ratio_{window}s'] = (
                large_taker_volume / (total_volume + 1e-8)
            )
            
        return features
        
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when insufficient data."""
        features = {}
        for window in self.windows:
            features[f'taker_maker_ratio_{window}s'] = 1.0
            features[f'taker_buy_sell_imbalance_{window}s'] = 0.0
            features[f'maker_buy_sell_imbalance_{window}s'] = 0.0
            features[f'aggressive_buy_ratio_{window}s'] = 0.25
            features[f'aggressive_sell_ratio_{window}s'] = 0.25
            features[f'avg_taker_trade_size_{window}s'] = 0.0
            features[f'avg_maker_trade_size_{window}s'] = 0.0
            features[f'large_taker_volume_ratio_{window}s'] = 0.0
        return features
        
    def _set_window_features(self, features: dict, window: int, values: dict) -> None:
        """Set features for a specific window with defaults."""
        defaults = {
            'taker_maker_ratio': 1.0,
            'taker_buy_sell_imbalance': 0.0,
            'maker_buy_sell_imbalance': 0.0,
            'aggressive_buy_ratio': 0.25,
            'aggressive_sell_ratio': 0.25,
            'avg_taker_trade_size': 0.0,
            'avg_maker_trade_size': 0.0,
            'large_taker_volume_ratio': 0.0
        }
        
        for key, default_value in defaults.items():
            features[f'{key}_{window}s'] = values.get(key, default_value)


class OrderFlowImbalance:
    """
    Order flow imbalance features based on order book and trade data.
    
    Captures the imbalance between buying and selling pressure,
    which can predict short-term price movements.
    """
    
    def __init__(self):
        self.last_orderbook = {}
        self.flow_history = {}  # Symbol -> deque of (timestamp, imbalance)
        
    def update_orderbook(self, symbol: str, bids: dict, asks: dict) -> None:
        """Update order book snapshot."""
        self.last_orderbook[symbol] = {
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now().timestamp()
        }
        
    def calculate_features(self, symbol: str, trades: List[dict]) -> Dict[str, float]:
        """Calculate order flow imbalance features."""
        features = {}
        
        if symbol not in self.last_orderbook:
            return self._get_default_features()
            
        ob = self.last_orderbook[symbol]
        bids = ob['bids']
        asks = ob['asks']
        
        # Volume-weighted bid/ask imbalance at different levels
        levels = [1, 5, 10, 25]  # Top N levels
        
        for n_levels in levels:
            bid_prices = sorted(bids.keys(), reverse=True)[:n_levels]
            ask_prices = sorted(asks.keys())[:n_levels]
            
            bid_volume = sum(bids.get(p, 0) for p in bid_prices)
            ask_volume = sum(asks.get(p, 0) for p in ask_prices)
            
            # Basic imbalance
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)
            features[f'volume_imbalance_{n_levels}'] = imbalance
            
            # Weighted by distance from mid
            if bid_prices and ask_prices:
                mid_price = (bid_prices[0] + ask_prices[0]) / 2
                
                weighted_bid_volume = sum(
                    bids.get(p, 0) * np.exp(-abs(p - mid_price) / mid_price)
                    for p in bid_prices
                )
                weighted_ask_volume = sum(
                    asks.get(p, 0) * np.exp(-abs(p - mid_price) / mid_price)
                    for p in ask_prices
                )
                
                weighted_imbalance = (weighted_bid_volume - weighted_ask_volume) / (
                    weighted_bid_volume + weighted_ask_volume + 1e-8
                )
                features[f'weighted_imbalance_{n_levels}'] = weighted_imbalance
        
        # Microprice (volume-weighted mid price)
        if bids and asks:
            best_bid = max(bids.keys())
            best_ask = min(asks.keys())
            best_bid_size = bids[best_bid]
            best_ask_size = asks[best_ask]
            
            microprice = (best_bid * best_ask_size + best_ask * best_bid_size) / (
                best_bid_size + best_ask_size + 1e-8
            )
            mid_price = (best_bid + best_ask) / 2
            
            # Microprice deviation from mid
            features['microprice_deviation'] = (microprice - mid_price) / mid_price
        else:
            features['microprice_deviation'] = 0.0
            
        # Trade flow vs book imbalance
        if trades:
            recent_buy_volume = sum(t['size'] for t in trades if t['side'] == 'buy')
            recent_sell_volume = sum(t['size'] for t in trades if t['side'] == 'sell')
            trade_imbalance = (recent_buy_volume - recent_sell_volume) / (
                recent_buy_volume + recent_sell_volume + 1e-8
            )
            
            # Compare trade flow with book imbalance
            book_imbalance = features.get('volume_imbalance_5', 0)
            features['flow_book_divergence'] = trade_imbalance - book_imbalance
        else:
            features['flow_book_divergence'] = 0.0
            
        return features
        
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when insufficient data."""
        features = {}
        levels = [1, 5, 10, 25]
        for n_levels in levels:
            features[f'volume_imbalance_{n_levels}'] = 0.0
            features[f'weighted_imbalance_{n_levels}'] = 0.0
        features['microprice_deviation'] = 0.0
        features['flow_book_divergence'] = 0.0
        return features


class AdvancedFeatureAggregator:
    """
    Aggregates all advanced features into a single interface.
    """
    
    def __init__(self):
        self.oi_features = OpenInterestFeatures()
        self.taker_maker_features = TakerMakerFlowFeatures()
        self.flow_imbalance = OrderFlowImbalance()
        
    async def update_oi(self, symbol: str, oi: float) -> None:
        """Update open interest data."""
        timestamp = datetime.now().timestamp()
        self.oi_features.update(symbol, oi, timestamp)
        
    async def update_trade(
        self,
        symbol: str,
        side: str,
        is_taker: bool,
        size: float,
        price: float
    ) -> None:
        """Update trade data."""
        timestamp = datetime.now().timestamp()
        self.taker_maker_features.update_trade(
            symbol, side, is_taker, size, price, timestamp
        )
        
    async def update_orderbook(
        self,
        symbol: str,
        bids: dict,
        asks: dict
    ) -> None:
        """Update order book data."""
        self.flow_imbalance.update_orderbook(symbol, bids, asks)
        
    async def calculate_all_features(
        self,
        symbol: str,
        recent_trades: Optional[List[dict]] = None
    ) -> Dict[str, float]:
        """Calculate all advanced features."""
        features = {}
        
        # OI features
        oi_feats = self.oi_features.calculate_features(symbol)
        features.update(oi_feats)
        
        # Taker/Maker features
        tm_feats = self.taker_maker_features.calculate_features(symbol)
        features.update(tm_feats)
        
        # Order flow imbalance
        flow_feats = self.flow_imbalance.calculate_features(
            symbol, recent_trades or []
        )
        features.update(flow_feats)
        
        logger.debug(f"Calculated {len(features)} advanced features for {symbol}")
        
        return features