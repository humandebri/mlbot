"""
Micro-liquidity feature engine optimized for high-frequency trading signals.

Features computed:
- Bid/ask spread dynamics
- Order book depth ratios
- Liquidity concentration metrics
- Market impact estimators
"""

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


class MicroLiquidityEngine:
    """
    High-performance micro-liquidity feature engine.
    
    Optimized for:
    - Real-time orderbook analysis
    - Memory-efficient depth calculations  
    - Sub-millisecond feature computation
    - Cost-effective processing
    """
    
    def __init__(self, window_size: int = 300):
        """
        Initialize micro-liquidity engine.
        
        Args:
            window_size: Rolling window size in seconds
        """
        self.window_size = window_size
        
        # Per-symbol data storage (memory-optimized)
        self.orderbook_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
        self.depth_ratios: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
        self.impact_estimates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Feature cache for fast access
        self.latest_features: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.last_update: Dict[str, float] = defaultdict(float)
        
        # Performance optimization parameters
        self.min_update_interval = 0.1  # Don't update faster than 100ms
        self.depth_levels = 5  # Analyze top 5 price levels
        
    def process_orderbook(self, symbol: str, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Process orderbook update and compute micro-liquidity features.
        
        Args:
            symbol: Trading symbol
            data: Orderbook data from ingestor
            
        Returns:
            Dictionary of computed features
        """
        try:
            current_time = time.time()
            
            # Rate limiting for performance
            if current_time - self.last_update[symbol] < self.min_update_interval:
                return self.latest_features[symbol]
            
            # Extract orderbook components
            bids = data.get("bids_top5", [])
            asks = data.get("asks_top5", [])
            mid_price = data.get("mid_price", 0)
            spread = data.get("spread", 0)
            timestamp = data.get("timestamp", current_time)
            
            if not bids or not asks or mid_price <= 0:
                return {}
            
            # Store raw data
            self._store_orderbook_data(symbol, {
                "bids": bids,
                "asks": asks,
                "mid_price": mid_price,
                "spread": spread,
                "timestamp": timestamp
            })
            
            # Compute all micro-liquidity features
            features = {}
            
            # 1. Spread features
            features.update(self._compute_spread_features(symbol, spread, mid_price))
            
            # 2. Depth features  
            features.update(self._compute_depth_features(symbol, bids, asks))
            
            # 3. Liquidity concentration features
            features.update(self._compute_concentration_features(symbol, bids, asks))
            
            # 4. Market impact features
            features.update(self._compute_impact_features(symbol, bids, asks, mid_price))
            
            # 5. Orderbook asymmetry features
            features.update(self._compute_asymmetry_features(symbol, bids, asks))
            
            # Cache results
            self.latest_features[symbol] = features
            self.last_update[symbol] = current_time
            
            return features
            
        except Exception as e:
            logger.warning(f"Error computing micro-liquidity features for {symbol}", exception=e)
            return {}
    
    def _store_orderbook_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Store orderbook data efficiently."""
        # Store full orderbook snapshot
        self.orderbook_data[symbol].append(data)
        
        # Store derived metrics for rolling calculations
        self.spread_history[symbol].append(data["spread"])
        
        # Calculate and store depth ratio
        bid_depth = sum(float(bid[1]) for bid in data["bids"][:self.depth_levels])
        ask_depth = sum(float(ask[1]) for ask in data["asks"][:self.depth_levels])
        total_depth = bid_depth + ask_depth
        
        depth_ratio = bid_depth / total_depth if total_depth > 0 else 0.5
        self.depth_ratios[symbol].append(depth_ratio)
    
    def _compute_spread_features(self, symbol: str, spread: float, mid_price: float) -> Dict[str, float]:
        """Compute spread-related features."""
        features = {}
        
        # Current spread metrics
        features["spread_bps"] = (spread / mid_price) * 10000 if mid_price > 0 else 0
        features["spread_abs"] = spread
        
        # Rolling spread statistics
        if len(self.spread_history[symbol]) >= 10:
            spreads = list(self.spread_history[symbol])
            spread_array = np.array(spreads[-60:], dtype=np.float32)  # Last 60 updates (~1min)
            
            features["spread_mean_1m"] = float(np.mean(spread_array))
            features["spread_std_1m"] = float(np.std(spread_array))
            features["spread_min_1m"] = float(np.min(spread_array))
            features["spread_max_1m"] = float(np.max(spread_array))
            
            # Spread momentum
            if len(spread_array) >= 20:
                recent_mean = np.mean(spread_array[-10:])
                older_mean = np.mean(spread_array[-20:-10])
                features["spread_momentum"] = float((recent_mean - older_mean) / older_mean) if older_mean > 0 else 0
            
            # Spread percentile position
            features["spread_percentile"] = float(
                np.percentile(spread_array, 50) if len(spread_array) > 5 else 0
            )
        
        return features
    
    def _compute_depth_features(self, symbol: str, bids: List, asks: List) -> Dict[str, float]:
        """Compute order book depth features."""
        features = {}
        
        try:
            # Level-by-level depth analysis
            for i in range(min(len(bids), len(asks), self.depth_levels)):
                bid_price = float(bids[i][0])
                bid_size = float(bids[i][1])
                ask_price = float(asks[i][0])
                ask_size = float(asks[i][1])
                
                features[f"bid_size_L{i+1}"] = bid_size
                features[f"ask_size_L{i+1}"] = ask_size
                features[f"size_ratio_L{i+1}"] = bid_size / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0.5
            
            # Aggregate depth metrics
            total_bid_depth = sum(float(bid[1]) for bid in bids[:self.depth_levels])
            total_ask_depth = sum(float(ask[1]) for ask in asks[:self.depth_levels])
            total_depth = total_bid_depth + total_ask_depth
            
            features["total_depth"] = total_depth
            features["bid_depth"] = total_bid_depth
            features["ask_depth"] = total_ask_depth
            features["depth_imbalance"] = (total_bid_depth - total_ask_depth) / total_depth if total_depth > 0 else 0
            
            # Depth concentration (top level vs total)
            if bids and asks and total_depth > 0:
                top_level_depth = float(bids[0][1]) + float(asks[0][1])
                features["depth_concentration"] = top_level_depth / total_depth
            
            # Rolling depth statistics
            if len(self.depth_ratios[symbol]) >= 10:
                ratios = list(self.depth_ratios[symbol])
                ratio_array = np.array(ratios[-60:], dtype=np.float32)
                
                features["depth_ratio_mean"] = float(np.mean(ratio_array))
                features["depth_ratio_std"] = float(np.std(ratio_array))
                features["depth_ratio_trend"] = float(
                    np.corrcoef(np.arange(len(ratio_array)), ratio_array)[0, 1]
                    if len(ratio_array) > 5 else 0
                )
        
        except Exception as e:
            logger.warning(f"Error computing depth features for {symbol}", exception=e)
        
        return features
    
    def _compute_concentration_features(self, symbol: str, bids: List, asks: List) -> Dict[str, float]:
        """Compute liquidity concentration metrics."""
        features = {}
        
        try:
            if len(bids) < 2 or len(asks) < 2:
                return features
            
            # Bid side concentration
            bid_sizes = [float(bid[1]) for bid in bids[:self.depth_levels]]
            bid_total = sum(bid_sizes)
            
            if bid_total > 0:
                # Herfindahl index for bid concentration
                bid_shares = [size / bid_total for size in bid_sizes]
                features["bid_concentration_hhi"] = sum(share ** 2 for share in bid_shares)
                
                # Top level dominance
                features["bid_top_level_share"] = bid_shares[0] if bid_shares else 0
            
            # Ask side concentration
            ask_sizes = [float(ask[1]) for ask in asks[:self.depth_levels]]
            ask_total = sum(ask_sizes)
            
            if ask_total > 0:
                ask_shares = [size / ask_total for size in ask_sizes]
                features["ask_concentration_hhi"] = sum(share ** 2 for share in ask_shares)
                features["ask_top_level_share"] = ask_shares[0] if ask_shares else 0
            
            # Overall concentration asymmetry
            bid_hhi = features.get("bid_concentration_hhi", 0)
            ask_hhi = features.get("ask_concentration_hhi", 0)
            features["concentration_asymmetry"] = bid_hhi - ask_hhi
        
        except Exception as e:
            logger.warning(f"Error computing concentration features for {symbol}", exception=e)
        
        return features
    
    def _compute_impact_features(self, symbol: str, bids: List, asks: List, mid_price: float) -> Dict[str, float]:
        """Compute market impact estimation features."""
        features = {}
        
        try:
            if not bids or not asks or mid_price <= 0:
                return features
            
            # Define order sizes for impact estimation (in USD)
            impact_sizes = [1000, 5000, 10000, 50000]  # $1k, $5k, $10k, $50k
            
            for size_usd in impact_sizes:
                size_coins = size_usd / mid_price
                
                # Calculate bid impact (selling)
                bid_impact = self._calculate_side_impact(bids, size_coins, "bid")
                features[f"bid_impact_{size_usd//1000}k"] = bid_impact
                
                # Calculate ask impact (buying)  
                ask_impact = self._calculate_side_impact(asks, size_coins, "ask")
                features[f"ask_impact_{size_usd//1000}k"] = ask_impact
                
                # Average impact
                features[f"avg_impact_{size_usd//1000}k"] = (bid_impact + ask_impact) / 2
            
            # Store impact estimates for trend analysis
            impact_1k = features.get("avg_impact_1k", 0)
            self.impact_estimates[symbol].append(impact_1k)
            
            # Impact trend
            if len(self.impact_estimates[symbol]) >= 10:
                impacts = list(self.impact_estimates[symbol])
                impact_array = np.array(impacts[-30:], dtype=np.float32)  # Last 30 samples
                
                if len(impact_array) > 5:
                    features["impact_trend"] = float(
                        np.corrcoef(np.arange(len(impact_array)), impact_array)[0, 1]
                    )
                    features["impact_volatility"] = float(np.std(impact_array))
        
        except Exception as e:
            logger.warning(f"Error computing impact features for {symbol}", exception=e)
        
        return features
    
    def _calculate_side_impact(self, levels: List, target_size: float, side: str) -> float:
        """Calculate market impact for one side of the book."""
        try:
            remaining_size = target_size
            total_cost = 0.0
            
            for price_str, size_str in levels:
                price = float(price_str)
                available_size = float(size_str)
                
                if remaining_size <= 0:
                    break
                
                executed_size = min(remaining_size, available_size)
                total_cost += executed_size * price
                remaining_size -= executed_size
            
            if remaining_size > 0:
                # Not enough liquidity - penalize heavily
                return 1.0  # 100% impact
            
            # Calculate average execution price
            avg_price = total_cost / target_size if target_size > 0 else 0
            
            # Impact relative to best price
            best_price = float(levels[0][0]) if levels else 0
            if best_price > 0:
                if side == "bid":
                    # Selling: positive impact means worse price
                    impact = (best_price - avg_price) / best_price
                else:
                    # Buying: positive impact means worse price  
                    impact = (avg_price - best_price) / best_price
                
                return max(0, impact)  # Only positive impacts
            
        except Exception as e:
            logger.warning(f"Error calculating {side} impact", exception=e)
        
        return 0.0
    
    def _compute_asymmetry_features(self, symbol: str, bids: List, asks: List) -> Dict[str, float]:
        """Compute orderbook asymmetry features."""
        features = {}
        
        try:
            if len(bids) < 2 or len(asks) < 2:
                return features
            
            # Price level asymmetry
            bid_prices = [float(bid[0]) for bid in bids[:3]]
            ask_prices = [float(ask[0]) for ask in asks[:3]]
            
            if bid_prices and ask_prices:
                best_bid, best_ask = bid_prices[0], ask_prices[0]
                mid = (best_bid + best_ask) / 2
                
                # Distance asymmetry
                bid_distances = [(mid - price) / mid for price in bid_prices]
                ask_distances = [(price - mid) / mid for price in ask_prices]
                
                features["price_asymmetry"] = np.mean(ask_distances) - np.mean(bid_distances)
            
            # Size asymmetry by level
            for i in range(min(len(bids), len(asks), 3)):
                bid_size = float(bids[i][1])
                ask_size = float(asks[i][1])
                total_size = bid_size + ask_size
                
                if total_size > 0:
                    features[f"size_asymmetry_L{i+1}"] = (bid_size - ask_size) / total_size
            
            # Weighted average asymmetry
            total_bid_size = sum(float(bid[1]) for bid in bids[:3])
            total_ask_size = sum(float(ask[1]) for ask in asks[:3])
            total_size = total_bid_size + total_ask_size
            
            if total_size > 0:
                features["weighted_asymmetry"] = (total_bid_size - total_ask_size) / total_size
        
        except Exception as e:
            logger.warning(f"Error computing asymmetry features for {symbol}", exception=e)
        
        return features
    
    def get_feature_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of computed features for monitoring."""
        return {
            "symbol": symbol,
            "feature_count": len(self.latest_features.get(symbol, {})),
            "last_update": self.last_update.get(symbol, 0),
            "orderbook_samples": len(self.orderbook_data.get(symbol, [])),
            "spread_samples": len(self.spread_history.get(symbol, [])),
            "latest_features": list(self.latest_features.get(symbol, {}).keys())
        }