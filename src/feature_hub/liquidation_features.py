"""
Advanced liquidation features engine for wick-hunting strategy.

Enhanced features:
- Multi-timeframe liquidation cascade analysis
- Directional liquidation pressure
- Market structure breakdowns  
- Liquidation-price correlation patterns
- Cross-symbol liquidation spillover effects
"""

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


class LiquidationFeatureEngine:
    """
    Advanced liquidation analysis engine for ML feature generation.
    
    Optimized for:
    - Multi-timeframe cascade detection
    - Cross-asset liquidation analysis
    - Memory-efficient rolling calculations
    - High-frequency signal extraction
    """
    
    def __init__(self, max_history: int = 1800):  # 30 minutes at 1-second resolution
        """
        Initialize liquidation feature engine.
        
        Args:
            max_history: Maximum liquidation events to keep per symbol
        """
        self.max_history = max_history
        
        # Per-symbol liquidation data
        self.liquidation_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_history))
        self.volume_windows: Dict[str, Dict[str, deque]] = defaultdict(lambda: {
            "5s": deque(maxlen=5),
            "30s": deque(maxlen=30),
            "1m": deque(maxlen=60),
            "5m": deque(maxlen=300),
        })
        
        # Price correlation tracking
        self.price_at_liquidation: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.liquidation_impacts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        
        # Cross-symbol analysis
        self.symbol_correlations: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.spillover_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Feature cache
        self.latest_features: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.last_update: Dict[str, float] = defaultdict(float)
        
        # Performance optimization
        self.min_update_interval = 1.0  # Update every second max
        self.cascade_window = 60  # Look for cascades within 60 seconds
        
        # Liquidation thresholds for significance
        self.size_percentiles: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.update_percentiles_interval = 300  # Update every 5 minutes
        self.last_percentile_update: Dict[str, float] = defaultdict(float)
    
    def process_liquidation(self, symbol: str, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Process liquidation event and compute advanced features.
        
        Args:
            symbol: Trading symbol
            data: Liquidation data from ingestor
            
        Returns:
            Dictionary of computed features
        """
        try:
            current_time = time.time()
            
            # Rate limiting
            if current_time - self.last_update[symbol] < self.min_update_interval:
                return self.latest_features[symbol]
            
            # Extract liquidation data
            liquidation = {
                "timestamp": data.get("timestamp", current_time),
                "symbol": symbol,
                "side": data.get("side", ""),
                "size": float(data.get("size", 0)),
                "price": float(data.get("price", 0)),
                "liquidation_time": data.get("liquidation_time", current_time),
                "spike_detected": data.get("spike_detected", False),
                "spike_severity": float(data.get("spike_severity", 0)),
                "spike_type": data.get("spike_type", ""),
                "liq_metrics": data.get("liq_metrics", {})
            }
            
            if liquidation["size"] <= 0 or liquidation["price"] <= 0:
                return {}
            
            # Store liquidation event
            self.liquidation_events[symbol].append(liquidation)
            
            # Update volume windows
            self._update_volume_windows(symbol, liquidation)
            
            # Update size percentiles periodically
            self._update_size_percentiles(symbol, current_time)
            
            # Compute advanced features
            features = {}
            
            # 1. Enhanced cascade analysis
            features.update(self._compute_cascade_features(symbol, liquidation))
            
            # 2. Directional pressure analysis
            features.update(self._compute_directional_features(symbol))
            
            # 3. Market structure breakdown detection
            features.update(self._compute_structure_features(symbol, liquidation))
            
            # 4. Cross-timeframe analysis
            features.update(self._compute_timeframe_features(symbol))
            
            # 5. Size-based segmentation
            features.update(self._compute_size_features(symbol, liquidation))
            
            # 6. Price impact correlation
            features.update(self._compute_price_correlation_features(symbol, liquidation))
            
            # 7. Cross-symbol spillover (if multiple symbols tracked)
            if len(self.liquidation_events) > 1:
                features.update(self._compute_spillover_features(symbol, liquidation))
            
            # Cache results
            self.latest_features[symbol] = features
            self.last_update[symbol] = current_time
            
            return features
            
        except Exception as e:
            logger.warning(f"Error computing liquidation features for {symbol}", exception=e)
            return {}
    
    def _update_volume_windows(self, symbol: str, liquidation: Dict[str, Any]) -> None:
        """Update rolling volume windows efficiently."""
        current_time = liquidation["timestamp"]
        size = liquidation["size"]
        
        # Add to all windows
        for window_name, window_deque in self.volume_windows[symbol].items():
            window_deque.append({
                "timestamp": current_time,
                "size": size,
                "side": liquidation["side"],
                "price": liquidation["price"]
            })
    
    def _update_size_percentiles(self, symbol: str, current_time: float) -> None:
        """Update size percentiles for threshold analysis."""
        if current_time - self.last_percentile_update[symbol] < self.update_percentiles_interval:
            return
        
        events = list(self.liquidation_events[symbol])
        if len(events) < 50:  # Need minimum data
            return
        
        sizes = [event["size"] for event in events[-500:]]  # Last 500 events
        size_array = np.array(sizes, dtype=np.float32)
        
        self.size_percentiles[symbol] = {
            "p50": float(np.percentile(size_array, 50)),
            "p75": float(np.percentile(size_array, 75)),
            "p90": float(np.percentile(size_array, 90)),
            "p95": float(np.percentile(size_array, 95)),
            "p99": float(np.percentile(size_array, 99)),
        }
        
        self.last_percentile_update[symbol] = current_time
    
    def _compute_cascade_features(self, symbol: str, current_liq: Dict[str, Any]) -> Dict[str, float]:
        """Compute liquidation cascade analysis features."""
        features = {}
        
        try:
            events = list(self.liquidation_events[symbol])
            if len(events) < 5:
                return features
            
            current_time = current_liq["timestamp"]
            
            # Define cascade time windows
            cascade_windows = [5, 15, 30, 60]  # seconds
            
            for window_sec in cascade_windows:
                window_start = current_time - window_sec
                window_events = [e for e in events if e["timestamp"] >= window_start]
                
                if not window_events:
                    continue
                
                # Cascade intensity metrics
                total_volume = sum(e["size"] for e in window_events)
                event_count = len(window_events)
                
                features[f"cascade_volume_{window_sec}s"] = total_volume
                features[f"cascade_count_{window_sec}s"] = event_count
                features[f"cascade_intensity_{window_sec}s"] = total_volume / window_sec  # Volume per second
                
                # Cascade acceleration (recent vs earlier events)
                if window_sec >= 30 and len(window_events) >= 4:
                    half_window = window_sec // 2
                    mid_time = current_time - half_window
                    
                    recent_events = [e for e in window_events if e["timestamp"] >= mid_time]
                    earlier_events = [e for e in window_events if e["timestamp"] < mid_time]
                    
                    if recent_events and earlier_events:
                        recent_volume = sum(e["size"] for e in recent_events)
                        earlier_volume = sum(e["size"] for e in earlier_events)
                        
                        # Normalize by time period
                        recent_rate = recent_volume / half_window
                        earlier_rate = earlier_volume / half_window
                        
                        if earlier_rate > 0:
                            features[f"cascade_acceleration_{window_sec}s"] = recent_rate / earlier_rate
                
                # Side-specific cascade analysis
                buy_liquidations = [e for e in window_events if e["side"].lower() == "buy"]
                sell_liquidations = [e for e in window_events if e["side"].lower() == "sell"]
                
                buy_volume = sum(e["size"] for e in buy_liquidations)
                sell_volume = sum(e["size"] for e in sell_liquidations)
                total_volume = buy_volume + sell_volume
                
                if total_volume > 0:
                    features[f"cascade_sell_ratio_{window_sec}s"] = sell_volume / total_volume
                    features[f"cascade_directional_imbalance_{window_sec}s"] = (sell_volume - buy_volume) / total_volume
                
                # Price range during cascade
                if window_events:
                    prices = [e["price"] for e in window_events]
                    price_range = max(prices) - min(prices)
                    avg_price = sum(prices) / len(prices)
                    features[f"cascade_price_range_{window_sec}s"] = price_range / avg_price if avg_price > 0 else 0
            
            # Multi-timeframe cascade correlation
            if "cascade_volume_5s" in features and "cascade_volume_60s" in features:
                vol_5s = features["cascade_volume_5s"]
                vol_60s = features["cascade_volume_60s"]
                features["cascade_concentration"] = vol_5s / vol_60s if vol_60s > 0 else 0
        
        except Exception as e:
            logger.warning(f"Error computing cascade features for {symbol}", exception=e)
        
        return features
    
    def _compute_directional_features(self, symbol: str) -> Dict[str, float]:
        """Compute directional liquidation pressure features."""
        features = {}
        
        try:
            # Analyze different timeframes
            for window_name, window_deque in self.volume_windows[symbol].items():
                window_events = list(window_deque)
                if len(window_events) < 2:
                    continue
                
                # Directional analysis
                buy_events = [e for e in window_events if e["side"].lower() == "buy"]
                sell_events = [e for e in window_events if e["side"].lower() == "sell"]
                
                buy_volume = sum(e["size"] for e in buy_events)
                sell_volume = sum(e["size"] for e in sell_events)
                total_volume = buy_volume + sell_volume
                
                if total_volume > 0:
                    # Basic directional metrics
                    features[f"dir_sell_dominance_{window_name}"] = sell_volume / total_volume
                    features[f"dir_imbalance_{window_name}"] = (sell_volume - buy_volume) / total_volume
                    
                    # Directional momentum (weighted by recency)
                    if len(window_events) >= 5:
                        weights = np.linspace(0.5, 1.0, len(window_events))  # More weight to recent
                        
                        sell_weighted = sum(
                            w * e["size"] for w, e in zip(weights, window_events) 
                            if e["side"].lower() == "sell"
                        )
                        buy_weighted = sum(
                            w * e["size"] for w, e in zip(weights, window_events)
                            if e["side"].lower() == "buy"
                        )
                        total_weighted = sell_weighted + buy_weighted
                        
                        if total_weighted > 0:
                            features[f"dir_momentum_{window_name}"] = (sell_weighted - buy_weighted) / total_weighted
                
                # Directional clustering (are liquidations of same side clustered?)
                if len(window_events) >= 10:
                    sides = [1 if e["side"].lower() == "sell" else -1 for e in window_events]
                    
                    # Calculate runs of same direction
                    runs = []
                    current_run = 1
                    
                    for i in range(1, len(sides)):
                        if sides[i] == sides[i-1]:
                            current_run += 1
                        else:
                            runs.append(current_run)
                            current_run = 1
                    runs.append(current_run)
                    
                    if runs:
                        avg_run_length = np.mean(runs)
                        max_run_length = max(runs)
                        features[f"dir_clustering_{window_name}"] = avg_run_length
                        features[f"dir_max_run_{window_name}"] = max_run_length
        
        except Exception as e:
            logger.warning(f"Error computing directional features for {symbol}", exception=e)
        
        return features
    
    def _compute_structure_features(self, symbol: str, current_liq: Dict[str, Any]) -> Dict[str, float]:
        """Compute market structure breakdown detection features."""
        features = {}
        
        try:
            events = list(self.liquidation_events[symbol])
            if len(events) < 20:
                return features
            
            current_time = current_liq["timestamp"]
            
            # Analyze liquidation size distribution over time
            recent_events = [e for e in events if current_time - e["timestamp"] <= 300]  # Last 5 minutes
            older_events = [e for e in events if 300 < current_time - e["timestamp"] <= 600]  # 5-10 minutes ago
            
            if len(recent_events) >= 10 and len(older_events) >= 10:
                recent_sizes = [e["size"] for e in recent_events]
                older_sizes = [e["size"] for e in older_events]
                
                # Size distribution shift analysis
                recent_large = sum(1 for size in recent_sizes if size > np.percentile(older_sizes, 90))
                recent_total = len(recent_sizes)
                
                features["structure_large_liq_ratio"] = recent_large / recent_total if recent_total > 0 else 0
                
                # Average size trend
                recent_avg = np.mean(recent_sizes)
                older_avg = np.mean(older_sizes)
                features["structure_size_trend"] = recent_avg / older_avg if older_avg > 0 else 1.0
                
                # Frequency change
                recent_freq = len(recent_events) / 300  # Events per second
                older_freq = len(older_events) / 300
                features["structure_freq_change"] = recent_freq / older_freq if older_freq > 0 else 1.0
            
            # Spike pattern analysis
            spike_events = [e for e in events if e.get("spike_detected", False)]
            if len(spike_events) >= 3:
                recent_spikes = [e for e in spike_events if current_time - e["timestamp"] <= 180]  # Last 3 minutes
                
                if recent_spikes:
                    # Spike frequency acceleration
                    spike_times = [e["timestamp"] for e in recent_spikes]
                    if len(spike_times) >= 2:
                        time_intervals = np.diff(spike_times)
                        features["structure_spike_frequency"] = len(recent_spikes) / 180  # Spikes per second
                        
                        if len(time_intervals) >= 2:
                            # Are spikes getting closer together?
                            early_intervals = time_intervals[:len(time_intervals)//2]
                            late_intervals = time_intervals[len(time_intervals)//2:]
                            
                            if early_intervals.size > 0 and late_intervals.size > 0:
                                early_avg = np.mean(early_intervals)
                                late_avg = np.mean(late_intervals)
                                features["structure_spike_acceleration"] = early_avg / late_avg if late_avg > 0 else 1.0
                    
                    # Spike severity escalation
                    recent_severities = [e.get("spike_severity", 0) for e in recent_spikes]
                    if len(recent_severities) >= 3:
                        # Is severity increasing?
                        x = np.arange(len(recent_severities))
                        correlation = np.corrcoef(x, recent_severities)[0, 1] if len(recent_severities) > 2 else 0
                        features["structure_severity_trend"] = correlation
            
            # Price level breakdown analysis
            if len(events) >= 50:
                # Group liquidations by price levels
                prices = [e["price"] for e in events[-50:]]  # Last 50 events
                price_array = np.array(prices)
                
                # Find support/resistance breaks
                percentiles = [10, 25, 75, 90]
                for p in percentiles:
                    level = np.percentile(price_array[:-10], p)  # Historical level
                    recent_prices = price_array[-10:]  # Recent prices
                    
                    if p <= 50:  # Support levels
                        breaks = sum(1 for price in recent_prices if price < level)
                    else:  # Resistance levels
                        breaks = sum(1 for price in recent_prices if price > level)
                    
                    features[f"structure_level_break_p{p}"] = breaks / len(recent_prices)
        
        except Exception as e:
            logger.warning(f"Error computing structure features for {symbol}", exception=e)
        
        return features
    
    def _compute_timeframe_features(self, symbol: str) -> Dict[str, float]:
        """Compute cross-timeframe liquidation analysis."""
        features = {}
        
        try:
            # Compare activity across different timeframes
            timeframes = {
                "micro": 5,    # 5 seconds
                "short": 30,   # 30 seconds  
                "medium": 180, # 3 minutes
                "long": 900    # 15 minutes
            }
            
            current_time = time.time()
            events = list(self.liquidation_events[symbol])
            
            volumes = {}
            counts = {}
            
            for tf_name, tf_seconds in timeframes.items():
                window_start = current_time - tf_seconds
                window_events = [e for e in events if e["timestamp"] >= window_start]
                
                volumes[tf_name] = sum(e["size"] for e in window_events)
                counts[tf_name] = len(window_events)
            
            # Cross-timeframe ratios
            if volumes["long"] > 0:
                features["tf_micro_concentration"] = volumes["micro"] / volumes["long"]
                features["tf_short_concentration"] = volumes["short"] / volumes["long"]
                features["tf_medium_concentration"] = volumes["medium"] / volumes["long"]
            
            # Activity acceleration across timeframes
            if volumes["medium"] > 0 and volumes["long"] > 0:
                medium_rate = volumes["medium"] / timeframes["medium"]
                long_rate = volumes["long"] / timeframes["long"]
                features["tf_acceleration"] = medium_rate / long_rate if long_rate > 0 else 1.0
            
            # Count-based analysis
            if counts["long"] > 0:
                features["tf_event_concentration"] = counts["micro"] / counts["long"]
                
            # Temporal clustering analysis
            if len(events) >= 20:
                recent_events = [e for e in events if current_time - e["timestamp"] <= 300]
                if len(recent_events) >= 10:
                    timestamps = [e["timestamp"] for e in recent_events]
                    intervals = np.diff(timestamps)
                    
                    if len(intervals) > 0:
                        # Coefficient of variation for intervals (clustering measure)
                        mean_interval = np.mean(intervals)
                        std_interval = np.std(intervals)
                        features["tf_clustering_cv"] = std_interval / mean_interval if mean_interval > 0 else 0
        
        except Exception as e:
            logger.warning(f"Error computing timeframe features for {symbol}", exception=e)
        
        return features
    
    def _compute_size_features(self, symbol: str, current_liq: Dict[str, Any]) -> Dict[str, float]:
        """Compute size-based liquidation features."""
        features = {}
        
        try:
            current_size = current_liq["size"]
            percentiles = self.size_percentiles.get(symbol, {})
            
            if not percentiles:
                return features
            
            # Current liquidation size classification
            if current_size >= percentiles.get("p99", float("inf")):
                features["size_class"] = 5  # Mega liquidation
            elif current_size >= percentiles.get("p95", float("inf")):
                features["size_class"] = 4  # Large liquidation
            elif current_size >= percentiles.get("p90", float("inf")):
                features["size_class"] = 3  # Notable liquidation
            elif current_size >= percentiles.get("p75", float("inf")):
                features["size_class"] = 2  # Medium liquidation
            else:
                features["size_class"] = 1  # Small liquidation
            
            # Size percentile position
            for percentile_name, percentile_value in percentiles.items():
                if percentile_value > 0:
                    features[f"size_vs_{percentile_name}"] = current_size / percentile_value
            
            # Recent large liquidation analysis
            events = list(self.liquidation_events[symbol])
            current_time = current_liq["timestamp"]
            
            recent_events = [e for e in events if current_time - e["timestamp"] <= 60]  # Last minute
            if recent_events:
                recent_sizes = [e["size"] for e in recent_events]
                large_threshold = percentiles.get("p90", 0)
                
                large_count = sum(1 for size in recent_sizes if size >= large_threshold)
                features["large_liq_count_1m"] = large_count
                features["large_liq_ratio_1m"] = large_count / len(recent_events)
                
                # Largest liquidation in recent period
                max_recent_size = max(recent_sizes)
                features["max_size_1m"] = max_recent_size
                features["current_vs_max_1m"] = current_size / max_recent_size if max_recent_size > 0 else 0
        
        except Exception as e:
            logger.warning(f"Error computing size features for {symbol}", exception=e)
        
        return features
    
    def _compute_price_correlation_features(self, symbol: str, current_liq: Dict[str, Any]) -> Dict[str, float]:
        """Compute price-liquidation correlation features."""
        features = {}
        
        try:
            # Store price at liquidation for correlation analysis
            self.price_at_liquidation[symbol].append({
                "timestamp": current_liq["timestamp"],
                "price": current_liq["price"],
                "size": current_liq["size"],
                "side": current_liq["side"]
            })
            
            price_data = list(self.price_at_liquidation[symbol])
            if len(price_data) < 20:
                return features
            
            # Price momentum around liquidations
            recent_data = price_data[-20:]
            prices = [d["price"] for d in recent_data]
            sizes = [d["size"] for d in recent_data]
            
            if len(prices) >= 10:
                # Price-size correlation
                correlation = np.corrcoef(prices, sizes)[0, 1] if len(prices) > 2 else 0
                features["price_size_correlation"] = correlation
                
                # Price momentum during liquidation activity
                price_changes = np.diff(prices)
                if len(price_changes) > 0:
                    features["price_momentum_during_liq"] = np.mean(price_changes)
                    features["price_volatility_during_liq"] = np.std(price_changes)
            
            # Directional price impact
            sell_events = [d for d in recent_data if d["side"].lower() == "sell"]
            buy_events = [d for d in recent_data if d["side"].lower() == "buy"]
            
            if len(sell_events) >= 3 and len(buy_events) >= 3:
                sell_prices = [d["price"] for d in sell_events]
                buy_prices = [d["price"] for d in buy_events]
                
                # Are sell liquidations happening at lower prices?
                avg_sell_price = np.mean(sell_prices)
                avg_buy_price = np.mean(buy_prices)
                overall_avg = np.mean(prices)
                
                if overall_avg > 0:
                    features["sell_liq_price_bias"] = (avg_sell_price - overall_avg) / overall_avg
                    features["buy_liq_price_bias"] = (avg_buy_price - overall_avg) / overall_avg
        
        except Exception as e:
            logger.warning(f"Error computing price correlation features for {symbol}", exception=e)
        
        return features
    
    def _compute_spillover_features(self, symbol: str, current_liq: Dict[str, Any]) -> Dict[str, float]:
        """Compute cross-symbol liquidation spillover features."""
        features = {}
        
        try:
            current_time = current_liq["timestamp"]
            
            # Look for concurrent liquidations in other symbols
            spillover_window = 10  # seconds
            spillover_count = 0
            spillover_volume = 0.0
            
            for other_symbol, other_events in self.liquidation_events.items():
                if other_symbol == symbol:
                    continue
                
                # Find liquidations in other symbols within spillover window
                concurrent_events = [
                    e for e in other_events 
                    if abs(e["timestamp"] - current_time) <= spillover_window
                ]
                
                spillover_count += len(concurrent_events)
                spillover_volume += sum(e["size"] for e in concurrent_events)
            
            features["spillover_count"] = spillover_count
            features["spillover_volume"] = spillover_volume
            
            # Update spillover tracking
            self.spillover_events[symbol].append({
                "timestamp": current_time,
                "spillover_count": spillover_count,
                "spillover_volume": spillover_volume
            })
            
            # Spillover trend analysis
            spillover_data = list(self.spillover_events[symbol])
            if len(spillover_data) >= 10:
                recent_spillovers = [d["spillover_count"] for d in spillover_data[-10:]]
                features["spillover_trend"] = np.mean(recent_spillovers)
                
                # Spillover correlation
                if len(spillover_data) >= 20:
                    timestamps = [d["timestamp"] for d in spillover_data[-20:]]
                    spillover_counts = [d["spillover_count"] for d in spillover_data[-20:]]
                    
                    # Are spillovers increasing over time?
                    x = np.arange(len(spillover_counts))
                    correlation = np.corrcoef(x, spillover_counts)[0, 1] if len(spillover_counts) > 2 else 0
                    features["spillover_acceleration"] = correlation
        
        except Exception as e:
            logger.warning(f"Error computing spillover features for {symbol}", exception=e)
        
        return features
    
    def get_feature_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of computed features for monitoring."""
        return {
            "symbol": symbol,
            "feature_count": len(self.latest_features.get(symbol, {})),
            "last_update": self.last_update.get(symbol, 0),
            "liquidation_events": len(self.liquidation_events.get(symbol, [])),
            "size_percentiles": self.size_percentiles.get(symbol, {}),
            "spillover_tracking": len(self.spillover_events.get(symbol, [])),
            "latest_features": list(self.latest_features.get(symbol, {}).keys())
        }