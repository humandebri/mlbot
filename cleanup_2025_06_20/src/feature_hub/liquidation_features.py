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
from ..common.decorators import with_error_handling
from ..common.performance import profile_performance
from ..common.error_handler import error_context
from ..common.exceptions import FeatureError, TradingBotError
from ..common.performance import performance_context, MemoryOptimizer
from ..common.types import Symbol, FeatureDict
from ..common.utils import safe_float, safe_int, get_utc_timestamp, clamp

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
        self.max_history = safe_int(max_history, 1800)
        
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
        
        # Advanced liquidation analysis
        self.size_distribution: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.asymmetry_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
        self.cluster_analysis: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    @profile_performance(include_memory=True)
    @with_error_handling(FeatureError)
    def process_liquidation(self, symbol: Symbol, data: Dict[str, Any]) -> FeatureDict:
        """
        Process liquidation event and compute advanced features.
        
        Args:
            symbol: Trading symbol
            data: Liquidation data from ingestor
            
        Returns:
            Dictionary of computed features
        """
        with performance_context(f"liquidation_processing_{symbol}"):
            current_time = get_utc_timestamp() / 1000.0  # Convert to seconds
            
            # Rate limiting
            if current_time - self.last_update[symbol] < self.min_update_interval:
                return self.latest_features[symbol]
            
            # Extract liquidation data with safe conversions
            liquidation = {
                "timestamp": safe_float(data.get("timestamp", current_time)),
                "symbol": symbol,
                "side": data.get("side", ""),
                "size": safe_float(data.get("size", 0)),
                "price": safe_float(data.get("price", 0)),
                "liquidation_time": safe_float(data.get("liquidation_time", current_time)),
                "spike_detected": data.get("spike_detected", False),
                "spike_severity": safe_float(data.get("spike_severity", 0)),
                "spike_type": data.get("spike_type", ""),
                "liq_metrics": data.get("liq_metrics", {})
            }
            
            # Validate critical data
            if liquidation["size"] <= 0 or liquidation["price"] <= 0:
                logger.debug(f"Invalid liquidation data for {symbol}: size={liquidation['size']}, price={liquidation['price']}")
                return {}
            
            with error_context({"symbol": symbol, "liquidation": liquidation}):
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
                
                # 8. Advanced size distribution analysis
                features.update(self._compute_size_distribution_features(symbol))
                
                # 9. Long/short asymmetry analysis
                features.update(self._compute_asymmetry_features(symbol))
                
                # 10. Liquidation clustering analysis
                features.update(self._compute_clustering_features(symbol))
                
                # Validate and clamp feature values
                features = self._validate_features(features)
                
                # Cache results
                self.latest_features[symbol] = features
                self.last_update[symbol] = current_time
                
                # Periodic memory optimization
                if len(self.liquidation_events) % 100 == 0:
                    MemoryOptimizer.optimize_memory()
                
                return features
    
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
    
    def _update_size_percentiles(self, symbol: Symbol, current_time: float) -> None:
        """Update size percentiles for threshold analysis."""
        if current_time - self.last_percentile_update[symbol] < self.update_percentiles_interval:
            return
        
        with error_context({"symbol": symbol, "operation": "update_size_percentiles"}):
            events = list(self.liquidation_events[symbol])
            if len(events) < 50:  # Need minimum data
                return
            
            # Use safe conversion and limit data for memory efficiency
            sizes = [safe_float(event["size"]) for event in events[-500:]]  # Last 500 events
            sizes = [s for s in sizes if s > 0]  # Filter out invalid sizes
            
            if len(sizes) < 10:
                return
            
            size_array = np.array(sizes, dtype=np.float32)
            
            try:
                self.size_percentiles[symbol] = {
                    "p50": float(np.percentile(size_array, 50)),
                    "p75": float(np.percentile(size_array, 75)),
                    "p90": float(np.percentile(size_array, 90)),
                    "p95": float(np.percentile(size_array, 95)),
                    "p99": float(np.percentile(size_array, 99)),
                }
                
                self.last_percentile_update[symbol] = current_time
                
            except Exception as e:
                logger.warning(f"Failed to compute percentiles for {symbol}", exception=e)
                # Keep existing percentiles if calculation fails
    
    def _compute_cascade_features(self, symbol: Symbol, current_liq: Dict[str, Any]) -> FeatureDict:
        """Compute liquidation cascade analysis features."""
        features = {}
        
        with error_context({"symbol": symbol, "operation": "compute_cascade_features"}):
            events = list(self.liquidation_events[symbol])
            if len(events) < 5:
                return features
            
            current_time = safe_float(current_liq["timestamp"])
            
            # Define cascade time windows
            cascade_windows = [5, 15, 30, 60]  # seconds
            
            for window_sec in cascade_windows:
                window_start = current_time - window_sec
                window_events = [e for e in events if safe_float(e["timestamp"]) >= window_start]
                
                if not window_events:
                    continue
                
                # Cascade intensity metrics with safe calculations
                sizes = [safe_float(e["size"]) for e in window_events]
                total_volume = sum(sizes)
                event_count = len(window_events)
                
                features[f"cascade_volume_{window_sec}s"] = total_volume
                features[f"cascade_count_{window_sec}s"] = event_count
                features[f"cascade_intensity_{window_sec}s"] = total_volume / window_sec if window_sec > 0 else 0
                
                # Cascade acceleration (recent vs earlier events)
                if window_sec >= 30 and len(window_events) >= 4:
                    half_window = window_sec // 2
                    mid_time = current_time - half_window
                    
                    recent_events = [e for e in window_events if safe_float(e["timestamp"]) >= mid_time]
                    earlier_events = [e for e in window_events if safe_float(e["timestamp"]) < mid_time]
                    
                    if recent_events and earlier_events:
                        recent_volume = sum(safe_float(e["size"]) for e in recent_events)
                        earlier_volume = sum(safe_float(e["size"]) for e in earlier_events)
                        
                        # Normalize by time period
                        recent_rate = recent_volume / half_window if half_window > 0 else 0
                        earlier_rate = earlier_volume / half_window if half_window > 0 else 0
                        
                        if earlier_rate > 0:
                            acceleration = recent_rate / earlier_rate
                            features[f"cascade_acceleration_{window_sec}s"] = clamp(acceleration, 0, 10)
                
                # Side-specific cascade analysis
                buy_liquidations = [e for e in window_events if str(e.get("side", "")).lower() == "buy"]
                sell_liquidations = [e for e in window_events if str(e.get("side", "")).lower() == "sell"]
                
                buy_volume = sum(safe_float(e["size"]) for e in buy_liquidations)
                sell_volume = sum(safe_float(e["size"]) for e in sell_liquidations)
                total_volume = buy_volume + sell_volume
                
                if total_volume > 0:
                    features[f"cascade_sell_ratio_{window_sec}s"] = sell_volume / total_volume
                    imbalance = (sell_volume - buy_volume) / total_volume
                    features[f"cascade_directional_imbalance_{window_sec}s"] = clamp(imbalance, -1, 1)
                
                # Price range during cascade
                if window_events:
                    prices = [safe_float(e["price"]) for e in window_events if safe_float(e["price"]) > 0]
                    if prices:
                        price_range = max(prices) - min(prices)
                        avg_price = sum(prices) / len(prices)
                        range_ratio = price_range / avg_price if avg_price > 0 else 0
                        features[f"cascade_price_range_{window_sec}s"] = clamp(range_ratio, 0, 1)
            
            # Multi-timeframe cascade correlation
            if "cascade_volume_5s" in features and "cascade_volume_60s" in features:
                vol_5s = features["cascade_volume_5s"]
                vol_60s = features["cascade_volume_60s"]
                concentration = vol_5s / vol_60s if vol_60s > 0 else 0
                features["cascade_concentration"] = clamp(concentration, 0, 1)
        
        return features
    
    def _validate_features(self, features: FeatureDict) -> FeatureDict:
        """
        Validate and sanitize feature values.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Validated feature dictionary
        """
        validated = {}
        
        for key, value in features.items():
            # Convert to safe float
            safe_value = safe_float(value)
            
            # Check for invalid values
            if np.isnan(safe_value) or np.isinf(safe_value):
                logger.debug(f"Invalid feature value for {key}: {value}")
                safe_value = 0.0
            
            # Clamp extreme values
            safe_value = clamp(safe_value, -1000, 1000)
            
            validated[key] = safe_value
        
        return validated
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get statistics about computed features."""
        stats = {
            "total_symbols": len(self.liquidation_events),
            "total_events": sum(len(events) for events in self.liquidation_events.values()),
            "feature_counts": {},
            "memory_usage": {}
        }
        
        for symbol, features in self.latest_features.items():
            stats["feature_counts"][symbol] = len(features)
        
        # Memory usage estimation
        for symbol in self.liquidation_events:
            event_count = len(self.liquidation_events[symbol])
            estimated_bytes = event_count * 200  # Rough estimate per event
            stats["memory_usage"][symbol] = f"{estimated_bytes / 1024:.1f} KB"
        
        return stats
    
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
    
    def _compute_size_distribution_features(self, symbol: str) -> Dict[str, float]:
        """Compute advanced size distribution analysis features."""
        features = {}
        
        try:
            events = list(self.liquidation_events[symbol])
            if len(events) < 30:
                return features
                
            # Get size data
            sizes = [e["size"] for e in events[-200:]]  # Last 200 events
            size_array = np.array(sizes)
            
            # Distribution shape analysis
            features["size_dist_skewness"] = float(np.abs(np.mean((size_array - np.mean(size_array)) ** 3) / (np.std(size_array) ** 3 + 1e-8)))
            features["size_dist_kurtosis"] = float(np.mean((size_array - np.mean(size_array)) ** 4) / (np.std(size_array) ** 4 + 1e-8) - 3)
            
            # Log-normal distribution test (common in financial data)
            log_sizes = np.log(size_array + 1e-8)
            features["size_dist_log_mean"] = float(np.mean(log_sizes))
            features["size_dist_log_std"] = float(np.std(log_sizes))
            
            # Bimodality detection (indicates different trader types)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(size_array)
            x_range = np.linspace(size_array.min(), size_array.max(), 100)
            density = kde(x_range)
            
            # Find local maxima in density
            peaks = []
            for i in range(1, len(density) - 1):
                if density[i] > density[i-1] and density[i] > density[i+1]:
                    peaks.append(i)
            
            features["size_dist_n_modes"] = len(peaks)
            features["size_dist_bimodal"] = 1.0 if len(peaks) >= 2 else 0.0
            
            # Tail analysis
            p95 = np.percentile(size_array, 95)
            tail_sizes = size_array[size_array > p95]
            if len(tail_sizes) > 0:
                features["size_dist_tail_weight"] = float(np.sum(tail_sizes) / np.sum(size_array))
                features["size_dist_tail_count_ratio"] = len(tail_sizes) / len(size_array)
            
            # Recent distribution shift
            if len(sizes) >= 100:
                recent_sizes = size_array[-50:]
                older_sizes = size_array[-100:-50]
                
                # Kolmogorov-Smirnov test for distribution shift
                from scipy.stats import ks_2samp
                ks_stat, _ = ks_2samp(recent_sizes, older_sizes)
                features["size_dist_shift_ks"] = float(ks_stat)
                
                # Mean and variance shift
                features["size_dist_mean_shift"] = float(np.mean(recent_sizes) / (np.mean(older_sizes) + 1e-8))
                features["size_dist_var_shift"] = float(np.var(recent_sizes) / (np.var(older_sizes) + 1e-8))
                
        except Exception as e:
            logger.warning(f"Error computing size distribution features for {symbol}", exception=e)
            
        return features
    
    def _compute_asymmetry_features(self, symbol: str) -> Dict[str, float]:
        """Compute long/short liquidation asymmetry features."""
        features = {}
        
        try:
            events = list(self.liquidation_events[symbol])
            if len(events) < 20:
                return features
                
            # Separate long and short liquidations
            long_liquidations = [e for e in events if e["side"].lower() == "buy"]
            short_liquidations = [e for e in events if e["side"].lower() == "sell"]
            
            if len(long_liquidations) >= 5 and len(short_liquidations) >= 5:
                # Size asymmetry
                long_sizes = [e["size"] for e in long_liquidations[-100:]]
                short_sizes = [e["size"] for e in short_liquidations[-100:]]
                
                avg_long_size = np.mean(long_sizes)
                avg_short_size = np.mean(short_sizes)
                
                features["asymmetry_size_ratio"] = avg_long_size / (avg_short_size + 1e-8)
                features["asymmetry_size_diff"] = (avg_long_size - avg_short_size) / (avg_long_size + avg_short_size + 1e-8)
                
                # Volume asymmetry over different windows
                windows = [60, 300, 900]  # 1min, 5min, 15min
                current_time = events[-1]["timestamp"]
                
                for window in windows:
                    window_start = current_time - window
                    
                    long_vol = sum(e["size"] for e in long_liquidations if e["timestamp"] >= window_start)
                    short_vol = sum(e["size"] for e in short_liquidations if e["timestamp"] >= window_start)
                    total_vol = long_vol + short_vol
                    
                    if total_vol > 0:
                        features[f"asymmetry_volume_ratio_{window}s"] = short_vol / total_vol
                        features[f"asymmetry_volume_imbalance_{window}s"] = (short_vol - long_vol) / total_vol
                
                # Frequency asymmetry
                long_count = len([e for e in long_liquidations if e["timestamp"] >= current_time - 300])
                short_count = len([e for e in short_liquidations if e["timestamp"] >= current_time - 300])
                
                if long_count + short_count > 0:
                    features["asymmetry_count_ratio"] = short_count / (long_count + short_count)
                
                # Temporal asymmetry (are shorts clustering more than longs?)
                if len(long_liquidations) >= 10 and len(short_liquidations) >= 10:
                    long_times = [e["timestamp"] for e in long_liquidations[-20:]]
                    short_times = [e["timestamp"] for e in short_liquidations[-20:]]
                    
                    long_intervals = np.diff(long_times) if len(long_times) > 1 else [0]
                    short_intervals = np.diff(short_times) if len(short_times) > 1 else [0]
                    
                    if len(long_intervals) > 0 and len(short_intervals) > 0:
                        # Lower CV means more regular/clustered
                        long_cv = np.std(long_intervals) / (np.mean(long_intervals) + 1e-8)
                        short_cv = np.std(short_intervals) / (np.mean(short_intervals) + 1e-8)
                        
                        features["asymmetry_temporal_clustering"] = short_cv / (long_cv + 1e-8)
                
                # Track asymmetry changes
                self.asymmetry_metrics[symbol].append({
                    "timestamp": current_time,
                    "short_dominance": features.get("asymmetry_volume_ratio_300s", 0.5)
                })
                
                # Asymmetry trend
                if len(self.asymmetry_metrics[symbol]) >= 10:
                    recent_metrics = list(self.asymmetry_metrics[symbol])[-10:]
                    dominance_values = [m["short_dominance"] for m in recent_metrics]
                    
                    # Is short dominance increasing?
                    x = np.arange(len(dominance_values))
                    correlation = np.corrcoef(x, dominance_values)[0, 1] if len(dominance_values) > 2 else 0
                    features["asymmetry_trend"] = correlation
                    
        except Exception as e:
            logger.warning(f"Error computing asymmetry features for {symbol}", exception=e)
            
        return features
    
    def _compute_clustering_features(self, symbol: str) -> Dict[str, float]:
        """Compute liquidation clustering analysis features."""
        features = {}
        
        try:
            events = list(self.liquidation_events[symbol])
            if len(events) < 30:
                return features
                
            current_time = events[-1]["timestamp"]
            
            # Time-based clustering using DBSCAN concept
            recent_events = [e for e in events if current_time - e["timestamp"] <= 300]  # Last 5 minutes
            
            if len(recent_events) >= 10:
                timestamps = np.array([e["timestamp"] for e in recent_events])
                
                # Simple clustering: events within 2 seconds are considered a cluster
                cluster_threshold = 2.0  # seconds
                clusters = []
                current_cluster = [timestamps[0]]
                
                for i in range(1, len(timestamps)):
                    if timestamps[i] - timestamps[i-1] <= cluster_threshold:
                        current_cluster.append(timestamps[i])
                    else:
                        if len(current_cluster) >= 2:  # Minimum cluster size
                            clusters.append(current_cluster)
                        current_cluster = [timestamps[i]]
                
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                
                features["cluster_count_5m"] = len(clusters)
                
                if clusters:
                    cluster_sizes = [len(c) for c in clusters]
                    features["cluster_avg_size"] = np.mean(cluster_sizes)
                    features["cluster_max_size"] = max(cluster_sizes)
                    
                    # Cluster intensity (events per cluster per second)
                    cluster_intensities = []
                    for cluster in clusters:
                        duration = max(cluster) - min(cluster) + 0.1  # Avoid division by zero
                        intensity = len(cluster) / duration
                        cluster_intensities.append(intensity)
                    
                    features["cluster_avg_intensity"] = np.mean(cluster_intensities)
                    features["cluster_max_intensity"] = max(cluster_intensities)
                    
                    # Inter-cluster intervals
                    if len(clusters) >= 2:
                        cluster_starts = [min(c) for c in clusters]
                        inter_cluster_intervals = np.diff(sorted(cluster_starts))
                        
                        features["cluster_avg_interval"] = np.mean(inter_cluster_intervals)
                        features["cluster_interval_cv"] = np.std(inter_cluster_intervals) / (np.mean(inter_cluster_intervals) + 1e-8)
                
                # Size-based clustering (large liquidations triggering cascades)
                sizes = np.array([e["size"] for e in recent_events])
                large_threshold = np.percentile(sizes, 80) if len(sizes) >= 5 else np.mean(sizes)
                
                # Find cascades following large liquidations
                cascade_events = 0
                for i, event in enumerate(recent_events):
                    if event["size"] >= large_threshold:
                        # Count events within 10 seconds after this large liquidation
                        cascade_end = event["timestamp"] + 10
                        following_events = [e for e in recent_events[i+1:] 
                                          if e["timestamp"] <= cascade_end]
                        if len(following_events) >= 3:  # Minimum cascade size
                            cascade_events += 1
                
                features["cluster_cascade_triggers"] = cascade_events
                
                # Directional clustering (do liquidations of same side cluster?)
                for side in ["buy", "sell"]:
                    side_events = [e for e in recent_events if e["side"].lower() == side]
                    if len(side_events) >= 5:
                        side_times = [e["timestamp"] for e in side_events]
                        side_intervals = np.diff(side_times) if len(side_times) > 1 else [0]
                        
                        if len(side_intervals) > 0:
                            # Lower CV indicates more clustering
                            cv = np.std(side_intervals) / (np.mean(side_intervals) + 1e-8)
                            features[f"cluster_{side}_cv"] = cv
                            
                            # Burst detection (many events in short time)
                            short_intervals = sum(1 for interval in side_intervals if interval < 1.0)
                            features[f"cluster_{side}_burst_ratio"] = short_intervals / len(side_intervals)
                
            # Update cluster analysis cache
            self.cluster_analysis[symbol] = {
                "last_update": current_time,
                "cluster_count": features.get("cluster_count_5m", 0),
                "features": features.copy()
            }
            
        except Exception as e:
            logger.warning(f"Error computing clustering features for {symbol}", exception=e)
            
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
            "latest_features": list(self.latest_features.get(symbol, {}).keys()),
            "cluster_analysis": self.cluster_analysis.get(symbol, {})
        }