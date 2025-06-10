"""High-performance liquidation feed processor optimized for spike detection."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..common.config import settings
from ..common.logging import get_logger
from ..common.monitoring import (
    increment_counter, 
    observe_histogram, 
    set_gauge,
    LIQUIDATION_VOLUME_5S,
    LIQUIDATION_SPIKE_Z_SCORE,
    LIQUIDATION_SIDE_RATIO,
    LIQUIDATION_SPIKES_DETECTED,
    LIQUIDATION_SPIKE_SEVERITY,
)

logger = get_logger(__name__)


@dataclass
class LiquidationEvent:
    """Structured liquidation event data."""
    
    symbol: str
    side: str  # Buy/Sell
    size: float
    price: float
    timestamp: float
    
    @classmethod
    def from_bybit_data(cls, data: Dict[str, Any]) -> Optional['LiquidationEvent']:
        """Create LiquidationEvent from Bybit WebSocket data."""
        try:
            return cls(
                symbol=data.get("symbol", ""),
                side=data.get("side", ""),
                size=float(data.get("size", 0)),
                price=float(data.get("price", 0)),
                timestamp=float(data.get("updatedTime", 0)) / 1000,  # Convert to seconds
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid liquidation data: {e}", data=data)
            return None


class LiquidationSpikeDetector:
    """
    Efficient liquidation spike detection with rolling statistics.
    
    Optimized for:
    - Real-time spike detection
    - Memory-efficient rolling windows
    - Statistical outlier identification
    - Cost-effective computation
    """
    
    def __init__(self, window_seconds: int = 300, spike_threshold: float = 2.0):
        """
        Initialize spike detector.
        
        Args:
            window_seconds: Rolling window size in seconds
            spike_threshold: Z-score threshold for spike detection
        """
        self.window_seconds = window_seconds
        self.spike_threshold = spike_threshold
        
        # Per-symbol data structures (memory-efficient)
        self.liquidation_volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.side_ratios: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
        
        # Statistics cache (updated incrementally)
        self.volume_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.last_stat_update: Dict[str, float] = defaultdict(float)
        self.stat_update_interval = 5.0  # Update stats every 5 seconds
        
        # Spike detection results
        self.recent_spikes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def process_liquidation(self, event: LiquidationEvent) -> Dict[str, Any]:
        """
        Process liquidation event and detect spikes efficiently.
        
        Returns:
            Dictionary with spike analysis results
        """
        symbol = event.symbol
        current_time = event.timestamp
        
        # Add to rolling windows
        self._add_to_windows(event)
        
        # Clean old data
        self._cleanup_old_data(symbol, current_time)
        
        # Update statistics incrementally
        if current_time - self.last_stat_update[symbol] > self.stat_update_interval:
            self._update_statistics(symbol)
            self.last_stat_update[symbol] = current_time
        
        # Calculate current metrics
        metrics = self._calculate_current_metrics(symbol, current_time)
        
        # Detect spikes
        spike_info = self._detect_spike(symbol, metrics)
        
        # Update monitoring metrics
        self._update_monitoring_metrics(symbol, metrics, spike_info)
        
        return {
            "symbol": symbol,
            "timestamp": current_time,
            "metrics": metrics,
            "spike_detected": spike_info["detected"],
            "spike_severity": spike_info["severity"],
            "spike_info": spike_info,
        }
    
    def _add_to_windows(self, event: LiquidationEvent) -> None:
        """Add liquidation event to rolling windows efficiently."""
        symbol = event.symbol
        timestamp = event.timestamp
        
        # Add volume data
        self.liquidation_volumes[symbol].append(event.size)
        self.volume_timestamps[symbol].append(timestamp)
        
        # Calculate side ratio for recent events (last 60 seconds)
        recent_threshold = timestamp - 60
        recent_sells = 0
        recent_buys = 0
        
        # Only calculate if we have recent data
        for i, ts in enumerate(reversed(self.volume_timestamps[symbol])):
            if ts < recent_threshold:
                break
            
            # Get corresponding volume (reverse index)
            idx = len(self.liquidation_volumes[symbol]) - 1 - i
            if idx >= 0:
                volume = self.liquidation_volumes[symbol][idx]
                if event.side.lower() == "sell":
                    recent_sells += volume
                else:
                    recent_buys += volume
        
        # Calculate and store side ratio
        total_volume = recent_sells + recent_buys
        side_ratio = recent_sells / total_volume if total_volume > 0 else 0.5
        self.side_ratios[symbol].append(side_ratio)
    
    def _cleanup_old_data(self, symbol: str, current_time: float) -> None:
        """Remove data outside the rolling window."""
        cutoff_time = current_time - self.window_seconds
        
        # Clean volume data
        while (self.volume_timestamps[symbol] and 
               self.volume_timestamps[symbol][0] < cutoff_time):
            self.volume_timestamps[symbol].popleft()
            if self.liquidation_volumes[symbol]:
                self.liquidation_volumes[symbol].popleft()
    
    def _update_statistics(self, symbol: str) -> None:
        """Update rolling statistics incrementally."""
        if not self.liquidation_volumes[symbol]:
            return
        
        volumes = list(self.liquidation_volumes[symbol])
        
        if len(volumes) < 10:  # Need minimum data for statistics
            self.volume_stats[symbol] = {
                "mean": 0.0,
                "std": 1.0,
                "count": len(volumes)
            }
            return
        
        # Calculate statistics efficiently
        volumes_array = np.array(volumes, dtype=np.float32)
        
        self.volume_stats[symbol] = {
            "mean": float(np.mean(volumes_array)),
            "std": max(float(np.std(volumes_array)), 0.01),  # Avoid division by zero
            "count": len(volumes),
            "percentile_95": float(np.percentile(volumes_array, 95)),
            "percentile_99": float(np.percentile(volumes_array, 99)),
        }
    
    def _calculate_current_metrics(self, symbol: str, current_time: float) -> Dict[str, float]:
        """Calculate current liquidation metrics efficiently."""
        # Time windows for different metrics
        windows = {
            "5s": current_time - 5,
            "30s": current_time - 30,
            "60s": current_time - 60,
            "300s": current_time - 300,
        }
        
        metrics = {}
        
        # Calculate volume for each window
        for window_name, start_time in windows.items():
            volume = 0.0
            count = 0
            
            for i, timestamp in enumerate(self.volume_timestamps[symbol]):
                if timestamp >= start_time:
                    if i < len(self.liquidation_volumes[symbol]):
                        volume += self.liquidation_volumes[symbol][i]
                        count += 1
            
            metrics[f"liq_vol_{window_name}"] = volume
            metrics[f"liq_count_{window_name}"] = count
        
        # Current side ratio
        if self.side_ratios[symbol]:
            metrics["liq_side_ratio"] = self.side_ratios[symbol][-1]
        else:
            metrics["liq_side_ratio"] = 0.5
        
        # Calculate Z-scores for spike detection
        stats = self.volume_stats[symbol]
        if stats.get("std", 0) > 0:
            metrics["liq_spike_z_5s"] = (metrics["liq_vol_5s"] - stats["mean"]) / stats["std"]
            metrics["liq_spike_z_30s"] = (metrics["liq_vol_30s"] - stats["mean"]) / stats["std"]
        else:
            metrics["liq_spike_z_5s"] = 0.0
            metrics["liq_spike_z_30s"] = 0.0
        
        return metrics
    
    def _detect_spike(self, symbol: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect liquidation spikes with multiple criteria."""
        spike_info = {
            "detected": False,
            "severity": 0.0,
            "type": "none",
            "criteria": {}
        }
        
        # Z-score based detection (primary)
        z_score_5s = metrics.get("liq_spike_z_5s", 0.0)
        z_score_30s = metrics.get("liq_spike_z_30s", 0.0)
        
        spike_info["criteria"]["z_score_5s"] = z_score_5s
        spike_info["criteria"]["z_score_30s"] = z_score_30s
        
        # Detect different types of spikes
        if z_score_5s > self.spike_threshold:
            spike_info["detected"] = True
            spike_info["severity"] = z_score_5s
            spike_info["type"] = "instant_spike"
        
        elif z_score_30s > self.spike_threshold * 0.7:
            spike_info["detected"] = True
            spike_info["severity"] = z_score_30s
            spike_info["type"] = "sustained_spike"
        
        # Side ratio based detection (directional spike)
        side_ratio = metrics.get("liq_side_ratio", 0.5)
        if side_ratio > 0.8 or side_ratio < 0.2:
            directional_strength = abs(side_ratio - 0.5) * 2  # 0-1 scale
            
            if directional_strength > 0.6 and metrics.get("liq_vol_5s", 0) > 0:
                if not spike_info["detected"] or spike_info["severity"] < directional_strength * 3:
                    spike_info["detected"] = True
                    spike_info["severity"] = directional_strength * 3
                    spike_info["type"] = "directional_spike"
        
        spike_info["criteria"]["side_ratio"] = side_ratio
        spike_info["criteria"]["directional_strength"] = abs(side_ratio - 0.5) * 2
        
        # Store recent spike for analysis
        if spike_info["detected"]:
            self.recent_spikes[symbol].append({
                "timestamp": time.time(),
                "severity": spike_info["severity"],
                "type": spike_info["type"],
                "metrics": metrics.copy()
            })
            
            # Keep only recent spikes (last hour)
            current_time = time.time()
            self.recent_spikes[symbol] = [
                spike for spike in self.recent_spikes[symbol]
                if current_time - spike["timestamp"] < 3600
            ]
        
        return spike_info
    
    def _update_monitoring_metrics(
        self, 
        symbol: str, 
        metrics: Dict[str, float], 
        spike_info: Dict[str, Any]
    ) -> None:
        """Update Prometheus metrics for monitoring."""
        # Update volume metrics
        LIQUIDATION_VOLUME_5S.labels(symbol=symbol).set(metrics.get("liq_vol_5s", 0))
        
        LIQUIDATION_SPIKE_Z_SCORE.labels(symbol=symbol).set(metrics.get("liq_spike_z_5s", 0))
        
        LIQUIDATION_SIDE_RATIO.labels(symbol=symbol).set(metrics.get("liq_side_ratio", 0.5))
        
        # Spike detection metrics
        if spike_info["detected"]:
            LIQUIDATION_SPIKES_DETECTED.labels(symbol=symbol, type=spike_info["type"]).inc()
            LIQUIDATION_SPIKE_SEVERITY.labels(symbol=symbol).observe(spike_info["severity"])
    
    def get_recent_spike_summary(self, symbol: str, minutes: int = 60) -> Dict[str, Any]:
        """Get summary of recent spikes for the symbol."""
        current_time = time.time()
        cutoff_time = current_time - (minutes * 60)
        
        recent = [
            spike for spike in self.recent_spikes[symbol]
            if spike["timestamp"] >= cutoff_time
        ]
        
        if not recent:
            return {
                "spike_count": 0,
                "max_severity": 0.0,
                "spike_types": {},
                "avg_severity": 0.0
            }
        
        spike_types = defaultdict(int)
        severities = []
        
        for spike in recent:
            spike_types[spike["type"]] += 1
            severities.append(spike["severity"])
        
        return {
            "spike_count": len(recent),
            "max_severity": max(severities),
            "avg_severity": sum(severities) / len(severities),
            "spike_types": dict(spike_types),
            "recent_spikes": recent[-5:]  # Last 5 spikes
        }