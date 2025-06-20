"""
Label generation for liquidation-driven trading ML models.

Generates expPNL (expected PnL) labels for predicting optimal limit order placement
during liquidation events and price wicks.

Core Logic:
- Detect liquidation spikes and price wicks
- Calculate expected profitability of limit orders at different delta levels
- Generate forward-looking labels with multiple time horizons
- Account for transaction costs and slippage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LabelConfig:
    """Configuration for label generation."""
    
    # Delta levels for limit order placement (percentage from current price)
    delta_levels: List[float] = None
    
    # Forward-looking time windows (seconds)
    time_windows: List[int] = None
    
    # Transaction costs
    maker_fee: float = 0.0001  # 0.01% maker fee
    taker_fee: float = 0.0006  # 0.06% taker fee
    slippage: float = 0.0002   # 0.02% slippage
    
    # Liquidation spike detection
    spike_threshold: float = 2.0  # Z-score threshold for spike detection
    min_spike_volume: float = 1000.0  # Minimum liquidation volume
    spike_duration_max: int = 10  # Maximum spike duration in seconds
    
    # Label generation
    min_profit_threshold: float = 0.005  # 0.5% minimum profit for positive label
    max_loss_threshold: float = -0.01   # -1% maximum acceptable loss
    label_smoothing: float = 0.1        # Label smoothing factor
    
    # Risk management
    max_position_size: float = 0.1      # Maximum position size as fraction of capital
    stop_loss_pct: float = 0.02         # 2% stop loss
    take_profit_pct: float = 0.01       # 1% take profit
    
    # Data quality
    min_data_points: int = 100          # Minimum data points for label generation
    outlier_percentile: float = 0.99    # Remove top 1% outliers
    
    def __post_init__(self):
        if self.delta_levels is None:
            self.delta_levels = [0.02, 0.05, 0.10]  # 2%, 5%, 10%
        
        if self.time_windows is None:
            self.time_windows = [60, 300, 900]  # 1min, 5min, 15min


class LabelGenerator:
    """
    Generate training labels for liquidation-driven trading ML models.
    
    Focuses on:
    - Liquidation spike detection and classification
    - Expected PnL calculation for limit order strategies
    - Multi-horizon profitability prediction
    - Risk-adjusted label generation
    """
    
    def __init__(self, config: Optional[LabelConfig] = None):
        """
        Initialize label generator.
        
        Args:
            config: Label generation configuration
        """
        self.config = config or LabelConfig()
        
        # Statistics tracking
        self.label_stats = {}
        self.spike_events = []
        self.generation_time = 0.0
        
        logger.info("Label generator initialized", config=self.config.__dict__)
    
    def generate_labels(
        self, 
        market_data: pd.DataFrame,
        features: pd.DataFrame,
        symbol: str = "BTCUSDT"
    ) -> pd.DataFrame:
        """
        Generate training labels from market data and features.
        
        Args:
            market_data: OHLCV and liquidation data
            features: Computed features from feature engines
            symbol: Trading symbol
            
        Returns:
            DataFrame with generated labels
        """
        logger.info("Generating labels", 
                   market_data_shape=market_data.shape,
                   features_shape=features.shape,
                   symbol=symbol)
        
        start_time = pd.Timestamp.now()
        
        try:
            # 1. Validate input data
            self._validate_input_data(market_data, features)
            
            # 2. Detect liquidation spikes
            spike_events = self._detect_liquidation_spikes(market_data)
            
            # 3. Generate base labels for each delta level and time window
            labels_list = []
            
            for delta in self.config.delta_levels:
                for time_window in self.config.time_windows:
                    label_col = f"expPNL_delta_{int(delta*100)}_window_{time_window}"
                    
                    labels = self._calculate_expPNL(
                        market_data=market_data,
                        features=features,
                        delta=delta,
                        time_window=time_window,
                        spike_events=spike_events
                    )
                    
                    labels_list.append(pd.Series(labels, name=label_col, index=features.index))
            
            # 4. Combine all labels
            labels_df = pd.concat(labels_list, axis=1)
            
            # 5. Generate additional derived labels
            labels_df = self._generate_derived_labels(labels_df, market_data)
            
            # 6. Apply label smoothing and quality filters
            labels_df = self._apply_label_processing(labels_df)
            
            # 7. Compute label statistics
            self._compute_label_statistics(labels_df)
            
            self.generation_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            logger.info("Labels generated successfully",
                       label_shape=labels_df.shape,
                       generation_time=self.generation_time,
                       spike_events=len(spike_events),
                       label_stats=self.label_stats)
            
            return labels_df
            
        except Exception as e:
            logger.error("Error generating labels", exception=e)
            raise
    
    def _validate_input_data(self, market_data: pd.DataFrame, features: pd.DataFrame) -> None:
        """Validate input data quality."""
        if market_data.empty or features.empty:
            raise ValueError("Input data cannot be empty")
        
        if len(market_data) < self.config.min_data_points:
            raise ValueError(f"Insufficient data points: {len(market_data)} < {self.config.min_data_points}")
        
        # Check for required columns
        required_market_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_market_cols if col not in market_data.columns]
        if missing_cols:
            logger.warning("Missing market data columns", missing=missing_cols)
        
        # Check for timestamp alignment
        if not market_data.index.equals(features.index):
            logger.warning("Market data and features indices do not align")
    
    def _detect_liquidation_spikes(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect liquidation spikes in market data.
        
        Returns:
            List of spike event dictionaries
        """
        spike_events = []
        
        # Calculate price volatility
        returns = market_data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        # Calculate volume anomalies
        volume_sma = market_data['volume'].rolling(window=20).mean()
        volume_std = market_data['volume'].rolling(window=20).std()
        volume_zscore = (market_data['volume'] - volume_sma) / volume_std
        
        # Detect liquidation volume spikes if available
        if 'liquidation_volume' in market_data.columns:
            liq_volume = market_data['liquidation_volume'].fillna(0)
            liq_sma = liq_volume.rolling(window=20).mean()
            liq_std = liq_volume.rolling(window=20).std()
            liq_zscore = (liq_volume - liq_sma) / liq_std
        else:
            liq_zscore = pd.Series(0, index=market_data.index)
        
        # Detect price wicks (large high-low ranges)
        wick_ratio = (market_data['high'] - market_data['low']) / market_data['close']
        wick_threshold = wick_ratio.quantile(0.95)  # Top 5% of wicks
        
        # Combine spike detection criteria
        spike_conditions = (
            (volume_zscore > self.config.spike_threshold) |
            (liq_zscore > self.config.spike_threshold) |
            (wick_ratio > wick_threshold) |
            (np.abs(returns) > returns.quantile(0.99))  # Extreme price movements
        )
        
        spike_indices = market_data.index[spike_conditions]
        
        for idx in spike_indices:
            try:
                spike_info = {
                    'timestamp': idx,
                    'price': market_data.loc[idx, 'close'],
                    'volume': market_data.loc[idx, 'volume'],
                    'liquidation_volume': liq_volume.loc[idx] if 'liquidation_volume' in market_data.columns else 0,
                    'wick_ratio': wick_ratio.loc[idx],
                    'price_change': returns.loc[idx],
                    'volume_zscore': volume_zscore.loc[idx],
                    'liq_zscore': liq_zscore.loc[idx]
                }
                
                # Add spike classification
                if liq_zscore.loc[idx] > self.config.spike_threshold:
                    spike_info['type'] = 'liquidation_spike'
                elif wick_ratio.loc[idx] > wick_threshold:
                    spike_info['type'] = 'price_wick'
                elif volume_zscore.loc[idx] > self.config.spike_threshold:
                    spike_info['type'] = 'volume_spike'
                else:
                    spike_info['type'] = 'price_movement'
                
                spike_events.append(spike_info)
                
            except Exception as e:
                logger.warning(f"Error processing spike at {idx}", exception=e)
                continue
        
        logger.info("Liquidation spikes detected", 
                   total_spikes=len(spike_events),
                   spike_types={spike['type'] for spike in spike_events})
        
        return spike_events
    
    def _calculate_expPNL(
        self,
        market_data: pd.DataFrame,
        features: pd.DataFrame,
        delta: float,
        time_window: int,
        spike_events: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Calculate expected PnL for limit order strategy.
        
        Args:
            market_data: Market OHLCV data
            features: Feature matrix
            delta: Price delta level (e.g., 0.02 for 2%)
            time_window: Forward-looking time window in seconds
            spike_events: Detected spike events
            
        Returns:
            Array of expPNL values
        """
        expPNL = np.zeros(len(features))
        
        prices = market_data['close'].values
        highs = market_data['high'].values
        lows = market_data['low'].values
        
        # Convert time window to number of data points (assuming 1-second intervals)
        window_size = min(time_window, len(prices) - 1)
        
        for i in range(len(prices) - window_size):
            current_price = prices[i]
            
            # Define limit order prices
            buy_limit_price = current_price * (1 - delta)   # Buy below current price
            sell_limit_price = current_price * (1 + delta)  # Sell above current price
            
            # Look forward to see if orders would be filled and profitable
            future_lows = lows[i+1:i+1+window_size]
            future_highs = highs[i+1:i+1+window_size]
            future_prices = prices[i+1:i+1+window_size]
            
            # Check if buy limit order would be filled
            buy_filled = np.any(future_lows <= buy_limit_price)
            
            # Check if sell limit order would be filled  
            sell_filled = np.any(future_highs >= sell_limit_price)
            
            buy_profit = 0.0
            sell_profit = 0.0
            
            if buy_filled:
                # Find when buy order is filled
                fill_idx = np.argmax(future_lows <= buy_limit_price)
                fill_price = buy_limit_price
                
                # Calculate profit from subsequent price movement
                remaining_prices = future_prices[fill_idx:]
                if len(remaining_prices) > 0:
                    max_profit_price = np.max(remaining_prices)
                    # Assume we sell at market when price recovers
                    buy_profit = (max_profit_price - fill_price) / fill_price
                    
                    # Apply transaction costs
                    buy_profit -= (self.config.maker_fee + self.config.taker_fee + self.config.slippage)
                    
                    # Apply stop loss
                    min_price_after_fill = np.min(remaining_prices)
                    if (min_price_after_fill - fill_price) / fill_price < -self.config.stop_loss_pct:
                        buy_profit = -self.config.stop_loss_pct - self.config.taker_fee
            
            if sell_filled:
                # Find when sell order is filled
                fill_idx = np.argmax(future_highs >= sell_limit_price)
                fill_price = sell_limit_price
                
                # Calculate profit from subsequent price movement
                remaining_prices = future_prices[fill_idx:]
                if len(remaining_prices) > 0:
                    min_profit_price = np.min(remaining_prices)
                    # Assume we buy back at market when price drops
                    sell_profit = (fill_price - min_profit_price) / fill_price
                    
                    # Apply transaction costs
                    sell_profit -= (self.config.maker_fee + self.config.taker_fee + self.config.slippage)
                    
                    # Apply stop loss
                    max_price_after_fill = np.max(remaining_prices)
                    if (max_price_after_fill - fill_price) / fill_price > self.config.stop_loss_pct:
                        sell_profit = -self.config.stop_loss_pct - self.config.taker_fee
            
            # Combine buy and sell opportunities (can't do both simultaneously)
            expPNL[i] = max(buy_profit, sell_profit)
            
            # Boost expPNL near spike events
            current_time = features.index[i]
            near_spike = any(
                abs((spike['timestamp'] - current_time).total_seconds()) < 30
                for spike in spike_events
            )
            
            if near_spike and expPNL[i] > 0:
                expPNL[i] *= 1.2  # 20% boost for spike proximity
        
        return expPNL
    
    def _generate_derived_labels(self, labels_df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate additional derived labels."""
        enhanced_labels = labels_df.copy()
        
        # 1. Binary classification labels
        for col in labels_df.columns:
            # Positive/negative labels
            enhanced_labels[f"{col}_binary"] = (labels_df[col] > self.config.min_profit_threshold).astype(int)
            
            # Multi-class labels (loss, neutral, small profit, large profit)
            enhanced_labels[f"{col}_multiclass"] = pd.cut(
                labels_df[col],
                bins=[-np.inf, -0.005, 0.002, 0.01, np.inf],
                labels=[0, 1, 2, 3]  # loss, neutral, small profit, large profit
            ).astype(int)
        
        # 2. Risk-adjusted labels
        if 'close' in market_data.columns:
            volatility = market_data['close'].pct_change().rolling(window=20).std()
            
            for col in labels_df.columns:
                if 'expPNL' in col:
                    # Sharpe-like ratio
                    enhanced_labels[f"{col}_sharpe"] = labels_df[col] / (volatility + 1e-8)
        
        # 3. Aggregate labels across time windows
        delta_levels = self.config.delta_levels
        
        for delta in delta_levels:
            delta_str = f"delta_{int(delta*100)}"
            delta_cols = [col for col in labels_df.columns if delta_str in col]
            
            if len(delta_cols) > 1:
                # Average across time windows
                enhanced_labels[f"expPNL_{delta_str}_avg"] = labels_df[delta_cols].mean(axis=1)
                
                # Maximum across time windows
                enhanced_labels[f"expPNL_{delta_str}_max"] = labels_df[delta_cols].max(axis=1)
                
                # Consistency score (how often positive across windows)
                enhanced_labels[f"expPNL_{delta_str}_consistency"] = (
                    labels_df[delta_cols] > self.config.min_profit_threshold
                ).sum(axis=1) / len(delta_cols)
        
        return enhanced_labels
    
    def _apply_label_processing(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Apply label smoothing and quality filters."""
        processed_labels = labels_df.copy()
        
        # 1. Remove outliers
        outlier_threshold = self.config.outlier_percentile
        
        for col in labels_df.columns:
            if 'expPNL' in col and not any(suffix in col for suffix in ['_binary', '_multiclass']):
                upper_bound = labels_df[col].quantile(outlier_threshold)
                lower_bound = labels_df[col].quantile(1 - outlier_threshold)
                
                # Clip outliers
                processed_labels[col] = labels_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 2. Apply label smoothing for binary labels
        smoothing = self.config.label_smoothing
        
        for col in labels_df.columns:
            if '_binary' in col:
                # Smooth binary labels: 0 → smoothing, 1 → 1-smoothing
                processed_labels[col] = labels_df[col] * (1 - 2*smoothing) + smoothing
        
        # 3. Handle missing values
        processed_labels = processed_labels.fillna(0)
        
        return processed_labels
    
    def _compute_label_statistics(self, labels_df: pd.DataFrame) -> None:
        """Compute and store label statistics."""
        self.label_stats = {}
        
        for col in labels_df.columns:
            col_data = labels_df[col].dropna()
            
            self.label_stats[col] = {
                "count": len(col_data),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "q25": float(col_data.quantile(0.25)),
                "q50": float(col_data.quantile(0.50)),
                "q75": float(col_data.quantile(0.75)),
                "positive_ratio": float((col_data > 0).mean()) if 'binary' not in col else None,
                "profitable_ratio": float((col_data > self.config.min_profit_threshold).mean()) if 'expPNL' in col else None
            }
            
            # Class distribution for classification labels
            if '_binary' in col or '_multiclass' in col:
                value_counts = col_data.value_counts(normalize=True).to_dict()
                self.label_stats[col]["class_distribution"] = value_counts
    
    def get_label_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of generated labels."""
        return {
            "config": self.label_stats.__dict__,
            "generation_time": self.generation_time,
            "spike_events_detected": len(self.spike_events),
            "spike_event_types": [event['type'] for event in self.spike_events],
            "label_statistics": self.label_stats,
            "label_quality_metrics": self._compute_label_quality_metrics()
        }
    
    def _compute_label_quality_metrics(self) -> Dict[str, Any]:
        """Compute label quality metrics."""
        quality_metrics = {}
        
        for label_name, stats in self.label_stats.items():
            if 'expPNL' in label_name and '_binary' not in label_name:
                # Signal-to-noise ratio
                snr = abs(stats["mean"]) / (stats["std"] + 1e-8)
                
                # Information ratio
                positive_ratio = stats.get("positive_ratio", 0)
                info_ratio = abs(positive_ratio - 0.5) * 2  # How far from random
                
                quality_metrics[label_name] = {
                    "signal_to_noise_ratio": snr,
                    "information_ratio": info_ratio,
                    "label_balance": min(positive_ratio, 1 - positive_ratio) * 2 if positive_ratio else 0
                }
        
        return quality_metrics