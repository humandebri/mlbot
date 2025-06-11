"""
Fractional Differentiation for time series stationarity.

Based on the research from "Advances in Financial Machine Learning" by Marcos López de Prado.
This module implements fractional differentiation to achieve stationarity while preserving
as much memory/information as possible from the original time series.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from numba import jit
import warnings


class FractionalDifferentiator:
    """
    Fractional differentiation transformer for time series data.
    
    This technique allows us to achieve stationarity (required for ML models)
    while preserving more information than traditional integer differentiation.
    
    Attributes:
        d: Fractional differentiation order (0 < d < 1)
        threshold: Weight threshold for computational efficiency
        window_type: 'expanding' or 'fixed'
    """
    
    def __init__(
        self, 
        d: float = 0.5, 
        threshold: float = 1e-4,
        window_type: str = 'fixed',
        window_size: Optional[int] = None
    ):
        """
        Initialize the fractional differentiator.
        
        Args:
            d: Differentiation order (0=no diff, 1=first diff)
            threshold: Minimum weight threshold (ignore weights below this)
            window_type: 'expanding' or 'fixed' window
            window_size: Size of fixed window (required if window_type='fixed')
        """
        if not 0 <= d <= 1:
            raise ValueError("d must be between 0 and 1")
        
        self.d = d
        self.threshold = threshold
        self.window_type = window_type
        self.window_size = window_size
        
        if window_type == 'fixed' and window_size is None:
            raise ValueError("window_size must be specified for fixed window")
    
    @staticmethod
    @jit(nopython=True)
    def _get_weights(d: float, size: int) -> np.ndarray:
        """
        Calculate fractional differentiation weights using the binomial series.
        
        The weights follow: w_k = -w_{k-1} * (d - k + 1) / k
        """
        weights = np.zeros(size)
        weights[0] = 1.0
        
        for k in range(1, size):
            weights[k] = -weights[k-1] * (d - k + 1) / k
            
        return weights
    
    def _get_weights_ffd(self, d: float, threshold: float) -> np.ndarray:
        """
        Get weights for fixed-width window fractional differentiation.
        
        Stops calculation when weight magnitude falls below threshold.
        """
        weights = [1.0]
        k = 1
        
        while True:
            weight = -weights[-1] * (d - k + 1) / k
            if abs(weight) < threshold:
                break
            weights.append(weight)
            k += 1
            
        return np.array(weights[::-1])  # Reverse for oldest to newest
    
    def transform(
        self, 
        series: Union[pd.Series, np.ndarray],
        return_weights: bool = False
    ) -> Union[pd.Series, Tuple[pd.Series, np.ndarray]]:
        """
        Apply fractional differentiation to the series.
        
        Args:
            series: Input time series
            return_weights: Whether to return the weights used
            
        Returns:
            Fractionally differentiated series (and optionally weights)
        """
        if isinstance(series, pd.Series):
            values = series.values
            index = series.index
        else:
            values = series
            index = None
            
        n = len(values)
        
        if self.window_type == 'expanding':
            # Expanding window implementation
            weights = self._get_weights(self.d, n)
            
            # Apply threshold
            weight_sum = np.cumsum(np.abs(weights[::-1]))
            weight_sum /= weight_sum[-1]
            skip = np.argmax(weight_sum > self.threshold)
            
            # Calculate fractionally differentiated series
            output = np.full(n, np.nan)
            
            for i in range(skip, n):
                if i < len(weights):
                    w = weights[:i+1]
                    output[i] = np.dot(w[::-1], values[:i+1])
                else:
                    w = weights
                    output[i] = np.dot(w[::-1], values[i-len(weights)+1:i+1])
                    
        else:  # fixed window
            # Fixed window implementation
            weights = self._get_weights_ffd(self.d, self.threshold)
            window_size = len(weights)
            
            output = np.full(n, np.nan)
            
            for i in range(window_size - 1, n):
                output[i] = np.dot(weights, values[i-window_size+1:i+1])
        
        # Convert back to pandas if input was pandas
        if index is not None:
            output_series = pd.Series(output, index=index)
        else:
            output_series = output
            
        if return_weights:
            return output_series, weights
        return output_series
    
    def find_optimal_d(
        self,
        series: Union[pd.Series, np.ndarray],
        d_range: Tuple[float, float] = (0.0, 1.0),
        step: float = 0.1,
        adf_threshold: float = 0.05
    ) -> dict:
        """
        Find optimal differentiation order that achieves stationarity
        while preserving maximum information.
        
        Uses Augmented Dickey-Fuller test to check stationarity.
        
        Args:
            series: Input time series
            d_range: Range of d values to test
            step: Step size for d values
            adf_threshold: P-value threshold for ADF test
            
        Returns:
            Dictionary with optimal d and test statistics
        """
        from statsmodels.tsa.stattools import adfuller
        
        results = []
        d_values = np.arange(d_range[0], d_range[1] + step, step)
        
        for d in d_values:
            # Apply fractional differentiation
            self.d = d
            diff_series = self.transform(series)
            
            # Remove NaN values for ADF test
            clean_series = diff_series[~np.isnan(diff_series)]
            
            if len(clean_series) < 20:  # Too few observations
                continue
                
            # Perform ADF test
            try:
                adf_stat, p_value, _, _, _ = adfuller(clean_series)
                
                # Calculate correlation with original series (information preservation)
                correlation = np.corrcoef(
                    series[-len(clean_series):],
                    clean_series
                )[0, 1]
                
                results.append({
                    'd': d,
                    'adf_stat': adf_stat,
                    'p_value': p_value,
                    'correlation': correlation,
                    'is_stationary': p_value < adf_threshold
                })
            except Exception as e:
                warnings.warn(f"ADF test failed for d={d}: {e}")
                continue
        
        # Find minimum d that achieves stationarity
        stationary_results = [r for r in results if r['is_stationary']]
        
        if stationary_results:
            optimal = min(stationary_results, key=lambda x: x['d'])
        else:
            # If no stationary series found, choose d with lowest p-value
            optimal = min(results, key=lambda x: x['p_value'])
            warnings.warn("No stationary series found in the given range")
        
        return {
            'optimal_d': optimal['d'],
            'results': results,
            'optimal_stats': optimal
        }


def apply_fractional_diff_features(
    df: pd.DataFrame,
    columns: list,
    d_values: Optional[dict] = None,
    auto_optimize: bool = True
) -> pd.DataFrame:
    """
    Apply fractional differentiation to multiple features.
    
    Args:
        df: DataFrame with features
        columns: Columns to apply fractional diff
        d_values: Pre-specified d values for each column
        auto_optimize: Whether to auto-find optimal d
        
    Returns:
        DataFrame with additional fractionally differentiated features
    """
    frac_diff = FractionalDifferentiator()
    result_df = df.copy()
    
    if d_values is None:
        d_values = {}
    
    for col in columns:
        if col not in df.columns:
            warnings.warn(f"Column {col} not found in DataFrame")
            continue
            
        # Get or optimize d value
        if col in d_values:
            d = d_values[col]
        elif auto_optimize:
            print(f"Optimizing d for {col}...")
            opt_result = frac_diff.find_optimal_d(df[col])
            d = opt_result['optimal_d']
            print(f"Optimal d for {col}: {d:.3f}")
        else:
            d = 0.5  # Default value
        
        # Apply fractional differentiation
        frac_diff.d = d
        result_df[f'{col}_fracdiff_{d:.2f}'] = frac_diff.transform(df[col])
    
    return result_df