"""
Feature binning/discretization for reducing overfitting.

This module implements various binning strategies to convert continuous features
into discrete categories, reducing noise and improving model generalization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor
import warnings


class FeatureBinner:
    """
    Advanced feature binning with multiple strategies.
    
    Supports:
    - Equal width binning
    - Equal frequency (quantile) binning
    - K-means clustering binning
    - Decision tree-based binning
    - Custom boundary binning
    """
    
    def __init__(self):
        self.bin_edges_ = {}
        self.bin_strategies_ = {}
        self.n_bins_ = {}
        
    def fit_transform(
        self,
        X: pd.DataFrame,
        features: List[str],
        strategies: Optional[Dict[str, str]] = None,
        n_bins: Union[int, Dict[str, int]] = 10,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit binning and transform features.
        
        Args:
            X: DataFrame with features
            features: List of features to bin
            strategies: Dict mapping features to strategies
            n_bins: Number of bins (global or per feature)
            y: Target variable (required for 'tree' strategy)
            
        Returns:
            DataFrame with binned features
        """
        if strategies is None:
            strategies = {feat: 'quantile' for feat in features}
            
        if isinstance(n_bins, int):
            n_bins = {feat: n_bins for feat in features}
            
        X_binned = X.copy()
        
        for feature in features:
            if feature not in X.columns:
                warnings.warn(f"Feature {feature} not found")
                continue
                
            strategy = strategies.get(feature, 'quantile')
            bins = n_bins.get(feature, 10)
            
            # Apply binning based on strategy
            if strategy == 'equal_width':
                X_binned[f'{feature}_bin'], edges = self._equal_width_binning(
                    X[feature], bins
                )
            elif strategy == 'quantile':
                X_binned[f'{feature}_bin'], edges = self._quantile_binning(
                    X[feature], bins
                )
            elif strategy == 'kmeans':
                X_binned[f'{feature}_bin'], edges = self._kmeans_binning(
                    X[feature], bins
                )
            elif strategy == 'tree':
                if y is None:
                    raise ValueError("Target y required for tree-based binning")
                X_binned[f'{feature}_bin'], edges = self._tree_binning(
                    X[feature], y, bins
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Store binning information
            self.bin_edges_[feature] = edges
            self.bin_strategies_[feature] = strategy
            self.n_bins_[feature] = bins
            
            # Add bin statistics
            X_binned = self._add_bin_statistics(X_binned, feature, X[feature])
            
        return X_binned
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted binning.
        
        Args:
            X: DataFrame with features
            
        Returns:
            DataFrame with binned features
        """
        X_binned = X.copy()
        
        for feature, edges in self.bin_edges_.items():
            if feature not in X.columns:
                continue
                
            # Apply binning using stored edges
            X_binned[f'{feature}_bin'] = pd.cut(
                X[feature],
                bins=edges,
                labels=False,
                include_lowest=True
            )
            
            # Handle values outside training range
            X_binned[f'{feature}_bin'] = X_binned[f'{feature}_bin'].fillna(-1)
            
            # Add bin statistics
            X_binned = self._add_bin_statistics(X_binned, feature, X[feature])
            
        return X_binned
    
    def _equal_width_binning(
        self, 
        series: pd.Series, 
        n_bins: int
    ) -> Tuple[pd.Series, np.ndarray]:
        """Equal width binning."""
        min_val = series.min()
        max_val = series.max()
        
        # Add small epsilon to avoid edge cases
        eps = (max_val - min_val) * 1e-10
        edges = np.linspace(min_val - eps, max_val + eps, n_bins + 1)
        
        binned = pd.cut(series, bins=edges, labels=False, include_lowest=True)
        
        return binned, edges
    
    def _quantile_binning(
        self, 
        series: pd.Series, 
        n_bins: int
    ) -> Tuple[pd.Series, np.ndarray]:
        """Quantile-based (equal frequency) binning."""
        # Remove NaN for quantile calculation
        clean_series = series.dropna()
        
        # Calculate quantiles
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = clean_series.quantile(quantiles).values
        
        # Ensure unique edges
        edges = np.unique(edges)
        
        if len(edges) < 2:
            warnings.warn(f"Too few unique values for quantile binning, using equal width")
            return self._equal_width_binning(series, n_bins)
        
        binned = pd.cut(series, bins=edges, labels=False, include_lowest=True, duplicates='drop')
        
        return binned, edges
    
    def _kmeans_binning(
        self, 
        series: pd.Series, 
        n_bins: int
    ) -> Tuple[pd.Series, np.ndarray]:
        """K-means clustering-based binning."""
        from sklearn.cluster import KMeans
        
        # Reshape for sklearn
        clean_series = series.dropna()
        X_reshape = clean_series.values.reshape(-1, 1)
        
        # Fit K-means
        kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
        kmeans.fit(X_reshape)
        
        # Get cluster boundaries
        centers = kmeans.cluster_centers_.flatten()
        centers_sorted = np.sort(centers)
        
        # Calculate edges as midpoints between centers
        edges = [series.min()]
        for i in range(len(centers_sorted) - 1):
            edges.append((centers_sorted[i] + centers_sorted[i+1]) / 2)
        edges.append(series.max())
        edges = np.array(edges)
        
        binned = pd.cut(series, bins=edges, labels=False, include_lowest=True)
        
        return binned, edges
    
    def _tree_binning(
        self, 
        series: pd.Series, 
        y: pd.Series,
        max_bins: int
    ) -> Tuple[pd.Series, np.ndarray]:
        """Decision tree-based optimal binning."""
        # Prepare data
        mask = ~(series.isna() | y.isna())
        X_clean = series[mask].values.reshape(-1, 1)
        y_clean = y[mask].values
        
        # Fit decision tree with limited leaves
        tree = DecisionTreeRegressor(
            max_leaf_nodes=max_bins,
            min_samples_leaf=len(X_clean) // (max_bins * 2),
            random_state=42
        )
        tree.fit(X_clean, y_clean)
        
        # Extract split points
        tree_splits = []
        
        def extract_splits(node_id=0):
            if tree.tree_.feature[node_id] != -2:  # Not a leaf
                tree_splits.append(tree.tree_.threshold[node_id])
                extract_splits(tree.tree_.children_left[node_id])
                extract_splits(tree.tree_.children_right[node_id])
        
        extract_splits()
        tree_splits = sorted(set(tree_splits))
        
        # Create edges
        edges = [series.min()] + tree_splits + [series.max()]
        edges = np.array(sorted(set(edges)))
        
        binned = pd.cut(series, bins=edges, labels=False, include_lowest=True)
        
        return binned, edges
    
    def _add_bin_statistics(
        self, 
        df: pd.DataFrame, 
        feature: str,
        original_values: pd.Series
    ) -> pd.DataFrame:
        """Add bin-related statistics as features."""
        bin_col = f'{feature}_bin'
        
        if bin_col not in df.columns:
            return df
        
        # Bin count (frequency)
        bin_counts = df[bin_col].value_counts()
        df[f'{feature}_bin_count'] = df[bin_col].map(bin_counts)
        
        # Bin mean value
        bin_means = df.groupby(bin_col)[feature].transform('mean')
        df[f'{feature}_bin_mean'] = bin_means
        
        # Distance from bin center
        df[f'{feature}_bin_distance'] = abs(original_values - bin_means)
        
        # Bin range (width)
        if feature in self.bin_edges_:
            edges = self.bin_edges_[feature]
            bin_widths = {}
            for i in range(len(edges) - 1):
                bin_widths[i] = edges[i+1] - edges[i]
            df[f'{feature}_bin_width'] = df[bin_col].map(bin_widths)
        
        return df


class AdaptiveBinner:
    """
    Adaptive binning that adjusts bins based on data distribution and importance.
    """
    
    def __init__(self, min_samples_per_bin: int = 30):
        self.min_samples_per_bin = min_samples_per_bin
        self.feature_importance_ = {}
        self.optimal_bins_ = {}
        
    def find_optimal_bins(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str],
        max_bins: int = 20,
        cv_folds: int = 3
    ) -> Dict[str, int]:
        """
        Find optimal number of bins for each feature using cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            features: Features to optimize
            max_bins: Maximum number of bins to try
            cv_folds: Number of CV folds
            
        Returns:
            Dict mapping features to optimal number of bins
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestRegressor
        
        optimal_bins = {}
        
        for feature in features:
            if feature not in X.columns:
                continue
                
            scores = []
            n_bins_range = range(2, min(max_bins + 1, 
                                        len(X[feature].unique())))
            
            for n_bins in n_bins_range:
                # Create binned feature
                binner = FeatureBinner()
                X_temp = X[[feature]].copy()
                X_binned = binner.fit_transform(
                    X_temp, 
                    [feature], 
                    n_bins=n_bins
                )
                
                # Evaluate with simple model
                rf = RandomForestRegressor(
                    n_estimators=50, 
                    max_depth=5,
                    random_state=42
                )
                
                # Use binned feature and bin statistics
                feature_cols = [col for col in X_binned.columns 
                               if col.startswith(feature)]
                
                cv_score = cross_val_score(
                    rf, 
                    X_binned[feature_cols], 
                    y,
                    cv=cv_folds,
                    scoring='neg_mean_squared_error'
                ).mean()
                
                scores.append((n_bins, -cv_score))  # Convert to positive
            
            # Find optimal bins (minimum MSE)
            optimal_n = min(scores, key=lambda x: x[1])[0]
            optimal_bins[feature] = optimal_n
            
            print(f"Optimal bins for {feature}: {optimal_n}")
            
        self.optimal_bins_ = optimal_bins
        return optimal_bins


def create_interaction_bins(
    df: pd.DataFrame,
    feature_pairs: List[Tuple[str, str]],
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Create binned interaction features for pairs of variables.
    
    Args:
        df: Input DataFrame
        feature_pairs: List of (feature1, feature2) tuples
        n_bins: Number of bins for each dimension
        
    Returns:
        DataFrame with interaction bins
    """
    result_df = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 not in df.columns or feat2 not in df.columns:
            continue
            
        # Create 2D bins
        # First bin each feature
        bins1 = pd.qcut(df[feat1], q=n_bins, labels=False, duplicates='drop')
        bins2 = pd.qcut(df[feat2], q=n_bins, labels=False, duplicates='drop')
        
        # Create interaction bin
        interaction_bin = bins1 * n_bins + bins2
        result_df[f'{feat1}_{feat2}_interaction_bin'] = interaction_bin
        
        # Add interaction statistics
        interaction_mean = df.groupby(interaction_bin)[feat1].transform('mean') * \
                          df.groupby(interaction_bin)[feat2].transform('mean')
        result_df[f'{feat1}_{feat2}_interaction_mean'] = interaction_mean
        
    return result_df