"""
Data preprocessing pipeline for ML model training and inference.

Optimized for:
- High-frequency financial data
- Time series feature engineering
- Missing value handling
- Outlier detection and treatment
- Feature scaling and normalization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for liquidation-driven trading ML models.
    
    Features:
    - Missing value imputation with multiple strategies
    - Outlier detection and treatment (IQR, Z-score, Isolation Forest)
    - Feature scaling and normalization
    - Time series feature engineering
    - Technical indicator computation
    - Data quality validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data preprocessor.
        
        Args:
            config: Custom configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Scalers and transformers
        self.scalers = {}
        self.imputers = {}
        self.fitted = False
        
        # Feature statistics for monitoring
        self.feature_stats = {}
        self.outlier_bounds = {}
        
        # Performance tracking
        self.preprocessing_time = 0.0
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration."""
        return {
            # Missing value handling
            "missing_value_strategy": "knn",  # 'mean', 'median', 'mode', 'knn', 'forward_fill'
            "missing_threshold": 0.3,  # Drop features with >30% missing values
            
            # Outlier detection
            "outlier_method": "iqr",  # 'iqr', 'zscore', 'isolation_forest'
            "outlier_threshold": 1.5,  # IQR multiplier or Z-score threshold
            "outlier_treatment": "clip",  # 'remove', 'clip', 'transform'
            
            # Feature scaling
            "scaling_method": "robust",  # 'standard', 'robust', 'quantile', 'minmax'
            "feature_range": (-1, 1),  # For MinMax scaling
            
            # Time series features
            "rolling_windows": [5, 10, 20, 60],  # Rolling window sizes in seconds
            "lag_features": [1, 3, 5, 10],  # Lag periods
            "diff_periods": [1, 5, 10],  # Difference periods
            
            # Technical indicators
            "enable_technical_indicators": True,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "bollinger_period": 20,
            
            # Data quality
            "min_variance_threshold": 1e-6,  # Remove near-constant features
            "max_correlation_threshold": 0.95,  # Remove highly correlated features
            
            # Performance
            "enable_feature_selection": True,
            "max_features": 200,  # Maximum number of features to keep
            "validation_split": 0.2  # For feature selection validation
        }
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            data: Training feature dataframe
            target: Optional target series for supervised feature selection
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting data preprocessor", 
                   data_shape=data.shape,
                   config=self.config)
        
        start_time = pd.Timestamp.now()
        
        try:
            # 1. Basic data quality checks
            self._validate_input_data(data)
            
            # 2. Handle missing values
            data_clean = self._fit_missing_value_imputation(data)
            
            # 3. Detect and fit outlier treatment
            data_clean = self._fit_outlier_detection(data_clean)
            
            # 4. Compute time series features
            data_enhanced = self._compute_time_series_features(data_clean)
            
            # 5. Technical indicators
            if self.config["enable_technical_indicators"]:
                data_enhanced = self._compute_technical_indicators(data_enhanced)
            
            # 6. Feature scaling
            data_scaled = self._fit_feature_scaling(data_enhanced)
            
            # 7. Feature selection
            if self.config["enable_feature_selection"] and target is not None:
                selected_features = self._fit_feature_selection(data_scaled, target)
                self.selected_features = selected_features
            else:
                self.selected_features = list(data_scaled.columns)
            
            # 8. Store feature statistics
            self._compute_feature_statistics(data_scaled)
            
            self.fitted = True
            self.preprocessing_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            logger.info("Data preprocessor fitted successfully",
                       processing_time=self.preprocessing_time,
                       final_features=len(self.selected_features),
                       feature_stats=self.feature_stats)
            
            return self
            
        except Exception as e:
            logger.error("Error fitting data preprocessor", exception=e)
            raise
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed dataframe
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        start_time = pd.Timestamp.now()
        
        try:
            # Apply same transformations as fit
            data_clean = self._transform_missing_values(data)
            data_clean = self._transform_outliers(data_clean)
            data_enhanced = self._compute_time_series_features(data_clean)
            
            if self.config["enable_technical_indicators"]:
                data_enhanced = self._compute_technical_indicators(data_enhanced)
            
            data_scaled = self._transform_feature_scaling(data_enhanced)
            
            # Select only fitted features
            data_final = data_scaled[self.selected_features]
            
            transform_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            logger.debug("Data transformed successfully",
                        input_shape=data.shape,
                        output_shape=data_final.shape,
                        transform_time=transform_time)
            
            return data_final
            
        except Exception as e:
            logger.error("Error transforming data", exception=e)
            raise
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            data: Training data
            target: Optional target for supervised preprocessing
            
        Returns:
            Transformed dataframe
        """
        return self.fit(data, target).transform(data)
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data quality."""
        if data.empty:
            raise ValueError("Input data is empty")
        
        if data.shape[0] < 100:
            logger.warning("Small dataset for preprocessing", rows=data.shape[0])
        
        # Check for infinite values
        inf_cols = data.columns[np.isinf(data).any()]
        if len(inf_cols) > 0:
            logger.warning("Infinite values detected", columns=inf_cols.tolist())
            data[inf_cols] = data[inf_cols].replace([np.inf, -np.inf], np.nan)
        
        # Log missing value summary
        missing_summary = data.isnull().sum()
        missing_features = missing_summary[missing_summary > 0]
        if len(missing_features) > 0:
            logger.info("Missing values detected", 
                       missing_features=missing_features.to_dict())
    
    def _fit_missing_value_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit missing value imputation strategy."""
        strategy = self.config["missing_value_strategy"]
        threshold = self.config["missing_threshold"]
        
        # Drop features with too many missing values
        missing_pct = data.isnull().sum() / len(data)
        features_to_drop = missing_pct[missing_pct > threshold].index
        
        if len(features_to_drop) > 0:
            logger.info("Dropping features with high missing values",
                       dropped_features=features_to_drop.tolist())
            data = data.drop(columns=features_to_drop)
        
        # Fit imputers for remaining features
        numeric_features = data.select_dtypes(include=[np.number]).columns
        
        if strategy == "knn":
            self.imputers["numeric"] = KNNImputer(n_neighbors=5)
        elif strategy in ["mean", "median"]:
            self.imputers["numeric"] = SimpleImputer(strategy=strategy)
        else:
            self.imputers["numeric"] = SimpleImputer(strategy="median")
        
        if len(numeric_features) > 0:
            self.imputers["numeric"].fit(data[numeric_features])
        
        return data
    
    def _transform_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform missing values using fitted imputers."""
        numeric_features = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) > 0 and "numeric" in self.imputers:
            data[numeric_features] = self.imputers["numeric"].transform(data[numeric_features])
        
        return data
    
    def _fit_outlier_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit outlier detection and treatment."""
        method = self.config["outlier_method"]
        threshold = self.config["outlier_threshold"]
        
        numeric_features = data.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            feature_data = data[feature].dropna()
            
            if method == "iqr":
                Q1 = feature_data.quantile(0.25)
                Q3 = feature_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
            elif method == "zscore":
                mean = feature_data.mean()
                std = feature_data.std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            
            else:  # Default to IQR
                Q1 = feature_data.quantile(0.25)
                Q3 = feature_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            
            self.outlier_bounds[feature] = {
                "lower": lower_bound,
                "upper": upper_bound
            }
        
        return data
    
    def _transform_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform outliers using fitted bounds."""
        treatment = self.config["outlier_treatment"]
        
        for feature, bounds in self.outlier_bounds.items():
            if feature in data.columns:
                if treatment == "clip":
                    data[feature] = data[feature].clip(
                        lower=bounds["lower"],
                        upper=bounds["upper"]
                    )
                elif treatment == "remove":
                    # Mark outliers as NaN for imputation
                    outlier_mask = (
                        (data[feature] < bounds["lower"]) |
                        (data[feature] > bounds["upper"])
                    )
                    data.loc[outlier_mask, feature] = np.nan
        
        return data
    
    def _compute_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute time series features."""
        enhanced_data = data.copy()
        
        # Ensure we have a datetime index or timestamp column
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                enhanced_data.index = pd.to_datetime(data['timestamp'])
            else:
                logger.warning("No timestamp information found for time series features")
                return enhanced_data
        
        numeric_features = data.select_dtypes(include=[np.number]).columns
        
        # Rolling statistics
        for window in self.config["rolling_windows"]:
            for feature in numeric_features[:10]:  # Limit to first 10 features for performance
                enhanced_data[f"{feature}_roll_mean_{window}"] = data[feature].rolling(window).mean()
                enhanced_data[f"{feature}_roll_std_{window}"] = data[feature].rolling(window).std()
                enhanced_data[f"{feature}_roll_max_{window}"] = data[feature].rolling(window).max()
                enhanced_data[f"{feature}_roll_min_{window}"] = data[feature].rolling(window).min()
        
        # Lag features
        for lag in self.config["lag_features"]:
            for feature in numeric_features[:5]:  # Limit lag features
                enhanced_data[f"{feature}_lag_{lag}"] = data[feature].shift(lag)
        
        # Difference features
        for period in self.config["diff_periods"]:
            for feature in numeric_features[:5]:  # Limit difference features
                enhanced_data[f"{feature}_diff_{period}"] = data[feature].diff(period)
                enhanced_data[f"{feature}_pct_change_{period}"] = data[feature].pct_change(period)
        
        # Drop NaN values created by rolling/lag operations
        enhanced_data = enhanced_data.dropna()
        
        return enhanced_data
    
    def _compute_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators."""
        enhanced_data = data.copy()
        
        # Look for price-like features
        price_features = [col for col in data.columns if any(
            keyword in col.lower() for keyword in ['price', 'close', 'open', 'high', 'low']
        )]
        
        for price_col in price_features[:3]:  # Limit to first 3 price features
            if price_col in data.columns:
                # RSI
                rsi_period = self.config["rsi_period"]
                delta = data[price_col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                enhanced_data[f"{price_col}_rsi"] = 100 - (100 / (1 + rs))
                
                # Simple moving averages
                enhanced_data[f"{price_col}_sma_10"] = data[price_col].rolling(10).mean()
                enhanced_data[f"{price_col}_sma_20"] = data[price_col].rolling(20).mean()
                
                # Bollinger Bands
                bb_period = self.config["bollinger_period"]
                bb_sma = data[price_col].rolling(bb_period).mean()
                bb_std = data[price_col].rolling(bb_period).std()
                enhanced_data[f"{price_col}_bb_upper"] = bb_sma + (bb_std * 2)
                enhanced_data[f"{price_col}_bb_lower"] = bb_sma - (bb_std * 2)
                enhanced_data[f"{price_col}_bb_ratio"] = (data[price_col] - bb_sma) / (bb_std * 2)
        
        return enhanced_data
    
    def _fit_feature_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit feature scaling."""
        method = self.config["scaling_method"]
        numeric_features = data.select_dtypes(include=[np.number]).columns
        
        if method == "standard":
            self.scalers["numeric"] = StandardScaler()
        elif method == "robust":
            self.scalers["numeric"] = RobustScaler()
        elif method == "quantile":
            self.scalers["numeric"] = QuantileTransformer(output_distribution='uniform')
        else:
            self.scalers["numeric"] = RobustScaler()  # Default
        
        if len(numeric_features) > 0:
            self.scalers["numeric"].fit(data[numeric_features])
        
        return data
    
    def _transform_feature_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scalers."""
        scaled_data = data.copy()
        numeric_features = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) > 0 and "numeric" in self.scalers:
            scaled_data[numeric_features] = self.scalers["numeric"].transform(data[numeric_features])
        
        return scaled_data
    
    def _fit_feature_selection(self, data: pd.DataFrame, target: pd.Series) -> List[str]:
        """Fit feature selection using multiple criteria."""
        from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
        from sklearn.ensemble import RandomForestRegressor
        
        numeric_features = data.select_dtypes(include=[np.number]).columns
        max_features = min(self.config["max_features"], len(numeric_features))
        
        # 1. Remove low variance features
        variance_threshold = self.config["min_variance_threshold"]
        variances = data[numeric_features].var()
        high_variance_features = variances[variances > variance_threshold].index
        
        # 2. Remove highly correlated features
        correlation_threshold = self.config["max_correlation_threshold"]
        correlation_matrix = data[high_variance_features].corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > correlation_threshold)]
        
        uncorrelated_features = [f for f in high_variance_features if f not in to_drop]
        
        # 3. Statistical feature selection
        if len(uncorrelated_features) > max_features:
            selector = SelectKBest(score_func=f_regression, k=max_features)
            selector.fit(data[uncorrelated_features], target)
            selected_features = [uncorrelated_features[i] for i in selector.get_support(indices=True)]
        else:
            selected_features = uncorrelated_features
        
        logger.info("Feature selection completed",
                   original_features=len(numeric_features),
                   high_variance_features=len(high_variance_features),
                   uncorrelated_features=len(uncorrelated_features),
                   final_selected=len(selected_features))
        
        return list(selected_features)
    
    def _compute_feature_statistics(self, data: pd.DataFrame) -> None:
        """Compute and store feature statistics for monitoring."""
        numeric_features = data.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            self.feature_stats[feature] = {
                "mean": float(data[feature].mean()),
                "std": float(data[feature].std()),
                "min": float(data[feature].min()),
                "max": float(data[feature].max()),
                "q25": float(data[feature].quantile(0.25)),
                "q50": float(data[feature].quantile(0.50)),
                "q75": float(data[feature].quantile(0.75)),
                "skew": float(data[feature].skew()),
                "kurtosis": float(data[feature].kurtosis())
            }
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps and selected features."""
        return {
            "config": self.config,
            "fitted": self.fitted,
            "selected_features": getattr(self, 'selected_features', []),
            "feature_count": len(getattr(self, 'selected_features', [])),
            "preprocessing_time": self.preprocessing_time,
            "feature_statistics": self.feature_stats,
            "outlier_bounds": self.outlier_bounds
        }