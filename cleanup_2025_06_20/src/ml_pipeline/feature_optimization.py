"""
Feature optimization and engineering for liquidation-driven trading ML models.

Advanced feature selection, transformation, and optimization techniques to maximize
model performance while maintaining computational efficiency for real-time inference.

Features:
- Multi-strategy feature selection (statistical, ML-based, sequential)
- Advanced feature engineering (interactions, transformations)
- Hyperparameter optimization with Optuna
- Real-time inference optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_regression, mutual_info_regression, chi2
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit
import optuna

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for feature optimization."""
    
    # Feature selection
    max_features: int = 100
    min_features: int = 20
    selection_methods: List[str] = None
    selection_percentile: float = 50.0
    
    # Feature engineering
    enable_interactions: bool = True
    interaction_degree: int = 2
    enable_polynomials: bool = False
    polynomial_degree: int = 2
    
    # Dimensionality reduction
    enable_pca: bool = False
    pca_variance_threshold: float = 0.95
    enable_ica: bool = False
    ica_components: int = 50
    
    # Optimization
    optuna_trials: int = 100
    cv_folds: int = 5
    scoring_metric: str = "neg_mean_squared_error"
    
    # Performance constraints
    max_computation_time_ms: float = 5.0  # Max feature computation time
    memory_limit_mb: float = 512.0        # Max memory usage
    
    # Real-time optimization
    enable_fast_features: bool = True
    prioritize_speed: bool = True
    batch_processing: bool = True
    
    def __post_init__(self):
        if self.selection_methods is None:
            self.selection_methods = [
                "statistical", "tree_based", "lasso", "rfe"
            ]


class FeatureOptimizer:
    """
    Advanced feature optimization for high-frequency trading ML models.
    
    Optimizes features for:
    - Predictive power maximization
    - Computational efficiency
    - Real-time inference speed
    - Memory usage minimization
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize feature optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        
        # Store fitted components
        self.fitted_selectors = {}
        self.fitted_transformers = {}
        self.feature_rankings = {}
        self.optimization_results = {}
        
        # Performance tracking
        self.optimization_time = 0.0
        self.selected_features = []
        self.feature_importance_scores = {}
        
        logger.info("Feature optimizer initialized", config=self.config.__dict__)
    
    def optimize_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive feature optimization pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Fraction of data for validation
            
        Returns:
            Tuple of (optimized_features, optimization_report)
        """
        logger.info("Starting feature optimization",
                   input_features=X.shape[1],
                   samples=X.shape[0],
                   target_stats={"mean": y.mean(), "std": y.std()})
        
        start_time = pd.Timestamp.now()
        
        try:
            # 1. Data validation
            X_clean, y_clean = self._validate_and_clean_data(X, y)
            
            # 2. Split data for optimization
            split_idx = int(len(X_clean) * (1 - validation_split))
            X_train, X_val = X_clean[:split_idx], X_clean[split_idx:]
            y_train, y_val = y_clean[:split_idx], y_clean[split_idx:]
            
            # 3. Generate engineered features
            X_engineered = self._engineer_features(X_train)
            
            # 4. Apply multiple feature selection strategies
            selected_features = self._multi_strategy_feature_selection(
                X_engineered, y_train
            )
            
            # 5. Optimize feature subset with Optuna
            if len(selected_features) > self.config.max_features:
                optimized_features = self._optuna_feature_optimization(
                    X_engineered[selected_features], y_train, X_val, y_val
                )
            else:
                optimized_features = selected_features
            
            # 6. Apply dimensionality reduction if needed
            if self.config.enable_pca or self.config.enable_ica:
                X_final, optimized_features = self._apply_dimensionality_reduction(
                    X_engineered[optimized_features], optimized_features
                )
            else:
                X_final = X_engineered[optimized_features]
            
            # 7. Validate performance constraints
            self._validate_performance_constraints(X_final)
            
            # 8. Generate optimization report
            optimization_report = self._generate_optimization_report(
                X, X_final, optimized_features, y_train
            )
            
            self.optimization_time = (pd.Timestamp.now() - start_time).total_seconds()
            self.selected_features = optimized_features
            
            logger.info("Feature optimization completed",
                       final_features=len(optimized_features),
                       optimization_time=self.optimization_time,
                       feature_reduction=f"{X.shape[1]} → {len(optimized_features)}")
            
            return X_final, optimization_report
            
        except Exception as e:
            logger.error("Error in feature optimization", exception=e)
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted optimization pipeline.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if not self.selected_features:
            raise ValueError("Optimizer must be fitted before transform")
        
        try:
            # Apply same transformations as fit
            X_engineered = self._engineer_features(X, fit=False)
            
            # Select optimized features
            X_selected = X_engineered[self.selected_features]
            
            # Apply dimensionality reduction if fitted
            if self.config.enable_pca and 'pca' in self.fitted_transformers:
                X_transformed = pd.DataFrame(
                    self.fitted_transformers['pca'].transform(X_selected),
                    index=X_selected.index,
                    columns=[f"pca_{i}" for i in range(self.fitted_transformers['pca'].n_components_)]
                )
                return X_transformed
            
            if self.config.enable_ica and 'ica' in self.fitted_transformers:
                X_transformed = pd.DataFrame(
                    self.fitted_transformers['ica'].transform(X_selected),
                    index=X_selected.index,
                    columns=[f"ica_{i}" for i in range(self.fitted_transformers['ica'].n_components_)]
                )
                return X_transformed
            
            return X_selected
            
        except Exception as e:
            logger.error("Error transforming features", exception=e)
            raise
    
    def _validate_and_clean_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and clean input data."""
        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx].copy()
        y_clean = y[valid_idx].copy()
        
        # Remove infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(X_clean.median())
        
        # Ensure numeric data
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        X_clean = X_clean[numeric_cols]
        
        logger.info("Data validation completed",
                   original_samples=len(X),
                   clean_samples=len(X_clean),
                   features=len(X_clean.columns))
        
        return X_clean, y_clean
    
    def _engineer_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Generate engineered features."""
        engineered_features = X.copy()
        
        # 1. Interaction features
        if self.config.enable_interactions:
            engineered_features = self._generate_interaction_features(
                engineered_features, fit=fit
            )
        
        # 2. Polynomial features (limited to avoid explosion)
        if self.config.enable_polynomials:
            engineered_features = self._generate_polynomial_features(
                engineered_features, fit=fit
            )
        
        # 3. Financial-specific features
        engineered_features = self._generate_financial_features(engineered_features)
        
        # 4. Time-based features
        engineered_features = self._generate_temporal_features(engineered_features)
        
        return engineered_features
    
    def _generate_interaction_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Generate interaction features between important variables."""
        if not self.config.enable_interactions:
            return X
        
        # Select top features for interactions (to limit feature explosion)
        if fit:
            # Use variance as a simple importance measure
            feature_importance = X.var().sort_values(ascending=False)
            top_features = feature_importance.head(20).index.tolist()
            self.fitted_transformers['interaction_features'] = top_features
        else:
            top_features = self.fitted_transformers.get('interaction_features', X.columns[:20])
        
        interaction_data = X.copy()
        
        # Generate pairwise interactions for top features
        for i, feat1 in enumerate(top_features[:10]):  # Limit to first 10
            for feat2 in top_features[i+1:]:
                if feat1 in X.columns and feat2 in X.columns:
                    # Multiplication interaction
                    interaction_data[f"{feat1}_x_{feat2}"] = X[feat1] * X[feat2]
                    
                    # Ratio interaction (with small epsilon for stability)
                    epsilon = 1e-8
                    interaction_data[f"{feat1}_div_{feat2}"] = X[feat1] / (X[feat2] + epsilon)
        
        return interaction_data
    
    def _generate_polynomial_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Generate polynomial features for top variables."""
        if not self.config.enable_polynomials:
            return X
        
        # Limit polynomial features to top 5 features to avoid explosion
        if fit:
            feature_importance = X.var().sort_values(ascending=False)
            poly_features = feature_importance.head(5).index.tolist()
            self.fitted_transformers['poly_features'] = poly_features
        else:
            poly_features = self.fitted_transformers.get('poly_features', X.columns[:5])
        
        poly_data = X.copy()
        
        for feature in poly_features:
            if feature in X.columns:
                # Square terms
                poly_data[f"{feature}_squared"] = X[feature] ** 2
                
                # Square root (for positive values)
                positive_values = X[feature] > 0
                poly_data[f"{feature}_sqrt"] = np.where(
                    positive_values, np.sqrt(X[feature]), 0
                )
                
                # Log transform (for positive values)
                poly_data[f"{feature}_log"] = np.where(
                    positive_values, np.log(X[feature] + 1), 0
                )
        
        return poly_data
    
    def _generate_financial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate finance-specific features."""
        financial_data = X.copy()
        
        # Look for price and volume related features
        price_features = [col for col in X.columns if any(
            keyword in col.lower() for keyword in ['price', 'close', 'open', 'high', 'low']
        )]
        
        volume_features = [col for col in X.columns if any(
            keyword in col.lower() for keyword in ['volume', 'size', 'amount']
        )]
        
        # Price-based features
        for price_col in price_features[:3]:  # Limit to first 3
            if price_col in X.columns:
                # Momentum indicators
                financial_data[f"{price_col}_momentum_3"] = X[price_col].pct_change(3)
                financial_data[f"{price_col}_momentum_5"] = X[price_col].pct_change(5)
                
                # Volatility proxy
                financial_data[f"{price_col}_volatility"] = X[price_col].rolling(10).std()
        
        # Volume-based features
        for vol_col in volume_features[:3]:  # Limit to first 3
            if vol_col in X.columns:
                # Volume trend
                financial_data[f"{vol_col}_trend"] = X[vol_col].rolling(5).mean() / X[vol_col].rolling(20).mean()
                
                # Volume acceleration
                financial_data[f"{vol_col}_acceleration"] = X[vol_col].diff().diff()
        
        return financial_data
    
    def _generate_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features."""
        temporal_data = X.copy()
        
        # If we have a datetime index, extract time features
        if isinstance(X.index, pd.DatetimeIndex):
            temporal_data['hour'] = X.index.hour
            temporal_data['minute'] = X.index.minute
            temporal_data['day_of_week'] = X.index.dayofweek
            
            # Cyclical encoding
            temporal_data['hour_sin'] = np.sin(2 * np.pi * X.index.hour / 24)
            temporal_data['hour_cos'] = np.cos(2 * np.pi * X.index.hour / 24)
            temporal_data['minute_sin'] = np.sin(2 * np.pi * X.index.minute / 60)
            temporal_data['minute_cos'] = np.cos(2 * np.pi * X.index.minute / 60)
        
        return temporal_data
    
    def _multi_strategy_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Apply multiple feature selection strategies and combine results."""
        all_selected_features = set()
        
        # 1. Statistical feature selection
        if "statistical" in self.config.selection_methods:
            selector = SelectKBest(score_func=f_regression, k=min(50, X.shape[1]//2))
            selector.fit(X, y)
            stat_features = X.columns[selector.get_support()].tolist()
            all_selected_features.update(stat_features)
            self.fitted_selectors['statistical'] = selector
        
        # 2. Tree-based feature selection
        if "tree_based" in self.config.selection_methods:
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            # Select features above median importance
            importance_threshold = np.median(rf.feature_importances_)
            tree_features = X.columns[rf.feature_importances_ > importance_threshold].tolist()
            all_selected_features.update(tree_features)
            self.fitted_selectors['tree_based'] = rf
        
        # 3. LASSO feature selection
        if "lasso" in self.config.selection_methods:
            lasso = LassoCV(cv=3, random_state=42, n_jobs=-1)
            lasso.fit(X, y)
            
            # Select features with non-zero coefficients
            lasso_features = X.columns[lasso.coef_ != 0].tolist()
            all_selected_features.update(lasso_features)
            self.fitted_selectors['lasso'] = lasso
        
        # 4. Recursive Feature Elimination
        if "rfe" in self.config.selection_methods:
            estimator = RandomForestRegressor(n_estimators=20, random_state=42)
            rfe = RFE(estimator, n_features_to_select=min(30, X.shape[1]//3))
            rfe.fit(X, y)
            
            rfe_features = X.columns[rfe.support_].tolist()
            all_selected_features.update(rfe_features)
            self.fitted_selectors['rfe'] = rfe
        
        # Combine and rank features
        selected_features = list(all_selected_features)
        
        # Rank features by frequency of selection
        feature_votes = {}
        for method in self.config.selection_methods:
            if method in self.fitted_selectors:
                if method == "statistical":
                    selected = X.columns[self.fitted_selectors[method].get_support()]
                elif method == "tree_based":
                    importance_threshold = np.median(self.fitted_selectors[method].feature_importances_)
                    selected = X.columns[self.fitted_selectors[method].feature_importances_ > importance_threshold]
                elif method == "lasso":
                    selected = X.columns[self.fitted_selectors[method].coef_ != 0]
                elif method == "rfe":
                    selected = X.columns[self.fitted_selectors[method].support_]
                
                for feature in selected:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Sort by votes and take top features
        ranked_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, _ in ranked_features[:self.config.max_features]]
        
        logger.info("Multi-strategy feature selection completed",
                   methods=self.config.selection_methods,
                   total_selected=len(selected_features),
                   feature_votes=dict(ranked_features[:10]))
        
        return selected_features
    
    def _optuna_feature_optimization(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series
    ) -> List[str]:
        """Optimize feature subset using Optuna."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        
        def objective(trial):
            # Select subset of features
            n_features = trial.suggest_int("n_features", 
                                         self.config.min_features, 
                                         min(self.config.max_features, len(X_train.columns)))
            
            # Use importance-based selection with random sampling
            if hasattr(self, 'fitted_selectors') and 'tree_based' in self.fitted_selectors:
                rf = self.fitted_selectors['tree_based']
                feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
                
                # Sample features based on importance (higher importance = higher probability)
                probabilities = feature_importance / feature_importance.sum()
                selected_features = np.random.choice(
                    X_train.columns, 
                    size=n_features, 
                    replace=False, 
                    p=probabilities
                ).tolist()
            else:
                # Random selection fallback
                selected_features = np.random.choice(
                    X_train.columns, 
                    size=n_features, 
                    replace=False
                ).tolist()
            
            # Train model with selected features
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 10, 100),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train[selected_features], y_train)
            y_pred = model.predict(X_val[selected_features])
            
            # Optimize for MSE
            mse = mean_squared_error(y_val, y_pred)
            
            # Add penalty for too many features (encourage parsimony)
            feature_penalty = len(selected_features) * 0.001
            
            return mse + feature_penalty
        
        # Run optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config.optuna_trials, show_progress_bar=False)
        
        # Get best features
        best_trial = study.best_trial
        best_n_features = best_trial.params["n_features"]
        
        # Re-run with best parameters
        if hasattr(self, 'fitted_selectors') and 'tree_based' in self.fitted_selectors:
            rf = self.fitted_selectors['tree_based']
            feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
            best_features = feature_importance.nlargest(best_n_features).index.tolist()
        else:
            best_features = X_train.columns[:best_n_features].tolist()
        
        self.optimization_results['optuna'] = {
            "best_score": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials)
        }
        
        logger.info("Optuna feature optimization completed",
                   best_score=study.best_value,
                   best_features=best_n_features,
                   trials=len(study.trials))
        
        return best_features
    
    def _apply_dimensionality_reduction(
        self, 
        X: pd.DataFrame, 
        feature_names: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Apply dimensionality reduction techniques."""
        reduced_X = X.copy()
        reduced_features = feature_names.copy()
        
        # PCA
        if self.config.enable_pca:
            pca = PCA()
            pca.fit(X)
            
            # Find number of components for desired variance
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= self.config.pca_variance_threshold) + 1
            
            pca_final = PCA(n_components=n_components)
            X_pca = pca_final.fit_transform(X)
            
            reduced_X = pd.DataFrame(
                X_pca,
                index=X.index,
                columns=[f"pca_{i}" for i in range(n_components)]
            )
            reduced_features = list(reduced_X.columns)
            self.fitted_transformers['pca'] = pca_final
            
            logger.info("PCA applied",
                       original_features=len(feature_names),
                       pca_components=n_components,
                       variance_explained=cumsum_variance[n_components-1])
        
        # ICA (if not using PCA)
        elif self.config.enable_ica:
            n_components = min(self.config.ica_components, X.shape[1])
            ica = FastICA(n_components=n_components, random_state=42)
            X_ica = ica.fit_transform(X)
            
            reduced_X = pd.DataFrame(
                X_ica,
                index=X.index,
                columns=[f"ica_{i}" for i in range(n_components)]
            )
            reduced_features = list(reduced_X.columns)
            self.fitted_transformers['ica'] = ica
            
            logger.info("ICA applied",
                       original_features=len(feature_names),
                       ica_components=n_components)
        
        return reduced_X, reduced_features
    
    def _validate_performance_constraints(self, X: pd.DataFrame) -> None:
        """Validate that feature set meets performance constraints."""
        import psutil
        import time
        
        # Test computation time
        start_time = time.perf_counter()
        _ = X.sum(axis=1)  # Simple computation test
        computation_time = (time.perf_counter() - start_time) * 1000  # ms
        
        if computation_time > self.config.max_computation_time_ms:
            logger.warning("Feature computation time exceeds constraint",
                          actual_time=computation_time,
                          max_allowed=self.config.max_computation_time_ms)
        
        # Test memory usage
        memory_usage = X.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        if memory_usage > self.config.memory_limit_mb:
            logger.warning("Feature memory usage exceeds constraint",
                          actual_memory=memory_usage,
                          max_allowed=self.config.memory_limit_mb)
        
        logger.info("Performance validation completed",
                   computation_time_ms=computation_time,
                   memory_usage_mb=memory_usage)
    
    def _generate_optimization_report(
        self, 
        X_original: pd.DataFrame,
        X_final: pd.DataFrame, 
        selected_features: List[str],
        y: pd.Series
    ) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        return {
            "feature_reduction": {
                "original_features": X_original.shape[1],
                "final_features": X_final.shape[1],
                "reduction_ratio": 1 - (X_final.shape[1] / X_original.shape[1])
            },
            "selected_features": selected_features,
            "selection_methods": self.config.selection_methods,
            "optimization_time": self.optimization_time,
            "optimization_results": self.optimization_results,
            "feature_importance_scores": self.feature_importance_scores,
            "performance_metrics": {
                "computation_time_ms": getattr(self, 'computation_time', 0),
                "memory_usage_mb": getattr(self, 'memory_usage', 0)
            },
            "configuration": self.config.__dict__
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from fitted selectors."""
        importance_scores = {}
        
        if 'tree_based' in self.fitted_selectors:
            rf = self.fitted_selectors['tree_based']
            for feature, importance in zip(rf.feature_names_in_, rf.feature_importances_):
                importance_scores[feature] = float(importance)
        
        return importance_scores