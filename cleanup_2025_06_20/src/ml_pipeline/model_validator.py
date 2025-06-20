"""
Comprehensive model validation framework for liquidation-driven trading models.

Advanced validation techniques including:
- Time series cross-validation with proper temporal splits
- Statistical significance testing
- Robustness analysis across market regimes
- Data drift detection
- Out-of-sample performance validation
- A/B testing framework for model comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.stats import ks_2samp, anderson_ksamp

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    
    # Cross-validation settings
    cv_folds: int = 5
    cv_gap: int = 0  # Gap between train and test in time series CV
    min_train_size: int = 1000
    test_size_ratio: float = 0.2
    
    # Bootstrap settings
    bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95
    
    # Statistical tests
    significance_level: float = 0.05
    min_effect_size: float = 0.01  # Minimum meaningful effect size
    
    # Robustness testing
    noise_levels: List[float] = None  # Noise levels for robustness testing
    market_regimes: List[str] = None  # Market regimes to test
    
    # Drift detection
    drift_detection_window: int = 100
    drift_threshold: float = 0.05
    ks_test_threshold: float = 0.05
    
    # Performance thresholds
    min_sharpe_ratio: float = 1.0
    max_drawdown_threshold: float = 0.15
    min_information_ratio: float = 0.5
    
    # Output settings
    save_validation_report: bool = True
    generate_plots: bool = False
    output_path: str = "validation_results"
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.01, 0.02, 0.05, 0.10]
        
        if self.market_regimes is None:
            self.market_regimes = ["bull", "bear", "sideways", "high_vol", "low_vol"]


class ModelValidator:
    """
    Comprehensive model validation framework for financial ML models.
    
    Provides rigorous validation including:
    - Time-aware cross-validation
    - Statistical significance testing
    - Robustness analysis
    - Data drift detection
    - Performance benchmarking
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize model validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        
        # Validation results storage
        self.cv_results = {}
        self.statistical_tests = {}
        self.robustness_results = {}
        self.drift_analysis = {}
        self.performance_benchmarks = {}
        
        # Output directory
        self.output_dir = Path(self.config.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Model validator initialized", config=self.config.__dict__)
    
    def validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor: Optional[Any] = None,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model validation.
        
        Args:
            model: Trained model to validate
            X: Feature matrix
            y: Target variable
            preprocessor: Optional data preprocessor
            market_data: Optional market data for regime analysis
            
        Returns:
            Comprehensive validation results
        """
        logger.info("Starting comprehensive model validation",
                   data_shape=X.shape,
                   target_stats={"mean": y.mean(), "std": y.std()})
        
        try:
            # 1. Time series cross-validation
            cv_results = self._time_series_cross_validation(model, X, y, preprocessor)
            
            # 2. Statistical significance testing
            statistical_tests = self._statistical_significance_tests(cv_results, y)
            
            # 3. Robustness analysis
            robustness_results = self._robustness_analysis(model, X, y, preprocessor)
            
            # 4. Data drift detection
            drift_analysis = self._data_drift_analysis(X, y)
            
            # 5. Market regime analysis
            regime_analysis = {}
            if market_data is not None:
                regime_analysis = self._market_regime_analysis(model, X, y, market_data, preprocessor)
            
            # 6. Performance benchmarking
            performance_benchmarks = self._performance_benchmarking(cv_results)
            
            # 7. Generate validation report
            validation_report = self._generate_validation_report(
                cv_results, statistical_tests, robustness_results,
                drift_analysis, regime_analysis, performance_benchmarks
            )
            
            # 8. Save results
            if self.config.save_validation_report:
                self._save_validation_results(validation_report)
            
            logger.info("Model validation completed successfully",
                       cv_score=cv_results.get("mean_score", 0),
                       significance_p_value=statistical_tests.get("t_test_p_value", 1),
                       passed_validation=validation_report.get("validation_passed", False))
            
            return validation_report
            
        except Exception as e:
            logger.error("Model validation failed", exception=e)
            raise
    
    def _time_series_cross_validation(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor: Optional[Any]
    ) -> Dict[str, Any]:
        """Perform time series cross-validation."""
        
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(
            n_splits=self.config.cv_folds,
            gap=self.config.cv_gap,
            test_size=int(len(X) * self.config.test_size_ratio / self.config.cv_folds)
        )
        
        cv_scores = []
        fold_results = []
        predictions_out_of_sample = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            try:
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Apply preprocessing if available
                if preprocessor:
                    X_train_processed = preprocessor.fit_transform(X_train, y_train)
                    X_test_processed = preprocessor.transform(X_test)
                else:
                    X_train_processed = X_train
                    X_test_processed = X_test
                
                # Train model
                model_copy = self._clone_model(model)
                model_copy.fit(X_train_processed, y_train)
                
                # Make predictions
                y_pred = model_copy.predict(X_test_processed)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Calculate financial metrics
                ic = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
                
                fold_result = {
                    "fold": fold,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "information_coefficient": ic,
                    "predictions": y_pred,
                    "actual": y_test.values
                }
                
                fold_results.append(fold_result)
                cv_scores.append(mse)  # Use MSE as primary metric
                predictions_out_of_sample.extend(list(zip(y_test.values, y_pred)))
                
                logger.debug(f"Fold {fold} completed",
                           train_size=len(train_idx),
                           test_size=len(test_idx),
                           mse=mse,
                           r2=r2)
                
            except Exception as e:
                logger.warning(f"Error in fold {fold}", exception=e)
                continue
        
        if not cv_scores:
            raise ValueError("All cross-validation folds failed")
        
        # Aggregate results
        cv_results = {
            "fold_results": fold_results,
            "scores": cv_scores,
            "mean_score": np.mean(cv_scores),
            "std_score": np.std(cv_scores),
            "median_score": np.median(cv_scores),
            "min_score": np.min(cv_scores),
            "max_score": np.max(cv_scores),
            "out_of_sample_predictions": predictions_out_of_sample,
            "cv_r2_scores": [f["r2"] for f in fold_results],
            "cv_ic_scores": [f["information_coefficient"] for f in fold_results]
        }
        
        return cv_results
    
    def _statistical_significance_tests(self, cv_results: Dict[str, Any], y: pd.Series) -> Dict[str, Any]:
        """Perform statistical significance tests on model performance."""
        
        scores = cv_results["scores"]
        
        # T-test against zero (null hypothesis: model has no predictive power)
        t_stat, t_p_value = stats.ttest_1samp(scores, 0)
        
        # Bootstrap confidence interval
        bootstrap_scores = []
        for _ in range(self.config.bootstrap_samples):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_scores.append(np.mean(bootstrap_sample))
        
        confidence_level = self.config.bootstrap_confidence
        lower_ci = np.percentile(bootstrap_scores, (1 - confidence_level) / 2 * 100)
        upper_ci = np.percentile(bootstrap_scores, (1 + confidence_level) / 2 * 100)
        
        # Effect size calculation (Cohen's d)
        baseline_std = y.std()
        effect_size = abs(np.mean(scores)) / baseline_std if baseline_std > 0 else 0
        
        # Shapiro-Wilk test for normality of residuals
        if cv_results["out_of_sample_predictions"]:
            actual_values = [p[0] for p in cv_results["out_of_sample_predictions"]]
            predicted_values = [p[1] for p in cv_results["out_of_sample_predictions"]]
            residuals = np.array(actual_values) - np.array(predicted_values)
            
            if len(residuals) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limit for performance
            else:
                shapiro_stat, shapiro_p = np.nan, np.nan
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        statistical_tests = {
            "t_test_statistic": t_stat,
            "t_test_p_value": t_p_value,
            "is_significant": t_p_value < self.config.significance_level,
            "effect_size": effect_size,
            "meaningful_effect": effect_size > self.config.min_effect_size,
            "bootstrap_ci_lower": lower_ci,
            "bootstrap_ci_upper": upper_ci,
            "shapiro_statistic": shapiro_stat,
            "shapiro_p_value": shapiro_p,
            "residuals_normal": shapiro_p > self.config.significance_level if not np.isnan(shapiro_p) else None
        }
        
        return statistical_tests
    
    def _robustness_analysis(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor: Optional[Any]
    ) -> Dict[str, Any]:
        """Analyze model robustness to noise and perturbations."""
        
        robustness_results = {
            "noise_sensitivity": {},
            "feature_stability": {},
            "outlier_sensitivity": {}
        }
        
        # Test sensitivity to input noise
        baseline_score = self._evaluate_model(model, X, y, preprocessor)
        
        for noise_level in self.config.noise_levels:
            noise_scores = []
            
            for _ in range(10):  # Multiple runs for statistical stability
                # Add Gaussian noise to features
                X_noisy = X.copy()
                for col in X_noisy.select_dtypes(include=[np.number]).columns:
                    noise = np.random.normal(0, noise_level * X_noisy[col].std(), size=len(X_noisy))
                    X_noisy[col] += noise
                
                noisy_score = self._evaluate_model(model, X_noisy, y, preprocessor)
                noise_scores.append(noisy_score)
            
            robustness_results["noise_sensitivity"][noise_level] = {
                "mean_score": np.mean(noise_scores),
                "std_score": np.std(noise_scores),
                "score_degradation": baseline_score - np.mean(noise_scores)
            }
        
        # Feature stability analysis (permutation importance)
        if hasattr(model, 'feature_importances_'):
            feature_importance_stability = self._analyze_feature_importance_stability(
                model, X, y, preprocessor
            )
            robustness_results["feature_stability"] = feature_importance_stability
        
        # Outlier sensitivity
        outlier_sensitivity = self._analyze_outlier_sensitivity(model, X, y, preprocessor)
        robustness_results["outlier_sensitivity"] = outlier_sensitivity
        
        return robustness_results
    
    def _data_drift_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Detect data drift over time."""
        
        drift_results = {
            "feature_drift": {},
            "target_drift": {},
            "overall_drift_score": 0.0
        }
        
        window_size = self.config.drift_detection_window
        
        if len(X) < 2 * window_size:
            logger.warning("Insufficient data for drift analysis")
            return drift_results
        
        # Split data into early and late periods
        split_point = len(X) // 2
        X_early = X.iloc[:split_point]
        X_late = X.iloc[split_point:]
        y_early = y.iloc[:split_point]
        y_late = y.iloc[split_point:]
        
        # Test for feature drift using Kolmogorov-Smirnov test
        feature_drift_scores = []
        
        for column in X.select_dtypes(include=[np.number]).columns:
            if X_early[column].std() > 0 and X_late[column].std() > 0:
                ks_stat, ks_p_value = ks_2samp(X_early[column].dropna(), X_late[column].dropna())
                
                drift_results["feature_drift"][column] = {
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_p_value,
                    "drift_detected": ks_p_value < self.config.ks_test_threshold,
                    "early_mean": X_early[column].mean(),
                    "late_mean": X_late[column].mean(),
                    "mean_shift": abs(X_late[column].mean() - X_early[column].mean()) / X_early[column].std()
                }
                
                feature_drift_scores.append(ks_stat)
        
        # Test for target drift
        if y_early.std() > 0 and y_late.std() > 0:
            target_ks_stat, target_ks_p = ks_2samp(y_early.dropna(), y_late.dropna())
            
            drift_results["target_drift"] = {
                "ks_statistic": target_ks_stat,
                "ks_p_value": target_ks_p,
                "drift_detected": target_ks_p < self.config.ks_test_threshold,
                "early_mean": y_early.mean(),
                "late_mean": y_late.mean(),
                "mean_shift": abs(y_late.mean() - y_early.mean()) / y_early.std()
            }
        
        # Calculate overall drift score
        if feature_drift_scores:
            drift_results["overall_drift_score"] = np.mean(feature_drift_scores)
        
        return drift_results
    
    def _market_regime_analysis(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        market_data: pd.DataFrame,
        preprocessor: Optional[Any]
    ) -> Dict[str, Any]:
        """Analyze model performance across different market regimes."""
        
        regime_results = {}
        
        # Define market regimes based on market data
        regimes = self._identify_market_regimes(market_data)
        
        for regime_name, regime_mask in regimes.items():
            if regime_mask.sum() < 50:  # Skip regimes with insufficient data
                continue
            
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            if len(X_regime) > 0:
                regime_score = self._evaluate_model(model, X_regime, y_regime, preprocessor)
                
                regime_results[regime_name] = {
                    "sample_size": len(X_regime),
                    "performance_score": regime_score,
                    "target_mean": y_regime.mean(),
                    "target_std": y_regime.std()
                }
        
        return regime_results
    
    def _performance_benchmarking(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark model performance against thresholds and baselines."""
        
        mean_score = cv_results["mean_score"]
        r2_scores = cv_results.get("cv_r2_scores", [])
        ic_scores = cv_results.get("cv_ic_scores", [])
        
        # Calculate Sharpe-like ratio for predictions
        if cv_results["out_of_sample_predictions"]:
            actual_values = [p[0] for p in cv_results["out_of_sample_predictions"]]
            predicted_values = [p[1] for p in cv_results["out_of_sample_predictions"]]
            
            prediction_returns = np.diff(predicted_values) / np.abs(predicted_values[:-1])
            prediction_returns = prediction_returns[np.isfinite(prediction_returns)]
            
            if len(prediction_returns) > 0:
                prediction_sharpe = np.mean(prediction_returns) / np.std(prediction_returns) if np.std(prediction_returns) > 0 else 0
            else:
                prediction_sharpe = 0
        else:
            prediction_sharpe = 0
        
        # Information ratio approximation
        avg_ic = np.mean(ic_scores) if ic_scores else 0
        ic_std = np.std(ic_scores) if ic_scores else 1
        information_ratio = avg_ic / ic_std if ic_std > 0 else 0
        
        benchmarks = {
            "mean_mse": mean_score,
            "mean_r2": np.mean(r2_scores) if r2_scores else 0,
            "mean_information_coefficient": avg_ic,
            "information_ratio": information_ratio,
            "prediction_sharpe": prediction_sharpe,
            "meets_sharpe_threshold": prediction_sharpe >= self.config.min_sharpe_ratio,
            "meets_ic_threshold": information_ratio >= self.config.min_information_ratio,
            "performance_consistency": 1 - (cv_results["std_score"] / cv_results["mean_score"]) if cv_results["mean_score"] > 0 else 0
        }
        
        return benchmarks
    
    def _evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor: Optional[Any]
    ) -> float:
        """Evaluate model performance on given data."""
        try:
            if preprocessor:
                X_processed = preprocessor.transform(X)
            else:
                X_processed = X
            
            predictions = model.predict(X_processed)
            return mean_squared_error(y, predictions)
            
        except Exception as e:
            logger.warning("Error evaluating model", exception=e)
            return float('inf')
    
    def _clone_model(self, model: Any) -> Any:
        """Clone model for cross-validation."""
        # This is a simplified clone - in practice, you'd use model-specific cloning
        try:
            from sklearn.base import clone
            return clone(model)
        except:
            # Fallback for non-sklearn models
            return model
    
    def _identify_market_regimes(self, market_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Identify market regimes based on market data."""
        regimes = {}
        
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            
            # Bull/Bear markets
            sma_50 = market_data['close'].rolling(window=50).mean()
            regimes['bull'] = market_data['close'] > sma_50
            regimes['bear'] = market_data['close'] <= sma_50
            
            # High/Low volatility
            vol_median = volatility.median()
            regimes['high_vol'] = volatility > vol_median
            regimes['low_vol'] = volatility <= vol_median
            
            # Sideways market (low trend strength)
            trend_strength = abs(returns.rolling(window=20).mean() / volatility)
            trend_median = trend_strength.median()
            regimes['sideways'] = trend_strength <= trend_median
        
        return regimes
    
    def _analyze_feature_importance_stability(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor: Optional[Any]
    ) -> Dict[str, Any]:
        """Analyze stability of feature importance across bootstrapped samples."""
        
        if not hasattr(model, 'feature_importances_'):
            return {}
        
        importance_samples = []
        
        for _ in range(50):  # Bootstrap samples
            # Bootstrap sample
            sample_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            
            # Train model
            model_copy = self._clone_model(model)
            if preprocessor:
                X_processed = preprocessor.fit_transform(X_sample, y_sample)
            else:
                X_processed = X_sample
            
            model_copy.fit(X_processed, y_sample)
            
            if hasattr(model_copy, 'feature_importances_'):
                importance_samples.append(model_copy.feature_importances_)
        
        if importance_samples:
            importance_array = np.array(importance_samples)
            
            return {
                "mean_importance": np.mean(importance_array, axis=0).tolist(),
                "std_importance": np.std(importance_array, axis=0).tolist(),
                "importance_stability": 1 - np.mean(np.std(importance_array, axis=0))
            }
        
        return {}
    
    def _analyze_outlier_sensitivity(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor: Optional[Any]
    ) -> Dict[str, Any]:
        """Analyze model sensitivity to outliers."""
        
        # Get baseline performance
        baseline_score = self._evaluate_model(model, X, y, preprocessor)
        
        # Remove top 5% outliers in target
        outlier_threshold = y.quantile(0.95)
        outlier_mask = y <= outlier_threshold
        
        X_no_outliers = X[outlier_mask]
        y_no_outliers = y[outlier_mask]
        
        no_outliers_score = self._evaluate_model(model, X_no_outliers, y_no_outliers, preprocessor)
        
        return {
            "baseline_score": baseline_score,
            "no_outliers_score": no_outliers_score,
            "outlier_sensitivity": abs(baseline_score - no_outliers_score) / baseline_score if baseline_score > 0 else 0,
            "outliers_removed": len(X) - len(X_no_outliers)
        }
    
    def _generate_validation_report(
        self,
        cv_results: Dict[str, Any],
        statistical_tests: Dict[str, Any],
        robustness_results: Dict[str, Any],
        drift_analysis: Dict[str, Any],
        regime_analysis: Dict[str, Any],
        performance_benchmarks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        # Determine if model passes validation
        validation_criteria = [
            statistical_tests.get("is_significant", False),
            statistical_tests.get("meaningful_effect", False),
            performance_benchmarks.get("meets_ic_threshold", False),
            cv_results.get("mean_score", float('inf')) < float('inf')
        ]
        
        validation_passed = all(validation_criteria)
        
        # Generate recommendations
        recommendations = []
        
        if not statistical_tests.get("is_significant", False):
            recommendations.append("Model lacks statistical significance - consider more data or feature engineering")
        
        if not statistical_tests.get("meaningful_effect", False):
            recommendations.append("Effect size is too small - model may not be practically useful")
        
        if drift_analysis.get("overall_drift_score", 0) > self.config.drift_threshold:
            recommendations.append("Significant data drift detected - model may need retraining")
        
        if performance_benchmarks.get("performance_consistency", 0) < 0.7:
            recommendations.append("Model performance is inconsistent across folds - consider regularization")
        
        return {
            "validation_passed": validation_passed,
            "cross_validation": cv_results,
            "statistical_tests": statistical_tests,
            "robustness_analysis": robustness_results,
            "data_drift": drift_analysis,
            "market_regime_analysis": regime_analysis,
            "performance_benchmarks": performance_benchmarks,
            "recommendations": recommendations,
            "validation_summary": {
                "mean_cv_score": cv_results.get("mean_score", 0),
                "statistical_significance": statistical_tests.get("is_significant", False),
                "effect_size": statistical_tests.get("effect_size", 0),
                "information_ratio": performance_benchmarks.get("information_ratio", 0),
                "drift_detected": drift_analysis.get("overall_drift_score", 0) > self.config.drift_threshold
            }
        }
    
    def _save_validation_results(self, validation_report: Dict[str, Any]) -> None:
        """Save validation results to files."""
        import json
        
        # Save main report
        report_path = self.output_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        # Save detailed CV results
        cv_path = self.output_dir / "cross_validation_results.json"
        with open(cv_path, 'w') as f:
            json.dump(validation_report["cross_validation"], f, indent=2, default=str)
        
        logger.info("Validation results saved", output_dir=str(self.output_dir))