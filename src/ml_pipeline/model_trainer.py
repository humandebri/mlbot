"""
CatBoost model training system with hyperparameter optimization and ONNX conversion.

High-performance training pipeline optimized for liquidation-driven trading models
with time series validation, feature importance analysis, and production deployment.

Features:
- CatBoost training with automatic hyperparameter optimization
- Time series cross-validation
- ONNX conversion for fast inference
- Model versioning and performance tracking
- Production-ready model deployment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

import catboost as cb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import optuna
import onnx
import onnxruntime as ort

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Advanced CatBoost model training system for liquidation-driven trading.
    
    Optimized for:
    - Time series financial data
    - High-frequency trading requirements
    - Production deployment
    - Real-time inference speed
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Model storage
        self.best_model = None
        self.models = {}  # Store multiple model versions
        self.training_history = []
        self.feature_importance = {}
        
        # Performance tracking
        self.training_time = 0.0
        self.validation_scores = {}
        self.optimization_results = {}
        
        # Paths
        self.model_dir = Path(self.config["model_save_path"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Model trainer initialized", config=self.config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            # Model parameters
            "objective": "RMSE",  # RMSE, MAE, MultiClass, Logloss
            "eval_metric": "RMSE",
            "iterations": 1000,
            "learning_rate": 0.1,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "bagging_temperature": 1.0,
            "border_count": 128,
            
            # Training parameters
            "early_stopping_rounds": 100,
            "use_best_model": True,
            "verbose": False,
            "random_seed": 42,
            "thread_count": -1,
            
            # Cross-validation
            "cv_folds": 5,
            "cv_gap": 0,  # Gap between train and validation for time series
            "test_size": 0.2,
            "validation_size": 0.1,
            
            # Hyperparameter optimization
            "enable_optuna": True,
            "optuna_trials": 100,
            "optuna_timeout": 3600,  # 1 hour
            "optuna_direction": "minimize",
            
            # Feature handling
            "auto_class_weights": True,
            "categorical_features": [],
            "text_features": [],
            "feature_selection_threshold": 0.001,
            
            # Model saving
            "model_save_path": "models",
            "save_model_format": ["cbm", "onnx"],  # CatBoost model, ONNX
            "model_version": "v1.0",
            
            # Performance
            "gpu_training": False,
            "task_type": "CPU",  # CPU or GPU
            "devices": "0",
            
            # Production optimization
            "optimize_for_inference": True,
            "target_inference_time_ms": 1.0,
            "max_model_size_mb": 100.0,
            
            # Monitoring
            "track_feature_importance": True,
            "save_training_plots": True,
            "enable_shap": False  # SHAP analysis (can be slow)
        }
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train CatBoost model with comprehensive optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Sample weights (optional)
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting model training",
                   train_shape=X_train.shape,
                   target_stats={"mean": y_train.mean(), "std": y_train.std()},
                   config=self.config)
        
        start_time = pd.Timestamp.now()
        
        try:
            # 1. Prepare data
            train_pool, val_pool = self._prepare_catboost_pools(
                X_train, y_train, X_val, y_val, sample_weight
            )
            
            # 2. Optimize hyperparameters if enabled
            if self.config["enable_optuna"]:
                best_params = self._optimize_hyperparameters(train_pool, val_pool)
                self.config.update(best_params)
            
            # 3. Train final model with best parameters
            final_model = self._train_final_model(train_pool, val_pool)
            
            # 4. Evaluate model performance
            train_scores = self._evaluate_model(final_model, train_pool, "train")
            val_scores = self._evaluate_model(final_model, val_pool, "validation") if val_pool else {}
            
            # 5. Analyze feature importance
            feature_importance = self._analyze_feature_importance(final_model, X_train.columns)
            
            # 6. Convert to ONNX if requested
            onnx_model = None
            if "onnx" in self.config["save_model_format"]:
                onnx_model = self._convert_to_onnx(final_model, X_train)
            
            # 7. Save model
            model_paths = self._save_model(final_model, onnx_model)
            
            # 8. Generate training report
            training_results = self._generate_training_report(
                final_model, train_scores, val_scores, feature_importance, model_paths
            )
            
            self.best_model = final_model
            self.feature_importance = feature_importance
            self.training_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            logger.info("Model training completed successfully",
                       training_time=self.training_time,
                       train_score=train_scores.get("rmse", 0),
                       val_score=val_scores.get("rmse", 0),
                       feature_count=len(feature_importance))
            
            return training_results
            
        except Exception as e:
            logger.error("Error in model training", exception=e)
            raise
    
    def _prepare_catboost_pools(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        sample_weight: Optional[np.ndarray]
    ) -> Tuple[cb.Pool, Optional[cb.Pool]]:
        """Prepare CatBoost data pools."""
        
        # Identify categorical features
        categorical_features = []
        for col in X_train.columns:
            if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                categorical_features.append(col)
        
        # Create training pool
        train_pool = cb.Pool(
            data=X_train,
            label=y_train,
            weight=sample_weight,
            cat_features=categorical_features + self.config["categorical_features"],
            feature_names=list(X_train.columns)
        )
        
        # Create validation pool
        val_pool = None
        if X_val is not None and y_val is not None:
            val_pool = cb.Pool(
                data=X_val,
                label=y_val,
                cat_features=categorical_features + self.config["categorical_features"],
                feature_names=list(X_val.columns)
            )
        
        return train_pool, val_pool
    
    def _optimize_hyperparameters(
        self, 
        train_pool: cb.Pool, 
        val_pool: Optional[cb.Pool]
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            params = {
                "iterations": self.config["iterations"],
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "random_strength": trial.suggest_float("random_strength", 0.1, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
                "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 255]),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                
                # Fixed parameters
                "objective": self.config["objective"],
                "eval_metric": self.config["eval_metric"],
                "early_stopping_rounds": self.config["early_stopping_rounds"],
                "use_best_model": True,
                "verbose": False,
                "random_seed": self.config["random_seed"],
                "thread_count": self.config["thread_count"],
                "task_type": self.config["task_type"]
            }
            
            # Train model
            model = cb.CatBoostRegressor(**params)
            
            if val_pool:
                model.fit(train_pool, eval_set=val_pool, verbose=False)
                # Get validation score
                val_predictions = model.predict(val_pool)
                val_actual = val_pool.get_label()
                score = mean_squared_error(val_actual, val_predictions)
            else:
                # Use cross-validation
                cv_results = cb.cv(
                    params=params,
                    pool=train_pool,
                    fold_count=self.config["cv_folds"],
                    shuffle=False,  # Important for time series
                    partition_random_seed=self.config["random_seed"],
                    plot=False,
                    verbose=False
                )
                score = cv_results.iloc[-1]['test-RMSE-mean']
            
            return score
        
        # Run optimization
        study = optuna.create_study(direction=self.config["optuna_direction"])
        study.optimize(
            objective, 
            n_trials=self.config["optuna_trials"],
            timeout=self.config["optuna_timeout"],
            show_progress_bar=False
        )
        
        best_params = study.best_params
        self.optimization_results = {
            "best_score": study.best_value,
            "best_params": best_params,
            "n_trials": len(study.trials),
            "optimization_time": study.trials[-1].datetime_complete - study.trials[0].datetime_start
        }
        
        logger.info("Hyperparameter optimization completed",
                   best_score=study.best_value,
                   best_params=best_params,
                   trials=len(study.trials))
        
        return best_params
    
    def _train_final_model(self, train_pool: cb.Pool, val_pool: Optional[cb.Pool]) -> cb.CatBoostRegressor:
        """Train final model with optimized parameters."""
        
        # Prepare model parameters
        model_params = {
            "objective": self.config["objective"],
            "eval_metric": self.config["eval_metric"],
            "iterations": self.config["iterations"],
            "learning_rate": self.config["learning_rate"],
            "depth": self.config["depth"],
            "l2_leaf_reg": self.config["l2_leaf_reg"],
            "random_strength": self.config["random_strength"],
            "bagging_temperature": self.config["bagging_temperature"],
            "border_count": self.config["border_count"],
            "early_stopping_rounds": self.config["early_stopping_rounds"],
            "use_best_model": self.config["use_best_model"],
            "verbose": self.config["verbose"],
            "random_seed": self.config["random_seed"],
            "thread_count": self.config["thread_count"],
            "task_type": self.config["task_type"]
        }
        
        # Add GPU support if enabled
        if self.config["gpu_training"] and self.config["task_type"] == "GPU":
            model_params["devices"] = self.config["devices"]
        
        # Add auto class weights for classification
        if self.config["auto_class_weights"] and "Class" in self.config["objective"]:
            model_params["auto_class_weights"] = "Balanced"
        
        # Create and train model
        model = cb.CatBoostRegressor(**model_params)
        
        # Train with validation set if available
        if val_pool:
            model.fit(train_pool, eval_set=val_pool, verbose=self.config["verbose"])
        else:
            model.fit(train_pool, verbose=self.config["verbose"])
        
        return model
    
    def _evaluate_model(self, model: cb.CatBoostRegressor, pool: cb.Pool, dataset_name: str) -> Dict[str, float]:
        """Evaluate model performance on given dataset."""
        predictions = model.predict(pool)
        actual = pool.get_label()
        
        # Regression metrics
        metrics = {
            "rmse": np.sqrt(mean_squared_error(actual, predictions)),
            "mae": mean_absolute_error(actual, predictions),
            "r2": r2_score(actual, predictions),
            "mse": mean_squared_error(actual, predictions)
        }
        
        # Additional financial metrics
        returns_actual = np.diff(actual) / actual[:-1]
        returns_pred = np.diff(predictions) / predictions[:-1]
        
        if len(returns_actual) > 0 and len(returns_pred) > 0:
            # Direction accuracy
            direction_actual = np.sign(returns_actual)
            direction_pred = np.sign(returns_pred)
            metrics["direction_accuracy"] = np.mean(direction_actual == direction_pred)
            
            # Information coefficient
            ic = np.corrcoef(returns_actual, returns_pred)[0, 1]
            metrics["information_coefficient"] = ic if not np.isnan(ic) else 0.0
        
        logger.info(f"Model evaluation on {dataset_name}",
                   **{k: round(v, 4) for k, v in metrics.items()})
        
        return metrics
    
    def _analyze_feature_importance(self, model: cb.CatBoostRegressor, feature_names: List[str]) -> Dict[str, float]:
        """Analyze feature importance."""
        
        # Get feature importance
        importance_scores = model.get_feature_importance()
        
        # Create importance dictionary
        feature_importance = {}
        for feature, importance in zip(feature_names, importance_scores):
            feature_importance[feature] = float(importance)
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        # Log top features
        top_features = list(feature_importance.items())[:10]
        logger.info("Top 10 feature importance", top_features=top_features)
        
        return feature_importance
    
    def _convert_to_onnx(self, model: cb.CatBoostRegressor, X_sample: pd.DataFrame) -> Optional[bytes]:
        """Convert CatBoost model to ONNX format."""
        try:
            import tempfile
            import os
            
            # Save CatBoost model temporarily
            with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as tmp_file:
                model.save_model(tmp_file.name)
                
                # Convert to ONNX
                onnx_model_path = tmp_file.name.replace('.cbm', '.onnx')
                model.save_model(
                    onnx_model_path,
                    format="onnx",
                    export_parameters={
                        'onnx_domain': 'ai.catboost',
                        'onnx_model_version': 1,
                        'onnx_doc_string': f'CatBoost model for liquidation trading',
                        'onnx_graph_name': 'CatBoostModel'
                    }
                )
                
                # Read ONNX model
                with open(onnx_model_path, 'rb') as f:
                    onnx_model_bytes = f.read()
                
                # Verify ONNX model
                onnx_model = onnx.load_from_string(onnx_model_bytes)
                onnx.checker.check_model(onnx_model)
                
                # Test inference speed
                ort_session = ort.InferenceSession(onnx_model_bytes)
                
                # Performance test
                import time
                test_input = X_sample.iloc[:1].values.astype(np.float32)
                input_name = ort_session.get_inputs()[0].name
                
                start_time = time.perf_counter()
                for _ in range(100):  # 100 inferences
                    _ = ort_session.run(None, {input_name: test_input})
                avg_inference_time = (time.perf_counter() - start_time) / 100 * 1000  # ms
                
                logger.info("ONNX conversion successful",
                           model_size_mb=len(onnx_model_bytes) / 1024 / 1024,
                           avg_inference_time_ms=avg_inference_time)
                
                # Cleanup
                os.unlink(tmp_file.name)
                os.unlink(onnx_model_path)
                
                return onnx_model_bytes
                
        except Exception as e:
            logger.error("ONNX conversion failed", exception=e)
            return None
    
    def _save_model(self, model: cb.CatBoostRegressor, onnx_model: Optional[bytes]) -> Dict[str, str]:
        """Save model in various formats."""
        model_paths = {}
        
        # Create versioned directory
        version_dir = self.model_dir / f"{self.config['model_version']}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CatBoost model
        if "cbm" in self.config["save_model_format"]:
            cbm_path = version_dir / "model.cbm"
            model.save_model(str(cbm_path))
            model_paths["catboost"] = str(cbm_path)
        
        # Save ONNX model
        if onnx_model and "onnx" in self.config["save_model_format"]:
            onnx_path = version_dir / "model.onnx"
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model)
            model_paths["onnx"] = str(onnx_path)
        
        # Save model metadata
        metadata = {
            "model_version": self.config["model_version"],
            "training_time": self.training_time,
            "config": self.config,
            "feature_importance": self.feature_importance,
            "optimization_results": self.optimization_results,
            "model_paths": model_paths
        }
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        model_paths["metadata"] = str(metadata_path)
        
        logger.info("Model saved successfully", model_paths=model_paths)
        
        return model_paths
    
    def _generate_training_report(
        self,
        model: cb.CatBoostRegressor,
        train_scores: Dict[str, float],
        val_scores: Dict[str, float],
        feature_importance: Dict[str, float],
        model_paths: Dict[str, str]
    ) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        
        return {
            "model_info": {
                "model_type": "CatBoostRegressor",
                "version": self.config["model_version"],
                "training_time": self.training_time,
                "feature_count": len(feature_importance),
                "model_size": model.calc_feature_statistics().get("model_size", 0)
            },
            "performance": {
                "train_scores": train_scores,
                "validation_scores": val_scores,
                "best_iteration": model.get_best_iteration() if hasattr(model, 'get_best_iteration') else None
            },
            "feature_analysis": {
                "feature_importance": feature_importance,
                "top_10_features": dict(list(feature_importance.items())[:10])
            },
            "optimization": self.optimization_results,
            "model_artifacts": model_paths,
            "config": self.config
        }
    
    def predict(self, X: pd.DataFrame, use_onnx: bool = False) -> np.ndarray:
        """Make predictions using trained model."""
        if self.best_model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        if use_onnx:
            # Use ONNX model for inference
            onnx_path = self.model_dir / f"{self.config['model_version']}" / "model.onnx"
            if onnx_path.exists():
                ort_session = ort.InferenceSession(str(onnx_path))
                input_name = ort_session.get_inputs()[0].name
                predictions = ort_session.run(None, {input_name: X.values.astype(np.float32)})[0]
                return predictions.flatten()
        
        # Use CatBoost model
        return self.best_model.predict(X)
    
    def load_model(self, model_version: str) -> cb.CatBoostRegressor:
        """Load saved model."""
        model_path = self.model_dir / model_version / "model.cbm"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = cb.CatBoostRegressor()
        model.load_model(str(model_path))
        
        # Load metadata
        metadata_path = self.model_dir / model_version / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.config = metadata.get("config", self.config)
                self.feature_importance = metadata.get("feature_importance", {})
        
        self.best_model = model
        logger.info("Model loaded successfully", version=model_version)
        
        return model