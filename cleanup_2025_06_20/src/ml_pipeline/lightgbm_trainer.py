"""
LightGBM model trainer for liquidation prediction.

LightGBM advantages:
- Much faster training (10-100x faster than CatBoost)
- Lower memory usage
- Native support for categorical features
- Excellent performance on large datasets
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import json
import joblib
from pathlib import Path
import optuna
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


class LightGBMTrainer:
    """
    LightGBM trainer optimized for high-frequency trading data.
    
    Features:
    - Fast training on large datasets
    - Hyperparameter optimization with Optuna
    - Time series cross-validation
    - Feature importance analysis
    - Model persistence and versioning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM trainer.
        
        Args:
            config: Custom configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.feature_importance = None
        self.training_history = []
        self.best_params = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default LightGBM configuration."""
        return {
            # LightGBM parameters
            "objective": "regression",
            "metric": ["rmse", "mae", "mape"],
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "num_threads": -1,  # Use all available cores
            
            # Training parameters
            "num_boost_round": 1000,
            "early_stopping_rounds": 50,
            "valid_split": 0.2,
            
            # Advanced parameters for speed
            "max_bin": 255,  # Default, can reduce for more speed
            "min_data_in_leaf": 20,
            "min_sum_hessian_in_leaf": 1e-3,
            "feature_pre_filter": False,
            "force_col_wise": True,  # Better for many features
            
            # GPU parameters (if available)
            "device": "cpu",  # Change to "gpu" if CUDA available
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            
            # Feature engineering
            "categorical_features": "auto",
            "zero_as_missing": False,
            
            # Optimization
            "enable_hyperopt": True,
            "hyperopt_trials": 100,
            "hyperopt_timeout": 3600,  # 1 hour
        }
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_valid: Validation features (optional)
            y_valid: Validation labels (optional)
            categorical_features: List of categorical feature names
            
        Returns:
            Training results dictionary
        """
        logger.info(
            "Starting LightGBM training",
            train_shape=X_train.shape,
            config=self.config
        )
        
        start_time = datetime.now()
        
        # Prepare datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_features or "auto",
            free_raw_data=False
        )
        
        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(
                X_valid,
                label=y_valid,
                categorical_feature=categorical_features or "auto",
                reference=train_data,
                free_raw_data=False
            )
            valid_sets = [train_data, valid_data]
            valid_names = ["train", "valid"]
        else:
            # Use part of training data for validation
            n_valid = int(len(X_train) * self.config["valid_split"])
            X_valid = X_train.iloc[-n_valid:]
            y_valid = y_train.iloc[-n_valid:]
            X_train = X_train.iloc[:-n_valid]
            y_train = y_train.iloc[:-n_valid]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
            valid_sets = [train_data, valid_data]
            valid_names = ["train", "valid"]
        
        # Set up parameters
        params = {
            k: v for k, v in self.config.items()
            if k not in ["num_boost_round", "early_stopping_rounds", "valid_split",
                        "enable_hyperopt", "hyperopt_trials", "hyperopt_timeout"]
        }
        
        # Hyperparameter optimization
        if self.config["enable_hyperopt"]:
            logger.info("Starting hyperparameter optimization...")
            self.best_params = self._optimize_hyperparameters(
                train_data, valid_data, params
            )
            params.update(self.best_params)
        
        # Train model
        logger.info("Training final model...")
        evals_result = {}
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config["num_boost_round"],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(self.config["early_stopping_rounds"]),
                lgb.log_evaluation(100)
            ],
            evals_result=evals_result
        )
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(importance_type='gain'),
            'split': self.model.feature_importance(importance_type='split')
        }).sort_values('importance', ascending=False)
        
        # Training results
        training_time = (datetime.now() - start_time).total_seconds()
        
        results = {
            'training_time': training_time,
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'feature_importance': self.feature_importance.to_dict(),
            'params': params,
            'evals_result': evals_result,
            'model_size_mb': self._get_model_size() / (1024 * 1024)
        }
        
        logger.info(
            "LightGBM training completed",
            time=training_time,
            best_iteration=self.model.best_iteration,
            best_score=self.model.best_score
        )
        
        return results
    
    def _optimize_hyperparameters(
        self,
        train_data: lgb.Dataset,
        valid_data: lgb.Dataset,
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            train_data: Training dataset
            valid_data: Validation dataset
            base_params: Base parameters
            
        Returns:
            Best parameters found
        """
        def objective(trial):
            params = base_params.copy()
            
            # Suggest hyperparameters
            params.update({
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            })
            
            # Train with suggested parameters
            model = lgb.train(
                params,
                train_data,
                num_boost_round=300,  # Fewer rounds for optimization
                valid_sets=[valid_data],
                callbacks=[
                    lgb.early_stopping(20),
                    lgb.log_evaluation(0)  # Suppress output
                ]
            )
            
            # Return validation score
            return model.best_score['valid_0'][params['metric'][0]]
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config['hyperopt_trials'],
            timeout=self.config['hyperopt_timeout'],
            n_jobs=1  # LightGBM already uses multiple threads
        )
        
        logger.info(
            "Hyperparameter optimization completed",
            best_params=study.best_params,
            best_value=study.best_value
        )
        
        return study.best_params
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with trained model.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def save_model(self, path: str) -> None:
        """
        Save trained model.
        
        Args:
            path: Save path
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'config': self.config,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'feature_names': list(self.model.feature_name()),
            'training_date': datetime.now().isoformat()
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, path: str) -> None:
        """
        Load trained model.
        
        Args:
            path: Model path
        """
        model_path = Path(path)
        
        # Load model
        self.model = lgb.Booster(model_file=str(model_path))
        
        # Load metadata
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            self.config = metadata.get('config', self.config)
            self.best_params = metadata.get('best_params')
            
            if metadata.get('feature_importance'):
                self.feature_importance = pd.DataFrame(metadata['feature_importance'])
        
        logger.info(f"Model loaded from {model_path}")
    
    def _get_model_size(self) -> float:
        """Get model size in bytes."""
        if self.model is None:
            return 0
        
        # Save to temporary file and check size
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            self.model.save_model(tmp.name)
            return Path(tmp.name).stat().st_size
    
    def convert_to_onnx(self, X_sample: pd.DataFrame, output_path: str) -> None:
        """
        Convert LightGBM model to ONNX format for fast inference.
        
        Args:
            X_sample: Sample input for shape inference
            output_path: ONNX model output path
        """
        try:
            import onnxmltools
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            if self.model is None:
                raise ValueError("No model to convert")
            
            # Define input types
            initial_types = [
                ('features', FloatTensorType([None, X_sample.shape[1]]))
            ]
            
            # Convert model
            onnx_model = onnxmltools.convert_lightgbm(
                self.model,
                initial_types=initial_types,
                target_opset=12
            )
            
            # Save ONNX model
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            logger.info(f"Model converted to ONNX and saved to {output_path}")
            
        except ImportError:
            logger.error("onnxmltools not installed. Run: pip install onnxmltools")
            raise


def compare_with_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Compare LightGBM with CatBoost performance.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Comparison results
    """
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import time
    
    results = {}
    
    # Train LightGBM
    logger.info("Training LightGBM...")
    lgb_start = time.time()
    
    lgb_trainer = LightGBMTrainer()
    lgb_trainer.train(X_train, y_train)
    lgb_pred = lgb_trainer.predict(X_test)
    
    lgb_time = time.time() - lgb_start
    
    results['lightgbm'] = {
        'training_time': lgb_time,
        'rmse': mean_squared_error(y_test, lgb_pred, squared=False),
        'mae': mean_absolute_error(y_test, lgb_pred),
        'model_size_mb': lgb_trainer._get_model_size() / (1024 * 1024)
    }
    
    # Train CatBoost
    logger.info("Training CatBoost...")
    cb_start = time.time()
    
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        verbose=False
    )
    cb_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    cb_pred = cb_model.predict(X_test)
    
    cb_time = time.time() - cb_start
    
    results['catboost'] = {
        'training_time': cb_time,
        'rmse': mean_squared_error(y_test, cb_pred, squared=False),
        'mae': mean_absolute_error(y_test, cb_pred)
    }
    
    # Calculate speedup
    results['speedup'] = cb_time / lgb_time
    
    logger.info(
        "Comparison completed",
        lightgbm_time=lgb_time,
        catboost_time=cb_time,
        speedup=results['speedup']
    )
    
    return results