#!/usr/bin/env python3
"""
Simple but effective ensemble model for profit prediction.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Tuple
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class SimpleProfitPredictor:
    """Simple but effective profit prediction model."""
    
    def __init__(self, profit_threshold: float = 0.005):
        self.profit_threshold = profit_threshold
        self.scaler = StandardScaler()
        self.models = {}
        
    def load_data(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Load cryptocurrency data."""
        logger.info(f"Loading data for {symbol}")
        
        conn = duckdb.connect("data/historical_data.duckdb")
        query = f"""
        SELECT 
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM klines_{symbol.lower()}
        WHERE timestamp >= '2024-01-01'
          AND timestamp <= '2024-04-30'
        ORDER BY timestamp
        """
        
        data = conn.execute(query).df()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        conn.close()
        
        logger.info(f"Loaded {len(data)} records")
        return data
    
    def create_essential_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create essential features that are proven to work."""
        
        # Basic price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['oc_ratio'] = (data['close'] - data['open']) / data['open']
        
        # Multi-timeframe returns
        for period in [1, 3, 5, 10, 15, 30]:
            data[f'return_{period}'] = data['close'].pct_change(period)
        
        # Volatility features
        for window in [5, 10, 20, 30]:
            data[f'vol_{window}'] = data['returns'].rolling(window).std()
        
        # Simple moving averages
        for ma in [5, 10, 20]:
            data[f'sma_{ma}'] = data['close'].rolling(ma).mean()
            data[f'price_vs_sma_{ma}'] = (data['close'] - data[f'sma_{ma}']) / data[f'sma_{ma}']
        
        # Momentum indicators
        data['momentum_3'] = data['close'].pct_change(3)
        data['momentum_5'] = data['close'].pct_change(5)
        data['momentum_10'] = data['close'].pct_change(10)
        
        # Volume features
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['log_volume'] = np.log(data['volume'] + 1)
        
        # RSI approximation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Band position
        bb_middle = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # Trend indicators
        data['trend_strength'] = (data['sma_5'] - data['sma_20']) / data['sma_20']
        data['price_above_ma'] = (data['close'] > data['sma_20']).astype(int)
        
        # Volume-price interaction
        data['volume_price_change'] = data['volume_ratio'] * abs(data['returns'])
        
        # Market regime
        data['high_vol'] = (data['vol_20'] > data['vol_20'].rolling(50).quantile(0.8)).astype(int)
        data['low_vol'] = (data['vol_20'] < data['vol_20'].rolling(50).quantile(0.2)).astype(int)
        
        # Time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        
        return data
    
    def create_profit_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create profit prediction targets."""
        
        transaction_cost = 0.0012
        
        # Future returns for different horizons
        for horizon in [3, 5, 10]:
            future_return = data['close'].pct_change(horizon).shift(-horizon)
            
            # Long and short profitability
            long_profit = future_return - transaction_cost
            short_profit = -future_return - transaction_cost
            
            # Profitable in either direction
            data[f'profitable_{horizon}m'] = ((long_profit > self.profit_threshold) | 
                                            (short_profit > self.profit_threshold)).astype(int)
        
        return data
    
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'profitable_5m') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        
        # Get feature columns
        feature_cols = [col for col in data.columns 
                       if not col.startswith('profitable_') 
                       and col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Start with a conservative approach - only use complete cases
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        # Remove rows with any NaN in target
        valid_target = ~y.isnull()
        X = X[valid_target]
        y = y[valid_target]
        
        # For features, fill NaN with median or forward fill
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(method='ffill').fillna(0)
        
        # Remove any remaining NaN rows
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")
        logger.info(f"Target: {target_col}, Positive rate: {y.mean():.2%}")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train multiple models and return performance."""
        
        logger.info("Training ensemble models")
        
        # Models to train
        models = {
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'catboost': cb.CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                class_weights=[1, 5],
                random_seed=42,
                verbose=False
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Cross-validation results
        cv_results = {}
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for name, model in models.items():
            logger.info(f"Training {name}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
            
            # Manual CV for more detailed metrics
            precisions = []
            recalls = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                
                precisions.append(precision)
                recalls.append(recall)
            
            cv_results[name] = {
                'auc_mean': cv_scores.mean(),
                'auc_std': cv_scores.std(),
                'precision_mean': np.mean(precisions),
                'precision_std': np.std(precisions),
                'recall_mean': np.mean(recalls),
                'recall_std': np.std(recalls)
            }
            
            # Train final model on all data
            model.fit(X, y)
            self.models[name] = model
            
            logger.info(f"{name} - AUC: {cv_results[name]['auc_mean']:.3f}±{cv_results[name]['auc_std']:.3f}")
        
        # Create ensemble
        ensemble_models = [(name, model) for name, model in self.models.items()]
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble.fit(X, y)
        self.models['ensemble'] = ensemble
        
        # Ensemble CV
        ensemble_cv_scores = cross_val_score(ensemble, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
        cv_results['ensemble'] = {
            'auc_mean': ensemble_cv_scores.mean(),
            'auc_std': ensemble_cv_scores.std(),
            'precision_mean': 0,  # Would need manual calculation
            'precision_std': 0,
            'recall_mean': 0,
            'recall_std': 0
        }
        
        logger.info(f"Ensemble - AUC: {cv_results['ensemble']['auc_mean']:.3f}±{cv_results['ensemble']['auc_std']:.3f}")
        
        return cv_results
    
    def save_models(self, model_dir: str = "models/simple_ensemble"):
        """Save models."""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_file = model_path / f"{name}_model.pkl"
            joblib.dump(model, model_file)
            logger.info(f"Saved {name} model")


def main():
    """Train simple ensemble model."""
    
    logger.info("Starting simple ensemble profit prediction")
    
    # Initialize predictor
    predictor = SimpleProfitPredictor(profit_threshold=0.005)
    
    # Load and prepare data
    data = predictor.load_data()
    data = predictor.create_essential_features(data)
    data = predictor.create_profit_targets(data)
    
    # Prepare training data
    X, y = predictor.prepare_data(data, target_col='profitable_5m')
    
    if len(X) == 0:
        logger.error("No valid training data available")
        return
    
    # Train models
    cv_results = predictor.train_models(X, y)
    
    # Save models
    predictor.save_models()
    
    # Print results
    print("\n" + "="*70)
    print("シンプルアンサンブル収益予測モデル結果")
    print("="*70)
    
    print(f"\n📊 データ:")
    print(f"  サンプル数: {len(X):,}")
    print(f"  特徴量数: {len(X.columns)}")
    print(f"  陽性率: {y.mean():.2%}")
    
    print(f"\n🎯 モデル性能:")
    for model_name, scores in cv_results.items():
        print(f"  {model_name.upper():15}: AUC={scores['auc_mean']:.3f}±{scores['auc_std']:.3f}")
    
    # Find best model
    best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]['auc_mean'])
    best_score = cv_results[best_model_name]['auc_mean']
    
    print(f"\n🏆 最高性能: {best_model_name.upper()} (AUC: {best_score:.3f})")
    
    if best_score > 0.6:
        print(f"  ✅ AUC {best_score:.3f}は良好")
    elif best_score > 0.55:
        print(f"  ⚠️  AUC {best_score:.3f}は普通")
    else:
        print(f"  ❌ AUC {best_score:.3f}は要改善")
    
    print(f"\n💾 保存先: models/simple_ensemble/")
    
    return predictor, cv_results


if __name__ == "__main__":
    main()