#!/usr/bin/env python3
"""
Ensemble model for profitable cryptocurrency trading prediction.

Uses multiple models and advanced feature engineering to improve profitability.
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
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


class EnsembleProfitPredictor:
    """Ensemble model for profit prediction with advanced features."""
    
    def __init__(self, profit_threshold: float = 0.005, lookback_window: int = 30):
        self.profit_threshold = profit_threshold
        self.lookback_window = lookback_window
        self.scalers = {}
        self.models = {}
        self.feature_importance = {}
        
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
    
    def engineer_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features for profit prediction."""
        
        logger.info("Engineering comprehensive features")
        
        # Basic price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['oc_ratio'] = (data['close'] - data['open']) / data['open']
        
        # Multi-timeframe returns and volatilities
        timeframes = [1, 3, 5, 10, 15, 30, 60]
        for tf in timeframes:
            data[f'return_{tf}'] = data['close'].pct_change(tf)
            data[f'vol_{tf}'] = data['returns'].rolling(tf).std()
            data[f'vol_rank_{tf}'] = data[f'vol_{tf}'].rolling(tf*3).rank(pct=True)
        
        # Price momentum and acceleration
        for tf in [3, 5, 10, 15]:
            data[f'momentum_{tf}'] = data['close'].pct_change(tf)
            data[f'acceleration_{tf}'] = data[f'momentum_{tf}'] - data[f'momentum_{tf}'].shift(tf)
        
        # Moving averages and distances
        ma_periods = [5, 10, 20, 50]
        for ma in ma_periods:
            data[f'sma_{ma}'] = data['close'].rolling(ma).mean()
            data[f'ema_{ma}'] = data['close'].ewm(span=ma).mean()
            data[f'distance_sma_{ma}'] = (data['close'] - data[f'sma_{ma}']) / data[f'sma_{ma}']
            data[f'distance_ema_{ma}'] = (data['close'] - data[f'ema_{ma}']) / data[f'ema_{ma}']
        
        # Trend indicators
        data['sma_trend_5_20'] = (data['sma_5'] - data['sma_20']) / data['sma_20']
        data['sma_trend_10_50'] = (data['sma_10'] - data['sma_50']) / data['sma_50']
        data['price_above_sma20'] = (data['close'] > data['sma_20']).astype(int)
        data['price_above_sma50'] = (data['close'] > data['sma_50']).astype(int)
        
        # RSI-like indicator
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
        data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
        data['rsi_divergence'] = data['rsi'].diff(5)
        
        # Bollinger Bands-like indicators
        for window in [10, 20]:
            rolling_mean = data['close'].rolling(window).mean()
            rolling_std = data['close'].rolling(window).std()
            data[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
            data[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
            data[f'bb_position_{window}'] = (data['close'] - data[f'bb_lower_{window}']) / (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}'] + 1e-8)
            data[f'bb_squeeze_{window}'] = rolling_std / rolling_mean
        
        # Volume features
        data['volume_ma_10'] = data['volume'].rolling(10).mean()
        data['volume_ma_20'] = data['volume'].rolling(20).mean()
        data['volume_ratio_10'] = data['volume'] / data['volume_ma_10']
        data['volume_ratio_20'] = data['volume'] / data['volume_ma_20']
        data['volume_spike'] = (data['volume_ratio_20'] > 2).astype(int)
        data['log_volume'] = np.log(data['volume'] + 1)
        data['volume_trend'] = data['volume'].pct_change(5)
        
        # Volume-price relationship
        data['vp_correlation'] = data['returns'].rolling(20).corr(data['volume'].pct_change())
        data['volume_weighted_price'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
        data['vwp_distance'] = (data['close'] - data['volume_weighted_price']) / data['volume_weighted_price']
        
        # Price action patterns
        data['doji'] = (abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) < 0.1).astype(int)
        data['hammer'] = ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8) > 0.6).astype(int)
        data['shooting_star'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8) > 0.6).astype(int)
        
        # Gap analysis
        data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        data['gap_up'] = (data['gap'] > 0.003).astype(int)
        data['gap_down'] = (data['gap'] < -0.003).astype(int)
        data['gap_magnitude'] = abs(data['gap'])
        
        # Statistical features
        for window in [10, 20]:
            data[f'skewness_{window}'] = data['returns'].rolling(window).skew()
            data[f'kurtosis_{window}'] = data['returns'].rolling(window).kurt()
            data[f'range_pct_{window}'] = (data['high'].rolling(window).max() - data['low'].rolling(window).min()) / data['close']
        
        # Regime indicators (use existing volatility features)
        vol_20_window = data['returns'].rolling(20).std()
        data['high_vol_regime'] = (vol_20_window > vol_20_window.rolling(100).quantile(0.8)).astype(int)
        data['low_vol_regime'] = (vol_20_window < vol_20_window.rolling(100).quantile(0.2)).astype(int)
        data['trending_market'] = (abs(data['sma_trend_5_20']) > 0.01).astype(int)
        
        # Time-based features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['market_session'] = 0
        data.loc[(data['hour'] >= 1) & (data['hour'] <= 8), 'market_session'] = 1  # Asia
        data.loc[(data['hour'] >= 8) & (data['hour'] <= 16), 'market_session'] = 2  # Europe/US
        data.loc[(data['hour'] >= 16) & (data['hour'] <= 24), 'market_session'] = 3  # US/After
        
        # Interaction features
        data['vol_momentum_interaction'] = data['vol_10'] * abs(data['momentum_5'])
        data['rsi_volume_interaction'] = data['rsi'] * data['volume_ratio_10']
        data['trend_vol_interaction'] = data['sma_trend_5_20'] * data['vol_10']
        
        logger.info(f"Engineered {len(data.columns)} total features")
        return data
    
    def create_profit_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated profit targets."""
        
        logger.info("Creating profit targets")
        
        transaction_cost = 0.0012  # 0.12% round-trip cost
        
        # Multiple horizons for ensemble learning
        horizons = [3, 5, 10, 15]
        
        for horizon in horizons:
            future_return = data['close'].pct_change(horizon).shift(-horizon)
            
            # Profitable trades (either direction)
            long_profit = future_return - transaction_cost
            short_profit = -future_return - transaction_cost
            
            # Different profit thresholds
            data[f'profitable_{horizon}m'] = ((long_profit > self.profit_threshold) | (short_profit > self.profit_threshold)).astype(int)
            data[f'strong_profit_{horizon}m'] = ((long_profit > self.profit_threshold*2) | (short_profit > self.profit_threshold*2)).astype(int)
            
            # Direction indicators
            data[f'long_favorable_{horizon}m'] = (long_profit > short_profit).astype(int)
            
            # Profit magnitude (for regression)
            data[f'max_profit_{horizon}m'] = np.maximum(long_profit, short_profit)
        
        # Meta-target: consistent profitability across timeframes
        data['multi_horizon_profit'] = (
            data['profitable_3m'] & data['profitable_5m'] & data['profitable_10m']
        ).astype(int)
        
        return data
    
    def prepare_training_data(self, data: pd.DataFrame, target_col: str = 'profitable_5m') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        
        # Get feature columns (exclude targets and metadata)
        feature_cols = [col for col in data.columns if not col.startswith(('profitable_', 'strong_profit_', 'max_profit_', 'long_favorable_'))]
        feature_cols = [col for col in feature_cols if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        # Remove rows with NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")
        logger.info(f"Target: {target_col}, Positive rate: {y.mean():.2%}")
        
        return X, y
    
    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train ensemble of models."""
        
        logger.info("Training ensemble models")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Scale features for some models
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        self.scalers['main'] = scaler
        
        # Model 1: LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Model 2: CatBoost
        cb_model = cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            border_count=254,
            random_seed=42,
            class_weights=[1, 10],  # Higher weight for positive class
            verbose=False
        )
        
        # Model 3: Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Model 4: Logistic Regression
        lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # Train individual models
        models = {
            'lightgbm': lgb_model,
            'catboost': cb_model,
            'random_forest': rf_model,
            'logistic': lr_model
        }
        
        cv_scores = {}
        feature_importance = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}")
            
            cv_aucs = []
            cv_precisions = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Use scaled data for logistic regression
                if name == 'logistic':
                    X_train_use = X_scaled.iloc[train_idx]
                    X_val_use = X_scaled.iloc[val_idx]
                else:
                    X_train_use = X_train
                    X_val_use = X_val
                
                model.fit(X_train_use, y_train)
                
                # Predictions
                y_pred_proba = model.predict_proba(X_val_use)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Metrics
                auc = roc_auc_score(y_val, y_pred_proba)
                precision = (y_pred & y_val).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
                
                cv_aucs.append(auc)
                cv_precisions.append(precision)
            
            cv_scores[name] = {
                'auc_mean': np.mean(cv_aucs),
                'auc_std': np.std(cv_aucs),
                'precision_mean': np.mean(cv_precisions),
                'precision_std': np.std(cv_precisions)
            }
            
            # Final training on full dataset
            if name == 'logistic':
                model.fit(X_scaled, y)
            else:
                model.fit(X, y)
            
            self.models[name] = model
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance[name] = pd.Series(model.feature_importances_, index=X.columns)
            elif hasattr(model, 'coef_'):
                feature_importance[name] = pd.Series(abs(model.coef_[0]), index=X.columns)
            
            logger.info(f"{name} - AUC: {cv_scores[name]['auc_mean']:.3f}±{cv_scores[name]['auc_std']:.3f}, "
                       f"Precision: {cv_scores[name]['precision_mean']:.3f}±{cv_scores[name]['precision_std']:.3f}")
        
        # Create ensemble model (Voting Classifier)
        ensemble_models = [
            ('lightgbm', lgb_model),
            ('catboost', cb_model),
            ('random_forest', rf_model)
        ]
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        ensemble.fit(X, y)
        self.models['ensemble'] = ensemble
        
        # Ensemble cross-validation
        cv_aucs = []
        cv_precisions = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            ensemble_cv = VotingClassifier(
                estimators=ensemble_models,
                voting='soft'
            )
            ensemble_cv.fit(X_train, y_train)
            
            y_pred_proba = ensemble_cv.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            auc = roc_auc_score(y_val, y_pred_proba)
            precision = (y_pred & y_val).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
            
            cv_aucs.append(auc)
            cv_precisions.append(precision)
        
        cv_scores['ensemble'] = {
            'auc_mean': np.mean(cv_aucs),
            'auc_std': np.std(cv_aucs),
            'precision_mean': np.mean(cv_precisions),
            'precision_std': np.std(cv_precisions)
        }
        
        logger.info(f"Ensemble - AUC: {cv_scores['ensemble']['auc_mean']:.3f}±{cv_scores['ensemble']['auc_std']:.3f}, "
                   f"Precision: {cv_scores['ensemble']['precision_mean']:.3f}±{cv_scores['ensemble']['precision_std']:.3f}")
        
        self.feature_importance = feature_importance
        
        return cv_scores
    
    def save_models(self, model_dir: str = "models/ensemble"):
        """Save all trained models."""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_file = model_path / f"{name}_model.pkl"
            joblib.dump(model, model_file)
            logger.info(f"Saved {name} model to {model_file}")
        
        # Save scalers
        scaler_file = model_path / "scalers.pkl"
        joblib.dump(self.scalers, scaler_file)
        logger.info(f"Saved scalers to {scaler_file}")
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance for each model."""
        if not self.feature_importance:
            logger.warning("No feature importance data available")
            return
        
        n_models = len(self.feature_importance)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, importance) in enumerate(self.feature_importance.items()):
            top_features = importance.nlargest(top_n)
            
            axes[i].barh(range(len(top_features)), top_features.values, alpha=0.7)
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features.index)
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(f'{model_name.title()} Feature Importance')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results/ensemble_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Train ensemble profit prediction model."""
    
    logger.info("Starting ensemble profit prediction model training")
    
    # Initialize predictor
    predictor = EnsembleProfitPredictor(profit_threshold=0.005)
    
    # Load and prepare data
    data = predictor.load_data()
    data = predictor.engineer_comprehensive_features(data)
    data = predictor.create_profit_targets(data)
    
    # Prepare training data
    X, y = predictor.prepare_training_data(data, target_col='profitable_5m')
    
    # Train ensemble models
    cv_scores = predictor.train_ensemble_models(X, y)
    
    # Save models
    predictor.save_models()
    
    # Plot feature importance
    predictor.plot_feature_importance()
    
    # Print results
    print("\n" + "="*80)
    print("アンサンブル収益予測モデル訓練結果")
    print("="*80)
    
    print(f"\n📊 データセット:")
    print(f"  総サンプル数: {len(X):,}")
    print(f"  特徴量数: {len(X.columns)}")
    print(f"  陽性率: {y.mean():.2%}")
    
    print(f"\n🎯 モデル性能 (Cross-Validation):")
    for model_name, scores in cv_scores.items():
        print(f"  {model_name.upper():12}: AUC={scores['auc_mean']:.3f}±{scores['auc_std']:.3f}, "
              f"Precision={scores['precision_mean']:.3f}±{scores['precision_std']:.3f}")
    
    # Find best model
    best_model = max(cv_scores.items(), key=lambda x: x[1]['auc_mean'])
    print(f"\n🏆 最高性能モデル: {best_model[0].upper()}")
    print(f"   AUC: {best_model[1]['auc_mean']:.3f}")
    print(f"   Precision: {best_model[1]['precision_mean']:.3f}")
    
    print(f"\n💾 保存済み:")
    print(f"   モデル: models/ensemble/")
    print(f"   特徴量重要度: backtest_results/ensemble_feature_importance.png")
    
    print(f"\n💡 次のステップ:")
    print(f"   1. 最高性能モデルでバックテストを実行")
    print(f"   2. 動的ポジションサイジングの実装")
    print(f"   3. リスク管理ルールの最適化")


if __name__ == "__main__":
    main()