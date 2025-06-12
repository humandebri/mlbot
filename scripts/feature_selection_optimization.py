#!/usr/bin/env python3
"""
Feature selection and optimization to improve model performance and reduce overfitting.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
import joblib
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class FeatureSelector:
    """Feature selection and optimization for profit prediction."""
    
    def __init__(self):
        self.feature_importance = {}
        self.selected_features = []
        self.correlation_matrix = None
        
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load training data with all features."""
        logger.info("Loading training data for feature selection")
        
        conn = duckdb.connect("data/historical_data.duckdb")
        query = """
        SELECT 
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM klines_btcusdt
        WHERE timestamp >= '2024-01-01'
          AND timestamp <= '2024-04-30'
        ORDER BY timestamp
        """
        
        data = conn.execute(query).df()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        conn.close()
        
        # Create comprehensive features
        data = self.create_comprehensive_features(data)
        data = self.create_profit_targets(data)
        
        # Prepare X, y
        feature_cols = [col for col in data.columns 
                       if not col.startswith('profitable_') 
                       and col not in ['open', 'high', 'low', 'close', 'volume']]
        
        X = data[feature_cols].copy()
        y = data['profitable_5m'].copy()
        
        # Remove NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")
        logger.info(f"Positive rate: {y.mean():.2%}")
        
        return X, y
    
    def create_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set."""
        
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
            data[f'vol_rank_{window}'] = data[f'vol_{window}'].rolling(window*2).rank(pct=True)
        
        # Moving averages
        for ma in [5, 10, 20, 50]:
            data[f'sma_{ma}'] = data['close'].rolling(ma).mean()
            data[f'ema_{ma}'] = data['close'].ewm(span=ma).mean()
            data[f'price_vs_sma_{ma}'] = (data['close'] - data[f'sma_{ma}']) / data[f'sma_{ma}']
            data[f'price_vs_ema_{ma}'] = (data['close'] - data[f'ema_{ma}']) / data[f'ema_{ma}']
        
        # Momentum features
        for period in [3, 5, 10, 15]:
            data[f'momentum_{period}'] = data['close'].pct_change(period)
            data[f'acceleration_{period}'] = data[f'momentum_{period}'] - data[f'momentum_{period}'].shift(period)
        
        # Trend indicators
        data['trend_strength_5_20'] = (data['sma_5'] - data['sma_20']) / data['sma_20']
        data['trend_strength_10_50'] = (data['sma_10'] - data['sma_50']) / data['sma_50']
        data['price_above_sma20'] = (data['close'] > data['sma_20']).astype(int)
        data['price_above_sma50'] = (data['close'] > data['sma_50']).astype(int)
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
        data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
        
        # Bollinger Bands
        for window in [10, 20]:
            rolling_mean = data['close'].rolling(window).mean()
            rolling_std = data['close'].rolling(window).std()
            data[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
            data[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
            data[f'bb_position_{window}'] = (data['close'] - data[f'bb_lower_{window}']) / (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}'] + 1e-8)
            data[f'bb_width_{window}'] = (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}']) / rolling_mean
        
        # Volume features
        data['volume_ma_10'] = data['volume'].rolling(10).mean()
        data['volume_ma_20'] = data['volume'].rolling(20).mean()
        data['volume_ratio_10'] = data['volume'] / data['volume_ma_10']
        data['volume_ratio_20'] = data['volume'] / data['volume_ma_20']
        data['volume_spike'] = (data['volume_ratio_20'] > 2).astype(int)
        data['log_volume'] = np.log(data['volume'] + 1)
        data['volume_trend'] = data['volume'].pct_change(5)
        
        # Volume-price relationships
        data['volume_price_change'] = data['volume_ratio_20'] * abs(data['returns'])
        data['volume_momentum'] = data['volume_ratio_10'] * data['momentum_3']
        
        # Market regime indicators
        data['high_vol_regime'] = (data['vol_20'] > data['vol_20'].rolling(50).quantile(0.8)).astype(int)
        data['low_vol_regime'] = (data['vol_20'] < data['vol_20'].rolling(50).quantile(0.2)).astype(int)
        data['trending_market'] = (abs(data['trend_strength_5_20']) > 0.01).astype(int)
        
        # Statistical features
        for window in [10, 20]:
            data[f'skewness_{window}'] = data['returns'].rolling(window).skew()
            data[f'kurtosis_{window}'] = data['returns'].rolling(window).kurt()
        
        # Time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        return data
    
    def create_profit_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create profit targets."""
        transaction_cost = 0.0008  # Lower fee assumption
        
        for horizon in [3, 5, 10]:
            future_return = data['close'].pct_change(horizon).shift(-horizon)
            long_profit = future_return - transaction_cost
            short_profit = -future_return - transaction_cost
            data[f'profitable_{horizon}m'] = ((long_profit > 0.005) | (short_profit > 0.005)).astype(int)
        
        return data
    
    def analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze feature importance using multiple methods."""
        logger.info("Analyzing feature importance")
        
        importance_results = {}
        
        # 1. Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        importance_results['random_forest'] = rf_importance
        
        # 2. Statistical significance (F-test)
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        f_scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
        importance_results['f_test'] = f_scores
        
        # 3. Correlation analysis
        self.correlation_matrix = X.corr()
        
        # 4. Combined score
        # Normalize scores to 0-1 range
        rf_norm = (rf_importance - rf_importance.min()) / (rf_importance.max() - rf_importance.min())
        f_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min())
        
        combined_score = (rf_norm + f_norm) / 2
        combined_score = combined_score.sort_values(ascending=False)
        importance_results['combined'] = combined_score
        
        self.feature_importance = importance_results
        
        logger.info(f"Feature importance analysis completed")
        return importance_results
    
    def remove_redundant_features(self, X: pd.DataFrame, correlation_threshold: float = 0.8) -> List[str]:
        """Remove highly correlated features."""
        logger.info(f"Removing features with correlation > {correlation_threshold}")
        
        # Find highly correlated features
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop
        to_drop = []
        for column in upper_triangle.columns:
            if column in to_drop:
                continue
                
            correlated_features = upper_triangle.index[upper_triangle[column] > correlation_threshold].tolist()
            
            if correlated_features:
                # Keep the feature with highest importance, drop others
                feature_group = [column] + correlated_features
                
                if 'combined' in self.feature_importance:
                    importance_scores = self.feature_importance['combined'][feature_group]
                    features_to_drop = importance_scores.iloc[1:].index.tolist()  # Keep the most important one
                    to_drop.extend(features_to_drop)
        
        remaining_features = [col for col in X.columns if col not in to_drop]
        
        logger.info(f"Removed {len(to_drop)} redundant features, kept {len(remaining_features)}")
        return remaining_features
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 20) -> List[str]:
        """Select best features using RFE and importance scores."""
        logger.info(f"Selecting top {n_features} features")
        
        # Use combined importance score
        if 'combined' in self.feature_importance:
            top_features = self.feature_importance['combined'].head(n_features).index.tolist()
        else:
            # Fallback to RF importance
            rf_importance = self.feature_importance['random_forest']
            top_features = rf_importance.head(n_features).index.tolist()
        
        # Filter top_features to only include those available in X
        available_top_features = [f for f in top_features if f in X.columns]
        
        # Validate with RFE
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rfe = RFE(estimator=rf, n_features_to_select=min(n_features, len(available_top_features)))
        rfe.fit(X[available_top_features], y)
        
        rfe_selected = [feature for feature, selected in zip(available_top_features, rfe.support_) if selected]
        
        logger.info(f"Selected {len(rfe_selected)} features after RFE validation")
        self.selected_features = rfe_selected
        
        return rfe_selected
    
    def validate_feature_selection(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str]) -> Dict:
        """Validate feature selection with cross-validation."""
        logger.info("Validating feature selection")
        
        # Compare full model vs selected features
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Full model
        rf_full = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        scores_full = cross_val_score(rf_full, X, y, cv=tscv, scoring='roc_auc')
        
        # Selected features model
        rf_selected = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        scores_selected = cross_val_score(rf_selected, X[selected_features], y, cv=tscv, scoring='roc_auc')
        
        results = {
            'full_model_auc': scores_full.mean(),
            'full_model_std': scores_full.std(),
            'selected_model_auc': scores_selected.mean(),
            'selected_model_std': scores_selected.std(),
            'n_features_full': len(X.columns),
            'n_features_selected': len(selected_features),
            'performance_ratio': scores_selected.mean() / scores_full.mean()
        }
        
        logger.info(f"Full model AUC: {results['full_model_auc']:.3f}±{results['full_model_std']:.3f}")
        logger.info(f"Selected model AUC: {results['selected_model_auc']:.3f}±{results['selected_model_std']:.3f}")
        
        return results
    
    def plot_feature_analysis(self):
        """Plot feature importance and correlation analysis."""
        
        if not self.feature_importance:
            logger.warning("No feature importance data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Top 20 features by combined score
        if 'combined' in self.feature_importance:
            top_features = self.feature_importance['combined'].head(20)
            axes[0, 0].barh(range(len(top_features)), top_features.values)
            axes[0, 0].set_yticks(range(len(top_features)))
            axes[0, 0].set_yticklabels(top_features.index)
            axes[0, 0].set_title('Top 20 Features (Combined Score)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Random Forest importance
        if 'random_forest' in self.feature_importance:
            rf_top = self.feature_importance['random_forest'].head(15)
            axes[0, 1].barh(range(len(rf_top)), rf_top.values)
            axes[0, 1].set_yticks(range(len(rf_top)))
            axes[0, 1].set_yticklabels(rf_top.index)
            axes[0, 1].set_title('Random Forest Feature Importance')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Correlation heatmap (top features)
        if self.correlation_matrix is not None and 'combined' in self.feature_importance:
            top_10_features = self.feature_importance['combined'].head(10).index
            corr_subset = self.correlation_matrix.loc[top_10_features, top_10_features]
            
            sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=axes[1, 0])
            axes[1, 0].set_title('Correlation Matrix (Top 10 Features)')
        
        # 4. Feature importance comparison
        if len(self.feature_importance) >= 2:
            comparison_features = self.feature_importance['combined'].head(10).index
            
            rf_values = self.feature_importance['random_forest'][comparison_features]
            combined_values = self.feature_importance['combined'][comparison_features]
            
            x = np.arange(len(comparison_features))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, rf_values, width, label='Random Forest', alpha=0.7)
            axes[1, 1].bar(x + width/2, combined_values, width, label='Combined Score', alpha=0.7)
            
            axes[1, 1].set_xlabel('Features')
            axes[1, 1].set_ylabel('Importance Score')
            axes[1, 1].set_title('Feature Importance Comparison')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(comparison_features, rotation=45, ha='right')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_optimized_model(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str]) -> RandomForestClassifier:
        """Train optimized model with selected features."""
        logger.info("Training optimized model with selected features")
        
        # Train on selected features
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X[selected_features], y)
        
        # Save optimized model
        model_dir = Path("models/optimized_ensemble")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, model_dir / "optimized_random_forest.pkl")
        
        # Save selected features
        pd.Series(selected_features).to_csv(model_dir / "selected_features.csv", index=False)
        
        logger.info(f"Saved optimized model and selected features to {model_dir}")
        
        return model


def main():
    """Main feature selection optimization process."""
    
    logger.info("Starting feature selection optimization")
    
    # Initialize feature selector
    selector = FeatureSelector()
    
    # Load training data
    X, y = selector.load_training_data()
    
    # Analyze feature importance
    importance_results = selector.analyze_feature_importance(X, y)
    
    # Remove redundant features
    non_redundant_features = selector.remove_redundant_features(X, correlation_threshold=0.85)
    X_filtered = X[non_redundant_features]
    
    # Select best features
    selected_features = selector.select_best_features(X_filtered, y, n_features=25)
    
    # Validate feature selection
    validation_results = selector.validate_feature_selection(X_filtered, y, selected_features)
    
    # Train optimized model
    optimized_model = selector.train_optimized_model(X_filtered, y, selected_features)
    
    # Plot analysis
    selector.plot_feature_analysis()
    
    # Print results
    print("\n" + "="*80)
    print("🔍 特徴量選択最適化結果")
    print("="*80)
    
    print(f"\n📊 特徴量削減:")
    print(f"  元の特徴量数: {len(X.columns)}")
    print(f"  冗長除去後: {len(non_redundant_features)}")
    print(f"  最終選択: {len(selected_features)}")
    print(f"  削減率: {(1 - len(selected_features)/len(X.columns))*100:.1f}%")
    
    print(f"\n🎯 性能比較:")
    print(f"  全特徴量モデル: AUC={validation_results['full_model_auc']:.3f}±{validation_results['full_model_std']:.3f}")
    print(f"  選択特徴量モデル: AUC={validation_results['selected_model_auc']:.3f}±{validation_results['selected_model_std']:.3f}")
    print(f"  性能比率: {validation_results['performance_ratio']:.3f}")
    
    if validation_results['performance_ratio'] >= 0.99:
        print(f"  ✅ 性能維持で特徴量大幅削減成功")
    elif validation_results['performance_ratio'] >= 0.95:
        print(f"  ⚠️ 軽微な性能低下、過学習リスク軽減")
    else:
        print(f"  ❌ 性能低下大、特徴量選択見直し必要")
    
    print(f"\n🏆 選択された特徴量 (Top 10):")
    for i, feature in enumerate(selected_features[:10]):
        importance_score = importance_results['combined'][feature]
        print(f"  {i+1:2d}. {feature:<25} (重要度: {importance_score:.3f})")
    
    print(f"\n💾 保存先:")
    print(f"  最適化モデル: models/optimized_ensemble/optimized_random_forest.pkl")
    print(f"  選択特徴量: models/optimized_ensemble/selected_features.csv")
    print(f"  分析結果: backtest_results/feature_analysis.png")
    
    return selector, validation_results


if __name__ == "__main__":
    main()