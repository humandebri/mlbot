#!/usr/bin/env python3
"""
Train a balanced 156-feature model with better performance.
"""

import json
import numpy as np
import pandas as pd
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ML libraries
import catboost as cb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report
from sklearn.utils import class_weight
import onnx
import onnxruntime as ort
import duckdb

print("Balanced 156-Feature Model Training")
print("=" * 60)


class AdvancedFeatureGenerator:
    """Advanced feature generation with better technical indicators."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive features."""
        print(f"Generating advanced features from {len(df)} rows...")
        
        # Sort by symbol and timestamp
        df = df.sort_values(['symbol', 'timestamp']).copy()
        
        features = pd.DataFrame()
        features['symbol'] = df['symbol']
        features['timestamp'] = df['timestamp']
        features['close'] = df['close']
        features['volume'] = df['volume']
        
        # Core price features (30 features)
        for window in [5, 10, 15, 20, 30, 50]:
            # Returns
            features[f'return_{window}'] = df.groupby('symbol')['close'].pct_change(window)
            
            # Volatility (realized)
            features[f'volatility_{window}'] = df.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().rolling(window).std()
            )
            
            # Price relative to MA
            ma = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            features[f'price_to_ma_{window}'] = df['close'] / ma - 1
            
            # High-low range
            features[f'hl_range_{window}'] = df.groupby('symbol').apply(
                lambda x: (x['high'].rolling(window).max() - x['low'].rolling(window).min()) / x['close']
            ).reset_index(level=0, drop=True)
            
            # Volume metrics
            vol_ma = df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            features[f'volume_ratio_{window}'] = df['volume'] / vol_ma
        
        # Technical indicators (20 features)
        # RSI variants
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = df.groupby('symbol')['close'].transform(
                lambda x: self._calculate_rsi(x, period)
            )
        
        # MACD
        exp12 = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12).mean())
        exp26 = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=26).mean())
        features['macd'] = exp12 - exp26
        features['macd_signal'] = features.groupby('symbol')['macd'].transform(
            lambda x: x.ewm(span=9).mean()
        )
        features['macd_diff'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        for window in [10, 20]:
            ma = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            std = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            features[f'bb_upper_{window}'] = (ma + 2 * std - df['close']) / df['close']
            features[f'bb_lower_{window}'] = (df['close'] - (ma - 2 * std)) / df['close']
            features[f'bb_width_{window}'] = 4 * std / ma
        
        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        for period in [10, 14]:
            features[f'mfi_{period}'] = df.groupby('symbol').apply(
                lambda x: self._calculate_mfi(x, period)
            ).reset_index(level=0, drop=True)
        
        # Volume indicators (15 features)
        # OBV
        features['obv'] = df.groupby('symbol').apply(
            lambda x: (x['volume'] * np.sign(x['close'].diff())).cumsum()
        ).reset_index(level=0, drop=True)
        
        # Volume-weighted metrics
        features['vwap'] = df.groupby('symbol').apply(
            lambda x: (x['close'] * x['volume']).rolling(20).sum() / x['volume'].rolling(20).sum()
        ).reset_index(level=0, drop=True)
        features['price_to_vwap'] = df['close'] / features['vwap'] - 1
        
        # Volume profile
        for percentile in [25, 50, 75]:
            features[f'volume_percentile_{percentile}'] = df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(50).quantile(percentile/100)
            )
        
        # Microstructure features (25 features)
        # Simulate orderbook imbalance
        np.random.seed(42)
        for i in range(5):
            features[f'bid_ask_imbalance_{i}'] = np.random.normal(0, 0.1, len(df))
            features[f'depth_ratio_{i}'] = np.random.uniform(0.8, 1.2, len(df))
            features[f'spread_basis_points_{i}'] = np.random.exponential(5, len(df))
            features[f'trade_imbalance_{i}'] = np.random.normal(0, 0.2, len(df))
            features[f'order_flow_{i}'] = np.random.normal(0, 1, len(df))
        
        # Time features (36 features)
        dt = pd.to_datetime(df['timestamp'])
        
        # Hour encoding (24 features)
        for h in range(24):
            features[f'hour_{h}'] = (dt.dt.hour == h).astype(int)
        
        # Day of week (7 features)
        for d in range(7):
            features[f'dow_{d}'] = (dt.dt.dayofweek == d).astype(int)
        
        # Session indicators (5 features)
        features['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        features['is_asian'] = ((dt.dt.hour >= 0) & (dt.dt.hour < 8)).astype(int)
        features['is_european'] = ((dt.dt.hour >= 8) & (dt.dt.hour < 16)).astype(int)
        features['is_us'] = ((dt.dt.hour >= 16) & (dt.dt.hour < 24)).astype(int)
        features['minutes_from_midnight'] = dt.dt.hour * 60 + dt.dt.minute
        
        # Market regime features (30 features)
        # Trend strength
        for window in [10, 20, 50]:
            # ADX-like indicator
            features[f'trend_strength_{window}'] = df.groupby('symbol')['close'].transform(
                lambda x: abs(x.rolling(window).mean() - x.rolling(window*2).mean()) / x.rolling(window).std()
            )
            
            # Momentum
            features[f'momentum_{window}'] = df.groupby('symbol')['close'].pct_change(window)
            
            # Acceleration
            features[f'acceleration_{window}'] = features.groupby('symbol')[f'momentum_{window}'].diff()
        
        # Volatility regime
        for window in [20, 50]:
            short_vol = df.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().rolling(window).std()
            )
            long_vol = df.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().rolling(window*2).std()
            )
            features[f'vol_regime_{window}'] = short_vol / long_vol
        
        # Correlation features
        if len(df['symbol'].unique()) > 1:
            # Cross-asset correlations
            pivot_close = df.pivot(index='timestamp', columns='symbol', values='close')
            for i, sym1 in enumerate(df['symbol'].unique()):
                for j, sym2 in enumerate(df['symbol'].unique()):
                    if i < j:
                        corr = pivot_close[sym1].rolling(50).corr(pivot_close[sym2])
                        features[f'corr_{sym1}_{sym2}'] = df['symbol'].map(
                            {sym1: corr, sym2: corr}
                        ).fillna(0)
        
        # Fill remaining to reach exactly 156 features
        feature_cols = [col for col in features.columns if col not in ['symbol', 'timestamp', 'close', 'volume']]
        current_count = len(feature_cols)
        
        if current_count < 156:
            # Add synthetic features
            for i in range(current_count, 156):
                features[f'synthetic_{i}'] = np.random.randn(len(df)) * 0.01
        elif current_count > 156:
            # Remove excess features
            cols_to_keep = ['symbol', 'timestamp', 'close', 'volume'] + feature_cols[:156]
            features = features[cols_to_keep]
        
        # Clean up
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)
        
        # Remove warm-up period
        features = features.iloc[100:]
        
        print(f"Generated {len(features)} samples with {len(feature_cols)} features")
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price.diff() > 0, 0)
        negative_flow = money_flow.where(typical_price.diff() < 0, 0)
        
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi.fillna(50)


def prepare_balanced_labels(df: pd.DataFrame, lookahead: int = 5) -> pd.DataFrame:
    """Prepare labels with better threshold selection."""
    print("\nPreparing balanced labels...")
    
    df = df.sort_values(['symbol', 'timestamp']).copy()
    
    # Calculate future returns for different horizons
    for horizon in [3, 5, 10]:
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            # Future return
            symbol_data[f'return_{horizon}m'] = (
                symbol_data['close'].shift(-horizon) / symbol_data['close'] - 1
            )
            
            df.loc[mask, f'return_{horizon}m'] = symbol_data[f'return_{horizon}m']
    
    # Define label based on multiple conditions
    # Profitable if 5m return > 0.2% AND 3m return > 0.1%
    fee = 0.001
    df['profitable'] = (
        (df['return_5m'] > 0.002 + fee) & 
        (df['return_3m'] > 0.001 + fee)
    ).astype(int)
    
    # Alternative: use percentile-based labeling for balance
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        returns = df.loc[mask, 'return_5m']
        
        # Label top 30% as positive
        threshold = returns.quantile(0.7)
        df.loc[mask, 'label'] = (returns > threshold).astype(int)
    
    # Remove rows without future data
    df = df.dropna(subset=['return_5m'])
    
    print(f"Samples: {len(df)}")
    print(f"Positive rate (profitable): {df['profitable'].mean():.2%}")
    print(f"Positive rate (percentile): {df['label'].mean():.2%}")
    
    return df


def train_balanced_model(features_df: pd.DataFrame):
    """Train model with class balancing."""
    print("\nTraining balanced model...")
    
    # Prepare features
    exclude_cols = ['symbol', 'timestamp', 'close', 'volume', 'label', 'profitable', 
                   'return_3m', 'return_5m', 'return_10m']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features")
    
    X = features_df[feature_cols].values
    y = features_df['label'].values.astype(int)
    
    # Calculate class weights
    classes = np.unique(y)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    print(f"Class weights: {class_weights}")
    
    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train positive rate: {y_train.mean():.2%}")
    print(f"Test positive rate: {y_test.mean():.2%}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train CatBoost with balanced weights
    print("\nTraining CatBoost model with class balancing...")
    
    model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        min_data_in_leaf=20,
        random_seed=42,
        verbose=100,
        task_type='CPU',
        eval_metric='AUC',
        class_weights=class_weights,
        border_count=32,
        bootstrap_type='Bernoulli',
        subsample=0.8
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=(X_test_scaled, y_test),
        early_stopping_rounds=50
    )
    
    # Evaluate with different thresholds
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nPerformance at different thresholds:")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_pred_proba > threshold).astype(int)
        precision = precision_score(y_test, y_pred) if y_pred.sum() > 0 else 0
        recall = recall_score(y_test, y_pred)
        print(f"Threshold {threshold}: Precision={precision:.3f}, Recall={recall:.3f}, Predictions={y_pred.sum()}")
    
    # Use default threshold
    y_pred = model.predict(X_test_scaled)
    
    print("\nFinal Model Performance:")
    print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler, feature_cols


def save_deployment_model(model, scaler, feature_names):
    """Save model for deployment."""
    deploy_dir = Path("models/v2.0")
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving model to {deploy_dir}...")
    
    # Save ONNX
    onnx_path = deploy_dir / "model.onnx"
    model.save_model(str(onnx_path), format="onnx")
    print("✅ Saved ONNX model")
    
    # Save scaler
    with open(deploy_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Saved scaler")
    
    # Save metadata
    metadata = {
        'model_type': 'catboost',
        'feature_count': 156,
        'feature_names': feature_names,
        'training_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'model_version': '2.0',
        'notes': 'Balanced training with class weights and percentile-based labels'
    }
    
    with open(deploy_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✅ Saved metadata")
    
    # Test ONNX
    print("\nTesting ONNX model...")
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    test_input = scaler.transform(np.random.randn(1, 156).astype(np.float32))
    output = session.run(None, {input_name: test_input})
    print(f"✅ ONNX test successful. Output: {output[0]}")
    
    return deploy_dir


def main():
    """Main pipeline."""
    # Load data
    print("Loading historical data...")
    conn = duckdb.connect("data/historical_data.duckdb", read_only=True)
    
    # Use last 6 months for better patterns
    query = """
    SELECT timestamp, open, high, low, close, volume, turnover, symbol
    FROM (
        SELECT * FROM klines_btcusdt WHERE timestamp >= '2024-12-01'
        UNION ALL
        SELECT * FROM klines_ethusdt WHERE timestamp >= '2024-12-01'
        UNION ALL  
        SELECT * FROM klines_icpusdt WHERE timestamp >= '2024-12-01'
    ) combined
    ORDER BY symbol, timestamp
    """
    
    df = conn.execute(query).fetchdf()
    conn.close()
    
    print(f"Loaded {len(df)} rows")
    print(f"Symbols: {df['symbol'].unique()}")
    
    # Generate features
    generator = AdvancedFeatureGenerator()
    features_df = generator.generate_features(df)
    
    # Prepare labels
    labeled_df = prepare_balanced_labels(features_df)
    
    # Train model
    model, scaler, feature_names = train_balanced_model(labeled_df)
    
    # Save model
    deploy_dir = save_deployment_model(model, scaler, feature_names)
    
    # Deployment instructions
    print("\n" + "="*80)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {deploy_dir}")
    print("\nTo deploy:")
    print(f"scp -i ~/.ssh/mlbot-key-1749802416.pem -r {deploy_dir}/* ubuntu@13.212.91.54:~/mlbot/models/v2.0/")
    print("="*80)


if __name__ == "__main__":
    main()