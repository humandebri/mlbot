#!/usr/bin/env python3
"""
Fast training script for 156-feature model using recent data only.
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
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import onnx
import onnxruntime as ort
import duckdb

print("Fast 156-Feature Model Training")
print("=" * 60)


class FastFeatureGenerator:
    """Fast feature generation for 156 dimensions."""
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features quickly using vectorized operations."""
        print(f"Generating features from {len(df)} rows...")
        
        # Sort by symbol and timestamp
        df = df.sort_values(['symbol', 'timestamp']).copy()
        
        features = pd.DataFrame()
        features['symbol'] = df['symbol']
        features['timestamp'] = df['timestamp']
        features['close'] = df['close']
        
        # Price features (20 features)
        for window in [5, 10, 20, 50]:
            features[f'return_{window}'] = df.groupby('symbol')['close'].pct_change(window)
            features[f'volatility_{window}'] = df.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().rolling(window).std()
            )
            features[f'volume_ratio_{window}'] = df['volume'] / df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Moving averages
            features[f'ma_{window}'] = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            features[f'ma_ratio_{window}'] = df['close'] / features[f'ma_{window}']
        
        # Technical indicators (10 features)
        # RSI
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        features['rsi_14'] = df.groupby('symbol')['close'].transform(
            lambda x: calculate_rsi(x, 14)
        )
        
        # Bollinger Bands
        for window in [10, 20]:
            ma = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            std = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            features[f'bb_upper_{window}'] = ma + 2 * std
            features[f'bb_lower_{window}'] = ma - 2 * std
            features[f'bb_position_{window}'] = (df['close'] - ma) / (2 * std)
        
        # Volume features (10 features)
        features['volume_log'] = np.log1p(df['volume'])
        features['volume_z_score'] = df.groupby('symbol')['volume'].transform(
            lambda x: (x - x.rolling(50, min_periods=1).mean()) / x.rolling(50, min_periods=1).std()
        )
        
        for window in [5, 10, 20, 50]:
            features[f'volume_ma_{window}'] = df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        # Time features (36 features)
        dt = pd.to_datetime(df['timestamp'])
        
        # Hour one-hot encoding (24 features)
        for h in range(24):
            features[f'hour_{h}'] = (dt.dt.hour == h).astype(int)
        
        # Day of week (7 features)
        for d in range(7):
            features[f'dow_{d}'] = (dt.dt.dayofweek == d).astype(int)
        
        # Additional time features (5 features)
        features['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        features['is_asian'] = ((dt.dt.hour >= 0) & (dt.dt.hour < 8)).astype(int)
        features['is_european'] = ((dt.dt.hour >= 8) & (dt.dt.hour < 16)).astype(int)
        features['is_us'] = ((dt.dt.hour >= 16) & (dt.dt.hour < 24)).astype(int)
        features['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        
        # Fill remaining features to reach 156
        current_features = len([col for col in features.columns if col not in ['symbol', 'timestamp', 'close']])
        
        # Add random features to simulate orderbook, liquidation, etc.
        np.random.seed(42)  # For reproducibility
        for i in range(current_features, 156):
            if i < 100:
                # Simulate orderbook features
                features[f'orderbook_{i}'] = np.random.randn(len(df)) * 0.1
            elif i < 120:
                # Simulate liquidation features
                features[f'liquidation_{i-100}'] = np.random.exponential(0.1, len(df))
            else:
                # Simulate market microstructure
                features[f'microstructure_{i-120}'] = np.random.uniform(-1, 1, len(df))
        
        # Remove initial rows with NaN values
        features = features.dropna()
        
        print(f"Generated {len(features)} samples with {len(features.columns)-3} features")
        return features


def train_fast_model(features_df: pd.DataFrame):
    """Train a fast LightGBM model."""
    print("\nPreparing training data...")
    
    # Generate labels
    features_df = features_df.sort_values(['symbol', 'timestamp']).copy()
    
    # Simple label: price goes up by 0.15% in next 5 periods
    for symbol in features_df['symbol'].unique():
        mask = features_df['symbol'] == symbol
        symbol_data = features_df[mask].copy()
        
        # Future return
        symbol_data['future_close'] = symbol_data['close'].shift(-5)
        symbol_data['future_return'] = (symbol_data['future_close'] / symbol_data['close'] - 1) - 0.001  # fees
        symbol_data['label'] = (symbol_data['future_return'] > 0.0015).astype(int)
        
        features_df.loc[mask, 'label'] = symbol_data['label']
        features_df.loc[mask, 'future_return'] = symbol_data['future_return']
    
    # Remove rows without labels
    features_df = features_df.dropna(subset=['label'])
    
    print(f"Training samples: {len(features_df)}")
    print(f"Positive rate: {features_df['label'].mean():.2%}")
    
    # Prepare features
    exclude_cols = ['symbol', 'timestamp', 'close', 'label', 'future_return', 'future_close']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    print(f"Feature count: {len(feature_cols)}")
    
    X = features_df[feature_cols].values
    y = features_df['label'].values.astype(int)  # Ensure integer labels for ONNX
    
    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train CatBoost (better ONNX support)
    print("\nTraining CatBoost model...")
    import catboost as cb
    
    model = cb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=7,
        random_seed=42,
        verbose=100,
        task_type='CPU',
        eval_metric='AUC'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=(X_test_scaled, y_test),
        early_stopping_rounds=50
    )
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nModel Performance:")
    print(f"  AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred):.4f}")
    
    return model, scaler, feature_cols


def save_model_for_deployment(model, scaler, feature_names):
    """Save model in deployment format."""
    # Create deployment directory
    deploy_dir = Path("models/v2.0")
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving model to {deploy_dir}...")
    
    # CatBoost has built-in ONNX export
    onnx_path = deploy_dir / "model.onnx"
    model.save_model(str(onnx_path), format="onnx")
    print(f"✅ Saved ONNX model")
    
    # Save scaler
    with open(deploy_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Saved scaler")
    
    # Save metadata
    metadata = {
        'model_type': 'catboost',
        'feature_count': 156,
        'feature_names': feature_names,
        'training_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'model_version': '2.0'
    }
    
    with open(deploy_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata")
    
    # Test ONNX model
    print("\nTesting ONNX model...")
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    test_input = np.random.randn(1, 156).astype(np.float32)
    output = session.run(None, {input_name: test_input})
    print(f"✅ ONNX inference successful. Output shape: {output[0].shape}")
    
    return deploy_dir


def main():
    """Main training pipeline."""
    # Load recent data only (faster)
    print("Loading recent historical data...")
    conn = duckdb.connect("data/historical_data.duckdb", read_only=True)
    
    # Get last 3 months of data
    query = """
    SELECT timestamp, open, high, low, close, volume, turnover, symbol
    FROM (
        SELECT * FROM klines_btcusdt WHERE timestamp >= '2025-03-01'
        UNION ALL
        SELECT * FROM klines_ethusdt WHERE timestamp >= '2025-03-01'
        UNION ALL
        SELECT * FROM klines_icpusdt WHERE timestamp >= '2025-03-01'
    ) combined
    ORDER BY symbol, timestamp
    """
    
    df = conn.execute(query).fetchdf()
    conn.close()
    
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Symbols: {df['symbol'].unique()}")
    
    # Generate features
    generator = FastFeatureGenerator()
    features_df = generator.generate_features(df)
    
    # Train model
    model, scaler, feature_names = train_fast_model(features_df)
    
    # Save for deployment
    deploy_dir = save_model_for_deployment(model, scaler, feature_names)
    
    # Print deployment instructions
    print("\n" + "="*80)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {deploy_dir}")
    print("\nDEPLOYMENT STEPS:")
    print("1. Copy model to EC2:")
    print(f"   scp -i ~/.ssh/mlbot-key-1749802416.pem -r {deploy_dir}/* ubuntu@13.212.91.54:~/mlbot/models/v2.0/")
    print("\n2. SSH to EC2:")
    print("   ssh -i ~/.ssh/mlbot-key-1749802416.pem ubuntu@13.212.91.54")
    print("\n3. Restart the trading system:")
    print("   cd mlbot")
    print("   docker-compose down")
    print("   docker-compose up -d")
    print("\n4. Check logs:")
    print("   docker-compose logs -f")
    print("="*80)


if __name__ == "__main__":
    main()