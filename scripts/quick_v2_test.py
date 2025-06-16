#!/usr/bin/env python3
"""
Quick test of v2.0 model performance vs previous models.
"""

import sys
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import duckdb
import onnxruntime as ort
from sklearn.metrics import roc_auc_score
import joblib

print("Quick V2.0 Model Performance Analysis")
print("=" * 60)


def load_v2_model():
    """Load v2.0 ONNX model."""
    model_dir = Path("models/v2.0")
    
    # Load metadata
    with open(model_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model = ort.InferenceSession(str(model_dir / "model.onnx"))
    
    # Load scaler
    with open(model_dir / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler, metadata


def load_old_ensemble_model():
    """Load the high-performing ensemble model."""
    model_dir = Path("models/simple_ensemble")
    
    # Try to load the best performing model
    for model_name in ['random_forest_model.pkl', 'catboost_model.pkl', 'ensemble_model.pkl']:
        model_path = model_dir / model_name
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"Loaded old model: {model_name}")
            return model, model_name
    
    return None, None


def generate_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate simple but effective features."""
    print("Generating simple features...")
    
    feature_data = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
        
        if len(symbol_df) < 100:
            continue
        
        for i in range(50, len(symbol_df)):
            features = {}
            current = symbol_df.iloc[i]
            
            # Basic info
            features['symbol'] = symbol
            features['timestamp'] = current['timestamp']
            features['close'] = current['close']
            
            # Price features
            for window in [5, 10, 20]:
                if i >= window:
                    prev_price = symbol_df.iloc[i-window]['close']
                    features[f'return_{window}'] = (current['close'] - prev_price) / prev_price
                    
                    # Volatility
                    returns = symbol_df['close'].iloc[i-window:i].pct_change().dropna()
                    features[f'volatility_{window}'] = returns.std() if len(returns) > 0 else 0
                    
                    # Volume ratio
                    vol_avg = symbol_df['volume'].iloc[i-window:i].mean()
                    features[f'volume_ratio_{window}'] = current['volume'] / vol_avg if vol_avg > 0 else 1
                else:
                    features[f'return_{window}'] = 0
                    features[f'volatility_{window}'] = 0
                    features[f'volume_ratio_{window}'] = 1
            
            # Time features
            timestamp = pd.to_datetime(current['timestamp'])
            features['hour'] = timestamp.hour
            features['day_of_week'] = timestamp.dayofweek
            features['is_weekend'] = 1 if timestamp.dayofweek >= 5 else 0
            
            feature_data.append(features)
    
    return pd.DataFrame(feature_data)


def create_156_features_from_simple(simple_features: pd.DataFrame) -> pd.DataFrame:
    """Convert simple features to 156-dimensional format for v2.0 model."""
    print("Converting to 156-dimensional features...")
    
    df = simple_features.copy()
    
    # Basic features (already have ~15)
    base_features = [col for col in df.columns if col not in ['symbol', 'timestamp', 'close']]
    
    # Pad with zeros to reach 156 features
    target_features = 156
    current_features = len(base_features)
    
    for i in range(current_features, target_features):
        df[f'padding_{i}'] = 0
    
    print(f"Created {target_features} features from {current_features} base features")
    return df


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create labels for evaluation."""
    df = df.sort_values(['symbol', 'timestamp']).copy()
    
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        symbol_data = df[mask].copy()
        
        # Future price (5 minutes ahead)
        symbol_data['future_close'] = symbol_data['close'].shift(-5)
        
        # Future return with fees
        symbol_data['future_return'] = (symbol_data['future_close'] / symbol_data['close'] - 1) - 0.001
        
        # Binary label
        symbol_data['label'] = (symbol_data['future_return'] > 0.0015).astype(int)
        
        df.loc[mask, 'future_return'] = symbol_data['future_return']
        df.loc[mask, 'label'] = symbol_data['label']
    
    return df.dropna(subset=['label'])


def test_models():
    """Test both v2.0 and old models."""
    
    # Load data
    print("\nLoading test data...")
    conn = duckdb.connect("data/historical_data.duckdb", read_only=True)
    
    # Use recent data for testing
    query = """
    SELECT * FROM klines_btcusdt
    WHERE timestamp >= '2024-05-01' AND timestamp < '2024-05-15'
    ORDER BY timestamp
    LIMIT 10000
    """
    
    try:
        df = conn.execute(query).fetchdf()
        print(f"Loaded {len(df)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    finally:
        conn.close()
    
    # Generate features
    simple_features = generate_simple_features(df)
    print(f"Generated {len(simple_features)} feature vectors")
    
    # Create labels
    labeled_data = prepare_labels(simple_features)
    print(f"Created {len(labeled_data)} labeled samples")
    print(f"Positive rate: {labeled_data['label'].mean():.3f}")
    
    # Test old model first
    print("\n" + "="*40)
    print("TESTING OLD ENSEMBLE MODEL")
    print("="*40)
    
    old_model, old_model_name = load_old_ensemble_model()
    
    if old_model is not None:
        # Prepare features for old model (simple features)
        feature_cols = [col for col in labeled_data.columns 
                       if col not in ['symbol', 'timestamp', 'close', 'label', 'future_return', 'future_close']]
        
        X_simple = labeled_data[feature_cols].values
        y_true = labeled_data['label'].values
        
        # Get predictions
        if hasattr(old_model, 'predict_proba'):
            y_proba_old = old_model.predict_proba(X_simple)[:, 1]
        else:
            y_proba_old = old_model.predict(X_simple)
        
        auc_old = roc_auc_score(y_true, y_proba_old)
        
        print(f"Old model ({old_model_name}):")
        print(f"  AUC: {auc_old:.4f}")
        print(f"  Features used: {len(feature_cols)}")
        print(f"  Prediction range: [{y_proba_old.min():.4f}, {y_proba_old.max():.4f}]")
    
        # Simulate trading
        threshold = 0.7
        signals = (y_proba_old > threshold).sum()
        if signals > 0:
            signal_returns = labeled_data[y_proba_old > threshold]['future_return']
            avg_return = signal_returns.mean()
            win_rate = (signal_returns > 0).mean()
            
            print(f"  Trading simulation (threshold {threshold}):")
            print(f"    Signals: {signals}")
            print(f"    Average return: {avg_return:.4f}")
            print(f"    Win rate: {win_rate:.3f}")
    
    # Test v2.0 model
    print("\n" + "="*40)
    print("TESTING V2.0 MODEL")
    print("="*40)
    
    v2_model, v2_scaler, v2_metadata = load_v2_model()
    
    print(f"V2.0 Model metadata:")
    print(f"  Training AUC: {v2_metadata.get('performance', {}).get('auc', 'N/A')}")
    print(f"  Feature count: {v2_metadata.get('feature_count', 'N/A')}")
    
    # Convert to 156 features
    features_156 = create_156_features_from_simple(labeled_data)
    
    # Prepare features for v2.0 model
    exclude_cols = ['symbol', 'timestamp', 'close', 'label', 'future_return', 'future_close']
    feature_cols_156 = [col for col in features_156.columns if col not in exclude_cols]
    
    X_156 = features_156[feature_cols_156].values
    X_156_scaled = v2_scaler.transform(X_156)
    
    # Get predictions
    input_name = v2_model.get_inputs()[0].name
    output_name = v2_model.get_outputs()[0].name
    
    y_proba_v2 = v2_model.run([output_name], {input_name: X_156_scaled.astype(np.float32)})[0]
    
    if y_proba_v2.shape[1] == 2:
        y_proba_v2 = y_proba_v2[:, 1]
    else:
        y_proba_v2 = y_proba_v2.flatten()
    
    auc_v2 = roc_auc_score(y_true, y_proba_v2)
    
    print(f"V2.0 model:")
    print(f"  Test AUC: {auc_v2:.4f}")
    print(f"  Features used: {len(feature_cols_156)}")
    print(f"  Prediction range: [{y_proba_v2.min():.4f}, {y_proba_v2.max():.4f}]")
    
    # Simulate trading
    threshold = 0.7
    signals_v2 = (y_proba_v2 > threshold).sum()
    if signals_v2 > 0:
        signal_returns_v2 = labeled_data[y_proba_v2 > threshold]['future_return']
        avg_return_v2 = signal_returns_v2.mean()
        win_rate_v2 = (signal_returns_v2 > 0).mean()
        
        print(f"  Trading simulation (threshold {threshold}):")
        print(f"    Signals: {signals_v2}")
        print(f"    Average return: {avg_return_v2:.4f}")
        print(f"    Win rate: {win_rate_v2:.3f}")
    else:
        print(f"  No signals generated at threshold {threshold}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    if old_model is not None:
        print(f"Old Model AUC:  {auc_old:.4f}")
    print(f"V2.0 Model AUC: {auc_v2:.4f}")
    
    if old_model is not None:
        auc_diff = auc_v2 - auc_old
        print(f"AUC Difference: {auc_diff:.4f} ({'worse' if auc_diff < 0 else 'better'})")
        
        if auc_diff < -0.1:
            print("\n🚨 SIGNIFICANT PERFORMANCE DEGRADATION DETECTED!")
            print("Possible causes:")
            print("1. Feature quality issues (many random/padding features in v2.0)")
            print("2. Overfitting during training")
            print("3. Data distribution mismatch")
            print("4. Model complexity issues")


if __name__ == "__main__":
    test_models()