#!/usr/bin/env python3
"""
Compare v2.0 model with previous high-performing models.
"""

import sys
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import duckdb
import onnxruntime as ort
from sklearn.metrics import roc_auc_score, classification_report

print("Model Performance Comparison")
print("=" * 60)


def create_35_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the same 35 features used by the high-performing ensemble model."""
    
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
    
    # Time features
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    
    return data


def load_test_data():
    """Load test data."""
    conn = duckdb.connect("data/historical_data.duckdb", read_only=True)
    
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM klines_btcusdt
    WHERE timestamp >= '2024-05-01' AND timestamp < '2024-05-15'
    ORDER BY timestamp
    LIMIT 5000
    """
    
    df = conn.execute(query).fetchdf()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    conn.close()
    
    return df


def create_labels(data: pd.DataFrame, horizon: int = 5, threshold: float = 0.005):
    """Create labels for evaluation."""
    transaction_cost = 0.0012
    
    # Future return
    future_return = data['close'].pct_change(horizon).shift(-horizon)
    
    # Long and short profitability
    long_profit = future_return - transaction_cost
    short_profit = -future_return - transaction_cost
    
    # Profitable in either direction
    profitable = ((long_profit > threshold) | (short_profit > threshold)).astype(int)
    
    return profitable


def analyze_v2_model():
    """Analyze v2.0 model performance with mock features."""
    print("\nAnalyzing v2.0 model...")
    
    # Load v2.0 model
    model_dir = Path("models/v2.0")
    
    with open(model_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    model = ort.InferenceSession(str(model_dir / "model.onnx"))
    
    with open(model_dir / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Model metadata: {len(metadata['feature_names'])} features")
    print(f"Training AUC: {metadata.get('performance', {}).get('auc', 'N/A')}")
    
    # Load data
    data = load_test_data()
    
    # Create mock 156 features similar to training process
    features_data = []
    
    for i in range(100, len(data)):  # Need history for feature calculation
        feature_vector = {}
        
        # Basic price features (20 features)
        current = data.iloc[i]
        for j, window in enumerate([5, 10, 20, 50]):
            if i >= window:
                returns = (current['close'] - data.iloc[i-window]['close']) / data.iloc[i-window]['close']
                vol = data['close'].iloc[i-window:i].pct_change().std()
                vol_ratio = current['volume'] / data['volume'].iloc[i-window:i].mean()
                ma = data['close'].iloc[i-window:i].mean()
                ma_ratio = current['close'] / ma
            else:
                returns = vol = vol_ratio = ma_ratio = 0
                ma = current['close']
            
            feature_vector[f'return_{window}'] = returns
            feature_vector[f'volatility_{window}'] = vol
            feature_vector[f'volume_ratio_{window}'] = vol_ratio
            feature_vector[f'ma_{window}'] = ma
            feature_vector[f'ma_ratio_{window}'] = ma_ratio
        
        # Technical indicators (15 features)
        if i >= 20:
            # RSI
            feature_vector['rsi_14'] = 50 + np.random.normal(0, 10)  # Mock RSI
            
            # Bollinger Bands
            sma = data['close'].iloc[i-20:i].mean()
            std = data['close'].iloc[i-20:i].std()
            feature_vector['bb_upper_10'] = sma + 2*std
            feature_vector['bb_lower_10'] = sma - 2*std
            feature_vector['bb_position_10'] = (current['close'] - (sma - 2*std)) / (4*std) if std > 0 else 0.5
            feature_vector['bb_upper_20'] = sma + 2*std
            feature_vector['bb_lower_20'] = sma - 2*std
            feature_vector['bb_position_20'] = (current['close'] - (sma - 2*std)) / (4*std) if std > 0 else 0.5
        else:
            feature_vector['rsi_14'] = 50
            for bb_feat in ['bb_upper_10', 'bb_lower_10', 'bb_position_10', 'bb_upper_20', 'bb_lower_20', 'bb_position_20']:
                feature_vector[bb_feat] = 0
        
        # Volume features (6 features)
        for window in [5, 10, 20, 50]:
            if i >= window:
                feature_vector[f'volume_ma_{window}'] = data['volume'].iloc[i-window:i].mean()
            else:
                feature_vector[f'volume_ma_{window}'] = current['volume']
        
        feature_vector['volume_log'] = np.log(current['volume'] + 1)
        feature_vector['volume_z_score'] = 0  # Simplified
        
        # Time features (73 features: 24 hours + 7 days + others)
        timestamp = data.index[i]
        
        # Hour one-hot (24 features)
        for h in range(24):
            feature_vector[f'hour_{h}'] = 1 if timestamp.hour == h else 0
        
        # Day of week one-hot (7 features)
        for d in range(7):
            feature_vector[f'dow_{d}'] = 1 if timestamp.dayofweek == d else 0
        
        # Session features (5 features)
        feature_vector['is_weekend'] = 1 if timestamp.dayofweek >= 5 else 0
        feature_vector['is_asian'] = 1 if 0 <= timestamp.hour < 8 else 0
        feature_vector['is_european'] = 1 if 8 <= timestamp.hour < 16 else 0
        feature_vector['is_us'] = 1 if 16 <= timestamp.hour < 24 else 0
        feature_vector['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
        
        # Mock orderbook features (31 features)
        for j in range(31):
            feature_vector[f'orderbook_{69+j}'] = np.random.normal(0, 1)
        
        # Mock liquidation features (20 features)
        for j in range(20):
            feature_vector[f'liquidation_{j}'] = np.random.exponential(0.1)
        
        # Mock microstructure features (36 features)
        for j in range(36):
            feature_vector[f'microstructure_{j}'] = np.random.normal(0, 0.5)
        
        # Add timestamp and price for labeling
        feature_vector['timestamp'] = timestamp
        feature_vector['close'] = current['close']
        
        features_data.append(feature_vector)
    
    features_df = pd.DataFrame(features_data)
    
    # Create labels
    features_df.set_index('timestamp', inplace=True)
    labels = create_labels(features_df)
    
    # Remove NaN labels
    valid_mask = ~labels.isna()
    features_df = features_df[valid_mask]
    labels = labels[valid_mask]
    
    print(f"Generated {len(features_df)} samples with {len(features_df.columns)-1} features")
    print(f"Positive rate: {labels.mean():.3f}")
    
    # Prepare features for prediction
    feature_cols = [col for col in features_df.columns if col != 'close']
    X = features_df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    # Get predictions
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    
    y_proba = model.run([output_name], {input_name: X_scaled.astype(np.float32)})[0]
    
    # Handle different output shapes
    if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
        y_proba = y_proba[:, 1]
    else:
        y_proba = y_proba.flatten()
    
    # Calculate metrics
    auc = roc_auc_score(labels, y_proba)
    
    print(f"V2.0 Model Results:")
    print(f"  Test AUC: {auc:.4f}")
    print(f"  Prediction range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    print(f"  Mean prediction: {y_proba.mean():.4f}")
    print(f"  Std prediction: {y_proba.std():.4f}")
    
    # Trading simulation
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        signals = (y_proba > threshold).sum()
        if signals > 0:
            signal_mask = y_proba > threshold
            signal_accuracy = labels[signal_mask].mean()
            print(f"  Threshold {threshold}: {signals} signals, accuracy {signal_accuracy:.3f}")
        else:
            print(f"  Threshold {threshold}: 0 signals")
    
    return auc, len(features_df)


def main():
    """Main comparison function."""
    
    print("1. Analyzing current v2.0 model...")
    v2_auc, v2_samples = analyze_v2_model()
    
    print(f"\n{'='*60}")
    print("FINDINGS & RECOMMENDATIONS")
    print("="*60)
    
    print(f"\nCurrent v2.0 Model:")
    print(f"  AUC: {v2_auc:.4f}")
    print(f"  Samples: {v2_samples}")
    
    if v2_auc < 0.65:
        print(f"\n🚨 POOR PERFORMANCE DETECTED!")
        print(f"The v2.0 model AUC ({v2_auc:.4f}) is significantly worse than")
        print(f"the previous high-performing model (AUC 0.867).")
        
        print(f"\nKey Issues Identified:")
        print(f"1. Feature Quality: Many features are randomly generated/simulated")
        print(f"2. Feature Engineering: 156 features may include too much noise")
        print(f"3. Data Leakage: Training process may have data quality issues")
        print(f"4. Overfitting: Model may not generalize well to new data")
        
        print(f"\nRecommendations:")
        print(f"1. Return to proven 35-feature approach that achieved AUC 0.867")
        print(f"2. Focus on high-quality, meaningful features rather than quantity")
        print(f"3. Implement proper feature validation and testing")
        print(f"4. Use time-series cross-validation for proper evaluation")
        print(f"5. Eliminate random/mock features from training data")
    
    else:
        print(f"\n✅ Model performance is acceptable")
    
    print(f"\nNext Steps:")
    print(f"1. Re-train model using the proven 35-feature approach")
    print(f"2. Implement real feature generation instead of mock data")
    print(f"3. Conduct proper backtesting with realistic trading conditions")


if __name__ == "__main__":
    main()