#!/usr/bin/env python3
"""
CNN-based profit prediction model using TensorFlow.
Uses 1D CNN for time series pattern recognition.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense, 
                                     Dropout, BatchNormalization, LSTM, 
                                     Bidirectional, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class CNNProfitPredictor:
    """CNN model for predicting profitable trading opportunities."""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = None
        
    def create_sequences(self, data, features):
        """Create sequences for CNN input."""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length - 5):
            # Get sequence of features
            seq = data[features].iloc[i:i+self.sequence_length].values
            sequences.append(seq)
            
            # Calculate target (profitable if 5-bar return > 0.3% after fees)
            entry_price = data['close'].iloc[i + self.sequence_length]
            exit_price = data['close'].iloc[i + self.sequence_length + 5]
            pnl = (exit_price - entry_price) / entry_price - 0.0012  # fees
            targets.append(1 if pnl > 0.003 else 0)  # 0.3% profit threshold
        
        return np.array(sequences), np.array(targets)
    
    def build_cnn_model(self, input_shape):
        """Build 1D CNN model for time series."""
        model = Sequential([
            # First Conv block
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   input_shape=input_shape, padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # Second Conv block
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # Third Conv block
            Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dropout(0.3),
            
            # Dense layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def build_lstm_cnn_hybrid(self, input_shape):
        """Build hybrid CNN-LSTM model."""
        model = Sequential([
            # CNN feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   input_shape=input_shape, padding='same'),
            BatchNormalization(),
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            
            # Bidirectional LSTM
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, model_type='cnn'):
        """Train the model."""
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        if model_type == 'cnn':
            self.model = self.build_cnn_model(input_shape)
        elif model_type == 'hybrid':
            self.model = self.build_lstm_cnn_hybrid(input_shape)
        else:
            raise ValueError("model_type must be 'cnn' or 'hybrid'")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_auc', 
            patience=20, 
            restore_best_weights=True,
            mode='max'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
        )
        
        # Class weights for imbalanced data
        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        class_weight = {0: 1.0, 1: pos_weight}
        
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weight,
            verbose=1
        )
        
        return history


def create_technical_features(data):
    """Create technical features for CNN."""
    # Price features
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['hl_ratio'] = (data['high'] - data['low']) / data['close']
    data['oc_ratio'] = (data['close'] - data['open']) / data['open']
    
    # Momentum
    for period in [3, 5, 10, 20]:
        data[f'momentum_{period}'] = data['close'].pct_change(period)
    
    # Volatility
    for window in [5, 10, 20]:
        data[f'volatility_{window}'] = data['returns'].rolling(window).std()
        data[f'volatility_ratio_{window}'] = (
            data[f'volatility_{window}'] / 
            data[f'volatility_{window}'].rolling(50).mean()
        )
    
    # Moving averages and ratios
    for ma in [5, 10, 20, 50]:
        data[f'sma_{ma}'] = data['close'].rolling(ma).mean()
        data[f'price_to_sma_{ma}'] = data['close'] / data[f'sma_{ma}']
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_diff'] = data['macd'] - data['macd_signal']
    
    # Bollinger Bands
    bb_mean = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    data['bb_upper'] = bb_mean + (bb_std * 2)
    data['bb_lower'] = bb_mean - (bb_std * 2)
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / bb_mean
    
    # Volume features
    data['volume_ma'] = data['volume'].rolling(20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data['volume_trend'] = data['volume'].rolling(5).mean() / data['volume'].rolling(20).mean()
    
    # Price patterns
    data['higher_high'] = ((data['high'] > data['high'].shift(1)) & 
                          (data['high'].shift(1) > data['high'].shift(2))).astype(int)
    data['lower_low'] = ((data['low'] < data['low'].shift(1)) & 
                        (data['low'].shift(1) < data['low'].shift(2))).astype(int)
    
    # Support/Resistance levels
    data['resistance_distance'] = (data['high'].rolling(20).max() - data['close']) / data['close']
    data['support_distance'] = (data['close'] - data['low'].rolling(20).min()) / data['close']
    
    # Market microstructure
    data['spread'] = (data['high'] - data['low']) / data['close']
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['weighted_close'] = (data['high'] + data['low'] + 2 * data['close']) / 4
    
    return data


def train_cnn_model():
    """Train CNN model for profit prediction."""
    
    logger.info("Starting CNN model training")
    
    # Load data
    conn = duckdb.connect("data/historical_data.duckdb")
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM klines_btcusdt
    WHERE timestamp >= '2024-01-01' AND timestamp <= '2024-07-31'
    ORDER BY timestamp
    """
    
    data = conn.execute(query).df()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    conn.close()
    
    logger.info(f"Loaded {len(data)} records")
    
    # Create features
    data = create_technical_features(data)
    
    # Select features for CNN
    feature_cols = [col for col in data.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume']]
    
    # Remove NaN
    data = data.dropna()
    logger.info(f"Data after removing NaN: {len(data)} records")
    
    # Initialize CNN predictor
    cnn = CNNProfitPredictor(sequence_length=60)
    
    # Create sequences
    logger.info("Creating sequences for CNN...")
    X, y = cnn.create_sequences(data, feature_cols)
    logger.info(f"Created {len(X)} sequences")
    logger.info(f"Positive class ratio: {y.mean():.2%}")
    
    # Normalize features
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_normalized = cnn.scaler.fit_transform(X_reshaped)
    X = X_normalized.reshape(X.shape)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    print("\n" + "="*80)
    print("🧠 CNNモデル訓練開始")
    print("="*80)
    
    # Train CNN model
    print("\n📊 1D CNNモデル訓練中...")
    history_cnn = cnn.train(X_train, y_train, X_val, y_val, model_type='cnn')
    
    # Evaluate CNN
    cnn_eval = cnn.model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ CNN結果:")
    print(f"  テスト精度: {cnn_eval[1]:.4f}")
    print(f"  テストAUC: {cnn_eval[2]:.4f}")
    
    # Save CNN model
    cnn.model.save("models/cnn_profit_model.h5")
    logger.info("Saved CNN model")
    
    # Train hybrid model
    print("\n📊 CNN-LSTMハイブリッドモデル訓練中...")
    cnn_hybrid = CNNProfitPredictor(sequence_length=60)
    cnn_hybrid.scaler = cnn.scaler  # Use same scaler
    history_hybrid = cnn_hybrid.train(X_train, y_train, X_val, y_val, model_type='hybrid')
    
    # Evaluate hybrid
    hybrid_eval = cnn_hybrid.model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ ハイブリッドモデル結果:")
    print(f"  テスト精度: {hybrid_eval[1]:.4f}")
    print(f"  テストAUC: {hybrid_eval[2]:.4f}")
    
    # Save hybrid model
    cnn_hybrid.model.save("models/cnn_lstm_hybrid_model.h5")
    logger.info("Saved hybrid model")
    
    # Model comparison
    print(f"\n📊 モデル比較:")
    print(f"{'モデル':<20} {'精度':<10} {'AUC':<10}")
    print("-" * 40)
    print(f"{'1D CNN':<20} {cnn_eval[1]:<10.4f} {cnn_eval[2]:<10.4f}")
    print(f"{'CNN-LSTM Hybrid':<20} {hybrid_eval[1]:<10.4f} {hybrid_eval[2]:<10.4f}")
    
    # Select best model
    if hybrid_eval[2] > cnn_eval[2]:
        best_model = cnn_hybrid.model
        best_name = "CNN-LSTM Hybrid"
        best_auc = hybrid_eval[2]
    else:
        best_model = cnn.model
        best_name = "1D CNN"
        best_auc = cnn_eval[2]
    
    print(f"\n🏆 最良モデル: {best_name} (AUC: {best_auc:.4f})")
    
    # Quick profitability test
    print("\n" + "="*80)
    print("💰 収益性クイックテスト")
    print("="*80)
    
    # Load test data
    conn = duckdb.connect("data/historical_data.duckdb")
    test_query = """
    SELECT timestamp, open, high, low, close, volume
    FROM klines_btcusdt
    WHERE timestamp >= '2024-08-01' AND timestamp <= '2024-09-30'
    ORDER BY timestamp
    """
    
    test_data = conn.execute(test_query).df()
    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
    test_data.set_index('timestamp', inplace=True)
    conn.close()
    
    # Create features
    test_data = create_technical_features(test_data)
    test_data = test_data.dropna()
    
    # Create sequences
    X_live, _ = cnn.create_sequences(test_data, feature_cols)
    X_live_reshaped = X_live.reshape(-1, X_live.shape[-1])
    X_live_normalized = cnn.scaler.transform(X_live_reshaped)
    X_live = X_live_normalized.reshape(X_live.shape)
    
    # Get predictions
    predictions = best_model.predict(X_live)
    
    # Simulate trading
    capital = 100000
    position_size = 0.02
    confidence_threshold = 0.7
    trades = []
    
    high_confidence_indices = np.where(predictions > confidence_threshold)[0]
    logger.info(f"Found {len(high_confidence_indices)} high confidence signals")
    
    for idx in high_confidence_indices[:100]:  # Test first 100 signals
        if idx + 60 + 5 >= len(test_data):
            continue
        
        entry_idx = idx + 60  # After sequence
        entry_price = test_data['close'].iloc[entry_idx]
        exit_price = test_data['close'].iloc[entry_idx + 5]
        
        # Direction based on model confidence and momentum
        momentum = test_data['momentum_5'].iloc[entry_idx]
        direction = 'long' if momentum > 0 else 'short'
        
        if direction == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        
        pnl_pct -= 0.0012  # fees
        
        position_value = capital * position_size
        pnl_dollar = position_value * pnl_pct
        
        trades.append({
            'pnl_dollar': pnl_dollar,
            'pnl_pct': pnl_pct * 100,
            'confidence': float(predictions[idx])
        })
        
        capital += pnl_dollar
    
    if trades:
        trades_df = pd.DataFrame(trades)
        total_return = (capital - 100000) / 100000 * 100
        win_rate = len(trades_df[trades_df['pnl_dollar'] > 0]) / len(trades_df) * 100
        avg_confidence = trades_df['confidence'].mean()
        
        print(f"\n📊 バックテスト結果 (2ヶ月):")
        print(f"  取引数: {len(trades_df)}")
        print(f"  総収益率: {total_return:.2f}%")
        print(f"  月次収益率: {total_return/2:.2f}%")
        print(f"  勝率: {win_rate:.1f}%")
        print(f"  平均信頼度: {avg_confidence:.3f}")
        
        if total_return > 0:
            print(f"\n🎉 収益性達成！CNNモデルによる改善成功")
            print(f"  推定年次収益: {total_return * 6:.1f}%")
        else:
            print(f"\n⚠️ まだ損失ですが、さらなる最適化余地あり")
    
    print(f"\n💡 CNNモデルの利点:")
    print(f"  ✅ 時系列パターンの自動学習")
    print(f"  ✅ 複雑な非線形関係の捕捉")
    print(f"  ✅ ノイズに対する堅牢性")
    print(f"  ✅ 特徴量エンジニアリングの自動化")
    
    return best_model, best_auc


if __name__ == "__main__":
    # Verify TensorFlow installation
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Metal GPU support: {tf.config.list_physical_devices('GPU')}")
    
    # Train model
    train_cnn_model()