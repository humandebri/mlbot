#!/usr/bin/env python3
"""
CNN-based profit prediction model for cryptocurrency trading.

This model uses CNN to capture sequential patterns in price data
for better profit prediction than traditional models.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
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
    
    def __init__(self, sequence_length: int = 60, profit_threshold: float = 0.005):
        self.sequence_length = sequence_length
        self.profit_threshold = profit_threshold
        self.scaler = RobustScaler()
        self.model = None
        self.history = None
        
    def load_and_prepare_data(self, symbol: str = "BTCUSDT") -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare data for CNN training."""
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
          AND timestamp <= '2024-03-31'  # 3 months for training
        ORDER BY timestamp
        """
        
        data = conn.execute(query).df()
        conn.close()
        
        logger.info(f"Loaded {len(data)} records")
        return self.engineer_features(data)
    
    def engineer_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Engineer features for CNN input."""
        
        # Calculate basic features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['oc_ratio'] = (data['close'] - data['open']) / data['open']
        
        # Multi-timeframe features
        for period in [1, 3, 5, 10, 15, 30]:
            data[f'return_{period}'] = data['close'].pct_change(period)
            data[f'vol_{period}'] = data['returns'].rolling(period).std()
        
        # Volume features
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['log_volume'] = np.log(data['volume'] + 1)
        
        # Price momentum
        data['momentum_3'] = data['close'].pct_change(3)
        data['momentum_5'] = data['close'].pct_change(5)
        data['momentum_10'] = data['close'].pct_change(10)
        
        # Moving averages
        for ma in [5, 10, 20]:
            data[f'sma_{ma}'] = data['close'].rolling(ma).mean()
            data[f'price_vs_sma_{ma}'] = (data['close'] - data[f'sma_{ma}']) / data[f'sma_{ma}']
        
        # RSI approximation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_normalized'] = (data['rsi'] - 50) / 50
        
        # Bollinger Bands approximation
        data['bb_middle'] = data['close'].rolling(20).mean()
        data['bb_std'] = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Create target variable (profitable trades)
        transaction_cost = 0.0012  # 0.12% round-trip cost
        
        # Future returns at different horizons
        for horizon in [3, 5, 10]:
            future_return = data['close'].pct_change(horizon).shift(-horizon)
            
            # Long profitability
            long_profit = future_return - transaction_cost
            # Short profitability
            short_profit = -future_return - transaction_cost
            
            # Either direction profitable
            data[f'profitable_{horizon}m'] = ((long_profit > self.profit_threshold) | 
                                            (short_profit > self.profit_threshold)).astype(int)
            
            # Direction of best trade
            data[f'best_direction_{horizon}m'] = np.where(
                long_profit > short_profit, 1, 0  # 1 for long, 0 for short
            )
        
        # Select feature columns for CNN
        feature_cols = [
            'returns', 'log_returns', 'hl_ratio', 'oc_ratio',
            'return_1', 'return_3', 'return_5', 'return_10', 'return_15', 'return_30',
            'vol_1', 'vol_3', 'vol_5', 'vol_10', 'vol_15', 'vol_30',
            'volume_ratio', 'log_volume',
            'momentum_3', 'momentum_5', 'momentum_10',
            'price_vs_sma_5', 'price_vs_sma_10', 'price_vs_sma_20',
            'rsi_normalized', 'bb_position'
        ]
        
        # Remove NaN rows
        data = data.dropna()
        
        # Prepare features and targets
        X = data[feature_cols].values
        y = data['profitable_5m'].values  # Use 5-minute horizon as main target
        
        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Positive rate: {y.mean():.2%}")
        
        return X, y
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for CNN input."""
        
        X_seq = []
        y_seq = []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        logger.info(f"Created {len(X_seq)} sequences of length {self.sequence_length}")
        
        return X_seq, y_seq
    
    def build_cnn_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build CNN model for profit prediction."""
        
        model = Sequential([
            # First CNN block
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # Second CNN block
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Third CNN block
            Conv1D(filters=256, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            # Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile with class weights for imbalanced data
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the CNN model."""
        
        logger.info("Creating sequences for CNN training")
        X_seq, y_seq = self.create_sequences(X, y)
        
        # Scale features
        logger.info("Scaling features")
        X_seq_scaled = np.zeros_like(X_seq)
        for i in range(X_seq.shape[2]):  # Scale each feature
            X_seq_scaled[:, :, i] = self.scaler.fit_transform(X_seq[:, :, i])
        
        # Train/validation split (time-based)
        split_idx = int(len(X_seq_scaled) * 0.8)
        X_train = X_seq_scaled[:split_idx]
        y_train = y_seq[:split_idx]
        X_val = X_seq_scaled[split_idx:]
        y_val = y_seq[split_idx:]
        
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        logger.info(f"Training positive rate: {y_train.mean():.2%}")
        logger.info(f"Validation positive rate: {y_val.mean():.2%}")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_cnn_model(input_shape)
        
        logger.info(f"Built CNN model with input shape: {input_shape}")
        print(self.model.summary())
        
        # Calculate class weights for imbalanced data
        pos_weight = len(y_train) / (2 * y_train.sum())
        neg_weight = len(y_train) / (2 * (len(y_train) - y_train.sum()))
        class_weight = {0: neg_weight, 1: pos_weight}
        
        logger.info(f"Class weights: {class_weight}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train model
        logger.info("Starting CNN training")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        val_predictions = self.model.predict(X_val)
        val_pred_binary = (val_predictions > 0.5).astype(int)
        
        # Calculate metrics
        val_accuracy = (val_pred_binary.flatten() == y_val).mean()
        val_precision = (val_pred_binary.flatten() & y_val).sum() / val_pred_binary.sum() if val_pred_binary.sum() > 0 else 0
        val_recall = (val_pred_binary.flatten() & y_val).sum() / y_val.sum() if y_val.sum() > 0 else 0
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
        
        results = {
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'positive_rate_train': y_train.mean(),
            'positive_rate_val': y_val.mean(),
            'model_params': self.model.count_params()
        }
        
        logger.info(f"Training completed. Validation metrics:")
        logger.info(f"  Accuracy: {val_accuracy:.3f}")
        logger.info(f"  Precision: {val_precision:.3f}")
        logger.info(f"  Recall: {val_recall:.3f}")
        logger.info(f"  F1-Score: {val_f1:.3f}")
        
        return results
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results/cnn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path: str):
        """Save the trained model."""
        if self.model is not None:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.warning("No model to save")


def main():
    """Train CNN profit prediction model."""
    
    logger.info("Starting CNN profit prediction model training")
    
    # Initialize predictor
    predictor = CNNProfitPredictor(sequence_length=60, profit_threshold=0.005)
    
    # Load and prepare data
    X, y = predictor.load_and_prepare_data()
    
    # Train model
    results = predictor.train_model(X, y)
    
    # Plot training history
    predictor.plot_training_history()
    
    # Save model
    model_path = "models/cnn_profit_predictor.h5"
    Path("models").mkdir(exist_ok=True)
    predictor.save_model(model_path)
    
    # Print final results
    print("\n" + "="*60)
    print("CNN収益予測モデル訓練結果")
    print("="*60)
    
    print(f"\n📊 データセット:")
    print(f"  訓練サンプル数: {results['train_samples']:,}")
    print(f"  検証サンプル数: {results['val_samples']:,}")
    print(f"  訓練陽性率: {results['positive_rate_train']:.2%}")
    print(f"  検証陽性率: {results['positive_rate_val']:.2%}")
    
    print(f"\n🎯 モデル性能:")
    print(f"  精度 (Accuracy): {results['val_accuracy']:.3f}")
    print(f"  適合率 (Precision): {results['val_precision']:.3f}")
    print(f"  再現率 (Recall): {results['val_recall']:.3f}")
    print(f"  F1スコア: {results['val_f1']:.3f}")
    print(f"  モデルパラメータ数: {results['model_params']:,}")
    
    print(f"\n💡 改善点:")
    if results['val_precision'] > 0.5:
        print(f"  ✅ 適合率{results['val_precision']:.1%}は良好")
    else:
        print(f"  ⚠️  適合率{results['val_precision']:.1%}は要改善")
    
    if results['val_recall'] > 0.3:
        print(f"  ✅ 再現率{results['val_recall']:.1%}は良好")
    else:
        print(f"  ⚠️  再現率{results['val_recall']:.1%}は要改善")
    
    if results['val_f1'] > 0.4:
        print(f"  ✅ F1スコア{results['val_f1']:.1%}は良好")
    else:
        print(f"  ⚠️  F1スコア{results['val_f1']:.1%}は要改善")
    
    print(f"\n💾 保存済み:")
    print(f"  モデル: {model_path}")
    print(f"  訓練履歴: backtest_results/cnn_training_history.png")


if __name__ == "__main__":
    main()