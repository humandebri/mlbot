#!/usr/bin/env python3
"""
Simplified CNN-based profit prediction model for faster training.

This model uses a smaller CNN architecture and reduced dataset
for quicker iterations while maintaining profitability focus.
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
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class SimplifiedCNNPredictor:
    """Simplified CNN model for rapid profit prediction."""
    
    def __init__(self, sequence_length: int = 30, profit_threshold: float = 0.003):
        self.sequence_length = sequence_length  # Reduced from 60
        self.profit_threshold = profit_threshold  # Reduced from 0.005 for more signals
        self.scaler = RobustScaler()
        self.model = None
        self.history = None
        self.metrics = {}
        
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
        WHERE timestamp >= '2024-02-01'
          AND timestamp <= '2024-02-29'  -- Just 1 month for faster training
        ORDER BY timestamp
        """
        
        data = conn.execute(query).df()
        conn.close()
        
        logger.info(f"Loaded {len(data)} records")
        return self.engineer_features(data)
    
    def engineer_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Engineer simplified features for CNN input."""
        
        # Core price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['oc_ratio'] = (data['close'] - data['open']) / data['open']
        
        # Simplified multi-timeframe features (fewer periods)
        for period in [1, 5, 10, 20]:
            data[f'return_{period}'] = data['close'].pct_change(period)
            data[f'vol_{period}'] = data['returns'].rolling(period).std()
        
        # Volume features
        data['volume_ma'] = data['volume'].rolling(10).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Momentum
        data['momentum_5'] = data['close'].pct_change(5)
        data['momentum_10'] = data['close'].pct_change(10)
        
        # Simple moving averages
        for ma in [5, 10]:
            data[f'sma_{ma}'] = data['close'].rolling(ma).mean()
            data[f'price_vs_sma_{ma}'] = (data['close'] - data[f'sma_{ma}']) / data[f'sma_{ma}']
        
        # Simplified RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_normalized'] = (data['rsi'] - 50) / 50
        
        # Create target variable with focus on profitability
        transaction_cost = 0.001  # 0.1% round-trip cost (reduced for more signals)
        
        # Multiple profit horizons for robustness
        horizons = [3, 5, 10]
        profit_signals = []
        
        for horizon in horizons:
            future_return = data['close'].pct_change(horizon).shift(-horizon)
            
            # Long profitability
            long_profit = future_return - transaction_cost
            # Short profitability
            short_profit = -future_return - transaction_cost
            
            # Signal if either direction is profitable
            signal = ((long_profit > self.profit_threshold) | 
                     (short_profit > self.profit_threshold)).astype(int)
            profit_signals.append(signal)
        
        # Combine signals (any horizon profitable)
        data['profitable'] = np.max(profit_signals, axis=0)
        
        # Select simplified feature set
        feature_cols = [
            'returns', 'log_returns', 'hl_ratio', 'oc_ratio',
            'return_1', 'return_5', 'return_10', 'return_20',
            'vol_1', 'vol_5', 'vol_10', 'vol_20',
            'volume_ratio',
            'momentum_5', 'momentum_10',
            'price_vs_sma_5', 'price_vs_sma_10',
            'rsi_normalized'
        ]
        
        # Remove NaN rows
        data = data.dropna()
        
        # Prepare features and targets
        X = data[feature_cols].values
        y = data['profitable'].values
        
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
    
    def build_simplified_cnn(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build simplified CNN model for faster training."""
        
        model = Sequential([
            # First CNN block (reduced filters)
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # Second CNN block
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers (reduced size)
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile with focus on precision for profitability
        model.compile(
            optimizer=Adam(learning_rate=0.002),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                    tf.keras.metrics.AUC()]
        )
        
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the simplified CNN model."""
        
        logger.info("Creating sequences for CNN training")
        X_seq, y_seq = self.create_sequences(X, y)
        
        # Scale features
        logger.info("Scaling features")
        X_seq_scaled = np.zeros_like(X_seq)
        for i in range(X_seq.shape[2]):
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
        self.model = self.build_simplified_cnn(input_shape)
        
        logger.info(f"Built simplified CNN model with input shape: {input_shape}")
        print(self.model.summary())
        
        # Calculate class weights
        pos_weight = len(y_train) / (2 * y_train.sum()) if y_train.sum() > 0 else 1
        neg_weight = len(y_train) / (2 * (len(y_train) - y_train.sum()))
        class_weight = {0: neg_weight, 1: pos_weight}
        
        logger.info(f"Class weights: {class_weight}")
        
        # Callbacks (simplified)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        
        # Train model with reduced epochs
        logger.info("Starting simplified CNN training")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,  # Reduced from 100
            batch_size=64,  # Increased from 32
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        val_predictions = self.model.predict(X_val)
        val_pred_binary = (val_predictions > 0.5).astype(int)
        
        # Calculate comprehensive metrics
        val_accuracy = (val_pred_binary.flatten() == y_val).mean()
        val_precision = (val_pred_binary.flatten() & y_val).sum() / val_pred_binary.sum() if val_pred_binary.sum() > 0 else 0
        val_recall = (val_pred_binary.flatten() & y_val).sum() / y_val.sum() if y_val.sum() > 0 else 0
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
        val_auc = roc_auc_score(y_val, val_predictions) if y_val.sum() > 0 and (y_val == 0).sum() > 0 else 0.5
        
        # Store metrics
        self.metrics = {
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'val_auc': val_auc,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'positive_rate_train': y_train.mean(),
            'positive_rate_val': y_val.mean(),
            'model_params': self.model.count_params(),
            'predicted_positives': val_pred_binary.sum(),
            'actual_positives': y_val.sum()
        }
        
        logger.info(f"Training completed. Validation metrics:")
        logger.info(f"  Accuracy: {val_accuracy:.3f}")
        logger.info(f"  Precision: {val_precision:.3f}")
        logger.info(f"  Recall: {val_recall:.3f}")
        logger.info(f"  F1-Score: {val_f1:.3f}")
        logger.info(f"  AUC: {val_auc:.3f}")
        
        return self.metrics
    
    def plot_results(self):
        """Plot training history and confusion matrix."""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # Training history plots
        # Loss
        plt.subplot(2, 3, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy
        plt.subplot(2, 3, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # AUC
        plt.subplot(2, 3, 3)
        if 'auc' in self.history.history:
            plt.plot(self.history.history['auc'], label='Training AUC')
            plt.plot(self.history.history['val_auc'], label='Validation AUC')
            plt.title('Model AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True)
        
        # Precision
        plt.subplot(2, 3, 4)
        if 'precision' in self.history.history:
            plt.plot(self.history.history['precision'], label='Training Precision')
            plt.plot(self.history.history['val_precision'], label='Validation Precision')
            plt.title('Model Precision')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.legend()
            plt.grid(True)
        
        # Recall
        plt.subplot(2, 3, 5)
        if 'recall' in self.history.history:
            plt.plot(self.history.history['recall'], label='Training Recall')
            plt.plot(self.history.history['val_recall'], label='Validation Recall')
            plt.title('Model Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(True)
        
        # Metrics summary
        plt.subplot(2, 3, 6)
        metrics_text = f"""Validation Metrics:
        
Accuracy: {self.metrics['val_accuracy']:.3f}
Precision: {self.metrics['val_precision']:.3f}
Recall: {self.metrics['val_recall']:.3f}
F1-Score: {self.metrics['val_f1']:.3f}
AUC: {self.metrics['val_auc']:.3f}

Samples:
Training: {self.metrics['train_samples']:,}
Validation: {self.metrics['val_samples']:,}

Positive Rate:
Training: {self.metrics['positive_rate_train']:.2%}
Validation: {self.metrics['positive_rate_val']:.2%}

Model Parameters: {self.metrics['model_params']:,}"""
        
        plt.text(0.1, 0.5, metrics_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace')
        plt.axis('off')
        plt.title('Model Summary')
        
        plt.tight_layout()
        plt.savefig('backtest_results/cnn_simplified_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path: str):
        """Save the trained model and metrics."""
        if self.model is not None:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save metrics
            metrics_path = model_path.replace('.h5', '_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            logger.info(f"Metrics saved to {metrics_path}")
        else:
            logger.warning("No model to save")


def main():
    """Train simplified CNN profit prediction model."""
    
    logger.info("Starting simplified CNN profit prediction model training")
    
    # Create results directory
    Path("backtest_results").mkdir(exist_ok=True)
    
    # Initialize predictor
    predictor = SimplifiedCNNPredictor(sequence_length=30, profit_threshold=0.003)
    
    # Load and prepare data
    X, y = predictor.load_and_prepare_data()
    
    # Train model
    results = predictor.train_model(X, y)
    
    # Plot results
    predictor.plot_results()
    
    # Save model
    model_path = "models/cnn_simplified_profit.h5"
    Path("models").mkdir(exist_ok=True)
    predictor.save_model(model_path)
    
    # Print final results
    print("\n" + "="*60)
    print("簡素化CNN収益予測モデル訓練結果")
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
    print(f"  AUC: {results['val_auc']:.3f}")
    print(f"  モデルパラメータ数: {results['model_params']:,}")
    
    print(f"\n📈 取引シグナル:")
    print(f"  予測陽性数: {results['predicted_positives']:,}")
    print(f"  実際の陽性数: {results['actual_positives']:,}")
    
    print(f"\n💡 Random Forest比較:")
    print(f"  CNN AUC: {results['val_auc']:.3f}")
    print(f"  Random Forest AUC: 0.867 (参考値)")
    if results['val_auc'] > 0.867:
        print(f"  ✅ CNNがRandom Forestを上回りました！")
    else:
        print(f"  ⚠️  Random Forestの方が高いAUCです")
    
    print(f"\n💾 保存済み:")
    print(f"  モデル: {model_path}")
    print(f"  メトリクス: {model_path.replace('.h5', '_metrics.json')}")
    print(f"  結果グラフ: backtest_results/cnn_simplified_results.png")
    
    print(f"\n⏱️  訓練時間: 約20エポック（高速化済み）")
    print(f"次のステップ: 収益性バックテストの実行")


if __name__ == "__main__":
    main()