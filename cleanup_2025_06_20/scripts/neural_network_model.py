#!/usr/bin/env python3
"""
Neural Network model for profitable cryptocurrency trading with standard fees.
Uses LSTM and advanced architectures for better time series prediction.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")


class CryptoDataset(Dataset):
    """Custom dataset for cryptocurrency time series."""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class AdvancedLSTM(nn.Module):
    """Advanced LSTM with attention mechanism for profit prediction."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super(AdvancedLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Feature extraction
        features = self.feature_net(context)
        
        # Output
        output = self.output(features)
        
        return output


class TransformerModel(nn.Module):
    """Transformer-based model for time series prediction."""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dropout=0.3):
        super(TransformerModel, self).__init__()
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer
        x = self.transformer(x)
        
        # Global pooling
        x = torch.mean(x, dim=1)
        
        # Output
        output = self.output_net(x)
        
        return output


class NeuralNetworkTrader:
    """Neural network-based trading system."""
    
    def __init__(self, sequence_length=60, model_type='lstm'):
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.device = device
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare training data."""
        logger.info("Loading data for neural network training")
        
        conn = duckdb.connect("data/historical_data.duckdb")
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM klines_btcusdt
        WHERE timestamp >= '2024-01-01' AND timestamp <= '2024-04-30'
        ORDER BY timestamp
        """
        
        data = conn.execute(query).df()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        conn.close()
        
        # Engineer features
        data = self.engineer_features(data)
        
        # Create sequences and labels
        X, y = self.create_sequences(data)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for neural network."""
        
        # Price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['oc_ratio'] = (data['close'] - data['open']) / data['open']
        
        # Multi-timeframe returns
        for period in [1, 3, 5, 10, 15, 30]:
            data[f'return_{period}'] = data['close'].pct_change(period)
        
        # Volatility
        for window in [5, 10, 20, 30]:
            data[f'vol_{window}'] = data['returns'].rolling(window).std()
            data[f'vol_rank_{window}'] = data[f'vol_{window}'].rolling(window*2).rank(pct=True)
        
        # Price momentum
        for period in [3, 5, 10, 15]:
            data[f'momentum_{period}'] = data['close'].pct_change(period)
            data[f'momentum_acc_{period}'] = data[f'momentum_{period}'].diff()
        
        # Moving averages
        for ma in [5, 10, 20, 50]:
            data[f'sma_{ma}'] = data['close'].rolling(ma).mean()
            data[f'price_vs_sma_{ma}'] = (data['close'] - data[f'sma_{ma}']) / data[f'sma_{ma}']
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_ma'] = data['rsi'].rolling(10).mean()
        
        # Bollinger Bands
        for window in [10, 20]:
            bb_mean = data['close'].rolling(window).mean()
            bb_std = data['close'].rolling(window).std()
            data[f'bb_upper_{window}'] = bb_mean + (bb_std * 2)
            data[f'bb_lower_{window}'] = bb_mean - (bb_std * 2)
            data[f'bb_position_{window}'] = (data['close'] - data[f'bb_lower_{window}']) / (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}'] + 1e-8)
            data[f'bb_width_{window}'] = (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}']) / bb_mean
        
        # Volume features
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['volume_momentum'] = data['volume'].pct_change(5)
        data['log_volume'] = np.log(data['volume'] + 1)
        
        # Advanced indicators
        data['price_acceleration'] = data['returns'].diff()
        data['volatility_ratio'] = data['vol_5'] / data['vol_20']
        
        # Market microstructure
        data['spread'] = (data['high'] - data['low']) / data['close']
        data['mid_price'] = (data['high'] + data['low']) / 2
        data['price_vs_mid'] = (data['close'] - data['mid_price']) / data['mid_price']
        
        # Trend indicators
        data['trend_strength'] = (data['sma_5'] - data['sma_20']) / data['sma_20']
        data['trend_consistency'] = data['returns'].rolling(10).apply(lambda x: np.sum(x > 0) / len(x))
        
        # Pattern recognition features
        data['higher_high'] = ((data['high'] > data['high'].shift(1)) & 
                              (data['high'].shift(1) > data['high'].shift(2))).astype(int)
        data['lower_low'] = ((data['low'] < data['low'].shift(1)) & 
                            (data['low'].shift(1) < data['low'].shift(2))).astype(int)
        
        # Time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        # Fill NaN
        data = data.fillna(method='ffill').fillna(0)
        
        return data
    
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for neural network training."""
        
        # Select features
        feature_cols = [col for col in data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(data[feature_cols])
        
        # Create target (profit in next 5-10 periods)
        future_returns = []
        for i in range(5, 11):
            future_returns.append(data['close'].pct_change(i).shift(-i))
        
        # Combined target: max profitable direction
        future_returns_array = np.array(future_returns)
        future_returns_df = pd.DataFrame(future_returns_array.T, index=data.index)
        
        # Consider transaction costs
        transaction_cost = 0.001  # 0.1% one-way
        
        # Profitable if return exceeds costs
        long_profitable = (future_returns_df > transaction_cost * 2).any(axis=1)
        short_profitable = (future_returns_df < -transaction_cost * 2).any(axis=1)
        
        # Target: 1 if any profitable opportunity exists
        targets = (long_profitable | short_profitable).astype(int).values
        
        # Create sequences
        sequences = []
        labels = []
        
        for i in range(self.sequence_length, len(scaled_features) - 10):
            sequences.append(scaled_features[i-self.sequence_length:i])
            labels.append(targets[i])
        
        return np.array(sequences), np.array(labels)
    
    def build_model(self, input_dim: int):
        """Build neural network model."""
        
        if self.model_type == 'lstm':
            self.model = AdvancedLSTM(
                input_dim=input_dim,
                hidden_dim=128,
                num_layers=3,
                dropout=0.3
            ).to(self.device)
        elif self.model_type == 'transformer':
            self.model = TransformerModel(
                input_dim=input_dim,
                d_model=128,
                nhead=8,
                num_layers=3,
                dropout=0.3
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Built {self.model_type} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 64):
        """Train neural network model."""
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        logger.info(f"Positive rate - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}")
        
        # Create datasets
        train_dataset = CryptoDataset(X_train, y_train)
        val_dataset = CryptoDataset(X_val, y_val)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Build model
        self.build_model(input_dim=X.shape[2])
        
        # Loss and optimizer
        pos_weight = torch.tensor([len(y_train) / (2 * y_train.sum())], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == batch_y).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / len(X_val)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'models/nn_{self.model_type}_best.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= 10:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2%}")
        
        # Load best model
        self.model.load_state_dict(torch.load(f'models/nn_{self.model_type}_best.pth'))
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor).squeeze().cpu().numpy()
            
        return predictions
    
    def save_model(self, path: str):
        """Save model and scaler."""
        import joblib
        
        model_dir = Path(path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), path)
        
        # Save scaler
        scaler_path = str(Path(path).parent / 'nn_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Saved model to {path}")
        logger.info(f"Saved scaler to {scaler_path}")


def main():
    """Train and evaluate neural network models."""
    
    logger.info("Starting neural network training for profitable trading")
    
    # Try both LSTM and Transformer
    results = {}
    
    for model_type in ['lstm', 'transformer']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type.upper()} model")
        logger.info(f"{'='*60}")
        
        # Initialize trader
        trader = NeuralNetworkTrader(sequence_length=60, model_type=model_type)
        
        # Load and prepare data
        X, y = trader.load_and_prepare_data()
        
        # Train model
        history = trader.train(X, y, epochs=50, batch_size=64)
        
        # Save model
        trader.save_model(f'models/nn_{model_type}_final.pth')
        
        # Store results
        results[model_type] = {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'history': history
        }
    
    # Print summary
    print("\n" + "="*80)
    print("🧠 ニューラルネットワーク訓練結果")
    print("="*80)
    
    for model_type, result in results.items():
        print(f"\n📊 {model_type.upper()}モデル:")
        print(f"  最終検証損失: {result['final_val_loss']:.4f}")
        print(f"  最終検証精度: {result['final_val_accuracy']:.2%}")
        print(f"  改善可能性: {'高' if result['final_val_accuracy'] > 0.55 else '中'}")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for i, (model_type, result) in enumerate(results.items()):
        history = result['history']
        
        axes[i].plot(history['train_loss'], label='Train Loss')
        axes[i].plot(history['val_loss'], label='Val Loss')
        axes[i].set_title(f'{model_type.upper()} Training History')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results/nn_training_history.png', dpi=300, bbox_inches='tight')
    
    print(f"\n💾 保存完了:")
    print(f"  モデル: models/nn_lstm_final.pth, models/nn_transformer_final.pth")
    print(f"  訓練履歴: backtest_results/nn_training_history.png")
    print(f"\n🚀 次のステップ: バックテストで実際の収益性を検証")
    
    return results


if __name__ == "__main__":
    main()