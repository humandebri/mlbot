#!/usr/bin/env python3
"""
Fast and efficient neural network model for cryptocurrency trading.
Optimized for speed and practical results.
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
from sklearn.metrics import accuracy_score, roc_auc_score
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
    """Fast dataset for cryptocurrency features."""
    
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FastNN(nn.Module):
    """Fast and efficient neural network for trading signals."""
    
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(FastNN, self).__init__()
        
        # Simple but effective architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)


class FastNeuralTrader:
    """Fast neural network trading system."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.device = device
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare training data efficiently."""
        logger.info("Loading data for fast neural network training")
        
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
        
        # Engineer essential features
        data = self.engineer_features(data)
        
        # Create targets and features
        X, y = self.create_features_and_targets(data)
        
        logger.info(f"Created {len(X)} samples with {X.shape[1]} features")
        return X, y
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create essential features quickly."""
        
        # Basic price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['oc_ratio'] = (data['close'] - data['open']) / data['open']
        
        # Multi-timeframe returns
        for period in [1, 3, 5, 10]:
            data[f'return_{period}'] = data['close'].pct_change(period)
        
        # Simple volatility
        for window in [5, 10, 20]:
            data[f'vol_{window}'] = data['returns'].rolling(window).std()
        
        # Moving averages
        for ma in [5, 10, 20]:
            data[f'sma_{ma}'] = data['close'].rolling(ma).mean()
            data[f'price_vs_sma_{ma}'] = (data['close'] - data[f'sma_{ma}']) / data[f'sma_{ma}']
        
        # RSI (simplified)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_middle = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # Volume features
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Momentum indicators
        data['momentum_3'] = data['close'].pct_change(3)
        data['momentum_5'] = data['close'].pct_change(5)
        
        # Simple trend
        data['trend_strength'] = (data['sma_5'] - data['sma_20']) / data['sma_20']
        
        # Time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        
        # Fill NaN
        data = data.fillna(method='ffill').fillna(0)
        
        return data
    
    def create_features_and_targets(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create features and targets for neural network."""
        
        # Select features
        feature_cols = [col for col in data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(data[feature_cols])
        
        # Create profitable trading target (simplified)
        future_returns_5 = data['close'].pct_change(5).shift(-5)
        future_returns_10 = data['close'].pct_change(10).shift(-10)
        
        # Consider transaction costs
        transaction_cost = 0.001  # 0.1% one-way
        min_profit_threshold = transaction_cost * 2.5  # 0.25% minimum profit
        
        # Target: 1 if profitable opportunity exists
        profitable_long = (future_returns_5 > min_profit_threshold) | (future_returns_10 > min_profit_threshold)
        profitable_short = (future_returns_5 < -min_profit_threshold) | (future_returns_10 < -min_profit_threshold)
        
        targets = (profitable_long | profitable_short).astype(int).values
        
        # Remove NaN rows
        valid_indices = ~np.isnan(targets)
        X = scaled_features[valid_indices]
        y = targets[valid_indices]
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, batch_size: int = 128):
        """Train neural network quickly."""
        
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
        self.model = FastNN(input_dim=X.shape[1], hidden_dim=64, dropout=0.3).to(self.device)
        logger.info(f"Built FastNN model with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Loss and optimizer
        pos_weight = torch.tensor([len(y_train) / (2 * y_train.sum())], dtype=torch.float32).to(self.device)
        criterion = nn.BCELoss()  # Use BCELoss since model already has sigmoid
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_auc': []}
        
        # Training loop
        best_val_auc = 0
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
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            
            val_accuracy = accuracy_score(val_targets, val_predictions > 0.5)
            val_auc = roc_auc_score(val_targets, val_predictions)
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_auc'].append(val_auc)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping based on AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/fast_nn_best.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= 7:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.3f}, Val AUC: {val_auc:.3f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('models/fast_nn_best.pth'))
        
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
        scaler_path = str(Path(path).parent / 'fast_nn_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Saved model to {path}")


def main():
    """Train and evaluate fast neural network model."""
    
    logger.info("Starting fast neural network training")
    
    # Initialize trader
    trader = FastNeuralTrader()
    
    # Load and prepare data
    X, y = trader.load_and_prepare_data()
    
    # Train model
    history = trader.train(X, y, epochs=25, batch_size=128)
    
    # Save model
    trader.save_model('models/fast_nn_final.pth')
    
    # Print results
    final_auc = history['val_auc'][-1]
    final_accuracy = history['val_accuracy'][-1]
    best_auc = max(history['val_auc'])
    
    print("\n" + "="*60)
    print("🚀 高速ニューラルネットワーク訓練完了")
    print("="*60)
    print(f"📊 最終検証AUC: {final_auc:.3f}")
    print(f"📊 最終検証精度: {final_accuracy:.2%}")
    print(f"🏆 最高AUC: {best_auc:.3f}")
    print(f"⚡ パラメータ数: {sum(p.numel() for p in trader.model.parameters()):,}")
    
    # Assessment
    if best_auc > 0.65:
        assessment = "🎉 優秀な性能！"
    elif best_auc > 0.60:
        assessment = "✅ 良好な性能"
    elif best_auc > 0.55:
        assessment = "⚠️ 改善可能"
    else:
        assessment = "❌ さらなる改善が必要"
    
    print(f"\n🎯 評価: {assessment}")
    print(f"💾 モデル保存: models/fast_nn_final.pth")
    print(f"\n🚀 次のステップ: バックテストで収益性検証")
    
    return trader, history


if __name__ == "__main__":
    main()