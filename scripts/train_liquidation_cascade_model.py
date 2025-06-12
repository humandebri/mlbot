#!/usr/bin/env python3
"""
Train a CatBoost model for liquidation cascade detection using historical data.

This script:
1. Loads historical OHLCV data
2. Simulates liquidation events based on price movements and leverage
3. Extracts features for cascade detection
4. Trains a CatBoost model
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import duckdb
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging
from src.ml_pipeline.model_trainer import ModelTrainer
from src.storage.duckdb_manager import DuckDBManager

setup_logging()
logger = get_logger(__name__)


class LiquidationDataGenerator:
    """Generate synthetic liquidation events from historical price data."""
    
    def __init__(self, leverage_distribution: Dict[float, float] = None):
        """
        Initialize liquidation data generator.
        
        Args:
            leverage_distribution: Distribution of leverage levels and their probabilities
        """
        self.leverage_distribution = leverage_distribution or {
            2: 0.1,   # 10% use 2x leverage
            5: 0.2,   # 20% use 5x leverage
            10: 0.3,  # 30% use 10x leverage
            20: 0.25, # 25% use 20x leverage
            50: 0.1,  # 10% use 50x leverage
            100: 0.05 # 5% use 100x leverage
        }
        
    def generate_liquidations(self, price_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate liquidation events from price data.
        
        Args:
            price_data: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            DataFrame with liquidation events
        """
        logger.info(f"Generating liquidation events for {symbol}")
        
        liquidations = []
        
        # Calculate price changes and volatility
        price_data = price_data.sort_values('timestamp')
        price_data['returns'] = price_data['close'].pct_change()
        price_data['volatility'] = price_data['returns'].rolling(window=60).std()
        price_data['high_low_pct'] = (price_data['high'] - price_data['low']) / price_data['close']
        
        # Identify potential liquidation points
        for idx in range(100, len(price_data) - 1):  # Start after some history
            row = price_data.iloc[idx]
            
            # Skip if missing data
            if pd.isna(row['returns']) or pd.isna(row['volatility']):
                continue
            
            # Simulate long liquidations (price drops)
            if row['returns'] < -0.001:  # Price drop > 0.1%
                # Calculate liquidations at different leverage levels
                for leverage, prob in self.leverage_distribution.items():
                    liquidation_threshold = 1 / leverage  # e.g., 10x leverage = 10% drop
                    
                    # Check if this drop would trigger liquidation
                    if abs(row['returns']) >= liquidation_threshold * 0.8:  # 80% of threshold
                        # Estimate liquidation size based on volatility and volume
                        size = np.random.exponential(
                            scale=float(row['volume']) * prob * abs(row['returns']) * 0.001
                        )
                        
                        if size > 0:
                            liquidations.append({
                                'timestamp': row['timestamp'],
                                'symbol': symbol,
                                'side': 'Buy',  # Long liquidation
                                'size': size,
                                'price': row['low'],  # Liquidation near low
                                'leverage': leverage,
                                'price_change': row['returns'],
                                'volatility': row['volatility'],
                                'volume': row['volume']
                            })
            
            # Simulate short liquidations (price rises)
            elif row['returns'] > 0.001:  # Price rise > 0.1%
                for leverage, prob in self.leverage_distribution.items():
                    liquidation_threshold = 1 / leverage
                    
                    if row['returns'] >= liquidation_threshold * 0.8:
                        size = np.random.exponential(
                            scale=float(row['volume']) * prob * row['returns'] * 0.001
                        )
                        
                        if size > 0:
                            liquidations.append({
                                'timestamp': row['timestamp'],
                                'symbol': symbol,
                                'side': 'Sell',  # Short liquidation
                                'size': size,
                                'price': row['high'],  # Liquidation near high
                                'leverage': leverage,
                                'price_change': row['returns'],
                                'volatility': row['volatility'],
                                'volume': row['volume']
                            })
        
        if not liquidations:
            logger.warning(f"No liquidations generated for {symbol}")
            return pd.DataFrame()
        
        liq_df = pd.DataFrame(liquidations)
        logger.info(f"Generated {len(liq_df)} liquidation events for {symbol}")
        
        return liq_df


class CascadeFeatureExtractor:
    """Extract features for liquidation cascade detection."""
    
    def __init__(self):
        self.window_sizes = [60, 300, 900, 1800]  # 1min, 5min, 15min, 30min
        
    def extract_features(self, liquidations: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from liquidation and price data.
        
        Args:
            liquidations: DataFrame with liquidation events
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with features
        """
        features_list = []
        
        # Sort by timestamp
        liquidations = liquidations.sort_values('timestamp')
        price_data = price_data.sort_values('timestamp')
        
        # Calculate volatility if not present
        if 'volatility' not in price_data.columns:
            price_data['returns'] = price_data['close'].pct_change()
            price_data['volatility'] = price_data['returns'].rolling(window=60).std()
            price_data['volatility'] = price_data['volatility'].fillna(0)
        
        # Add price features to liquidations
        liquidations = liquidations.merge(
            price_data[['timestamp', 'close', 'volume', 'volatility']].rename(
                columns={'close': 'current_price', 'volume': 'current_volume'}
            ),
            on='timestamp',
            how='left'
        )
        
        for idx in range(len(liquidations)):
            current_liq = liquidations.iloc[idx]
            current_time = current_liq['timestamp']
            
            features = {
                'timestamp': current_time,
                'symbol': current_liq['symbol']
            }
            
            # 1. Recent liquidation activity
            for window in self.window_sizes:
                window_start = current_time - pd.Timedelta(seconds=window)
                recent_liqs = liquidations[
                    (liquidations['timestamp'] >= window_start) & 
                    (liquidations['timestamp'] < current_time)
                ]
                
                # Basic statistics
                features[f'liq_count_{window}s'] = len(recent_liqs)
                features[f'liq_volume_{window}s'] = recent_liqs['size'].sum()
                
                if len(recent_liqs) > 0:
                    features[f'liq_avg_size_{window}s'] = recent_liqs['size'].mean()
                    features[f'liq_max_size_{window}s'] = recent_liqs['size'].max()
                    
                    # Directional analysis
                    buy_liqs = recent_liqs[recent_liqs['side'] == 'Buy']
                    sell_liqs = recent_liqs[recent_liqs['side'] == 'Sell']
                    
                    total_vol = recent_liqs['size'].sum()
                    features[f'sell_ratio_{window}s'] = sell_liqs['size'].sum() / total_vol if total_vol > 0 else 0.5
                    
                    # Cascade acceleration
                    if window >= 300 and len(recent_liqs) >= 5:
                        half_window = window // 2
                        mid_time = current_time - pd.Timedelta(seconds=half_window)
                        
                        recent_half = recent_liqs[recent_liqs['timestamp'] >= mid_time]
                        older_half = recent_liqs[recent_liqs['timestamp'] < mid_time]
                        
                        if len(recent_half) > 0 and len(older_half) > 0:
                            recent_rate = len(recent_half) / half_window
                            older_rate = len(older_half) / half_window
                            features[f'cascade_acceleration_{window}s'] = recent_rate / older_rate if older_rate > 0 else 1.0
                else:
                    features[f'liq_avg_size_{window}s'] = 0
                    features[f'liq_max_size_{window}s'] = 0
                    features[f'sell_ratio_{window}s'] = 0.5
                    if window >= 300:
                        features[f'cascade_acceleration_{window}s'] = 1.0
            
            # 2. Price features around liquidation
            price_window = price_data[
                (price_data['timestamp'] >= current_time - pd.Timedelta(minutes=30)) &
                (price_data['timestamp'] <= current_time)
            ]
            
            if len(price_window) > 1:
                # Price momentum
                features['price_change_5m'] = (
                    price_window.iloc[-1]['close'] - price_window.iloc[max(0, len(price_window)-6)]['close']
                ) / price_window.iloc[max(0, len(price_window)-6)]['close']
                
                features['price_change_15m'] = (
                    price_window.iloc[-1]['close'] - price_window.iloc[max(0, len(price_window)-16)]['close']
                ) / price_window.iloc[max(0, len(price_window)-16)]['close']
                
                # Volatility
                features['current_volatility'] = price_window['volatility'].iloc[-1] if 'volatility' in price_window else 0
                features['volatility_change'] = (
                    price_window['volatility'].iloc[-1] - price_window['volatility'].mean()
                ) / price_window['volatility'].mean() if price_window['volatility'].mean() > 0 else 0
                
                # Volume
                features['volume_ratio'] = (
                    price_window['volume'].iloc[-1] / price_window['volume'].mean()
                ) if price_window['volume'].mean() > 0 else 1.0
            else:
                features['price_change_5m'] = 0
                features['price_change_15m'] = 0
                features['current_volatility'] = 0
                features['volatility_change'] = 0
                features['volume_ratio'] = 1.0
            
            # 3. Liquidation clustering
            recent_times = liquidations[
                (liquidations['timestamp'] >= current_time - pd.Timedelta(minutes=5)) &
                (liquidations['timestamp'] < current_time)
            ]['timestamp']
            
            if len(recent_times) >= 2:
                time_diffs = recent_times.diff().dt.total_seconds().dropna()
                features['liq_clustering'] = time_diffs.std() / time_diffs.mean() if time_diffs.mean() > 0 else 0
            else:
                features['liq_clustering'] = 0
            
            # 4. Target: Cascade in next 60 seconds
            future_liqs = liquidations[
                (liquidations['timestamp'] > current_time) &
                (liquidations['timestamp'] <= current_time + pd.Timedelta(seconds=60))
            ]
            
            # Define cascade as significant liquidation activity
            features['cascade_occurred'] = 1 if (
                len(future_liqs) >= 5 or 
                future_liqs['size'].sum() > liquidations['size'].quantile(0.9) * 3
            ) else 0
            
            features['cascade_volume'] = future_liqs['size'].sum()
            features['cascade_count'] = len(future_liqs)
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)


def prepare_training_data(start_date: str = "2021-01-01", end_date: str = "2025-06-11") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare training data from historical market data.
    
    Returns:
        Tuple of (features, labels)
    """
    logger.info("Loading historical data from DuckDB")
    
    # Connect to DuckDB
    conn = duckdb.connect("data/historical_data.duckdb")
    
    # Load price data for all symbols
    all_features = []
    all_labels = []
    
    symbols = ["BTCUSDT", "ETHUSDT", "ICPUSDT"]
    
    for symbol in symbols:
        logger.info(f"Processing {symbol}")
        
        # Load OHLCV data
        query = f"""
        SELECT 
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM klines_{symbol.lower()}
        WHERE timestamp >= '{start_date}'
          AND timestamp <= '{end_date}'
        ORDER BY timestamp
        """
        
        price_data = conn.execute(query).df()
        
        if len(price_data) == 0:
            logger.warning(f"No data found for {symbol}")
            continue
        
        logger.info(f"Loaded {len(price_data)} price records for {symbol}")
        
        # Generate synthetic liquidations
        generator = LiquidationDataGenerator()
        liquidations = generator.generate_liquidations(price_data, symbol)
        
        if len(liquidations) == 0:
            continue
        
        # Extract features
        extractor = CascadeFeatureExtractor()
        features_df = extractor.extract_features(liquidations, price_data)
        
        if len(features_df) == 0:
            continue
        
        # Separate features and labels
        label_cols = ['cascade_occurred', 'cascade_volume', 'cascade_count']
        feature_cols = [col for col in features_df.columns if col not in label_cols + ['timestamp', 'symbol']]
        
        features = features_df[feature_cols]
        labels = features_df['cascade_occurred']  # Binary classification
        
        all_features.append(features)
        all_labels.append(labels)
        
        logger.info(f"Extracted {len(features)} samples for {symbol}")
    
    conn.close()
    
    if not all_features:
        raise ValueError("No features extracted from any symbol")
    
    # Combine all data
    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)
    
    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Feature count: {len(X.columns)}")
    logger.info(f"Positive samples: {y.sum()} ({y.mean()*100:.2f}%)")
    
    return X, y


def train_cascade_model(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Train CatBoost model for cascade detection.
    
    Returns:
        Training results
    """
    # Split data chronologically (80/20 split)
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Configure model for binary classification using regression
    config = {
        "objective": "RMSE",  # Using RMSE for binary targets
        "eval_metric": "RMSE",
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "bagging_temperature": 0.7,
        "border_count": 128,
        "early_stopping_rounds": 50,
        "use_best_model": True,
        "verbose": True,
        "random_seed": 42,
        "thread_count": -1,
        "task_type": "CPU",  # CPU or GPU
        "gpu_training": False,  # Use CPU for training
        "enable_optuna": True,
        "optuna_trials": 50,
        "optuna_direction": "minimize",  # Minimize RMSE
        "optuna_timeout": 3600,  # 1 hour timeout for optimization
        "model_version": f"cascade_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "save_model_format": ["cbm", "onnx"],
        "model_save_path": "models/cascade_detection",
        "categorical_features": [],  # No categorical features in our dataset
        "auto_class_weights": True  # Handle imbalanced classes
    }
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train model
    results = trainer.train_model(X_train, y_train, X_val, y_val)
    
    return results


def main():
    """Main training pipeline."""
    logger.info("Starting liquidation cascade model training")
    
    try:
        # Prepare data
        logger.info("Preparing training data...")
        X, y = prepare_training_data()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Train model
        logger.info("Training CatBoost model...")
        results = train_cascade_model(X, y)
        
        # Print results
        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        
        print("\nModel Performance:")
        if 'validation_scores' in results['performance']:
            val_scores = results['performance']['validation_scores']
            for metric, value in val_scores.items():
                print(f"  Validation {metric}: {value:.4f}")
            # Add cascade detection performance interpretation
            if 'rmse' in val_scores:
                print(f"\nNote: Using RMSE for binary cascade detection (threshold ~0.5)")
                print(f"Lower RMSE indicates better cascade prediction accuracy")
        
        print("\nTop 10 Features:")
        top_features = list(results['feature_analysis']['top_10_features'].items())
        for i, (feature, importance) in enumerate(top_features):
            print(f"  {i+1}. {feature}: {importance:.2f}")
        
        print(f"\nModel saved to: {results['model_artifacts']['catboost']}")
        
        if 'optimization' in results and results['optimization']:
            print(f"\nHyperparameter optimization:")
            print(f"  Best score: {results['optimization']['best_score']:.4f}")
            print(f"  Trials: {results['optimization']['n_trials']}")
        
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        logger.error("Training failed", exception=e)
        raise


if __name__ == "__main__":
    main()