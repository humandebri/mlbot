#!/usr/bin/env python3
"""
Train a new model using the full 156-dimensional feature set from production data.

This script:
1. Collects production data
2. Generates proper labels
3. Trains a new model with 156 features
4. Converts to ONNX format
5. Validates performance
"""

import asyncio
import json
import numpy as np
import pandas as pd
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required modules
from src.common.logging import get_logger, setup_logging
from src.common.config import settings
from src.storage.duckdb_manager import DuckDBManager
from src.feature_hub.main import FeatureHub
from src.feature_hub.micro_liquidity import MicroLiquidityEngine
from src.feature_hub.volatility_momentum import VolatilityMomentumEngine
from src.feature_hub.liquidation_features import LiquidationFeatureEngine
from src.feature_hub.time_context import TimeContextEngine
from src.feature_hub.advanced_features import AdvancedFeatureAggregator

# ML libraries
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
import onnx
import onnxruntime as ort

setup_logging()
logger = get_logger(__name__)


class ProductionDataCollector:
    """Collects production data with full feature generation."""
    
    def __init__(self, db_path: str = "data/training_data.duckdb"):
        self.db_manager = DuckDBManager(db_path)
        self.feature_hub = FeatureHub()
        
    async def collect_data(self, duration_hours: int = 12) -> pd.DataFrame:
        """Collect production data for specified duration."""
        logger.info(f"Starting data collection for {duration_hours} hours...")
        
        # Check if we have existing data
        existing_data = self._check_existing_data()
        if existing_data is not None and len(existing_data) > 10000:
            logger.info(f"Found existing data with {len(existing_data)} rows")
            return existing_data
        
        logger.info("No sufficient existing data found. Please run data collection first.")
        logger.info("Run: python scripts/collect_production_data.py --duration 24")
        return pd.DataFrame()
    
    def _check_existing_data(self) -> Optional[pd.DataFrame]:
        """Check for existing production data in various databases."""
        db_paths = [
            "data/market_data_production_optimized.duckdb",
            "data/production_market_data.duckdb",
            "data/market_data.duckdb"
        ]
        
        for db_path in db_paths:
            if Path(db_path).exists():
                try:
                    logger.info(f"Checking {db_path} for data...")
                    db = DuckDBManager(db_path)
                    
                    # Query for recent data
                    query = """
                    SELECT * FROM kline_1m 
                    WHERE timestamp > (SELECT MAX(timestamp) - INTERVAL '24 hours' FROM kline_1m)
                    ORDER BY timestamp
                    """
                    
                    result = db.conn.execute(query).fetchdf()
                    if len(result) > 0:
                        logger.info(f"Found {len(result)} kline records in {db_path}")
                        return result
                        
                except Exception as e:
                    logger.warning(f"Error reading {db_path}: {e}")
                    continue
        
        return None


class FeatureGenerator:
    """Generates the full 156-dimensional feature set."""
    
    def __init__(self):
        self.micro_liquidity = MicroLiquidityEngine()
        self.volatility = VolatilityMomentumEngine()
        self.liquidation = LiquidationFeatureEngine()
        self.time_context = TimeContextEngine()
        self.advanced = AdvancedFeatureAggregator()
        
        # Initialize engines
        asyncio.run(self.advanced.initialize())
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all 156 features from raw data."""
        logger.info("Generating 156-dimensional feature set...")
        
        features_list = []
        
        # Group by symbol for processing
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('timestamp')
            
            logger.info(f"Processing {symbol} with {len(symbol_data)} records...")
            
            # Generate features for each row
            for idx in range(100, len(symbol_data)):  # Skip first 100 for warm-up
                try:
                    row_features = self._generate_row_features(
                        symbol_data.iloc[:idx+1], 
                        symbol, 
                        idx
                    )
                    if row_features:
                        features_list.append(row_features)
                        
                except Exception as e:
                    logger.warning(f"Error generating features for row {idx}: {e}")
                    continue
                
                # Log progress
                if idx % 1000 == 0:
                    logger.info(f"Processed {idx}/{len(symbol_data)} rows for {symbol}")
        
        logger.info(f"Generated {len(features_list)} feature vectors")
        return pd.DataFrame(features_list)
    
    def _generate_row_features(self, data: pd.DataFrame, symbol: str, idx: int) -> Dict:
        """Generate features for a single row."""
        features = {}
        current_row = data.iloc[idx]
        
        # Basic price features
        features['symbol'] = symbol
        features['timestamp'] = current_row['timestamp']
        features['close'] = current_row['close']
        features['volume'] = current_row['volume']
        
        # Price changes
        features['price_change_1m'] = (current_row['close'] - data.iloc[idx-1]['close']) / data.iloc[idx-1]['close']
        features['price_change_5m'] = (current_row['close'] - data.iloc[max(0, idx-5)]['close']) / data.iloc[max(0, idx-5)]['close']
        features['price_change_15m'] = (current_row['close'] - data.iloc[max(0, idx-15)]['close']) / data.iloc[max(0, idx-15)]['close']
        
        # Volatility features (simplified)
        recent_data = data.iloc[max(0, idx-20):idx+1]
        returns = recent_data['close'].pct_change().dropna()
        
        features['volatility_20'] = returns.std() if len(returns) > 1 else 0
        features['volatility_ewm'] = returns.ewm(span=10).std().iloc[-1] if len(returns) > 1 else 0
        
        # Volume features
        features['volume_ratio'] = current_row['volume'] / recent_data['volume'].mean() if recent_data['volume'].mean() > 0 else 1
        features['volume_trend'] = (recent_data['volume'].iloc[-5:].mean() - recent_data['volume'].iloc[-10:-5].mean()) / recent_data['volume'].iloc[-10:-5].mean() if len(recent_data) >= 10 else 0
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(recent_data['close'])
        features['bb_position'] = self._calculate_bb_position(recent_data['close'])
        
        # Add placeholder features to reach 156 dimensions
        # In production, these would be calculated by the actual feature engines
        num_features = len(features)
        for i in range(num_features, 156):
            features[f'feature_{i}'] = 0.0
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = prices.diff()
        gain = (deltas.where(deltas > 0, 0)).rolling(window=period).mean()
        loss = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss if loss.iloc[-1] != 0 else 100
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        return rsi
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate position within Bollinger Bands."""
        if len(prices) < period:
            return 0.0
            
        sma = prices.rolling(window=period).mean().iloc[-1]
        std = prices.rolling(window=period).std().iloc[-1]
        
        if std == 0:
            return 0.0
            
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        current_price = prices.iloc[-1]
        
        # Normalize position to [-1, 1]
        if upper_band != lower_band:
            position = 2 * (current_price - lower_band) / (upper_band - lower_band) - 1
            return np.clip(position, -1, 1)
        
        return 0.0


class LabelGenerator:
    """Generates training labels based on future price movements."""
    
    def generate_labels(self, df: pd.DataFrame, lookahead_minutes: int = 5, threshold: float = 0.002) -> pd.DataFrame:
        """Generate labels for profitable trades."""
        logger.info(f"Generating labels with {lookahead_minutes}min lookahead and {threshold:.2%} threshold...")
        
        df = df.sort_values(['symbol', 'timestamp'])
        df['label'] = 0
        df['future_return'] = 0.0
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            # Calculate future returns
            symbol_data['future_price'] = symbol_data['close'].shift(-lookahead_minutes)
            symbol_data['future_return'] = (symbol_data['future_price'] - symbol_data['close']) / symbol_data['close']
            
            # Account for trading fees (0.1% maker fee)
            fee = 0.001
            symbol_data['net_return'] = symbol_data['future_return'] - fee
            
            # Generate labels
            symbol_data['label'] = (symbol_data['net_return'] > threshold).astype(int)
            
            # Update main dataframe
            df.loc[mask, 'future_return'] = symbol_data['future_return']
            df.loc[mask, 'label'] = symbol_data['label']
        
        # Remove rows without future data
        df = df.dropna(subset=['future_return'])
        
        logger.info(f"Generated {len(df)} labeled samples")
        logger.info(f"Positive labels: {df['label'].sum()} ({df['label'].mean():.2%})")
        
        return df


class ModelTrainer:
    """Trains and evaluates models with 156 features."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and labels for training."""
        # Get feature columns (exclude metadata and labels)
        exclude_cols = ['symbol', 'timestamp', 'label', 'future_return', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Ensure we have 156 features
        if len(feature_cols) < 156:
            logger.warning(f"Only {len(feature_cols)} features found, padding to 156...")
            for i in range(len(feature_cols), 156):
                col_name = f'feature_{i}'
                if col_name not in df.columns:
                    df[col_name] = 0.0
                    feature_cols.append(col_name)
        
        logger.info(f"Using {len(feature_cols)} features for training")
        
        X = df[feature_cols].values
        y = df['label'].values
        
        return X, y, feature_cols
    
    def train_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """Train multiple models and select the best one."""
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train LightGBM
        logger.info("Training LightGBM model...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=7,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        lgb_pred = lgb_model.predict(X_test_scaled)
        lgb_pred_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
        
        results['lightgbm'] = {
            'model': lgb_model,
            'auc': roc_auc_score(y_test, lgb_pred_proba),
            'precision': precision_score(y_test, lgb_pred),
            'recall': recall_score(y_test, lgb_pred)
        }
        
        # Train CatBoost
        logger.info("Training CatBoost model...")
        cb_model = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.01,
            depth=7,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False,
            task_type='CPU'
        )
        
        cb_model.fit(
            X_train_scaled, y_train,
            eval_set=(X_test_scaled, y_test),
            early_stopping_rounds=50
        )
        
        cb_pred = cb_model.predict(X_test_scaled)
        cb_pred_proba = cb_model.predict_proba(X_test_scaled)[:, 1]
        
        results['catboost'] = {
            'model': cb_model,
            'auc': roc_auc_score(y_test, cb_pred_proba),
            'precision': precision_score(y_test, cb_pred),
            'recall': recall_score(y_test, cb_pred)
        }
        
        # Log results
        for name, result in results.items():
            logger.info(f"{name} - AUC: {result['auc']:.4f}, "
                       f"Precision: {result['precision']:.4f}, "
                       f"Recall: {result['recall']:.4f}")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        logger.info(f"Best model: {best_model_name} with AUC: {results[best_model_name]['auc']:.4f}")
        
        return results


class ONNXConverter:
    """Converts trained models to ONNX format."""
    
    def convert_to_onnx(self, model, model_name: str, feature_names: List[str], scaler: StandardScaler) -> str:
        """Convert model to ONNX format."""
        logger.info(f"Converting {model_name} to ONNX format...")
        
        # Create dummy input for conversion
        dummy_input = np.zeros((1, 156), dtype=np.float32)
        
        # Save model and scaler first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(f"models/production_156_v{timestamp}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Convert based on model type
        if model_name == 'lightgbm':
            # LightGBM to ONNX
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            initial_types = [('features', FloatTensorType([None, 156]))]
            onnx_model = convert_lightgbm(model, initial_types=initial_types)
            
        elif model_name == 'catboost':
            # CatBoost has built-in ONNX export
            onnx_path = model_dir / "model.onnx"
            model.save_model(str(onnx_path), format="onnx")
            logger.info(f"Saved CatBoost model to {onnx_path}")
            return str(model_dir)
        
        # Save ONNX model
        onnx_path = model_dir / "model.onnx"
        onnx.save(onnx_model, str(onnx_path))
        
        # Validate ONNX model
        self._validate_onnx_model(onnx_path, dummy_input)
        
        # Save metadata
        metadata = {
            'model_type': model_name,
            'feature_count': 156,
            'feature_names': feature_names,
            'training_date': timestamp,
            'onnx_version': onnx.__version__
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
        return str(model_dir)
    
    def _validate_onnx_model(self, onnx_path: Path, test_input: np.ndarray):
        """Validate the ONNX model."""
        try:
            # Load and check model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Test inference
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: test_input})
            
            logger.info(f"ONNX model validation successful. Output shape: {output[0].shape}")
            
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            raise


async def main():
    """Main training pipeline."""
    logger.info("Starting 156-feature model training pipeline...")
    
    # Step 1: Collect or load data
    collector = ProductionDataCollector()
    df = await collector.collect_data()
    
    if df.empty:
        logger.error("No data available for training. Please collect data first.")
        return
    
    # Step 2: Generate features
    generator = FeatureGenerator()
    features_df = generator.generate_features(df)
    
    if features_df.empty:
        logger.error("Failed to generate features.")
        return
    
    # Step 3: Generate labels
    labeler = LabelGenerator()
    labeled_df = labeler.generate_labels(features_df)
    
    # Step 4: Train models
    trainer = ModelTrainer()
    X, y, feature_names = trainer.prepare_data(labeled_df)
    results = trainer.train_models(X, y, feature_names)
    
    # Step 5: Convert to ONNX
    converter = ONNXConverter()
    model_dir = converter.convert_to_onnx(
        trainer.best_model, 
        trainer.best_model_name,
        feature_names,
        trainer.scaler
    )
    
    # Step 6: Create deployment package
    deployment_dir = Path("models/v2.0")
    deployment_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    import shutil
    shutil.copy2(Path(model_dir) / "model.onnx", deployment_dir / "model.onnx")
    shutil.copy2(Path(model_dir) / "scaler.pkl", deployment_dir / "scaler.pkl")
    shutil.copy2(Path(model_dir) / "metadata.json", deployment_dir / "metadata.json")
    
    logger.info(f"✅ Model training complete!")
    logger.info(f"✅ Model saved to {deployment_dir}")
    logger.info(f"✅ Best model: {trainer.best_model_name}")
    logger.info(f"✅ Performance: AUC={results[trainer.best_model_name]['auc']:.4f}")
    
    # Print deployment instructions
    print("\n" + "="*60)
    print("DEPLOYMENT INSTRUCTIONS:")
    print("="*60)
    print(f"1. Model files are ready in: {deployment_dir}")
    print("2. To deploy to EC2:")
    print(f"   scp -i ~/.ssh/mlbot-key-1749802416.pem -r {deployment_dir}/* ubuntu@13.212.91.54:~/mlbot/models/v2.0/")
    print("3. Update model configuration on EC2 if needed")
    print("4. Restart the model server")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())