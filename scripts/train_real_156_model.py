#!/usr/bin/env python3
"""
Real 156-feature model training using actual market data
Based on the high-performance 44-feature model but extended properly
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from datetime import datetime, timedelta
import duckdb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import catboost as cb
import json
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealFeatureGenerator:
    """Generate 156 real market-based features"""
    
    def __init__(self):
        self.feature_names = []
        
    def calculate_returns(self, df):
        """Enhanced return features (20 features)"""
        features = {}
        
        # Basic returns
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['hl_ratio'] = (df['high'] - df['low']) / df['close']
        features['oc_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Multi-timeframe returns
        for period in [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60]:
            features[f'return_{period}'] = df['close'].pct_change(period)
            
        # Return volatility
        for window in [5, 10, 20, 30]:
            features[f'return_vol_{window}'] = features['returns'].rolling(window).std()
            
        return features
    
    def calculate_price_features(self, df):
        """Enhanced price-based features (30 features)"""
        features = {}
        
        # Moving averages
        for period in [3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
            sma = df['close'].rolling(period).mean()
            ema = df['close'].ewm(span=period).mean()
            features[f'price_vs_sma_{period}'] = (df['close'] - sma) / sma
            features[f'price_vs_ema_{period}'] = (df['close'] - ema) / ema
        
        # Price percentiles
        for period in [10, 20, 30, 50]:
            features[f'price_percentile_{period}'] = df['close'].rolling(period).rank() / period
            
        # Price momentum
        for period in [3, 5, 8, 10, 15, 20]:
            features[f'momentum_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
            
        return features
    
    def calculate_volume_features(self, df):
        """Enhanced volume features (25 features)"""
        features = {}
        
        # Basic volume metrics
        features['log_volume'] = np.log(df['volume'] + 1)
        features['volume_price_trend'] = df['volume'] * np.sign(features.get('returns', df['close'].pct_change()))
        
        # Volume ratios
        for period in [3, 5, 8, 10, 15, 20, 25, 30]:
            vol_ma = df['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / vol_ma
            
        # Volume moving averages
        for period in [5, 10, 20, 30, 50]:
            features[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            
        # Volume volatility
        for period in [5, 10, 20]:
            features[f'volume_vol_{period}'] = df['volume'].rolling(period).std()
            
        # VWAP features
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        for period in [10, 20, 30]:
            vwap = (typical_price * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
            features[f'vwap_ratio_{period}'] = df['close'] / vwap
            
        return features
    
    def calculate_volatility_features(self, df):
        """Enhanced volatility features (20 features)"""
        features = {}
        
        # True Range and ATR
        hl = df['high'] - df['low']
        hc = np.abs(df['high'] - df['close'].shift(1))
        lc = np.abs(df['low'] - df['close'].shift(1))
        true_range = np.maximum(hl, np.maximum(hc, lc))
        
        for period in [5, 10, 14, 20, 30]:
            features[f'atr_{period}'] = true_range.rolling(period).mean()
            features[f'atr_ratio_{period}'] = true_range / features[f'atr_{period}']
            
        # Volatility regimes
        for period in [10, 20, 30]:
            vol = df['close'].pct_change().rolling(period).std()
            vol_ma = vol.rolling(50).mean()
            features[f'vol_regime_{period}'] = vol / vol_ma
            
        # GARCH-like features
        returns = df['close'].pct_change()
        for lag in [1, 2, 3, 5]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
            
        return features
    
    def calculate_technical_indicators(self, df):
        """Technical indicators (25 features)"""
        features = {}
        
        # RSI variations
        for period in [9, 14, 21, 30]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
        # MACD variations
        for fast, slow in [(12, 26), (8, 17), (19, 39)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9).mean()
            features[f'macd_{fast}_{slow}'] = macd_line
            features[f'macd_signal_{fast}_{slow}'] = signal_line
            features[f'macd_hist_{fast}_{slow}'] = macd_line - signal_line
            
        # Bollinger Bands
        for period in [10, 20, 30]:
            for std_dev in [1.5, 2.0, 2.5]:
                sma = df['close'].rolling(period).mean()
                std = df['close'].rolling(period).std()
                upper = sma + (std * std_dev)
                lower = sma - (std * std_dev)
                features[f'bb_position_{period}_{std_dev}'] = (df['close'] - lower) / (upper - lower)
                features[f'bb_width_{period}_{std_dev}'] = (upper - lower) / sma
                
        return features
    
    def calculate_microstructure_features(self, df):
        """Market microstructure features (20 features)"""
        features = {}
        
        # Bid-ask spread proxies using high-low
        for period in [1, 3, 5, 10]:
            features[f'hl_spread_{period}'] = (df['high'] - df['low']).rolling(period).mean()
            features[f'hl_spread_ratio_{period}'] = features[f'hl_spread_{period}'] / df['close']
            
        # Price efficiency measures
        for period in [5, 10, 20]:
            # Variance ratio test approximation
            returns = df['close'].pct_change()
            var_1 = returns.rolling(period).var()
            var_k = returns.rolling(period * 2).var()
            features[f'variance_ratio_{period}'] = var_k / (var_1 * 2)
            
        # Amihud illiquidity ratio approximation
        for period in [5, 10, 20]:
            illiq = np.abs(df['close'].pct_change()) / (df['volume'] * df['close'])
            features[f'illiquidity_{period}'] = illiq.rolling(period).mean()
            
        # Market impact proxies
        for period in [3, 5, 10]:
            price_impact = np.abs(df['close'].pct_change()) / np.log(df['volume'] + 1)
            features[f'price_impact_{period}'] = price_impact.rolling(period).mean()
            
        return features
    
    def calculate_regime_features(self, df):
        """Market regime features (16 features)"""
        features = {}
        
        # Trend strength
        for period in [10, 20, 30, 50]:
            returns = df['close'].pct_change()
            trend_strength = returns.rolling(period).sum() / returns.rolling(period).std()
            features[f'trend_strength_{period}'] = trend_strength
            
        # Volatility regimes
        for period in [10, 20, 30]:
            vol = df['close'].pct_change().rolling(period).std()
            vol_threshold = vol.rolling(100).quantile(0.8)
            features[f'high_vol_regime_{period}'] = (vol > vol_threshold).astype(float)
            features[f'low_vol_regime_{period}'] = (vol < vol.rolling(100).quantile(0.2)).astype(float)
            
        # Trending vs ranging markets
        for period in [20, 30, 50]:
            price_range = df['high'].rolling(period).max() - df['low'].rolling(period).min()
            atr = (df['high'] - df['low']).rolling(period).mean()
            features[f'trending_market_{period}'] = (price_range / atr) > 1.5
            
        return features
    
    def calculate_time_features(self):
        """Time-based features (same as original)"""
        # This should be calculated from actual timestamps
        # For now, using placeholder
        features = {}
        features['hour_sin'] = 0.0  # Will be updated with real data
        features['hour_cos'] = 0.0  # Will be updated with real data
        features['is_weekend'] = 0.0  # Will be updated with real data
        return features
    
    def generate_all_features(self, df):
        """Generate all 156 features"""
        all_features = {}
        
        # Generate feature groups
        all_features.update(self.calculate_returns(df))
        all_features.update(self.calculate_price_features(df))
        all_features.update(self.calculate_volume_features(df))
        all_features.update(self.calculate_volatility_features(df))
        all_features.update(self.calculate_technical_indicators(df))
        all_features.update(self.calculate_microstructure_features(df))
        all_features.update(self.calculate_regime_features(df))
        all_features.update(self.calculate_time_features())
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(all_features, index=df.index)
        
        # Store feature names
        self.feature_names = list(feature_df.columns)
        logger.info(f"Generated {len(self.feature_names)} features")
        
        return feature_df

def load_market_data():
    """Load actual market data from DuckDB"""
    db_path = Path("data/historical_data.duckdb")
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return None
    
    try:
        conn = duckdb.connect(str(db_path))
        
        # Load recent market data
        query = """
        SELECT 
            timestamp,
            'BTCUSDT' as symbol,
            open,
            high,
            low,
            close,
            volume
        FROM klines_btcusdt 
        WHERE timestamp >= (current_timestamp - INTERVAL '90 days')
        UNION ALL
        SELECT 
            timestamp,
            'ETHUSDT' as symbol,
            open,
            high,
            low,
            close,
            volume
        FROM klines_ethusdt 
        WHERE timestamp >= (current_timestamp - INTERVAL '90 days')
        ORDER BY timestamp
        """
        
        df = conn.execute(query).fetchdf()
        conn.close()
        
        if df.empty:
            logger.error("No market data found")
            return None
            
        logger.info(f"Loaded {len(df)} market data points")
        return df
        
    except Exception as e:
        logger.error(f"Error loading market data: {e}")
        return None

def generate_labels(df, profit_threshold=0.003, horizons=[10, 15, 20]):
    """Generate profit labels with multiple horizons"""
    labels = []
    
    for horizon in horizons:
        # Calculate future returns
        future_returns = df['close'].pct_change(horizon).shift(-horizon)
        
        # Binary classification: profitable or not
        profit_labels = (future_returns > profit_threshold).astype(int)
        labels.append(profit_labels)
    
    # Combine labels (profitable in any horizon)
    combined_labels = pd.concat(labels, axis=1).max(axis=1)
    return combined_labels

def train_models(X, y):
    """Train multiple models and return the best one"""
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)  # Reduced for speed
    
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),
        'lightgbm': lgb.LGBMClassifier(
            objective='binary',
            n_estimators=100,  # Reduced for speed
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=42,
            verbose=-1
        ),
        'catboost': cb.CatBoostClassifier(
            iterations=100,  # Reduced for speed
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=False
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Remove rows with NaN
            valid_train = ~(X_train.isna().any(axis=1) | y_train.isna())
            valid_val = ~(X_val.isna().any(axis=1) | y_val.isna())
            
            X_train = X_train[valid_train]
            y_train = y_train[valid_train]
            X_val = X_val[valid_val]
            y_val = y_val[valid_val]
            
            if len(X_train) == 0 or len(X_val) == 0:
                continue
                
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            scores.append(score)
        
        if scores:
            results[name] = {
                'auc_mean': np.mean(scores),
                'auc_std': np.std(scores)
            }
            
            # Train final model on all data
            valid_all = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[valid_all]
            y_clean = y[valid_all]
            
            final_model = models[name]
            final_model.fit(X_clean, y_clean)
            trained_models[name] = final_model
            
            logger.info(f"{name}: AUC = {results[name]['auc_mean']:.4f} ± {results[name]['auc_std']:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc_mean'])
    best_model = trained_models[best_model_name]
    
    logger.info(f"Best model: {best_model_name} with AUC {results[best_model_name]['auc_mean']:.4f}")
    
    return best_model, best_model_name, results

def main():
    """Main training pipeline"""
    logger.info("Starting real 156-feature model training...")
    
    # Load market data
    market_data = load_market_data()
    if market_data is None:
        logger.error("Failed to load market data. Exiting.")
        return
    
    # Process each symbol separately and combine
    all_features = []
    all_labels = []
    
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        symbol_data = market_data[market_data['symbol'] == symbol].copy()
        symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'])
        symbol_data = symbol_data.sort_values('timestamp').reset_index(drop=True)
        
        if len(symbol_data) < 100:
            logger.warning(f"Insufficient data for {symbol}")
            continue
        
        # Generate features
        feature_generator = RealFeatureGenerator()
        features = feature_generator.generate_all_features(symbol_data)
        
        # Generate labels
        labels = generate_labels(symbol_data)
        
        # Align features and labels
        min_len = min(len(features), len(labels))
        features = features.iloc[:min_len]
        labels = labels.iloc[:min_len]
        
        all_features.append(features)
        all_labels.append(labels)
    
    if not all_features:
        logger.error("No features generated. Exiting.")
        return
    
    # Combine all features and labels
    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)
    
    # Remove rows with too many NaN values
    nan_threshold = 0.1  # Allow up to 10% NaN values
    valid_rows = X.isna().sum(axis=1) / len(X.columns) < nan_threshold
    X = X[valid_rows]
    y = y[valid_rows]
    
    # Fill remaining NaN values  
    X = X.ffill().fillna(0)
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Remove extreme outliers (beyond 99.9th percentile)
    for col in X.columns:
        if X[col].dtype in [np.float64, np.float32]:
            q99 = X[col].quantile(0.999)
            q01 = X[col].quantile(0.001)
            X[col] = X[col].clip(lower=q01, upper=q99)
    
    logger.info(f"Final dataset: {len(X)} samples, {len(X.columns)} features")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")
    
    if len(X) < 100:
        logger.error("Insufficient data for training")
        return
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Train models
    best_model, best_model_name, results = train_models(X_scaled, y)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"models/real_156_features_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and scaler
    joblib.dump(best_model, output_dir / "model.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    
    # Save metadata
    metadata = {
        "model_type": "real_156_features",
        "feature_count": len(X.columns),
        "feature_names": list(X.columns),
        "training_date": timestamp,
        "model_version": "4.0_real_features",
        "best_model": best_model_name,
        "performance": results,
        "config": {
            "profit_threshold": 0.003,
            "transaction_cost": 0.0008,
            "horizons": [10, 15, 20],
            "approach": "real_market_data_156_features"
        }
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {output_dir}")
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Performance: AUC = {results[best_model_name]['auc_mean']:.4f}")
    
    # Convert to ONNX
    try:
        import onnx
        import skl2onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        initial_type = [('float_input', FloatTensorType([None, len(X.columns)]))]
        onnx_model = convert_sklearn(best_model, initial_types=initial_type)
        
        with open(output_dir / "model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        logger.info("ONNX model saved successfully")
        
    except Exception as e:
        logger.warning(f"Could not convert to ONNX: {e}")
    
    return output_dir, best_model_name, results[best_model_name]['auc_mean']

if __name__ == "__main__":
    result = main()
    if result:
        output_dir, model_name, auc_score = result
        print(f"\n=== Training Complete ===")
        print(f"Model: {model_name}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Output: {output_dir}")
        print("========================")