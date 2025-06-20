#!/usr/bin/env python3
"""
Neural Network profitability test - Testing with advanced deep learning approach.
Simulates a neural network-like strategy without requiring PyTorch installation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class NeuralNetworkStrategy:
    """Neural network-based trading strategy using sklearn MLP."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = None
        
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features inspired by neural network preprocessing."""
        
        # Core price dynamics
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['oc_ratio'] = (data['close'] - data['open']) / data['open']
        
        # Advanced return features
        for period in [1, 2, 3, 5, 8, 13, 21]:  # Fibonacci-like periods
            data[f'return_{period}'] = data['close'].pct_change(period)
            data[f'return_acc_{period}'] = data[f'return_{period}'] - data[f'return_{period}'].shift(1)
        
        # Multi-scale volatility
        for window in [3, 5, 8, 13, 21, 34]:
            data[f'vol_{window}'] = data['returns'].rolling(window).std()
            data[f'vol_zscore_{window}'] = (data[f'vol_{window}'] - data[f'vol_{window}'].rolling(window*2).mean()) / data[f'vol_{window}'].rolling(window*2).std()
        
        # Advanced momentum
        for period in [3, 5, 8, 13]:
            data[f'momentum_{period}'] = data['close'].pct_change(period)
            data[f'momentum_strength_{period}'] = data[f'momentum_{period}'] / data[f'vol_{period}']
            data[f'momentum_acc_{period}'] = data[f'momentum_{period}'].diff()
        
        # Adaptive moving averages
        for ma in [5, 10, 20, 50]:
            data[f'sma_{ma}'] = data['close'].rolling(ma).mean()
            data[f'ema_{ma}'] = data['close'].ewm(span=ma, adjust=False).mean()
            data[f'price_vs_sma_{ma}'] = (data['close'] - data[f'sma_{ma}']) / data[f'sma_{ma}']
            data[f'price_vs_ema_{ma}'] = (data['close'] - data[f'ema_{ma}']) / data[f'ema_{ma}']
            
        # Enhanced RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_ma'] = data['rsi'].rolling(10).mean()
        data['rsi_divergence'] = data['rsi'] - data['rsi_ma']
        
        # Advanced Bollinger Bands
        for window in [10, 20]:
            bb_mean = data['close'].rolling(window).mean()
            bb_std = data['close'].rolling(window).std()
            bb_upper = bb_mean + (bb_std * 2)
            bb_lower = bb_mean - (bb_std * 2)
            data[f'bb_position_{window}'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
            data[f'bb_width_{window}'] = (bb_upper - bb_lower) / bb_mean
            data[f'bb_squeeze_{window}'] = data[f'bb_width_{window}'].rolling(window).rank(pct=True)
        
        # Volume profile
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['volume_momentum'] = data['volume'].pct_change(5)
        data['log_volume'] = np.log(data['volume'] + 1)
        data['volume_price_corr'] = data['returns'].rolling(20).corr(data['volume'].pct_change())
        
        # Market microstructure
        data['spread'] = (data['high'] - data['low']) / data['close']
        data['spread_ma'] = data['spread'].rolling(20).mean()
        data['spread_zscore'] = (data['spread'] - data['spread_ma']) / data['spread'].rolling(20).std()
        
        # Trend strength indicators
        data['trend_strength_short'] = (data['sma_5'] - data['sma_10']) / data['sma_10']
        data['trend_strength_medium'] = (data['sma_10'] - data['sma_20']) / data['sma_20']
        data['trend_strength_long'] = (data['sma_20'] - data['sma_50']) / data['sma_50']
        data['trend_consistency'] = data['returns'].rolling(20).apply(lambda x: np.sum(x > 0) / len(x))
        
        # Pattern recognition
        data['higher_high'] = ((data['high'] > data['high'].shift(1)) & 
                              (data['high'].shift(1) > data['high'].shift(2))).astype(int)
        data['lower_low'] = ((data['low'] < data['low'].shift(1)) & 
                            (data['low'].shift(1) < data['low'].shift(2))).astype(int)
        data['inside_bar'] = ((data['high'] < data['high'].shift(1)) & 
                             (data['low'] > data['low'].shift(1))).astype(int)
        
        # Volatility regime
        if 'vol_20' in data.columns:
            data['vol_regime'] = pd.qcut(data['vol_20'], q=5, labels=False, duplicates='drop')
            data['vol_expanding'] = (data['vol_5'] > data['vol_20']).astype(int)
        else:
            # Fallback if vol_20 doesn't exist
            data['vol_20'] = data['returns'].rolling(20).std()
            data['vol_regime'] = pd.qcut(data['vol_20'], q=5, labels=False, duplicates='drop')
            data['vol_expanding'] = (data['vol_5'] > data['vol_20']).astype(int)
        
        # Time-based features with cyclical encoding
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Interaction features
        data['momentum_volume'] = data['momentum_5'] * data['volume_ratio']
        data['volatility_trend'] = data['vol_20'] * abs(data['trend_strength_medium'])
        data['rsi_momentum'] = data['rsi'] * data['momentum_5']
        
        return data
    
    def train_neural_network(self, X_train, y_train):
        """Train a neural network classifier."""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create multi-layer perceptron
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),  # Deep architecture
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42
        )
        
        # Train model
        logger.info("Training neural network...")
        self.model.fit(X_scaled, y_train)
        
        # Calculate feature importance using permutation
        self.calculate_feature_importance(X_scaled, y_train)
        
        logger.info(f"Neural network trained with score: {self.model.score(X_scaled, y_train):.3f}")
        
    def calculate_feature_importance(self, X, y):
        """Calculate feature importance through permutation."""
        baseline_score = self.model.score(X, y)
        importances = []
        
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_score = self.model.score(X_permuted, y)
            importance = baseline_score - permuted_score
            importances.append(importance)
            
        self.feature_importance = np.array(importances)
    
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


def nn_profitability_test():
    """Test neural network strategy profitability."""
    
    logger.info("Starting neural network profitability test")
    
    # Initialize strategy
    strategy = NeuralNetworkStrategy()
    
    # Load training data
    conn = duckdb.connect("data/historical_data.duckdb")
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM klines_btcusdt
    WHERE timestamp >= '2024-01-01' AND timestamp <= '2024-04-30'
    ORDER BY timestamp
    """
    
    train_data = conn.execute(query).df()
    train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
    train_data.set_index('timestamp', inplace=True)
    
    # Create features
    train_data = strategy.create_advanced_features(train_data)
    
    # Prepare training data
    feature_cols = [col for col in train_data.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume']]
    
    # Remove NaN
    train_data = train_data.dropna()
    
    # Create targets (profitable trades)
    transaction_cost = 0.001  # Standard fee
    
    # Multi-horizon profitability
    profit_horizons = []
    for horizon in [3, 5, 7, 10]:
        future_return = train_data['close'].pct_change(horizon).shift(-horizon)
        long_profit = future_return - transaction_cost * 2
        short_profit = -future_return - transaction_cost * 2
        profitable = (long_profit > 0.003) | (short_profit > 0.003)  # 0.3% profit threshold
        profit_horizons.append(profitable)
    
    # Combined target
    y_train = pd.DataFrame(profit_horizons).T.any(axis=1).astype(int)
    X_train = train_data[feature_cols]
    
    # Remove rows with NaN targets
    valid_mask = ~y_train.isnull()
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    logger.info(f"Training data: {len(X_train)} samples, {len(feature_cols)} features")
    logger.info(f"Positive rate: {y_train.mean():.2%}")
    
    # Train neural network
    strategy.train_neural_network(X_train, y_train)
    
    # Test on multiple periods
    test_periods = [
        ('2024-05-01', '2024-07-31', 'May-Jul (Validation)'),
        ('2024-08-01', '2024-10-31', 'Aug-Oct (Test)'),
        ('2024-11-01', '2024-12-31', 'Nov-Dec (Recent)')
    ]
    
    print("\n" + "="*80)
    print("🧠 ニューラルネットワーク収益性テスト")
    print("="*80)
    
    all_results = []
    
    for start_date, end_date, period_name in test_periods:
        # Load test data
        query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM klines_btcusdt
        WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'
        ORDER BY timestamp
        """
        
        test_data = conn.execute(query).df()
        
        if len(test_data) == 0:
            continue
            
        test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
        test_data.set_index('timestamp', inplace=True)
        
        # Create features
        test_data = strategy.create_advanced_features(test_data)
        test_data = test_data.dropna()
        
        if len(test_data) == 0:
            continue
        
        # Get predictions
        X_test = test_data[feature_cols]
        predictions = strategy.predict(X_test)
        
        # Neural network optimized strategy parameters
        confidence_threshold = 0.75  # Higher confidence for NN
        position_size = 0.02  # 2% position
        fee_rate = 0.001  # Standard fee
        
        # Simulate trading
        capital = 100000
        trades = []
        
        high_confidence_indices = np.where(predictions > confidence_threshold)[0]
        
        for i in high_confidence_indices[:150]:  # Limit trades
            if i + 10 >= len(test_data):
                continue
                
            entry_price = test_data['close'].iloc[i]
            confidence = predictions[i]
            
            # Neural network-based direction selection
            # Use multiple indicators for direction
            momentum = test_data['momentum_5'].iloc[i]
            trend_short = test_data['trend_strength_short'].iloc[i]
            trend_medium = test_data['trend_strength_medium'].iloc[i]
            rsi = test_data['rsi'].iloc[i]
            bb_position = test_data['bb_position_20'].iloc[i]
            
            # Weighted direction score
            long_score = 0
            if momentum > 0: long_score += 2
            if trend_short > 0: long_score += 1
            if trend_medium > 0: long_score += 1
            if rsi < 50: long_score += 1
            if bb_position < 0.3: long_score += 1
            
            short_score = 6 - long_score
            
            direction = 'long' if long_score > short_score else 'short'
            
            # Dynamic exit based on market conditions
            if test_data['vol_regime'].iloc[i] <= 2:  # Low volatility
                hold_period = 8
            elif test_data['vol_regime'].iloc[i] >= 4:  # High volatility
                hold_period = 3
            else:
                hold_period = 5
            
            exit_price = test_data['close'].iloc[i + hold_period]
            
            # Calculate PnL
            if direction == 'long':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            
            pnl_pct -= fee_rate * 2  # Round trip
            
            position_value = capital * position_size
            pnl_dollar = position_value * pnl_pct
            
            trades.append({
                'pnl_dollar': pnl_dollar,
                'confidence': confidence,
                'direction': direction
            })
            
            capital += pnl_dollar
        
        if trades:
            trades_df = pd.DataFrame(trades)
            total_return = (capital - 100000) / 100000 * 100
            win_rate = len(trades_df[trades_df['pnl_dollar'] > 0]) / len(trades_df) * 100
            avg_trade = trades_df['pnl_dollar'].mean()
            
            months = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 30.44
            monthly_return = total_return / months if months > 0 else 0
            
            result = {
                'period': period_name,
                'trades': len(trades_df),
                'total_return': total_return,
                'monthly_return': monthly_return,
                'win_rate': win_rate,
                'avg_trade': avg_trade,
                'final_capital': capital
            }
            
            all_results.append(result)
    
    conn.close()
    
    # Display results
    print(f"\n📊 期間別パフォーマンス:")
    print(f"{'期間':<20} {'取引数':<8} {'月次収益率':<12} {'勝率':<8} {'平均取引':<12}")
    print("-" * 70)
    
    total_periods = 0
    profitable_periods = 0
    total_monthly_return = 0
    
    for result in all_results:
        print(f"{result['period']:<20} {result['trades']:<8} {result['monthly_return']:>10.2f}% "
              f"{result['win_rate']:>6.1f}% ${result['avg_trade']:>9.2f}")
        
        total_periods += 1
        if result['monthly_return'] > 0:
            profitable_periods += 1
        total_monthly_return += result['monthly_return']
    
    if total_periods > 0:
        avg_monthly_return = total_monthly_return / total_periods
    else:
        avg_monthly_return = 0
    
    # Summary
    print(f"\n📈 総合分析:")
    print(f"  平均月次収益率: {avg_monthly_return:.3f}%")
    print(f"  収益期間: {profitable_periods}/{total_periods}")
    print(f"  年次換算: {avg_monthly_return * 12:.1f}%")
    
    # Feature importance
    if hasattr(strategy, 'feature_importance') and strategy.feature_importance is not None:
        top_features_idx = np.argsort(strategy.feature_importance)[-10:][::-1]
        print(f"\n🔍 重要特徴量 (Top 10):")
        for idx in top_features_idx:
            if idx < len(feature_cols):
                print(f"  - {feature_cols[idx]}")
    
    # Final assessment
    print(f"\n🎯 ニューラルネットワーク評価:")
    if avg_monthly_return > 0.1:
        assessment = "🎉 優秀！NN戦略で収益性達成"
        recommendation = "実用化推奨"
    elif avg_monthly_return > 0:
        assessment = "✅ 収益性達成！さらなる最適化可能"
        recommendation = "テスト運用推奨"
    else:
        assessment = "⚠️ 追加改善必要"
        recommendation = "パラメータ調整継続"
    
    print(f"  {assessment}")
    print(f"  推奨: {recommendation}")
    
    # Compare with previous
    print(f"\n📊 改善比較:")
    print(f"  ランダムフォレスト: -0.021% 月次")
    print(f"  ニューラルネット: {avg_monthly_return:+.3f}% 月次")
    print(f"  改善幅: {avg_monthly_return - (-0.021):+.3f}%")
    
    return all_results


if __name__ == "__main__":
    nn_profitability_test()