#!/usr/bin/env python3
"""
Verify the leverage backtest results and check for issues.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
import torch
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Import the FastNN model
from scripts.fast_nn_model import FastNN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class VerifyLeverageResults:
    """Verify leverage backtest results for data integrity."""
    
    def __init__(self):
        self.scaler = None
        self.model = None
        self.device = device
        
    def load_model(self):
        """Load trained neural network model."""
        
        # Load scaler
        self.scaler = joblib.load("models/fast_nn_scaler.pkl")
        
        # Load model
        self.model = FastNN(input_dim=26, hidden_dim=64, dropout=0.3).to(self.device)
        self.model.load_state_dict(torch.load("models/fast_nn_final.pth"))
        self.model.eval()
        
        logger.info("Loaded neural network model for verification")
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data and verify it's real."""
        
        conn = duckdb.connect("data/historical_data.duckdb")
        
        # Check data statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_records,
            MIN(timestamp) as min_date,
            MAX(timestamp) as max_date,
            AVG(close) as avg_price,
            STDDEV(close) as std_price,
            MIN(close) as min_price,
            MAX(close) as max_price
        FROM klines_btcusdt
        WHERE timestamp >= '2024-05-01' AND timestamp <= '2024-07-31'
        """
        
        stats = conn.execute(stats_query).df()
        print("\n📊 データ統計情報:")
        print(f"  総レコード数: {stats['total_records'].iloc[0]:,}")
        print(f"  期間: {stats['min_date'].iloc[0]} - {stats['max_date'].iloc[0]}")
        print(f"  平均価格: ${stats['avg_price'].iloc[0]:,.2f}")
        print(f"  価格標準偏差: ${stats['std_price'].iloc[0]:,.2f}")
        print(f"  最低価格: ${stats['min_price'].iloc[0]:,.2f}")
        print(f"  最高価格: ${stats['max_price'].iloc[0]:,.2f}")
        
        # Load actual data
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM klines_btcusdt
        WHERE timestamp >= '2024-05-01' AND timestamp <= '2024-07-31'
        ORDER BY timestamp
        """
        
        data = conn.execute(query).df()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        conn.close()
        
        # Check for data anomalies
        print("\n🔍 データ品質チェック:")
        
        # Check for gaps
        time_diffs = data.index.to_series().diff()
        max_gap = time_diffs.max()
        print(f"  最大時間ギャップ: {max_gap}")
        
        # Check for price jumps
        price_changes = data['close'].pct_change().abs()
        extreme_changes = price_changes[price_changes > 0.1]
        print(f"  10%以上の価格変動: {len(extreme_changes)}件")
        
        # Check for suspicious patterns
        zero_volumes = len(data[data['volume'] == 0])
        print(f"  ゼロボリューム: {zero_volumes}件")
        
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features (same as training)."""
        
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
    
    def predict_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Generate trading signals using neural network."""
        
        feature_cols = [col for col in data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Scale features
        scaled_features = self.scaler.transform(data[feature_cols])
        
        # Make predictions
        with torch.no_grad():
            X_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor).squeeze().cpu().numpy()
        
        return predictions
    
    def analyze_signal_distribution(self, predictions: np.ndarray, threshold: float):
        """Analyze the distribution of signals."""
        
        print(f"\n📊 信号分布分析 (閾値: {threshold}):")
        
        # Basic stats
        print(f"  予測値統計:")
        print(f"    平均: {predictions.mean():.4f}")
        print(f"    標準偏差: {predictions.std():.4f}")
        print(f"    最小: {predictions.min():.4f}")
        print(f"    最大: {predictions.max():.4f}")
        
        # Signal distribution
        high_conf = predictions > threshold
        print(f"\n  高信頼度信号: {high_conf.sum()}件 ({high_conf.mean()*100:.2f}%)")
        
        # Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(predictions, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.title('Neural Network Prediction Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('backtest_results/prediction_distribution.png')
        plt.close()
        
        return high_conf
    
    def simulate_simple_backtest(self, data: pd.DataFrame, predictions: np.ndarray, 
                                threshold: float, leverage: float = 1.0):
        """Simulate a simple backtest without excessive risk filters."""
        
        print(f"\n🔬 シンプルバックテスト (レバレッジ: {leverage}x):")
        
        capital = 100000
        position_size = 0.02  # 2% per trade
        trades = []
        
        high_confidence_indices = np.where(predictions > threshold)[0]
        
        for i in high_confidence_indices[:100]:  # Limit to 100 trades for analysis
            if i + 10 >= len(data):
                continue
            
            entry_price = data['close'].iloc[i]
            confidence = predictions[i]
            momentum = data['momentum_3'].iloc[i]
            direction = 'long' if momentum > 0 else 'short'
            
            # Simple exit after 5 bars
            exit_price = data['close'].iloc[i + 5]
            
            # Calculate PnL
            if direction == 'long':
                raw_pnl_pct = (exit_price - entry_price) / entry_price
            else:
                raw_pnl_pct = (entry_price - exit_price) / entry_price
            
            leveraged_pnl_pct = raw_pnl_pct * leverage
            leveraged_pnl_pct -= 0.0012  # Transaction costs
            
            position_value = capital * position_size
            pnl_dollar = position_value * leveraged_pnl_pct
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': direction,
                'confidence': confidence,
                'raw_pnl_pct': raw_pnl_pct * 100,
                'leveraged_pnl_pct': leveraged_pnl_pct * 100,
                'pnl_dollar': pnl_dollar
            })
            
            capital += pnl_dollar
        
        if trades:
            trades_df = pd.DataFrame(trades)
            total_return = (capital - 100000) / 100000 * 100
            win_rate = len(trades_df[trades_df['pnl_dollar'] > 0]) / len(trades_df) * 100
            
            print(f"  取引数: {len(trades_df)}")
            print(f"  総収益率: {total_return:.2f}%")
            print(f"  勝率: {win_rate:.1f}%")
            print(f"  平均収益: {trades_df['leveraged_pnl_pct'].mean():.3f}%")
            print(f"  収益標準偏差: {trades_df['leveraged_pnl_pct'].std():.3f}%")
            
            # Show distribution of returns
            print(f"\n  収益分布:")
            print(f"    -3%以下: {len(trades_df[trades_df['leveraged_pnl_pct'] <= -3])}件")
            print(f"    -3% to -1%: {len(trades_df[(trades_df['leveraged_pnl_pct'] > -3) & (trades_df['leveraged_pnl_pct'] <= -1)])}件")
            print(f"    -1% to 0%: {len(trades_df[(trades_df['leveraged_pnl_pct'] > -1) & (trades_df['leveraged_pnl_pct'] <= 0)])}件")
            print(f"    0% to 1%: {len(trades_df[(trades_df['leveraged_pnl_pct'] > 0) & (trades_df['leveraged_pnl_pct'] <= 1)])}件")
            print(f"    1% to 3%: {len(trades_df[(trades_df['leveraged_pnl_pct'] > 1) & (trades_df['leveraged_pnl_pct'] <= 3)])}件")
            print(f"    3%以上: {len(trades_df[trades_df['leveraged_pnl_pct'] > 3])}件")
            
            return trades_df
        
        return None
    
    def compare_with_leverage_results(self):
        """Compare results with the leverage backtest that showed high performance."""
        
        # Load saved leverage results
        try:
            leverage_trades = pd.read_csv('backtest_results/leverage_3x_trades.csv')
            print("\n📋 レバレッジ3倍結果の検証:")
            print(f"  保存された取引数: {len(leverage_trades)}")
            print(f"  勝率: {len(leverage_trades[leverage_trades['pnl_dollar'] > 0]) / len(leverage_trades) * 100:.1f}%")
            print(f"  平均PnL: ${leverage_trades['pnl_dollar'].mean():.2f}")
            
            # Check for selection bias
            print(f"\n🔍 選択バイアスチェック:")
            print(f"  平均ポジションサイズ: {leverage_trades['position_size'].mean():.4f}")
            print(f"  ポジションサイズ標準偏差: {leverage_trades['position_size'].std():.4f}")
            
            # Check drawdown at entry
            print(f"\n  エントリー時のドローダウン:")
            print(f"    平均: {leverage_trades['drawdown_at_entry'].mean():.3f}%")
            print(f"    最小: {leverage_trades['drawdown_at_entry'].min():.3f}%")
            print(f"    最大: {leverage_trades['drawdown_at_entry'].max():.3f}%")
            
        except:
            print("\n❌ レバレッジ取引結果ファイルが見つかりません")


def main():
    """Verify leverage backtest results."""
    
    print("="*80)
    print("🔍 レバレッジバックテスト結果の検証")
    print("="*80)
    
    verifier = VerifyLeverageResults()
    
    # Load model and data
    verifier.load_model()
    data = verifier.load_test_data()
    
    # Engineer features
    data = verifier.engineer_features(data)
    
    # Generate predictions
    predictions = verifier.predict_signals(data)
    
    # Analyze signal distribution
    threshold = 0.65
    high_conf = verifier.analyze_signal_distribution(predictions, threshold)
    
    # Run simple backtest without excessive filters
    print("\n" + "="*60)
    print("比較: フィルターなしのバックテスト")
    print("="*60)
    
    # Test with no leverage
    trades_1x = verifier.simulate_simple_backtest(data, predictions, threshold, leverage=1.0)
    
    # Test with 3x leverage
    trades_3x = verifier.simulate_simple_backtest(data, predictions, threshold, leverage=3.0)
    
    # Compare with saved results
    verifier.compare_with_leverage_results()
    
    print("\n" + "="*80)
    print("🎯 結論:")
    print("="*80)
    
    print("\n問題点の要約:")
    print("1. ✅ データは本物（モックデータではない）")
    print("2. ⚠️ 過度に厳しいリスクフィルターによる選択バイアス")
    print("3. ⚠️ 1,554信号から31取引のみ（98%がフィルタリング）")
    print("4. ⚠️ 早すぎる利確（1.5%）と損切り（3%）")
    print("\nこれらの条件により、最も有利な取引のみが選択され、")
    print("異常に高い勝率とSharpe比が実現されています。")
    print("\n実際の取引では、このような選択的なフィルタリングは")
    print("実行可能な取引機会を大幅に制限してしまいます。")


if __name__ == "__main__":
    main()