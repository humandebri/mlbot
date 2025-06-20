#!/usr/bin/env python3
"""
Extended backtest with longer period for comprehensive profitability analysis.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def extended_profitability_test():
    """Extended backtest over multiple months."""
    
    logger.info("Running extended profitability test")
    
    # Load best model (Random Forest showed best performance)
    try:
        model = joblib.load("models/simple_ensemble/random_forest_model.pkl")
        logger.info("Loaded Random Forest model")
    except:
        logger.error("Could not load model")
        return

    # Load extended test data (3 months: May-July 2024)
    conn = duckdb.connect("data/historical_data.duckdb")
    query = """
    SELECT 
        timestamp,
        open,
        high,
        low,
        close,
        volume
    FROM klines_btcusdt
    WHERE timestamp >= '2024-05-01'
      AND timestamp <= '2024-07-31'  -- 3 months for statistical significance
    ORDER BY timestamp
    """
    
    data = conn.execute(query).df()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    conn.close()
    
    logger.info(f"Loaded {len(data)} test records over 3 months")
    
    # Create same features as training
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['hl_ratio'] = (data['high'] - data['low']) / data['close']
    data['oc_ratio'] = (data['close'] - data['open']) / data['open']
    
    for period in [1, 3, 5, 10, 15, 30]:
        data[f'return_{period}'] = data['close'].pct_change(period)
    
    for window in [5, 10, 20, 30]:
        data[f'vol_{window}'] = data['returns'].rolling(window).std()
    
    for ma in [5, 10, 20]:
        data[f'sma_{ma}'] = data['close'].rolling(ma).mean()
        data[f'price_vs_sma_{ma}'] = (data['close'] - data[f'sma_{ma}']) / data[f'sma_{ma}']
    
    data['momentum_3'] = data['close'].pct_change(3)
    data['momentum_5'] = data['close'].pct_change(5)
    data['momentum_10'] = data['close'].pct_change(10)
    
    data['volume_ma'] = data['volume'].rolling(20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data['log_volume'] = np.log(data['volume'] + 1)
    
    # RSI
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
    
    data['trend_strength'] = (data['sma_5'] - data['sma_20']) / data['sma_20']
    data['price_above_ma'] = (data['close'] > data['sma_20']).astype(int)
    data['volume_price_change'] = data['volume_ratio'] * abs(data['returns'])
    data['high_vol'] = (data['vol_20'] > data['vol_20'].rolling(50).quantile(0.8)).astype(int)
    data['low_vol'] = (data['vol_20'] < data['vol_20'].rolling(50).quantile(0.2)).astype(int)
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    
    # Fill NaN
    feature_cols = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    for col in feature_cols:
        data[col] = data[col].fillna(data[col].median())
    
    # Remove NaN rows
    data = data.dropna()
    
    # Get predictions
    X = data[feature_cols]
    predictions = model.predict_proba(X)[:, 1]
    
    # Multiple confidence threshold analysis
    confidence_thresholds = [0.65, 0.70, 0.75, 0.80, 0.85]
    results = {}
    
    for threshold in confidence_thresholds:
        logger.info(f"Testing confidence threshold: {threshold}")
        
        # Trading simulation
        capital = 100000
        position_size = 0.02  # 2%
        trades = []
        
        high_confidence_indices = np.where(predictions > threshold)[0]
        logger.info(f"Found {len(high_confidence_indices)} signals at {threshold} threshold")
        
        # Test more signals (up to 200)
        for i in high_confidence_indices[:200]:
            if i + 5 >= len(data):  # Need next 5 bars
                continue
                
            entry_price = data['close'].iloc[i]
            confidence = predictions[i]
            
            # Direction based on momentum
            direction = 'long' if data['return_3'].iloc[i] > 0 else 'short'
            
            # Simulate 5-bar hold
            exit_price = data['close'].iloc[i + 5]
            
            # Calculate PnL
            if direction == 'long':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            
            # Subtract fees
            pnl_pct -= 0.0012  # 0.12% round-trip
            
            # Calculate dollar PnL
            position_value = capital * position_size
            pnl_dollar = position_value * pnl_pct
            
            trades.append({
                'timestamp': data.index[i],
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'confidence': confidence,
                'pnl_pct': pnl_pct * 100,
                'pnl_dollar': pnl_dollar
            })
            
            capital += pnl_dollar
        
        if trades:
            trades_df = pd.DataFrame(trades)
            total_return = (capital - 100000) / 100000 * 100
            win_rate = len(trades_df[trades_df['pnl_dollar'] > 0]) / len(trades_df) * 100
            avg_trade = trades_df['pnl_dollar'].mean()
            total_pnl = trades_df['pnl_dollar'].sum()
            
            results[threshold] = {
                'trades': len(trades_df),
                'total_return': total_return,
                'win_rate': win_rate,
                'avg_trade': avg_trade,
                'total_pnl': total_pnl,
                'final_capital': capital
            }
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("🚀 拡張バックテスト結果 (3ヶ月: 2024年5-7月)")
    print("="*80)
    
    print(f"\n📊 テスト期間: 2024年5月1日 - 7月31日 (3ヶ月)")
    print(f"  データポイント: {len(data):,}")
    print(f"  初期資本: $100,000")
    
    print(f"\n🎯 信頼度閾値別パフォーマンス:")
    print(f"{'閾値':<8} {'取引数':<8} {'収益率':<10} {'勝率':<8} {'平均取引':<12} {'総利益':<12}")
    print("-" * 70)
    
    best_threshold = None
    best_return = -float('inf')
    
    for threshold, result in results.items():
        print(f"{threshold:<8} {result['trades']:<8} {result['total_return']:>8.2f}% "
              f"{result['win_rate']:>6.1f}% ${result['avg_trade']:>9.2f} "
              f"${result['total_pnl']:>10.2f}")
        
        if result['total_return'] > best_return:
            best_return = result['total_return']
            best_threshold = threshold
    
    # Analysis
    print(f"\n📈 最適パフォーマンス:")
    if best_threshold and best_return > 0:
        best_result = results[best_threshold]
        print(f"  🏆 最適閾値: {best_threshold}")
        print(f"  💰 最高収益率: {best_return:.2f}%")
        print(f"  📊 取引数: {best_result['trades']}")
        print(f"  🎯 勝率: {best_result['win_rate']:.1f}%")
        print(f"  💵 総利益: ${best_result['total_pnl']:,.2f}")
        
        # Monthly equivalent
        monthly_return = best_return / 3  # 3 months
        print(f"\n📅 月次換算:")
        print(f"  月次収益率: {monthly_return:.2f}%")
        print(f"  年次収益率: {monthly_return * 12:.2f}%")
        
        if monthly_return > 2:
            assessment = "🎉 非常に優秀！実用化推奨"
        elif monthly_return > 1:
            assessment = "✅ 良好！実用可能"
        elif monthly_return > 0:
            assessment = "⚠️ 微益だが改善の余地あり"
        else:
            assessment = "❌ さらなる改善が必要"
        
        print(f"\n🎯 総合評価: {assessment}")
        
    else:
        print(f"  ❌ 全ての閾値で損失")
        print(f"  最小損失: {best_return:.2f}% (閾値: {best_threshold})")
        
        print(f"\n🔧 改善提案:")
        print(f"  1. 特徴量エンジニアリングの改良")
        print(f"  2. モデルハイパーパラメータの最適化") 
        print(f"  3. より保守的なポジションサイジング")
        print(f"  4. 異なる市場条件での追加検証")
    
    # Performance consistency analysis
    print(f"\n📊 パフォーマンス分析:")
    profitable_thresholds = sum(1 for r in results.values() if r['total_return'] > 0)
    print(f"  収益性閾値数: {profitable_thresholds}/{len(confidence_thresholds)}")
    
    if profitable_thresholds > 0:
        avg_profitable_return = np.mean([r['total_return'] for r in results.values() if r['total_return'] > 0])
        print(f"  平均収益率: {avg_profitable_return:.2f}%")
    
    # Risk assessment
    all_returns = [r['total_return'] for r in results.values()]
    return_std = np.std(all_returns)
    print(f"  収益率標準偏差: {return_std:.2f}%")
    
    if return_std < 1:
        print(f"  ✅ 安定したパフォーマンス")
    elif return_std < 3:
        print(f"  ⚠️ 中程度のボラティリティ")
    else:
        print(f"  ❌ 高いボラティリティ - リスク管理強化が必要")
    
    return results


if __name__ == "__main__":
    extended_profitability_test()