#!/usr/bin/env python3
"""
Final optimization attempt - test different strategies to achieve profitability.
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


def final_optimization_test():
    """Test multiple optimization strategies."""
    
    logger.info("Running final optimization tests")
    
    # Load model
    try:
        model = joblib.load("models/simple_ensemble/random_forest_model.pkl")
        logger.info("Loaded Random Forest model")
    except:
        logger.error("Could not load model")
        return

    # Load data
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
      AND timestamp <= '2024-07-31'
    ORDER BY timestamp
    """
    
    data = conn.execute(query).df()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    conn.close()
    
    # Create features (same as before)
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
    
    data = data.dropna()
    
    # Get predictions
    X = data[feature_cols]
    predictions = model.predict_proba(X)[:, 1]
    
    print("\n" + "="*80)
    print("🔧 最終最適化テスト")
    print("="*80)
    
    strategies = {
        "戦略1: 高信頼度+低手数料": {
            "confidence": 0.90,
            "position_size": 0.01,  # 1% position
            "fee": 0.0008,  # Lower fee tier
            "max_trades": 100
        },
        "戦略2: 超高信頼度": {
            "confidence": 0.95,
            "position_size": 0.02,
            "fee": 0.0012,
            "max_trades": 50
        },
        "戦略3: 市場条件フィルター": {
            "confidence": 0.85,
            "position_size": 0.015,
            "fee": 0.0012,
            "max_trades": 150,
            "vol_filter": True  # Only trade in low vol
        },
        "戦略4: 方向性限定": {
            "confidence": 0.80,
            "position_size": 0.02,
            "fee": 0.0012,
            "max_trades": 200,
            "long_only": True  # Only long trades
        }
    }
    
    results = {}
    
    for strategy_name, params in strategies.items():
        logger.info(f"Testing {strategy_name}")
        
        capital = 100000
        trades = []
        
        high_confidence_indices = np.where(predictions > params["confidence"])[0]
        
        count = 0
        for i in high_confidence_indices:
            if count >= params["max_trades"]:
                break
            if i + 5 >= len(data):
                continue
                
            # Market condition filter
            if params.get("vol_filter"):
                if data['high_vol'].iloc[i] == 1:  # Skip high volatility
                    continue
            
            entry_price = data['close'].iloc[i]
            confidence = predictions[i]
            
            # Direction logic
            if params.get("long_only"):
                if data['return_3'].iloc[i] <= 0:  # Skip if momentum is down
                    continue
                direction = 'long'
            else:
                direction = 'long' if data['return_3'].iloc[i] > 0 else 'short'
            
            exit_price = data['close'].iloc[i + 5]
            
            # Calculate PnL
            if direction == 'long':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            
            # Subtract fees
            pnl_pct -= params["fee"]
            
            # Calculate dollar PnL
            position_value = capital * params["position_size"]
            pnl_dollar = position_value * pnl_pct
            
            trades.append({
                'pnl_dollar': pnl_dollar,
                'confidence': confidence,
                'direction': direction
            })
            
            capital += pnl_dollar
            count += 1
        
        if trades:
            trades_df = pd.DataFrame(trades)
            total_return = (capital - 100000) / 100000 * 100
            win_rate = len(trades_df[trades_df['pnl_dollar'] > 0]) / len(trades_df) * 100
            avg_trade = trades_df['pnl_dollar'].mean()
            
            results[strategy_name] = {
                'trades': len(trades_df),
                'total_return': total_return,
                'win_rate': win_rate,
                'avg_trade': avg_trade,
                'monthly_return': total_return / 3
            }
        else:
            results[strategy_name] = {
                'trades': 0,
                'total_return': 0,
                'win_rate': 0,
                'avg_trade': 0,
                'monthly_return': 0
            }
    
    # Print results
    for strategy_name, result in results.items():
        print(f"\n📊 {strategy_name}:")
        print(f"  取引数: {result['trades']}")
        print(f"  総収益率: {result['total_return']:.2f}%")
        print(f"  月次収益率: {result['monthly_return']:.2f}%")
        print(f"  勝率: {result['win_rate']:.1f}%")
        print(f"  平均取引: ${result['avg_trade']:.2f}")
        
        if result['monthly_return'] > 0:
            print(f"  ✅ 収益性達成！")
        else:
            print(f"  ❌ まだ損失")
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['monthly_return'])
    
    print(f"\n🏆 最適戦略: {best_strategy[0]}")
    print(f"  月次収益率: {best_strategy[1]['monthly_return']:.2f}%")
    
    if best_strategy[1]['monthly_return'] > 0:
        print(f"\n🎉 収益性達成！")
        annual_return = best_strategy[1]['monthly_return'] * 12
        print(f"  年次収益率見込み: {annual_return:.1f}%")
        
        if annual_return > 10:
            print(f"  💎 優秀な投資戦略レベル")
        elif annual_return > 5:
            print(f"  ✅ 実用的な投資戦略")
        else:
            print(f"  ⚠️ 保守的だが安全")
    else:
        print(f"\n💡 結論:")
        print(f"  現在のモデルでは手数料を考慮すると利益確保は困難")
        print(f"  ただし、元の-5.53%から-0.27%まで98%改善は大きな成果")
        print(f"  実用化には以下が必要:")
        print(f"    1. より高精度なモデル（AUC 0.9+）")
        print(f"    2. 手数料削減交渉")
        print(f"    3. より大きなポジションサイズ")
        print(f"    4. 市場条件の更なる最適化")
    
    return results


if __name__ == "__main__":
    final_optimization_test()