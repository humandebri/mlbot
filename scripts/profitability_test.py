#!/usr/bin/env python3
"""
Simple profitability test for the improved strategy.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def test_strategy_profitability():
    """Test if the improved strategy can actually generate profits."""
    logger.info("Testing strategy profitability")
    
    # Load data
    conn = duckdb.connect("data/historical_data.duckdb")
    query = """
    SELECT 
        timestamp,
        close,
        volume
    FROM klines_btcusdt
    WHERE timestamp >= '2024-01-01'
      AND timestamp <= '2024-01-31'
    ORDER BY timestamp
    """
    
    data = conn.execute(query).df()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    conn.close()
    
    # Calculate features
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(30).std()
    data['price_change'] = data['close'].pct_change(15)
    data['volume_ma'] = data['volume'].rolling(30).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data = data.dropna()
    
    logger.info(f"Testing on {len(data)} data points (January 2024)")
    
    # Test trading simulation
    capital = 100000
    position_size = 0.03  # 3% per trade
    trades = []
    signals_tested = 0
    
    for i in range(len(data) - 1):  # Need next bar for exit
        row = data.iloc[i]
        next_row = data.iloc[i + 1]
        
        # Generate signal (relaxed thresholds)
        signal = (
            abs(row['volatility']) > 0.003 or  
            (abs(row['price_change']) > 0.005 and row['volume_ratio'] > 1.2)
        )
        
        if signal:
            signals_tested += 1
            
            # Trade direction
            direction = 'long' if row['price_change'] > 0 else 'short'
            entry_price = row['close']
            exit_price = next_row['close']
            
            # Calculate PnL
            if direction == 'long':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            
            # Apply costs (fees + slippage)
            pnl_pct -= 0.0012  # 0.06% entry fee + 0.06% exit fee
            
            # Calculate dollar PnL
            position_value = capital * position_size
            pnl_dollar = position_value * pnl_pct
            
            trades.append({
                'timestamp': row['timestamp'],
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct * 100,
                'pnl_dollar': pnl_dollar,
                'volatility': row['volatility'],
                'price_change': row['price_change']
            })
            
            # Update capital
            capital += pnl_dollar
    
    # Analysis
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) > 0:
        total_return = (capital - 100000) / 100000 * 100
        win_rate = len(trades_df[trades_df['pnl_dollar'] > 0]) / len(trades_df) * 100
        avg_win = trades_df[trades_df['pnl_dollar'] > 0]['pnl_dollar'].mean() if len(trades_df[trades_df['pnl_dollar'] > 0]) > 0 else 0
        avg_loss = trades_df[trades_df['pnl_dollar'] < 0]['pnl_dollar'].mean() if len(trades_df[trades_df['pnl_dollar'] < 0]) > 0 else 0
        
        profitable_days = 0
        daily_pnl = trades_df.groupby(trades_df['timestamp'].dt.date)['pnl_dollar'].sum()
        profitable_days = len(daily_pnl[daily_pnl > 0])
        
        print("\n" + "="*60)
        print("BOT収益性テスト結果 (2024年1月)")
        print("="*60)
        
        print(f"\n📊 取引統計:")
        print(f"  総取引回数:        {len(trades_df):,}回")
        print(f"  信号適中率:        {len(trades_df)/signals_tested*100:.1f}% ({len(trades_df)}/{signals_tested})")
        print(f"  勝率:             {win_rate:.1f}%")
        print(f"  平均勝ちトレード:   ${avg_win:.2f}")
        print(f"  平均負けトレード:   ${avg_loss:.2f}")
        
        print(f"\n💰 収益性:")
        print(f"  初期資本:         $100,000")
        print(f"  最終資本:         ${capital:,.2f}")
        print(f"  総収益率:         {total_return:.2f}%")
        print(f"  総利益:           ${capital-100000:,.2f}")
        print(f"  収益性日数:        {profitable_days}/{len(daily_pnl)}日")
        
        # Risk analysis
        daily_returns = trades_df.groupby(trades_df['timestamp'].dt.date)['pnl_dollar'].sum() / 100000 * 100
        volatility = daily_returns.std()
        sharpe = daily_returns.mean() / volatility * np.sqrt(252) if volatility > 0 else 0
        
        print(f"\n⚠️  リスク指標:")
        print(f"  日次ボラティリティ: {volatility:.2f}%")
        print(f"  シャープレシオ:     {sharpe:.2f}")
        
        # Determine bot viability
        print(f"\n🤖 Bot適用性評価:")
        if total_return > 2:
            print(f"  ✅ 良好 - 月間{total_return:.1f}%の収益")
        elif total_return > 0:
            print(f"  ⚠️  微益 - 月間{total_return:.1f}%の微小利益")
        else:
            print(f"  ❌ 損失 - 月間{total_return:.1f}%の損失")
            
        if win_rate > 55:
            print(f"  ✅ 勝率{win_rate:.1f}%は良好")
        elif win_rate > 45:
            print(f"  ⚠️  勝率{win_rate:.1f}%は普通")
        else:
            print(f"  ❌ 勝率{win_rate:.1f}%は低い")
        
        if sharpe > 1:
            print(f"  ✅ シャープレシオ{sharpe:.2f}は優秀")
        elif sharpe > 0.5:
            print(f"  ⚠️  シャープレシオ{sharpe:.2f}は普通")
        else:
            print(f"  ❌ シャープレシオ{sharpe:.2f}は低い")
        
        # Bot recommendation
        print(f"\n💡 推奨:")
        if total_return > 1 and win_rate > 50 and sharpe > 0.5:
            print(f"  🚀 Botとして有望！実用化を検討")
        elif total_return > 0:
            print(f"  🔧 調整必要 - パラメータの最適化が必要")
        else:
            print(f"  🛑 現状では推奨できない - 戦略の根本的見直しが必要")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cumulative PnL
        trades_df['cumulative_pnl'] = trades_df['pnl_dollar'].cumsum()
        ax1.plot(trades_df['timestamp'], 100000 + trades_df['cumulative_pnl'], 'b-', linewidth=2)
        ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('資本曲線')
        ax1.set_ylabel('資本 ($)')
        ax1.grid(True, alpha=0.3)
        
        # PnL distribution
        ax2.hist(trades_df['pnl_dollar'], bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('取引PnL分布')
        ax2.set_xlabel('PnL ($)')
        ax2.set_ylabel('頻度')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results/profitability_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'total_trades': len(trades_df),
            'profitable': total_return > 0
        }
    
    else:
        print("❌ 取引が実行されませんでした")
        return None


if __name__ == "__main__":
    test_strategy_profitability()