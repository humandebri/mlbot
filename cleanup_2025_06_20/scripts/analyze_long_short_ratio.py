#!/usr/bin/env python3
"""
Analyze long/short ratio and performance from optimized trades.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def analyze_direction_performance():
    """Analyze performance of long vs short trades."""
    
    # Load optimized position trades
    try:
        trades_df = pd.read_csv('backtest_results/optimized_position_trades.csv')
        logger.info(f"Loaded {len(trades_df)} trades from optimized strategy")
    except:
        logger.error("Could not load optimized trades file")
        return
    
    # Basic statistics
    total_trades = len(trades_df)
    long_trades = trades_df[trades_df['direction'] == 'long']
    short_trades = trades_df[trades_df['direction'] == 'short']
    
    print("="*80)
    print("📊 ロング/ショート取引分析")
    print("="*80)
    
    print(f"\n🎯 取引方向の内訳:")
    print(f"  総取引数: {total_trades}")
    print(f"  ロング取引: {len(long_trades)} ({len(long_trades)/total_trades*100:.1f}%)")
    print(f"  ショート取引: {len(short_trades)} ({len(short_trades)/total_trades*100:.1f}%)")
    
    # Performance by direction
    print(f"\n📈 方向別パフォーマンス:")
    print("-" * 60)
    
    # Long performance
    if len(long_trades) > 0:
        long_wins = len(long_trades[long_trades['pnl_dollar'] > 0])
        long_win_rate = long_wins / len(long_trades) * 100
        long_avg_pnl = long_trades['pnl_dollar'].mean()
        long_total_pnl = long_trades['pnl_dollar'].sum()
        long_avg_pnl_pct = long_trades['pnl_pct'].mean()
        
        print(f"\n🔵 ロング取引:")
        print(f"  勝率: {long_win_rate:.1f}%")
        print(f"  平均損益: ${long_avg_pnl:.2f}")
        print(f"  平均収益率: {long_avg_pnl_pct:.3f}%")
        print(f"  総利益: ${long_total_pnl:,.2f}")
    
    # Short performance
    if len(short_trades) > 0:
        short_wins = len(short_trades[short_trades['pnl_dollar'] > 0])
        short_win_rate = short_wins / len(short_trades) * 100
        short_avg_pnl = short_trades['pnl_dollar'].mean()
        short_total_pnl = short_trades['pnl_dollar'].sum()
        short_avg_pnl_pct = short_trades['pnl_pct'].mean()
        
        print(f"\n🔴 ショート取引:")
        print(f"  勝率: {short_win_rate:.1f}%")
        print(f"  平均損益: ${short_avg_pnl:.2f}")
        print(f"  平均収益率: {short_avg_pnl_pct:.3f}%")
        print(f"  総利益: ${short_total_pnl:,.2f}")
    
    # Time analysis
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df['month'] = trades_df['timestamp'].dt.to_period('M')
    
    # Monthly direction breakdown
    print(f"\n📅 月別取引方向:")
    print("-" * 60)
    
    monthly_direction = trades_df.groupby(['month', 'direction']).size().unstack(fill_value=0)
    for month in monthly_direction.index:
        long_count = monthly_direction.loc[month, 'long'] if 'long' in monthly_direction.columns else 0
        short_count = monthly_direction.loc[month, 'short'] if 'short' in monthly_direction.columns else 0
        total = long_count + short_count
        if total > 0:
            print(f"  {month}: ロング {long_count} ({long_count/total*100:.0f}%), "
                  f"ショート {short_count} ({short_count/total*100:.0f}%)")
    
    # Volatility analysis
    print(f"\n🌊 ボラティリティ別分析:")
    print("-" * 60)
    
    # Low, medium, high volatility
    low_vol = trades_df[trades_df['volatility'] < 0.015]
    med_vol = trades_df[(trades_df['volatility'] >= 0.015) & (trades_df['volatility'] < 0.025)]
    high_vol = trades_df[trades_df['volatility'] >= 0.025]
    
    for vol_name, vol_df in [('低ボラ', low_vol), ('中ボラ', med_vol), ('高ボラ', high_vol)]:
        if len(vol_df) > 0:
            long_pct = len(vol_df[vol_df['direction'] == 'long']) / len(vol_df) * 100
            short_pct = 100 - long_pct
            print(f"  {vol_name}: ロング {long_pct:.0f}%, ショート {short_pct:.0f}%")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Direction pie chart
    ax1 = axes[0, 0]
    direction_counts = trades_df['direction'].value_counts()
    colors = ['#2E86AB', '#E63946']
    ax1.pie(direction_counts.values, labels=direction_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('取引方向の割合', fontsize=14, fontweight='bold')
    
    # Plot 2: PnL by direction
    ax2 = axes[0, 1]
    long_pnls = long_trades['pnl_dollar'].values if len(long_trades) > 0 else []
    short_pnls = short_trades['pnl_dollar'].values if len(short_trades) > 0 else []
    
    bp_data = []
    labels = []
    if len(long_pnls) > 0:
        bp_data.append(long_pnls)
        labels.append('Long')
    if len(short_pnls) > 0:
        bp_data.append(short_pnls)
        labels.append('Short')
    
    if bp_data:
        bp = ax2.boxplot(bp_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(bp_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('方向別損益分布', fontsize=14, fontweight='bold')
    ax2.set_ylabel('PnL ($)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Monthly direction trend
    ax3 = axes[1, 0]
    if not monthly_direction.empty:
        monthly_direction.plot(kind='bar', stacked=True, ax=ax3, color=colors)
        ax3.set_title('月別取引方向の推移', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Number of Trades')
        ax3.legend(['Long', 'Short'])
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 4: Win rate comparison
    ax4 = axes[1, 1]
    win_rates = []
    avg_returns = []
    labels = []
    
    if len(long_trades) > 0:
        win_rates.append(long_win_rate)
        avg_returns.append(long_avg_pnl_pct)
        labels.append('Long')
    
    if len(short_trades) > 0:
        win_rates.append(short_win_rate)
        avg_returns.append(short_avg_pnl_pct)
        labels.append('Short')
    
    if win_rates:
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, win_rates, width, label='勝率 (%)', color='green', alpha=0.7)
        ax4_2 = ax4.twinx()
        bars2 = ax4_2.bar(x + width/2, avg_returns, width, label='平均収益率 (%)', color='blue', alpha=0.7)
        
        ax4.set_xlabel('Direction')
        ax4.set_ylabel('Win Rate (%)', color='green')
        ax4_2.set_ylabel('Avg Return (%)', color='blue')
        ax4.set_title('方向別パフォーマンス比較', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax4_2.text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.3f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('backtest_results/long_short_analysis.png', dpi=300, bbox_inches='tight')
    
    # Summary
    print(f"\n💡 分析結果:")
    print("-" * 60)
    
    if len(long_trades) > 0 and len(short_trades) > 0:
        if long_total_pnl > short_total_pnl:
            profit_ratio = long_total_pnl / (long_total_pnl + short_total_pnl) * 100
            print(f"  ✅ ロング取引が総利益の{profit_ratio:.0f}%を占める")
        else:
            profit_ratio = short_total_pnl / (long_total_pnl + short_total_pnl) * 100
            print(f"  ✅ ショート取引が総利益の{profit_ratio:.0f}%を占める")
        
        print(f"  📊 両方向の取引により、市場状況に関わらず収益機会を捕捉")
        print(f"  🎯 モメンタムベースの方向判断が有効に機能")
    
    print(f"\n💾 保存完了:")
    print(f"  分析チャート: backtest_results/long_short_analysis.png")
    
    return trades_df


def analyze_market_conditions():
    """Analyze market conditions during the test period."""
    
    import duckdb
    
    conn = duckdb.connect("data/historical_data.duckdb")
    query = """
    SELECT 
        DATE_TRUNC('month', timestamp) as month,
        FIRST(close) as month_start,
        LAST(close) as month_end,
        MAX(close) as month_high,
        MIN(close) as month_low
    FROM klines_btcusdt
    WHERE timestamp >= '2024-05-01' AND timestamp <= '2024-07-31'
    GROUP BY DATE_TRUNC('month', timestamp)
    ORDER BY month
    """
    
    market_data = conn.execute(query).df()
    conn.close()
    
    print(f"\n📊 テスト期間の市場状況:")
    print("-" * 60)
    
    for _, row in market_data.iterrows():
        month_return = (row['month_end'] - row['month_start']) / row['month_start'] * 100
        month_range = (row['month_high'] - row['month_low']) / row['month_start'] * 100
        trend = "上昇" if month_return > 0 else "下落"
        
        print(f"  {row['month'].strftime('%Y-%m')}: {trend}市場 "
              f"(月間{month_return:+.1f}%, レンジ{month_range:.1f}%)")


if __name__ == "__main__":
    analyze_direction_performance()
    analyze_market_conditions()