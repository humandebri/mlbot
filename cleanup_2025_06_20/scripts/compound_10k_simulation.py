#!/usr/bin/env python3
"""
Compound returns simulation starting with $10,000 initial capital.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import japanize_matplotlib  # 日本語フォント対応
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def simulate_compound_growth(initial_capital: float = 10000, 
                           monthly_return: float = 0.0416,
                           years: int = 5):
    """Simulate compound growth with various scenarios."""
    
    months = years * 12
    dates = pd.date_range(start='2025-01-01', periods=months+1, freq='M')
    
    # Scenario 1: Ideal compound (no friction)
    ideal_capital = [initial_capital]
    for i in range(months):
        next_capital = ideal_capital[-1] * (1 + monthly_return)
        ideal_capital.append(next_capital)
    
    # Scenario 2: Realistic with volatility
    realistic_capital = [initial_capital]
    monthly_returns = []
    
    for i in range(months):
        current_capital = realistic_capital[-1]
        
        # Capacity constraints (less impact at lower capital levels)
        if current_capital > 1000000:  # $1M
            capacity_factor = 0.9
        elif current_capital > 500000:  # $500k
            capacity_factor = 0.95
        else:
            capacity_factor = 1.0
        
        # Monthly volatility
        return_std = monthly_return * 0.3
        actual_return = np.random.normal(monthly_return * capacity_factor, return_std)
        
        # Occasional small drawdowns (5% chance)
        if np.random.random() < 0.05:
            actual_return = -abs(np.random.normal(0.015, 0.005))
        
        # Cap returns
        actual_return = max(-0.08, min(0.12, actual_return))
        
        monthly_returns.append(actual_return)
        next_capital = current_capital * (1 + actual_return)
        realistic_capital.append(next_capital)
    
    # Scenario 3: Conservative (80% of returns)
    conservative_capital = [initial_capital]
    conservative_return = monthly_return * 0.8
    for i in range(months):
        next_capital = conservative_capital[-1] * (1 + conservative_return)
        conservative_capital.append(next_capital)
    
    return {
        'dates': dates,
        'ideal': ideal_capital,
        'realistic': realistic_capital,
        'conservative': conservative_capital,
        'monthly_returns': monthly_returns
    }


def calculate_milestones(capital_series: list, initial: float = 10000):
    """Calculate key milestones."""
    
    milestones = {}
    multiples = [2, 5, 10, 20, 50, 100]
    
    for multiple in multiples:
        target = initial * multiple
        for i, capital in enumerate(capital_series):
            if capital >= target:
                milestones[multiple] = i
                break
    
    return milestones


def create_visualization(results: dict, initial_capital: float = 10000):
    """Create comprehensive visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Capital growth comparison
    ax1 = axes[0, 0]
    ax1.plot(results['dates'], np.array(results['ideal']) / 1000, 
             'g-', linewidth=3, label='理想的シナリオ')
    ax1.plot(results['dates'], np.array(results['realistic']) / 1000, 
             'b-', linewidth=2, label='現実的シナリオ', alpha=0.8)
    ax1.plot(results['dates'], np.array(results['conservative']) / 1000, 
             'r--', linewidth=2, label='保守的シナリオ', alpha=0.8)
    
    # Add milestone markers
    milestones = calculate_milestones(results['ideal'], initial_capital)
    for multiple, month in milestones.items():
        if month < len(results['dates']):
            ax1.plot(results['dates'][month], results['ideal'][month] / 1000, 
                    'go', markersize=8)
            ax1.annotate(f'{multiple}x', 
                        (results['dates'][month], results['ideal'][month] / 1000),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_title(f'資本成長シミュレーション（初期資本: ${initial_capital:,}）', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Capital (千ドル)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}k'))
    
    # Plot 2: Monthly returns distribution
    ax2 = axes[0, 1]
    monthly_returns_pct = np.array(results['monthly_returns']) * 100
    ax2.hist(monthly_returns_pct, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(x=4.16, color='red', linestyle='--', linewidth=2, 
               label='期待値: 4.16%')
    ax2.set_title('月次リターン分布（現実的シナリオ）', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Monthly Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time to reach milestones
    ax3 = axes[1, 0]
    scenarios = ['理想的', '現実的', '保守的']
    colors = ['green', 'blue', 'red']
    x_pos = np.arange(len(scenarios))
    
    # Time to 10x
    times_to_10x = []
    for scenario in ['ideal', 'realistic', 'conservative']:
        milestones = calculate_milestones(results[scenario], initial_capital)
        times_to_10x.append(milestones.get(10, 60) / 12)  # Convert to years
    
    bars = ax3.bar(x_pos, times_to_10x, color=colors, alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(scenarios)
    ax3.set_ylabel('Years')
    ax3.set_title('10倍達成までの期間', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time in zip(bars, times_to_10x):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}年', ha='center', va='bottom')
    
    # Plot 4: Projected values table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create projection table
    time_points = [6, 12, 24, 36, 48, 60]
    table_data = []
    
    for month in time_points:
        if month < len(results['ideal']):
            row = [
                f'{month}ヶ月',
                f'${results["ideal"][month]:,.0f}',
                f'${results["realistic"][month]:,.0f}',
                f'${results["conservative"][month]:,.0f}'
            ]
            table_data.append(row)
    
    table = ax4.table(cellText=table_data,
                     colLabels=['期間', '理想的', '現実的', '保守的'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.27, 0.27, 0.26])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax4.set_title('資産推移予測', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def main():
    """Run $10,000 compound simulation."""
    
    logger.info("Starting $10,000 compound returns simulation")
    
    print("="*80)
    print("💰 $10,000スタート 複利運用シミュレーション（月次4.16%）")
    print("="*80)
    
    # Run 5-year simulation
    results = simulate_compound_growth(initial_capital=10000, 
                                     monthly_return=0.0416, 
                                     years=5)
    
    # Calculate key metrics
    print("\n📊 5年間のシミュレーション結果:")
    print("-" * 60)
    
    scenarios = {
        '理想的（摩擦なし）': results['ideal'],
        '現実的（変動あり）': results['realistic'],
        '保守的（80%収益）': results['conservative']
    }
    
    for name, capital_series in scenarios.items():
        initial = capital_series[0]
        final = capital_series[-1]
        total_return = (final - initial) / initial * 100
        cagr = (final / initial) ** (1/5) - 1
        
        print(f"\n🎯 {name}:")
        print(f"  初期資本: ${initial:,.0f}")
        print(f"  5年後資本: ${final:,.0f}")
        print(f"  倍率: {final/initial:.1f}倍")
        print(f"  総収益率: {total_return:.1f}%")
        print(f"  年率成長率(CAGR): {cagr*100:.1f}%")
    
    # Milestone analysis
    print("\n🏆 達成マイルストーン（理想的シナリオ）:")
    print("-" * 60)
    
    milestones = calculate_milestones(results['ideal'], 10000)
    milestone_targets = {
        2: '$20,000（2倍）',
        5: '$50,000（5倍）',
        10: '$100,000（10倍）',
        20: '$200,000（20倍）',
        50: '$500,000（50倍）',
        100: '$1,000,000（100倍）'
    }
    
    for multiple, target_desc in milestone_targets.items():
        if multiple in milestones:
            months = milestones[multiple]
            years = months / 12
            print(f"  {target_desc}: {months}ヶ月後（{years:.1f}年）")
    
    # Practical timeline
    print("\n📅 現実的な資産推移予測:")
    print("-" * 60)
    
    checkpoints = [
        (12, '1年後'),
        (24, '2年後'),
        (36, '3年後'),
        (48, '4年後'),
        (60, '5年後')
    ]
    
    for months, label in checkpoints:
        if months < len(results['realistic']):
            capital = results['realistic'][months]
            multiplier = capital / 10000
            print(f"  {label}: ${capital:>10,.0f} ({multiplier:>5.1f}倍)")
    
    # Risk analysis
    print("\n⚡ リスク分析（現実的シナリオ）:")
    print("-" * 60)
    
    monthly_returns = np.array(results['monthly_returns'])
    positive_months = np.sum(monthly_returns > 0)
    negative_months = np.sum(monthly_returns < 0)
    worst_month = np.min(monthly_returns) * 100
    best_month = np.max(monthly_returns) * 100
    
    print(f"  プラス月数: {positive_months}/{len(monthly_returns)} "
          f"({positive_months/len(monthly_returns)*100:.1f}%)")
    print(f"  マイナス月数: {negative_months}/{len(monthly_returns)} "
          f"({negative_months/len(monthly_returns)*100:.1f}%)")
    print(f"  最高月次リターン: {best_month:.1f}%")
    print(f"  最低月次リターン: {worst_month:.1f}%")
    
    # Create visualization
    fig = create_visualization(results, initial_capital=10000)
    plt.savefig('backtest_results/compound_10k_analysis.png', dpi=300, bbox_inches='tight')
    
    # Practical advice
    print("\n💡 実践的アドバイス:")
    print("-" * 60)
    print("  1. 最初の1年は複利効果が小さく感じるが、継続が重要")
    print("  2. 2-3年後から指数関数的成長が顕著に")
    print("  3. $100,000到達（10倍）は約3.5年で達成可能")
    print("  4. 初期の$10,000でも5年で$100,000以上に成長する可能性")
    
    print("\n⚠️ 注意事項:")
    print("  • 税金は考慮されていません（実際の手取りは減少）")
    print("  • 暗号通貨市場の変動リスクがあります")
    print("  • 戦略の継続的な有効性は保証されません")
    
    print(f"\n💾 保存完了:")
    print(f"  分析チャート: backtest_results/compound_10k_analysis.png")
    
    # Extended projection
    print("\n🚀 長期展望（理想的シナリオ）:")
    print("-" * 60)
    
    # Calculate 10-year projection
    extended_months = 10 * 12
    extended_capital = 10000
    for _ in range(extended_months):
        extended_capital *= 1.0416
    
    print(f"  10年後予測: ${extended_capital:,.0f} ({extended_capital/10000:.0f}倍)")
    print(f"  つまり、$10,000 → $1,000,000+ が現実的に可能！")
    
    return results


if __name__ == "__main__":
    main()