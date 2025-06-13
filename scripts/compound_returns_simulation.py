#!/usr/bin/env python3
"""
Compound returns simulation for optimized 3x leverage strategy.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class CompoundReturnsSimulator:
    """Simulate compound returns with realistic considerations."""
    
    def __init__(self, monthly_return: float = 0.0416, initial_capital: float = 100000):
        self.monthly_return = monthly_return
        self.initial_capital = initial_capital
        
    def simulate_ideal_compound(self, years: int = 3) -> pd.DataFrame:
        """Simulate ideal compound returns without any friction."""
        
        months = years * 12
        dates = pd.date_range(start='2025-01-01', periods=months+1, freq='M')
        
        capital = [self.initial_capital]
        for i in range(months):
            next_capital = capital[-1] * (1 + self.monthly_return)
            capital.append(next_capital)
        
        df = pd.DataFrame({
            'date': dates,
            'capital': capital,
            'monthly_gain': [0] + [capital[i+1] - capital[i] for i in range(months)]
        })
        
        return df
    
    def simulate_realistic_compound(self, years: int = 3, 
                                  volatility_factor: float = 0.3,
                                  max_drawdown_prob: float = 0.05,
                                  capacity_limit: float = 10000000) -> pd.DataFrame:
        """Simulate realistic compound returns with market conditions."""
        
        months = years * 12
        dates = pd.date_range(start='2025-01-01', periods=months+1, freq='M')
        
        capital = [self.initial_capital]
        monthly_returns = []
        
        for i in range(months):
            current_capital = capital[-1]
            
            # Apply capacity constraints (reduced returns as capital grows)
            if current_capital > capacity_limit * 0.1:
                capacity_factor = 1 - min(0.5, (current_capital - capacity_limit * 0.1) / capacity_limit)
            else:
                capacity_factor = 1.0
            
            # Add realistic volatility
            return_std = self.monthly_return * volatility_factor
            actual_return = np.random.normal(self.monthly_return * capacity_factor, return_std)
            
            # Occasional drawdowns
            if np.random.random() < max_drawdown_prob:
                actual_return = -abs(np.random.normal(0.02, 0.01))  # 2% average drawdown
            
            # Cap extreme returns
            actual_return = max(-0.10, min(0.15, actual_return))  # -10% to +15% cap
            
            monthly_returns.append(actual_return)
            next_capital = current_capital * (1 + actual_return)
            capital.append(next_capital)
        
        df = pd.DataFrame({
            'date': dates,
            'capital': capital,
            'monthly_return': [0] + monthly_returns,
            'monthly_gain': [0] + [capital[i+1] - capital[i] for i in range(months)]
        })
        
        return df
    
    def simulate_with_withdrawals(self, years: int = 3,
                                monthly_withdrawal: float = 5000,
                                withdrawal_start_month: int = 12) -> pd.DataFrame:
        """Simulate compound returns with regular withdrawals."""
        
        months = years * 12
        dates = pd.date_range(start='2025-01-01', periods=months+1, freq='M')
        
        capital = [self.initial_capital]
        withdrawals = [0]
        
        for i in range(months):
            current_capital = capital[-1]
            
            # Calculate returns
            returns = current_capital * self.monthly_return
            
            # Apply withdrawal if applicable
            if i >= withdrawal_start_month - 1:
                withdrawal = min(monthly_withdrawal, returns * 0.5)  # Max 50% of returns
            else:
                withdrawal = 0
            
            withdrawals.append(withdrawal)
            next_capital = current_capital + returns - withdrawal
            capital.append(next_capital)
        
        df = pd.DataFrame({
            'date': dates,
            'capital': capital,
            'withdrawal': withdrawals,
            'monthly_gain': [0] + [capital[i+1] - capital[i] + withdrawals[i+1] for i in range(months)]
        })
        
        return df
    
    def calculate_statistics(self, df: pd.DataFrame) -> dict:
        """Calculate key statistics from simulation."""
        
        initial = df['capital'].iloc[0]
        final = df['capital'].iloc[-1]
        total_return = (final - initial) / initial
        
        # Calculate CAGR
        years = len(df) / 12
        cagr = (final / initial) ** (1/years) - 1
        
        # Calculate max drawdown
        peak = df['capital'].expanding().max()
        drawdown = (df['capital'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Monthly statistics
        if 'monthly_return' in df.columns:
            monthly_returns = df['monthly_return'][1:]  # Exclude first month
            avg_monthly = monthly_returns.mean()
            monthly_std = monthly_returns.std()
            sharpe_monthly = avg_monthly / monthly_std * np.sqrt(12) if monthly_std > 0 else 0
        else:
            avg_monthly = self.monthly_return
            monthly_std = 0
            sharpe_monthly = 0
        
        # Total withdrawals if applicable
        total_withdrawals = df['withdrawal'].sum() if 'withdrawal' in df.columns else 0
        
        return {
            'initial_capital': initial,
            'final_capital': final,
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'avg_monthly_return': avg_monthly,
            'monthly_volatility': monthly_std,
            'sharpe_ratio': sharpe_monthly,
            'total_withdrawals': total_withdrawals
        }
    
    def plot_compound_analysis(self, simulations: dict):
        """Create comprehensive compound returns visualization."""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Capital growth comparison
        ax1 = axes[0, 0]
        for name, df in simulations.items():
            ax1.plot(df['date'], df['capital'] / 1000, label=name, linewidth=2)
        
        ax1.set_title('資本成長比較（複利効果）', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital (千ドル)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}k'))
        
        # Plot 2: Monthly returns distribution (realistic only)
        ax2 = axes[0, 1]
        if 'realistic' in simulations and 'monthly_return' in simulations['realistic'].columns:
            monthly_returns = simulations['realistic']['monthly_return'][1:] * 100
            ax2.hist(monthly_returns, bins=30, edgecolor='black', alpha=0.7, color='green')
            ax2.axvline(x=self.monthly_return * 100, color='red', linestyle='--', 
                       label=f'期待値: {self.monthly_return*100:.1f}%')
            ax2.set_title('月次リターン分布（現実的シナリオ）', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Monthly Return (%)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative returns
        ax3 = axes[1, 0]
        for name, df in simulations.items():
            cumulative_return = (df['capital'] / self.initial_capital - 1) * 100
            ax3.plot(df['date'], cumulative_return, label=name, linewidth=2)
        
        ax3.set_title('累積収益率', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}%'))
        
        # Plot 4: Monthly gains
        ax4 = axes[1, 1]
        for name, df in simulations.items():
            if 'monthly_gain' in df.columns:
                ax4.plot(df['date'][1:], df['monthly_gain'][1:] / 1000, 
                        label=name, linewidth=2, alpha=0.7)
        
        ax4.set_title('月次利益推移', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Monthly Gain (千ドル)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}k'))
        
        plt.tight_layout()
        return fig


def main():
    """Run compound returns simulation."""
    
    logger.info("Starting compound returns simulation")
    
    print("="*80)
    print("💰 複利運用シミュレーション（月次4.16%）")
    print("="*80)
    
    # Initialize simulator
    simulator = CompoundReturnsSimulator(monthly_return=0.0416, initial_capital=100000)
    
    # Run different scenarios
    simulations = {}
    
    # 1. Ideal compound (no friction)
    simulations['理想的（摩擦なし）'] = simulator.simulate_ideal_compound(years=3)
    
    # 2. Realistic compound (with volatility and capacity constraints)
    simulations['現実的'] = simulator.simulate_realistic_compound(
        years=3, 
        volatility_factor=0.3,
        max_drawdown_prob=0.05,
        capacity_limit=10000000  # $10M capacity limit
    )
    
    # 3. With monthly withdrawals
    simulations['月次出金あり（$5k/月）'] = simulator.simulate_with_withdrawals(
        years=3,
        monthly_withdrawal=5000,
        withdrawal_start_month=12
    )
    
    # Calculate statistics
    print("\n📊 シミュレーション結果（3年間）:")
    print("-" * 80)
    
    for name, df in simulations.items():
        stats = simulator.calculate_statistics(df)
        
        print(f"\n🎯 {name}:")
        print(f"  初期資本: ${stats['initial_capital']:,.0f}")
        print(f"  最終資本: ${stats['final_capital']:,.0f}")
        print(f"  総収益率: {stats['total_return']*100:.1f}%")
        print(f"  年率成長率(CAGR): {stats['cagr']*100:.1f}%")
        
        if stats['max_drawdown'] < 0:
            print(f"  最大ドローダウン: {stats['max_drawdown']*100:.1f}%")
        
        if stats['monthly_volatility'] > 0:
            print(f"  月次ボラティリティ: {stats['monthly_volatility']*100:.1f}%")
            print(f"  Sharpe比: {stats['sharpe_ratio']:.2f}")
        
        if stats['total_withdrawals'] > 0:
            print(f"  総出金額: ${stats['total_withdrawals']:,.0f}")
    
    # Time-based milestones
    print("\n📈 理想的シナリオでの資産推移:")
    print("-" * 50)
    
    ideal_df = simulations['理想的（摩擦なし）']
    milestones = [6, 12, 18, 24, 30, 36]
    
    for month in milestones:
        if month <= len(ideal_df) - 1:
            capital = ideal_df.iloc[month]['capital']
            multiplier = capital / 100000
            print(f"  {month:2d}ヶ月後: ${capital:>12,.0f} ({multiplier:>5.1f}倍)")
    
    # Plot results
    fig = simulator.plot_compound_analysis(simulations)
    plt.savefig('backtest_results/compound_returns_analysis.png', dpi=300, bbox_inches='tight')
    
    # Additional analysis: Break-even with withdrawals
    print("\n💸 出金戦略分析:")
    print("-" * 50)
    
    # Find sustainable withdrawal rate
    sustainable_withdrawal = simulator.monthly_return * simulator.initial_capital * 0.5
    print(f"  持続可能な月次出金額: ${sustainable_withdrawal:,.0f}")
    print(f"  年間出金可能額: ${sustainable_withdrawal * 12:,.0f}")
    
    # Project 10-year scenario
    print("\n🚀 長期予測（10年、理想的シナリオ）:")
    long_term = simulator.simulate_ideal_compound(years=10)
    final_10y = long_term.iloc[-1]['capital']
    print(f"  10年後の予想資本: ${final_10y:,.0f}")
    print(f"  倍率: {final_10y/100000:.1f}倍")
    
    # Risk warnings
    print("\n⚠️ 重要な注意事項:")
    print("  1. 市場容量の制約により、資本が大きくなると収益率が低下する可能性")
    print("  2. 暗号通貨市場の変動により、想定外の損失が発生する可能性")
    print("  3. 規制変更やプラットフォームリスクの考慮が必要")
    print("  4. 税金の影響は含まれていません（実際の手取りは減少）")
    
    print(f"\n💾 保存完了:")
    print(f"  分析チャート: backtest_results/compound_returns_analysis.png")
    
    # Create summary table
    summary_df = pd.DataFrame({
        'シナリオ': ['理想的', '現実的', '出金あり'],
        '3年後資本': [
            simulations['理想的（摩擦なし）'].iloc[-1]['capital'],
            simulations['現実的'].iloc[-1]['capital'],
            simulations['月次出金あり（$5k/月）'].iloc[-1]['capital']
        ],
        '倍率': [
            simulations['理想的（摩擦なし）'].iloc[-1]['capital'] / 100000,
            simulations['現実的'].iloc[-1]['capital'] / 100000,
            simulations['月次出金あり（$5k/月）'].iloc[-1]['capital'] / 100000
        ]
    })
    
    summary_df.to_csv('backtest_results/compound_returns_summary.csv', index=False)
    print(f"  サマリーデータ: backtest_results/compound_returns_summary.csv")
    
    return simulations


if __name__ == "__main__":
    main()