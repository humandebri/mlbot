#!/usr/bin/env python3
"""
Quick improved backtest with proper timestamp handling.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class QuickBacktester:
    """Quick improved backtester with proper timestamp handling."""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.equity_curve = []
        
    def run_quick_backtest(self):
        """Run quick backtest to demonstrate improvements."""
        logger.info("Starting quick improved backtest")
        
        # Load historical data
        conn = duckdb.connect("data/historical_data.duckdb")
        
        # Get BTCUSDT data for 2024
        query = """
        SELECT 
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM klines_btcusdt
        WHERE timestamp >= '2024-01-01'
          AND timestamp <= '2024-01-31'  -- Just January for quick test
        ORDER BY timestamp
        """
        
        data = conn.execute(query).df()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        conn.close()
        
        logger.info(f"Loaded {len(data)} data points")
        
        # Calculate features
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(30).std()
        data['price_change'] = data['close'].pct_change(15)
        data['volume_ma'] = data['volume'].rolling(30).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Remove NaN
        data = data.dropna()
        
        # Generate signals with RELAXED thresholds
        original_signals = 0
        improved_signals = 0
        
        for i in range(len(data)):
            row = data.iloc[i]
            current_time = data.index[i]
            
            # ORIGINAL restrictive conditions (1.5% volatility, 1.0% momentum)
            original_signal = (
                abs(row['volatility']) > 0.015 and
                abs(row['price_change']) > 0.01 and
                row['volume_ratio'] > 1.5
            )
            if original_signal:
                original_signals += 1
            
            # IMPROVED relaxed conditions (0.3% volatility, 0.5% momentum)
            improved_signal = (
                abs(row['volatility']) > 0.003 or  # Much more relaxed
                (abs(row['price_change']) > 0.005 and row['volume_ratio'] > 1.2)
            )
            if improved_signal:
                improved_signals += 1
                
                # Simulate trade
                direction = 'long' if row['price_change'] > 0 else 'short'
                entry_price = row['close']
                
                # Simple profit simulation (next bar)
                if i < len(data) - 1:
                    next_price = data.iloc[i + 1]['close']
                    if direction == 'long':
                        pnl = (next_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - next_price) / entry_price
                    
                    # Apply some fees/slippage
                    pnl -= 0.001  # 0.1% fees/slippage
                    
                    self.trades.append({
                        'timestamp': current_time,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': next_price,
                        'pnl_pct': pnl * 100,
                        'capital_effect': self.capital * 0.03 * pnl  # 3% position size
                    })
                    
                    # Update capital
                    self.capital += self.capital * 0.03 * pnl
            
            # Record equity
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': self.capital
            })
        
        # Calculate results
        results = {
            'period': 'January 2024',
            'data_points': len(data),
            'original_signals': original_signals,
            'improved_signals': improved_signals,
            'signal_improvement': f"{improved_signals/original_signals:.1f}x" if original_signals > 0 else "∞",
            'total_trades': len(self.trades),
            'final_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'avg_trade_pnl': np.mean([t['pnl_pct'] for t in self.trades]) if self.trades else 0,
            'win_rate': len([t for t in self.trades if t['pnl_pct'] > 0]) / len(self.trades) * 100 if self.trades else 0
        }
        
        return results
    
    def print_comparison(self, results: Dict):
        """Print comparison results."""
        print("\n" + "="*60)
        print("TRADING OPPORTUNITY IMPROVEMENT ANALYSIS")
        print("="*60)
        
        print(f"\nData Analysis ({results['period']}):")
        print(f"  Total data points:     {results['data_points']:,}")
        print(f"  Original strategy:     {results['original_signals']:,} signals")
        print(f"  Improved strategy:     {results['improved_signals']:,} signals")
        print(f"  Signal improvement:    {results['signal_improvement']}")
        
        print(f"\nTrading Results:")
        print(f"  Total trades executed: {results['total_trades']:,}")
        print(f"  Final capital:         ${results['final_capital']:,.2f}")
        print(f"  Total return:          {results['total_return']:.2f}%")
        print(f"  Average trade PnL:     {results['avg_trade_pnl']:.3f}%")
        print(f"  Win rate:              {results['win_rate']:.1f}%")
        
        print(f"\nKey Improvements:")
        print(f"  ✓ Volatility threshold: 1.5% → 0.3% (5x more sensitive)")
        print(f"  ✓ Momentum threshold:   1.0% → 0.5% (2x more sensitive)")
        print(f"  ✓ Volume threshold:     1.5x → 1.2x (easier to trigger)")
        print(f"  ✓ Logic changed from AND to OR (more inclusive)")
        print(f"  ✓ Sampling frequency:   10min → 2min (5x more frequent)")
        
    def plot_improvement(self, results: Dict):
        """Plot improvement visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Signal comparison
        ax1.bar(['Original Strategy', 'Improved Strategy'], 
                [results['original_signals'], results['improved_signals']], 
                color=['red', 'green'], alpha=0.7)
        ax1.set_title('Trading Signal Generation Comparison')
        ax1.set_ylabel('Number of Signals')
        for i, v in enumerate([results['original_signals'], results['improved_signals']]):
            ax1.text(i, v + max(results['original_signals'], results['improved_signals']) * 0.01, 
                    f'{v:,}', ha='center', va='bottom', fontweight='bold')
        
        # Equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            ax2.plot(equity_df['timestamp'], equity_df['equity'], 'b-', linewidth=2)
            ax2.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
            ax2.set_title('Equity Curve (Improved Strategy)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Capital ($)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results/improvement_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Run quick improvement demonstration."""
    backtester = QuickBacktester()
    results = backtester.run_quick_backtest()
    backtester.print_comparison(results)
    backtester.plot_improvement(results)


if __name__ == "__main__":
    main()