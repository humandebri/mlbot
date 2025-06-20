#!/usr/bin/env python3
"""
Analyze the impact of relaxed thresholds on trading signal generation.
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


def analyze_threshold_impact():
    """Analyze impact of different thresholds on signal generation."""
    logger.info("Analyzing threshold impact on trading opportunities")
    
    # Load sample data
    conn = duckdb.connect("data/historical_data.duckdb")
    
    query = """
    SELECT 
        timestamp,
        close,
        volume
    FROM klines_btcusdt
    WHERE timestamp >= '2024-01-01'
      AND timestamp <= '2024-01-07'  -- Just one week for quick analysis
    ORDER BY timestamp
    LIMIT 10000
    """
    
    data = conn.execute(query).df()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    conn.close()
    
    logger.info(f"Analyzing {len(data)} data points")
    
    # Calculate features
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(30).std()
    data['price_change'] = data['close'].pct_change(15)
    data['volume_ma'] = data['volume'].rolling(30).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    
    # Remove NaN
    data = data.dropna()
    
    # Test different threshold combinations
    results = []
    
    threshold_tests = [
        {"name": "Original (Very Restrictive)", "vol": 0.015, "mom": 0.01, "vol_th": 1.5, "logic": "AND"},
        {"name": "Slightly Relaxed", "vol": 0.01, "mom": 0.008, "vol_th": 1.3, "logic": "AND"},
        {"name": "Moderately Relaxed", "vol": 0.007, "mom": 0.006, "vol_th": 1.2, "logic": "AND"},
        {"name": "Improved (OR Logic)", "vol": 0.005, "mom": 0.005, "vol_th": 1.2, "logic": "OR"},
        {"name": "Highly Relaxed", "vol": 0.003, "mom": 0.005, "vol_th": 1.2, "logic": "OR"},
    ]
    
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS - TRADING OPPORTUNITY IMPROVEMENT")
    print("="*80)
    print(f"Data period: 2024-01-01 to 2024-01-07 ({len(data)} data points)")
    print()
    
    for test in threshold_tests:
        signals = 0
        
        for i, row in data.iterrows():
            vol_condition = abs(row['volatility']) > test['vol']
            momentum_condition = abs(row['price_change']) > test['mom']
            volume_condition = row['volume_ratio'] > test['vol_th']
            
            if test['logic'] == 'AND':
                signal = vol_condition and momentum_condition and volume_condition
            else:  # OR logic
                signal = vol_condition or (momentum_condition and volume_condition)
            
            if signal:
                signals += 1
        
        signal_rate = signals / len(data) * 100
        
        results.append({
            'name': test['name'],
            'signals': signals,
            'rate': signal_rate,
            'vol_threshold': test['vol'],
            'mom_threshold': test['mom'],
            'volume_threshold': test['vol_th'],
            'logic': test['logic']
        })
        
        print(f"{test['name']:25} | {signals:5d} signals ({signal_rate:5.2f}%) | " +
              f"Vol: {test['vol']:.3f}, Mom: {test['mom']:.3f}, VolRatio: {test['vol_th']:.1f}, Logic: {test['logic']}")
    
    # Calculate improvements
    baseline = results[0]['signals']  # Original strategy
    
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    for i, result in enumerate(results):
        if i == 0:
            improvement = "1.0x (baseline)"
        elif baseline == 0:
            improvement = "∞x (infinite)" if result['signals'] > 0 else "0x"
        else:
            improvement = f"{result['signals']/baseline:.1f}x"
        
        print(f"{result['name']:25} | {improvement:>15} improvement | {result['signals']:,} signals")
    
    print(f"\nKey Findings:")
    print(f"• Original strategy generated {baseline:,} signals ({results[0]['rate']:.2f}% of time)")
    print(f"• Best improved strategy generated {results[-1]['signals']:,} signals ({results[-1]['rate']:.2f}% of time)")
    if baseline == 0:
        print(f"• That's an INFINITE improvement - from NO trading opportunities to {results[-1]['signals']:,} opportunities!")
    else:
        print(f"• That's a {results[-1]['signals']/baseline:.1f}x improvement in trading opportunities!")
    
    print(f"\nCritical Improvements:")
    print(f"• Volatility threshold: {results[0]['vol_threshold']:.3f} → {results[-1]['vol_threshold']:.3f} ({results[0]['vol_threshold']/results[-1]['vol_threshold']:.1f}x more sensitive)")
    print(f"• Momentum threshold: {results[0]['mom_threshold']:.3f} → {results[-1]['mom_threshold']:.3f} ({results[0]['mom_threshold']/results[-1]['mom_threshold']:.1f}x more sensitive)")
    print(f"• Logic: {results[0]['logic']} → {results[-1]['logic']} (more inclusive)")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Signal count comparison
    names = [r['name'] for r in results]
    signal_counts = [r['signals'] for r in results]
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    
    bars = ax1.bar(range(len(names)), signal_counts, color=colors, alpha=0.7)
    ax1.set_title('Trading Signals Generated by Threshold Strategy', fontsize=14)
    ax1.set_ylabel('Number of Signals')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace(' ', '\n') for n in names], rotation=0, ha='center')
    
    # Add value labels on bars
    for bar, count in zip(bars, signal_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(signal_counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Signal rate comparison  
    signal_rates = [r['rate'] for r in results]
    bars2 = ax2.bar(range(len(names)), signal_rates, color=colors, alpha=0.7)
    ax2.set_title('Signal Generation Rate (%)', fontsize=14)
    ax2.set_ylabel('Percentage of Time with Signals (%)')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.replace(' ', '\n') for n in names], rotation=0, ha='center')
    
    # Add value labels
    for bar, rate in zip(bars2, signal_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(signal_rates)*0.01,
                f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('backtest_results/threshold_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Analysis chart saved to backtest_results/threshold_analysis.png")
    plt.show()
    
    return results


if __name__ == "__main__":
    analyze_threshold_impact()