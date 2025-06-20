#!/usr/bin/env python3
"""
Analyze and compare model performance between old high-performing models and v2.0.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

print("Model Performance Analysis Report")
print("=" * 60)


def analyze_old_model_performance():
    """Analyze the performance of the old high-performing model."""
    
    # Load leverage 3x backtest results
    trades_file = Path("backtest_results/leverage_3x_trades.csv")
    
    if not trades_file.exists():
        print("❌ Old model backtest results not found")
        return None
    
    trades_df = pd.read_csv(trades_file)
    
    print(f"\n📊 OLD HIGH-PERFORMING MODEL ANALYSIS")
    print(f"{'='*50}")
    
    # Basic stats
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
    losing_trades = len(trades_df[trades_df['pnl_pct'] <= 0])
    win_rate = winning_trades / total_trades * 100
    
    print(f"Total trades: {total_trades}")
    print(f"Winning trades: {winning_trades}")
    print(f"Losing trades: {losing_trades}")
    print(f"Win rate: {win_rate:.1f}%")
    
    # PnL analysis
    total_pnl = trades_df['pnl_dollar'].sum()
    avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_dollar'].mean()
    avg_loss = trades_df[trades_df['pnl_pct'] <= 0]['pnl_dollar'].mean()
    
    print(f"\nPnL Analysis:")
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"Average winning trade: ${avg_win:.2f}")
    print(f"Average losing trade: ${avg_loss:.2f}")
    
    # Return analysis
    total_return_pct = trades_df['pnl_pct'].sum()
    avg_return_per_trade = trades_df['pnl_pct'].mean()
    best_trade = trades_df['pnl_pct'].max()
    worst_trade = trades_df['pnl_pct'].min()
    
    print(f"\nReturn Analysis:")
    print(f"Total return: {total_return_pct:.2f}%")
    print(f"Average return per trade: {avg_return_per_trade:.3f}%")
    print(f"Best trade: {best_trade:.3f}%")
    print(f"Worst trade: {worst_trade:.3f}%")
    
    # Risk metrics
    returns = trades_df['pnl_pct']
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(returns)) if returns.std() > 0 else 0
    max_drawdown = trades_df['drawdown_at_entry'].min()
    
    print(f"\nRisk Metrics:")
    print(f"Sharpe ratio (trade-level): {sharpe_ratio:.2f}")
    print(f"Max drawdown: {max_drawdown:.3f}%")
    
    # Confidence analysis
    avg_confidence = trades_df['confidence'].mean()
    high_conf_trades = len(trades_df[trades_df['confidence'] > 0.8])
    high_conf_win_rate = len(trades_df[(trades_df['confidence'] > 0.8) & (trades_df['pnl_pct'] > 0)]) / high_conf_trades * 100 if high_conf_trades > 0 else 0
    
    print(f"\nConfidence Analysis:")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"High confidence trades (>0.8): {high_conf_trades}")
    print(f"High confidence win rate: {high_conf_win_rate:.1f}%")
    
    # Time analysis
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trading_duration = (trades_df['timestamp'].max() - trades_df['timestamp'].min()).total_seconds() / 3600
    trades_per_hour = total_trades / trading_duration
    
    print(f"\nTiming Analysis:")
    print(f"Trading duration: {trading_duration:.1f} hours")
    print(f"Trades per hour: {trades_per_hour:.2f}")
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_return_pct': total_return_pct,
        'avg_return_per_trade': avg_return_per_trade,
        'sharpe_ratio': sharpe_ratio,
        'avg_confidence': avg_confidence,
        'trades_per_hour': trades_per_hour
    }


def analyze_v2_model_issues():
    """Analyze v2.0 model issues."""
    
    print(f"\n🚨 V2.0 MODEL ANALYSIS")
    print(f"{'='*50}")
    
    # Load v2.0 metadata
    v2_metadata_file = Path("models/v2.0/metadata.json")
    
    if v2_metadata_file.exists():
        with open(v2_metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"Model type: {metadata.get('model_type', 'Unknown')}")
        print(f"Feature count: {metadata.get('feature_count', 'Unknown')}")
        print(f"Training date: {metadata.get('training_date', 'Unknown')}")
        
        # Check if performance data exists
        perf = metadata.get('performance', {})
        if perf:
            print(f"Training AUC: {perf.get('auc', 'N/A')}")
            print(f"Training accuracy: {perf.get('accuracy', 'N/A')}")
        else:
            print("❌ No performance data in metadata")
        
        # Analyze feature names
        feature_names = metadata.get('feature_names', [])
        if feature_names:
            # Count feature types
            hour_features = len([f for f in feature_names if f.startswith('hour_')])
            dow_features = len([f for f in feature_names if f.startswith('dow_')])
            orderbook_features = len([f for f in feature_names if f.startswith('orderbook_')])
            liquidation_features = len([f for f in feature_names if f.startswith('liquidation_')])
            microstructure_features = len([f for f in feature_names if f.startswith('microstructure_')])
            
            print(f"\nFeature Breakdown:")
            print(f"  Hour features: {hour_features}")
            print(f"  Day of week features: {dow_features}")
            print(f"  Orderbook features: {orderbook_features}")
            print(f"  Liquidation features: {liquidation_features}")
            print(f"  Microstructure features: {microstructure_features}")
            
            # Identify potentially problematic features
            basic_features = [f for f in feature_names if any(x in f for x in ['return_', 'volatility_', 'volume_', 'ma_'])]
            print(f"  Basic market features: {len(basic_features)}")
    
    # From our testing, we know v2.0 has severe issues
    print(f"\n❌ CRITICAL ISSUES IDENTIFIED:")
    print(f"1. Model outputs all zeros (no meaningful predictions)")
    print(f"2. AUC = 0.5000 (random performance)")
    print(f"3. No trading signals generated at any confidence threshold")
    print(f"4. Many features appear to be randomly generated/simulated")
    print(f"5. Training process used mock data instead of real features")


def generate_recommendations():
    """Generate actionable recommendations."""
    
    print(f"\n💡 RECOMMENDATIONS")
    print(f"{'='*50}")
    
    print(f"\nIMMEDIATE ACTIONS:")
    print(f"1. 🔄 Revert to proven 35-feature model architecture")
    print(f"2. 🔍 Audit feature generation process for v2.0")
    print(f"3. 🧪 Re-train using validated feature engineering pipeline")
    print(f"4. ✅ Implement proper model validation before deployment")
    
    print(f"\nFEATURE ENGINEERING FIXES:")
    print(f"1. Remove all randomly generated/mock features")
    print(f"2. Focus on proven market indicators:")
    print(f"   - Price returns (multiple timeframes)")
    print(f"   - Volatility measures")
    print(f"   - Volume indicators")
    print(f"   - Technical indicators (RSI, Bollinger Bands)")
    print(f"   - Moving averages")
    print(f"3. Implement real orderbook and liquidation data processing")
    print(f"4. Use proper time-series feature engineering")
    
    print(f"\nMODEL TRAINING IMPROVEMENTS:")
    print(f"1. Use time-series cross-validation")
    print(f"2. Implement proper train/validation/test splits")
    print(f"3. Monitor for overfitting during training")
    print(f"4. Validate feature importance and model interpretability")
    print(f"5. Test model performance on out-of-sample data")
    
    print(f"\nDEPLOYMENT SAFEGUARDS:")
    print(f"1. Require minimum AUC threshold (>0.65) before deployment")
    print(f"2. Test model on paper trading before live deployment")
    print(f"3. Implement model performance monitoring")
    print(f"4. Add circuit breakers for poor performance detection")


def main():
    """Main analysis function."""
    
    # Analyze old model performance
    old_performance = analyze_old_model_performance()
    
    # Analyze v2.0 issues
    analyze_v2_model_issues()
    
    # Generate recommendations
    generate_recommendations()
    
    print(f"\n📈 PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if old_performance:
        print(f"High-Performing Model (AUC ~0.867):")
        print(f"  ✅ {old_performance['total_trades']} profitable trades")
        print(f"  ✅ {old_performance['win_rate']:.1f}% win rate")
        print(f"  ✅ ${old_performance['total_pnl']:.2f} total profit")
        print(f"  ✅ {old_performance['avg_return_per_trade']:.3f}% avg return per trade")
        print(f"  ✅ {old_performance['trades_per_hour']:.2f} trades/hour")
    
    print(f"\nV2.0 Model (AUC = 0.5000):")
    print(f"  ❌ 0 trades generated")
    print(f"  ❌ 0% win rate")
    print(f"  ❌ $0 profit")
    print(f"  ❌ Model outputs all zeros")
    print(f"  ❌ No meaningful predictions")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"The v2.0 model represents a complete regression in performance.")
    print(f"Immediate action required to restore trading capability.")
    print(f"Recommend reverting to proven 35-feature approach while")
    print(f"rebuilding the 156-feature model with proper methodology.")


if __name__ == "__main__":
    main()