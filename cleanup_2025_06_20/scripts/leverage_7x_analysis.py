#!/usr/bin/env python3
"""
Leverage 7x analysis with enhanced risk management for higher leverage.
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


class HighLeverageAnalyzer:
    """Analyze trading performance with 7x leverage."""
    
    def __init__(self, leverage: float = 7.0):
        self.leverage = leverage
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
        
        logger.info(f"Loaded neural network model for {self.leverage}x leverage analysis")
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data."""
        
        conn = duckdb.connect("data/historical_data.duckdb")
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
        
        logger.info(f"Loaded {len(data)} test records")
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
    
    def calculate_position_size_7x(self, confidence: float, recent_volatility: float, 
                                   current_drawdown: float, base_size: float = 0.015) -> float:
        """Calculate position size for 7x leverage (more conservative)."""
        
        # Reduced base position size for higher leverage (1.5%)
        position_size = base_size
        
        # Conservative confidence adjustment (0.7x to 1.3x)
        confidence_multiplier = 0.7 + (confidence - 0.5) * 1.2
        position_size *= confidence_multiplier
        
        # Stronger volatility adjustment for 7x leverage
        if recent_volatility > 0.035:  # High volatility
            position_size *= 0.6  # More conservative
        elif recent_volatility > 0.025:
            position_size *= 0.8
        elif recent_volatility < 0.015:  # Low volatility
            position_size *= 1.15
        
        # Aggressive drawdown adjustment for capital preservation
        if current_drawdown < -5:  # 5% drawdown
            position_size *= 0.7
        elif current_drawdown < -8:  # 8% drawdown
            position_size *= 0.5
        elif current_drawdown < -12:  # 12% drawdown
            position_size *= 0.3
        
        # Tighter position size limits for 7x leverage
        return min(max(position_size, 0.005), 0.02)  # Between 0.5% and 2%
    
    def run_7x_backtest(self, confidence_threshold: float = 0.65):
        """Run 7x leverage backtest with enhanced risk management."""
        
        # Load model and data
        self.load_model()
        data = self.load_test_data()
        
        # Engineer features
        data = self.engineer_features(data)
        
        # Generate predictions
        predictions = self.predict_signals(data)
        
        # Initialize trading variables
        initial_capital = 100000
        capital = initial_capital
        base_position_size = 0.015  # 1.5% base (conservative for 7x)
        trades = []
        equity_curve = [initial_capital]
        
        # Enhanced risk management for 7x leverage
        max_daily_loss = 0.07  # 7% daily stop (1% per leverage unit)
        consecutive_losses = 0
        max_consecutive_losses = 8  # Tighter limit
        daily_trades = 0
        max_daily_trades = 15  # Fewer trades for risk control
        
        # Additional risk metrics
        daily_pnl = 0
        max_position_value = capital * 0.15  # Max 15% in any position
        
        # Find high confidence signals
        high_confidence_indices = np.where(predictions > confidence_threshold)[0]
        logger.info(f"Found {len(high_confidence_indices)} signals at {confidence_threshold} threshold")
        
        current_date = None
        
        for signal_idx in high_confidence_indices:
            if signal_idx + 10 >= len(data):
                continue
            
            # Daily reset
            trade_date = data.index[signal_idx].date()
            if trade_date != current_date:
                current_date = trade_date
                daily_trades = 0
                daily_pnl = 0
            
            # Daily limits
            if daily_trades >= max_daily_trades:
                continue
            
            if daily_pnl < -max_daily_loss * initial_capital:
                continue  # Daily stop hit
            
            # Current market state
            entry_price = data['close'].iloc[signal_idx]
            confidence = predictions[signal_idx]
            recent_volatility = data['vol_20'].iloc[signal_idx]
            
            # Calculate current drawdown
            peak_equity = max(equity_curve)
            current_drawdown = (capital - peak_equity) / peak_equity * 100
            
            # Enhanced risk checks for 7x leverage
            if consecutive_losses >= max_consecutive_losses:
                consecutive_losses = max_consecutive_losses - 3  # Stronger penalty
            
            if current_drawdown < -15:  # 15% drawdown circuit breaker for 7x
                logger.warning(f"Circuit breaker triggered at {current_drawdown:.1f}% drawdown")
                break  # Stop trading
            
            # Position sizing
            position_size = self.calculate_position_size_7x(
                confidence, recent_volatility, current_drawdown, base_position_size
            )
            
            # Ensure position value doesn't exceed max
            position_value = min(capital * position_size, max_position_value)
            actual_position_size = position_value / capital
            
            # Direction based on momentum
            momentum = data['momentum_3'].iloc[signal_idx]
            direction = 'long' if momentum > 0 else 'short'
            
            # Enhanced exit strategy for 7x leverage
            entry_time = data.index[signal_idx]
            
            best_pnl = float('-inf')
            best_exit_time = None
            best_exit_price = None
            exit_reason = None
            
            # Tighter exit points for 7x leverage
            for exit_bars in [2, 3, 4, 5, 7]:
                if signal_idx + exit_bars >= len(data):
                    continue
                    
                exit_price = data['close'].iloc[signal_idx + exit_bars]
                exit_time = data.index[signal_idx + exit_bars]
                
                # Calculate raw PnL
                if direction == 'long':
                    raw_pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    raw_pnl_pct = (entry_price - exit_price) / entry_price
                
                # Apply leverage
                leveraged_pnl_pct = raw_pnl_pct * self.leverage
                
                # Higher transaction costs for 7x leverage
                leveraged_pnl_pct -= 0.002  # 0.2% round-trip
                
                # Tighter stop-loss for 7x (2.5% on leveraged position)
                if leveraged_pnl_pct < -0.025:
                    best_pnl = leveraged_pnl_pct
                    best_exit_time = exit_time
                    best_exit_price = exit_price
                    exit_reason = 'stop_loss'
                    break
                
                # Take profit earlier for 7x (2.5% gain)
                if leveraged_pnl_pct > 0.025:
                    best_pnl = leveraged_pnl_pct
                    best_exit_time = exit_time
                    best_exit_price = exit_price
                    exit_reason = 'take_profit'
                    break
                
                # Track best exit
                if leveraged_pnl_pct > best_pnl:
                    best_pnl = leveraged_pnl_pct
                    best_exit_time = exit_time
                    best_exit_price = exit_price
                    exit_reason = 'time_exit'
            
            # Calculate dollar PnL
            pnl_dollar = position_value * best_pnl
            
            # Risk check before adding trade
            if abs(pnl_dollar) > capital * 0.1:  # Single trade risk limit
                continue
            
            # Update consecutive losses counter
            if best_pnl < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
            
            trades.append({
                'timestamp': entry_time,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': best_exit_price,
                'exit_time': best_exit_time,
                'confidence': confidence,
                'position_size': actual_position_size,
                'pnl_pct': best_pnl * 100,
                'pnl_dollar': pnl_dollar,
                'position_value': position_value,
                'leverage': self.leverage,
                'volatility': recent_volatility,
                'exit_reason': exit_reason,
                'drawdown_at_entry': current_drawdown
            })
            
            capital += pnl_dollar
            equity_curve.append(capital)
            daily_trades += 1
            daily_pnl += pnl_dollar
        
        return trades, equity_curve
    
    def analyze_results(self, trades: list, equity_curve: list):
        """Analyze 7x leverage trading results."""
        
        if not trades:
            return None
        
        trades_df = pd.DataFrame(trades)
        initial_capital = 100000
        final_capital = equity_curve[-1]
        
        # Basic metrics
        total_return = (final_capital - initial_capital) / initial_capital * 100
        num_trades = len(trades_df)
        win_rate = len(trades_df[trades_df['pnl_dollar'] > 0]) / num_trades * 100
        avg_trade = trades_df['pnl_dollar'].mean()
        total_pnl = trades_df['pnl_dollar'].sum()
        
        # Risk metrics
        returns = trades_df['pnl_pct'].values / 100
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252/5)
        else:
            sharpe_ratio = 0
        
        # Drawdown analysis
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / initial_capital * 100
        max_drawdown = drawdown.min()
        
        # Additional risk metrics
        avg_winner = trades_df[trades_df['pnl_dollar'] > 0]['pnl_dollar'].mean() if len(trades_df[trades_df['pnl_dollar'] > 0]) > 0 else 0
        avg_loser = trades_df[trades_df['pnl_dollar'] < 0]['pnl_dollar'].mean() if len(trades_df[trades_df['pnl_dollar'] < 0]) > 0 else 0
        profit_factor = abs(avg_winner / avg_loser) if avg_loser != 0 else 0
        
        # Exit reason analysis
        exit_reasons = trades_df['exit_reason'].value_counts()
        
        # Calculate Calmar ratio
        calmar_ratio = (total_return / 3 * 12) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        results = {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'total_pnl': total_pnl,
            'final_capital': final_capital,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'trades_df': trades_df,
            'equity_curve': equity_curve,
            'exit_reasons': exit_reasons
        }
        
        return results
    
    def print_7x_results(self, results: dict):
        """Print comprehensive 7x leverage results."""
        
        if not results:
            print("❌ 取引が実行されませんでした")
            return
        
        print("\n" + "="*80)
        print(f"🚀 レバレッジ{self.leverage}倍 詳細分析結果 (3ヶ月)")
        print("="*80)
        
        print(f"\n💰 収益性メトリクス:")
        print(f"  総収益率: {results['total_return']:.2f}%")
        print(f"  月次収益率: {results['total_return']/3:.2f}%")
        print(f"  年次換算収益率: {results['total_return']/3*12:.2f}%")
        print(f"  総利益: ${results['total_pnl']:,.2f}")
        print(f"  最終資本: ${results['final_capital']:,.2f}")
        
        print(f"\n📊 取引統計:")
        print(f"  総取引数: {results['num_trades']}")
        print(f"  勝率: {results['win_rate']:.1f}%")
        print(f"  平均取引損益: ${results['avg_trade']:.2f}")
        print(f"  平均勝ち取引: ${results['avg_winner']:.2f}")
        print(f"  平均負け取引: ${results['avg_loser']:.2f}")
        print(f"  プロフィットファクター: {results['profit_factor']:.2f}")
        
        print(f"\n⚡ リスクメトリクス:")
        print(f"  Sharpe比: {results['sharpe_ratio']:.2f}")
        print(f"  Calmar比: {results['calmar_ratio']:.2f}")
        print(f"  最大ドローダウン: {results['max_drawdown']:.2f}%")
        
        print(f"\n🔄 決済理由分析:")
        for reason, count in results['exit_reasons'].items():
            percentage = count / results['num_trades'] * 100
            print(f"  {reason}: {count}件 ({percentage:.1f}%)")
        
        # Risk assessment
        monthly_return = results['total_return'] / 3
        
        print(f"\n🎯 総合評価:")
        if monthly_return > 5 and results['max_drawdown'] > -10:
            assessment = "🎉 非常に優秀！高レバレッジ活用成功"
        elif monthly_return > 3 and results['max_drawdown'] > -15:
            assessment = "🚀 優秀！実用化推奨"
        elif monthly_return > 1.5:
            assessment = "✅ 良好だがリスク注意"
        elif monthly_return > 0:
            assessment = "⚠️ 微益 - リスクに見合わない可能性"
        else:
            assessment = "❌ 損失 - レバレッジ過大"
        
        print(f"  {assessment}")
        
        print(f"\n⚠️ 高レバレッジリスク分析:")
        if results['max_drawdown'] < -10:
            print(f"  🔴 高いドローダウン（{results['max_drawdown']:.1f}%）- 資金管理重要")
        if results['sharpe_ratio'] < 1.5:
            print(f"  🔴 リスク調整後リターンが低い（Sharpe: {results['sharpe_ratio']:.2f}）")
        if results['profit_factor'] < 1.5:
            print(f"  🔴 プロフィットファクターが低い（{results['profit_factor']:.2f}）")
        
        # Compare with 3x leverage
        print(f"\n📊 レバレッジ比較:")
        print(f"  3倍時の月次収益（実績）: 1.90%")
        print(f"  7倍時の月次収益: {monthly_return:.2f}%")
        print(f"  収益向上率: {(monthly_return/1.90):.1f}x")
        
        return results


def main():
    """Run comprehensive 7x leverage analysis."""
    
    logger.info("Starting 7x leverage analysis")
    
    print("="*80)
    print("🔬 レバレッジ7倍 詳細分析")
    print("="*80)
    
    # Test multiple confidence thresholds
    thresholds = [0.6, 0.65, 0.7, 0.75]
    all_results = {}
    
    for threshold in thresholds:
        print(f"\n📊 テスト中: 信頼度閾値 {threshold}")
        
        analyzer = HighLeverageAnalyzer(leverage=7.0)
        trades, equity_curve = analyzer.run_7x_backtest(confidence_threshold=threshold)
        results = analyzer.analyze_results(trades, equity_curve)
        
        if results:
            print(f"  収益率: {results['total_return']:.2f}% | 取引数: {results['num_trades']} | DD: {results['max_drawdown']:.1f}%")
            all_results[threshold] = results
    
    # Find best threshold
    if all_results:
        # Select best based on risk-adjusted return (Calmar ratio)
        best_threshold = max(all_results.keys(), 
                           key=lambda k: all_results[k]['calmar_ratio'] if all_results[k]['calmar_ratio'] > 0 else -100)
        best_results = all_results[best_threshold]
        
        print(f"\n{'='*80}")
        print(f"🏆 最適結果詳細: 信頼度閾値 {best_threshold}")
        print(f"{'='*80}")
        
        analyzer = HighLeverageAnalyzer(leverage=7.0)
        analyzer.print_7x_results(best_results)
        
        # Visualization
        plt.figure(figsize=(20, 12))
        
        # Subplot 1: Equity curves for different thresholds
        plt.subplot(2, 3, 1)
        for threshold, results in all_results.items():
            equity = results['equity_curve']
            dates = pd.date_range(start='2024-05-01', periods=len(equity), freq='5min')[:len(equity)]
            plt.plot(dates, equity, label=f'Threshold {threshold}', linewidth=2)
        plt.axhline(y=100000, color='gray', linestyle='--', alpha=0.7)
        plt.title('レバレッジ7倍 エクイティカーブ比較')
        plt.xlabel('Date')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Drawdown analysis
        plt.subplot(2, 3, 2)
        equity_series = pd.Series(best_results['equity_curve'])
        running_max = equity_series.expanding().max()
        drawdown_pct = (equity_series - running_max) / running_max * 100
        dates = pd.date_range(start='2024-05-01', periods=len(drawdown_pct), freq='5min')[:len(drawdown_pct)]
        plt.fill_between(dates, drawdown_pct, 0, color='red', alpha=0.3)
        plt.plot(dates, drawdown_pct, color='red', linewidth=1)
        plt.title('ドローダウン推移')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Return distribution
        plt.subplot(2, 3, 3)
        best_trades = best_results['trades_df']
        plt.hist(best_trades['pnl_pct'], bins=50, edgecolor='black', alpha=0.7, color='green')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.title(f'収益分布 (7倍レバレッジ)')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Risk-Return comparison
        plt.subplot(2, 3, 4)
        leverages = []
        returns = []
        drawdowns = []
        sharpes = []
        
        # Add 3x results (from previous test)
        leverages.append(3)
        returns.append(1.90 * 3)  # 3-month return
        drawdowns.append(-0.18)
        sharpes.append(2.39)
        
        # Add 7x results
        for threshold, results in all_results.items():
            leverages.append(7)
            returns.append(results['total_return'])
            drawdowns.append(results['max_drawdown'])
            sharpes.append(results['sharpe_ratio'])
        
        scatter = plt.scatter(drawdowns, returns, c=sharpes, s=[l*20 for l in leverages], 
                            cmap='viridis', alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, label='Sharpe Ratio')
        plt.xlabel('Max Drawdown (%)')
        plt.ylabel('Total Return (%)')
        plt.title('リスク・リターン分析')
        plt.grid(True, alpha=0.3)
        
        # Subplot 5: Position size distribution
        plt.subplot(2, 3, 5)
        plt.scatter(best_trades.index, best_trades['position_size'] * 100, 
                   c=best_trades['volatility'], cmap='coolwarm', alpha=0.6)
        plt.colorbar(label='Volatility')
        plt.axhline(y=1.5, color='red', linestyle='--', label='Base size')
        plt.title('ポジションサイズ vs ボラティリティ')
        plt.xlabel('Trade Number')
        plt.ylabel('Position Size (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 6: Monthly performance
        plt.subplot(2, 3, 6)
        best_trades['month'] = pd.to_datetime(best_trades['timestamp']).dt.to_period('M')
        monthly_pnl = best_trades.groupby('month')['pnl_dollar'].sum()
        monthly_pnl.plot(kind='bar', color=['green' if x > 0 else 'red' for x in monthly_pnl])
        plt.title('月別損益')
        plt.xlabel('Month')
        plt.ylabel('PnL ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results/leverage_7x_analysis.png', dpi=300, bbox_inches='tight')
        
        # Save results
        best_trades.to_csv('backtest_results/leverage_7x_trades.csv', index=False)
        
        # Summary comparison table
        print(f"\n📊 レバレッジ比較サマリー:")
        print(f"{'指標':<20} {'3倍':<15} {'7倍 (最適)':<15} {'改善率':<10}")
        print("-" * 60)
        print(f"{'月次収益率':<20} {'1.90%':<15} {f'{best_results["total_return"]/3:.2f}%':<15} {f'{(best_results["total_return"]/3/1.90):.1f}x':<10}")
        print(f"{'Sharpe比':<20} {'2.39':<15} {f'{best_results["sharpe_ratio"]:.2f}':<15} {f'{(best_results["sharpe_ratio"]/2.39):.1f}x':<10}")
        print(f"{'最大DD':<20} {'-0.18%':<15} {f'{best_results["max_drawdown"]:.2f}%':<15} {f'{abs(best_results["max_drawdown"]/0.18):.1f}x':<10}")
        print(f"{'取引数':<20} {'697':<15} {f'{best_results["num_trades"]}':<15} {f'{(best_results["num_trades"]/697):.1f}x':<10}")
        
        print(f"\n💾 保存完了:")
        print(f"  取引データ: backtest_results/leverage_7x_trades.csv")
        print(f"  分析チャート: backtest_results/leverage_7x_analysis.png")
        
        return all_results
    
    else:
        print("❌ 有効な結果が得られませんでした")
        return None


if __name__ == "__main__":
    main()