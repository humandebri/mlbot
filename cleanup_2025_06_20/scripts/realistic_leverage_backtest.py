#!/usr/bin/env python3
"""
Realistic leverage backtest with balanced risk management.
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


class RealisticLeverageBacktester:
    """Realistic backtest with balanced risk management."""
    
    def __init__(self, leverage: float = 3.0):
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
        
        logger.info(f"Loaded neural network model for realistic {self.leverage}x leverage trading")
    
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
    
    def calculate_position_size(self, confidence: float, recent_volatility: float, 
                               current_drawdown: float, base_size: float = 0.02) -> float:
        """Calculate realistic position size."""
        
        # Base position size (2% with leverage)
        position_size = base_size
        
        # Mild confidence adjustment (0.8x to 1.2x)
        confidence_multiplier = 0.8 + (confidence - 0.5) * 0.8
        position_size *= confidence_multiplier
        
        # Volatility adjustment (mild)
        if recent_volatility > 0.04:  # Very high volatility
            position_size *= 0.8
        elif recent_volatility < 0.015:  # Low volatility
            position_size *= 1.1
        
        # Drawdown adjustment (only in extreme cases)
        if current_drawdown < -10:  # 10% drawdown
            position_size *= 0.7
        elif current_drawdown < -15:  # 15% drawdown
            position_size *= 0.5
        
        # Cap position size
        return min(max(position_size, 0.01), 0.03)  # Between 1% and 3%
    
    def run_realistic_backtest(self, confidence_threshold: float = 0.65):
        """Run realistic leverage backtest."""
        
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
        base_position_size = 0.02  # 2% base
        trades = []
        equity_curve = [initial_capital]
        
        # Balanced risk management
        max_daily_loss = 0.05  # 5% daily stop (more realistic)
        consecutive_losses = 0
        max_consecutive_losses = 10  # Allow more consecutive losses
        daily_trades = 0
        max_daily_trades = 20  # Limit daily trades
        
        # Find high confidence signals
        high_confidence_indices = np.where(predictions > confidence_threshold)[0]
        logger.info(f"Found {len(high_confidence_indices)} signals at {confidence_threshold} threshold")
        
        current_date = None
        
        for signal_idx in high_confidence_indices:
            if signal_idx + 10 >= len(data):
                continue
            
            # Daily trade limit
            trade_date = data.index[signal_idx].date()
            if trade_date != current_date:
                current_date = trade_date
                daily_trades = 0
            
            if daily_trades >= max_daily_trades:
                continue
            
            # Current market state
            entry_price = data['close'].iloc[signal_idx]
            confidence = predictions[signal_idx]
            recent_volatility = data['vol_20'].iloc[signal_idx]
            
            # Calculate current drawdown
            peak_equity = max(equity_curve)
            current_drawdown = (capital - peak_equity) / peak_equity * 100
            
            # Balanced risk checks (less restrictive)
            if consecutive_losses >= max_consecutive_losses:
                consecutive_losses = max_consecutive_losses - 2  # Reset but with penalty
            
            if current_drawdown < -20:  # 20% drawdown circuit breaker
                continue
            
            # Position sizing
            position_size = self.calculate_position_size(
                confidence, recent_volatility, current_drawdown, base_position_size
            )
            
            # Direction based on momentum
            momentum = data['momentum_3'].iloc[signal_idx]
            direction = 'long' if momentum > 0 else 'short'
            
            # Realistic exit strategy
            entry_time = data.index[signal_idx]
            
            best_pnl = float('-inf')
            best_exit_time = None
            best_exit_price = None
            exit_reason = None
            
            # Multiple exit points
            for exit_bars in [3, 5, 7, 10]:
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
                
                # Transaction costs
                leveraged_pnl_pct -= 0.0015  # 0.15% round-trip
                
                # Realistic stop-loss (5% on leveraged position)
                if leveraged_pnl_pct < -0.05:
                    best_pnl = leveraged_pnl_pct
                    best_exit_time = exit_time
                    best_exit_price = exit_price
                    exit_reason = 'stop_loss'
                    break
                
                # Realistic take profit (3% gain)
                if leveraged_pnl_pct > 0.03:
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
            
            # Calculate position value and dollar PnL
            position_value = capital * position_size
            pnl_dollar = position_value * best_pnl
            
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
                'position_size': position_size,
                'pnl_pct': best_pnl * 100,
                'pnl_dollar': pnl_dollar,
                'position_value': position_value,
                'leverage': self.leverage,
                'volatility': recent_volatility,
                'exit_reason': exit_reason
            })
            
            capital += pnl_dollar
            equity_curve.append(capital)
            daily_trades += 1
            
            # Daily loss check (less restrictive)
            daily_loss_pct = (capital - initial_capital) / initial_capital
            if daily_loss_pct < -max_daily_loss:
                logger.warning(f"Daily loss limit approached: {daily_loss_pct:.2%}")
        
        return trades, equity_curve
    
    def analyze_results(self, trades: list, equity_curve: list):
        """Analyze realistic trading results."""
        
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
        
        # Exit reason analysis
        exit_reasons = trades_df['exit_reason'].value_counts()
        
        results = {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'total_pnl': total_pnl,
            'final_capital': final_capital,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades_df': trades_df,
            'equity_curve': equity_curve,
            'exit_reasons': exit_reasons
        }
        
        return results
    
    def print_realistic_results(self, results: dict):
        """Print realistic trading results."""
        
        if not results:
            print("❌ 取引が実行されませんでした")
            return
        
        print("\n" + "="*80)
        print(f"📊 現実的なレバレッジ{self.leverage}倍バックテスト結果 (3ヶ月)")
        print("="*80)
        
        print(f"\n📈 パフォーマンスサマリー:")
        print(f"  💰 総収益率: {results['total_return']:.2f}%")
        print(f"  📅 月次収益率: {results['total_return']/3:.2f}%")
        print(f"  🚀 年次換算収益率: {results['total_return']/3*12:.2f}%")
        print(f"  💵 総利益: ${results['total_pnl']:,.2f}")
        print(f"  💲 最終資本: ${results['final_capital']:,.2f}")
        
        print(f"\n📊 取引統計:")
        print(f"  📊 総取引数: {results['num_trades']}")
        print(f"  🎯 勝率: {results['win_rate']:.1f}%")
        print(f"  💲 平均取引損益: ${results['avg_trade']:.2f}")
        print(f"  📊 Sharpe比: {results['sharpe_ratio']:.2f}")
        print(f"  📉 最大ドローダウン: {results['max_drawdown']:.2f}%")
        
        print(f"\n🔄 決済理由分析:")
        for reason, count in results['exit_reasons'].items():
            percentage = count / results['num_trades'] * 100
            print(f"  {reason}: {count}件 ({percentage:.1f}%)")
        
        # Risk assessment
        monthly_return = results['total_return'] / 3
        
        print(f"\n🎯 収益性評価:")
        if monthly_return > 2:
            assessment = "🚀 優秀！レバレッジ活用成功"
        elif monthly_return > 1:
            assessment = "✅ 良好！実用可能"
        elif monthly_return > 0.5:
            assessment = "⚠️ 改善の余地あり"
        elif monthly_return > 0:
            assessment = "📊 微益 - リスク管理要改善"
        else:
            assessment = "❌ 損失 - 戦略見直し必要"
        
        print(f"  {assessment}")
        
        print(f"\n⚡ レバレッジ効果分析:")
        base_return = results['total_return'] / self.leverage
        print(f"  レバレッジなし推定収益: {base_return:.2f}%")
        print(f"  レバレッジ効果倍率: {self.leverage:.1f}x")
        
        if results['max_drawdown'] < -10:
            print(f"\n⚠️ リスク警告:")
            print(f"  高いドローダウン（{results['max_drawdown']:.1f}%）に注意")
        
        return results


def main():
    """Run realistic leverage backtest."""
    
    logger.info("Starting realistic leverage backtest analysis")
    
    print("="*80)
    print("🔬 現実的なレバレッジバックテスト分析")
    print("="*80)
    
    # Test with different confidence thresholds
    thresholds = [0.6, 0.65, 0.7]
    all_results = {}
    
    for threshold in thresholds:
        print(f"\n📊 テスト中: 信頼度閾値 {threshold}")
        
        backtester = RealisticLeverageBacktester(leverage=3.0)
        trades, equity_curve = backtester.run_realistic_backtest(confidence_threshold=threshold)
        results = backtester.analyze_results(trades, equity_curve)
        
        if results:
            print(f"  収益率: {results['total_return']:.2f}% | 取引数: {results['num_trades']} | 勝率: {results['win_rate']:.1f}%")
            all_results[threshold] = results
    
    # Find best threshold
    if all_results:
        best_threshold = max(all_results.keys(), key=lambda k: all_results[k]['total_return'])
        best_results = all_results[best_threshold]
        
        print(f"\n{'='*80}")
        print(f"🏆 最適結果詳細: 信頼度閾値 {best_threshold}")
        print(f"{'='*80}")
        
        backtester = RealisticLeverageBacktester(leverage=3.0)
        backtester.print_realistic_results(best_results)
        
        # Plot comparison
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Equity curves comparison
        plt.subplot(2, 2, 1)
        for threshold, results in all_results.items():
            equity = results['equity_curve']
            trades_df = results['trades_df']
            dates = pd.date_range(start='2024-05-01', periods=len(equity), freq='5min')[:len(equity)]
            plt.plot(dates, equity, label=f'Threshold {threshold}', linewidth=2)
        plt.axhline(y=100000, color='gray', linestyle='--', alpha=0.7)
        plt.title('エクイティカーブ比較')
        plt.xlabel('Date')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Return distribution
        plt.subplot(2, 2, 2)
        best_trades = best_results['trades_df']
        plt.hist(best_trades['pnl_pct'], bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title(f'収益分布 (閾値 {best_threshold})')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Win rate by month
        plt.subplot(2, 2, 3)
        best_trades['month'] = pd.to_datetime(best_trades['timestamp']).dt.to_period('M')
        monthly_stats = best_trades.groupby('month').agg({
            'pnl_dollar': lambda x: (x > 0).mean() * 100
        })
        monthly_stats.plot(kind='bar')
        plt.title('月別勝率')
        plt.xlabel('Month')
        plt.ylabel('Win Rate (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Position size distribution
        plt.subplot(2, 2, 4)
        plt.scatter(best_trades.index, best_trades['position_size'], alpha=0.6)
        plt.axhline(y=0.02, color='red', linestyle='--', label='Base size')
        plt.title('ポジションサイズ推移')
        plt.xlabel('Trade Number')
        plt.ylabel('Position Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results/realistic_leverage_analysis.png', dpi=300, bbox_inches='tight')
        
        # Save best results
        best_trades.to_csv('backtest_results/realistic_leverage_trades.csv', index=False)
        
        print(f"\n💾 保存完了:")
        print(f"  取引データ: backtest_results/realistic_leverage_trades.csv")
        print(f"  分析チャート: backtest_results/realistic_leverage_analysis.png")
    
    else:
        print("❌ 有効な結果が得られませんでした")


if __name__ == "__main__":
    main()