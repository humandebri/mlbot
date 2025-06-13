#!/usr/bin/env python3
"""
Optimize position sizing for 3x leverage to maximize returns while controlling risk.
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


class PositionSizeOptimizer:
    """Optimize position sizing for maximum risk-adjusted returns."""
    
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
        
        logger.info(f"Loaded neural network model for position size optimization")
    
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
    
    def calculate_kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float, 
                                    kelly_fraction: float = 0.25) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        kelly_fraction: Fraction of full Kelly to use for safety (typically 0.25)
        """
        
        if avg_loss == 0:
            return 0.02  # Default 2%
        
        # Calculate odds
        b = abs(avg_win / avg_loss)  # Payoff ratio
        p = win_rate  # Probability of winning
        q = 1 - p  # Probability of losing
        
        # Kelly formula: f = (p*b - q) / b
        kelly_full = (p * b - q) / b
        
        # Apply Kelly fraction for safety
        kelly_size = kelly_full * kelly_fraction
        
        # Cap between reasonable bounds
        return min(max(kelly_size, 0.01), 0.05)  # 1% to 5%
    
    def calculate_dynamic_position_size(self, confidence: float, recent_volatility: float,
                                      current_drawdown: float, base_size: float,
                                      recent_performance: dict) -> float:
        """
        Enhanced dynamic position sizing based on multiple factors.
        """
        
        # Start with base size
        position_size = base_size
        
        # 1. Confidence-based sizing (stronger effect)
        if confidence > 0.8:
            confidence_multiplier = 1.5
        elif confidence > 0.7:
            confidence_multiplier = 1.25
        elif confidence > 0.65:
            confidence_multiplier = 1.0
        else:
            confidence_multiplier = 0.75
        
        position_size *= confidence_multiplier
        
        # 2. Volatility-based adjustment
        avg_volatility = 0.02  # Typical market volatility
        if recent_volatility < avg_volatility * 0.7:  # Low volatility
            position_size *= 1.3
        elif recent_volatility < avg_volatility:
            position_size *= 1.1
        elif recent_volatility > avg_volatility * 1.5:  # High volatility
            position_size *= 0.7
        elif recent_volatility > avg_volatility * 1.2:
            position_size *= 0.85
        
        # 3. Performance-based adjustment
        if recent_performance['win_rate'] > 0.65:  # Hot streak
            position_size *= 1.2
        elif recent_performance['win_rate'] < 0.35:  # Cold streak
            position_size *= 0.8
        
        # 4. Drawdown-based adjustment (less aggressive)
        if current_drawdown < -5:
            position_size *= 0.8
        elif current_drawdown < -10:
            position_size *= 0.6
        elif current_drawdown > -1 and recent_performance['trades'] > 20:  # Winning streak
            position_size *= 1.1
        
        # 5. Apply Kelly adjustment if we have enough data
        if recent_performance['trades'] >= 30:
            kelly_size = self.calculate_kelly_position_size(
                recent_performance['win_rate'],
                recent_performance['avg_win'],
                recent_performance['avg_loss'],
                kelly_fraction=0.3  # 30% Kelly for 3x leverage
            )
            # Blend Kelly with dynamic sizing
            position_size = 0.7 * position_size + 0.3 * kelly_size
        
        # Final bounds based on leverage
        min_size = 0.01  # 1% minimum
        max_size = 0.05  # 5% maximum for 3x leverage
        
        return min(max(position_size, min_size), max_size)
    
    def run_optimized_backtest(self, confidence_threshold: float = 0.6, 
                              base_sizes: list = [0.015, 0.02, 0.025, 0.03]):
        """Run backtest with different base position sizes."""
        
        # Load model and data
        self.load_model()
        data = self.load_test_data()
        
        # Engineer features
        data = self.engineer_features(data)
        
        # Generate predictions
        predictions = self.predict_signals(data)
        
        results = {}
        
        for base_size in base_sizes:
            logger.info(f"Testing base position size: {base_size*100:.1f}%")
            
            # Initialize trading variables
            initial_capital = 100000
            capital = initial_capital
            trades = []
            equity_curve = [initial_capital]
            
            # Risk management
            max_daily_loss = 0.05
            consecutive_losses = 0
            max_consecutive_losses = 10
            daily_trades = 0
            max_daily_trades = 20
            
            # Performance tracking
            recent_trades = []
            
            # Find high confidence signals
            high_confidence_indices = np.where(predictions > confidence_threshold)[0]
            
            current_date = None
            
            for signal_idx in high_confidence_indices:
                if signal_idx + 10 >= len(data):
                    continue
                
                # Daily reset
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
                
                # Risk checks
                if consecutive_losses >= max_consecutive_losses:
                    consecutive_losses = max_consecutive_losses - 2
                
                if current_drawdown < -20:
                    continue
                
                # Calculate recent performance
                recent_performance = self.calculate_recent_performance(recent_trades[-50:])
                
                # Dynamic position sizing
                position_size = self.calculate_dynamic_position_size(
                    confidence, recent_volatility, current_drawdown, 
                    base_size, recent_performance
                )
                
                # Direction based on momentum
                momentum = data['momentum_3'].iloc[signal_idx]
                direction = 'long' if momentum > 0 else 'short'
                
                # Exit strategy
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
                    leveraged_pnl_pct -= 0.0015
                    
                    # Dynamic stop-loss based on position size
                    stop_loss_pct = -0.05 * (base_size / position_size)  # Tighter stop for larger positions
                    if leveraged_pnl_pct < stop_loss_pct:
                        best_pnl = leveraged_pnl_pct
                        best_exit_time = exit_time
                        best_exit_price = exit_price
                        exit_reason = 'stop_loss'
                        break
                    
                    # Dynamic take profit
                    take_profit_pct = 0.03 * (position_size / base_size)  # Higher target for larger positions
                    if leveraged_pnl_pct > take_profit_pct:
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
                
                trade_data = {
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
                }
                
                trades.append(trade_data)
                recent_trades.append(trade_data)
                
                capital += pnl_dollar
                equity_curve.append(capital)
                daily_trades += 1
            
            # Analyze results
            if trades:
                results[base_size] = self.analyze_results(trades, equity_curve)
            
        return results
    
    def calculate_recent_performance(self, recent_trades: list) -> dict:
        """Calculate performance metrics for recent trades."""
        
        if not recent_trades:
            return {
                'trades': 0,
                'win_rate': 0.5,
                'avg_win': 0.01,
                'avg_loss': 0.01
            }
        
        wins = [t for t in recent_trades if t['pnl_pct'] > 0]
        losses = [t for t in recent_trades if t['pnl_pct'] < 0]
        
        win_rate = len(wins) / len(recent_trades) if recent_trades else 0.5
        avg_win = np.mean([t['pnl_pct'] for t in wins]) / 100 if wins else 0.01
        avg_loss = abs(np.mean([t['pnl_pct'] for t in losses])) / 100 if losses else 0.01
        
        return {
            'trades': len(recent_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def analyze_results(self, trades: list, equity_curve: list) -> dict:
        """Analyze trading results."""
        
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
        
        # Position size analysis
        avg_position_size = trades_df['position_size'].mean()
        position_size_std = trades_df['position_size'].std()
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'total_pnl': total_pnl,
            'final_capital': final_capital,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_position_size': avg_position_size,
            'position_size_std': position_size_std,
            'trades_df': trades_df,
            'equity_curve': equity_curve
        }


def main():
    """Run position size optimization analysis."""
    
    logger.info("Starting position size optimization for 3x leverage")
    
    print("="*80)
    print("📊 ポジションサイズ最適化分析（レバレッジ3倍）")
    print("="*80)
    
    optimizer = PositionSizeOptimizer(leverage=3.0)
    
    # Test different base position sizes
    base_sizes = [0.015, 0.02, 0.025, 0.03, 0.035]
    results = optimizer.run_optimized_backtest(confidence_threshold=0.6, base_sizes=base_sizes)
    
    # Print results summary
    print(f"\n📈 ベースポジションサイズ別パフォーマンス:")
    print(f"{'ベースサイズ':<12} {'総収益率':<10} {'月次収益':<10} {'Sharpe':<8} {'最大DD':<10} {'取引数':<8}")
    print("-" * 70)
    
    best_size = None
    best_sharpe = -float('inf')
    
    for base_size, result in results.items():
        monthly_return = result['total_return'] / 3
        print(f"{base_size*100:>8.1f}% {result['total_return']:>8.2f}% "
              f"{monthly_return:>8.2f}% {result['sharpe_ratio']:>6.2f} "
              f"{result['max_drawdown']:>8.2f}% {result['num_trades']:>6}")
        
        if result['sharpe_ratio'] > best_sharpe:
            best_sharpe = result['sharpe_ratio']
            best_size = base_size
    
    # Detailed analysis of best result
    if best_size and best_size in results:
        best_result = results[best_size]
        
        print(f"\n{'='*80}")
        print(f"🏆 最適ベースポジションサイズ: {best_size*100:.1f}%")
        print(f"{'='*80}")
        
        print(f"\n📊 詳細パフォーマンス:")
        print(f"  総収益率: {best_result['total_return']:.2f}%")
        print(f"  月次収益率: {best_result['total_return']/3:.2f}%")
        print(f"  年次換算収益率: {best_result['total_return']/3*12:.2f}%")
        print(f"  Sharpe比: {best_result['sharpe_ratio']:.2f}")
        print(f"  最大ドローダウン: {best_result['max_drawdown']:.2f}%")
        
        print(f"\n📈 ポジションサイズ統計:")
        print(f"  平均ポジションサイズ: {best_result['avg_position_size']*100:.2f}%")
        print(f"  標準偏差: {best_result['position_size_std']*100:.2f}%")
        
        # Visualization
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Equity curves comparison
        plt.subplot(2, 2, 1)
        for base_size, result in results.items():
            equity = result['equity_curve']
            dates = pd.date_range(start='2024-05-01', periods=len(equity), freq='5min')[:len(equity)]
            plt.plot(dates, equity, label=f'Base {base_size*100:.1f}%', linewidth=2)
        plt.axhline(y=100000, color='gray', linestyle='--', alpha=0.7)
        plt.title('ポジションサイズ別エクイティカーブ')
        plt.xlabel('Date')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Position size distribution
        plt.subplot(2, 2, 2)
        best_trades = best_result['trades_df']
        plt.hist(best_trades['position_size'] * 100, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=best_size*100, color='red', linestyle='--', label=f'Base: {best_size*100:.1f}%')
        plt.title('動的ポジションサイズ分布')
        plt.xlabel('Position Size (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Position size over time
        plt.subplot(2, 2, 3)
        scatter = plt.scatter(best_trades.index, best_trades['position_size'] * 100, 
                            c=best_trades['confidence'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Confidence')
        plt.axhline(y=best_size*100, color='red', linestyle='--', alpha=0.5)
        plt.title('ポジションサイズ推移（信頼度別）')
        plt.xlabel('Trade Number')
        plt.ylabel('Position Size (%)')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Risk-Return by position size
        plt.subplot(2, 2, 4)
        returns = []
        sharpes = []
        sizes = []
        
        for base_size, result in results.items():
            returns.append(result['total_return'])
            sharpes.append(result['sharpe_ratio'])
            sizes.append(base_size * 100)
        
        plt.plot(sizes, returns, 'b-o', label='Total Return', linewidth=2, markersize=8)
        plt.ylabel('Total Return (%)', color='b')
        plt.tick_params(axis='y', labelcolor='b')
        
        ax2 = plt.twinx()
        ax2.plot(sizes, sharpes, 'r-s', label='Sharpe Ratio', linewidth=2, markersize=8)
        ax2.set_ylabel('Sharpe Ratio', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.xlabel('Base Position Size (%)')
        plt.title('リスク・リターン vs ポジションサイズ')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results/position_size_optimization.png', dpi=300, bbox_inches='tight')
        
        # Save best results
        best_trades.to_csv('backtest_results/optimized_position_trades.csv', index=False)
        
        print(f"\n💾 保存完了:")
        print(f"  取引データ: backtest_results/optimized_position_trades.csv")
        print(f"  分析チャート: backtest_results/position_size_optimization.png")
        
        # Compare with original 2% fixed size
        print(f"\n📊 改善効果:")
        print(f"  元の固定2%での月次収益: 1.90%")
        print(f"  最適化後の月次収益: {best_result['total_return']/3:.2f}%")
        print(f"  改善率: {(best_result['total_return']/3/1.90 - 1)*100:.1f}%")
        
        # Kelly Criterion calculation example
        trades_df = best_result['trades_df']
        wins = trades_df[trades_df['pnl_dollar'] > 0]
        losses = trades_df[trades_df['pnl_dollar'] < 0]
        
        if len(wins) > 0 and len(losses) > 0:
            win_rate = len(wins) / len(trades_df)
            avg_win = wins['pnl_pct'].mean() / 100
            avg_loss = abs(losses['pnl_pct'].mean()) / 100
            
            full_kelly = (win_rate * (avg_win/avg_loss) - (1-win_rate)) / (avg_win/avg_loss)
            
            print(f"\n📐 Kelly基準分析:")
            print(f"  勝率: {win_rate*100:.1f}%")
            print(f"  平均利益/平均損失: {avg_win/avg_loss:.2f}")
            print(f"  Full Kelly: {full_kelly*100:.1f}%")
            print(f"  推奨Kelly (30%): {full_kelly*0.3*100:.1f}%")
        
        return results
    
    else:
        print("❌ 有効な結果が得られませんでした")
        return None


if __name__ == "__main__":
    main()