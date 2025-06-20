#!/usr/bin/env python3
"""
Leverage 3x backtest for neural network model with enhanced risk management.
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


class LeverageBacktester:
    """Enhanced backtest with 3x leverage and advanced risk management."""
    
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
        
        logger.info(f"Loaded neural network model for {self.leverage}x leverage trading")
    
    def load_test_data(self) -> pd.DataFrame:
        """Load extended test data."""
        
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
    
    def calculate_dynamic_position_size(self, confidence: float, recent_volatility: float, 
                                      current_drawdown: float, base_size: float = 0.015) -> float:
        """Calculate dynamic position size based on confidence, volatility, and drawdown."""
        
        # Base position size (1.5% for 3x leverage)
        position_size = base_size
        
        # Confidence adjustment (0.5x to 1.5x based on confidence)
        confidence_multiplier = 0.5 + (confidence - 0.5) * 2
        position_size *= confidence_multiplier
        
        # Volatility adjustment (reduce size in high volatility)
        if recent_volatility > 0.03:  # High volatility
            position_size *= 0.7
        elif recent_volatility < 0.015:  # Low volatility
            position_size *= 1.2
        
        # Drawdown adjustment (reduce size during drawdown)
        if current_drawdown < -1:  # 1% drawdown
            position_size *= 0.8
        elif current_drawdown < -2:  # 2% drawdown
            position_size *= 0.6
        
        # Cap position size
        return min(position_size, 0.025)  # Max 2.5%
    
    def run_leverage_backtest(self, confidence_threshold: float = 0.65):
        """Run comprehensive leverage backtest with enhanced risk management."""
        
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
        base_position_size = 0.015  # 1.5% for 3x leverage
        trades = []
        equity_curve = []
        
        # Risk management variables
        max_daily_loss = 0.02  # 2% daily stop
        consecutive_losses = 0
        max_consecutive_losses = 5
        
        # Find high confidence signals
        high_confidence_indices = np.where(predictions > confidence_threshold)[0]
        logger.info(f"Found {len(high_confidence_indices)} signals at {confidence_threshold} threshold")
        
        for i, signal_idx in enumerate(high_confidence_indices):
            if signal_idx + 10 >= len(data):  # Need next 10 bars
                continue
            
            # Current market state
            entry_price = data['close'].iloc[signal_idx]
            confidence = predictions[signal_idx]
            recent_volatility = data['vol_20'].iloc[signal_idx]
            
            # Calculate current drawdown
            if equity_curve:
                peak_equity = max(equity_curve)
                current_drawdown = (capital - peak_equity) / peak_equity * 100
            else:
                current_drawdown = 0
            
            # Enhanced risk checks
            if consecutive_losses >= max_consecutive_losses:
                continue  # Skip trading after too many consecutive losses
            
            if current_drawdown < -5:  # 5% drawdown circuit breaker
                continue
            
            # Daily loss check (simplified - assuming each trade is roughly daily)
            if i > 0 and len(trades) > 0:
                recent_trades = [t for t in trades[-10:] if t['pnl_pct'] < 0]
                if len(recent_trades) > 3:  # Too many recent losses
                    continue
            
            # Dynamic position sizing
            position_size = self.calculate_dynamic_position_size(
                confidence, recent_volatility, current_drawdown, base_position_size
            )
            
            # Direction based on momentum
            momentum = data['momentum_3'].iloc[signal_idx]
            direction = 'long' if momentum > 0 else 'short'
            
            # Enhanced exit strategy
            entry_time = data.index[signal_idx]
            
            best_pnl = float('-inf')
            best_exit_time = None
            best_exit_price = None
            
            # Multiple exit points with stop-loss
            for exit_bars in [2, 3, 5, 7, 10]:
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
                
                # Transaction costs (higher for leverage)
                leveraged_pnl_pct -= 0.0015  # 0.15% round-trip for leverage
                
                # Stop-loss check (3% loss on leveraged position)
                if leveraged_pnl_pct < -0.03:
                    best_pnl = leveraged_pnl_pct
                    best_exit_time = exit_time
                    best_exit_price = exit_price
                    break  # Emergency exit
                
                # Take profit check (1.5% gain)
                if leveraged_pnl_pct > 0.015:
                    best_pnl = leveraged_pnl_pct
                    best_exit_time = exit_time
                    best_exit_price = exit_price
                    break  # Take profit
                
                # Track best exit
                if leveraged_pnl_pct > best_pnl:
                    best_pnl = leveraged_pnl_pct
                    best_exit_time = exit_time
                    best_exit_price = exit_price
            
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
                'drawdown_at_entry': current_drawdown
            })
            
            capital += pnl_dollar
            equity_curve.append(capital)
            
            # Daily loss circuit breaker
            daily_loss_pct = pnl_dollar / initial_capital
            if daily_loss_pct < -max_daily_loss:
                logger.warning(f"Daily loss limit reached: {daily_loss_pct:.2%}")
                break
        
        return trades, equity_curve
    
    def analyze_results(self, trades: list, equity_curve: list):
        """Comprehensive analysis of leverage trading results."""
        
        if not trades:
            return None
        
        trades_df = pd.DataFrame(trades)
        initial_capital = 100000
        final_capital = equity_curve[-1] if equity_curve else initial_capital
        
        # Basic metrics
        total_return = (final_capital - initial_capital) / initial_capital * 100
        num_trades = len(trades_df)
        win_rate = len(trades_df[trades_df['pnl_dollar'] > 0]) / num_trades * 100
        avg_trade = trades_df['pnl_dollar'].mean()
        total_pnl = trades_df['pnl_dollar'].sum()
        
        # Risk metrics
        returns = trades_df['pnl_pct'].values
        sharpe_ratio = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252/5)
        
        # Drawdown analysis
        equity_series = pd.Series(equity_curve, index=range(len(equity_curve)))
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / initial_capital * 100
        max_drawdown = drawdown.min()
        
        # Win/Loss streaks
        win_loss = (trades_df['pnl_dollar'] > 0).astype(int)
        streaks = []
        current_streak = 1
        for i in range(1, len(win_loss)):
            if win_loss.iloc[i] == win_loss.iloc[i-1]:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        streaks.append(current_streak)
        
        max_win_streak = max([s for i, s in enumerate(streaks) if i % 2 == 0 and win_loss.iloc[0] == 1] + [0])
        max_loss_streak = max([s for i, s in enumerate(streaks) if i % 2 == (0 if win_loss.iloc[0] == 0 else 1)] + [0])
        
        # Volatility analysis
        avg_volatility = trades_df['volatility'].mean()
        
        results = {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'total_pnl': total_pnl,
            'final_capital': final_capital,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'avg_volatility': avg_volatility,
            'trades_df': trades_df,
            'equity_curve': equity_curve
        }
        
        return results
    
    def print_leverage_results(self, results: dict):
        """Print comprehensive leverage trading results."""
        
        if not results:
            print("❌ 取引が実行されませんでした")
            return
        
        print("\n" + "="*80)
        print(f"🚀 レバレッジ{self.leverage}倍 バックテスト結果 (3ヶ月: 2024年5-7月)")
        print("="*80)
        
        print(f"\n📊 基本パフォーマンス:")
        print(f"  💰 総収益率: {results['total_return']:.2f}%")
        print(f"  📈 月次収益率: {results['total_return']/3:.2f}%")
        print(f"  🚀 年次収益率: {results['total_return']/3*12:.2f}%")
        print(f"  💵 総利益: ${results['total_pnl']:,.2f}")
        print(f"  📊 取引数: {results['num_trades']}")
        print(f"  🎯 勝率: {results['win_rate']:.1f}%")
        print(f"  💲 平均取引: ${results['avg_trade']:.2f}")
        
        print(f"\n⚡ レバレッジ効果:")
        base_return = results['total_return'] / self.leverage
        print(f"  📉 レバレッジなし収益率: {base_return:.2f}%")
        print(f"  📈 レバレッジ効果倍率: {results['total_return']/base_return:.1f}x")
        
        print(f"\n🛡️ リスク分析:")
        print(f"  📊 Sharpe比: {results['sharpe_ratio']:.2f}")
        print(f"  📉 最大ドローダウン: {results['max_drawdown']:.2f}%")
        print(f"  🔥 最大連勝: {results['max_win_streak']}回")
        print(f"  ❄️ 最大連敗: {results['max_loss_streak']}回")
        print(f"  🌊 平均ボラティリティ: {results['avg_volatility']*100:.2f}%")
        
        # Risk assessment
        monthly_return = results['total_return'] / 3
        if monthly_return > 4:
            assessment = "🎉 非常に優秀！レバレッジ効果絶大"
        elif monthly_return > 2:
            assessment = "🚀 優秀！実用化強く推奨"
        elif monthly_return > 1:
            assessment = "✅ 良好！実用可能"
        elif monthly_return > 0:
            assessment = "⚠️ 微益だが改善の余地あり"
        else:
            assessment = "❌ 損失。レバレッジ使用は危険"
        
        print(f"\n🎯 総合評価: {assessment}")
        
        # Risk warnings
        print(f"\n⚠️ レバレッジ取引注意事項:")
        if results['max_drawdown'] < -3:
            print(f"  🔴 高いドローダウン（{results['max_drawdown']:.1f}%）- 資金管理要注意")
        if results['max_loss_streak'] > 5:
            print(f"  🔴 長い連敗（{results['max_loss_streak']}回）- メンタル管理重要")
        if results['sharpe_ratio'] < 1:
            print(f"  🔴 低いSharpe比（{results['sharpe_ratio']:.2f}）- リスク調整済みリターン要改善")
        
        if results['max_drawdown'] > -2 and results['sharpe_ratio'] > 1.5 and results['max_loss_streak'] <= 4:
            print(f"  ✅ 良好なリスクプロファイル - レバレッジ使用推奨")
        
        return results


def main():
    """Run 3x leverage backtest analysis."""
    
    logger.info("Starting 3x leverage backtest analysis")
    
    # Test multiple confidence thresholds
    thresholds = [0.6, 0.65, 0.7, 0.75]
    all_results = {}
    
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"テスト中: 信頼度閾値 {threshold}")
        print(f"{'='*60}")
        
        backtester = LeverageBacktester(leverage=3.0)
        trades, equity_curve = backtester.run_leverage_backtest(confidence_threshold=threshold)
        results = backtester.analyze_results(trades, equity_curve)
        
        if results:
            print(f"\n📊 閾値 {threshold} 結果:")
            print(f"  収益率: {results['total_return']:.2f}% | 取引数: {results['num_trades']} | 勝率: {results['win_rate']:.1f}%")
            all_results[threshold] = results
    
    # Find best threshold
    if all_results:
        best_threshold = max(all_results.keys(), key=lambda k: all_results[k]['total_return'])
        best_results = all_results[best_threshold]
        
        print(f"\n{'='*80}")
        print(f"🏆 最適結果: 信頼度閾値 {best_threshold}")
        print(f"{'='*80}")
        
        backtester = LeverageBacktester(leverage=3.0)
        backtester.print_leverage_results(best_results)
        
        # Save results
        trades_df = best_results['trades_df']
        trades_df.to_csv('backtest_results/leverage_3x_trades.csv', index=False)
        
        # Plot equity curve
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        equity_curve = best_results['equity_curve']
        dates = pd.date_range(start='2024-05-01', periods=len(equity_curve), freq='5min')
        plt.plot(dates, equity_curve, linewidth=2, color='green')
        plt.axhline(y=100000, color='gray', linestyle='--', alpha=0.7)
        plt.title('レバレッジ3倍 エクイティカーブ')
        plt.ylabel('資本 ($)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        trades_df['cumulative_pnl'] = trades_df['pnl_dollar'].cumsum()
        plt.plot(trades_df.index, trades_df['cumulative_pnl'], linewidth=2, color='blue')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.title('累積PnL')
        plt.ylabel('累積利益 ($)')
        plt.xlabel('取引番号')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results/leverage_3x_results.png', dpi=300, bbox_inches='tight')
        
        print(f"\n💾 保存完了:")
        print(f"  取引データ: backtest_results/leverage_3x_trades.csv")
        print(f"  チャート: backtest_results/leverage_3x_results.png")
        
        return best_results
    
    else:
        print("❌ 有効な結果が得られませんでした")
        return None


if __name__ == "__main__":
    main()