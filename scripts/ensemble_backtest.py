#!/usr/bin/env python3
"""
Backtest the improved ensemble model to validate profitability.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Tuple
import joblib
from dataclasses import dataclass
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@dataclass
class Trade:
    """Trade data structure."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp = None
    symbol: str = "BTCUSDT"
    side: str = "long"  # long or short
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    pnl: float = 0.0
    fees: float = 0.0
    confidence: float = 0.0
    status: str = "open"  # open, closed


class EnsembleBacktester:
    """Backtest the ensemble model with proper risk management."""
    
    def __init__(self, initial_capital: float = 100000.0, 
                 position_size_pct: float = 0.02,  # 2% per trade
                 max_positions: int = 3,
                 confidence_threshold: float = 0.7,  # Higher threshold for quality
                 fee_rate: float = 0.0006):
        
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.confidence_threshold = confidence_threshold
        self.fee_rate = fee_rate
        
        # Trading state
        self.trades = []
        self.open_positions = {}
        self.equity_curve = []
        
        # Models
        self.models = {}
        self.scaler = None
        
    def load_models(self, model_dir: str = "models/simple_ensemble"):
        """Load trained ensemble models."""
        model_path = Path(model_dir)
        
        model_files = {
            'lightgbm': 'lightgbm_model.pkl',
            'catboost': 'catboost_model.pkl', 
            'random_forest': 'random_forest_model.pkl',
            'ensemble': 'ensemble_model.pkl'
        }
        
        for name, filename in model_files.items():
            file_path = model_path / filename
            if file_path.exists():
                self.models[name] = joblib.load(file_path)
                logger.info(f"Loaded {name} model")
            else:
                logger.warning(f"Model file not found: {file_path}")
        
        if not self.models:
            raise ValueError("No models loaded successfully")
            
        logger.info(f"Loaded {len(self.models)} models")
    
    def load_test_data(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Load test data (different from training period)."""
        logger.info(f"Loading test data for {symbol}")
        
        conn = duckdb.connect("data/historical_data.duckdb")
        
        # Use May 2024 as test period (after training period)
        query = f"""
        SELECT 
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM klines_{symbol.lower()}
        WHERE timestamp >= '2024-05-01'
          AND timestamp <= '2024-05-31'
        ORDER BY timestamp
        """
        
        data = conn.execute(query).df()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        conn.close()
        
        logger.info(f"Loaded {len(data)} test records")
        return data
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create same features as training (must match exactly)."""
        
        # Basic price features  
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['oc_ratio'] = (data['close'] - data['open']) / data['open']
        
        # Multi-timeframe returns
        for period in [1, 3, 5, 10, 15, 30]:
            data[f'return_{period}'] = data['close'].pct_change(period)
        
        # Volatility features
        for window in [5, 10, 20, 30]:
            data[f'vol_{window}'] = data['returns'].rolling(window).std()
        
        # Simple moving averages
        for ma in [5, 10, 20]:
            data[f'sma_{ma}'] = data['close'].rolling(ma).mean()
            data[f'price_vs_sma_{ma}'] = (data['close'] - data[f'sma_{ma}']) / data[f'sma_{ma}']
        
        # Momentum indicators
        data['momentum_3'] = data['close'].pct_change(3)
        data['momentum_5'] = data['close'].pct_change(5)
        data['momentum_10'] = data['close'].pct_change(10)
        
        # Volume features
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['log_volume'] = np.log(data['volume'] + 1)
        
        # RSI approximation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Band position
        bb_middle = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # Trend indicators
        data['trend_strength'] = (data['sma_5'] - data['sma_20']) / data['sma_20']
        data['price_above_ma'] = (data['close'] > data['sma_20']).astype(int)
        
        # Volume-price interaction
        data['volume_price_change'] = data['volume_ratio'] * abs(data['returns'])
        
        # Market regime
        data['high_vol'] = (data['vol_20'] > data['vol_20'].rolling(50).quantile(0.8)).astype(int)
        data['low_vol'] = (data['vol_20'] < data['vol_20'].rolling(50).quantile(0.2)).astype(int)
        
        # Time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        
        # Handle NaN values same as training
        feature_cols = [col for col in data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        for col in feature_cols:
            if data[col].dtype in ['float64', 'int64']:
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(method='ffill').fillna(0)
        
        return data
    
    def get_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from all models."""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[:, 1]
                else:
                    pred_proba = model.predict(X)
                predictions[name] = pred_proba
            except Exception as e:
                logger.warning(f"Prediction error for {name}: {e}")
                predictions[name] = np.zeros(len(X))
        
        return predictions
    
    def generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using ensemble predictions."""
        
        # Get feature columns (same as training)
        feature_cols = [col for col in data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        X = data[feature_cols].copy()
        
        # Remove rows with NaN
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        
        if len(X) == 0:
            logger.warning("No valid data for prediction")
            return pd.DataFrame()
        
        # Get predictions
        predictions = self.get_predictions(X)
        
        signals = []
        
        for i in range(len(X)):
            timestamp = X.index[i]
            
            # Use ensemble prediction (or best model if ensemble not available)
            if 'ensemble' in predictions:
                confidence = predictions['ensemble'][i]
            elif 'random_forest' in predictions:
                confidence = predictions['random_forest'][i]
            else:
                confidence = list(predictions.values())[0][i]
            
            # Generate signal if confidence is high enough
            if confidence > self.confidence_threshold:
                
                # Simple direction logic based on recent momentum
                recent_return = data.loc[timestamp, 'return_3']
                
                if recent_return > 0:
                    direction = 'long'  # Momentum continuation
                else:
                    direction = 'short'  # Momentum continuation
                
                signals.append({
                    'timestamp': timestamp,
                    'direction': direction,
                    'confidence': confidence,
                    'price': data.loc[timestamp, 'close']
                })
        
        signals_df = pd.DataFrame(signals)
        logger.info(f"Generated {len(signals_df)} trading signals")
        
        return signals_df
    
    def execute_trade(self, signal: Dict, current_price: float) -> Trade:
        """Execute a trade based on signal."""
        
        # Calculate position size
        position_value = self.capital * self.position_size_pct
        position_size = position_value / current_price
        
        # Calculate fees
        entry_fee = position_value * self.fee_rate
        
        # Create trade
        trade = Trade(
            entry_time=signal['timestamp'],
            symbol="BTCUSDT",
            side=signal['direction'],
            entry_price=current_price,
            size=position_size,
            confidence=signal['confidence'],
            fees=entry_fee
        )
        
        # Update capital
        self.capital -= entry_fee
        
        # Store trade
        self.trades.append(trade)
        self.open_positions[signal['timestamp']] = trade
        
        logger.info(f"Opened {trade.side} position at {current_price:.2f} (confidence: {trade.confidence:.2f})")
        
        return trade
    
    def close_trade(self, trade: Trade, current_price: float, current_time: pd.Timestamp, reason: str = ""):
        """Close an open trade."""
        
        # Calculate PnL
        if trade.side == 'long':
            pnl = (current_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - current_price) * trade.size
        
        # Calculate exit fee
        exit_fee = abs(pnl) * self.fee_rate if pnl != 0 else current_price * trade.size * self.fee_rate
        pnl -= exit_fee
        
        # Update trade
        trade.exit_time = current_time
        trade.exit_price = current_price
        trade.pnl = pnl
        trade.fees += exit_fee
        trade.status = 'closed'
        
        # Update capital
        self.capital += pnl
        
        # Remove from open positions
        del self.open_positions[trade.entry_time]
        
        hold_time = (current_time - trade.entry_time).total_seconds() / 3600
        logger.info(f"Closed {trade.side} position at {current_price:.2f}, PnL: ${pnl:.2f}, Hold: {hold_time:.1f}h, Reason: {reason}")
        
        return pnl
    
    def run_backtest(self) -> Dict:
        """Run the ensemble model backtest."""
        
        logger.info("Starting ensemble model backtest")
        
        # Load models
        self.load_models()
        
        # Load test data
        data = self.load_test_data()
        
        # Create features
        data = self.create_features(data)
        
        # Generate signals
        signals = self.generate_trading_signals(data)
        
        if len(signals) == 0:
            logger.warning("No trading signals generated")
            return self.calculate_metrics()
        
        # Execute trading simulation
        for _, signal in signals.iterrows():
            timestamp = signal['timestamp']
            current_price = signal['price']
            
            # Check if we can open new position
            if (len(self.open_positions) < self.max_positions and 
                timestamp not in self.open_positions):
                
                self.execute_trade(signal.to_dict(), current_price)
            
            # Check existing positions for exit conditions
            for entry_time, trade in list(self.open_positions.items()):
                
                # Get current price for this trade
                if timestamp in data.index:
                    current_price_trade = data.loc[timestamp, 'close']
                    
                    # Exit conditions
                    pnl_pct = 0
                    if trade.side == 'long':
                        pnl_pct = (current_price_trade - trade.entry_price) / trade.entry_price
                    else:
                        pnl_pct = (trade.entry_price - current_price_trade) / trade.entry_price
                    
                    should_close = False
                    reason = ""
                    
                    # Stop loss (1.5%)
                    if pnl_pct <= -0.015:
                        should_close = True
                        reason = "stop_loss"
                    
                    # Take profit (2.5%)
                    elif pnl_pct >= 0.025:
                        should_close = True
                        reason = "take_profit"
                    
                    # Time exit (4 hours)
                    elif (timestamp - trade.entry_time).total_seconds() > 14400:
                        should_close = True
                        reason = "time_exit"
                    
                    if should_close:
                        self.close_trade(trade, current_price_trade, timestamp, reason)
            
            # Record equity
            total_equity = self.capital
            for trade in self.open_positions.values():
                if timestamp in data.index:
                    current_price_equity = data.loc[timestamp, 'close']
                    if trade.side == 'long':
                        unrealized_pnl = (current_price_equity - trade.entry_price) * trade.size
                    else:
                        unrealized_pnl = (trade.entry_price - current_price_equity) * trade.size
                    total_equity += unrealized_pnl
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': total_equity,
                'realized_capital': self.capital,
                'open_positions': len(self.open_positions)
            })
        
        # Close remaining positions
        for trade in list(self.open_positions.values()):
            final_price = data['close'].iloc[-1]
            final_time = data.index[-1]
            self.close_trade(trade, final_price, final_time, "final_close")
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics."""
        
        if not self.trades:
            return {
                'total_return': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_trade': 0.0,
                'final_capital': self.capital
            }
        
        # Basic metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        closed_trades = [t for t in self.trades if t.status == 'closed']
        
        if closed_trades:
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            losing_trades = [t for t in closed_trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(closed_trades)
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            avg_trade = np.mean([t.pnl for t in closed_trades])
            
            # Sharpe-like ratio
            if len(self.equity_curve) > 1:
                equity_df = pd.DataFrame(self.equity_curve)
                equity_df['returns'] = equity_df['equity'].pct_change()
                daily_ret_std = equity_df['returns'].std()
                sharpe = (total_return / 30) / (daily_ret_std * np.sqrt(30)) if daily_ret_std > 0 else 0  # Approximation for 1 month
            else:
                sharpe = 0
        else:
            win_rate = avg_win = avg_loss = avg_trade = sharpe = 0
        
        return {
            'total_return': total_return,
            'final_capital': self.capital,
            'total_trades': len(closed_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'sharpe_ratio': sharpe,
            'winning_trades': len(winning_trades) if closed_trades else 0,
            'losing_trades': len(losing_trades) if closed_trades else 0,
            'equity_curve': self.equity_curve,
            'trades': [self._trade_to_dict(t) for t in self.trades]
        }
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert trade to dictionary."""
        return {
            'entry_time': str(trade.entry_time),
            'exit_time': str(trade.exit_time) if trade.exit_time else None,
            'side': trade.side,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'pnl': trade.pnl,
            'confidence': trade.confidence,
            'status': trade.status
        }
    
    def plot_results(self, results: Dict):
        """Plot backtest results."""
        if not results['equity_curve']:
            logger.warning("No equity curve to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Equity curve
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        ax1.plot(equity_df['timestamp'], equity_df['equity'], 'b-', linewidth=2, label='Total Equity')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('Ensemble Model Backtest - Equity Curve')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Trade PnL distribution
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            closed_trades = trades_df[trades_df['status'] == 'closed']
            
            if len(closed_trades) > 0:
                ax2.hist(closed_trades['pnl'], bins=20, alpha=0.7, edgecolor='black')
                ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax2.set_title('Trade PnL Distribution')
                ax2.set_xlabel('PnL ($)')
                ax2.set_ylabel('Frequency')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results/ensemble_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Run ensemble model backtest."""
    
    logger.info("Starting ensemble model backtest")
    
    # Initialize backtester with conservative settings
    backtester = EnsembleBacktester(
        initial_capital=100000.0,
        position_size_pct=0.02,  # 2% per trade (conservative)
        max_positions=3,
        confidence_threshold=0.7,  # High confidence threshold
        fee_rate=0.0006
    )
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Print results
    print("\n" + "="*70)
    print("アンサンブルモデル バックテスト結果")
    print("="*70)
    
    print(f"\n📊 トレード統計:")
    print(f"  総取引数: {results['total_trades']}")
    print(f"  勝率: {results['win_rate']*100:.1f}%")
    print(f"  勝ちトレード: {results['winning_trades']}")
    print(f"  負けトレード: {results['losing_trades']}")
    
    print(f"\n💰 収益性:")
    print(f"  初期資本: ${backtester.initial_capital:,.2f}")
    print(f"  最終資本: ${results['final_capital']:,.2f}")
    print(f"  総収益率: {results['total_return']*100:.2f}%")
    print(f"  平均取引利益: ${results['avg_trade']:.2f}")
    print(f"  平均勝ち: ${results['avg_win']:.2f}")
    print(f"  平均負け: ${results['avg_loss']:.2f}")
    
    print(f"\n⚡ リスク指標:")
    print(f"  シャープレシオ: {results['sharpe_ratio']:.2f}")
    
    # Performance assessment
    print(f"\n🎯 性能評価:")
    if results['total_return'] > 0.05:  # 5%+ return in 1 month
        print(f"  ✅ 優秀 - 月間{results['total_return']*100:.1f}%の収益")
    elif results['total_return'] > 0:
        print(f"  ⚠️  微益 - 月間{results['total_return']*100:.1f}%の微小利益")
    else:
        print(f"  ❌ 損失 - 月間{results['total_return']*100:.1f}%の損失")
    
    if results['win_rate'] > 0.6:
        print(f"  ✅ 勝率{results['win_rate']*100:.1f}%は優秀")
    elif results['win_rate'] > 0.4:
        print(f"  ⚠️  勝率{results['win_rate']*100:.1f}%は普通")
    else:
        print(f"  ❌ 勝率{results['win_rate']*100:.1f}%は低い")
    
    # Overall assessment
    if results['total_return'] > 0.02 and results['win_rate'] > 0.5:
        print(f"\n🚀 総合評価: Bot実用化推奨！")
    elif results['total_return'] > 0:
        print(f"\n🔧 総合評価: パラメータ調整でさらに改善可能")
    else:
        print(f"\n⚠️  総合評価: さらなる改善が必要")
    
    # Plot results
    backtester.plot_results(results)
    
    return results


if __name__ == "__main__":
    main()