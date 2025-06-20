#!/usr/bin/env python3
"""
Simplified backtest for liquidation cascade model using price features directly.

This script uses actual price movements and volatility to generate trading signals,
without synthetic liquidation generation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import duckdb
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging
from src.ml_pipeline.model_trainer import ModelTrainer

setup_logging()
logger = get_logger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float = 0.0
    fees: float = 0.0
    status: str = 'open'  # 'open', 'closed'


class SimplifiedBacktester:
    """
    Simplified backtester using price-based features for cascade detection.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 position_size_pct: float = 0.02,  # 2% per trade
                 max_positions: int = 3,
                 fee_rate: float = 0.0006,  # 0.06% taker fee
                 slippage_bps: float = 10,  # 10 basis points slippage
                 stop_loss_pct: float = 0.015,  # 1.5% stop loss
                 take_profit_pct: float = 0.025,  # 2.5% take profit
                 volatility_threshold: float = 0.015,  # 1.5% volatility to trigger
                 momentum_threshold: float = 0.01):  # 1% momentum threshold
        """Initialize backtester."""
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps / 10000.0
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.volatility_threshold = volatility_threshold
        self.momentum_threshold = momentum_threshold
        
        # Trading state
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.equity_curve = []
        self.signals = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0
    
    def extract_price_features(self, price_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract features from price data."""
        df = price_data[price_data['symbol'] == symbol].copy()
        df = df.sort_values('timestamp')
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Price momentum features
        df['price_change_5m'] = df['close'].pct_change(5)
        df['price_change_15m'] = df['close'].pct_change(15)
        df['price_change_30m'] = df['close'].pct_change(30)
        df['price_change_60m'] = df['close'].pct_change(60)
        
        # Volatility features
        df['volatility_5m'] = df['returns'].rolling(5).std()
        df['volatility_15m'] = df['returns'].rolling(15).std()
        df['volatility_30m'] = df['returns'].rolling(30).std()
        df['volatility_60m'] = df['returns'].rolling(60).std()
        df['volatility_change'] = df['volatility_30m'].pct_change(10)
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(30).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # High-low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_ma'] = df['hl_range'].rolling(30).mean()
        
        # Trend features
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        
        # Cascade-like features (simulated)
        df['liq_count_60s'] = df['volatility_5m'] * 100  # Proxy for liquidation activity
        df['liq_count_300s'] = df['volatility_15m'] * 100
        df['liq_count_900s'] = df['volatility_30m'] * 100
        df['liq_count_1800s'] = df['volatility_60m'] * 100
        
        # Cascade acceleration (volatility increasing)
        df['cascade_acceleration_300s'] = df['volatility_5m'] / (df['volatility_15m'] + 1e-8)
        df['cascade_acceleration_900s'] = df['volatility_15m'] / (df['volatility_30m'] + 1e-8)
        
        # Volume surge as liquidation proxy
        df['liq_volume_60s'] = df['volume_ratio'] * df['volatility_5m']
        df['liq_max_size_300s'] = df['hl_range'] * df['volume']
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def generate_trading_signals(self, features: pd.DataFrame, model: Optional[ModelTrainer] = None) -> pd.DataFrame:
        """Generate trading signals from features."""
        signals = []
        
        for idx in range(len(features)):
            row = features.iloc[idx]
            
            # Use model predictions if available
            if model is not None:
                feature_cols = ['price_change_5m', 'price_change_15m', 'volatility_change', 
                               'volume_ratio', 'liq_count_60s', 'liq_count_300s', 'liq_count_900s',
                               'liq_count_1800s', 'cascade_acceleration_300s', 'cascade_acceleration_900s',
                               'liq_volume_60s', 'liq_max_size_300s']
                
                # Add missing features with defaults
                X = pd.DataFrame([row[feature_cols]], columns=feature_cols)
                
                # Fill any missing with 0
                X = X.fillna(0)
                
                # Add dummy features to match training (29 features total)
                while len(X.columns) < 29:
                    X[f'dummy_{len(X.columns)}'] = 0
                
                try:
                    prediction = model.predict(X)[0]
                    cascade_signal = prediction > 0.5  # Binary threshold
                except Exception as e:
                    logger.debug(f"Prediction error: {e}")
                    cascade_signal = False
            else:
                # Rule-based signals
                cascade_signal = (
                    abs(row['volatility_30m']) > self.volatility_threshold and
                    abs(row['price_change_15m']) > self.momentum_threshold and
                    row['volume_ratio'] > 1.5
                )
            
            if cascade_signal:
                # Determine direction based on recent momentum
                if row['price_change_5m'] < 0 and row['price_change_15m'] < 0:
                    direction = 'short'  # Expecting further drop
                elif row['price_change_5m'] > 0 and row['price_change_15m'] > 0:
                    direction = 'long'  # Expecting bounce
                else:
                    continue  # Skip mixed signals
                
                signals.append({
                    'timestamp': row.name,
                    'symbol': row['symbol'],
                    'direction': direction,
                    'volatility': row['volatility_30m'],
                    'momentum': row['price_change_15m'],
                    'volume_ratio': row['volume_ratio']
                })
        
        return pd.DataFrame(signals)
    
    def run_backtest(self, 
                    model_path: Optional[str] = None,
                    start_date: str = "2024-01-01",
                    end_date: str = "2024-12-30") -> Dict:
        """Run simplified backtest."""
        logger.info(f"Starting simplified backtest from {start_date} to {end_date}")
        
        # Load model if provided
        model = None
        if model_path:
            try:
                model_version = Path(model_path).name
                model = ModelTrainer({
                    'model_save_path': str(Path(model_path).parent),
                    'model_version': model_version
                })
                model.load_model(model_version)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using rule-based signals.")
        
        # Load historical data
        conn = duckdb.connect("data/historical_data.duckdb")
        
        all_features = []
        symbols = ["BTCUSDT", "ETHUSDT", "ICPUSDT"]
        
        for symbol in symbols:
            query = f"""
            SELECT 
                timestamp,
                '{symbol}' as symbol,
                open,
                high,
                low,
                close,
                volume
            FROM klines_{symbol.lower()}
            WHERE timestamp >= '{start_date}'
              AND timestamp <= '{end_date}'
            ORDER BY timestamp
            """
            
            price_data = conn.execute(query).df()
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            price_data.set_index('timestamp', inplace=True)
            
            # Extract features
            features = self.extract_price_features(price_data.reset_index(), symbol)
            features['symbol'] = symbol
            all_features.append(features)
            
            logger.info(f"Extracted {len(features)} feature points for {symbol}")
        
        conn.close()
        
        # Combine all features
        all_features_df = pd.concat(all_features, ignore_index=False)
        all_features_df = all_features_df.sort_index()
        
        # Generate trading signals
        signals_df = self.generate_trading_signals(all_features_df, model)
        logger.info(f"Generated {len(signals_df)} trading signals")
        
        # Store signals for analysis
        self.signals = signals_df.to_dict('records')
        
        # Simulate trading
        timestamps = all_features_df.index.unique()[::10]  # Check every 10 minutes
        
        for timestamp in timestamps:
            # Check exit conditions for open positions
            for symbol, trade in list(self.open_positions.items()):
                current_data = all_features_df[
                    (all_features_df.index == timestamp) & 
                    (all_features_df['symbol'] == symbol)
                ]
                
                if len(current_data) > 0:
                    current_price = current_data['close'].iloc[0]
                    self.check_and_close_position(trade, current_price, timestamp)
            
            # Check for new signals
            current_signals = signals_df[signals_df['timestamp'] == timestamp]
            
            for _, signal in current_signals.iterrows():
                if len(self.open_positions) < self.max_positions:
                    if signal['symbol'] not in self.open_positions:
                        current_data = all_features_df[
                            (all_features_df.index == timestamp) & 
                            (all_features_df['symbol'] == signal['symbol'])
                        ]
                        
                        if len(current_data) > 0:
                            current_price = current_data['close'].iloc[0]
                            self.execute_trade(signal.to_dict(), current_price, signal['symbol'])
            
            # Record equity
            total_value = self.capital
            for trade in self.open_positions.values():
                current_data = all_features_df[
                    (all_features_df.index == timestamp) & 
                    (all_features_df['symbol'] == trade.symbol)
                ]
                if len(current_data) > 0:
                    current_price = current_data['close'].iloc[0]
                    if trade.side == 'long':
                        unrealized_pnl = (current_price - trade.entry_price) * trade.size
                    else:
                        unrealized_pnl = (trade.entry_price - current_price) * trade.size
                    total_value += unrealized_pnl
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': total_value,
                'capital': self.capital,
                'open_positions': len(self.open_positions)
            })
        
        # Close remaining positions
        for symbol, trade in list(self.open_positions.items()):
            last_data = all_features_df[all_features_df['symbol'] == symbol].iloc[-1]
            self.close_position(trade, last_data['close'], last_data.name)
        
        return self.calculate_metrics()
    
    def execute_trade(self, signal: Dict, current_price: float, symbol: str) -> Optional[Trade]:
        """Execute trade based on signal."""
        # Calculate position size
        position_value = self.capital * self.position_size_pct
        position_size = position_value / current_price
        
        # Apply slippage
        if signal['direction'] == 'short':
            entry_price = current_price * (1 - self.slippage_bps)
        else:
            entry_price = current_price * (1 + self.slippage_bps)
        
        # Calculate fees
        fees = position_value * self.fee_rate
        
        # Create trade
        trade = Trade(
            entry_time=signal['timestamp'],
            exit_time=None,
            symbol=symbol,
            side=signal['direction'],
            entry_price=entry_price,
            exit_price=None,
            size=position_size,
            fees=fees,
            status='open'
        )
        
        # Update capital
        self.capital -= fees
        
        # Store trade
        self.trades.append(trade)
        self.open_positions[symbol] = trade
        self.total_trades += 1
        
        logger.info(f"Opened {trade.side} position in {symbol} at {entry_price:.2f}")
        
        return trade
    
    def check_and_close_position(self, trade: Trade, current_price: float, current_time: pd.Timestamp):
        """Check if position should be closed."""
        if trade.side == 'long':
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - current_price) / trade.entry_price
        
        # Check exit conditions
        should_close = False
        
        if pnl_pct <= -self.stop_loss_pct:
            should_close = True
            logger.debug(f"Stop loss triggered for {trade.symbol}")
        elif pnl_pct >= self.take_profit_pct:
            should_close = True
            logger.debug(f"Take profit triggered for {trade.symbol}")
        elif current_time - trade.entry_time > pd.Timedelta(hours=4):
            should_close = True
            logger.debug(f"Time exit triggered for {trade.symbol}")
        
        if should_close:
            self.close_position(trade, current_price, current_time)
    
    def close_position(self, trade: Trade, current_price: float, current_time: pd.Timestamp):
        """Close an open position."""
        # Apply slippage
        if trade.side == 'long':
            exit_price = current_price * (1 - self.slippage_bps)
            pnl = (exit_price - trade.entry_price) * trade.size
        else:
            exit_price = current_price * (1 + self.slippage_bps)
            pnl = (trade.entry_price - exit_price) * trade.size
        
        # Calculate exit fees
        exit_fees = abs(pnl) * self.fee_rate
        pnl -= exit_fees
        
        # Update trade
        trade.exit_time = current_time
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.fees += exit_fees
        trade.status = 'closed'
        
        # Update capital
        self.capital += pnl - exit_fees
        self.total_pnl += pnl
        self.total_fees += trade.fees
        
        # Update statistics
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Remove from open positions
        del self.open_positions[trade.symbol]
        
        logger.info(f"Closed {trade.side} position in {trade.symbol} at {exit_price:.2f}, PnL: ${pnl:.2f}")
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics."""
        if not self.equity_curve:
            return {
                'total_return': 0.0,
                'final_capital': self.capital,
                'total_pnl': self.total_pnl,
                'total_fees': self.total_fees,
                'total_trades': self.total_trades,
                'signal_count': len(self.signals),
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'equity_curve': [],
                'trades': []
            }
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio
        if len(equity_df) > 1:
            daily_returns = equity_df['returns'].dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # Win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Average trade
        closed_trades = [t for t in self.trades if t.status == 'closed']
        avg_win = np.mean([t.pnl for t in closed_trades if t.pnl > 0]) if any(t.pnl > 0 for t in closed_trades) else 0
        avg_loss = np.mean([t.pnl for t in closed_trades if t.pnl < 0]) if any(t.pnl < 0 for t in closed_trades) else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in closed_trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_return': total_return,
            'final_capital': self.capital,
            'total_pnl': self.total_pnl,
            'total_fees': self.total_fees,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'signal_count': len(self.signals),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': equity_df.to_dict('records'),
            'trades': [self._trade_to_dict(t) for t in self.trades]
        }
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert trade to dictionary."""
        return {
            'entry_time': str(trade.entry_time),
            'exit_time': str(trade.exit_time) if trade.exit_time else None,
            'symbol': trade.symbol,
            'side': trade.side,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'size': trade.size,
            'pnl': trade.pnl,
            'fees': trade.fees,
            'status': trade.status
        }
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Plot backtest results."""
        if not results['equity_curve']:
            logger.warning("No equity curve data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert equity curve back to DataFrame
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # 1. Equity curve
        ax1 = axes[0, 0]
        ax1.plot(equity_df['timestamp'], equity_df['equity'], 'b-', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Equity Curve', fontsize=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Drawdown
        ax2 = axes[0, 1]
        equity_df['drawdown_pct'] = equity_df['drawdown'] * 100
        ax2.fill_between(equity_df['timestamp'], equity_df['drawdown_pct'], 0, 
                        color='red', alpha=0.3)
        ax2.set_title('Drawdown (%)', fontsize=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade PnL distribution
        ax3 = axes[1, 0]
        trades_df = pd.DataFrame(results['trades'])
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            closed_trades = trades_df[trades_df['status'] == 'closed']
            if len(closed_trades) > 0:
                ax3.hist(closed_trades['pnl'], bins=20, alpha=0.7, edgecolor='black')
                ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax3.set_title('Trade PnL Distribution', fontsize=14)
                ax3.set_xlabel('PnL ($)')
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)
        
        # 4. Monthly returns
        ax4 = axes[1, 1]
        equity_df.set_index('timestamp', inplace=True)
        monthly_returns = equity_df['equity'].resample('M').agg(['first', 'last'])
        monthly_returns['return'] = (monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first'] * 100
        
        colors = ['green' if r > 0 else 'red' for r in monthly_returns['return']]
        ax4.bar(range(len(monthly_returns)), monthly_returns['return'], color=colors, alpha=0.7)
        ax4.set_title('Monthly Returns (%)', fontsize=14)
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Return (%)')
        ax4.grid(True, alpha=0.3)
        
        # Set month labels
        month_labels = [idx.strftime('%Y-%m') for idx in monthly_returns.index]
        ax4.set_xticks(range(len(month_labels)))
        ax4.set_xticklabels(month_labels, rotation=45, ha='right')
        
        plt.suptitle('Liquidation Cascade Strategy Backtest Results', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self, results: Dict):
        """Print backtest summary."""
        print("\n" + "="*60)
        print("SIMPLIFIED BACKTEST RESULTS")
        print("="*60)
        
        print(f"\nPerformance Summary:")
        print(f"  Initial Capital:    ${self.initial_capital:,.2f}")
        print(f"  Final Capital:      ${results['final_capital']:,.2f}")
        print(f"  Total Return:       {results['total_return']*100:.2f}%")
        print(f"  Total PnL:          ${results['total_pnl']:,.2f}")
        print(f"  Total Fees:         ${results['total_fees']:,.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:       {results['max_drawdown']*100:.2f}%")
        print(f"  Profit Factor:      {results['profit_factor']:.2f}")
        
        print(f"\nTrading Statistics:")
        print(f"  Total Signals:      {results['signal_count']}")
        print(f"  Total Trades:       {results['total_trades']}")
        print(f"  Winning Trades:     {results['winning_trades']}")
        print(f"  Losing Trades:      {results['losing_trades']}")
        print(f"  Win Rate:           {results['win_rate']*100:.2f}%")
        print(f"  Average Win:        ${results['avg_win']:,.2f}")
        print(f"  Average Loss:       ${results['avg_loss']:,.2f}")


def main():
    """Run simplified backtest."""
    logger.info("Starting simplified cascade backtest")
    
    # Test both with and without model
    configs = [
        {
            'name': 'Rule-Based Strategy',
            'model_path': None,
            'volatility_threshold': 0.015,
            'momentum_threshold': 0.01
        },
        {
            'name': 'ML Model Strategy',
            'model_path': 'models/cascade_detection/cascade_v1_20250612_150231',
            'volatility_threshold': 0.02,
            'momentum_threshold': 0.015
        }
    ]
    
    results_list = []
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print('='*60)
        
        backtester = SimplifiedBacktester(
            initial_capital=100000.0,
            position_size_pct=0.02,
            max_positions=3,
            fee_rate=0.0006,
            slippage_bps=10,
            stop_loss_pct=0.015,
            take_profit_pct=0.025,
            volatility_threshold=config['volatility_threshold'],
            momentum_threshold=config['momentum_threshold']
        )
        
        results = backtester.run_backtest(
            model_path=config['model_path'],
            start_date="2024-01-01",
            end_date="2024-12-30"
        )
        
        results['strategy_name'] = config['name']
        results_list.append(results)
        
        # Print summary
        backtester.print_summary(results)
        
        # Save results
        results_path = Path("backtest_results")
        results_path.mkdir(exist_ok=True)
        
        strategy_name = config['name'].lower().replace(' ', '_').replace('-', '_')
        
        # Save detailed results
        with open(results_path / f"{strategy_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Plot results
        backtester.plot_results(results, save_path=str(results_path / f"{strategy_name}_chart.png"))
    
    # Compare strategies
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON")
    print('='*60)
    
    comparison_df = pd.DataFrame([
        {
            'Strategy': r['strategy_name'],
            'Total Return': f"{r['total_return']*100:.2f}%",
            'Sharpe Ratio': f"{r['sharpe_ratio']:.2f}",
            'Max Drawdown': f"{r['max_drawdown']*100:.2f}%",
            'Win Rate': f"{r['win_rate']*100:.2f}%",
            'Total Trades': r['total_trades'],
            'Profit Factor': f"{r['profit_factor']:.2f}"
        }
        for r in results_list
    ])
    
    print(comparison_df.to_string(index=False))
    
    logger.info("Backtest completed successfully")


if __name__ == "__main__":
    main()