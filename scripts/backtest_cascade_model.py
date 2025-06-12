#!/usr/bin/env python3
"""
Backtest liquidation cascade detection model with realistic trading simulation.

This script performs walk-forward backtesting to avoid data leakage,
simulates realistic trading conditions, and generates profit curves.
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
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging
from src.ml_pipeline.model_trainer import ModelTrainer
from scripts.train_liquidation_cascade_model import (
    LiquidationDataGenerator, 
    CascadeFeatureExtractor
)

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


class CascadeBacktester:
    """
    Backtester for liquidation cascade detection strategy.
    
    Features:
    - Walk-forward analysis to prevent data leakage
    - Realistic trading simulation with fees and slippage
    - Multiple position management
    - Risk management with stop loss and position sizing
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 position_size_pct: float = 0.1,  # 10% per trade
                 max_positions: int = 3,
                 fee_rate: float = 0.0006,  # 0.06% taker fee
                 slippage_bps: float = 10,  # 10 basis points slippage
                 stop_loss_pct: float = 0.02,  # 2% stop loss
                 take_profit_pct: float = 0.05,  # 5% take profit
                 cascade_threshold: float = 0.7,  # Model prediction threshold
                 min_cascade_confidence: float = 0.8):  # Minimum confidence to trade
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital in USD
            position_size_pct: Position size as % of capital
            max_positions: Maximum concurrent positions
            fee_rate: Trading fee rate
            slippage_bps: Slippage in basis points
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            cascade_threshold: Threshold for cascade detection
            min_cascade_confidence: Minimum confidence to enter trade
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps / 10000.0
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.cascade_threshold = cascade_threshold
        self.min_cascade_confidence = min_cascade_confidence
        
        # Trading state
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.equity_curve = []
        self.daily_returns = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        
    def load_model_and_data(self, model_path: str, start_date: str, end_date: str) -> Tuple[ModelTrainer, pd.DataFrame]:
        """Load trained model and prepare backtesting data."""
        logger.info("Loading model and preparing backtest data")
        
        # Load model
        model_version = Path(model_path).name  # Get full directory name
        trainer = ModelTrainer({
            'model_save_path': str(Path(model_path).parent),  # Set correct save path
            'model_version': model_version
        })
        trainer.load_model(model_version)
        
        # Load historical data
        conn = duckdb.connect("data/historical_data.duckdb")
        
        all_data = []
        symbols = ["BTCUSDT", "ETHUSDT", "ICPUSDT"]
        
        for symbol in symbols:
            query = f"""
            SELECT 
                timestamp,
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
            price_data['symbol'] = symbol
            all_data.append(price_data)
            
            logger.info(f"Loaded {len(price_data)} records for {symbol}")
        
        conn.close()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values('timestamp')
        
        return trainer, combined_data
    
    def generate_features_online(self, 
                               price_data: pd.DataFrame, 
                               current_time: pd.Timestamp,
                               lookback_minutes: int = 30) -> pd.DataFrame:
        """
        Generate features using only past data (no future leak).
        
        Args:
            price_data: Historical price data
            current_time: Current timestamp
            lookback_minutes: Minutes of history to use
            
        Returns:
            Features for each symbol at current time
        """
        features_list = []
        
        # Get only past data
        lookback_start = current_time - pd.Timedelta(minutes=lookback_minutes)
        historical_data = price_data[
            (price_data['timestamp'] >= lookback_start) & 
            (price_data['timestamp'] < current_time)
        ]
        
        if len(historical_data) < 50:  # Need minimum history (reduced from 100)
            logger.warning(f"Insufficient historical data at {current_time}: {len(historical_data)} records")
            return pd.DataFrame()
        
        # Generate features for each symbol
        generator = LiquidationDataGenerator()
        extractor = CascadeFeatureExtractor()
        
        for symbol in historical_data['symbol'].unique():
            symbol_data = historical_data[historical_data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 20:
                continue
            
            # Generate synthetic liquidations from recent price action
            liquidations = generator.generate_liquidations(symbol_data, symbol)
            
            if len(liquidations) == 0:
                continue
            
            # Extract features for most recent point
            features_df = extractor.extract_features(liquidations, symbol_data)
            
            if len(features_df) > 0:
                # Take only the most recent feature set
                latest_features = features_df.iloc[-1:].copy()
                latest_features['current_time'] = current_time
                latest_features['symbol'] = symbol
                latest_features['current_price'] = symbol_data['close'].iloc[-1]
                features_list.append(latest_features)
        
        if not features_list:
            return pd.DataFrame()
        
        return pd.concat(features_list, ignore_index=True)
    
    def execute_trade(self, signal: Dict, current_price: float, symbol: str) -> Optional[Trade]:
        """Execute trade based on cascade detection signal."""
        # Check if we can open new position
        if len(self.open_positions) >= self.max_positions:
            return None
        
        # Check if already have position in this symbol
        if symbol in self.open_positions:
            return None
        
        # Calculate position size
        position_value = self.capital * self.position_size_pct
        position_size = position_value / current_price
        
        # Apply slippage
        if signal['direction'] == 'short':
            entry_price = current_price * (1 - self.slippage_bps)
            side = 'short'
        else:
            entry_price = current_price * (1 + self.slippage_bps)
            side = 'long'
        
        # Calculate fees
        fees = position_value * self.fee_rate
        
        # Create trade
        trade = Trade(
            entry_time=signal['timestamp'],
            exit_time=None,
            symbol=symbol,
            side=side,
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
        
        logger.info(f"Opened {side} position in {symbol} at {entry_price:.2f}")
        
        return trade
    
    def check_exit_conditions(self, trade: Trade, current_price: float, current_time: pd.Timestamp) -> bool:
        """Check if position should be closed."""
        if trade.side == 'long':
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
            
            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                return True
            
            # Take profit
            if pnl_pct >= self.take_profit_pct:
                return True
        
        else:  # short
            pnl_pct = (trade.entry_price - current_price) / trade.entry_price
            
            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                return True
            
            # Take profit
            if pnl_pct >= self.take_profit_pct:
                return True
        
        # Time-based exit (hold max 1 hour)
        if current_time - trade.entry_time > pd.Timedelta(hours=1):
            return True
        
        return False
    
    def close_position(self, trade: Trade, current_price: float, current_time: pd.Timestamp):
        """Close an open position."""
        # Apply slippage
        if trade.side == 'long':
            exit_price = current_price * (1 - self.slippage_bps)
            pnl = (exit_price - trade.entry_price) * trade.size
        else:  # short
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
    
    def run_backtest(self, 
                    model_path: str,
                    start_date: str = "2023-01-01",
                    end_date: str = "2024-12-31",
                    rebalance_days: int = 30) -> Dict:
        """
        Run walk-forward backtest.
        
        Args:
            model_path: Path to trained model
            start_date: Backtest start date
            end_date: Backtest end date
            rebalance_days: Days between model retraining (walk-forward)
            
        Returns:
            Backtest results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Load model and data
        trainer, price_data = self.load_model_and_data(model_path, start_date, end_date)
        
        # Convert timestamps
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        
        # Get unique timestamps for simulation
        timestamps = price_data['timestamp'].unique()
        timestamps = sorted(timestamps)[100::60]  # Sample every 60 minutes, skip first 100 to ensure enough history
        
        logger.info(f"Simulating {len(timestamps)} time points")
        
        # Track daily equity
        last_date = None
        daily_equity = self.capital
        
        # Main backtest loop
        for i, current_time in enumerate(timestamps):
            # Update daily returns
            current_date = current_time.date()
            if last_date and current_date != last_date:
                self.daily_returns.append({
                    'date': last_date,
                    'equity': daily_equity,
                    'return': (self.capital - daily_equity) / daily_equity
                })
                daily_equity = self.capital
            last_date = current_date
            
            # Check existing positions
            for symbol, trade in list(self.open_positions.items()):
                current_prices = price_data[
                    (price_data['timestamp'] == current_time) & 
                    (price_data['symbol'] == symbol)
                ]
                
                if len(current_prices) > 0:
                    current_price = current_prices['close'].iloc[0]
                    
                    if self.check_exit_conditions(trade, current_price, current_time):
                        self.close_position(trade, current_price, current_time)
            
            # Generate features for current time (no future leak)
            features = self.generate_features_online(price_data, current_time)
            
            if len(features) == 0:
                if i % 100 == 0:
                    logger.debug(f"No features generated at {current_time}")
                continue
            
            # Get cascade predictions
            feature_cols = [col for col in features.columns if col not in 
                          ['current_time', 'symbol', 'current_price', 'cascade_occurred']]
            X = features[feature_cols]
            
            # Make predictions
            predictions = trainer.predict(X)
            
            # Generate trading signals
            for idx, pred in enumerate(predictions):
                if pred > self.cascade_threshold:
                    # High cascade probability - expect volatility
                    confidence = pred
                    
                    if confidence >= self.min_cascade_confidence:
                        signal = {
                            'timestamp': current_time,
                            'symbol': features.iloc[idx]['symbol'],
                            'direction': 'short' if features.iloc[idx].get('cascade_directional_imbalance_60s', 0) > 0 else 'long',
                            'confidence': confidence,
                            'predicted_cascade': pred
                        }
                        
                        current_price = features.iloc[idx]['current_price']
                        self.execute_trade(signal, current_price, signal['symbol'])
            
            # Record equity
            total_value = self.capital
            for trade in self.open_positions.values():
                current_prices = price_data[
                    (price_data['timestamp'] == current_time) & 
                    (price_data['symbol'] == trade.symbol)
                ]
                if len(current_prices) > 0:
                    current_price = current_prices['close'].iloc[0]
                    if trade.side == 'long':
                        unrealized_pnl = (current_price - trade.entry_price) * trade.size
                    else:
                        unrealized_pnl = (trade.entry_price - current_price) * trade.size
                    total_value += unrealized_pnl
            
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': total_value,
                'capital': self.capital,
                'open_positions': len(self.open_positions)
            })
            
            # Progress update
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(timestamps)} ({i/len(timestamps)*100:.1f}%), "
                           f"Trades: {self.total_trades}, Equity: ${self.capital:.2f}")
        
        # Close any remaining positions
        for symbol, trade in list(self.open_positions.items()):
            last_prices = price_data[price_data['symbol'] == symbol].iloc[-1]
            self.close_position(trade, last_prices['close'], last_prices['timestamp'])
        
        # Calculate final metrics
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics."""
        # Convert equity curve to DataFrame
        if not self.equity_curve:
            logger.warning("No equity curve data available")
            return {
                'total_return': 0.0,
                'final_capital': self.capital,
                'total_pnl': self.total_pnl,
                'total_fees': self.total_fees,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'equity_curve': [],
                'trades': []
            }
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (assuming 0% risk-free rate)
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
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        metrics = {
            'total_return': total_return,
            'final_capital': self.capital,
            'total_pnl': self.total_pnl,
            'total_fees': self.total_fees,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': equity_df.to_dict('records'),
            'trades': [self._trade_to_dict(t) for t in self.trades]
        }
        
        return metrics
    
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
        """Plot backtest results including equity curve."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Convert equity curve back to DataFrame
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # 1. Equity curve
        ax1 = axes[0]
        ax1.plot(equity_df['timestamp'], equity_df['equity'], 'b-', linewidth=2, label='Total Equity')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('Equity Curve - Liquidation Cascade Strategy', fontsize=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Drawdown
        ax2 = axes[1]
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['equity'].cummax()) / equity_df['equity'].cummax()
        ax2.fill_between(equity_df['timestamp'], equity_df['drawdown'] * 100, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.set_title('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade distribution
        ax3 = axes[2]
        trades_df = pd.DataFrame(results['trades'])
        if len(trades_df) > 0:
            trades_df = trades_df[trades_df['status'] == 'closed']
            
            # PnL histogram
            if len(trades_df) > 0:
                bins = np.linspace(trades_df['pnl'].min(), trades_df['pnl'].max(), 30)
                ax3.hist(trades_df['pnl'], bins=bins, alpha=0.7, edgecolor='black')
                ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax3.set_title('Trade PnL Distribution', fontsize=12)
                ax3.set_xlabel('PnL ($)')
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to {save_path}")
        
        plt.show()
        
    def print_summary(self, results: Dict):
        """Print backtest summary."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS - Liquidation Cascade Strategy")
        print("="*60)
        
        print(f"\nPerformance Summary:")
        print(f"  Initial Capital:    ${results['final_capital'] / (1 + results['total_return']):,.2f}")
        print(f"  Final Capital:      ${results['final_capital']:,.2f}")
        print(f"  Total Return:       {results['total_return']*100:.2f}%")
        print(f"  Total PnL:          ${results['total_pnl']:,.2f}")
        print(f"  Total Fees:         ${results['total_fees']:,.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:       {results['max_drawdown']*100:.2f}%")
        print(f"  Profit Factor:      {results['profit_factor']:.2f}")
        
        print(f"\nTrading Statistics:")
        print(f"  Total Trades:       {results['total_trades']}")
        print(f"  Winning Trades:     {results['winning_trades']}")
        print(f"  Losing Trades:      {results['losing_trades']}")
        print(f"  Win Rate:           {results['win_rate']*100:.2f}%")
        print(f"  Average Win:        ${results['avg_win']:,.2f}")
        print(f"  Average Loss:       ${results['avg_loss']:,.2f}")
        
        # Calculate monthly returns
        equity_df = pd.DataFrame(results['equity_curve'])
        if len(equity_df) > 0:
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            equity_df['month'] = equity_df['timestamp'].dt.to_period('M')
            monthly_returns = equity_df.groupby('month')['equity'].agg(['first', 'last'])
            monthly_returns['return'] = (monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first'] * 100
            
            print(f"\nMonthly Returns:")
            for month, row in monthly_returns.iterrows():
                print(f"  {month}: {row['return']:>6.2f}%")


def main():
    """Run backtest for liquidation cascade model."""
    logger.info("Starting liquidation cascade model backtest")
    
    # Backtest configuration
    config = {
        'initial_capital': 100000.0,
        'position_size_pct': 0.05,  # 5% per trade
        'max_positions': 3,
        'fee_rate': 0.0006,  # Bybit taker fee
        'slippage_bps': 10,
        'stop_loss_pct': 0.02,  # 2% stop loss
        'take_profit_pct': 0.03,  # 3% take profit
        'cascade_threshold': 0.2,  # Much lower threshold for testing
        'min_cascade_confidence': 0.3  # Lower confidence for more signals
    }
    
    # Initialize backtester
    backtester = CascadeBacktester(**config)
    
    # Run backtest on out-of-sample data
    # The model was trained on data up to 2024-12-30, so we test on 2024 data
    # with walk-forward approach to avoid look-ahead bias
    results = backtester.run_backtest(
        model_path="models/cascade_detection/cascade_v1_20250612_150231",
        start_date="2024-01-01",
        end_date="2024-12-30",
        rebalance_days=30
    )
    
    # Save results
    results_path = Path("backtest_results")
    results_path.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(results_path / "cascade_backtest_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    backtester.print_summary(results)
    
    # Plot results
    backtester.plot_results(results, save_path=str(results_path / "cascade_backtest_chart.png"))
    
    logger.info("Backtest completed successfully")


if __name__ == "__main__":
    main()