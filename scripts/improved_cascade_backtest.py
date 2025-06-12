#!/usr/bin/env python3
"""
Improved liquidation cascade backtest with higher trading frequency.

Key improvements:
1. Relaxed signal thresholds based on actual market data analysis
2. OR logic instead of restrictive AND logic
3. Removed restrictive direction filtering
4. Higher time sampling frequency
5. Better position management
6. Multiple signal types
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
    signal_type: str = 'cascade'  # Type of signal that generated this trade


class ImprovedBacktester:
    """
    Improved backtester with relaxed conditions for higher trading frequency.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 position_size_pct: float = 0.03,  # Increased from 2% to 3%
                 max_positions: int = 5,  # Increased from 3 to 5
                 fee_rate: float = 0.0006,
                 slippage_bps: float = 10,
                 stop_loss_pct: float = 0.015,
                 take_profit_pct: float = 0.025,
                 # RELAXED THRESHOLDS based on data analysis
                 volatility_threshold: float = 0.003,  # 0.3% from 1.5%
                 momentum_threshold: float = 0.005,    # 0.5% from 1.0%
                 volume_threshold: float = 1.2,       # 1.2x from 1.5x
                 # Additional thresholds for multiple signal types
                 strong_momentum_threshold: float = 0.008,  # 0.8%
                 volatility_spike_threshold: float = 0.004, # 0.4%
                 volatility_change_threshold: float = 0.5): # 50% increase
        """Initialize improved backtester."""
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps / 10000.0
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Relaxed thresholds
        self.volatility_threshold = volatility_threshold
        self.momentum_threshold = momentum_threshold
        self.volume_threshold = volume_threshold
        self.strong_momentum_threshold = strong_momentum_threshold
        self.volatility_spike_threshold = volatility_spike_threshold
        self.volatility_change_threshold = volatility_change_threshold
        
        # Trading state
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.equity_curve = []
        self.signals = []
        self.signal_stats = {'cascade': 0, 'momentum': 0, 'volatility': 0, 'trend': 0}
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0
    
    def extract_price_features(self, price_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract enhanced features from price data."""
        df = price_data[price_data['symbol'] == symbol].copy()
        df = df.sort_values('timestamp')
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Price momentum features (multiple timeframes)
        df['price_change_1m'] = df['close'].pct_change(1)
        df['price_change_3m'] = df['close'].pct_change(3)
        df['price_change_5m'] = df['close'].pct_change(5)
        df['price_change_10m'] = df['close'].pct_change(10)
        df['price_change_15m'] = df['close'].pct_change(15)
        df['price_change_30m'] = df['close'].pct_change(30)
        df['price_change_60m'] = df['close'].pct_change(60)
        
        # Volatility features (multiple windows)
        df['volatility_5m'] = df['returns'].rolling(5).std()
        df['volatility_10m'] = df['returns'].rolling(10).std()
        df['volatility_15m'] = df['returns'].rolling(15).std()
        df['volatility_30m'] = df['returns'].rolling(30).std()
        df['volatility_60m'] = df['returns'].rolling(60).std()
        
        # Volatility changes and spikes
        df['volatility_change_5m'] = df['volatility_5m'].pct_change(5)
        df['volatility_change_10m'] = df['volatility_10m'].pct_change(10)
        df['volatility_spike_5m'] = df['volatility_5m'] / df['volatility_30m']
        
        # Volume features
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_30'] = df['volume'].rolling(30).mean()
        df['volume_ratio_10'] = df['volume'] / df['volume_ma_10']
        df['volume_ratio_30'] = df['volume'] / df['volume_ma_30']
        df['volume_surge'] = df['volume'] / df['volume'].rolling(60).mean()
        
        # Price range and momentum
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_ma'] = df['hl_range'].rolling(20).mean()
        df['hl_expansion'] = df['hl_range'] / df['hl_range_ma']
        
        # Trend and momentum indicators
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['trend_short'] = (df['close'] - df['sma_10']) / df['sma_10']
        df['trend_medium'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['trend_strength'] = (df['sma_10'] - df['sma_20']) / df['sma_20']
        
        # Price acceleration
        df['momentum_acceleration'] = df['price_change_5m'] - df['price_change_10m']
        df['momentum_persistence'] = np.sign(df['price_change_5m']) == np.sign(df['price_change_15m'])
        
        # Enhanced cascade-like features
        df['price_velocity'] = abs(df['price_change_5m'])
        df['volatility_momentum'] = df['volatility_10m'] / df['volatility_30m']
        df['volume_price_correlation'] = df['volume_ratio_10'] * abs(df['price_change_5m'])
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def generate_multiple_signal_types(self, features: pd.DataFrame, model: Optional[ModelTrainer] = None) -> pd.DataFrame:
        """Generate multiple types of trading signals."""
        signals = []
        
        for idx in range(len(features)):
            row = features.iloc[idx]
            signal_generated = False
            signal_type = None
            confidence = 0.0
            
            # 1. CASCADE SIGNAL (Original but relaxed)
            cascade_signal = (
                abs(row['volatility_30m']) > self.volatility_threshold or  # Changed to OR
                (abs(row['price_change_15m']) > self.momentum_threshold and 
                 row['volume_ratio_30'] > self.volume_threshold)
            )
            
            if cascade_signal:
                signal_generated = True
                signal_type = 'cascade'
                confidence = min(abs(row['volatility_30m']) / self.volatility_threshold, 2.0)
                self.signal_stats['cascade'] += 1
            
            # 2. MOMENTUM SIGNAL (Strong directional moves with volume)
            momentum_signal = (
                abs(row['price_change_15m']) > self.strong_momentum_threshold and
                row['volume_ratio_10'] > 2.0 and
                row['momentum_persistence']  # Consistent direction
            )
            
            if momentum_signal and not signal_generated:
                signal_generated = True
                signal_type = 'momentum' 
                confidence = abs(row['price_change_15m']) / self.strong_momentum_threshold
                self.signal_stats['momentum'] += 1
            
            # 3. VOLATILITY SPIKE SIGNAL
            volatility_spike_signal = (
                row['volatility_10m'] > self.volatility_spike_threshold and
                row['volatility_change_5m'] > self.volatility_change_threshold and
                row['volume_ratio_10'] > 1.5
            )
            
            if volatility_spike_signal and not signal_generated:
                signal_generated = True
                signal_type = 'volatility'
                confidence = row['volatility_10m'] / self.volatility_spike_threshold
                self.signal_stats['volatility'] += 1
            
            # 4. TREND BREAKOUT SIGNAL
            trend_breakout_signal = (
                abs(row['trend_short']) > 0.01 and  # 1% away from 10-period average
                abs(row['momentum_acceleration']) > 0.005 and  # Accelerating
                row['hl_expansion'] > 1.3 and  # Range expansion
                row['volume_surge'] > 1.8  # Volume confirmation
            )
            
            if trend_breakout_signal and not signal_generated:
                signal_generated = True
                signal_type = 'trend'
                confidence = abs(row['trend_short']) / 0.01
                self.signal_stats['trend'] += 1
            
            # Generate signal if any condition met
            if signal_generated:
                # SIMPLIFIED DIRECTION LOGIC (removed restrictive filtering)
                if row['price_change_10m'] < 0:
                    direction = 'short'  # Recent downward momentum
                else:
                    direction = 'long'   # Recent upward momentum
                
                signals.append({
                    'timestamp': pd.Timestamp(row.name),
                    'symbol': row['symbol'],
                    'direction': direction,
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'volatility': row['volatility_30m'],
                    'momentum': row['price_change_15m'],
                    'volume_ratio': row.get('volume_ratio_30', row.get('volume_ratio_10', 1.0)),
                    'price_velocity': row['price_velocity'],
                    'trend_strength': row.get('trend_strength', 0.0)
                })
        
        return pd.DataFrame(signals)
    
    def run_backtest(self, 
                    model_path: Optional[str] = None,
                    start_date: str = "2024-01-01",
                    end_date: str = "2024-12-30") -> Dict:
        """Run improved backtest with higher frequency."""
        logger.info(f"Starting improved backtest from {start_date} to {end_date}")
        
        # Load model if provided (but we'll focus on rule-based for now)
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
            
            # Extract enhanced features
            features = self.extract_price_features(price_data.reset_index(), symbol)
            features['symbol'] = symbol
            all_features.append(features)
            
            logger.info(f"Extracted {len(features)} feature points for {symbol}")
        
        conn.close()
        
        # Combine all features
        all_features_df = pd.concat(all_features, ignore_index=False)
        all_features_df = all_features_df.sort_index()
        
        # Generate trading signals with multiple types
        signals_df = self.generate_multiple_signal_types(all_features_df, model)
        logger.info(f"Generated {len(signals_df)} trading signals")
        logger.info(f"Signal breakdown: {self.signal_stats}")
        
        # Store signals for analysis
        self.signals = signals_df.to_dict('records')
        
        # Simulate trading with HIGHER FREQUENCY (every 2 minutes instead of 10)
        timestamps = all_features_df.index.unique()[::2]  # Every 2 minutes
        
        logger.info(f"Simulating {len(timestamps)} time points (every 2 minutes)")
        
        for i, timestamp in enumerate(timestamps):
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
                # IMPROVED POSITION MANAGEMENT
                if len(self.open_positions) < self.max_positions:
                    # Allow re-entry in same symbol after some time
                    can_trade = True
                    if signal['symbol'] in self.open_positions:
                        can_trade = False
                    
                    # Check if we had a recent trade in this symbol (cooldown)
                    recent_trades = [t for t in self.trades if 
                                   t.symbol == signal['symbol'] and 
                                   t.exit_time and 
                                   (pd.Timestamp(timestamp) - pd.Timestamp(t.exit_time)).total_seconds() < 1800]  # 30 min cooldown
                    
                    if len(recent_trades) > 0:
                        can_trade = False
                    
                    if can_trade:
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
            
            # Progress update
            if i % 1000 == 0:
                logger.info(f"Progress: {i}/{len(timestamps)} ({i/len(timestamps)*100:.1f}%), "
                           f"Trades: {self.total_trades}, Signals: {len(self.signals)}, Equity: ${total_value:.2f}")
        
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
            status='open',
            signal_type=signal.get('signal_type', 'unknown')
        )
        
        # Update capital
        self.capital -= fees
        
        # Store trade
        self.trades.append(trade)
        self.open_positions[symbol] = trade
        self.total_trades += 1
        
        logger.info(f"Opened {trade.side} position in {symbol} at {entry_price:.2f} "
                   f"({signal.get('signal_type', 'unknown')} signal, confidence: {signal.get('confidence', 0):.2f})")
        
        return trade
    
    def check_and_close_position(self, trade: Trade, current_price: float, current_time: pd.Timestamp):
        """Check if position should be closed."""
        if trade.side == 'long':
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - current_price) / trade.entry_price
        
        # Check exit conditions
        should_close = False
        exit_reason = ""
        
        if pnl_pct <= -self.stop_loss_pct:
            should_close = True
            exit_reason = "stop_loss"
        elif pnl_pct >= self.take_profit_pct:
            should_close = True
            exit_reason = "take_profit"
        elif pd.Timestamp(current_time) - pd.Timestamp(trade.entry_time) > pd.Timedelta(hours=2):
            should_close = True
            exit_reason = "time_exit"
        
        if should_close:
            self.close_position(trade, current_price, current_time)
            logger.debug(f"{exit_reason.upper()} triggered for {trade.symbol} ({trade.signal_type})")
    
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
        exit_fees = abs(pnl) * self.fee_rate if pnl != 0 else current_price * trade.size * self.fee_rate
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
        
        hold_time = (pd.Timestamp(current_time) - pd.Timestamp(trade.entry_time)).total_seconds() / 3600  # hours
        logger.info(f"Closed {trade.side} position in {trade.symbol} at {exit_price:.2f}, "
                   f"PnL: ${pnl:.2f}, Hold: {hold_time:.1f}h, Type: {trade.signal_type}")
    
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
                'signal_stats': self.signal_stats,
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
        
        # Signal type performance
        signal_performance = {}
        for signal_type in self.signal_stats.keys():
            type_trades = [t for t in closed_trades if getattr(t, 'signal_type', '') == signal_type]
            if type_trades:
                type_pnl = sum(t.pnl for t in type_trades)
                type_wins = sum(1 for t in type_trades if t.pnl > 0)
                signal_performance[signal_type] = {
                    'total_pnl': type_pnl,
                    'trade_count': len(type_trades),
                    'win_rate': type_wins / len(type_trades),
                    'avg_pnl': type_pnl / len(type_trades)
                }
        
        return {
            'total_return': total_return,
            'final_capital': self.capital,
            'total_pnl': self.total_pnl,
            'total_fees': self.total_fees,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'signal_count': len(self.signals),
            'signal_stats': self.signal_stats,
            'signal_performance': signal_performance,
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
            'status': trade.status,
            'signal_type': getattr(trade, 'signal_type', 'unknown')
        }
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Plot improved backtest results."""
        if not results['equity_curve']:
            logger.warning("No equity curve data to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
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
        
        # 3. Signal type breakdown
        ax3 = axes[0, 2]
        signal_counts = list(results['signal_stats'].values())
        signal_labels = list(results['signal_stats'].keys())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        ax3.pie(signal_counts, labels=signal_labels, autopct='%1.1f%%', colors=colors)
        ax3.set_title('Signal Type Distribution', fontsize=14)
        
        # 4. Trade PnL distribution
        ax4 = axes[1, 0]
        trades_df = pd.DataFrame(results['trades'])
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            closed_trades = trades_df[trades_df['status'] == 'closed']
            if len(closed_trades) > 0:
                ax4.hist(closed_trades['pnl'], bins=20, alpha=0.7, edgecolor='black')
                ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax4.set_title('Trade PnL Distribution', fontsize=14)
                ax4.set_xlabel('PnL ($)')
                ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3)
        
        # 5. Monthly returns
        ax5 = axes[1, 1]
        equity_df.set_index('timestamp', inplace=True)
        monthly_returns = equity_df['equity'].resample('M').agg(['first', 'last'])
        monthly_returns['return'] = (monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first'] * 100
        
        if len(monthly_returns) > 0:
            colors = ['green' if r > 0 else 'red' for r in monthly_returns['return']]
            ax5.bar(range(len(monthly_returns)), monthly_returns['return'], color=colors, alpha=0.7)
            ax5.set_title('Monthly Returns (%)', fontsize=14)
            ax5.set_xlabel('Month')
            ax5.set_ylabel('Return (%)')
            ax5.grid(True, alpha=0.3)
            
            # Set month labels
            month_labels = [idx.strftime('%Y-%m') for idx in monthly_returns.index]
            ax5.set_xticks(range(len(month_labels)))
            ax5.set_xticklabels(month_labels, rotation=45, ha='right')
        
        # 6. Signal performance by type
        ax6 = axes[1, 2]
        if 'signal_performance' in results and results['signal_performance']:
            perf_data = results['signal_performance']
            types = list(perf_data.keys())
            pnls = [perf_data[t]['total_pnl'] for t in types]
            colors = ['green' if p > 0 else 'red' for p in pnls]
            
            ax6.bar(types, pnls, color=colors, alpha=0.7)
            ax6.set_title('PnL by Signal Type', fontsize=14)
            ax6.set_xlabel('Signal Type')
            ax6.set_ylabel('Total PnL ($)')
            ax6.grid(True, alpha=0.3)
            ax6.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Improved Liquidation Cascade Strategy Results', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self, results: Dict):
        """Print detailed backtest summary."""
        print("\n" + "="*60)
        print("IMPROVED BACKTEST RESULTS")
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
        print(f"  Signal->Trade Rate: {results['total_trades']/results['signal_count']*100:.1f}%" if results['signal_count'] > 0 else "  Signal->Trade Rate: N/A")
        print(f"  Winning Trades:     {results['winning_trades']}")
        print(f"  Losing Trades:      {results['losing_trades']}")
        print(f"  Win Rate:           {results['win_rate']*100:.2f}%")
        print(f"  Average Win:        ${results['avg_win']:,.2f}")
        print(f"  Average Loss:       ${results['avg_loss']:,.2f}")
        
        print(f"\nSignal Breakdown:")
        for signal_type, count in results['signal_stats'].items():
            print(f"  {signal_type.capitalize():12}: {count:4d} signals")
        
        print(f"\nSignal Performance:")
        if 'signal_performance' in results:
            for signal_type, perf in results['signal_performance'].items():
                print(f"  {signal_type.capitalize():12}: ${perf['total_pnl']:6.0f} PnL, "
                      f"{perf['trade_count']:2d} trades, {perf['win_rate']*100:4.1f}% win rate")


def main():
    """Run improved backtest."""
    logger.info("Starting improved liquidation cascade backtest")
    
    # Initialize improved backtester with relaxed parameters
    backtester = ImprovedBacktester(
        initial_capital=100000.0,
        position_size_pct=0.03,  # 3% per trade
        max_positions=5,         # Up to 5 positions
        fee_rate=0.0006,
        slippage_bps=10,
        stop_loss_pct=0.015,
        take_profit_pct=0.025,
        # Relaxed thresholds based on analysis
        volatility_threshold=0.003,    # 0.3% (was 1.5%)
        momentum_threshold=0.005,      # 0.5% (was 1.0%)
        volume_threshold=1.2,          # 1.2x (was 1.5x)
        strong_momentum_threshold=0.008,
        volatility_spike_threshold=0.004,
        volatility_change_threshold=0.5
    )
    
    # Run backtest
    results = backtester.run_backtest(
        model_path=None,  # Start with rule-based for clarity
        start_date="2024-01-01",
        end_date="2024-12-30"
    )
    
    # Print summary
    backtester.print_summary(results)
    
    # Save results
    results_path = Path("backtest_results")
    results_path.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(results_path / "improved_backtest_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Plot results
    backtester.plot_results(results, save_path=str(results_path / "improved_backtest_chart.png"))
    
    logger.info("Improved backtest completed successfully")


if __name__ == "__main__":
    main()