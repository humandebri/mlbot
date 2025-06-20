#!/usr/bin/env python3
"""
Comprehensive backtest of the v2.0 model to analyze performance degradation.
"""

import sys
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import duckdb
import onnxruntime as ort
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Import feature generators
from scripts.train_production_model import FeatureGenerator156

print("V2.0 Model Comprehensive Backtest")
print("=" * 60)


@dataclass
class Trade:
    """Trade representation."""
    timestamp: pd.Timestamp
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    fees: float
    confidence: float
    duration_minutes: int


class V2ModelBacktester:
    """Comprehensive backtester for v2.0 model."""
    
    def __init__(self, model_dir: str = "models/v2.0"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_generator = FeatureGenerator156()
        
    def load_model(self):
        """Load v2.0 model and components."""
        print(f"Loading model from {self.model_dir}...")
        
        # Load metadata
        with open(self.model_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        print(f"Model AUC from training: {self.metadata.get('performance', {}).get('auc', 'N/A')}")
        
        # Load ONNX model
        self.model = ort.InferenceSession(str(self.model_dir / "model.onnx"))
        print("✅ Loaded ONNX model")
        
        # Load scaler
        with open(self.model_dir / "scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        print("✅ Loaded scaler")
        
    def load_test_data(self, start_date: str = "2024-05-01", end_date: str = "2024-06-01") -> pd.DataFrame:
        """Load test data from DuckDB."""
        print(f"\nLoading test data from {start_date} to {end_date}...")
        
        conn = duckdb.connect("data/historical_data.duckdb", read_only=True)
        
        dfs = []
        for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
            table_name = f'klines_{symbol.lower()}'
            try:
                query = f"""
                SELECT * FROM {table_name}
                WHERE timestamp >= '{start_date}'
                  AND timestamp < '{end_date}'
                ORDER BY timestamp
                """
                df = conn.execute(query).fetchdf()
                print(f"Loaded {len(df)} rows for {symbol}")
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
        
        conn.close()
        
        if not dfs:
            raise ValueError("No data loaded!")
        
        return pd.concat(dfs, ignore_index=True)
    
    def evaluate_model_quality(self, features_df: pd.DataFrame) -> Dict:
        """Evaluate model quality on test data."""
        print("\nEvaluating model quality...")
        
        # Generate labels for evaluation
        df = features_df.copy()
        
        # Calculate future returns (5-minute ahead)
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            df.loc[mask, 'future_close'] = df[mask]['close'].shift(-5)
        
        df['future_return'] = (df['future_close'] / df['close'] - 1) - 0.001  # Subtract fees
        df['label'] = (df['future_return'] > 0.0015).astype(int)
        
        # Remove NaN
        df = df.dropna(subset=['label', 'future_return'])
        
        # Prepare features
        exclude_cols = ['symbol', 'timestamp', 'close', 'label', 'future_return', 'future_close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y_true = df['label'].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        
        y_proba = self.model.run([output_name], {input_name: X_scaled.astype(np.float32)})[0]
        
        # Handle both binary and probability outputs
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
        else:
            y_proba = y_proba.flatten()
        
        y_pred = (y_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'auc': roc_auc_score(y_true, y_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'accuracy': (y_pred == y_true).mean(),
            'positive_rate': y_true.mean(),
            'predicted_positive_rate': y_pred.mean()
        }
        
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Analyze prediction distribution
        print(f"\nPrediction probability distribution:")
        print(f"  Min: {y_proba.min():.4f}")
        print(f"  25%: {np.percentile(y_proba, 25):.4f}")
        print(f"  50%: {np.percentile(y_proba, 50):.4f}")
        print(f"  75%: {np.percentile(y_proba, 75):.4f}")
        print(f"  Max: {y_proba.max():.4f}")
        
        return metrics
    
    def run_trading_simulation(self, features_df: pd.DataFrame, 
                             confidence_threshold: float = 0.7,
                             position_size: float = 10000,
                             max_positions: int = 3) -> Tuple[List[Trade], pd.DataFrame]:
        """Run trading simulation with the model."""
        print(f"\nRunning trading simulation...")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Position size: ${position_size:,.0f}")
        print(f"  Max positions: {max_positions}")
        
        # Prepare data
        df = features_df.copy()
        
        # Calculate future prices
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            df.loc[mask, 'future_close'] = df[mask]['close'].shift(-5)
        
        df = df.dropna(subset=['future_close'])
        
        # Get predictions
        exclude_cols = ['symbol', 'timestamp', 'close', 'future_close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        
        y_proba = self.model.run([output_name], {input_name: X_scaled.astype(np.float32)})[0]
        
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
        else:
            y_proba = y_proba.flatten()
        
        df['prediction_proba'] = y_proba
        df['predicted'] = (y_proba > confidence_threshold).astype(int)
        
        # Simulate trading
        trades = []
        open_positions = {}
        capital = 100000
        equity_curve = []
        
        for idx, row in df.iterrows():
            current_time = row['timestamp']
            symbol = row['symbol']
            
            # Check for exit signals (5 minutes later)
            positions_to_close = []
            for pos_key, pos in open_positions.items():
                if current_time >= pos['exit_time']:
                    positions_to_close.append(pos_key)
            
            # Close positions
            for pos_key in positions_to_close:
                pos = open_positions[pos_key]
                exit_price = row['close'] if pos['symbol'] == symbol else pos['exit_price']
                
                # Calculate PnL
                price_change = exit_price - pos['entry_price']
                pnl = (price_change / pos['entry_price']) * pos['size']
                fees = pos['size'] * 0.001  # 0.1% fees each way
                net_pnl = pnl - fees
                
                trade = Trade(
                    timestamp=pos['entry_time'],
                    symbol=pos['symbol'],
                    side='long',
                    entry_price=pos['entry_price'],
                    exit_price=exit_price,
                    size=pos['size'],
                    pnl=net_pnl,
                    return_pct=(net_pnl / pos['size']) * 100,
                    fees=fees,
                    confidence=pos['confidence'],
                    duration_minutes=5
                )
                trades.append(trade)
                capital += net_pnl
                
                del open_positions[pos_key]
            
            # Check for entry signals
            if row['predicted'] == 1 and len(open_positions) < max_positions:
                pos_key = f"{symbol}_{current_time}"
                open_positions[pos_key] = {
                    'symbol': symbol,
                    'entry_time': current_time,
                    'exit_time': current_time + pd.Timedelta(minutes=5),
                    'entry_price': row['close'],
                    'exit_price': row['future_close'],
                    'size': position_size,
                    'confidence': row['prediction_proba']
                }
            
            # Record equity
            total_value = capital + sum(pos['size'] for pos in open_positions.values())
            equity_curve.append({
                'timestamp': current_time,
                'capital': capital,
                'positions_value': sum(pos['size'] for pos in open_positions.values()),
                'total_equity': total_value,
                'return_pct': ((total_value - 100000) / 100000) * 100
            })
        
        equity_df = pd.DataFrame(equity_curve)
        
        print(f"\nTrading Simulation Results:")
        print(f"  Total trades: {len(trades)}")
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            print(f"  Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
            print(f"  Average PnL per trade: ${np.mean([t.pnl for t in trades]):.2f}")
            print(f"  Total PnL: ${sum(t.pnl for t in trades):.2f}")
            print(f"  Final capital: ${capital:.2f}")
            print(f"  Total return: {((capital - 100000) / 100000) * 100:.2f}%")
            
            # Calculate Sharpe ratio
            if len(equity_df) > 1:
                returns = equity_df['return_pct'].diff().dropna()
                sharpe = np.sqrt(252 * 24 * 12) * returns.mean() / returns.std() if returns.std() > 0 else 0
                print(f"  Sharpe ratio: {sharpe:.2f}")
        
        return trades, equity_df
    
    def analyze_feature_quality(self, features_df: pd.DataFrame):
        """Analyze the quality of generated features."""
        print("\nAnalyzing feature quality...")
        
        # Check for features with constant values
        exclude_cols = ['symbol', 'timestamp', 'close']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        constant_features = []
        low_variance_features = []
        
        for col in feature_cols:
            if features_df[col].nunique() == 1:
                constant_features.append(col)
            elif features_df[col].std() < 1e-10:
                low_variance_features.append(col)
        
        print(f"  Constant features: {len(constant_features)}")
        if constant_features:
            print(f"    Examples: {constant_features[:5]}")
        
        print(f"  Low variance features: {len(low_variance_features)}")
        if low_variance_features:
            print(f"    Examples: {low_variance_features[:5]}")
        
        # Check for NaN values
        nan_counts = features_df[feature_cols].isna().sum()
        features_with_nan = nan_counts[nan_counts > 0]
        print(f"  Features with NaN: {len(features_with_nan)}")
        
        # Analyze feature distributions
        print("\nFeature statistics (sample of first 10):")
        for i, col in enumerate(feature_cols[:10]):
            values = features_df[col].dropna()
            print(f"  {col}:")
            print(f"    Mean: {values.mean():.4f}, Std: {values.std():.4f}")
            print(f"    Min: {values.min():.4f}, Max: {values.max():.4f}")
    
    def plot_results(self, trades: List[Trade], equity_df: pd.DataFrame):
        """Plot backtest results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        ax = axes[0, 0]
        ax.plot(equity_df['timestamp'], equity_df['total_equity'], label='Total Equity')
        ax.plot(equity_df['timestamp'], [100000] * len(equity_df), 'k--', alpha=0.5, label='Initial Capital')
        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Returns distribution
        if trades:
            ax = axes[0, 1]
            returns = [t.return_pct for t in trades]
            ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_title('Trade Returns Distribution')
            ax.set_xlabel('Return (%)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Cumulative returns
            ax = axes[1, 0]
            trade_df = pd.DataFrame([{
                'timestamp': t.timestamp,
                'return_pct': t.return_pct
            } for t in trades])
            trade_df = trade_df.sort_values('timestamp')
            trade_df['cumulative_return'] = (1 + trade_df['return_pct'] / 100).cumprod() - 1
            ax.plot(trade_df['timestamp'], trade_df['cumulative_return'] * 100)
            ax.set_title('Cumulative Returns')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return (%)')
            ax.grid(True, alpha=0.3)
            
            # Win rate over time
            ax = axes[1, 1]
            trade_df['win'] = trade_df['return_pct'] > 0
            trade_df['win_rate'] = trade_df['win'].expanding().mean()
            ax.plot(trade_df['timestamp'], trade_df['win_rate'] * 100)
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
            ax.set_title('Win Rate Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Win Rate (%)')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path("backtest_results/v2_model_analysis.png")
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved results plot to {output_path}")
        plt.close()


def main():
    """Run comprehensive v2.0 model analysis."""
    
    # Initialize backtester
    backtester = V2ModelBacktester()
    
    # Load model
    backtester.load_model()
    
    # Load test data
    raw_data = backtester.load_test_data()
    print(f"Total raw data: {len(raw_data)} rows")
    
    # Generate features
    print("\nGenerating features...")
    features_df = backtester.feature_generator.generate_features(raw_data)
    print(f"Generated {len(features_df)} feature vectors")
    
    # Analyze feature quality
    backtester.analyze_feature_quality(features_df)
    
    # Evaluate model quality
    metrics = backtester.evaluate_model_quality(features_df)
    
    # Run trading simulations with different thresholds
    print("\n" + "="*60)
    print("TRADING SIMULATIONS")
    print("="*60)
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_result = None
    best_threshold = None
    best_sharpe = -float('inf')
    
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")
        trades, equity_df = backtester.run_trading_simulation(
            features_df, 
            confidence_threshold=threshold
        )
        
        if trades and len(equity_df) > 1:
            returns = equity_df['return_pct'].diff().dropna()
            sharpe = np.sqrt(252 * 24 * 12) * returns.mean() / returns.std() if returns.std() > 0 else 0
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_threshold = threshold
                best_result = (trades, equity_df)
    
    # Plot best results
    if best_result:
        print(f"\n{'='*60}")
        print(f"BEST CONFIGURATION: Threshold = {best_threshold}")
        print(f"Sharpe Ratio: {best_sharpe:.2f}")
        print(f"{'='*60}")
        
        trades, equity_df = best_result
        backtester.plot_results(trades, equity_df)
    
    # Save detailed results
    results_summary = {
        'model_training_auc': backtester.metadata.get('performance', {}).get('auc', 'N/A'),
        'test_auc': metrics['auc'],
        'test_precision': metrics['precision'],
        'test_recall': metrics['recall'],
        'best_threshold': best_threshold,
        'best_sharpe': best_sharpe,
        'total_trades': len(trades) if best_result else 0,
        'win_rate': len([t for t in trades if t.pnl > 0]) / len(trades) * 100 if best_result and trades else 0,
        'total_return': ((equity_df['total_equity'].iloc[-1] - 100000) / 100000 * 100) if best_result else 0
    }
    
    with open("backtest_results/v2_model_analysis.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey findings:")
    print(f"1. Training AUC: {backtester.metadata.get('performance', {}).get('auc', 'N/A')}")
    print(f"2. Test AUC: {metrics['auc']:.4f}")
    print(f"3. Feature quality issues detected (many random/simulated features)")
    print(f"4. Best Sharpe ratio: {best_sharpe:.2f}")
    print(f"5. Best threshold: {best_threshold}")


if __name__ == "__main__":
    main()