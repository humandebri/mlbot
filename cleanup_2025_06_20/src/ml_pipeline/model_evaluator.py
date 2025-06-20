"""
Model evaluation and backtesting utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from ..common.logging import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and backtesting."""
    
    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate regression predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def simulate_trading(
        self,
        predictions: np.ndarray,
        actual: np.ndarray,
        timestamps: pd.DatetimeIndex,
        initial_capital: float = 10000,
        position_size: float = 0.1,
        fee_rate: float = 0.00055,
        confidence_threshold: float = 0.001
    ) -> Dict[str, Any]:
        """
        Simulate trading based on predictions.
        
        Args:
            predictions: Model predictions (expected returns)
            actual: Actual returns
            timestamps: Timestamps for each prediction
            initial_capital: Starting capital
            position_size: Position size as fraction of capital
            fee_rate: Trading fee rate
            confidence_threshold: Minimum prediction to trade
            
        Returns:
            Trading simulation results
        """
        capital = initial_capital
        positions = []
        returns = []
        
        for i in range(len(predictions)):
            pred = predictions[i]
            
            # Trade if prediction exceeds threshold
            if abs(pred) > confidence_threshold:
                # Calculate position size
                trade_size = capital * position_size
                
                # Simulate trade
                if pred > 0:  # Long
                    entry_fee = trade_size * fee_rate
                    exit_fee = trade_size * (1 + actual[i]) * fee_rate
                    pnl = trade_size * actual[i] - entry_fee - exit_fee
                else:  # Short
                    entry_fee = trade_size * fee_rate
                    exit_fee = trade_size * (1 - actual[i]) * fee_rate
                    pnl = -trade_size * actual[i] - entry_fee - exit_fee
                
                # Update capital
                capital += pnl
                returns.append(pnl / (capital - pnl))
                
                positions.append({
                    'timestamp': timestamps[i],
                    'side': 'long' if pred > 0 else 'short',
                    'size': trade_size,
                    'prediction': pred,
                    'actual': actual[i],
                    'pnl': pnl,
                    'capital': capital
                })
        
        if not returns:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0
            }
        
        # Calculate metrics
        returns_series = pd.Series(returns)
        capital_series = pd.Series([p['capital'] for p in positions])
        
        # Drawdown calculation
        peak = capital_series.expanding().max()
        drawdown = (capital_series - peak) / peak
        
        return {
            'total_return': (capital - initial_capital) / initial_capital,
            'sharpe_ratio': returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0,
            'max_drawdown': drawdown.min(),
            'win_rate': len([p for p in positions if p['pnl'] > 0]) / len(positions),
            'total_trades': len(positions),
            'avg_trade': np.mean([p['pnl'] for p in positions]),
            'best_trade': max([p['pnl'] for p in positions]),
            'worst_trade': min([p['pnl'] for p in positions]),
            'final_capital': capital
        }