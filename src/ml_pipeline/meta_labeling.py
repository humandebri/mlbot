"""
Meta-labeling framework for improved trading signal generation.

Meta-labeling separates the prediction problem into two parts:
1. Primary model: Predicts the side (direction) of the trade
2. Meta model: Predicts whether to take the trade and position size

This approach significantly improves precision and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
from datetime import datetime, timedelta


class MetaLabeler:
    """
    Meta-labeling implementation for liquidation-driven trading.
    
    The meta-labeling approach:
    1. Primary model predicts market direction (up/down)
    2. Meta model predicts whether the primary prediction is correct
    3. Combined signal provides high-precision trading decisions
    """
    
    def __init__(
        self,
        primary_threshold: float = 0.5,
        meta_threshold: float = 0.6,
        min_bet_size: float = 0.1,
        max_bet_size: float = 1.0
    ):
        """
        Initialize meta-labeler.
        
        Args:
            primary_threshold: Threshold for primary model predictions
            meta_threshold: Threshold for meta model predictions
            min_bet_size: Minimum position size (fraction)
            max_bet_size: Maximum position size (fraction)
        """
        self.primary_threshold = primary_threshold
        self.meta_threshold = meta_threshold
        self.min_bet_size = min_bet_size
        self.max_bet_size = max_bet_size
        
        # Models
        self.primary_model = None
        self.meta_model = None
        
        # Performance tracking
        self.primary_predictions = []
        self.meta_predictions = []
        self.outcomes = []
        
    def create_primary_labels(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        lookahead: int = 60,
        barrier_pct: float = 0.002
    ) -> pd.Series:
        """
        Create primary labels (market direction).
        
        Uses triple barrier method:
        - Upper barrier: profit target
        - Lower barrier: stop loss
        - Vertical barrier: time limit
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            lookahead: Maximum holding period (seconds)
            barrier_pct: Barrier width as percentage of price
            
        Returns:
            Series with labels: 1 (up), 0 (no trade), -1 (down)
        """
        labels = pd.Series(index=df.index, dtype=float)
        prices = df[price_col].values
        
        for i in range(len(df) - lookahead):
            entry_price = prices[i]
            future_prices = prices[i+1:i+lookahead+1]
            
            # Calculate barriers
            upper_barrier = entry_price * (1 + barrier_pct)
            lower_barrier = entry_price * (1 - barrier_pct)
            
            # Find which barrier is hit first
            upper_hits = np.where(future_prices >= upper_barrier)[0]
            lower_hits = np.where(future_prices <= lower_barrier)[0]
            
            if len(upper_hits) > 0 and len(lower_hits) > 0:
                # Both barriers hit - first one wins
                if upper_hits[0] < lower_hits[0]:
                    labels.iloc[i] = 1  # Up
                else:
                    labels.iloc[i] = -1  # Down
            elif len(upper_hits) > 0:
                labels.iloc[i] = 1  # Up
            elif len(lower_hits) > 0:
                labels.iloc[i] = -1  # Down
            else:
                # No barrier hit - check final price
                final_price = future_prices[-1]
                if final_price > entry_price * (1 + barrier_pct/2):
                    labels.iloc[i] = 1
                elif final_price < entry_price * (1 - barrier_pct/2):
                    labels.iloc[i] = -1
                else:
                    labels.iloc[i] = 0  # No trade
        
        return labels
    
    def create_meta_labels(
        self,
        df: pd.DataFrame,
        primary_predictions: pd.Series,
        primary_labels: pd.Series,
        bet_sizes: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Create meta-labels based on primary model predictions.
        
        Meta-labels indicate whether the primary model's prediction
        was correct and by how much.
        
        Args:
            df: Feature DataFrame
            primary_predictions: Primary model predictions
            primary_labels: Actual primary labels (ground truth)
            bet_sizes: Optional bet sizes from primary model
            
        Returns:
            DataFrame with meta-labels and additional info
        """
        meta_df = pd.DataFrame(index=df.index)
        
        # Binary meta-label: was primary prediction correct?
        meta_df['meta_label'] = (
            (primary_predictions > self.primary_threshold) & (primary_labels > 0) |
            (primary_predictions < -self.primary_threshold) & (primary_labels < 0)
        ).astype(int)
        
        # Confidence-weighted meta-label
        pred_confidence = np.abs(primary_predictions)
        outcome_magnitude = np.abs(primary_labels)
        
        # Meta-label incorporates both correctness and magnitude
        meta_df['weighted_meta_label'] = meta_df['meta_label'] * outcome_magnitude
        
        # Primary model confidence
        meta_df['primary_confidence'] = pred_confidence
        
        # Bet size (if provided)
        if bet_sizes is not None:
            meta_df['bet_size'] = bet_sizes
        else:
            # Default bet sizing based on confidence
            meta_df['bet_size'] = np.clip(
                pred_confidence,
                self.min_bet_size,
                self.max_bet_size
            )
        
        # Side prediction
        meta_df['predicted_side'] = np.sign(primary_predictions)
        meta_df['actual_side'] = np.sign(primary_labels)
        
        return meta_df
    
    def train_meta_model(
        self,
        features: pd.DataFrame,
        meta_labels: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the meta-model to predict primary model accuracy.
        
        Args:
            features: Feature DataFrame
            meta_labels: Meta-labels (binary)
            validation_split: Fraction for validation
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare data
        X = features.values
        y = meta_labels.values
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train meta-model (RandomForest for robustness)
        self.meta_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=50,
            class_weight='balanced',
            random_state=42
        )
        
        self.meta_model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.meta_model.predict(X_train)
        val_pred = self.meta_model.predict(X_val)
        
        metrics = {
            'train_precision': precision_score(y_train, train_pred),
            'train_recall': recall_score(y_train, train_pred),
            'train_f1': f1_score(y_train, train_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'feature_importance': dict(zip(
                features.columns,
                self.meta_model.feature_importances_
            ))
        }
        
        return metrics
    
    def generate_trading_signal(
        self,
        features: pd.DataFrame,
        primary_prediction: float,
        return_components: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, float]]]:
        """
        Generate final trading signal combining primary and meta predictions.
        
        Args:
            features: Current feature values
            primary_prediction: Primary model prediction
            return_components: Whether to return component predictions
            
        Returns:
            Trading signal dictionary (and optionally components)
        """
        # Get meta prediction
        if self.meta_model is None:
            warnings.warn("Meta model not trained, using primary signal only")
            meta_prob = 1.0 if abs(primary_prediction) > self.primary_threshold else 0.0
        else:
            meta_prob = self.meta_model.predict_proba(features.values.reshape(1, -1))[0, 1]
        
        # Determine if we should trade
        should_trade = (
            abs(primary_prediction) > self.primary_threshold and
            meta_prob > self.meta_threshold
        )
        
        # Calculate position size using Kelly-like criterion
        if should_trade:
            # Combine primary confidence and meta probability
            primary_conf = min(abs(primary_prediction), 1.0)
            combined_conf = primary_conf * meta_prob
            
            # Position size with risk scaling
            position_size = np.clip(
                combined_conf * self.max_bet_size,
                self.min_bet_size,
                self.max_bet_size
            )
            
            # Adjust for meta model confidence
            if meta_prob > 0.8:  # High confidence
                position_size *= 1.2
            elif meta_prob < 0.6:  # Low confidence
                position_size *= 0.8
                
            position_size = np.clip(position_size, self.min_bet_size, self.max_bet_size)
        else:
            position_size = 0.0
        
        signal = {
            'trade': should_trade,
            'side': 'buy' if primary_prediction > 0 else 'sell',
            'position_size': position_size,
            'confidence': meta_prob if should_trade else 0.0,
            'primary_prediction': primary_prediction,
            'meta_probability': meta_prob
        }
        
        if return_components:
            components = {
                'primary_raw': primary_prediction,
                'primary_confidence': min(abs(primary_prediction), 1.0),
                'meta_probability': meta_prob,
                'combined_confidence': signal['confidence']
            }
            return signal, components
            
        return signal
    
    def calculate_bet_size_dynamic(
        self,
        base_size: float,
        recent_performance: List[float],
        volatility: float,
        available_capital: float
    ) -> float:
        """
        Calculate dynamic bet size based on recent performance and market conditions.
        
        Args:
            base_size: Base position size from meta-labeling
            recent_performance: Recent P&L history
            volatility: Current market volatility
            available_capital: Available trading capital
            
        Returns:
            Adjusted position size
        """
        # Performance adjustment
        if len(recent_performance) >= 5:
            recent_pnl = np.array(recent_performance[-10:])
            
            # Winning streak bonus
            consecutive_wins = 0
            for pnl in reversed(recent_pnl):
                if pnl > 0:
                    consecutive_wins += 1
                else:
                    break
                    
            if consecutive_wins >= 3:
                performance_mult = 1.1
            elif consecutive_wins >= 5:
                performance_mult = 1.2
            else:
                performance_mult = 1.0
                
            # Drawdown reduction
            if np.sum(recent_pnl) < 0:
                drawdown_pct = abs(np.sum(recent_pnl)) / available_capital
                if drawdown_pct > 0.05:  # 5% drawdown
                    performance_mult *= 0.7
                elif drawdown_pct > 0.03:  # 3% drawdown
                    performance_mult *= 0.85
        else:
            performance_mult = 1.0
        
        # Volatility adjustment
        # Higher volatility = smaller position
        vol_mult = np.exp(-volatility * 10)  # Exponential decay with volatility
        vol_mult = np.clip(vol_mult, 0.5, 1.5)
        
        # Final size
        adjusted_size = base_size * performance_mult * vol_mult
        
        # Capital constraint
        max_size_by_capital = available_capital * 0.1  # Max 10% per trade
        final_size = min(adjusted_size, max_size_by_capital)
        
        return final_size
    
    def evaluate_meta_labeling_performance(
        self,
        predictions: pd.DataFrame,
        outcomes: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate the performance of meta-labeling strategy.
        
        Args:
            predictions: DataFrame with predictions and meta-labels
            outcomes: DataFrame with actual outcomes
            
        Returns:
            Performance metrics dictionary
        """
        # Filter to trades taken by meta-labeling
        meta_trades = predictions[predictions['meta_probability'] > self.meta_threshold]
        
        if len(meta_trades) == 0:
            return {'error': 'No trades taken by meta-labeling'}
        
        # Calculate metrics
        metrics = {}
        
        # Precision (accuracy of trades taken)
        correct_trades = outcomes.loc[meta_trades.index, 'profitable'].sum()
        metrics['precision'] = correct_trades / len(meta_trades)
        
        # Coverage (percentage of profitable trades captured)
        total_profitable = outcomes['profitable'].sum()
        metrics['coverage'] = correct_trades / total_profitable if total_profitable > 0 else 0
        
        # Average return
        returns = outcomes.loc[meta_trades.index, 'return']
        metrics['avg_return'] = returns.mean()
        metrics['total_return'] = returns.sum()
        metrics['sharpe_ratio'] = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Compare with primary model alone
        primary_trades = predictions[abs(predictions['primary_prediction']) > self.primary_threshold]
        primary_returns = outcomes.loc[primary_trades.index, 'return']
        
        metrics['primary_avg_return'] = primary_returns.mean()
        metrics['improvement_pct'] = (
            (metrics['avg_return'] - metrics['primary_avg_return']) / 
            abs(metrics['primary_avg_return']) * 100
            if metrics['primary_avg_return'] != 0 else 0
        )
        
        # Risk metrics
        metrics['max_drawdown'] = (returns.cumsum().cummax() - returns.cumsum()).max()
        metrics['trades_taken'] = len(meta_trades)
        metrics['avg_position_size'] = meta_trades['position_size'].mean()
        
        return metrics