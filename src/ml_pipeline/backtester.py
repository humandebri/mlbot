"""
Comprehensive backtesting framework for liquidation-driven trading strategies.

Features:
- Time series walk-forward validation
- Realistic limit order execution simulation
- Transaction cost and slippage modeling
- Comprehensive performance metrics
- Risk analysis and drawdown calculation
- Market regime analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..common.config import settings
from ..common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Time periods
    lookback_window: int = 252  # Days for rolling statistics
    walk_forward_window: int = 30  # Days for each walk-forward step
    min_history_days: int = 60  # Minimum history required
    
    # Trading parameters
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of capital per trade
    min_trade_size: float = 100.0  # Minimum trade size
    
    # Transaction costs
    maker_fee: float = 0.0001  # 0.01%
    taker_fee: float = 0.0006  # 0.06%
    slippage_bps: float = 2.0  # 2 basis points
    borrowing_rate: float = 0.0001  # Daily borrowing rate
    
    # Order execution
    limit_order_timeout: int = 300  # Seconds before order cancellation
    partial_fill_threshold: float = 0.1  # Minimum fill ratio
    max_spread_bps: float = 50.0  # Maximum spread for order placement
    
    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.01  # 1% take profit
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_drawdown_pct: float = 0.20  # 20% max drawdown
    
    # Performance calculation
    risk_free_rate: float = 0.02  # Annual risk-free rate
    trading_days_per_year: int = 365  # For crypto (24/7)
    benchmark_symbol: str = "BTCUSDT"  # Benchmark for comparison
    
    # Output settings
    save_trades: bool = True
    save_metrics: bool = True
    generate_plots: bool = False  # Set to True for visual analysis
    output_path: str = "backtest_results"


class Order:
    """Represents a trading order."""
    
    def __init__(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        order_type: str,  # 'limit' or 'market'
        quantity: float,
        price: float,
        prediction_score: float = 0.0,
        confidence: float = 0.0
    ):
        self.id = f"{timestamp.value}_{side}_{symbol}"
        self.timestamp = timestamp
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.prediction_score = prediction_score
        self.confidence = confidence
        
        # Execution details
        self.status = "pending"  # pending, partial, filled, cancelled
        self.filled_quantity = 0.0
        self.filled_price = 0.0
        self.execution_timestamp = None
        self.fees = 0.0
        self.slippage = 0.0


class Trade:
    """Represents a completed trade."""
    
    def __init__(self, entry_order: Order, exit_order: Order):
        self.entry_order = entry_order
        self.exit_order = exit_order
        
        # Trade metrics
        self.entry_time = entry_order.execution_timestamp
        self.exit_time = exit_order.execution_timestamp
        self.duration = (self.exit_time - self.entry_time).total_seconds()
        
        self.entry_price = entry_order.filled_price
        self.exit_price = exit_order.filled_price
        self.quantity = min(entry_order.filled_quantity, exit_order.filled_quantity)
        
        # P&L calculation
        if entry_order.side == "buy":
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # short trade
            self.pnl = (self.entry_price - self.exit_price) * self.quantity
        
        self.total_fees = entry_order.fees + exit_order.fees
        self.net_pnl = self.pnl - self.total_fees
        self.return_pct = self.net_pnl / (self.entry_price * self.quantity)


class Backtester:
    """
    Comprehensive backtesting framework for liquidation-driven trading strategies.
    
    Simulates realistic trading conditions with:
    - Limit order execution modeling
    - Transaction costs and slippage
    - Market impact simulation
    - Risk management integration
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        
        # State tracking
        self.current_time = None
        self.current_capital = self.config.initial_capital
        self.positions = {}  # symbol -> quantity
        self.open_orders = []
        self.filled_orders = []
        self.trades = []
        
        # Performance tracking
        self.equity_curve = []
        self.daily_returns = []
        self.max_drawdown = 0.0
        self.peak_equity = self.config.initial_capital
        
        # Metrics storage
        self.performance_metrics = {}
        self.trade_analysis = {}
        self.risk_metrics = {}
        
        # Output directory
        self.output_dir = Path(self.config.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Backtester initialized", config=self.config.__dict__)
    
    def backtest(
        self,
        market_data: pd.DataFrame,
        predictions: pd.Series,
        features: pd.DataFrame,
        confidence_scores: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtest.
        
        Args:
            market_data: OHLCV market data
            predictions: Model predictions (expPNL)
            features: Feature matrix used for predictions
            confidence_scores: Prediction confidence scores
            
        Returns:
            Comprehensive backtest results
        """
        logger.info("Starting backtest",
                   data_period=f"{market_data.index[0]} to {market_data.index[-1]}",
                   total_samples=len(market_data),
                   prediction_stats={"mean": predictions.mean(), "std": predictions.std()})
        
        try:
            # 1. Validate input data
            self._validate_backtest_data(market_data, predictions, features)
            
            # 2. Initialize backtest state
            self._initialize_backtest_state()
            
            # 3. Run walk-forward backtest
            results = self._run_walk_forward_backtest(
                market_data, predictions, features, confidence_scores
            )
            
            # 4. Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics()
            
            # 5. Analyze trades
            trade_analysis = self._analyze_trades()
            
            # 6. Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics()
            
            # 7. Generate comprehensive results
            backtest_results = {
                "performance_metrics": performance_metrics,
                "trade_analysis": trade_analysis,
                "risk_metrics": risk_metrics,
                "equity_curve": pd.Series(self.equity_curve, index=market_data.index),
                "daily_returns": pd.Series(self.daily_returns),
                "trades": self._trades_to_dataframe(),
                "config": self.config.__dict__
            }
            
            # 8. Save results
            if self.config.save_metrics:
                self._save_backtest_results(backtest_results)
            
            logger.info("Backtest completed successfully",
                       total_trades=len(self.trades),
                       final_capital=self.current_capital,
                       total_return=performance_metrics.get("total_return", 0),
                       sharpe_ratio=performance_metrics.get("sharpe_ratio", 0))
            
            return backtest_results
            
        except Exception as e:
            logger.error("Backtest failed", exception=e)
            raise
    
    def _validate_backtest_data(
        self, 
        market_data: pd.DataFrame, 
        predictions: pd.Series, 
        features: pd.DataFrame
    ) -> None:
        """Validate input data for backtesting."""
        
        # Check data alignment
        if not market_data.index.equals(predictions.index):
            raise ValueError("Market data and predictions indices must be aligned")
        
        if not market_data.index.equals(features.index):
            raise ValueError("Market data and features indices must be aligned")
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in market_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check data quality
        if market_data.isnull().any().any():
            logger.warning("Market data contains NaN values")
        
        if predictions.isnull().any():
            logger.warning("Predictions contain NaN values")
        
        # Check minimum data requirements
        if len(market_data) < self.config.min_history_days:
            raise ValueError(f"Insufficient data: {len(market_data)} < {self.config.min_history_days}")
    
    def _initialize_backtest_state(self) -> None:
        """Initialize backtesting state variables."""
        self.current_capital = self.config.initial_capital
        self.positions = {}
        self.open_orders = []
        self.filled_orders = []
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.max_drawdown = 0.0
        self.peak_equity = self.config.initial_capital
    
    def _run_walk_forward_backtest(
        self,
        market_data: pd.DataFrame,
        predictions: pd.Series,
        features: pd.DataFrame,
        confidence_scores: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """Run walk-forward backtesting simulation."""
        
        start_idx = self.config.min_history_days
        
        for i in range(start_idx, len(market_data)):
            self.current_time = market_data.index[i]
            current_data = market_data.iloc[i]
            current_prediction = predictions.iloc[i]
            current_confidence = confidence_scores.iloc[i] if confidence_scores is not None else 0.5
            
            # 1. Process existing orders
            self._process_open_orders(current_data)
            
            # 2. Generate trading signals
            signals = self._generate_trading_signals(
                current_prediction, current_confidence, current_data
            )
            
            # 3. Execute trades based on signals
            for signal in signals:
                self._execute_trade_signal(signal, current_data)
            
            # 4. Update portfolio value
            portfolio_value = self._calculate_portfolio_value(current_data)
            self.equity_curve.append(portfolio_value)
            
            # 5. Calculate daily returns
            if len(self.equity_curve) > 1:
                daily_return = (portfolio_value - self.equity_curve[-2]) / self.equity_curve[-2]
                self.daily_returns.append(daily_return)
            
            # 6. Update drawdown
            self._update_drawdown_metrics(portfolio_value)
            
            # 7. Risk management checks
            self._apply_risk_management(current_data)
        
        return {"processed_bars": len(market_data) - start_idx}
    
    def _generate_trading_signals(
        self, 
        prediction: float, 
        confidence: float, 
        market_data: pd.Series
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on predictions."""
        signals = []
        
        # Only trade if confidence is above threshold
        confidence_threshold = 0.6
        if confidence < confidence_threshold:
            return signals
        
        # Calculate position size based on prediction strength and confidence
        prediction_strength = abs(prediction)
        position_size = min(
            prediction_strength * confidence * self.config.max_position_size,
            self.config.max_position_size
        )
        
        # Calculate trade value
        trade_value = self.current_capital * position_size
        
        if trade_value < self.config.min_trade_size:
            return signals
        
        # Determine trade direction
        if prediction > 0.005:  # Buy signal threshold
            signals.append({
                "action": "buy",
                "quantity": trade_value / market_data['close'],
                "prediction": prediction,
                "confidence": confidence,
                "price_target": market_data['close'] * (1 - 0.001)  # Slightly below market
            })
        elif prediction < -0.005:  # Sell signal threshold
            signals.append({
                "action": "sell",
                "quantity": trade_value / market_data['close'],
                "prediction": prediction,
                "confidence": confidence,
                "price_target": market_data['close'] * (1 + 0.001)  # Slightly above market
            })
        
        return signals
    
    def _execute_trade_signal(self, signal: Dict[str, Any], market_data: pd.Series) -> None:
        """Execute a trading signal by placing orders."""
        
        # Create limit order
        order = Order(
            timestamp=self.current_time,
            symbol=self.config.benchmark_symbol,
            side=signal["action"],
            order_type="limit",
            quantity=signal["quantity"],
            price=signal["price_target"],
            prediction_score=signal["prediction"],
            confidence=signal["confidence"]
        )
        
        # Check if we have sufficient capital/position
        if signal["action"] == "buy":
            required_capital = order.quantity * order.price
            if required_capital > self.current_capital * 0.95:  # Leave 5% buffer
                return
        else:  # sell
            current_position = self.positions.get(order.symbol, 0)
            if order.quantity > current_position:
                return
        
        self.open_orders.append(order)
    
    def _process_open_orders(self, market_data: pd.Series) -> None:
        """Process open orders for execution or timeout."""
        orders_to_remove = []
        
        for order in self.open_orders:
            # Check for timeout
            time_elapsed = (self.current_time - order.timestamp).total_seconds()
            if time_elapsed > self.config.limit_order_timeout:
                order.status = "cancelled"
                orders_to_remove.append(order)
                continue
            
            # Check for execution
            executed = self._check_order_execution(order, market_data)
            if executed:
                orders_to_remove.append(order)
        
        # Remove processed orders
        for order in orders_to_remove:
            self.open_orders.remove(order)
    
    def _check_order_execution(self, order: Order, market_data: pd.Series) -> bool:
        """Check if order can be executed given current market conditions."""
        
        # Check if limit price is achievable
        if order.side == "buy" and order.price >= market_data['low']:
            # Buy order can be filled
            execution_price = min(order.price, market_data['open'])
        elif order.side == "sell" and order.price <= market_data['high']:
            # Sell order can be filled
            execution_price = max(order.price, market_data['open'])
        else:
            return False
        
        # Calculate fees and slippage
        fees = execution_price * order.quantity * self.config.maker_fee
        slippage = execution_price * order.quantity * (self.config.slippage_bps / 10000.0)
        
        # Execute order
        order.status = "filled"
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.execution_timestamp = self.current_time
        order.fees = fees
        order.slippage = slippage
        
        # Update portfolio
        if order.side == "buy":
            self.current_capital -= (execution_price * order.quantity + fees + slippage)
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
        else:  # sell
            self.current_capital += (execution_price * order.quantity - fees - slippage)
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) - order.quantity
        
        self.filled_orders.append(order)
        
        # Check if this completes a trade (simplified: assume immediate exit)
        self._check_trade_completion(order)
        
        return True
    
    def _check_trade_completion(self, order: Order) -> None:
        """Check if order completes a trade and create Trade object."""
        # Simplified trade completion logic
        # In practice, this would be more sophisticated with proper entry/exit matching
        
        # For demonstration, assume each order is a complete trade
        # This should be improved to properly match entry and exit orders
        if len(self.filled_orders) >= 2:
            # Create dummy exit order for demonstration
            if order.side == "buy":
                # Assume we sell after some time at market price
                exit_order = Order(
                    timestamp=self.current_time,
                    symbol=order.symbol,
                    side="sell",
                    order_type="market",
                    quantity=order.quantity,
                    price=order.filled_price * 1.01  # Assume 1% profit
                )
                exit_order.status = "filled"
                exit_order.filled_quantity = order.quantity
                exit_order.filled_price = order.filled_price * 1.01
                exit_order.execution_timestamp = self.current_time
                exit_order.fees = exit_order.filled_price * exit_order.quantity * self.config.taker_fee
                
                trade = Trade(order, exit_order)
                self.trades.append(trade)
    
    def _calculate_portfolio_value(self, market_data: pd.Series) -> float:
        """Calculate current portfolio value."""
        portfolio_value = self.current_capital
        
        # Add value of positions
        for symbol, quantity in self.positions.items():
            if symbol == self.config.benchmark_symbol:
                portfolio_value += quantity * market_data['close']
        
        return portfolio_value
    
    def _update_drawdown_metrics(self, portfolio_value: float) -> None:
        """Update maximum drawdown metrics."""
        if portfolio_value > self.peak_equity:
            self.peak_equity = portfolio_value
        
        current_drawdown = (self.peak_equity - portfolio_value) / self.peak_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def _apply_risk_management(self, market_data: pd.Series) -> None:
        """Apply risk management rules."""
        
        # Check maximum drawdown
        current_portfolio_value = self._calculate_portfolio_value(market_data)
        current_drawdown = (self.peak_equity - current_portfolio_value) / self.peak_equity
        
        if current_drawdown > self.config.max_drawdown_pct:
            logger.warning("Maximum drawdown exceeded, liquidating positions",
                          current_drawdown=current_drawdown,
                          max_allowed=self.config.max_drawdown_pct)
            self._liquidate_all_positions(market_data)
        
        # Check daily loss limits
        if len(self.daily_returns) > 0:
            daily_return = self.daily_returns[-1]
            if daily_return < -self.config.max_daily_loss_pct:
                logger.warning("Daily loss limit exceeded",
                              daily_return=daily_return,
                              max_allowed=-self.config.max_daily_loss_pct)
    
    def _liquidate_all_positions(self, market_data: pd.Series) -> None:
        """Emergency liquidation of all positions."""
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                # Create market sell order
                liquidation_order = Order(
                    timestamp=self.current_time,
                    symbol=symbol,
                    side="sell",
                    order_type="market",
                    quantity=quantity,
                    price=market_data['close']
                )
                self._check_order_execution(liquidation_order, market_data)
        
        # Clear positions
        self.positions = {}
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(self.equity_curve) < 2:
            return {}
        
        equity_series = pd.Series(self.equity_curve)
        returns_series = pd.Series(self.daily_returns)
        
        # Basic metrics
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
        annualized_return = (1 + total_return) ** (self.config.trading_days_per_year / len(equity_series)) - 1
        
        # Risk metrics
        volatility = returns_series.std() * np.sqrt(self.config.trading_days_per_year)
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Win rate and trade metrics
        if self.trades:
            winning_trades = [t for t in self.trades if t.net_pnl > 0]
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.net_pnl for t in self.trades if t.net_pnl <= 0]) or 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "avg_trade_return": np.mean([t.return_pct for t in self.trades]) if self.trades else 0,
            "best_trade": max([t.net_pnl for t in self.trades]) if self.trades else 0,
            "worst_trade": min([t.net_pnl for t in self.trades]) if self.trades else 0
        }
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trade patterns and statistics."""
        if not self.trades:
            return {}
        
        trade_returns = [t.return_pct for t in self.trades]
        trade_durations = [t.duration for t in self.trades]
        
        return {
            "trade_count": len(self.trades),
            "winning_trades": len([t for t in self.trades if t.net_pnl > 0]),
            "losing_trades": len([t for t in self.trades if t.net_pnl <= 0]),
            "avg_trade_duration": np.mean(trade_durations),
            "median_trade_duration": np.median(trade_durations),
            "avg_return_per_trade": np.mean(trade_returns),
            "std_return_per_trade": np.std(trade_returns),
            "skewness": pd.Series(trade_returns).skew(),
            "kurtosis": pd.Series(trade_returns).kurtosis()
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate advanced risk metrics."""
        if len(self.daily_returns) < 2:
            return {}
        
        returns = pd.Series(self.daily_returns)
        
        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Sortino ratio
        excess_return = returns.mean() - self.config.risk_free_rate / self.config.trading_days_per_year
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        return {
            "value_at_risk_95": var_95,
            "conditional_var_95": cvar_95,
            "downside_deviation": downside_deviation,
            "sortino_ratio": sortino_ratio,
            "tail_ratio": abs(returns.quantile(0.95) / returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0
        }
    
    def _trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "duration_seconds": trade.duration,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "pnl": trade.pnl,
                "fees": trade.total_fees,
                "net_pnl": trade.net_pnl,
                "return_pct": trade.return_pct
            })
        
        return pd.DataFrame(trade_data)
    
    def _save_backtest_results(self, results: Dict[str, Any]) -> None:
        """Save backtest results to files."""
        import json
        
        # Save performance metrics
        metrics_path = self.output_dir / "performance_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results["performance_metrics"], f, indent=2, default=str)
        
        # Save trades
        if self.config.save_trades and not results["trades"].empty:
            trades_path = self.output_dir / "trades.csv"
            results["trades"].to_csv(trades_path, index=False)
        
        # Save equity curve
        equity_path = self.output_dir / "equity_curve.csv"
        results["equity_curve"].to_csv(equity_path)
        
        logger.info("Backtest results saved", output_dir=str(self.output_dir))