"""
Risk management system for liquidation-driven trading.

Comprehensive risk controls including:
- Position sizing and leverage control
- Drawdown protection
- Exposure limits
- Risk metrics calculation
- Dynamic risk adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import warnings
warnings.filterwarnings('ignore')

from ..common.config import settings
from ..common.logging import get_logger
from ..common.monitoring import (
    RISK_VIOLATIONS, POSITION_VALUE, DAILY_PNL,
    increment_counter, set_gauge
)

logger = get_logger(__name__)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    
    # Position limits
    max_position_size_usd: float = 100000.0  # Maximum position size per symbol
    max_total_exposure_usd: float = 500000.0  # Maximum total exposure
    max_leverage: float = 10.0  # Maximum leverage allowed
    max_positions: int = 10  # Maximum number of concurrent positions
    
    # Loss limits
    max_daily_loss_usd: float = 10000.0  # Maximum daily loss
    max_drawdown_pct: float = 0.20  # Maximum drawdown (20%)
    stop_loss_pct: float = 0.02  # Default stop loss (2%)
    trailing_stop_pct: float = 0.015  # Trailing stop loss (1.5%)
    
    # Risk per trade
    risk_per_trade_pct: float = 0.01  # Risk 1% of capital per trade
    kelly_fraction: float = 0.25  # Fractional Kelly for position sizing
    
    # Correlation limits
    max_correlated_positions: int = 5  # Max positions in correlated assets
    correlation_threshold: float = 0.7  # Correlation threshold
    
    # Time-based limits
    cooldown_period_seconds: int = 300  # Cooldown after stop loss
    max_trades_per_hour: int = 50  # Rate limiting
    max_trades_per_day: int = 200
    
    # Volatility adjustments
    volatility_scaling: bool = True  # Scale position size by volatility
    volatility_lookback: int = 20  # Lookback period for volatility
    target_volatility: float = 0.15  # Target annualized volatility
    
    # Risk metrics
    var_confidence: float = 0.95  # Value at Risk confidence level
    cvar_confidence: float = 0.95  # Conditional VaR confidence
    
    # Emergency controls
    circuit_breaker_loss_pct: float = 0.05  # Halt trading after 5% loss
    emergency_liquidation: bool = True  # Enable emergency liquidation


@dataclass
class Position:
    """Trading position information."""
    
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    position_id: str
    timestamp: datetime
    
    # Risk parameters
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    max_position_value: float = 0.0
    
    # Performance tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    
    # Metadata
    strategy: str = "liquidation_cascade"
    tags: List[str] = field(default_factory=list)
    
    @property
    def position_value(self) -> float:
        """Current position value in USD."""
        return abs(self.quantity * self.current_price)
    
    @property
    def pnl(self) -> float:
        """Total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def pnl_pct(self) -> float:
        """PnL percentage."""
        entry_value = abs(self.quantity * self.entry_price)
        return (self.pnl / entry_value * 100) if entry_value > 0 else 0.0
    
    def update_price(self, new_price: float) -> None:
        """Update position with new price."""
        self.current_price = new_price
        
        # Update unrealized PnL
        if self.side == 'long':
            self.unrealized_pnl = self.quantity * (new_price - self.entry_price)
        else:  # short
            self.unrealized_pnl = self.quantity * (self.entry_price - new_price)
        
        # Update max position value for trailing stop
        self.max_position_value = max(self.max_position_value, self.position_value)


class RiskManager:
    """
    Comprehensive risk management system for trading operations.
    
    Features:
    - Real-time position and exposure monitoring
    - Dynamic position sizing
    - Stop loss and drawdown protection
    - Correlation-based risk limits
    - Emergency circuit breakers
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialize risk manager.
        
        Args:
            config: Risk management configuration
        """
        self.config = config or RiskConfig()
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        
        # Performance tracking
        self.daily_pnl: Dict[str, float] = defaultdict(float)  # Date -> PnL
        self.cumulative_pnl: float = 0.0
        self.peak_equity: float = 100000.0  # Starting capital
        self.current_equity: float = self.peak_equity
        
        # Trade tracking
        self.trade_count_hourly: Dict[datetime, int] = defaultdict(int)
        self.trade_count_daily: Dict[str, int] = defaultdict(int)
        self.last_stop_loss_time: Dict[str, datetime] = {}
        
        # Risk metrics
        self.current_var: float = 0.0
        self.current_cvar: float = 0.0
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
        
        # Circuit breaker state
        self.trading_halted: bool = False
        self.halt_reason: Optional[str] = None
        self.halt_time: Optional[datetime] = None
        
        # Correlation matrix cache
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.correlation_update_time: Optional[datetime] = None
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("Risk manager initialized", config=self.config.__dict__)
    
    async def check_order_risk(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "limit"
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Check if order passes risk controls.
        
        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Order price
            order_type: Order type
            
        Returns:
            Tuple of (approved, rejection_reason, risk_metrics)
        """
        async with self._lock:
            try:
                # Check if trading is halted
                if self.trading_halted:
                    return False, f"Trading halted: {self.halt_reason}", {}
                
                # Check rate limits
                rate_check = self._check_rate_limits()
                if not rate_check[0]:
                    return False, rate_check[1], {}
                
                # Check cooldown period
                cooldown_check = self._check_cooldown(symbol)
                if not cooldown_check[0]:
                    return False, cooldown_check[1], {}
                
                # Calculate position value
                position_value = quantity * price
                
                # Check position size limits
                size_check = self._check_position_size(symbol, position_value)
                if not size_check[0]:
                    return False, size_check[1], {}
                
                # Check total exposure
                exposure_check = self._check_total_exposure(position_value)
                if not exposure_check[0]:
                    return False, exposure_check[1], {}
                
                # Check correlation limits
                correlation_check = await self._check_correlation_limits(symbol)
                if not correlation_check[0]:
                    return False, correlation_check[1], {}
                
                # Check leverage
                leverage_check = self._check_leverage(position_value)
                if not leverage_check[0]:
                    return False, leverage_check[1], {}
                
                # Check daily loss limit
                loss_check = self._check_daily_loss_limit()
                if not loss_check[0]:
                    return False, loss_check[1], {}
                
                # Calculate risk metrics
                risk_metrics = self._calculate_order_risk_metrics(
                    symbol, side, quantity, price
                )
                
                # All checks passed
                logger.info("Order risk check passed",
                           symbol=symbol,
                           side=side,
                           quantity=quantity,
                           price=price,
                           risk_metrics=risk_metrics)
                
                return True, None, risk_metrics
                
            except Exception as e:
                logger.error("Error in order risk check", exception=e)
                return False, f"Risk check error: {str(e)}", {}
    
    async def add_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Position:
        """Add new position to risk tracking."""
        async with self._lock:
            position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                position_id=position_id,
                timestamp=datetime.now(),
                stop_loss=stop_loss or self._calculate_stop_loss(side, entry_price),
                take_profit=take_profit
            )
            
            self.positions[position_id] = position
            
            # Update metrics
            self._update_exposure_metrics()
            
            logger.info("Position added",
                       position_id=position_id,
                       symbol=symbol,
                       side=side,
                       quantity=quantity,
                       entry_price=entry_price)
            
            return position
    
    async def update_position(
        self,
        position_id: str,
        current_price: float,
        quantity_change: float = 0.0
    ) -> Optional[Position]:
        """Update existing position."""
        async with self._lock:
            if position_id not in self.positions:
                logger.warning("Position not found", position_id=position_id)
                return None
            
            position = self.positions[position_id]
            position.update_price(current_price)
            
            # Update quantity if changed
            if quantity_change != 0:
                position.quantity += quantity_change
            
            # Check trailing stop
            if position.trailing_stop_distance:
                self._update_trailing_stop(position)
            
            # Check stop loss and take profit
            action = self._check_exit_conditions(position)
            if action:
                logger.warning("Exit condition triggered",
                              position_id=position_id,
                              action=action,
                              current_price=current_price)
            
            # Update metrics
            self._update_exposure_metrics()
            
            return position
    
    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        quantity: Optional[float] = None,
        fees: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """Close position and calculate final PnL."""
        async with self._lock:
            if position_id not in self.positions:
                logger.warning("Position not found for closing", position_id=position_id)
                return None
            
            position = self.positions[position_id]
            close_quantity = quantity or position.quantity
            
            # Calculate realized PnL
            if position.side == 'long':
                trade_pnl = close_quantity * (exit_price - position.entry_price)
            else:  # short
                trade_pnl = close_quantity * (position.entry_price - exit_price)
            
            trade_pnl -= fees
            position.realized_pnl += trade_pnl
            position.fees_paid += fees
            
            # Update daily PnL
            today = datetime.now().strftime("%Y-%m-%d")
            self.daily_pnl[today] += trade_pnl
            self.cumulative_pnl += trade_pnl
            
            # Update equity
            self.current_equity += trade_pnl
            
            # Full or partial close
            if close_quantity >= position.quantity:
                # Full close
                self.position_history.append(position)
                del self.positions[position_id]
                
                # Check if stop loss was hit
                if position.stop_loss and exit_price <= position.stop_loss:
                    self.last_stop_loss_time[position.symbol] = datetime.now()
            else:
                # Partial close
                position.quantity -= close_quantity
            
            # Update metrics
            self._update_exposure_metrics()
            self._update_drawdown()
            
            # Set Prometheus metrics
            set_gauge(DAILY_PNL, self.daily_pnl[today])
            
            close_info = {
                "position_id": position_id,
                "symbol": position.symbol,
                "side": position.side,
                "quantity_closed": close_quantity,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "pnl": trade_pnl,
                "fees": fees,
                "total_pnl": position.pnl,
                "pnl_pct": position.pnl_pct
            }
            
            logger.info("Position closed", **close_info)
            
            return close_info
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        confidence: float = 0.5,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size using Kelly criterion and risk management.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            confidence: Model confidence (win probability)
            volatility: Current volatility
            
        Returns:
            Optimal position size in base currency
        """
        # Calculate risk per trade
        account_risk = self.current_equity * self.config.risk_per_trade_pct
        
        # Calculate position risk
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        # Base position size
        base_size = account_risk / risk_per_unit if risk_per_unit > 0 else 0
        
        # Apply Kelly criterion
        if confidence > 0.5:
            # Simplified Kelly: f = p - q/b
            # where p = win probability, q = loss probability, b = win/loss ratio
            win_loss_ratio = 2.0  # Assume 2:1 reward/risk
            kelly_fraction = confidence - (1 - confidence) / win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, 1))  # Bound between 0 and 1
            
            # Apply fractional Kelly
            kelly_size = base_size * kelly_fraction * self.config.kelly_fraction
        else:
            kelly_size = base_size * 0.5  # Reduce size for low confidence
        
        # Apply volatility scaling
        if self.config.volatility_scaling and volatility:
            vol_scalar = self.config.target_volatility / volatility
            vol_scalar = max(0.5, min(vol_scalar, 2.0))  # Bound scaling
            kelly_size *= vol_scalar
        
        # Apply position limits
        max_size = self.config.max_position_size_usd / entry_price
        final_size = min(kelly_size, max_size)
        
        # Check total exposure
        current_exposure = self._calculate_total_exposure()
        remaining_capacity = self.config.max_total_exposure_usd - current_exposure
        max_allowed = remaining_capacity / entry_price
        
        final_size = min(final_size, max_allowed)
        
        logger.debug("Position size calculated",
                    symbol=symbol,
                    base_size=base_size,
                    kelly_size=kelly_size,
                    final_size=final_size,
                    confidence=confidence,
                    volatility=volatility)
        
        return max(0, final_size)
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        total_exposure = self._calculate_total_exposure()
        position_count = len(self.positions)
        
        # Calculate current P&L
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        today = datetime.now().strftime("%Y-%m-%d")
        
        metrics = {
            "current_equity": self.current_equity,
            "total_exposure": total_exposure,
            "exposure_pct": total_exposure / self.current_equity if self.current_equity > 0 else 0,
            "position_count": position_count,
            "max_positions": self.config.max_positions,
            "total_unrealized_pnl": total_unrealized_pnl,
            "daily_pnl": self.daily_pnl[today],
            "cumulative_pnl": self.cumulative_pnl,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "current_var": self.current_var,
            "current_cvar": self.current_cvar,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason
        }
        
        # Add position details
        metrics["positions"] = {
            pid: {
                "symbol": p.symbol,
                "side": p.side,
                "quantity": p.quantity,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "unrealized_pnl": p.unrealized_pnl,
                "pnl_pct": p.pnl_pct
            }
            for pid, p in self.positions.items()
        }
        
        return metrics
    
    def _check_rate_limits(self) -> Tuple[bool, Optional[str]]:
        """Check trading rate limits."""
        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        current_day = now.strftime("%Y-%m-%d")
        
        # Check hourly limit
        hourly_count = self.trade_count_hourly[current_hour]
        if hourly_count >= self.config.max_trades_per_hour:
            return False, f"Hourly trade limit reached ({hourly_count}/{self.config.max_trades_per_hour})"
        
        # Check daily limit
        daily_count = self.trade_count_daily[current_day]
        if daily_count >= self.config.max_trades_per_day:
            return False, f"Daily trade limit reached ({daily_count}/{self.config.max_trades_per_day})"
        
        return True, None
    
    def _check_cooldown(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """Check if symbol is in cooldown period."""
        if symbol in self.last_stop_loss_time:
            time_since_stop = (datetime.now() - self.last_stop_loss_time[symbol]).total_seconds()
            if time_since_stop < self.config.cooldown_period_seconds:
                remaining = self.config.cooldown_period_seconds - time_since_stop
                return False, f"Symbol in cooldown for {remaining:.0f} more seconds"
        
        return True, None
    
    def _check_position_size(self, symbol: str, position_value: float) -> Tuple[bool, Optional[str]]:
        """Check position size limits."""
        if position_value > self.config.max_position_size_usd:
            return False, f"Position size ${position_value:.2f} exceeds limit ${self.config.max_position_size_usd:.2f}"
        
        # Check existing position
        existing_exposure = sum(
            p.position_value for p in self.positions.values() 
            if p.symbol == symbol
        )
        
        if existing_exposure + position_value > self.config.max_position_size_usd:
            return False, f"Total position in {symbol} would exceed limit"
        
        return True, None
    
    def _check_total_exposure(self, additional_value: float) -> Tuple[bool, Optional[str]]:
        """Check total exposure limits."""
        current_exposure = self._calculate_total_exposure()
        new_exposure = current_exposure + additional_value
        
        if new_exposure > self.config.max_total_exposure_usd:
            return False, f"Total exposure ${new_exposure:.2f} would exceed limit ${self.config.max_total_exposure_usd:.2f}"
        
        return True, None
    
    def _check_leverage(self, position_value: float) -> Tuple[bool, Optional[str]]:
        """Check leverage limits."""
        current_exposure = self._calculate_total_exposure()
        new_exposure = current_exposure + position_value
        leverage = new_exposure / self.current_equity if self.current_equity > 0 else 0
        
        if leverage > self.config.max_leverage:
            return False, f"Leverage {leverage:.1f}x would exceed limit {self.config.max_leverage}x"
        
        return True, None
    
    def _check_daily_loss_limit(self) -> Tuple[bool, Optional[str]]:
        """Check daily loss limits."""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_loss = -self.daily_pnl[today]  # Convert to positive for comparison
        
        if daily_loss >= self.config.max_daily_loss_usd:
            return False, f"Daily loss ${daily_loss:.2f} exceeds limit ${self.config.max_daily_loss_usd:.2f}"
        
        # Check circuit breaker
        daily_loss_pct = daily_loss / self.current_equity if self.current_equity > 0 else 0
        if daily_loss_pct >= self.config.circuit_breaker_loss_pct:
            self._trigger_circuit_breaker(f"Daily loss {daily_loss_pct:.1%} exceeded circuit breaker threshold")
            return False, "Circuit breaker triggered"
        
        return True, None
    
    async def _check_correlation_limits(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """Check correlation-based position limits."""
        # This is a simplified check - in production, you'd calculate actual correlations
        # For now, treat all crypto as correlated
        
        correlated_positions = len(self.positions)
        if correlated_positions >= self.config.max_correlated_positions:
            return False, f"Maximum correlated positions ({self.config.max_correlated_positions}) reached"
        
        return True, None
    
    def _calculate_stop_loss(self, side: str, entry_price: float) -> float:
        """Calculate default stop loss price."""
        if side == 'long':
            return entry_price * (1 - self.config.stop_loss_pct)
        else:  # short
            return entry_price * (1 + self.config.stop_loss_pct)
    
    def _update_trailing_stop(self, position: Position) -> None:
        """Update trailing stop loss."""
        if not position.trailing_stop_distance:
            return
        
        if position.side == 'long':
            new_stop = position.current_price - position.trailing_stop_distance
            if position.stop_loss is None or new_stop > position.stop_loss:
                position.stop_loss = new_stop
        else:  # short
            new_stop = position.current_price + position.trailing_stop_distance
            if position.stop_loss is None or new_stop < position.stop_loss:
                position.stop_loss = new_stop
    
    def _check_exit_conditions(self, position: Position) -> Optional[str]:
        """Check if position should be closed."""
        if position.stop_loss:
            if position.side == 'long' and position.current_price <= position.stop_loss:
                return "stop_loss"
            elif position.side == 'short' and position.current_price >= position.stop_loss:
                return "stop_loss"
        
        if position.take_profit:
            if position.side == 'long' and position.current_price >= position.take_profit:
                return "take_profit"
            elif position.side == 'short' and position.current_price <= position.take_profit:
                return "take_profit"
        
        return None
    
    def _calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure."""
        return sum(p.position_value for p in self.positions.values())
    
    def _update_exposure_metrics(self) -> None:
        """Update exposure and risk metrics."""
        total_exposure = self._calculate_total_exposure()
        
        # Update Prometheus metrics
        set_gauge(POSITION_VALUE, total_exposure)
        
        # Calculate VaR and CVaR (simplified)
        if self.position_history:
            returns = [p.pnl_pct for p in self.position_history[-100:]]  # Last 100 trades
            if returns:
                sorted_returns = sorted(returns)
                var_index = int(len(sorted_returns) * (1 - self.config.var_confidence))
                self.current_var = sorted_returns[var_index] if var_index < len(sorted_returns) else 0
                self.current_cvar = np.mean(sorted_returns[:var_index]) if var_index > 0 else 0
    
    def _update_drawdown(self) -> None:
        """Update drawdown metrics."""
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        self.current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Check max drawdown limit
        if self.current_drawdown > self.config.max_drawdown_pct:
            self._trigger_circuit_breaker(f"Maximum drawdown {self.current_drawdown:.1%} exceeded")
    
    def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger emergency trading halt."""
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_time = datetime.now()
        
        logger.critical("Circuit breaker triggered",
                       reason=reason,
                       current_equity=self.current_equity,
                       current_drawdown=self.current_drawdown)
        
        increment_counter(RISK_VIOLATIONS, violation_type="circuit_breaker")
        
        # TODO: Implement emergency position liquidation if configured
    
    def _calculate_order_risk_metrics(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Dict[str, Any]:
        """Calculate risk metrics for new order."""
        position_value = quantity * price
        
        # Calculate position risk
        stop_loss_price = self._calculate_stop_loss(side, price)
        position_risk = abs(price - stop_loss_price) * quantity
        risk_pct = position_risk / self.current_equity if self.current_equity > 0 else 0
        
        # Calculate new exposure
        current_exposure = self._calculate_total_exposure()
        new_exposure = current_exposure + position_value
        new_leverage = new_exposure / self.current_equity if self.current_equity > 0 else 0
        
        return {
            "position_value": position_value,
            "position_risk": position_risk,
            "risk_pct": risk_pct,
            "current_exposure": current_exposure,
            "new_exposure": new_exposure,
            "new_leverage": new_leverage,
            "stop_loss_price": stop_loss_price,
            "max_loss": position_risk
        }
    
    def update_trade_counts(self) -> None:
        """Update trade count for rate limiting."""
        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        current_day = now.strftime("%Y-%m-%d")
        
        self.trade_count_hourly[current_hour] += 1
        self.trade_count_daily[current_day] += 1
        
        # Clean old entries
        cutoff_time = now - timedelta(hours=2)
        old_hours = [h for h in self.trade_count_hourly if h < cutoff_time]
        for h in old_hours:
            del self.trade_count_hourly[h]