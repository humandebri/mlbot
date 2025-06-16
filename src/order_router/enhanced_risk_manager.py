"""
Enhanced Risk Manager with dynamic position sizing based on account equity.
"""

from typing import Optional, Dict, Any, Tuple
from .risk_manager import RiskManager, RiskConfig
from .dynamic_risk_config import DynamicRiskConfig
from ..common.logging import get_logger
from ..common.account_monitor import AccountMonitor

logger = get_logger(__name__)


class EnhancedRiskManager(RiskManager):
    """
    Enhanced risk manager that dynamically adjusts position limits based on:
    - Current account equity
    - Market volatility
    - Recent trading performance
    - Risk parameters as percentages rather than fixed values
    """
    
    def __init__(
        self, 
        dynamic_config: Optional[DynamicRiskConfig] = None,
        account_monitor: Optional[AccountMonitor] = None
    ):
        """
        Initialize enhanced risk manager.
        
        Args:
            dynamic_config: Dynamic risk configuration
            account_monitor: Account monitor for real-time balance
        """
        # Initialize with base config
        super().__init__()
        
        # Set dynamic config
        self.dynamic_config = dynamic_config or DynamicRiskConfig()
        self.account_monitor = account_monitor
        
        # Cache for dynamic limits
        self._cached_limits: Optional[Dict] = None
        self._limits_cache_time: Optional[float] = None
        self._cache_duration = 60.0  # Refresh limits every 60 seconds
        
        # Performance tracking for adaptive sizing
        self.recent_trades = []
        self.max_recent_trades = 100
        
        logger.info("Enhanced risk manager initialized with dynamic sizing")
    
    def get_current_limits(self) -> Dict[str, Any]:
        """
        Get current dynamic limits based on account equity.
        
        Returns:
            Dictionary with current limits
        """
        import time
        current_time = time.time()
        
        # Check cache
        if (self._cached_limits and self._limits_cache_time and 
            current_time - self._limits_cache_time < self._cache_duration):
            return self._cached_limits
        
        # Get current equity
        if self.account_monitor and self.account_monitor.current_balance:
            current_equity = self.account_monitor.current_balance.total_equity
        else:
            current_equity = self.current_equity
        
        # Calculate base limits
        base_limits = self.dynamic_config.calculate_dynamic_limits(current_equity)
        
        # Get market volatility (simplified - in production, calculate from recent data)
        market_volatility = self._estimate_market_volatility()
        
        # Get recent performance
        performance = self._calculate_recent_performance()
        
        # Adjust for conditions
        adjusted_limits = self.dynamic_config.adjust_for_market_conditions(
            base_limits, 
            market_volatility,
            performance
        )
        
        # Update base config with dynamic values
        self.config.max_position_size_usd = adjusted_limits['max_position_size_usd']
        self.config.max_total_exposure_usd = adjusted_limits['max_total_exposure_usd']
        self.config.max_positions = adjusted_limits['max_positions']
        self.config.max_daily_loss_usd = adjusted_limits['max_daily_loss_usd']
        self.config.max_trades_per_hour = adjusted_limits['max_trades_per_hour']
        self.config.max_trades_per_day = adjusted_limits['max_trades_per_day']
        
        # Cache the limits
        self._cached_limits = adjusted_limits
        self._limits_cache_time = current_time
        
        logger.info(
            "Dynamic limits updated",
            equity=current_equity,
            max_position_size=adjusted_limits['max_position_size_usd'],
            max_positions=adjusted_limits['max_positions'],
            max_exposure=adjusted_limits['max_total_exposure_usd']
        )
        
        return adjusted_limits
    
    async def check_order_risk(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "limit"
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Enhanced order risk check with dynamic limits.
        """
        # Update dynamic limits
        self.get_current_limits()
        
        # Call parent implementation
        return await super().check_order_risk(symbol, side, quantity, price, order_type)
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        confidence: float = 0.5,
        volatility: Optional[float] = None
    ) -> float:
        """
        Enhanced position size calculation with dynamic equity-based sizing.
        """
        # Update dynamic limits
        limits = self.get_current_limits()
        
        # Get current equity
        if self.account_monitor and self.account_monitor.current_balance:
            current_equity = self.account_monitor.current_balance.total_equity
        else:
            current_equity = self.current_equity
        
        # Calculate risk amount (1% of equity)
        risk_amount = limits['risk_per_trade_usd']
        
        # Calculate position risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            logger.warning("Invalid stop loss distance", 
                          entry=entry_price, 
                          stop_loss=stop_loss_price)
            return 0.0
        
        # Base position size (before Kelly)
        base_size = risk_amount / risk_per_unit
        
        # Apply Kelly criterion with dynamic calculation
        if confidence > 0.5:
            # Get position size as percentage of equity
            kelly_pct = self.dynamic_config.get_position_size_for_kelly(
                current_equity,
                confidence,
                win_loss_ratio=2.0  # Could be dynamic based on historical data
            )
            
            # Convert percentage to position size
            kelly_position_value = current_equity * kelly_pct
            kelly_size = kelly_position_value / entry_price
            
            # Use the smaller of risk-based or Kelly-based size
            position_size = min(base_size, kelly_size)
        else:
            # Low confidence - use reduced size
            position_size = base_size * 0.5
        
        # Apply volatility scaling
        if self.dynamic_config.volatility_scaling and volatility:
            vol_scalar = self.dynamic_config.target_volatility / volatility
            vol_scalar = max(
                self.dynamic_config.min_volatility_scalar,
                min(vol_scalar, self.dynamic_config.max_volatility_scalar)
            )
            position_size *= vol_scalar
        
        # Apply position limits (as % of equity)
        max_position_value = limits['max_position_size_usd']
        max_size = max_position_value / entry_price
        position_size = min(position_size, max_size)
        
        # Check total exposure
        current_exposure = self._calculate_total_exposure()
        remaining_capacity = limits['max_total_exposure_usd'] - current_exposure
        max_allowed = remaining_capacity / entry_price
        position_size = min(position_size, max_allowed)
        
        # Ensure minimum size
        min_size = limits['min_position_size_usd'] / entry_price
        if position_size < min_size:
            logger.debug(
                "Position size below minimum",
                calculated=position_size,
                minimum=min_size
            )
            return 0.0
        
        # Round to increment
        increment = self.dynamic_config.position_size_increment / entry_price
        position_size = round(position_size / increment) * increment
        
        logger.info(
            "Dynamic position size calculated",
            symbol=symbol,
            equity=current_equity,
            risk_amount=risk_amount,
            position_size=position_size,
            position_value=position_size * entry_price,
            confidence=confidence,
            volatility=volatility
        )
        
        return position_size
    
    def _estimate_market_volatility(self) -> float:
        """
        Estimate current market volatility.
        In production, this would analyze recent price data.
        """
        # Simplified implementation - return target volatility
        # In production: calculate from recent OHLC data
        return self.dynamic_config.target_volatility
    
    def _calculate_recent_performance(self) -> Dict[str, float]:
        """
        Calculate recent trading performance metrics.
        """
        if not self.position_history:
            return {
                'win_rate': 0.5,
                'current_drawdown_pct': 0.0,
                'avg_win_loss_ratio': 2.0
            }
        
        # Get recent closed positions
        recent = self.position_history[-self.max_recent_trades:]
        
        if not recent:
            return {
                'win_rate': 0.5,
                'current_drawdown_pct': 0.0,
                'avg_win_loss_ratio': 2.0
            }
        
        # Calculate win rate
        wins = sum(1 for p in recent if p.pnl > 0)
        win_rate = wins / len(recent) if recent else 0.5
        
        # Calculate average win/loss ratio
        winning_trades = [p.pnl for p in recent if p.pnl > 0]
        losing_trades = [abs(p.pnl) for p in recent if p.pnl < 0]
        
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 1
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 2.0
        
        return {
            'win_rate': win_rate,
            'current_drawdown_pct': self.current_drawdown,
            'avg_win_loss_ratio': win_loss_ratio,
            'total_trades': len(recent),
            'winning_trades': wins,
            'losing_trades': len(recent) - wins
        }
    
    def get_position_sizing_info(self) -> Dict[str, Any]:
        """
        Get detailed information about current position sizing parameters.
        """
        limits = self.get_current_limits()
        
        if self.account_monitor and self.account_monitor.current_balance:
            current_equity = self.account_monitor.current_balance.total_equity
        else:
            current_equity = self.current_equity
        
        performance = self._calculate_recent_performance()
        
        return {
            'current_equity': current_equity,
            'dynamic_limits': limits,
            'risk_parameters': {
                'risk_per_trade_pct': self.dynamic_config.risk_per_trade_pct * 100,
                'kelly_fraction': self.dynamic_config.kelly_fraction * 100,
                'max_leverage': self.dynamic_config.max_leverage,
                'stop_loss_pct': self.dynamic_config.default_stop_loss_pct * 100,
            },
            'position_limits': {
                'max_position_pct': self.dynamic_config.max_position_pct_per_symbol * 100,
                'max_exposure_pct': self.dynamic_config.max_total_exposure_pct * 100,
                'max_positions': limits['max_positions'],
                'positions_per_100k': self.dynamic_config.positions_per_100k,
            },
            'recent_performance': performance,
            'current_exposure': {
                'total_exposure_usd': self._calculate_total_exposure(),
                'position_count': len(self.positions),
                'exposure_pct': (self._calculate_total_exposure() / current_equity * 100) 
                               if current_equity > 0 else 0
            }
        }