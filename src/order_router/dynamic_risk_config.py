"""
Dynamic risk configuration that adapts to account balance and market conditions.
"""

from dataclasses import dataclass
from typing import Optional
import math
from ..common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DynamicRiskConfig:
    """
    Dynamic risk management configuration that scales with account equity.
    
    Instead of fixed limits, all parameters are calculated as percentages
    or ratios of current account equity.
    """
    
    # Core risk parameters (as percentages)
    risk_per_trade_pct: float = 0.01  # Risk 1% of capital per trade
    kelly_fraction: float = 0.25  # Fractional Kelly (25%)
    
    # Position sizing parameters (as % of equity)
    max_position_pct_per_symbol: float = 0.20  # Max 20% of equity per symbol
    max_total_exposure_pct: float = 1.0  # Max 100% of equity (with 3x leverage = 300% notional)
    
    # Dynamic position limits based on equity
    min_position_size_usd: float = 10.0  # Minimum position size
    position_size_increment: float = 1.0  # Position size must be multiple of this
    
    # Leverage and exposure
    max_leverage: float = 3.0  # Maximum 3x leverage
    target_leverage: float = 1.5  # Target 1.5x leverage
    
    # Risk limits (as % of equity)
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss
    max_drawdown_pct: float = 0.20  # Max 20% drawdown
    circuit_breaker_loss_pct: float = 0.03  # Halt at 3% intraday loss
    
    # Stop loss parameters
    default_stop_loss_pct: float = 0.02  # 2% stop loss
    trailing_stop_pct: float = 0.015  # 1.5% trailing stop
    
    # Volatility adjustments
    volatility_scaling: bool = True
    target_volatility: float = 0.15  # 15% annualized
    min_volatility_scalar: float = 0.5  # Min position size multiplier
    max_volatility_scalar: float = 2.0  # Max position size multiplier
    
    # Correlation and diversification
    max_correlated_positions: int = 5
    correlation_threshold: float = 0.7
    concentration_limit_pct: float = 0.30  # Max 30% in correlated assets
    
    # Dynamic position count based on equity
    base_position_count: int = 3  # Base number of positions
    positions_per_100k: int = 2  # Additional positions per $100k
    max_positions_absolute: int = 20  # Hard cap on positions
    
    # Time-based limits
    cooldown_period_seconds: int = 300  # 5 min cooldown after stop loss
    max_trades_per_hour_per_10k: int = 5  # Scale with account size
    max_trades_per_day_per_10k: int = 20
    
    def calculate_dynamic_limits(self, current_equity: float) -> dict:
        """
        Calculate dynamic position and risk limits based on current equity.
        
        Args:
            current_equity: Current account equity in USD
            
        Returns:
            Dictionary with calculated limits
        """
        # Ensure minimum equity
        equity = max(current_equity, 100.0)
        
        # Calculate position limits
        max_position_size_usd = equity * self.max_position_pct_per_symbol
        max_total_exposure_usd = equity * self.max_total_exposure_pct
        
        # Calculate loss limits
        max_daily_loss_usd = equity * self.max_daily_loss_pct
        circuit_breaker_usd = equity * self.circuit_breaker_loss_pct
        
        # Calculate dynamic position count
        # Base positions + additional based on equity
        additional_positions = int(equity / 100000) * self.positions_per_100k
        max_positions = min(
            self.base_position_count + additional_positions,
            self.max_positions_absolute
        )
        
        # Calculate rate limits (scale with account size)
        equity_10k_units = max(1, equity / 10000)
        max_trades_per_hour = int(self.max_trades_per_hour_per_10k * equity_10k_units)
        max_trades_per_day = int(self.max_trades_per_day_per_10k * equity_10k_units)
        
        # Calculate minimum meaningful position size
        # At least 0.5% of equity or $10, whichever is larger
        min_position_size = max(
            equity * 0.005,
            self.min_position_size_usd
        )
        
        return {
            # Position sizing
            'max_position_size_usd': max_position_size_usd,
            'max_total_exposure_usd': max_total_exposure_usd,
            'min_position_size_usd': min_position_size,
            'position_size_increment': self.position_size_increment,
            
            # Position count
            'max_positions': max_positions,
            'max_correlated_positions': min(self.max_correlated_positions, max_positions),
            
            # Loss limits
            'max_daily_loss_usd': max_daily_loss_usd,
            'circuit_breaker_usd': circuit_breaker_usd,
            'max_drawdown_usd': equity * self.max_drawdown_pct,
            
            # Rate limits
            'max_trades_per_hour': max_trades_per_hour,
            'max_trades_per_day': max_trades_per_day,
            
            # Risk per trade
            'risk_per_trade_usd': equity * self.risk_per_trade_pct,
            
            # Concentration limit
            'max_concentration_usd': equity * self.concentration_limit_pct,
            
            # Leverage info
            'max_leveraged_exposure': equity * self.max_leverage,
            'target_leveraged_exposure': equity * self.target_leverage,
        }
    
    def adjust_for_market_conditions(
        self, 
        base_limits: dict,
        market_volatility: float,
        recent_performance: Optional[dict] = None
    ) -> dict:
        """
        Adjust limits based on market conditions and recent performance.
        
        Args:
            base_limits: Base limits from calculate_dynamic_limits
            market_volatility: Current market volatility (annualized)
            recent_performance: Recent trading performance metrics
            
        Returns:
            Adjusted limits dictionary
        """
        adjusted = base_limits.copy()
        
        # Adjust for volatility
        if self.volatility_scaling and market_volatility > 0:
            vol_scalar = self.target_volatility / market_volatility
            vol_scalar = max(self.min_volatility_scalar, 
                           min(vol_scalar, self.max_volatility_scalar))
            
            # Scale position sizes
            adjusted['max_position_size_usd'] *= vol_scalar
            adjusted['risk_per_trade_usd'] *= vol_scalar
            
            # Tighten rate limits in high volatility
            if market_volatility > self.target_volatility * 1.5:
                adjusted['max_trades_per_hour'] = int(adjusted['max_trades_per_hour'] * 0.7)
                adjusted['max_trades_per_day'] = int(adjusted['max_trades_per_day'] * 0.7)
        
        # Adjust for recent performance
        if recent_performance:
            win_rate = recent_performance.get('win_rate', 0.5)
            recent_drawdown = recent_performance.get('current_drawdown_pct', 0)
            
            # Reduce size if performance is poor
            if win_rate < 0.4 or recent_drawdown > 0.10:
                performance_scalar = 0.5
                adjusted['max_position_size_usd'] *= performance_scalar
                adjusted['max_positions'] = max(1, int(adjusted['max_positions'] * 0.7))
                
                logger.warning(
                    "Reducing position limits due to poor performance",
                    win_rate=win_rate,
                    drawdown=recent_drawdown
                )
            
            # Increase size if performance is excellent
            elif win_rate > 0.6 and recent_drawdown < 0.05:
                performance_scalar = 1.2
                adjusted['max_position_size_usd'] *= performance_scalar
                
                logger.info(
                    "Increasing position limits due to strong performance",
                    win_rate=win_rate,
                    drawdown=recent_drawdown
                )
        
        return adjusted
    
    def get_position_size_for_kelly(
        self,
        current_equity: float,
        confidence: float,
        win_loss_ratio: float = 2.0
    ) -> float:
        """
        Calculate position size using Kelly criterion.
        
        Args:
            current_equity: Current account equity
            confidence: Win probability (0-1)
            win_loss_ratio: Expected win/loss ratio
            
        Returns:
            Recommended position size as % of equity
        """
        if confidence <= 0.5:
            return 0.0
        
        # Kelly formula: f = p - q/b
        # where p = win probability, q = loss probability, b = win/loss ratio
        q = 1 - confidence
        kelly_pct = confidence - q / win_loss_ratio
        
        # Apply fractional Kelly
        position_pct = kelly_pct * self.kelly_fraction
        
        # Cap at maximum per position
        position_pct = min(position_pct, self.max_position_pct_per_symbol)
        
        return max(0, position_pct)