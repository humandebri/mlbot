"""
Smart balance monitoring with adaptive frequency.
Optimizes API usage and costs while maintaining effectiveness.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from enum import Enum

from .account_monitor import AccountMonitor, AccountBalance
from .logging import get_logger

logger = get_logger(__name__)


class MonitoringMode(Enum):
    """Monitoring frequency modes."""
    CONSERVATIVE = "conservative"  # 1 hour
    NORMAL = "normal"             # 15 minutes  
    ACTIVE = "active"             # 5 minutes
    URGENT = "urgent"             # 1 minute


class SmartBalanceMonitor(AccountMonitor):
    """
    Smart balance monitor with adaptive frequency.
    
    Automatically adjusts monitoring frequency based on:
    - Market volatility
    - Recent trading activity
    - Risk levels
    - Time of day
    """
    
    def __init__(self):
        # Start with normal mode (15 minutes)
        super().__init__(check_interval=900)  # 15 minutes
        
        self.mode = MonitoringMode.NORMAL
        self.last_trade_time: Optional[datetime] = None
        self.recent_volatility = 0.0
        self.risk_level = "low"
        
        # Mode settings
        self.mode_intervals = {
            MonitoringMode.CONSERVATIVE: 3600,  # 1 hour
            MonitoringMode.NORMAL: 900,         # 15 minutes
            MonitoringMode.ACTIVE: 300,         # 5 minutes
            MonitoringMode.URGENT: 60           # 1 minute
        }
        
        # Adaptive thresholds
        self.high_volatility_threshold = 0.05  # 5%
        self.trade_activity_window = 3600      # 1 hour
        self.risk_escalation_threshold = 0.05  # 5% drawdown
        
        logger.info("Smart balance monitor initialized",
                   initial_mode=self.mode.value,
                   interval=self.check_interval)
    
    async def _monitor_loop(self) -> None:
        """Enhanced monitoring loop with adaptive frequency."""
        while self._running:
            try:
                # Update balance
                await self.update_balance()
                
                # Analyze market conditions
                await self._analyze_conditions()
                
                # Update monitoring mode
                await self._update_monitoring_mode()
                
                # Check risk limits
                await self._check_risk_limits()
                
                # Sleep until next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error("Error in smart monitor loop", exception=e)
                await asyncio.sleep(min(self.check_interval, 30))
    
    async def _analyze_conditions(self) -> None:
        """Analyze market and account conditions."""
        if not self.current_balance or len(self.balance_history) < 2:
            return
        
        # Calculate recent volatility
        recent_balances = self.balance_history[-10:]  # Last 10 readings
        if len(recent_balances) >= 2:
            returns = []
            for i in range(1, len(recent_balances)):
                if recent_balances[i-1].total_equity > 0:
                    ret = (recent_balances[i].total_equity - recent_balances[i-1].total_equity) / recent_balances[i-1].total_equity
                    returns.append(abs(ret))
            
            if returns:
                self.recent_volatility = sum(returns) / len(returns)
        
        # Check recent trading activity
        now = datetime.now()
        if self.last_trade_time and (now - self.last_trade_time).total_seconds() < self.trade_activity_window:
            logger.debug("Recent trading activity detected")
        
        # Calculate risk level
        if self.initial_balance and self.peak_balance > 0:
            current_drawdown = (self.peak_balance - self.current_balance.total_equity) / self.peak_balance
            if current_drawdown > self.risk_escalation_threshold:
                self.risk_level = "high"
            elif current_drawdown > self.risk_escalation_threshold / 2:
                self.risk_level = "medium"
            else:
                self.risk_level = "low"
    
    async def _update_monitoring_mode(self) -> None:
        """Update monitoring mode based on conditions."""
        old_mode = self.mode
        
        # Determine new mode
        new_mode = MonitoringMode.NORMAL  # Default
        
        # Check for urgent conditions
        if self.risk_level == "high":
            new_mode = MonitoringMode.URGENT
            logger.warning("Switching to URGENT monitoring", reason="high_risk")
        
        # Check for active conditions
        elif (self.recent_volatility > self.high_volatility_threshold or 
              self._has_recent_trading() or
              self.risk_level == "medium"):
            new_mode = MonitoringMode.ACTIVE
            logger.info("Switching to ACTIVE monitoring", 
                       volatility=self.recent_volatility,
                       recent_trading=self._has_recent_trading(),
                       risk_level=self.risk_level)
        
        # Check for conservative conditions
        elif (self.risk_level == "low" and 
              self.recent_volatility < self.high_volatility_threshold / 2 and
              not self._has_recent_trading() and
              self._is_quiet_hours()):
            new_mode = MonitoringMode.CONSERVATIVE
            logger.info("Switching to CONSERVATIVE monitoring", reason="quiet_conditions")
        
        # Update mode if changed
        if new_mode != old_mode:
            self.mode = new_mode
            self.check_interval = self.mode_intervals[new_mode]
            
            logger.info("Monitoring mode changed",
                       old_mode=old_mode.value,
                       new_mode=new_mode.value,
                       new_interval=self.check_interval)
            
            # Log API usage estimate
            daily_calls = 86400 / self.check_interval
            logger.info("API usage estimate",
                       daily_calls=int(daily_calls),
                       mode=new_mode.value)
    
    def _has_recent_trading(self) -> bool:
        """Check if there was recent trading activity."""
        if not self.last_trade_time:
            return False
        
        time_since_trade = (datetime.now() - self.last_trade_time).total_seconds()
        return time_since_trade < self.trade_activity_window
    
    def _is_quiet_hours(self) -> bool:
        """Check if it's during quiet market hours."""
        now = datetime.now()
        hour = now.hour
        
        # Quiet hours: 22:00 - 06:00 UTC (roughly Asian night)
        return hour >= 22 or hour <= 6
    
    def on_trade_executed(self, trade_info: Dict[str, Any]) -> None:
        """Called when a trade is executed."""
        self.last_trade_time = datetime.now()
        logger.info("Trade executed, updating monitor", 
                   symbol=trade_info.get('symbol'),
                   side=trade_info.get('side'))
        
        # Force balance update on next cycle
        asyncio.create_task(self._immediate_balance_check())
    
    async def _immediate_balance_check(self) -> None:
        """Perform immediate balance check after trade."""
        try:
            await asyncio.sleep(5)  # Wait a moment for settlement
            await self.update_balance()
            logger.info("Immediate balance check completed")
        except Exception as e:
            logger.error("Immediate balance check failed", exception=e)
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        daily_calls = 86400 / self.check_interval
        monthly_calls = daily_calls * 30
        monthly_cost = (monthly_calls / 1_000_000) * 3.50  # AWS API Gateway cost
        
        stats = super().get_performance_stats()
        stats.update({
            "monitoring_mode": self.mode.value,
            "check_interval": self.check_interval,
            "daily_api_calls": int(daily_calls),
            "monthly_api_calls": int(monthly_calls),
            "monthly_cost_usd": monthly_cost,
            "recent_volatility": self.recent_volatility,
            "risk_level": self.risk_level,
            "has_recent_trading": self._has_recent_trading()
        })
        
        return stats