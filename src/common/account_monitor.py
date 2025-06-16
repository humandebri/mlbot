"""
Account balance monitoring and management.
Tracks real account balance via Bybit API and adjusts position sizing accordingly.
"""

import asyncio
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
import hashlib
import hmac
import json

from .config import settings
from .logging import get_logger
from .monitoring import set_gauge

logger = get_logger(__name__)


@dataclass
class AccountBalance:
    """Account balance information."""
    
    total_equity: float
    available_balance: float
    unrealized_pnl: float
    used_margin: float
    currency: str = "USDT"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def free_margin_pct(self) -> float:
        """Free margin percentage."""
        if self.total_equity <= 0:
            return 0.0
        return self.available_balance / self.total_equity * 100


class AccountMonitor:
    """
    Real-time account balance monitoring.
    
    Features:
    - Periodic balance checking
    - Position size adjustment based on equity
    - Risk limit monitoring
    - Performance tracking
    """
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize account monitor.
        
        Args:
            check_interval: Balance check interval in seconds
        """
        self.check_interval = check_interval
        self.current_balance: Optional[AccountBalance] = None
        self.balance_history: List[AccountBalance] = []
        
        # Performance tracking
        self.initial_balance: Optional[float] = None
        self.peak_balance: float = 0.0
        self.drawdown_start: Optional[float] = None
        
        # Risk monitoring
        self.max_daily_loss_usd = float(settings.trading.max_daily_loss_usd)
        self.max_drawdown_pct = float(settings.trading.max_drawdown_pct)
        
        # API credentials
        self.api_key = settings.bybit.api_key
        self.api_secret = settings.bybit.api_secret
        self.base_url = settings.get_bybit_urls()["rest"]
        
        # Background task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("Account monitor initialized", 
                   check_interval=check_interval,
                   max_daily_loss=self.max_daily_loss_usd,
                   max_drawdown_pct=self.max_drawdown_pct)
    
    async def start(self) -> None:
        """Start balance monitoring."""
        if self._running:
            return
        
        self._running = True
        
        # Initial balance check
        await self.update_balance()
        
        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Account monitoring started")
    
    async def stop(self) -> None:
        """Stop balance monitoring."""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Account monitoring stopped")
    
    async def update_balance(self) -> Optional[AccountBalance]:
        """Update account balance from Bybit API."""
        try:
            balance_data = await self._get_wallet_balance()
            
            if not balance_data:
                logger.error("Failed to get balance data")
                return None
            
            # Parse USDT balance
            usdt_balance = None
            for coin in balance_data.get("list", [{}])[0].get("coin", []):
                if coin.get("coin") == "USDT":
                    usdt_balance = coin
                    break
            
            if not usdt_balance:
                logger.error("USDT balance not found")
                return None
            
            # Create balance object with safe float conversion
            def safe_float(value, default=0.0):
                """Convert value to float safely."""
                if value is None or value == '':
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            equity = safe_float(usdt_balance.get("equity"))
            available = safe_float(usdt_balance.get("availableToWithdraw"))
            unrealized_pnl = safe_float(usdt_balance.get("unrealisedPnl"))
            usd_value = safe_float(usdt_balance.get("usdValue"))
            
            balance = AccountBalance(
                total_equity=equity,
                available_balance=available,
                unrealized_pnl=unrealized_pnl,
                used_margin=usd_value - available
            )
            
            # Update current balance
            self.current_balance = balance
            self.balance_history.append(balance)
            
            # Set initial balance if first time
            if self.initial_balance is None:
                self.initial_balance = balance.total_equity
                logger.info("Initial balance set", balance=self.initial_balance)
            
            # Update peak balance
            if balance.total_equity > self.peak_balance:
                self.peak_balance = balance.total_equity
                self.drawdown_start = None
            
            # Update metrics (disabled for now due to monitoring setup issues)
            # set_gauge("account_equity", balance.total_equity)
            # set_gauge("available_balance", balance.available_balance)
            # set_gauge("unrealized_pnl", balance.unrealized_pnl)
            
            # Calculate returns
            if self.initial_balance and self.initial_balance > 0:
                total_return_pct = (balance.total_equity - self.initial_balance) / self.initial_balance * 100
                # set_gauge("total_return_pct", total_return_pct)
                pass
            
            logger.info("Balance updated",
                       equity=balance.total_equity,
                       available=balance.available_balance,
                       unrealized_pnl=balance.unrealized_pnl,
                       free_margin_pct=balance.free_margin_pct)
            
            # Keep only recent history
            if len(self.balance_history) > 1000:
                self.balance_history = self.balance_history[-500:]
            
            return balance
            
        except Exception as e:
            logger.error("Error updating balance", exception=e)
            # Log error instead of using increment_counter
            logger.error("Balance update error counter would increment here")
            return None
    
    async def _get_wallet_balance(self) -> Optional[Dict]:
        """Get wallet balance from Bybit API."""
        endpoint = "/v5/account/wallet-balance"
        params = {
            "accountType": "UNIFIED",
            "coin": "USDT"
        }
        
        return await self._make_authenticated_request("GET", endpoint, params)
    
    async def _make_authenticated_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make authenticated request to Bybit API."""
        
        if not self.api_key or not self.api_secret:
            logger.error("API credentials not configured")
            return None
        
        timestamp = str(int(time.time() * 1000))
        
        # Prepare query string
        if params:
            query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        else:
            query_string = ""
        
        # Create signature
        param_str = f"{timestamp}{self.api_key}5000{query_string}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Headers
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }
        
        # Make request
        url = f"{self.base_url}{endpoint}"
        if query_string:
            url += f"?{query_string}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0:
                            return data.get("result")
                        else:
                            logger.error("API error", 
                                       code=data.get("retCode"),
                                       message=data.get("retMsg"))
                    else:
                        logger.error("HTTP error", status=response.status)
                        
        except Exception as e:
            logger.error("Request failed", exception=e)
        
        return None
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await self.update_balance()
                
                # Check risk limits
                await self._check_risk_limits()
                
                # Sleep until next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error("Error in monitor loop", exception=e)
                await asyncio.sleep(min(self.check_interval, 30))
    
    async def _check_risk_limits(self) -> None:
        """Check and enforce risk limits."""
        if not self.current_balance or not self.initial_balance:
            return
        
        # Check daily loss
        today = datetime.now().date()
        today_balances = [b for b in self.balance_history 
                         if b.timestamp.date() == today]
        
        if len(today_balances) >= 2:
            daily_start = today_balances[0].total_equity
            current_equity = self.current_balance.total_equity
            daily_pnl = current_equity - daily_start
            
            if daily_pnl < -self.max_daily_loss_usd:
                logger.critical("Daily loss limit exceeded",
                               daily_pnl=daily_pnl,
                               limit=self.max_daily_loss_usd)
                # Risk limit violation - daily loss
                logger.critical("Risk limit violation counter would increment here", limit_type="daily_loss")
        
        # Check drawdown
        if self.peak_balance > 0:
            current_drawdown_pct = (self.peak_balance - self.current_balance.total_equity) / self.peak_balance * 100
            
            if current_drawdown_pct > self.max_drawdown_pct * 100:
                logger.critical("Drawdown limit exceeded",
                               current_drawdown_pct=current_drawdown_pct,
                               limit_pct=self.max_drawdown_pct * 100)
                # Risk limit violation - drawdown
                logger.critical("Risk limit violation counter would increment here", limit_type="drawdown")
    
    def get_current_position_size(self, base_pct: float = 0.05) -> float:
        """
        Calculate current position size based on available balance.
        
        Args:
            base_pct: Base position size percentage
            
        Returns:
            Position size in USD
        """
        if not self.current_balance:
            return 0.0
        
        # Use available balance for position sizing
        position_size = self.current_balance.available_balance * base_pct
        
        # Apply minimum position size
        min_size = 12.0  # Minimum for ICPUSDT
        
        return max(position_size, min_size)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.current_balance or not self.initial_balance:
            return {}
        
        current_equity = self.current_balance.total_equity
        total_return = (current_equity - self.initial_balance) / self.initial_balance * 100
        
        # Calculate drawdown
        max_drawdown_pct = 0.0
        if self.peak_balance > 0:
            max_drawdown_pct = (self.peak_balance - current_equity) / self.peak_balance * 100
        
        return {
            "initial_balance": self.initial_balance,
            "current_equity": current_equity,
            "available_balance": self.current_balance.available_balance,
            "unrealized_pnl": self.current_balance.unrealized_pnl,
            "total_return_pct": total_return,
            "peak_balance": self.peak_balance,
            "max_drawdown_pct": max_drawdown_pct,
            "free_margin_pct": self.current_balance.free_margin_pct
        }