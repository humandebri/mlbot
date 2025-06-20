"""
Position management system for tracking and managing trading positions.

Features:
- Real-time position tracking
- PnL calculation and monitoring
- Position aggregation across strategies
- Performance analytics
"""

import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

from ..common.config import settings
from ..common.logging import get_logger
from ..common.monitoring import (
    ACTIVE_POSITIONS, TOTAL_PNL, WIN_RATE,
    set_gauge, increment_counter
)

logger = get_logger(__name__)


@dataclass
class Trade:
    """Individual trade record."""
    
    trade_id: str
    position_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    order_type: str = "limit"
    fees: float = 0.0
    
    @property
    def value(self) -> float:
        """Trade value in USD."""
        return self.quantity * self.price


@dataclass
class PositionStats:
    """Position performance statistics."""
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_hold_time: timedelta = timedelta(0)
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0


class PositionManager:
    """
    Comprehensive position management system.
    
    Tracks all trading positions, calculates PnL in real-time,
    and provides performance analytics.
    """
    
    def __init__(self):
        """Initialize position manager."""
        
        # Active positions by ID
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # Trade history
        self.trades: List[Trade] = []
        self.trade_by_id: Dict[str, Trade] = {}
        
        # Position history (closed positions)
        self.closed_positions: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.daily_pnl: Dict[str, float] = defaultdict(float)
        self.symbol_pnl: Dict[str, float] = defaultdict(float)
        self.strategy_pnl: Dict[str, float] = defaultdict(float)
        
        # Position aggregation
        self.net_positions: Dict[str, float] = defaultdict(float)  # Symbol -> net quantity
        self.position_by_symbol: Dict[str, Set[str]] = defaultdict(set)  # Symbol -> position IDs
        
        # Performance stats cache
        self._stats_cache: Optional[PositionStats] = None
        self._stats_cache_time: Optional[datetime] = None
        self._stats_cache_ttl = 60  # seconds
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("Position manager initialized")
    
    async def open_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        strategy: str = "liquidation_cascade",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Open a new position.
        
        Args:
            position_id: Unique position identifier
            symbol: Trading symbol
            side: Position side ('long' or 'short')
            quantity: Position quantity
            entry_price: Entry price
            strategy: Strategy name
            metadata: Additional position metadata
            
        Returns:
            Position information
        """
        async with self._lock:
            if position_id in self.positions:
                logger.warning("Position already exists", position_id=position_id)
                return self.positions[position_id]
            
            position = {
                "position_id": position_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": entry_price,
                "entry_time": datetime.now(),
                "last_update": datetime.now(),
                "strategy": strategy,
                "status": "open",
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "fees_paid": 0.0,
                "trades": [],
                "metadata": metadata or {}
            }
            
            # Create entry trade
            entry_trade = Trade(
                trade_id=f"{position_id}_entry",
                position_id=position_id,
                symbol=symbol,
                side="buy" if side == "long" else "sell",
                quantity=quantity,
                price=entry_price,
                timestamp=datetime.now()
            )
            
            # Update tracking
            self.positions[position_id] = position
            self.trades.append(entry_trade)
            self.trade_by_id[entry_trade.trade_id] = entry_trade
            position["trades"].append(entry_trade.trade_id)
            
            # Update aggregations
            self._update_net_position(symbol, quantity if side == "long" else -quantity)
            self.position_by_symbol[symbol].add(position_id)
            
            # Update metrics
            set_gauge(ACTIVE_POSITIONS, len(self.positions))
            
            logger.info("Position opened",
                       position_id=position_id,
                       symbol=symbol,
                       side=side,
                       quantity=quantity,
                       entry_price=entry_price,
                       strategy=strategy)
            
            return position
    
    async def update_position(
        self,
        position_id: str,
        current_price: float,
        quantity_change: float = 0.0,
        fees: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """
        Update position with current market price.
        
        Args:
            position_id: Position ID
            current_price: Current market price
            quantity_change: Change in position quantity (for partial fills)
            fees: Additional fees
            
        Returns:
            Updated position or None if not found
        """
        async with self._lock:
            if position_id not in self.positions:
                logger.warning("Position not found", position_id=position_id)
                return None
            
            position = self.positions[position_id]
            position["current_price"] = current_price
            position["last_update"] = datetime.now()
            
            # Update quantity if changed
            if quantity_change != 0:
                old_quantity = position["quantity"]
                position["quantity"] += quantity_change
                
                # Create trade record for quantity change
                trade = Trade(
                    trade_id=f"{position_id}_adj_{int(datetime.now().timestamp())}",
                    position_id=position_id,
                    symbol=position["symbol"],
                    side="buy" if quantity_change > 0 else "sell",
                    quantity=abs(quantity_change),
                    price=current_price,
                    timestamp=datetime.now(),
                    fees=fees
                )
                
                self.trades.append(trade)
                self.trade_by_id[trade.trade_id] = trade
                position["trades"].append(trade.trade_id)
                
                # Update net position
                self._update_net_position(
                    position["symbol"],
                    quantity_change if position["side"] == "long" else -quantity_change
                )
            
            # Add fees
            position["fees_paid"] += fees
            
            # Calculate unrealized PnL
            if position["side"] == "long":
                position["unrealized_pnl"] = position["quantity"] * (current_price - position["entry_price"])
            else:  # short
                position["unrealized_pnl"] = position["quantity"] * (position["entry_price"] - current_price)
            
            position["unrealized_pnl"] -= position["fees_paid"]
            
            return position
    
    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        quantity: Optional[float] = None,
        fees: float = 0.0,
        reason: str = "manual"
    ) -> Optional[Dict[str, Any]]:
        """
        Close a position (fully or partially).
        
        Args:
            position_id: Position ID
            exit_price: Exit price
            quantity: Quantity to close (None for full close)
            fees: Trading fees
            reason: Close reason
            
        Returns:
            Closed position information
        """
        async with self._lock:
            if position_id not in self.positions:
                logger.warning("Position not found for closing", position_id=position_id)
                return None
            
            position = self.positions[position_id]
            close_quantity = quantity or position["quantity"]
            
            if close_quantity > position["quantity"]:
                logger.warning("Close quantity exceeds position size",
                              position_id=position_id,
                              close_quantity=close_quantity,
                              position_quantity=position["quantity"])
                close_quantity = position["quantity"]
            
            # Calculate realized PnL for this close
            if position["side"] == "long":
                trade_pnl = close_quantity * (exit_price - position["entry_price"])
            else:  # short
                trade_pnl = close_quantity * (position["entry_price"] - exit_price)
            
            trade_pnl -= fees
            
            # Create exit trade
            exit_trade = Trade(
                trade_id=f"{position_id}_exit_{int(datetime.now().timestamp())}",
                position_id=position_id,
                symbol=position["symbol"],
                side="sell" if position["side"] == "long" else "buy",
                quantity=close_quantity,
                price=exit_price,
                timestamp=datetime.now(),
                fees=fees
            )
            
            self.trades.append(exit_trade)
            self.trade_by_id[exit_trade.trade_id] = exit_trade
            position["trades"].append(exit_trade.trade_id)
            
            # Update position
            position["quantity"] -= close_quantity
            position["realized_pnl"] += trade_pnl
            position["fees_paid"] += fees
            position["last_update"] = datetime.now()
            
            # Update PnL tracking
            today = datetime.now().strftime("%Y-%m-%d")
            self.daily_pnl[today] += trade_pnl
            self.symbol_pnl[position["symbol"]] += trade_pnl
            self.strategy_pnl[position["strategy"]] += trade_pnl
            
            # Update net position
            self._update_net_position(
                position["symbol"],
                -close_quantity if position["side"] == "long" else close_quantity
            )
            
            # Full close
            if position["quantity"] <= 0:
                position["status"] = "closed"
                position["exit_time"] = datetime.now()
                position["exit_price"] = exit_price
                position["close_reason"] = reason
                position["hold_time"] = position["exit_time"] - position["entry_time"]
                position["total_pnl"] = position["realized_pnl"]
                
                # Move to closed positions
                self.closed_positions.append(position)
                del self.positions[position_id]
                self.position_by_symbol[position["symbol"]].discard(position_id)
                
                # Update win/loss stats
                if position["total_pnl"] > 0:
                    increment_counter("winning_trades", symbol=position["symbol"])
                else:
                    increment_counter("losing_trades", symbol=position["symbol"])
            
            # Update metrics
            set_gauge(ACTIVE_POSITIONS, len(self.positions))
            set_gauge(TOTAL_PNL, sum(self.daily_pnl.values()))
            
            # Invalidate stats cache
            self._stats_cache = None
            
            close_info = {
                "position_id": position_id,
                "symbol": position["symbol"],
                "side": position["side"],
                "quantity_closed": close_quantity,
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "trade_pnl": trade_pnl,
                "total_pnl": position["realized_pnl"],
                "fees": fees,
                "remaining_quantity": position["quantity"],
                "status": position["status"],
                "reason": reason
            }
            
            logger.info("Position closed", **close_info)
            
            return close_info
    
    async def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get position by ID."""
        async with self._lock:
            return self.positions.get(position_id)
    
    async def get_positions_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all positions for a symbol."""
        async with self._lock:
            position_ids = self.position_by_symbol.get(symbol, set())
            return [self.positions[pid] for pid in position_ids if pid in self.positions]
    
    async def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions."""
        async with self._lock:
            return list(self.positions.values())
    
    async def get_net_position(self, symbol: str) -> float:
        """Get net position for a symbol."""
        async with self._lock:
            return self.net_positions.get(symbol, 0.0)
    
    async def calculate_total_pnl(self) -> Dict[str, float]:
        """Calculate total PnL across all positions."""
        async with self._lock:
            realized_pnl = sum(p["realized_pnl"] for p in self.positions.values())
            unrealized_pnl = sum(p["unrealized_pnl"] for p in self.positions.values())
            closed_pnl = sum(p["total_pnl"] for p in self.closed_positions)
            
            return {
                "realized_pnl": realized_pnl + closed_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": realized_pnl + unrealized_pnl + closed_pnl
            }
    
    async def get_performance_stats(self, symbol: Optional[str] = None) -> PositionStats:
        """
        Get performance statistics.
        
        Args:
            symbol: Filter by symbol (None for all)
            
        Returns:
            Performance statistics
        """
        async with self._lock:
            # Check cache
            if (self._stats_cache is not None and 
                self._stats_cache_time is not None and
                (datetime.now() - self._stats_cache_time).total_seconds() < self._stats_cache_ttl and
                symbol is None):
                return self._stats_cache
            
            # Filter positions
            if symbol:
                positions = [p for p in self.closed_positions if p["symbol"] == symbol]
            else:
                positions = self.closed_positions
            
            if not positions:
                return PositionStats()
            
            # Calculate statistics
            stats = PositionStats()
            stats.total_trades = len(positions)
            
            pnls = [p["total_pnl"] for p in positions]
            stats.total_pnl = sum(pnls)
            stats.total_fees = sum(p["fees_paid"] for p in positions)
            
            winning_trades = [p for p in positions if p["total_pnl"] > 0]
            losing_trades = [p for p in positions if p["total_pnl"] <= 0]
            
            stats.winning_trades = len(winning_trades)
            stats.losing_trades = len(losing_trades)
            stats.win_rate = stats.winning_trades / stats.total_trades if stats.total_trades > 0 else 0
            
            if winning_trades:
                stats.avg_win = sum(p["total_pnl"] for p in winning_trades) / len(winning_trades)
            
            if losing_trades:
                stats.avg_loss = sum(p["total_pnl"] for p in losing_trades) / len(losing_trades)
            
            # Profit factor
            total_wins = sum(p["total_pnl"] for p in winning_trades) if winning_trades else 0
            total_losses = abs(sum(p["total_pnl"] for p in losing_trades)) if losing_trades else 1
            stats.profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            # Average hold time
            hold_times = [p["hold_time"] for p in positions if "hold_time" in p]
            if hold_times:
                avg_seconds = sum(ht.total_seconds() for ht in hold_times) / len(hold_times)
                stats.avg_hold_time = timedelta(seconds=avg_seconds)
            
            # Best/worst trades
            if pnls:
                stats.best_trade_pnl = max(pnls)
                stats.worst_trade_pnl = min(pnls)
            
            # Sharpe ratio (simplified daily)
            if len(self.daily_pnl) > 1:
                daily_returns = list(self.daily_pnl.values())
                if len(daily_returns) > 1:
                    returns_array = np.array(daily_returns)
                    stats.sharpe_ratio = (
                        np.mean(returns_array) / np.std(returns_array) * np.sqrt(365)
                        if np.std(returns_array) > 0 else 0
                    )
            
            # Max drawdown
            if pnls:
                cumulative_pnl = np.cumsum(pnls)
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = (cumulative_pnl - running_max) / np.maximum(running_max, 1)
                stats.max_drawdown = abs(np.min(drawdown))
            
            # Update cache
            if symbol is None:
                self._stats_cache = stats
                self._stats_cache_time = datetime.now()
            
            # Update metrics
            set_gauge(WIN_RATE, stats.win_rate)
            
            return stats
    
    async def get_daily_pnl(self, days: int = 30) -> pd.DataFrame:
        """Get daily PnL for the last N days."""
        async with self._lock:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days-1)
            
            date_range = pd.date_range(start_date, end_date)
            pnl_data = []
            
            for date in date_range:
                date_str = date.strftime("%Y-%m-%d")
                pnl_data.append({
                    "date": date,
                    "pnl": self.daily_pnl.get(date_str, 0.0)
                })
            
            return pd.DataFrame(pnl_data)
    
    async def get_symbol_performance(self) -> pd.DataFrame:
        """Get performance breakdown by symbol."""
        async with self._lock:
            data = []
            
            for symbol in set(p["symbol"] for p in self.closed_positions):
                symbol_positions = [p for p in self.closed_positions if p["symbol"] == symbol]
                
                if symbol_positions:
                    total_pnl = sum(p["total_pnl"] for p in symbol_positions)
                    trade_count = len(symbol_positions)
                    win_count = sum(1 for p in symbol_positions if p["total_pnl"] > 0)
                    
                    data.append({
                        "symbol": symbol,
                        "total_pnl": total_pnl,
                        "trade_count": trade_count,
                        "win_rate": win_count / trade_count if trade_count > 0 else 0,
                        "avg_pnl": total_pnl / trade_count if trade_count > 0 else 0
                    })
            
            return pd.DataFrame(data).sort_values("total_pnl", ascending=False)
    
    def _update_net_position(self, symbol: str, quantity_change: float) -> None:
        """Update net position for a symbol."""
        self.net_positions[symbol] += quantity_change
        
        # Clean up zero positions
        if abs(self.net_positions[symbol]) < 1e-8:
            del self.net_positions[symbol]