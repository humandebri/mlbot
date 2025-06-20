"""
Daily performance report for Discord notification.
"""

import asyncio
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import pytz

from .logging import get_logger
from .discord_notifier import discord_notifier
from .account_monitor import AccountMonitor

logger = get_logger(__name__)


@dataclass
class DailyStats:
    """Daily trading statistics."""
    
    # Balance info
    starting_balance: float
    ending_balance: float
    daily_pnl: float
    daily_return_pct: float
    
    # Trading activity
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Risk metrics
    max_drawdown: float
    max_position_size: float
    total_volume: float
    
    # Market activity
    signals_generated: int
    high_confidence_signals: int
    liquidation_spikes_detected: int


class DailyReportManager:
    """
    Manages daily performance reports.
    Sends comprehensive daily summary to Discord.
    """
    
    def __init__(
        self, 
        account_monitor: AccountMonitor,
        report_time: str = "09:00",  # UTC time
        timezone: str = "UTC"
    ):
        """
        Initialize daily report manager.
        
        Args:
            account_monitor: Account monitor instance
            report_time: Time to send daily report (HH:MM format)
            timezone: Timezone for report time
        """
        self.account_monitor = account_monitor
        self.report_time = report_time
        self.timezone = pytz.timezone(timezone)
        
        # Parse report time
        hour, minute = map(int, report_time.split(":"))
        self.report_hour = hour
        self.report_minute = minute
        
        # Daily tracking
        self.daily_start_balance: Optional[float] = None
        self.daily_trades: List[Dict] = []
        self.daily_signals: List[Dict] = []
        self.daily_liquidations: int = 0
        
        # Background task
        self._report_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(
            f"Daily report manager initialized - Report time: {report_time} {timezone}"
        )
    
    async def start(self):
        """Start daily report manager."""
        if self._running:
            return
        
        self._running = True
        
        # Set initial daily balance
        if self.account_monitor.current_balance:
            self.daily_start_balance = self.account_monitor.current_balance.total_equity
        
        # Start report task
        self._report_task = asyncio.create_task(self._report_loop())
        logger.info("Daily report manager started")
    
    async def stop(self):
        """Stop daily report manager."""
        self._running = False
        
        if self._report_task:
            self._report_task.cancel()
            try:
                await self._report_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Daily report manager stopped")
    
    async def _report_loop(self):
        """Background task to send daily reports."""
        while self._running:
            try:
                # Calculate next report time
                next_report = self._get_next_report_time()
                wait_seconds = (next_report - datetime.now(self.timezone)).total_seconds()
                
                if wait_seconds > 0:
                    logger.info(
                        f"Next daily report in {wait_seconds/3600:.1f} hours at {next_report}"
                    )
                    await asyncio.sleep(wait_seconds)
                
                # Send daily report
                await self.send_daily_report()
                
                # Reset daily tracking
                self._reset_daily_tracking()
                
            except Exception as e:
                logger.error(f"Error in daily report loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def _get_next_report_time(self) -> datetime:
        """Calculate next report time."""
        now = datetime.now(self.timezone)
        report_time = now.replace(
            hour=self.report_hour,
            minute=self.report_minute,
            second=0,
            microsecond=0
        )
        
        # If report time has passed today, schedule for tomorrow
        if report_time <= now:
            report_time += timedelta(days=1)
        
        return report_time
    
    async def send_daily_report(self):
        """Send comprehensive daily report to Discord."""
        try:
            if not self.account_monitor.current_balance:
                logger.warning("No balance data available for daily report")
                return
            
            # Calculate daily stats
            stats = self._calculate_daily_stats()
            
            # Format report message
            report_date = datetime.now(self.timezone).strftime("%Y-%m-%d")
            
            # Create main report fields
            fields = {
                "📊 残高推移": (
                    f"開始: ${stats.starting_balance:,.2f}\n"
                    f"終了: ${stats.ending_balance:,.2f}\n"
                    f"損益: ${stats.daily_pnl:+,.2f} ({stats.daily_return_pct:+.2f}%)"
                ),
                "📈 取引実績": (
                    f"総取引数: {stats.total_trades}\n"
                    f"勝ち: {stats.winning_trades} / 負け: {stats.losing_trades}\n"
                    f"勝率: {stats.win_rate:.1f}%"
                ),
                "🎯 シグナル": (
                    f"生成数: {stats.signals_generated}\n"
                    f"高信頼度: {stats.high_confidence_signals}\n"
                    f"清算スパイク: {stats.liquidation_spikes_detected}"
                ),
                "⚡ リスク指標": (
                    f"最大ドローダウン: {stats.max_drawdown:.2f}%\n"
                    f"最大ポジション: ${stats.max_position_size:,.2f}\n"
                    f"総取引高: ${stats.total_volume:,.2f}"
                )
            }
            
            # Add performance commentary
            description = self._generate_performance_commentary(stats)
            
            # Determine color based on daily performance
            if stats.daily_pnl > 0:
                color = "00ff00"  # Green for profit
            elif stats.daily_pnl < 0:
                color = "ff0000"  # Red for loss
            else:
                color = "ffff00"  # Yellow for breakeven
            
            # Send Discord notification
            discord_notifier.send_notification(
                title=f"📅 日次レポート - {report_date}",
                description=description,
                color=color,
                fields=fields
            )
            
            logger.info(
                f"Daily report sent - PnL: ${stats.daily_pnl:+,.2f} ({stats.daily_return_pct:+.2f}%)"
            )
            
        except Exception as e:
            logger.error(f"Error sending daily report: {e}", exc_info=True)
    
    def _calculate_daily_stats(self) -> DailyStats:
        """Calculate daily statistics."""
        current_balance = self.account_monitor.current_balance.total_equity
        starting_balance = self.daily_start_balance or current_balance
        
        daily_pnl = current_balance - starting_balance
        daily_return_pct = (daily_pnl / starting_balance * 100) if starting_balance > 0 else 0
        
        # Calculate trade statistics
        winning_trades = len([t for t in self.daily_trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.daily_trades if t.get('pnl', 0) < 0])
        total_trades = len(self.daily_trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate risk metrics
        max_drawdown = 0
        max_position_size = 0
        total_volume = 0
        
        if self.daily_trades:
            max_position_size = max(t.get('position_size', 0) for t in self.daily_trades)
            total_volume = sum(t.get('volume', 0) for t in self.daily_trades)
        
        # Get drawdown from account monitor
        if hasattr(self.account_monitor, 'max_drawdown'):
            max_drawdown = self.account_monitor.max_drawdown * 100
        
        # Count signals
        high_confidence_signals = len([s for s in self.daily_signals if s.get('confidence', 0) > 0.6])
        
        return DailyStats(
            starting_balance=starting_balance,
            ending_balance=current_balance,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            max_position_size=max_position_size,
            total_volume=total_volume,
            signals_generated=len(self.daily_signals),
            high_confidence_signals=high_confidence_signals,
            liquidation_spikes_detected=self.daily_liquidations
        )
    
    def _generate_performance_commentary(self, stats: DailyStats) -> str:
        """Generate performance commentary based on daily stats."""
        comments = []
        
        # Overall performance
        if stats.daily_return_pct > 5:
            comments.append("🎉 素晴らしいパフォーマンス！")
        elif stats.daily_return_pct > 2:
            comments.append("✅ 良好な一日でした。")
        elif stats.daily_return_pct > 0:
            comments.append("📈 プラスで終了。")
        elif stats.daily_return_pct > -2:
            comments.append("📉 小幅な損失。")
        else:
            comments.append("⚠️ 大きな損失。リスク管理を見直しましょう。")
        
        # Win rate commentary
        if stats.win_rate > 60:
            comments.append(f"勝率{stats.win_rate:.0f}%は優秀です。")
        elif stats.win_rate < 40 and stats.total_trades > 5:
            comments.append(f"勝率{stats.win_rate:.0f}%は改善が必要です。")
        
        # Activity level
        if stats.total_trades == 0:
            comments.append("本日は取引なし。")
        elif stats.total_trades > 20:
            comments.append("活発な取引日でした。")
        
        # Signal quality
        if stats.high_confidence_signals > 0 and stats.signals_generated > 0:
            signal_quality = stats.high_confidence_signals / stats.signals_generated * 100
            if signal_quality > 20:
                comments.append(f"高品質なシグナル生成率: {signal_quality:.0f}%")
        
        return " ".join(comments)
    
    def _reset_daily_tracking(self):
        """Reset daily tracking data."""
        if self.account_monitor.current_balance:
            self.daily_start_balance = self.account_monitor.current_balance.total_equity
        
        self.daily_trades.clear()
        self.daily_signals.clear()
        self.daily_liquidations = 0
        
        logger.info("Daily tracking data reset")
    
    def record_trade(self, trade_data: Dict):
        """Record a trade for daily statistics."""
        self.daily_trades.append({
            **trade_data,
            'timestamp': datetime.now()
        })
    
    def record_signal(self, signal_data: Dict):
        """Record a signal for daily statistics."""
        self.daily_signals.append({
            **signal_data,
            'timestamp': datetime.now()
        })
    
    def record_liquidation(self):
        """Record a liquidation spike detection."""
        self.daily_liquidations += 1