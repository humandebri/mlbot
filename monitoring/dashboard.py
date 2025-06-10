"""
Simple monitoring dashboard for the trading system.

Provides real-time visualization of:
- System health
- Trading performance
- Risk metrics
- Recent signals
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any
import os
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align

console = Console()


class TradingDashboard:
    """Real-time monitoring dashboard."""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.session = None
        self.running = False
    
    async def start(self):
        """Start the dashboard."""
        self.session = aiohttp.ClientSession()
        self.running = True
        
        try:
            with Live(self.generate_layout(), refresh_per_second=1) as live:
                while self.running:
                    live.update(self.generate_layout())
                    await asyncio.sleep(1)
        finally:
            await self.session.close()
    
    def generate_layout(self) -> Layout:
        """Generate dashboard layout."""
        layout = Layout()
        
        # Create header
        header = Panel(
            Align.center(
                Text("💎 Liquidation Trading Bot Dashboard 💎", style="bold cyan"),
                vertical="middle"
            ),
            height=3
        )
        
        # Create main sections
        layout.split_column(
            Layout(header, size=3),
            Layout(name="main")
        )
        
        # Split main into columns
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Split left column
        layout["left"].split_column(
            Layout(name="health", size=10),
            Layout(name="performance", size=15),
            Layout(name="risk", size=15)
        )
        
        # Split right column
        layout["right"].split_column(
            Layout(name="positions", size=20),
            Layout(name="signals", size=20)
        )
        
        # Populate sections
        asyncio.create_task(self.update_sections(layout))
        
        return layout
    
    async def update_sections(self, layout: Layout):
        """Update all dashboard sections."""
        try:
            # Get system status
            status = await self.fetch_data("/system/status")
            health = await self.fetch_data("/system/health")
            
            # Update health section
            layout["health"].update(self.create_health_panel(health))
            
            # Update trading sections if system is running
            if status and "trading_system" in status and isinstance(status["trading_system"], dict):
                trading = status["trading_system"]
                
                # Performance
                if "trading" in trading and "performance" in trading["trading"]:
                    perf = trading["trading"]["performance"]
                    layout["performance"].update(self.create_performance_panel(perf))
                
                # Risk
                if "trading" in trading and "risk" in trading["trading"]:
                    risk = trading["trading"]["risk"]
                    layout["risk"].update(self.create_risk_panel(risk))
                
                # Positions
                if "trading" in trading and "positions" in trading["trading"]:
                    positions = trading["trading"]["positions"]
                    layout["positions"].update(self.create_positions_panel(positions))
                
                # Signals
                if "recent_signals" in trading:
                    signals = trading["recent_signals"]
                    layout["signals"].update(self.create_signals_panel(signals))
            
        except Exception as e:
            console.print(f"Error updating dashboard: {e}")
    
    def create_health_panel(self, health: Dict[str, Any]) -> Panel:
        """Create health status panel."""
        if not health:
            return Panel("No health data available", title="System Health")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status", justify="center")
        
        components = health.get("components", {})
        for name, is_healthy in components.items():
            status = "✅ Healthy" if is_healthy else "❌ Unhealthy"
            style = "green" if is_healthy else "red"
            table.add_row(name, Text(status, style=style))
        
        overall = health.get("status", "unknown")
        overall_style = "green" if overall == "healthy" else "yellow" if overall == "degraded" else "red"
        
        return Panel(
            table,
            title=f"System Health - {overall.upper()}",
            title_align="left",
            border_style=overall_style
        )
    
    def create_performance_panel(self, perf: Dict[str, Any]) -> Panel:
        """Create performance metrics panel."""
        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Trades", str(perf.get("total_trades", 0)))
        table.add_row("Win Rate", f"{perf.get('win_rate', 0):.1%}")
        table.add_row("Profit Factor", f"{perf.get('profit_factor', 0):.2f}")
        table.add_row("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}")
        
        pnl = perf.get("total_pnl", 0)
        pnl_style = "green" if pnl >= 0 else "red"
        table.add_row("Total P&L", Text(f"${pnl:,.2f}", style=pnl_style))
        
        return Panel(table, title="Performance Metrics", border_style="blue")
    
    def create_risk_panel(self, risk: Dict[str, Any]) -> Panel:
        """Create risk metrics panel."""
        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Current Equity", f"${risk.get('current_equity', 0):,.2f}")
        table.add_row("Total Exposure", f"${risk.get('total_exposure', 0):,.2f}")
        table.add_row("Exposure %", f"{risk.get('exposure_pct', 0):.1%}")
        
        drawdown = risk.get("current_drawdown", 0)
        dd_style = "green" if drawdown < 0.05 else "yellow" if drawdown < 0.10 else "red"
        table.add_row("Drawdown", Text(f"{drawdown:.1%}", style=dd_style))
        
        table.add_row("Daily P&L", f"${risk.get('daily_pnl', 0):,.2f}")
        table.add_row("VaR (95%)", f"{risk.get('current_var', 0):.2%}")
        
        halted = risk.get("trading_halted", False)
        halt_text = "🛑 HALTED" if halted else "✅ Active"
        halt_style = "red bold" if halted else "green"
        table.add_row("Trading Status", Text(halt_text, style=halt_style))
        
        return Panel(table, title="Risk Management", border_style="yellow")
    
    def create_positions_panel(self, positions: list) -> Panel:
        """Create positions panel."""
        if not positions:
            return Panel("No active positions", title="Active Positions")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Symbol", style="cyan")
        table.add_column("Side")
        table.add_column("Size", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        
        for pos in positions[:10]:  # Show top 10
            side_style = "green" if pos["side"] == "long" else "red"
            pnl = pos.get("unrealized_pnl", 0)
            pnl_style = "green" if pnl >= 0 else "red"
            
            table.add_row(
                pos["symbol"],
                Text(pos["side"].upper(), style=side_style),
                f"{pos['quantity']:.4f}",
                f"${pos['entry_price']:,.2f}",
                f"${pos['current_price']:,.2f}",
                Text(f"${pnl:,.2f}", style=pnl_style)
            )
        
        return Panel(table, title=f"Active Positions ({len(positions)})", border_style="green")
    
    def create_signals_panel(self, signals: list) -> Panel:
        """Create recent signals panel."""
        if not signals:
            return Panel("No recent signals", title="Recent Signals")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Time", style="dim")
        table.add_column("Symbol", style="cyan")
        table.add_column("Prediction", justify="right")
        table.add_column("Confidence", justify="right")
        table.add_column("Type")
        
        for signal in signals[:10]:  # Show last 10
            timestamp = datetime.fromisoformat(signal["timestamp"])
            time_str = timestamp.strftime("%H:%M:%S")
            
            pred = signal["prediction"]
            pred_style = "green" if pred > 0 else "red"
            
            signal_type = "🔥 LIQ" if signal.get("liquidation_detected") else "📊 ML"
            urgency = signal.get("urgency", "normal")
            urgency_style = "red" if urgency == "high" else "yellow" if urgency == "normal" else "dim"
            
            table.add_row(
                time_str,
                signal["symbol"],
                Text(f"{pred:.3%}", style=pred_style),
                f"{signal['confidence']:.1%}",
                Text(signal_type, style=urgency_style)
            )
        
        return Panel(table, title="Recent Trading Signals", border_style="purple")
    
    async def fetch_data(self, endpoint: str) -> Dict[str, Any]:
        """Fetch data from API."""
        if not self.session:
            return {}
        
        try:
            url = f"{self.api_url}{endpoint}"
            async with self.session.get(url, timeout=2) as response:
                if response.status == 200:
                    return await response.json()
                return {}
        except:
            return {}


async def main():
    """Run the dashboard."""
    dashboard = TradingDashboard()
    
    try:
        await dashboard.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())