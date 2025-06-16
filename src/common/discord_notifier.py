"""Discord notification handler for trading events."""

import os
import asyncio
from typing import Optional, Dict, Any
from discord_webhook import DiscordWebhook, DiscordEmbed
from .logging import get_logger

logger = get_logger(__name__)


class DiscordNotifier:
    """Handles Discord notifications for trading events."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize Discord notifier.
        
        Args:
            webhook_url: Discord webhook URL. If not provided, reads from DISCORD_WEBHOOK env var.
        """
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK")
        self.enabled = bool(self.webhook_url)
        
        if not self.enabled:
            logger.warning("Discord notifications disabled - no webhook URL provided")
        else:
            logger.info("Discord notifications enabled")
    
    def send_notification(self, title: str, description: str, color: str = "03b2f8",
                         fields: Optional[Dict[str, Any]] = None) -> bool:
        """Send a notification to Discord.
        
        Args:
            title: Notification title
            description: Notification description
            color: Hex color code (default: blue)
            fields: Additional fields to include
            
        Returns:
            bool: True if sent successfully
        """
        if not self.enabled:
            return False
        
        try:
            webhook = DiscordWebhook(url=self.webhook_url)
            
            embed = DiscordEmbed(
                title=title,
                description=description,
                color=color
            )
            
            if fields:
                for name, value in fields.items():
                    embed.add_embed_field(name=name, value=str(value), inline=True)
            
            webhook.add_embed(embed)
            response = webhook.execute()
            
            if response.status_code == 200:
                logger.debug(f"Discord notification sent: {title}")
                return True
            else:
                logger.error(f"Failed to send Discord notification: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False
    
    def send_trade_signal(self, symbol: str, side: str, price: float, 
                         confidence: float, expected_pnl: float) -> bool:
        """Send trade signal notification."""
        color = "00ff00" if side.upper() == "BUY" else "ff0000"
        
        fields = {
            "Symbol": symbol,
            "Side": side.upper(),
            "Price": f"${price:,.2f}",
            "Confidence": f"{confidence:.2%}",
            "Expected PnL": f"{expected_pnl:.2%}"
        }
        
        return self.send_notification(
            title="🚨 Trade Signal",
            description=f"New {side.upper()} signal for {symbol}",
            color=color,
            fields=fields
        )
    
    def send_order_executed(self, symbol: str, side: str, price: float, 
                           quantity: float, order_id: str) -> bool:
        """Send order execution notification."""
        fields = {
            "Symbol": symbol,
            "Side": side.upper(),
            "Price": f"${price:,.2f}",
            "Quantity": f"{quantity}",
            "Order ID": order_id[:8] + "..."
        }
        
        return self.send_notification(
            title="✅ Order Executed",
            description=f"Order placed successfully",
            color="00ff00",
            fields=fields
        )
    
    def send_system_status(self, status: str, message: str) -> bool:
        """Send system status notification."""
        color = "00ff00" if status == "online" else "ff0000"
        emoji = "🟢" if status == "online" else "🔴"
        
        return self.send_notification(
            title=f"{emoji} System Status: {status.upper()}",
            description=message,
            color=color
        )
    
    def send_error(self, error_type: str, message: str) -> bool:
        """Send error notification."""
        return self.send_notification(
            title=f"❌ Error: {error_type}",
            description=message,
            color="ff0000"
        )
    
    def send_daily_summary(self, total_trades: int, total_pnl: float, 
                          win_rate: float, best_trade: Optional[Dict[str, Any]] = None) -> bool:
        """Send daily trading summary."""
        fields = {
            "Total Trades": total_trades,
            "Total PnL": f"${total_pnl:,.2f}",
            "Win Rate": f"{win_rate:.2%}"
        }
        
        if best_trade:
            fields["Best Trade"] = f"{best_trade['symbol']} +${best_trade['pnl']:,.2f}"
        
        emoji = "📈" if total_pnl > 0 else "📉"
        color = "00ff00" if total_pnl > 0 else "ff0000"
        
        return self.send_notification(
            title=f"{emoji} Daily Trading Summary",
            description=f"Today's Performance Report",
            color=color,
            fields=fields
        )


# Global notifier instance
discord_notifier = DiscordNotifier()