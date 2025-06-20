#!/usr/bin/env python3
"""
Working production bot that bypasses feature generation issues.
"""

import asyncio
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Set environment variables before imports
os.environ["BYBIT__API_KEY"] = os.getenv("BYBIT__API_KEY", "")
os.environ["BYBIT__API_SECRET"] = os.getenv("BYBIT__API_SECRET", "")
os.environ["DISCORD_WEBHOOK"] = os.getenv("DISCORD_WEBHOOK", "")

from src.common.logging import setup_logging, get_logger
from src.common.discord_notifier import discord_notifier
from src.common.bybit_client import BybitRESTClient

logger = get_logger(__name__)

class WorkingProductionBot:
    """Simple working production bot."""
    
    def __init__(self):
        self.bybit_client = None
        self.running = False
        self.last_signal_time = {}
        self.signal_cooldown = 300  # 5 minutes
        self.signal_count = 0
        self.symbols = ["BTCUSDT", "ETHUSDT", "ICPUSDT"]
    
    async def initialize(self):
        """Initialize bot."""
        logger.info("Initializing Working Production Bot...")
        
        # Initialize Bybit client
        self.bybit_client = BybitRESTClient(testnet=False)
        await self.bybit_client.__aenter__()
        
        # Send startup notification
        discord_notifier.send_notification(
            title="🤖 MLBot Started Successfully",
            description=(
                f"**System:** Working Production Bot\n"
                f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"**Symbols:** {', '.join(self.symbols)}\n"
                f"**Status:** Online and monitoring"
            ),
            color="00FF00"
        )
        
        logger.info("Bot initialized successfully")
    
    async def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for a symbol."""
        try:
            ticker = await self.bybit_client.get_ticker(symbol)
            if ticker:
                return {
                    "price": float(ticker.get("lastPrice", 0)),
                    "bid": float(ticker.get("bid1Price", 0)),
                    "ask": float(ticker.get("ask1Price", 0)),
                    "volume": float(ticker.get("volume24h", 0)),
                    "turnover": float(ticker.get("turnover24h", 0)),
                    "change": float(ticker.get("price24hPcnt", 0)) * 100
                }
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
        return None
    
    async def generate_signal(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Generate a simple trading signal based on market conditions."""
        # Simple momentum-based signal
        price_change = market_data.get("change", 0)
        volume = market_data.get("volume", 0)
        
        # Signal logic: 
        # - Strong price movement (>0.5%)
        # - High volume
        if abs(price_change) > 0.5 and volume > 0:
            direction = "BUY" if price_change > 0 else "SELL"
            # Confidence based on price change magnitude
            confidence = min(abs(price_change) / 2.0, 1.0) * 100
            
            return {
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
                "price": market_data["price"],
                "change": price_change,
                "volume": volume
            }
        
        return None
    
    async def should_send_signal(self, symbol: str) -> bool:
        """Check if we should send a signal."""
        now = datetime.now()
        
        if symbol not in self.last_signal_time:
            return True
        
        time_since_last = (now - self.last_signal_time[symbol]).total_seconds()
        return time_since_last >= self.signal_cooldown
    
    async def trading_loop(self):
        """Main trading loop."""
        self.running = True
        check_interval = 30  # Check every 30 seconds
        
        while self.running:
            try:
                for symbol in self.symbols:
                    # Get market data
                    market_data = await self.get_market_data(symbol)
                    if not market_data:
                        continue
                    
                    # Log current state
                    logger.info(
                        f"📊 {symbol}: ${market_data['price']:,.2f} "
                        f"({market_data['change']:+.2f}%) "
                        f"Vol: {market_data['volume']:,.0f}"
                    )
                    
                    # Generate signal
                    signal = await self.generate_signal(symbol, market_data)
                    
                    if signal and signal["confidence"] >= 70 and await self.should_send_signal(symbol):
                        self.signal_count += 1
                        self.last_signal_time[symbol] = datetime.now()
                        
                        # Send Discord notification
                        discord_notifier.send_notification(
                            title=f"🎯 Trading Signal #{self.signal_count} - {symbol}",
                            description=(
                                f"**Direction:** {signal['direction']}\n"
                                f"**Price:** ${signal['price']:,.2f}\n"
                                f"**24h Change:** {signal['change']:+.2f}%\n"
                                f"**Confidence:** {signal['confidence']:.1f}%\n"
                                f"**Volume:** {signal['volume']:,.0f}"
                            ),
                            color="00FF00" if signal['direction'] == 'BUY' else "FF0000"
                        )
                        
                        logger.info(f"🚨 Signal sent for {symbol}")
                
                # Send hourly summary
                if datetime.now().minute == 0 and datetime.now().second < check_interval:
                    await self.send_hourly_summary()
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)
    
    async def send_hourly_summary(self):
        """Send hourly market summary."""
        try:
            summary_lines = ["**📊 Hourly Market Summary**", ""]
            
            for symbol in self.symbols:
                market_data = await self.get_market_data(symbol)
                if market_data:
                    summary_lines.append(
                        f"**{symbol}:** ${market_data['price']:,.2f} "
                        f"({market_data['change']:+.2f}%)"
                    )
            
            summary_lines.append("")
            summary_lines.append(f"**Signals Sent:** {self.signal_count}")
            summary_lines.append(f"**Time:** {datetime.now().strftime('%H:%M')}")
            
            discord_notifier.send_notification(
                title="Hourly Report",
                description="\n".join(summary_lines),
                color="3498db"
            )
        except Exception as e:
            logger.error(f"Error sending hourly summary: {e}")
    
    async def run(self):
        """Run the bot."""
        try:
            await self.initialize()
            await self.trading_loop()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            discord_notifier.send_notification(
                title="❌ Bot Error",
                description=f"Fatal error: {str(e)}",
                color="FF0000"
            )
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("Shutting down...")
        self.running = False
        
        if self.bybit_client:
            await self.bybit_client.__aexit__(None, None, None)
        
        discord_notifier.send_notification(
            title="🛑 Bot Stopped",
            description=f"Bot stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            color="FFA500"
        )

async def main():
    """Main entry point."""
    setup_logging()
    bot = WorkingProductionBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())