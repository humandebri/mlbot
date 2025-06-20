#!/usr/bin/env python3
"""
Simplified Improved ML Production Bot
Uses real historical data without complex config dependencies
"""

import asyncio
import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))

# Import required components with minimal dependencies
import aiohttp
import redis.asyncio as redis
from improved_feature_generator import ImprovedFeatureGenerator


class DiscordNotifier:
    """Simple Discord notifier."""
    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK")
        self.session = None
    
    async def setup(self):
        self.session = aiohttp.ClientSession()
    
    async def send(self, title: str, description: str, color: str = "3498db"):
        if not self.webhook_url or not self.session:
            return
        
        color_int = int(color, 16) if color else 3498979
        
        payload = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color_int,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        
        try:
            async with self.session.post(self.webhook_url, json=payload) as resp:
                if resp.status != 204:
                    logger.error(f"Discord webhook failed: {resp.status}")
        except Exception as e:
            logger.error(f"Discord notification error: {e}")
    
    async def cleanup(self):
        if self.session:
            await self.session.close()


class SimpleBybitClient:
    """Simple Bybit REST API client."""
    def __init__(self):
        self.session = None
        self.base_url = "https://api.bybit.com"
    
    async def setup(self):
        self.session = aiohttp.ClientSession()
    
    async def get_ticker(self, symbol: str) -> Dict:
        if not self.session:
            return {}
        
        url = f"{self.base_url}/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}
        
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("result", {}).get("list"):
                        return data["result"]["list"][0]
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
        
        return {}
    
    async def cleanup(self):
        if self.session:
            await self.session.close()


class SimpleMLInference:
    """Simple ML inference using ONNX."""
    def __init__(self, model_path: str = "models/v3.1_improved/model.onnx"):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(self.model_path)
            logger.info(f"Loaded ONNX model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> float:
        if self.model is None:
            return 0.5
        
        try:
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: features})
            # Get probability of positive class
            if len(output[0].shape) > 1 and output[0].shape[1] == 2:
                return float(output[0][0][1])  # Probability of class 1
            else:
                return float(output[0][0])
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5


class SimpleImprovedBot:
    """Simplified ML production bot with improved features."""
    
    def __init__(self):
        self.discord = DiscordNotifier()
        self.bybit = SimpleBybitClient()
        self.ml_model = SimpleMLInference()
        self.feature_generator = ImprovedFeatureGenerator()
        
        # Configuration
        self.symbols = ["BTCUSDT", "ETHUSDT", "ICPUSDT"]
        self.min_confidence = 0.65
        self.signal_cooldown = 300  # 5 minutes
        self.check_interval = 10  # 10 seconds
        
        # Tracking
        self.last_signal_time = {}
        self.signal_count = 0
        self.prediction_count = 0
        self.running = False
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Simple Improved ML Bot...")
        
        try:
            # Setup async components
            await self.discord.setup()
            await self.bybit.setup()
            
            # Load ML model
            self.ml_model.load_model()
            
            # Load historical data
            logger.info("Loading historical data...")
            for symbol in self.symbols:
                self.feature_generator.update_historical_cache(symbol)
                logger.info(f"Loaded historical data for {symbol}")
            
            # Send startup notification
            await self.discord.send(
                title="🤖 Simple ML Bot Started",
                description=(
                    f"**System:** Simple Improved ML Bot\n"
                    f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"**Symbols:** {', '.join(self.symbols)}\n"
                    f"**Model:** v3.1_improved\n"
                    f"**Features:** Real historical data\n"
                    f"**Min Confidence:** {self.min_confidence * 100:.0f}%"
                ),
                color="00FF00"
            )
            
            logger.info("Bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    async def should_send_signal(self, symbol: str) -> bool:
        """Check if we should send a signal."""
        now = datetime.now()
        if symbol not in self.last_signal_time:
            return True
        
        time_since_last = (now - self.last_signal_time[symbol]).total_seconds()
        return time_since_last >= self.signal_cooldown
    
    async def process_symbol(self, symbol: str):
        """Process a single symbol."""
        try:
            # Get ticker data
            ticker = await self.bybit.get_ticker(symbol)
            if not ticker:
                logger.warning(f"No ticker data for {symbol}")
                return
            
            # Generate features
            features = self.feature_generator.generate_features(ticker, symbol)
            normalized = self.feature_generator.normalize_features(features)
            
            # Get prediction
            prediction = self.ml_model.predict(normalized.reshape(1, -1))
            self.prediction_count += 1
            
            # Calculate confidence
            confidence = abs(prediction - 0.5) * 2
            confidence = min(max(confidence, 0.0), 1.0)
            direction = "BUY" if prediction > 0.5 else "SELL"
            
            current_price = float(ticker.get("lastPrice", 0))
            
            logger.info(
                f"📊 {symbol}: pred={prediction:.4f}, "
                f"conf={confidence*100:.2f}%, dir={direction}, "
                f"price=${current_price:,.2f}"
            )
            
            # Check if we should send signal
            if confidence >= self.min_confidence and await self.should_send_signal(symbol):
                self.signal_count += 1
                self.last_signal_time[symbol] = datetime.now()
                
                # Send signal
                await self.send_trading_signal(
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    price=current_price,
                    prediction=prediction
                )
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    async def send_trading_signal(self, symbol: str, direction: str, confidence: float,
                                 price: float, prediction: float):
        """Send trading signal."""
        try:
            # Get 24h change
            ticker = await self.bybit.get_ticker(symbol)
            change_24h = float(ticker.get("price24hPcnt", 0)) * 100 if ticker else 0
            
            await self.discord.send(
                title=f"🎯 Signal #{self.signal_count} - {symbol}",
                description=(
                    f"**Direction:** {direction}\n"
                    f"**Confidence:** {confidence*100:.1f}%\n"
                    f"**ML Score:** {prediction:.4f}\n"
                    f"**Price:** ${price:,.2f}\n"
                    f"**24h Change:** {change_24h:+.2f}%\n"
                    f"**Time:** {datetime.now().strftime('%H:%M:%S')}"
                ),
                color="00FF00" if direction == "BUY" else "FF0000"
            )
            
            logger.info(f"Signal sent for {symbol}")
            
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
    
    async def trading_loop(self):
        """Main trading loop."""
        self.running = True
        last_report = datetime.now()
        
        while self.running:
            try:
                # Process each symbol
                for symbol in self.symbols:
                    await self.process_symbol(symbol)
                
                # Send hourly report
                if (datetime.now() - last_report).total_seconds() >= 3600:
                    await self.send_hourly_report()
                    last_report = datetime.now()
                
                # Wait before next iteration
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)
    
    async def send_hourly_report(self):
        """Send hourly report."""
        try:
            await self.discord.send(
                title="📊 Hourly Report",
                description=(
                    f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                    f"**Predictions:** {self.prediction_count}\n"
                    f"**Signals:** {self.signal_count}\n"
                    f"**Status:** Running smoothly"
                ),
                color="3498db"
            )
        except Exception as e:
            logger.error(f"Error sending report: {e}")
    
    async def run(self):
        """Run the bot."""
        try:
            await self.initialize()
            await self.trading_loop()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            await self.discord.send(
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
        
        await self.discord.send(
            title="🛑 Bot Stopped",
            description=(
                f"Bot stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Total predictions: {self.prediction_count}\n"
                f"Total signals: {self.signal_count}"
            ),
            color="FFA500"
        )
        
        # Cleanup
        await self.discord.cleanup()
        await self.bybit.cleanup()
        self.feature_generator.close()


async def main():
    """Main entry point."""
    bot = SimpleImprovedBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())