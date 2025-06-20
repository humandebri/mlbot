#!/usr/bin/env python3
"""
Simple Improved ML Bot with proper ONNX output handling
"""

import asyncio
import aiohttp
import numpy as np
import onnxruntime as ort
from datetime import datetime
import logging
import os
import json
from improved_feature_generator import ImprovedFeatureGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Discord webhook from environment
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "").strip()

class SimpleImprovedBot:
    def __init__(self):
        self.feature_generator = ImprovedFeatureGenerator()
        self.model_path = "models/v3.1_improved/model.onnx"
        self.session = None
        self.running = False
        self.symbols = ["BTCUSDT", "ETHUSDT", "ICPUSDT"]
        self.last_signal_time = {}
        self.signal_cooldown = 300  # 5 minutes
        self.min_confidence = 0.65
        self.signal_count = 0
        
    def load_model(self):
        """Load ONNX model."""
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info(f"Loaded ONNX model from {self.model_path}")
        logger.info(f"Model outputs: {self.output_names}")
        
    async def get_ticker(self, session, symbol):
        """Get ticker data from Bybit."""
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        async with session.get(url) as response:
            data = await response.json()
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                return data["result"]["list"][0]
        return None
        
    def predict(self, features):
        """Make prediction with ONNX model."""
        # Run inference
        outputs = self.session.run(None, {self.input_name: features.reshape(1, -1).astype(np.float32)})
        
        # Handle dual output model (class labels + probability dict)
        if len(outputs) > 1 and isinstance(outputs[1], list):
            # Extract probability of positive class from dictionary
            prob_dict = outputs[1][0]  # First prediction
            if isinstance(prob_dict, dict):
                prediction = prob_dict.get(1, 0.5)  # Get probability of class 1
            else:
                prediction = outputs[0][0]  # Fallback to class label
        else:
            # Single output model
            prediction = outputs[0][0]
            
        return float(prediction)
        
    async def send_discord(self, title, description, color="3498db"):
        """Send Discord notification."""
        if not DISCORD_WEBHOOK:
            return
            
        embed = {
            "title": title,
            "description": description,
            "color": int(color, 16),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        payload = {"embeds": [embed]}
        
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(DISCORD_WEBHOOK, json=payload)
            except Exception as e:
                logger.error(f"Discord error: {e}")
                
    async def trading_loop(self):
        """Main trading loop."""
        self.running = True
        
        async with aiohttp.ClientSession() as session:
            while self.running:
                try:
                    for symbol in self.symbols:
                        # Get ticker
                        ticker = await self.get_ticker(session, symbol)
                        if not ticker:
                            continue
                            
                        # Generate features
                        features = self.feature_generator.generate_features(ticker, symbol)
                        normalized = self.feature_generator.normalize_features(features)
                        
                        # Get prediction
                        prediction = self.predict(normalized)
                        
                        # Calculate confidence
                        confidence = abs(prediction - 0.5) * 2
                        direction = "BUY" if prediction > 0.5 else "SELL"
                        price = float(ticker.get("lastPrice", 0))
                        change_24h = float(ticker.get("price24hPcnt", 0)) * 100
                        
                        logger.info(
                            f"📊 {symbol}: pred={prediction:.4f}, "
                            f"conf={confidence*100:.2f}%, dir={direction}, "
                            f"price=${price:,.2f}"
                        )
                        
                        # Check if we should send signal
                        now = datetime.now()
                        last_signal = self.last_signal_time.get(symbol)
                        
                        if confidence >= self.min_confidence:
                            if not last_signal or (now - last_signal).total_seconds() >= self.signal_cooldown:
                                self.signal_count += 1
                                self.last_signal_time[symbol] = now
                                
                                # Get feature info
                                hist_data = self.feature_generator.historical_data.get(symbol)
                                features_info = ""
                                if hist_data is not None and len(hist_data) > 0:
                                    latest_vol = hist_data['returns'].tail(20).std() * 100
                                    latest_rsi = self.feature_generator.calculate_rsi(hist_data['close'], 14)
                                    features_info = (
                                        f"**20-Day Volatility:** {latest_vol:.2f}%\n"
                                        f"**RSI(14):** {latest_rsi:.1f}\n"
                                    )
                                
                                await self.send_discord(
                                    f"🎯 ML Signal #{self.signal_count} - {symbol}",
                                    (
                                        f"**Direction:** {direction}\n"
                                        f"**Confidence:** {confidence*100:.1f}%\n"
                                        f"**ML Score:** {prediction:.4f}\n"
                                        f"**Price:** ${price:,.2f}\n"
                                        f"**24h Change:** {change_24h:+.2f}%\n"
                                        f"{features_info}"
                                        f"**Time:** {now.strftime('%H:%M:%S')}\n"
                                        f"*Using real historical data (improved)*"
                                    ),
                                    "00FF00" if direction == "BUY" else "FF0000"
                                )
                                
                                logger.info(f"Signal sent for {symbol}")
                    
                    await asyncio.sleep(10)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    await asyncio.sleep(30)
                    
    async def run(self):
        """Run the bot."""
        logger.info("Initializing Simple Improved ML Bot...")
        
        # Load model
        self.load_model()
        
        # Load historical data
        logger.info("Loading historical data...")
        for symbol in self.symbols:
            self.feature_generator.update_historical_cache(symbol)
            logger.info(f"Loaded historical data for {symbol}")
            
        # Send startup notification
        await self.send_discord(
            "🤖 ML Trading Bot Started (Improved + Fixed)",
            (
                f"**System:** Simple Improved ML Bot\n"
                f"**Model:** v3.1_improved (Fixed ONNX output)\n"
                f"**Symbols:** {', '.join(self.symbols)}\n"
                f"**Features:** Using real historical data\n"
                f"**Min Confidence:** {self.min_confidence * 100:.0f}%\n"
                f"**Status:** Online"
            ),
            "00FF00"
        )
        
        logger.info("Bot initialized successfully")
        
        # Start trading loop
        await self.trading_loop()

async def main():
    bot = SimpleImprovedBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())