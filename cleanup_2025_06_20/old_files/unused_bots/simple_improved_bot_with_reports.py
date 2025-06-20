#!/usr/bin/env python3
"""
Simple Improved ML Bot with hourly reports and adjusted confidence threshold
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
        self.min_confidence = 0.45  # Lowered from 0.65 to 0.45 (45%)
        self.signal_count = 0
        self.prediction_count = 0
        self.prediction_history = []
        self.last_hourly_report = datetime.now()
        
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
    
    async def send_hourly_report(self):
        """Send hourly summary report."""
        try:
            # Calculate statistics from history
            recent_preds = [p for p in self.prediction_history 
                          if (datetime.now() - p["timestamp"]).total_seconds() < 3600]
            
            # Build report
            report_lines = [
                "**📊 ML Trading Report (Improved)**",
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
                "**System Performance:**",
                f"• Total Predictions: {self.prediction_count}",
                f"• Signals Sent: {self.signal_count}",
                f"• Recent Predictions (1h): {len(recent_preds)}",
                f"• Min Confidence: {self.min_confidence * 100:.0f}%",
                f"• Feature Generator: Using real historical data",
                ""
            ]
            
            # Symbol statistics
            if recent_preds:
                for symbol in self.symbols:
                    symbol_preds = [p for p in recent_preds if p["symbol"] == symbol]
                    if symbol_preds:
                        avg_pred = np.mean([p["prediction"] for p in symbol_preds])
                        avg_conf = np.mean([p["confidence"] for p in symbol_preds]) * 100
                        buy_count = sum(1 for p in symbol_preds if p["direction"] == "BUY")
                        sell_count = len(symbol_preds) - buy_count
                        latest_price = symbol_preds[-1]["price"]
                        
                        # Add historical data info
                        hist_info = ""
                        if symbol in self.feature_generator.historical_data:
                            hist_len = len(self.feature_generator.historical_data[symbol])
                            hist_info = f" ({hist_len:,} historical records)"
                        
                        report_lines.append(f"**{symbol}:**")
                        report_lines.append(f"  Price: ${latest_price:,.2f}")
                        report_lines.append(f"  Avg Prediction: {avg_pred:.4f}")
                        report_lines.append(f"  Avg Confidence: {avg_conf:.1f}%")
                        report_lines.append(f"  Buy/Sell: {buy_count}/{sell_count}")
                        report_lines.append(f"  Historical Data{hist_info}")
                        report_lines.append("")
            
            await self.send_discord(
                "📈 Hourly ML Report",
                "\n".join(report_lines),
                "3498db"
            )
            
            logger.info("Hourly report sent")
            
        except Exception as e:
            logger.error(f"Error sending hourly report: {e}")
                
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
                        self.prediction_count += 1
                        
                        # Calculate confidence
                        confidence = abs(prediction - 0.5) * 2
                        direction = "BUY" if prediction > 0.5 else "SELL"
                        price = float(ticker.get("lastPrice", 0))
                        change_24h = float(ticker.get("price24hPcnt", 0)) * 100
                        
                        # Store prediction
                        self.prediction_history.append({
                            "symbol": symbol,
                            "prediction": prediction,
                            "confidence": confidence,
                            "direction": direction,
                            "price": price,
                            "timestamp": datetime.now()
                        })
                        
                        # Keep only last 2000 predictions
                        if len(self.prediction_history) > 2000:
                            self.prediction_history = self.prediction_history[-2000:]
                        
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
                    
                    # Check if hourly report is due
                    if (datetime.now() - self.last_hourly_report).total_seconds() >= 3600:
                        await self.send_hourly_report()
                        self.last_hourly_report = datetime.now()
                    
                    await asyncio.sleep(10)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    await asyncio.sleep(30)
                    
    async def run(self):
        """Run the bot."""
        logger.info("Initializing Simple Improved ML Bot with Reports...")
        
        # Load model
        self.load_model()
        
        # Load historical data
        logger.info("Loading historical data...")
        for symbol in self.symbols:
            self.feature_generator.update_historical_cache(symbol)
            logger.info(f"Loaded historical data for {symbol}")
            
        # Send startup notification
        await self.send_discord(
            "🤖 ML Trading Bot Started (Improved + Reports)",
            (
                f"**System:** Simple Improved ML Bot with Reports\n"
                f"**Model:** v3.1_improved (Fixed ONNX output)\n"
                f"**Symbols:** {', '.join(self.symbols)}\n"
                f"**Features:** Using real historical data\n"
                f"**Min Confidence:** {self.min_confidence * 100:.0f}% (Adjusted)\n"
                f"**Hourly Reports:** Enabled\n"
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