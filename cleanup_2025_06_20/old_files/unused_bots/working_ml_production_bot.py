#!/usr/bin/env python3
"""
Working ML Production Bot - Simplified but functional ML trading system
"""

import asyncio
import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))

# Import required components
from src.common.logging import setup_logging, get_logger
from src.common.database import get_redis_client
from src.common.discord_notifier import discord_notifier
from src.common.bybit_client import BybitRESTClient
from src.common.config import settings
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from ml_feature_generator import MLFeatureGenerator

logger = get_logger(__name__)


class WorkingMLProductionBot:
    """Simplified ML production bot that actually works."""
    
    def __init__(self):
        self.redis_client = None
        self.bybit_client = None
        self.inference_engine = None
        self.running = False
        
        # Tracking
        self.last_signal_time = {}
        self.signal_cooldown = 300  # 5 minutes
        self.min_confidence = 0.65  # 65% (lower for testing)
        self.signal_count = 0
        self.prediction_count = 0
        self.last_hourly_report = datetime.now()
        self.prediction_history = []
        
        # Feature generators for each symbol
        self.feature_generators = {}
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Working ML Production Bot...")
        
        try:
            # Initialize Redis
            self.redis_client = await get_redis_client()
            
            # Initialize Bybit client
            self.bybit_client = BybitRESTClient(testnet=False)
            await self.bybit_client.__aenter__()
            
            # Initialize ML inference engine with proper config
            inference_config = InferenceConfig(
                model_path=settings.model.model_path,
                enable_batching=False,
                enable_thompson_sampling=False,  # Disable Thompson Sampling
                confidence_threshold=0.65,
                risk_adjustment=False  # Disable broken risk adjustment
            )
            self.inference_engine = InferenceEngine(inference_config)
            
            # Load the model
            self.inference_engine.load_model()
            
            # Initialize feature generators for each symbol
            for symbol in settings.bybit.symbols:
                self.feature_generators[symbol] = MLFeatureGenerator()
            
            # Send startup notification
            discord_notifier.send_notification(
                title="🤖 ML Trading Bot Started",
                description=(
                    f"**System:** Working ML Production Bot\n"
                    f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"**Symbols:** {', '.join(settings.bybit.symbols)}\n"
                    f"**ML Model:** v3.1_improved\n"
                    f"**Min Confidence:** {self.min_confidence * 100:.0f}%\n"
                    f"**Status:** Online and predicting"
                ),
                color="00FF00"
            )
            
            logger.info("Bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            raise
    
    async def get_ml_features(self, symbol: str) -> Optional[np.ndarray]:
        """Get ML features for a symbol using the correct feature generator."""
        try:
            # Get ticker data
            ticker = await self.bybit_client.get_ticker(symbol)
            if not ticker:
                return None
            
            # Get feature generator for this symbol
            feature_gen = self.feature_generators.get(symbol)
            if not feature_gen:
                logger.error(f"No feature generator for {symbol}")
                return None
            
            # Generate features using the ML feature generator
            features = feature_gen.generate_features(ticker)
            
            # Normalize features
            normalized_features = feature_gen.normalize_features(features)
            
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error getting ML features for {symbol}: {e}")
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
        check_interval = 10  # Check every 10 seconds
        
        while self.running:
            try:
                for symbol in settings.bybit.symbols:
                    # Get normalized ML features
                    feature_array = await self.get_ml_features(symbol)
                    if feature_array is None:
                        logger.warning(f"No features available for {symbol}")
                        continue
                    
                    # Get current price from ticker
                    ticker = await self.bybit_client.get_ticker(symbol)
                    current_price = float(ticker.get("lastPrice", 0)) if ticker else 0
                    
                    # Get ML prediction
                    try:
                        # Use the correct predict method
                        result = self.inference_engine.predict(feature_array.reshape(1, -1))
                        # Extract prediction from result dictionary
                        prediction = float(result["predictions"][0])
                        self.prediction_count += 1
                        
                        # Calculate confidence
                        confidence = abs(prediction - 0.5) * 2
                        confidence = min(max(confidence, 0.0), 1.0)
                        direction = "BUY" if prediction > 0.5 else "SELL"
                        
                        # Store prediction
                        self.prediction_history.append({
                            "symbol": symbol,
                            "prediction": prediction,
                            "confidence": confidence,
                            "direction": direction,
                            "price": current_price,
                            "timestamp": datetime.now()
                        })
                        
                        # Keep only last 1000 predictions
                        if len(self.prediction_history) > 1000:
                            self.prediction_history = self.prediction_history[-1000:]
                        
                        logger.info(
                            f"📊 {symbol}: pred={prediction:.4f}, "
                            f"conf={confidence*100:.2f}%, dir={direction}, "
                            f"price=${current_price:,.2f}"
                        )
                        
                        # Check if we should send a signal
                        if confidence >= self.min_confidence and await self.should_send_signal(symbol):
                            self.signal_count += 1
                            self.last_signal_time[symbol] = datetime.now()
                            
                            # Send signal
                            await self._send_trading_signal(
                                symbol=symbol,
                                direction=direction,
                                confidence=confidence,
                                price=current_price,
                                prediction=prediction
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing prediction for {symbol}: {e}")
                
                # Check if hourly report is due
                if (datetime.now() - self.last_hourly_report).total_seconds() >= 3600:
                    await self.send_hourly_report()
                    self.last_hourly_report = datetime.now()
                
                # Wait before next iteration
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)
    
    async def _send_trading_signal(self, symbol: str, direction: str, confidence: float, 
                                  price: float, prediction: float):
        """Send trading signal via Discord."""
        try:
            # Get 24h change
            ticker = await self.bybit_client.get_ticker(symbol)
            change_24h = float(ticker.get("price24hPcnt", 0)) * 100 if ticker else 0
            
            discord_notifier.send_notification(
                title=f"🎯 ML Signal #{self.signal_count} - {symbol}",
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
            
            logger.info(f"🚨 ML Signal #{self.signal_count} sent for {symbol}")
            
        except Exception as e:
            logger.error(f"Error sending trading signal: {e}")
    
    async def send_hourly_report(self):
        """Send hourly summary report."""
        try:
            # Calculate statistics from history
            recent_preds = [p for p in self.prediction_history 
                          if (datetime.now() - p["timestamp"]).total_seconds() < 3600]
            
            # Build report
            report_lines = [
                "**📊 ML Trading Report**",
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
                "**System Performance:**",
                f"• Total Predictions: {self.prediction_count}",
                f"• Signals Sent: {self.signal_count}",
                f"• Recent Predictions: {len(recent_preds)}",
                ""
            ]
            
            # Symbol statistics
            if recent_preds:
                for symbol in settings.bybit.symbols:
                    symbol_preds = [p for p in recent_preds if p["symbol"] == symbol]
                    if symbol_preds:
                        avg_conf = np.mean([p["confidence"] for p in symbol_preds]) * 100
                        buy_count = sum(1 for p in symbol_preds if p["direction"] == "BUY")
                        sell_count = len(symbol_preds) - buy_count
                        latest_price = symbol_preds[-1]["price"]
                        
                        report_lines.append(f"**{symbol}:**")
                        report_lines.append(f"  Price: ${latest_price:,.2f}")
                        report_lines.append(f"  Predictions: {len(symbol_preds)}")
                        report_lines.append(f"  Buy/Sell: {buy_count}/{sell_count}")
                        report_lines.append(f"  Avg Confidence: {avg_conf:.1f}%")
                        report_lines.append("")
            
            discord_notifier.send_notification(
                title="Hourly ML Report",
                description="\n".join(report_lines),
                color="3498db"
            )
            
        except Exception as e:
            logger.error(f"Error sending hourly report: {e}")
    
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
            description=f"ML Bot stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            color="FFA500"
        )


async def main():
    """Main entry point."""
    setup_logging()
    bot = WorkingMLProductionBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())