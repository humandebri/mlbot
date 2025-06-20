#!/usr/bin/env python3
"""
Fix the production bot to properly send Discord notifications and handle predictions.
"""

import asyncio
import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))

from src.common.logging import setup_logging, get_logger
from src.common.database import RedisManager
from src.common.discord_notifier import discord_notifier
from src.common.bybit_client import BybitRESTClient
from src.ml_pipeline.pytorch_inference_engine import PyTorchInferenceEngine
from src.common.config import settings

load_dotenv()
logger = get_logger(__name__)

class FixedProductionBot:
    """Fixed version of the production bot with proper Discord notifications."""
    
    def __init__(self):
        self.redis_manager = RedisManager()
        self.bybit_client = None
        self.inference_engine = None
        self.running = False
        
        # Tracking
        self.last_signal_time = {}
        self.signal_cooldown = 300  # 5 minutes between signals per symbol
        self.min_confidence = 0.7  # 70% minimum confidence
        self.signal_count = 0
        self.last_hourly_report = datetime.now()
        self.predictions_history = []
        
        # Manual scaler parameters for 44 features
        self.scaler_mean = [
            100000, 100000, 100000, 100000, 1000,  # OHLCV
            2.0, 100, 100, 0, 100000,  # Spread, bid/ask, imbalance, microprice
            50, 0.02, 0, 100, 0.5,  # RSI, volatility, vol_imbalance, trade_count, buy_ratio
            0.1, 0, 0, 0, 0,  # Large trades, liquidations
            0.0001, 1000000, 0, 0, 0,  # Funding, OI, price changes
            100, 500, 20, 100, 12,  # Volumes, trade counts, time
            30, 3, 0, 0, 0,  # Time features
            3600, 0, 1, 0, 0,  # Seconds to funding, correlations
            0, 0.5, 0, 1  # Momentum features
        ]
        
        self.scaler_std = [
            1000, 1000, 1000, 1000, 100,  # OHLCV
            1.0, 50, 50, 0.2, 1000,  # Spread, bid/ask, imbalance, microprice
            20, 0.01, 0.2, 50, 0.2,  # RSI, volatility, vol_imbalance, trade_count, buy_ratio
            0.1, 10, 0.1, 5, 5,  # Large trades, liquidations
            0.0002, 100000, 0.02, 0.01, 0.01,  # Funding, OI, price changes
            50, 250, 10, 50, 6,  # Volumes, trade counts, time
            15, 2, 0.5, 0.5, 0.5,  # Time features
            1800, 0.2, 0.5, 0.1, 0.1,  # Seconds to funding, correlations
            10, 0.2, 0.1, 0.5  # Momentum features
        ]
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Fixed Production Bot...")
        
        # Initialize Redis
        await self.redis_manager.connect()
        
        # Initialize Bybit client
        self.bybit_client = BybitRESTClient(testnet=False)
        await self.bybit_client.__aenter__()
        
        # Initialize inference engine
        self.inference_engine = PyTorchInferenceEngine(settings)
        
        # Send startup notification
        await discord_notifier.send_notification(
            title="🤖 MLBot Production System Started",
            message=(
                f"**System:** Fixed Production Bot\n"
                f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"**Symbols:** {', '.join(settings.bybit.symbols)}\n"
                f"**Min Confidence:** {self.min_confidence * 100:.0f}%\n"
                f"**Signal Cooldown:** {self.signal_cooldown}s"
            ),
            color=0x00FF00
        )
        
        logger.info("Initialization complete")
    
    def normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        """Normalize features to 44 dimensions."""
        # Expected feature names in order
        feature_names = [
            'open', 'high', 'low', 'close', 'volume',
            'spread_bps', 'bid_size', 'ask_size', 'imbalance', 'microprice',
            'rsi', 'volatility', 'volume_imbalance', 'trade_count', 'buy_ratio',
            'large_trade_ratio', 'liquidation_volume', 'liquidation_ratio', 
            'long_liquidations', 'short_liquidations',
            'funding_rate', 'open_interest', 'oi_change', 'price_change_1m', 'price_change_5m',
            'volume_1m', 'volume_5m', 'trade_count_1m', 'trade_count_5m', 'hour',
            'minute', 'day_of_week', 'is_asian_session', 'is_european_session', 'is_us_session',
            'seconds_to_funding', 'price_vol_corr', 'volume_momentum', 'rsi_divergence', 
            'spread_momentum', 'liquidation_momentum', 'microstructure_intensity', 
            'time_weighted_pressure', 'information_ratio'
        ]
        
        # Extract features in order
        feature_vector = []
        for name in feature_names:
            value = features.get(name, 0.0)
            feature_vector.append(value)
        
        # Convert to numpy array
        feature_array = np.array(feature_vector, dtype=np.float32)
        
        # Normalize
        normalized = (feature_array - np.array(self.scaler_mean)) / np.array(self.scaler_std)
        
        # Clip to reasonable range
        normalized = np.clip(normalized, -5, 5)
        
        return normalized
    
    async def get_features_for_symbol(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get latest features from Redis for a symbol."""
        try:
            # Get latest features from Redis
            latest_entries = await self.redis_manager.xrevrange(
                f"features:{symbol}",
                count=1
            )
            
            if not latest_entries:
                return None
            
            _, fields = latest_entries[0]
            features_str = fields.get("features", fields.get(b"features", ""))
            
            if isinstance(features_str, bytes):
                features_str = features_str.decode()
            
            if not features_str:
                return None
            
            features = json.loads(features_str)
            return features
            
        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {e}")
            return None
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price from Bybit."""
        try:
            ticker = await self.bybit_client.get_ticker(symbol)
            if ticker:
                return float(ticker.get("lastPrice", 0))
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
        return 0.0
    
    async def should_send_signal(self, symbol: str) -> bool:
        """Check if we should send a signal for this symbol."""
        now = datetime.now()
        
        if symbol not in self.last_signal_time:
            return True
        
        time_since_last = (now - self.last_signal_time[symbol]).total_seconds()
        return time_since_last >= self.signal_cooldown
    
    async def trading_loop(self):
        """Main trading loop."""
        self.running = True
        
        while self.running:
            try:
                for symbol in settings.bybit.symbols:
                    # Get features
                    features = await self.get_features_for_symbol(symbol)
                    if not features:
                        logger.warning(f"No features available for {symbol}")
                        continue
                    
                    # Get current price
                    current_price = await self.get_current_price(symbol)
                    if current_price == 0:
                        logger.warning(f"No price available for {symbol}")
                        continue
                    
                    # Normalize features
                    normalized_features = self.normalize_features(features)
                    
                    # Get prediction
                    try:
                        prediction = await self.inference_engine.predict_single_numpy(normalized_features)
                        
                        # Calculate confidence (0.5 = 50%, 1.0 = 100% buy, 0.0 = 100% sell)
                        confidence = abs(prediction - 0.5) * 2
                        direction = "BUY" if prediction > 0.5 else "SELL"
                        
                        # Store for reporting
                        self.predictions_history.append({
                            "symbol": symbol,
                            "prediction": prediction,
                            "confidence": confidence,
                            "direction": direction,
                            "price": current_price,
                            "timestamp": datetime.now()
                        })
                        
                        # Keep only last 1000 predictions
                        if len(self.predictions_history) > 1000:
                            self.predictions_history = self.predictions_history[-1000:]
                        
                        logger.info(
                            f"📊 {symbol}: {len(features)}特徴量, "
                            f"pred={prediction:.4f}, conf={confidence*100:.2f}%, "
                            f"価格=${current_price:,.2f}"
                        )
                        
                        # Check if we should send a signal
                        if confidence >= self.min_confidence and await self.should_send_signal(symbol):
                            self.signal_count += 1
                            self.last_signal_time[symbol] = datetime.now()
                            
                            # Send Discord notification
                            await discord_notifier.send_trade_signal(
                                symbol=symbol,
                                direction=direction,
                                confidence=confidence * 100,
                                expected_pnl=0.1,  # Placeholder
                                metadata={
                                    "prediction": prediction,
                                    "price": current_price,
                                    "features_count": len(features),
                                    "signal_number": self.signal_count
                                }
                            )
                            
                            logger.info(f"🚨 SIGNAL #{self.signal_count} sent for {symbol}")
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                # Check if hourly report is due
                if (datetime.now() - self.last_hourly_report).total_seconds() >= 3600:
                    await self.send_hourly_report()
                    self.last_hourly_report = datetime.now()
                
                # Wait before next iteration
                await asyncio.sleep(10)  # Check every 10 seconds instead of 3
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)
    
    async def send_hourly_report(self):
        """Send hourly summary report."""
        try:
            # Calculate statistics from history
            if not self.predictions_history:
                return
            
            recent_preds = [p for p in self.predictions_history 
                          if (datetime.now() - p["timestamp"]).total_seconds() < 3600]
            
            if not recent_preds:
                return
            
            # Group by symbol
            symbol_stats = {}
            for pred in recent_preds:
                symbol = pred["symbol"]
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        "predictions": [],
                        "buy_count": 0,
                        "sell_count": 0,
                        "avg_confidence": 0,
                        "current_price": pred["price"]
                    }
                
                symbol_stats[symbol]["predictions"].append(pred)
                if pred["direction"] == "BUY":
                    symbol_stats[symbol]["buy_count"] += 1
                else:
                    symbol_stats[symbol]["sell_count"] += 1
            
            # Build report
            report_lines = []
            report_lines.append(f"**📊 Hourly Trading Report**")
            report_lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Total Predictions: {len(recent_preds)}")
            report_lines.append(f"Signals Sent: {self.signal_count}")
            report_lines.append("")
            
            for symbol, stats in symbol_stats.items():
                avg_conf = np.mean([p["confidence"] for p in stats["predictions"]]) * 100
                report_lines.append(f"**{symbol}**")
                report_lines.append(f"  Current Price: ${stats['current_price']:,.2f}")
                report_lines.append(f"  Predictions: {len(stats['predictions'])}")
                report_lines.append(f"  Buy/Sell: {stats['buy_count']}/{stats['sell_count']}")
                report_lines.append(f"  Avg Confidence: {avg_conf:.1f}%")
                report_lines.append("")
            
            await discord_notifier.send_notification(
                title="Hourly System Report",
                message="\n".join(report_lines),
                color=0x3498db
            )
            
        except Exception as e:
            logger.error(f"Error sending hourly report: {e}")
    
    async def run(self):
        """Run the bot."""
        try:
            await self.initialize()
            await self.trading_loop()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            await discord_notifier.send_notification(
                title="❌ Bot Error",
                message=f"Fatal error occurred: {str(e)}",
                color=0xFF0000
            )
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.running = False
        
        if self.bybit_client:
            await self.bybit_client.__aexit__(None, None, None)
        
        if self.redis_manager:
            await self.redis_manager.close()
        
        await discord_notifier.send_notification(
            title="🛑 Bot Stopped",
            message=f"Bot stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            color=0xFFA500
        )

async def main():
    """Main entry point."""
    setup_logging()
    bot = FixedProductionBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())