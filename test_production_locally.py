#!/usr/bin/env python3
"""Test the production system locally to debug issues."""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))

load_dotenv()

async def test_system():
    """Test key components."""
    
    # Test 1: Check model predictions
    print("=" * 50)
    print("Testing model predictions...")
    
    from src.ml_pipeline.inference_engine import InferenceEngine
    from src.common.config import settings
    
    engine = InferenceEngine(settings)
    
    # Create test features (44 features)
    test_features = {
        'open': 100000.0,
        'high': 101000.0,
        'low': 99000.0,
        'close': 100500.0,
        'volume': 1000.0,
        'spread_bps': 1.5,
        'bid_size': 100.0,
        'ask_size': 120.0,
        'imbalance': -0.1,
        'microprice': 100500.0,
        'rsi': 55.0,
        'volatility': 0.02,
        'volume_imbalance': 0.1,
        'trade_count': 100,
        'buy_ratio': 0.55,
        'large_trade_ratio': 0.2,
        'liquidation_volume': 0.0,
        'liquidation_ratio': 0.0,
        'long_liquidations': 0,
        'short_liquidations': 0,
        'funding_rate': 0.0001,
        'open_interest': 1000000.0,
        'oi_change': 0.01,
        'price_change_1m': 0.001,
        'price_change_5m': 0.002,
        'volume_1m': 100.0,
        'volume_5m': 500.0,
        'trade_count_1m': 20,
        'trade_count_5m': 100,
        'hour': 10,
        'minute': 30,
        'day_of_week': 3,
        'is_asian_session': 0,
        'is_european_session': 1,
        'is_us_session': 0,
        'seconds_to_funding': 3600,
        'price_vol_corr': 0.1,
        'volume_momentum': 1.2,
        'rsi_divergence': 0.0,
        'spread_momentum': -0.1,
        'liquidation_momentum': 0.0,
        'microstructure_intensity': 0.5,
        'time_weighted_pressure': 0.1,
        'information_ratio': 1.5
    }
    
    try:
        prediction = await engine.predict_single(test_features)
        print(f"Prediction value: {prediction}")
        
        # Extract confidence
        import numpy as np
        confidence = abs(prediction - 0.5) * 2
        confidence = min(max(confidence, 0.0), 1.0) * 100
        
        print(f"Confidence: {confidence:.2f}%")
        print(f"Direction: {'BUY' if prediction > 0.5 else 'SELL'}")
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Check Discord notifier
    print("\n" + "=" * 50)
    print("Testing Discord notifier...")
    
    from src.common.discord_notifier import discord_notifier
    
    webhook = os.getenv("DISCORD_WEBHOOK")
    print(f"Discord webhook configured: {'Yes' if webhook else 'No'}")
    
    if webhook:
        print(f"Webhook URL length: {len(webhook)}")
        print(f"Webhook starts with: {webhook[:50]}...")
        
        # Test sending a message
        try:
            await discord_notifier.send_notification(
                title="Test Message from MLBot",
                message="Testing Discord notifications. If you see this, notifications are working!",
                color=0x00FF00  # Green
            )
            print("Discord test message sent successfully!")
        except Exception as e:
            print(f"Error sending Discord message: {e}")

if __name__ == "__main__":
    asyncio.run(test_system())