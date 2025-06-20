#!/usr/bin/env python3
"""
Debug version of ML bot to check feature generation and predictions
"""

import asyncio
import os
import sys
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))

# Import components
from src.common.logging import setup_logging, get_logger
from src.common.bybit_client import BybitRESTClient
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from ml_feature_generator import MLFeatureGenerator

logger = get_logger(__name__)

async def debug_features():
    """Debug feature generation and predictions."""
    # Initialize Bybit client
    bybit_client = BybitRESTClient(testnet=False)
    await bybit_client.__aenter__()
    
    # Initialize ML inference engine
    inference_config = InferenceConfig(
        model_path="models/v3.1_improved/model.onnx",
        enable_batching=False,
        enable_thompson_sampling=False,
        confidence_threshold=0.65
    )
    inference_engine = InferenceEngine(inference_config)
    inference_engine.load_model()
    
    # Test for each symbol
    symbols = ["BTCUSDT", "ETHUSDT", "ICPUSDT"]
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Testing {symbol}")
        print('='*60)
        
        # Get ticker data
        ticker = await bybit_client.get_ticker(symbol)
        print(f"\nTicker data:")
        print(f"  Price: ${float(ticker.get('lastPrice', 0)):,.2f}")
        print(f"  24h Change: {float(ticker.get('price24hPcnt', 0))*100:.2f}%")
        print(f"  Volume: {float(ticker.get('volume24h', 0)):,.0f}")
        
        # Generate features
        feature_gen = MLFeatureGenerator()
        features = feature_gen.generate_features(ticker)
        
        print(f"\nGenerated features (first 10):")
        for i, (name, value) in enumerate(list(features.items())[:10]):
            print(f"  {name}: {value:.6f}")
        
        # Normalize features
        normalized = feature_gen.normalize_features(features)
        print(f"\nNormalized features shape: {normalized.shape}")
        print(f"Normalized features (first 10): {normalized[:10]}")
        print(f"Min: {normalized.min():.4f}, Max: {normalized.max():.4f}, Mean: {normalized.mean():.4f}")
        
        # Get prediction
        result = inference_engine.predict(normalized.reshape(1, -1))
        prediction = float(result["predictions"][0])
        raw_prediction = float(result["raw_predictions"][0])
        
        print(f"\nPrediction results:")
        print(f"  Raw prediction: {raw_prediction:.6f}")
        print(f"  Risk-adjusted prediction: {prediction:.6f}")
        print(f"  Confidence score: {result.get('confidence_scores', [0])[0]:.4f}")
        print(f"  Inference time: {result['inference_time_ms']:.2f}ms")
        
        # Test with different feature values
        print(f"\nTesting with synthetic features...")
        
        # Bullish scenario
        bullish_features = features.copy()
        bullish_features["returns"] = 0.02
        bullish_features["momentum_3"] = 0.015
        bullish_features["rsi_14"] = 70
        bullish_normalized = feature_gen.normalize_features(bullish_features)
        bullish_result = inference_engine.predict(bullish_normalized.reshape(1, -1))
        print(f"  Bullish scenario prediction: {float(bullish_result['predictions'][0]):.6f}")
        
        # Bearish scenario
        bearish_features = features.copy()
        bearish_features["returns"] = -0.02
        bearish_features["momentum_3"] = -0.015
        bearish_features["rsi_14"] = 30
        bearish_normalized = feature_gen.normalize_features(bearish_features)
        bearish_result = inference_engine.predict(bearish_normalized.reshape(1, -1))
        print(f"  Bearish scenario prediction: {float(bearish_result['predictions'][0]):.6f}")
    
    # Test direct model inference
    print(f"\n{'='*60}")
    print("Testing direct model inference")
    print('='*60)
    
    # Create random features
    random_features = np.random.randn(1, 44).astype(np.float32)
    print(f"Random features shape: {random_features.shape}")
    
    direct_result = inference_engine.onnx_session.run(
        [inference_engine.output_name],
        {inference_engine.input_name: random_features}
    )
    print(f"Direct ONNX output: {direct_result[0]}")
    
    # Clean up
    await bybit_client.__aexit__(None, None, None)

async def main():
    """Main entry point."""
    setup_logging()
    await debug_features()

if __name__ == "__main__":
    asyncio.run(main())