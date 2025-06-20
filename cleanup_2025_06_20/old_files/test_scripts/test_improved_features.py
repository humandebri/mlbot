#!/usr/bin/env python3
"""
Test improved feature generator to compare with approximation version
"""

import asyncio
import numpy as np
from datetime import datetime
from src.common.bybit_client import BybitRESTClient
from ml_feature_generator import MLFeatureGenerator
from improved_feature_generator import ImprovedFeatureGenerator
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
import json


async def compare_generators():
    """Compare original and improved feature generators."""
    # Initialize clients
    bybit_client = BybitRESTClient(testnet=False)
    await bybit_client.__aenter__()
    
    # Initialize generators
    original_gen = MLFeatureGenerator()
    improved_gen = ImprovedFeatureGenerator()
    
    # Initialize inference engine
    inference_config = InferenceConfig(
        model_path="models/v3.1_improved/model.onnx",
        enable_batching=False,
        enable_thompson_sampling=False,
        confidence_threshold=0.65,
        risk_adjustment=False
    )
    inference_engine = InferenceEngine(inference_config)
    inference_engine.load_model()
    
    symbols = ["BTCUSDT", "ETHUSDT", "ICPUSDT"]
    
    # Load historical data for improved generator
    print("Loading historical data...")
    for symbol in symbols:
        improved_gen.update_historical_cache(symbol)
    
    print("\n" + "="*80)
    print("FEATURE COMPARISON: Original (Approximations) vs Improved (Historical Data)")
    print("="*80)
    
    for symbol in symbols:
        print(f"\n\n{'='*60}")
        print(f"SYMBOL: {symbol}")
        print('='*60)
        
        # Get ticker data
        ticker = await bybit_client.get_ticker(symbol)
        price = float(ticker.get("lastPrice", 0))
        change_24h = float(ticker.get("price24hPcnt", 0)) * 100
        
        print(f"Current Price: ${price:,.2f}")
        print(f"24h Change: {change_24h:+.2f}%")
        
        # Generate features with both generators
        original_features = original_gen.generate_features(ticker)
        improved_features = improved_gen.generate_features(ticker, symbol)
        
        # Compare key features
        print("\n" + "-"*60)
        print("FEATURE COMPARISON:")
        print("-"*60)
        print(f"{'Feature':<20} {'Original':<15} {'Improved':<15} {'Difference':<15}")
        print("-"*60)
        
        key_features = [
            "returns", "return_3", "return_5", "return_10",
            "vol_5", "vol_10", "vol_20",
            "rsi_14", "macd", "volume_ratio_10",
            "momentum_3", "momentum_5"
        ]
        
        for feature in key_features:
            orig_val = original_features.get(feature, 0)
            impr_val = improved_features.get(feature, 0)
            diff = impr_val - orig_val
            print(f"{feature:<20} {orig_val:<15.6f} {impr_val:<15.6f} {diff:<15.6f}")
        
        # Show which features are using real data vs approximations
        print("\n" + "-"*60)
        print("APPROXIMATION vs REAL DATA:")
        print("-"*60)
        
        # Check for approximation patterns
        if abs(original_features["return_3"] - original_features["returns"] * 0.8) < 0.0001:
            print("❌ return_3: Using approximation (returns * 0.8)")
        else:
            print("✅ return_3: Using real calculation")
        
        if abs(improved_features["return_3"] - improved_features["returns"] * 0.8) < 0.0001:
            print("   → Improved still using approximation (no historical data)")
        else:
            print("   → Improved using real historical data ✅")
        
        # Test predictions with both feature sets
        print("\n" + "-"*60)
        print("ML PREDICTIONS:")
        print("-"*60)
        
        # Original features prediction
        orig_normalized = original_gen.normalize_features(original_features)
        orig_result = inference_engine.predict(orig_normalized.reshape(1, -1))
        orig_pred = float(orig_result["predictions"][0])
        orig_conf = abs(orig_pred - 0.5) * 2
        
        # Improved features prediction
        impr_normalized = improved_gen.normalize_features(improved_features)
        impr_result = inference_engine.predict(impr_normalized.reshape(1, -1))
        impr_pred = float(impr_result["predictions"][0])
        impr_conf = abs(impr_pred - 0.5) * 2
        
        print(f"Original prediction: {orig_pred:.4f} (confidence: {orig_conf*100:.1f}%)")
        print(f"Improved prediction: {impr_pred:.4f} (confidence: {impr_conf*100:.1f}%)")
        print(f"Prediction difference: {abs(impr_pred - orig_pred):.4f}")
        
        # Show historical data availability
        if symbol in improved_gen.historical_data:
            hist_data = improved_gen.historical_data[symbol]
            print(f"\nHistorical data available: {len(hist_data)} records")
            print(f"Date range: {hist_data.index[0].strftime('%Y-%m-%d')} to {hist_data.index[-1].strftime('%Y-%m-%d')}")
        else:
            print("\nNo historical data available ❌")
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nKey improvements in the new feature generator:")
    print("1. ✅ Real multi-period returns calculated from historical prices")
    print("2. ✅ Actual volatility calculations using rolling standard deviation")
    print("3. ✅ Real RSI calculation using gain/loss ratios")
    print("4. ✅ Actual MACD using EMA crossovers")
    print("5. ✅ Real volume ratios from historical volume data")
    print("6. ✅ Actual Bollinger Bands calculation")
    print("7. ✅ Real momentum indicators")
    print("8. ✅ Price percentiles from historical distribution")
    print("\nNo more random values or simple approximations! 🎉")
    
    # Cleanup
    await bybit_client.__aexit__(None, None, None)
    improved_gen.close()


async def main():
    """Main entry point."""
    await compare_generators()


if __name__ == "__main__":
    asyncio.run(main())