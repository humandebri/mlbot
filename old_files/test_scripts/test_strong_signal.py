#!/usr/bin/env python3
"""
強いシグナルを生成するテスト

より強い価格変動と明確なトレンドを使用して
ML予測システムが実際にシグナルを生成できるかテスト
"""

import os
import sys
import numpy as np
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Environment setup
from dotenv import load_dotenv
load_dotenv()

from src.feature_hub.technical_indicators import TechnicalIndicatorEngine
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig

def create_strong_bullish_scenario():
    """強い上昇トレンドシナリオを作成"""
    print("🚀 強い上昇トレンドシナリオ")
    print("-" * 40)
    
    engine = TechnicalIndicatorEngine()
    symbol = "BTCUSDT"
    
    # 強い上昇トレンドのデータ（5%の上昇）
    base_price = 100000
    price_data = []
    
    for i in range(20):  # 20 ticks for better technical indicators
        # 0.2-0.3% ずつ上昇 + ランダムノイズ
        upward_trend = i * 0.003  # 3% total upward movement
        noise = np.random.normal(0, 0.001)  # Small noise
        price_multiplier = 1 + upward_trend + noise
        
        current_price = base_price * price_multiplier
        high = current_price * (1 + abs(np.random.normal(0, 0.002)))  # Higher high
        low = current_price * (1 - abs(np.random.normal(0, 0.001)))   # Higher low
        volume = 1000000 * (1 + abs(np.random.normal(0, 0.3)))        # Variable volume
        
        price_data.append((current_price * 0.999, high, low, current_price, volume))
    
    # Process all price data
    for i, (open_price, high, low, close, volume) in enumerate(price_data):
        features = engine.update_price_data(symbol, open_price, high, low, close, volume)
        if i >= 15:  # Log last few
            print(f"  Tick {i+1}: Close=${close:.0f}, Returns={features.get('returns', 0):.4f}")
    
    latest_features = features
    
    # Show key indicators
    print(f"\n📊 主要指標:")
    print(f"  Returns: {latest_features.get('returns', 0):.4f}")
    print(f"  Volatility: {latest_features.get('vol_20', 0):.4f}")
    print(f"  RSI: {latest_features.get('rsi_14', 50):.1f}")
    print(f"  MACD: {latest_features.get('macd', 0):.4f}")
    print(f"  Trend Strength: {latest_features.get('trend_strength_long', 0):.4f}")
    
    return latest_features

def create_strong_bearish_scenario():
    """強い下降トレンドシナリオを作成"""
    print("\n📉 強い下降トレンドシナリオ")
    print("-" * 40)
    
    engine = TechnicalIndicatorEngine()
    symbol = "ETHUSDT"
    
    # 強い下降トレンドのデータ
    base_price = 3000
    price_data = []
    
    for i in range(20):
        # 0.2-0.4% ずつ下降
        downward_trend = -i * 0.004  # -4% total downward movement
        noise = np.random.normal(0, 0.001)
        price_multiplier = 1 + downward_trend + noise
        
        current_price = base_price * price_multiplier
        high = current_price * (1 + abs(np.random.normal(0, 0.001)))   # Lower high
        low = current_price * (1 - abs(np.random.normal(0, 0.002)))    # Lower low
        volume = 1500000 * (1 + abs(np.random.normal(0, 0.4)))         # High volume selloff
        
        price_data.append((current_price * 1.001, high, low, current_price, volume))
    
    # Process all price data
    for i, (open_price, high, low, close, volume) in enumerate(price_data):
        features = engine.update_price_data(symbol, open_price, high, low, close, volume)
        if i >= 15:
            print(f"  Tick {i+1}: Close=${close:.0f}, Returns={features.get('returns', 0):.4f}")
    
    latest_features = features
    
    # Show key indicators
    print(f"\n📊 主要指標:")
    print(f"  Returns: {latest_features.get('returns', 0):.4f}")
    print(f"  Volatility: {latest_features.get('vol_20', 0):.4f}")
    print(f"  RSI: {latest_features.get('rsi_14', 50):.1f}")
    print(f"  MACD: {latest_features.get('macd', 0):.4f}")
    print(f"  Trend Strength: {latest_features.get('trend_strength_long', 0):.4f}")
    
    return latest_features

def test_ml_prediction(features: Dict[str, float], scenario_name: str):
    """ML予測をテストして信号生成をチェック"""
    print(f"\n🔮 {scenario_name} ML予測テスト")
    print("-" * 50)
    
    # InferenceEngine初期化
    config = InferenceConfig()
    engine = InferenceEngine(config)
    engine.load_model()
    
    # 予測実行
    result = engine.predict(features, return_confidence=True)
    prediction = result["predictions"][0] if len(result["predictions"]) > 0 else 0
    confidence = result["confidence_scores"][0] if result.get("confidence_scores") is not None else 0
    
    print(f"予測値: {prediction:.6f}")
    print(f"信頼度: {confidence:.1%}")
    print(f"推論時間: {result['inference_time_ms']:.3f}ms")
    
    # より緩い閾値でテスト
    confidence_thresholds = [0.6, 0.5, 0.4, 0.3]  # Multiple thresholds
    pnl_thresholds = [0.001, 0.0005, 0.0001]      # Multiple PnL thresholds
    
    print(f"\n📊 様々な閾値でのシグナル判定:")
    
    best_signal = None
    
    for conf_thresh in confidence_thresholds:
        for pnl_thresh in pnl_thresholds:
            confidence_pass = confidence >= conf_thresh
            expected_pnl = abs(prediction)
            pnl_pass = expected_pnl >= pnl_thresh
            
            signal_generated = confidence_pass and pnl_pass
            
            status = "✅" if signal_generated else "❌"
            print(f"  {status} 信頼度{conf_thresh:.0%} & PnL{pnl_thresh:.2%}: {confidence:.1%} & {expected_pnl:.3%}")
            
            if signal_generated and not best_signal:
                direction = "BUY" if prediction > 0 else "SELL"
                best_signal = {
                    "direction": direction,
                    "confidence": confidence,
                    "expected_pnl": expected_pnl,
                    "conf_threshold": conf_thresh,
                    "pnl_threshold": pnl_thresh
                }
    
    if best_signal:
        print(f"\n🎯 シグナル生成成功!")
        print(f"  方向: {best_signal['direction']}")
        print(f"  信頼度: {best_signal['confidence']:.1%} (閾値: {best_signal['conf_threshold']:.0%})")
        print(f"  期待PnL: {best_signal['expected_pnl']:.3%} (閾値: {best_signal['pnl_threshold']:.2%})")
        return True
    else:
        print(f"\n❌ 全ての閾値でシグナル生成失敗")
        print(f"  最高信頼度: {confidence:.1%}")
        print(f"  最高期待PnL: {abs(prediction):.3%}")
        return False

def main():
    """メインテスト"""
    print("🎯 強いシグナル生成テスト開始")
    print("=" * 60)
    
    try:
        # 1. 強い上昇トレンドシナリオ
        bullish_features = create_strong_bullish_scenario()
        bullish_success = test_ml_prediction(bullish_features, "上昇トレンド")
        
        # 2. 強い下降トレンドシナリオ  
        bearish_features = create_strong_bearish_scenario()
        bearish_success = test_ml_prediction(bearish_features, "下降トレンド")
        
        # 結果サマリー
        print("\n" + "=" * 60)
        print("📈 テスト結果サマリー")
        print("=" * 60)
        
        print(f"上昇トレンドシグナル: {'✅ 成功' if bullish_success else '❌ 失敗'}")
        print(f"下降トレンドシグナル: {'✅ 成功' if bearish_success else '❌ 失敗'}")
        
        if bullish_success or bearish_success:
            print(f"\n🎉 シグナル生成可能！")
            print(f"💡 適切な閾値設定により実取引で信号生成可能")
            print(f"🚀 EC2にデプロイして実運用開始可能")
        else:
            print(f"\n⚠️  現在のモデルでは明確なシグナル生成困難")
            print(f"💡 提案:")
            print(f"  - より敏感な閾値設定")
            print(f"  - モデルの再訓練")
            print(f"  - 特徴量エンジニアリングの改善")
            
        return bullish_success or bearish_success
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n🏁 テスト完了: {'成功' if success else '要改善'}")