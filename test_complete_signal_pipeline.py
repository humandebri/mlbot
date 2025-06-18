#!/usr/bin/env python3
"""
完全なシグナル生成パイプラインのテスト

修正された内容:
1. FeatureAdapter26を使用（26次元対応）
2. 動作する回帰器モデル（catboost_model.onnx）を使用
3. TechnicalIndicatorEngineからInferenceEngineまでの完全なフロー

期待される結果:
- 非ゼロの予測値
- シグナル生成
- Discord通知が送信される
"""

import os
import sys
import asyncio
import numpy as np
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Environment setup
from dotenv import load_dotenv
load_dotenv()

from src.feature_hub.technical_indicators import TechnicalIndicatorEngine
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
from src.ml_pipeline.feature_adapter_26 import FeatureAdapter26
from src.common.config import settings

def test_technical_indicators():
    """テクニカル指標の生成をテスト"""
    print("🔧 テクニカル指標エンジンのテスト")
    print("-" * 50)
    
    engine = TechnicalIndicatorEngine()
    
    # 現実的な価格データを生成
    symbol = "BTCUSDT"
    price_data = [
        (106000, 106500, 105800, 106200, 1000000),  # OHLCV
        (106200, 106800, 106000, 106600, 1200000),
        (106600, 107000, 106400, 106900, 1100000),
        (106900, 107200, 106700, 107100, 1300000),
        (107100, 107500, 106900, 107300, 1250000),
    ]
    
    features_list = []
    for i, (open_price, high, low, close, volume) in enumerate(price_data):
        features = engine.update_price_data(symbol, open_price, high, low, close, volume)
        features_list.append(features)
        print(f"  Tick {i+1}: {len(features)} features generated")
        
        # 主要な特徴量を表示
        print(f"    Returns: {features.get('returns', 0):.6f}")
        print(f"    RSI: {features.get('rsi_14', 50):.2f}")
        print(f"    MACD: {features.get('macd', 0):.6f}")
        print(f"    Volatility: {features.get('vol_20', 0):.6f}")
    
    latest_features = features_list[-1]
    print(f"\n✅ 最新特徴量: {len(latest_features)}個")
    return latest_features

def test_feature_adapter(features: Dict[str, float]):
    """FeatureAdapter26のテスト"""
    print(f"\n🔧 FeatureAdapter26のテスト")
    print("-" * 50)
    
    adapter = FeatureAdapter26()
    
    print(f"入力特徴量数: {len(features)}")
    print(f"入力特徴量例: {list(features.keys())[:10]}")
    
    # 26次元に変換
    adapted_features = adapter.adapt(features)
    
    print(f"出力形状: {adapted_features.shape}")
    print(f"出力型: {adapted_features.dtype}")
    print(f"出力範囲: [{np.min(adapted_features):.3f}, {np.max(adapted_features):.3f}]")
    print(f"非ゼロ値の数: {np.count_nonzero(adapted_features)}/26")
    
    # 統計情報
    stats = adapter.get_adaptation_stats(features)
    print(f"適応統計:")
    print(f"  マッチ率: {stats['match_rate']:.2%}")
    print(f"  マッチした特徴量: {stats['matched_features']}")
    print(f"  欠落特徴量: {stats['missing_features']}")
    
    print("✅ FeatureAdapter26テスト完了")
    return adapted_features

def test_inference_engine(features: Dict[str, float]):
    """InferenceEngineのテスト"""
    print(f"\n🔧 InferenceEngineのテスト")
    print("-" * 50)
    
    # 設定確認
    config = InferenceConfig()
    print(f"モデルパス: {config.model_path}")
    print(f"使用中の設定: {settings.model.model_path}")
    
    # InferenceEngine初期化
    engine = InferenceEngine(config)
    
    # モデル読み込み
    print("モデル読み込み中...")
    engine.load_model()
    
    # 入力情報確認
    input_info = engine.onnx_session.get_inputs()[0]
    output_info = engine.onnx_session.get_outputs()[0]
    print(f"モデル入力: {input_info.name} {input_info.shape} {input_info.type}")
    print(f"モデル出力: {output_info.name} {output_info.shape} {output_info.type}")
    
    # 予測実行
    print("予測実行中...")
    result = engine.predict(features, return_confidence=True)
    
    # 結果表示
    prediction = result["predictions"][0] if len(result["predictions"]) > 0 else 0
    confidence = result["confidence_scores"][0] if result.get("confidence_scores") is not None else 0
    
    print(f"予測値: {prediction:.6f}")
    print(f"信頼度: {confidence:.6f}")
    print(f"推論時間: {result['inference_time_ms']:.3f}ms")
    print(f"入力形状: {result['model_info']['input_shape']}")
    
    # 結果検証
    if prediction == 0:
        print("❌ 予測値が0 - まだ問題がある")
        return False
    else:
        print("✅ 非ゼロ予測値 - 正常動作")
        return True

def test_signal_generation(features: Dict[str, float]):
    """シグナル生成ロジックのテスト"""
    print(f"\n🔧 シグナル生成ロジックのテスト")
    print("-" * 50)
    
    # InferenceEngineで予測実行
    config = InferenceConfig()
    engine = InferenceEngine(config)
    engine.load_model()
    
    result = engine.predict(features, return_confidence=True)
    prediction = result["predictions"][0] if len(result["predictions"]) > 0 else 0
    confidence = result["confidence_scores"][0] if result.get("confidence_scores") is not None else 0
    
    print(f"ML予測: {prediction:.6f}")
    print(f"信頼度: {confidence:.6f}")
    
    # シグナル判定（実際のロジックを模擬）
    min_confidence = 0.6  # 60%
    min_expected_pnl = 0.001  # 0.1%
    
    print(f"\n📊 シグナル判定:")
    print(f"  信頼度閾値: {min_confidence:.1%}")
    print(f"  期待PnL閾値: {min_expected_pnl:.1%}")
    
    # 信頼度チェック
    confidence_pass = confidence >= min_confidence
    print(f"  信頼度チェック: {confidence:.1%} >= {min_confidence:.1%} = {'✅' if confidence_pass else '❌'}")
    
    # 期待PnLチェック（予測値を期待PnLとして使用）
    expected_pnl = abs(prediction)
    pnl_pass = expected_pnl >= min_expected_pnl
    print(f"  期待PnLチェック: {expected_pnl:.3%} >= {min_expected_pnl:.1%} = {'✅' if pnl_pass else '❌'}")
    
    # 最終判定
    signal_generated = confidence_pass and pnl_pass
    
    if signal_generated:
        direction = "BUY" if prediction > 0 else "SELL"
        print(f"\n🎯 シグナル生成: {direction}")
        print(f"  方向: {direction}")
        print(f"  信頼度: {confidence:.1%}")
        print(f"  期待PnL: {expected_pnl:.3%}")
        return True
    else:
        print(f"\n❌ シグナル生成されず")
        reasons = []
        if not confidence_pass:
            reasons.append(f"信頼度不足 ({confidence:.1%})")
        if not pnl_pass:
            reasons.append(f"期待PnL不足 ({expected_pnl:.3%})")
        print(f"  理由: {', '.join(reasons)}")
        return False

def main():
    """メインテスト関数"""
    print("🚀 完全シグナル生成パイプラインテスト開始")
    print("=" * 60)
    
    try:
        # 1. テクニカル指標生成
        features = test_technical_indicators()
        
        # 2. 特徴量適応
        adapted_features = test_feature_adapter(features)
        
        # 3. ML推論エンジン
        inference_success = test_inference_engine(features)
        
        # 4. シグナル生成
        signal_success = test_signal_generation(features)
        
        # 総合結果
        print("\n" + "=" * 60)
        print("📈 総合テスト結果")
        print("=" * 60)
        
        results = [
            ("テクニカル指標生成", len(features) > 40),
            ("特徴量適応", adapted_features.shape == (26,)),
            ("ML推論エンジン", inference_success),
            ("シグナル生成", signal_success)
        ]
        
        for test_name, success in results:
            status = "✅ 成功" if success else "❌ 失敗"
            print(f"  {test_name}: {status}")
        
        all_success = all(success for _, success in results)
        
        if all_success:
            print(f"\n🎉 全テスト成功！シグナル生成パイプライン正常動作")
            print(f"💡 EC2にデプロイして実際の取引を開始可能")
        else:
            print(f"\n⚠️  一部テスト失敗。問題箇所を修正が必要")
            
        return all_success
        
    except Exception as e:
        print(f"❌ テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)