#!/usr/bin/env python3
"""
モデル入力の詳細診断

FeatureAdapter26が生成する26次元特徴量が
モデルに適切に入力されているかを確認
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
from src.ml_pipeline.feature_adapter_26 import FeatureAdapter26
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
import onnxruntime as ort

def create_sample_features():
    """サンプル特徴量を生成"""
    print("🔧 サンプル特徴量生成")
    print("-" * 40)
    
    engine = TechnicalIndicatorEngine()
    symbol = "BTCUSDT"
    
    # 現実的な価格データを複数回投入
    price_data = [
        (100000, 100500, 99800, 100200, 1000000),
        (100200, 100800, 100000, 100600, 1200000),
        (100600, 101000, 100400, 100900, 1100000),
        (100900, 101200, 100700, 101100, 1300000),
        (101100, 101500, 100900, 101300, 1250000),
    ]
    
    for i, (open_price, high, low, close, volume) in enumerate(price_data):
        features = engine.update_price_data(symbol, open_price, high, low, close, volume)
    
    print(f"✅ 生成された特徴量: {len(features)}個")
    return features

def test_feature_adapter_details(features: Dict[str, float]):
    """FeatureAdapter26の詳細テスト"""
    print(f"\n🔧 FeatureAdapter26詳細テスト")
    print("-" * 40)
    
    adapter = FeatureAdapter26()
    
    print(f"入力特徴量:")
    for i, (key, value) in enumerate(list(features.items())[:10]):
        print(f"  {key}: {value}")
    if len(features) > 10:
        print(f"  ... and {len(features) - 10} more")
    
    # 26次元に変換
    adapted_features = adapter.adapt(features)
    
    print(f"\n出力26次元特徴量:")
    target_features = adapter.get_feature_names()
    for i, (name, value) in enumerate(zip(target_features, adapted_features)):
        print(f"  [{i:2d}] {name:20s}: {value:10.6f}")
    
    print(f"\n統計情報:")
    print(f"  範囲: [{np.min(adapted_features):.6f}, {np.max(adapted_features):.6f}]")
    print(f"  平均: {np.mean(adapted_features):.6f}")
    print(f"  標準偏差: {np.std(adapted_features):.6f}")
    print(f"  非ゼロ値: {np.count_nonzero(adapted_features)}/26")
    print(f"  無限値: {np.sum(np.isinf(adapted_features))}")
    print(f"  NaN値: {np.sum(np.isnan(adapted_features))}")
    
    return adapted_features

def test_model_direct_onnx(adapted_features):
    """ONNXモデルを直接テスト"""
    print(f"\n🔧 ONNX直接テスト")
    print("-" * 40)
    
    model_paths = [
        "models/v1.0/model.onnx",
        "models/catboost_model.onnx"
    ]
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            continue
            
        print(f"\n📁 テスト: {model_path}")
        print("-" * 20)
        
        try:
            session = ort.InferenceSession(model_path)
            
            # 入力情報
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            print(f"入力要求: {input_info.name} {input_info.shape} {input_info.type}")
            print(f"出力形式: {output_info.name} {output_info.shape} {output_info.type}")
            
            # 入力データ準備
            input_array = adapted_features.reshape(1, -1).astype(np.float32)
            print(f"実際の入力形状: {input_array.shape}")
            print(f"実際の入力型: {input_array.dtype}")
            print(f"入力値サンプル: {input_array[0][:5]}")
            
            # 予測実行
            result = session.run([output_info.name], {input_info.name: input_array})
            output = result[0]
            
            print(f"生出力: {output}")
            print(f"出力形状: {output.shape}")
            print(f"出力型: {output.dtype}")
            
            if len(output.shape) > 1:
                output_value = output[0][0] if output.shape[1] > 0 else 0
            else:
                output_value = output[0] if len(output) > 0 else 0
            
            print(f"予測値: {output_value}")
            
            # 複数のランダム入力で検証
            print(f"\n🔄 複数入力テスト:")
            for i in range(3):
                # わずかに変更した入力
                noise_input = input_array + np.random.normal(0, 0.01, input_array.shape).astype(np.float32)
                test_result = session.run([output_info.name], {input_info.name: noise_input})
                test_output = test_result[0]
                
                if len(test_output.shape) > 1:
                    test_value = test_output[0][0] if test_output.shape[1] > 0 else 0
                else:
                    test_value = test_output[0] if len(test_output) > 0 else 0
                
                print(f"  テスト{i+1}: {test_value:.8f}")
            
            # 極端な入力テスト
            print(f"\n🎯 極端入力テスト:")
            extreme_inputs = [
                ("全て0", np.zeros((1, 26), dtype=np.float32)),
                ("全て1", np.ones((1, 26), dtype=np.float32)),
                ("大きな値", np.ones((1, 26), dtype=np.float32) * 100),
                ("負の値", np.ones((1, 26), dtype=np.float32) * -1),
                ("ランダム", np.random.normal(0, 10, (1, 26)).astype(np.float32)),
            ]
            
            for name, extreme_input in extreme_inputs:
                try:
                    extreme_result = session.run([output_info.name], {input_info.name: extreme_input})
                    extreme_output = extreme_result[0]
                    
                    if len(extreme_output.shape) > 1:
                        extreme_value = extreme_output[0][0] if extreme_output.shape[1] > 0 else 0
                    else:
                        extreme_value = extreme_output[0] if len(extreme_output) > 0 else 0
                    
                    print(f"  {name:10s}: {extreme_value:.8f}")
                except Exception as e:
                    print(f"  {name:10s}: エラー - {e}")
                    
        except Exception as e:
            print(f"❌ エラー: {e}")

def analyze_feature_ranges():
    """特徴量の範囲分析"""
    print(f"\n📊 特徴量範囲分析")
    print("-" * 40)
    
    # より現実的な特徴量を手動で作成
    realistic_features = {
        # 基本価格特徴量
        "returns": 0.002,        # 0.2% return
        "log_returns": 0.00199,  # log return
        "close": 100000,         # BTC price
        "volume": 1000000,       # Volume
        "price_change_pct": 0.002,
        "high_low_ratio": 1.002,
        "volume_ratio": 1.1,
        "volatility_20": 0.015,
        
        # テクニカル指標
        "rsi_14": 65.0,
        "macd": 50.0,
        "bb_position_20": 0.2,
        "bb_width_20": 0.03,
        "sma_5": 99800,
        "sma_10": 99600,
        "sma_20": 99200,
        "close_to_sma5": 0.002,
        "close_to_sma10": 0.004,
        "close_to_sma20": 0.008,
        
        # 高度な特徴量
        "trend_strength_short": 0.1,
        "trend_strength_long": 0.15,
        "market_regime": 0.0,
        "momentum_3": 0.003,
        "momentum_5": 0.005,
        
        # 時間特徴量
        "hour_sin": 0.5,
        "hour_cos": 0.866,
        "is_weekend": 0.0
    }
    
    print("🎯 現実的な特徴量セット:")
    for key, value in realistic_features.items():
        print(f"  {key:20s}: {value:10.6f}")
    
    return realistic_features

def main():
    """メイン診断関数"""
    print("🔍 モデル入力詳細診断開始")
    print("=" * 60)
    
    try:
        # 1. 通常の特徴量生成
        features = create_sample_features()
        adapted_features = test_feature_adapter_details(features)
        
        # 2. ONNX直接テスト
        test_model_direct_onnx(adapted_features)
        
        # 3. 現実的な特徴量テスト
        realistic_features = analyze_feature_ranges()
        adapter = FeatureAdapter26()
        realistic_adapted = adapter.adapt(realistic_features)
        
        print(f"\n🎯 現実的特徴量でのテスト:")
        test_model_direct_onnx(realistic_adapted)
        
        print("\n" + "=" * 60)
        print("📈 診断完了")
        print("=" * 60)
        print("✅ モデルは動作するが予測値が非常に小さい")
        print("💡 可能性のある原因:")
        print("  1. モデルが異なる特徴量セットで訓練された")
        print("  2. 特徴量のスケール・範囲が期待値と異なる")
        print("  3. モデルが保守的すぎる（小さな予測値のみ出力）")
        print("  4. 特徴量エンジニアリングが不適切")
        
    except Exception as e:
        print(f"❌ 診断中にエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()