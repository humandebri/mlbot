#!/usr/bin/env python3
"""
緊急診断: なぜMLモデルが0%信頼度を返すのか
"""

import sys
import os
import numpy as np
import asyncio
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_hub.technical_indicators import TechnicalIndicatorEngine
from src.ml_pipeline.feature_adapter_44 import FeatureAdapter44

async def diagnose_model_predictions():
    """モデル予測の問題を診断"""
    
    print("🔍 MLモデル予測問題の緊急診断")
    print("=" * 60)
    
    # 1. 技術的指標エンジンをテスト
    print("\n1. 技術的指標生成テスト...")
    tech_engine = TechnicalIndicatorEngine()
    
    # 複数の価格データで履歴を構築
    test_prices = [
        (106000, 106500, 105500, 106250, 1000000),
        (106250, 106800, 105800, 106600, 1100000),
        (106600, 107000, 106200, 106800, 1200000),
        (106800, 107200, 106400, 107000, 1300000),
        (107000, 107500, 106700, 107200, 1400000)
    ]
    
    for i, (open_p, high, low, close, volume) in enumerate(test_prices):
        features = tech_engine.update_price_data("BTCUSDT", open_p, high, low, close, volume)
        print(f"   データポイント {i+1}: {len(features)} 特徴量生成")
    
    final_features = tech_engine.get_latest_features("BTCUSDT")
    print(f"   ✅ 最終特徴量数: {len(final_features)}")
    
    # 重要な指標の値を確認
    key_indicators = {
        "returns": final_features.get("returns", 0),
        "vol_20": final_features.get("vol_20", 0),
        "rsi_14": final_features.get("rsi_14", 0),
        "macd": final_features.get("macd", 0),
        "price_vs_sma_20": final_features.get("price_vs_sma_20", 0),
        "bb_position_20": final_features.get("bb_position_20", 0)
    }
    
    print("\n   主要指標の値:")
    for name, value in key_indicators.items():
        print(f"   - {name}: {value:.6f}")
    
    # 2. FeatureAdapter44をテスト
    print("\n2. 特徴量アダプターテスト...")
    adapter = FeatureAdapter44()
    adapted_features = adapter.adapt(final_features)
    
    print(f"   ✅ 出力形状: {adapted_features.shape}")
    print(f"   ✅ 全てゼロ?: {np.all(adapted_features == 0)}")
    print(f"   ✅ NaN値?: {np.any(np.isnan(adapted_features))}")
    print(f"   ✅ 無限値?: {np.any(np.isinf(adapted_features))}")
    
    # 値の範囲を確認
    print(f"   最小値: {np.min(adapted_features):.6f}")
    print(f"   最大値: {np.max(adapted_features):.6f}")
    print(f"   平均値: {np.mean(adapted_features):.6f}")
    print(f"   標準偏差: {np.std(adapted_features):.6f}")
    
    # 3. 手動スケーラーをテスト
    print("\n3. 手動スケーラーテスト...")
    
    scaler_path = "models/v3.1_improved/manual_scaler.json"
    if os.path.exists(scaler_path):
        with open(scaler_path, 'r') as f:
            scaler_data = json.load(f)
        
        means = np.array(scaler_data['means'])
        stds = np.array(scaler_data['stds'])
        
        print(f"   ✅ スケーラー読み込み成功")
        print(f"   平均値の範囲: {np.min(means):.6f} ~ {np.max(means):.6f}")
        print(f"   標準偏差の範囲: {np.min(stds):.6f} ~ {np.max(stds):.6f}")
        
        # 正規化を実行
        normalized = (adapted_features - means) / stds
        normalized = np.clip(normalized, -5, 5)
        
        print(f"\n   正規化後:")
        print(f"   最小値: {np.min(normalized):.6f}")
        print(f"   最大値: {np.max(normalized):.6f}")
        print(f"   平均値: {np.mean(normalized):.6f}")
        print(f"   標準偏差: {np.std(normalized):.6f}")
        
        # 4. ONNXモデルでテスト
        print("\n4. ONNXモデル直接テスト...")
        try:
            import onnxruntime as ort
            
            model_path = "models/v3.1_improved/model.onnx"
            if os.path.exists(model_path):
                session = ort.InferenceSession(model_path)
                
                # 予測実行
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                
                prediction = session.run([output_name], {input_name: normalized.reshape(1, -1).astype(np.float32)})
                raw_output = prediction[0][0]
                
                print(f"   ✅ モデル読み込み成功")
                print(f"   ✅ 生の出力: {raw_output}")
                print(f"   ✅ 出力範囲: 0-1か?: {0 <= raw_output <= 1}")
                
                # シグモイド適用が必要か確認
                sigmoid_output = 1 / (1 + np.exp(-raw_output))
                print(f"   シグモイド後: {sigmoid_output}")
                
                # 5. 信頼度計算をテスト
                print("\n5. 信頼度計算テスト...")
                
                # InferenceEngineの信頼度計算を模擬
                confidence = abs(raw_output - 0.5) * 2  # 0.5からの距離を信頼度とする
                expected_pnl = (raw_output - 0.5) * 0.02  # 2%の最大期待収益
                
                print(f"   信頼度: {confidence:.4f} ({confidence*100:.2f}%)")
                print(f"   期待PnL: {expected_pnl:.6f} ({expected_pnl*100:.4f}%)")
                
                # 6. 問題の診断
                print("\n6. 問題診断...")
                
                if confidence < 0.01:
                    print("   ❌ 信頼度が極めて低い - モデルが不確実")
                    print("   💡 解決策: 閾値を0.01以下に下げる")
                
                if abs(expected_pnl) < 0.0001:
                    print("   ❌ 期待PnLが極めて小さい")
                    print("   💡 解決策: PnL閾値を0.00001以下に下げる")
                
                if 0.49 <= raw_output <= 0.51:
                    print("   ❌ モデル出力が中立（0.5付近）")
                    print("   💡 解決策: より極端な市場条件が必要")
                
            else:
                print("   ❌ モデルファイルが見つからない")
                
        except Exception as e:
            print(f"   ❌ ONNXテストエラー: {e}")
    
    else:
        print("   ❌ 手動スケーラーが見つからない")
    
    print("\n" + "=" * 60)
    print("診断完了！上記の結果に基づいて修正が必要です。")


if __name__ == "__main__":
    asyncio.run(diagnose_model_predictions())