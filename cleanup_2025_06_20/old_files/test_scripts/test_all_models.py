#!/usr/bin/env python3
"""
全モデルをテストして回帰器を探す
"""

import os
import numpy as np
import onnxruntime as ort
from pathlib import Path

def test_all_models():
    """全ONNXモデルをテストして正しい回帰器を見つける"""
    
    print("🔍 全ONNXモデルテスト")
    print("=" * 60)
    
    # モデルパスを取得
    model_paths = [
        "models/v3.1_improved/model.onnx",
        "models/v2.0/model.onnx", 
        "models/real_156_features_20250615_230554/model.onnx",
        "models/v1.0/model.onnx",
        "models/cascade_detection/cascade_v1_20250612_150231/model.onnx",
        "models/catboost_model.onnx"
    ]
    
    working_models = []
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            continue
            
        print(f"\n📁 テスト: {model_path}")
        print("-" * 40)
        
        try:
            session = ort.InferenceSession(model_path)
            
            # 基本情報
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            print(f"入力: {input_info.name} {input_info.shape} {input_info.type}")
            print(f"出力: {output_info.name} {output_info.shape} {output_info.type}")
            
            # 入力次元確認
            expected_dims = None
            if hasattr(input_info.shape, '__len__') and len(input_info.shape) > 1:
                expected_dims = input_info.shape[1]
                if isinstance(expected_dims, str):  # 'None'など
                    expected_dims = 44  # デフォルト
            else:
                expected_dims = 44  # デフォルト
            
            print(f"期待される入力次元: {expected_dims}")
            
            # テスト入力生成
            if expected_dims == 44:
                test_input = np.random.normal(0, 1, 44).astype(np.float32).reshape(1, -1)
            elif expected_dims == 156:
                test_input = np.random.normal(0, 1, 156).astype(np.float32).reshape(1, -1)
            elif expected_dims == 26:
                test_input = np.random.normal(0, 1, 26).astype(np.float32).reshape(1, -1)
            else:
                print(f"  ❌ 不明な次元数: {expected_dims}")
                continue
            
            # 予測テスト
            result = session.run([output_info.name], {input_info.name: test_input})
            output = result[0]
            
            print(f"出力: {output} (形状: {output.shape}, 型: {output.dtype})")
            
            # 回帰器かどうかチェック
            is_regressor = False
            is_classifier = False
            
            # 出力型で判定
            if 'float' in str(output.dtype):
                is_regressor = True
                print("  ✅ 回帰器（float出力）")
            elif 'int' in str(output.dtype):
                is_classifier = True
                print("  ❌ 分類器（int出力）")
            
            # 出力値の範囲で判定
            if is_regressor:
                output_val = float(output.flatten()[0]) if len(output.flatten()) > 0 else 0
                if 0 <= output_val <= 1:
                    print(f"  ✅ 確率範囲の出力: {output_val:.6f}")
                    working_models.append((model_path, expected_dims, output_val))
                else:
                    print(f"  ⚠️  範囲外の出力: {output_val:.6f}")
            
            # 複数のテスト入力で確認
            test_results = []
            for i in range(3):
                if expected_dims == 44:
                    rand_input = np.random.normal(0, 1, 44).astype(np.float32).reshape(1, -1)
                elif expected_dims == 156:
                    rand_input = np.random.normal(0, 1, 156).astype(np.float32).reshape(1, -1)
                elif expected_dims == 26:
                    rand_input = np.random.normal(0, 1, 26).astype(np.float32).reshape(1, -1)
                
                test_result = session.run([output_info.name], {input_info.name: rand_input})
                test_output = test_result[0].flatten()[0] if len(test_result[0].flatten()) > 0 else 0
                test_results.append(test_output)
            
            print(f"  複数テスト結果: {test_results}")
            
            # 全て同じ値なら問題
            if len(set([round(float(x), 6) for x in test_results])) == 1:
                print("  ⚠️  全て同じ値 - モデルに問題の可能性")
            else:
                print("  ✅ 異なる値 - モデル正常動作")
                
        except Exception as e:
            print(f"  ❌ エラー: {e}")
    
    # 使用可能なモデルの要約
    print("\n" + "=" * 60)
    print("🎯 使用可能なモデル")
    print("=" * 60)
    
    if working_models:
        for model_path, dims, output_val in working_models:
            print(f"✅ {model_path}")
            print(f"   入力次元: {dims}, サンプル出力: {output_val:.6f}")
    else:
        print("❌ 使用可能な回帰モデルが見つかりませんでした")
        
        print("\n💡 代替案:")
        print("1. PyTorchモデルから回帰器のONNXを再生成")
        print("2. 簡単な回帰モデルを新規作成")
        print("3. 分類器を確率出力モードで使用")
    
    return working_models

if __name__ == "__main__":
    working_models = test_all_models()
    
    # 最適なモデルを推奨
    if working_models:
        best_model = working_models[0]  # 最初の動作モデル
        print(f"\n🎯 推奨モデル: {best_model[0]}")
        print(f"   次元: {best_model[1]}, 出力例: {best_model[2]:.6f}")