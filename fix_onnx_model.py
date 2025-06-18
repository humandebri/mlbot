#!/usr/bin/env python3
"""
ONNXモデル修復 - モデルが0を返す問題の解決
"""

import sys
import os
import numpy as np
import json
import onnxruntime as ort
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def diagnose_onnx_model():
    """ONNXモデルの問題を詳細に診断"""
    
    print("🔧 ONNXモデル修復開始")
    print("=" * 60)
    
    model_path = "models/v3.1_improved/model.onnx"
    scaler_path = "models/v3.1_improved/manual_scaler.json"
    
    # 1. モデル詳細情報
    print("\n1. モデル詳細情報")
    print("-" * 30)
    
    try:
        session = ort.InferenceSession(model_path)
        
        # 入力情報
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"入力名: {input_info.name}")
        print(f"入力形状: {input_info.shape}")
        print(f"入力タイプ: {input_info.type}")
        
        print(f"出力名: {output_info.name}")
        print(f"出力形状: {output_info.shape}")
        print(f"出力タイプ: {output_info.type}")
        
        # 2. 複数の入力パターンでテスト
        print("\n2. 複数入力パターンテスト")
        print("-" * 30)
        
        # テスト入力パターン
        test_inputs = [
            ("全て0", np.zeros(44, dtype=np.float32)),
            ("全て1", np.ones(44, dtype=np.float32)),
            ("ランダム小", np.random.normal(0, 0.1, 44).astype(np.float32)),
            ("ランダム大", np.random.normal(0, 1, 44).astype(np.float32)),
            ("順番", np.arange(44, dtype=np.float32) / 44),
            ("正規化済み", np.random.normal(0, 1, 44).astype(np.float32)),
        ]
        
        for name, input_data in test_inputs:
            # 入力形状を調整
            input_reshaped = input_data.reshape(1, -1)
            
            # 予測実行
            result = session.run([output_info.name], {input_info.name: input_reshaped})
            output = result[0][0]
            
            print(f"{name:12}: {output:.6f} (タイプ: {type(output)})")
            
            # 出力が常に0の場合、問題を特定
            if output == 0:
                print(f"  ⚠️  {name}でも0が返される")
        
        # 3. スケーラー適用テスト
        print("\n3. スケーラー適用テスト")
        print("-" * 30)
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)
            
            means = np.array(scaler_data['means'])
            stds = np.array(scaler_data['stds'])
            
            print(f"スケーラー平均値範囲: {np.min(means):.3f} ~ {np.max(means):.3f}")
            print(f"スケーラー標準偏差範囲: {np.min(stds):.3f} ~ {np.max(stds):.3f}")
            
            # 現実的な特徴量を生成
            realistic_features = np.array([
                0.001,   # returns
                0.0009,  # log_returns  
                1.002,   # hl_ratio
                0.002,   # oc_ratio
                0.001, 0.003, 0.005, 0.01, 0.02,  # multi-period returns
                0.01, 0.015, 0.02, 0.025,  # volatility
                1.0, 1.0,  # vol ratios
                0.0, 0.0, 0.0, 0.0,  # price vs sma
                0.0, 0.0,  # price vs ema
                0.0, 0.0,  # macd
                55.0, 52.0,  # rsi
                0.1, 0.02,  # bollinger bands
                1.1, 1.2,  # volume ratios
                12.0, 0.0,  # volume features
                0.01, 0.005, 0.002,  # momentum
                0.6, 0.4,  # percentiles
                0.1, 0.05,  # trend strength
                0.0, 0.0, 0.0,  # regimes
                0.5, 0.866, 0.0  # time features
            ], dtype=np.float32)
            
            print(f"現実的特徴量: {len(realistic_features)}個")
            
            # 正規化適用
            normalized = (realistic_features - means) / stds
            normalized = np.clip(normalized, -5, 5)
            
            print(f"正規化後範囲: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
            
            # 予測実行（float32にキャスト）
            input_float32 = normalized.reshape(1, -1).astype(np.float32)
            result = session.run([output_info.name], {input_info.name: input_float32})
            output = result[0][0]
            
            print(f"現実的特徴量での予測: {output:.6f}")
            print(f"出力型確認: {type(output)} (値: {output})")
            
            if output == 0:
                print("  🚨 現実的特徴量でも0が返される - モデルに根本的問題")
            else:
                print("  ✅ 非ゼロ値が返された - 特徴量の問題")
        
        # 4. モデル構造確認
        print("\n4. モデル構造診断")
        print("-" * 30)
        
        # ONNXモデルの詳細情報を確認
        try:
            import onnx
            model = onnx.load(model_path)
            
            print(f"モデルバージョン: {model.ir_version}")
            print(f"グラフノード数: {len(model.graph.node)}")
            print(f"初期化子数: {len(model.graph.initializer)}")
            
            # 最初と最後のノードを確認
            if model.graph.node:
                first_node = model.graph.node[0]
                last_node = model.graph.node[-1]
                
                print(f"最初のノード: {first_node.op_type}")
                print(f"最後のノード: {last_node.op_type}")
                
        except ImportError:
            print("onnxライブラリが無いため、構造確認をスキップ")
        
        # 5. 修復提案
        print("\n5. 修復提案")
        print("-" * 30)
        
        if all(session.run([output_info.name], {input_info.name: test[1].reshape(1, -1)})[0][0] == 0 
               for _, test in test_inputs):
            print("🚨 全入力で0が返される - モデル自体に問題")
            print("💡 修復方法:")
            print("  1. 異なるモデルファイルを使用")
            print("  2. モデルを再訓練")
            print("  3. PyTorchモデルから再変換")
        else:
            print("✅ 特定の入力で非ゼロ値 - 前処理の問題")
            print("💡 修復方法:")
            print("  1. 特徴量正規化の調整")
            print("  2. スケーラーパラメータの確認")
            print("  3. 特徴量の範囲確認")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_onnx_model()