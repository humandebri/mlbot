#!/usr/bin/env python3
"""
v3.1_improvedモデルの診断と修復可能性検証
- 分類器問題の詳細分析
- 回帰器への変換可能性検証
- 実際のシグナル生成能力テスト
"""

import sys
import asyncio
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_v31_improved_model():
    """v3.1_improvedモデルの詳細診断"""
    print("="*80)
    print("🔬 v3.1_improved モデル診断・修復検証")
    print("="*80)
    
    model_dir = Path("models/v3.1_improved")
    
    try:
        # 1. メタデータ確認
        print("\n1️⃣ メタデータ分析...")
        
        import json
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ v3.1_improved メタデータ:")
        print(f"   - モデル型: {metadata.get('model_type', 'unknown')}")
        print(f"   - バージョン: {metadata.get('model_version', 'unknown')}")
        print(f"   - 特徴量数: {metadata.get('feature_count', 'unknown')}")
        print(f"   - 訓練日: {metadata.get('training_date', 'unknown')}")
        
        if 'performance' in metadata:
            perf = metadata['performance']
            print(f"   - パフォーマンス:")
            if 'best_auc' in perf:
                print(f"     * AUC: {perf['best_auc']:.3f}")
            if 'best_model' in perf:
                print(f"     * ベストモデル: {perf['best_model']}")
        
        # 2. ONNXモデル構造分析
        print("\n2️⃣ ONNXモデル構造分析...")
        
        model_path = model_dir / "model.onnx"
        model = onnx.load(str(model_path))
        
        # 入力情報
        input_info = model.graph.input[0]
        input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        input_type = input_info.type.tensor_type.elem_type
        
        print(f"✅ ONNX構造:")
        print(f"   - 入力形状: {input_shape}")
        print(f"   - 入力型: {onnx.TensorProto.DataType.Name(input_type)}")
        
        # 出力情報
        for i, output_info in enumerate(model.graph.output):
            output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
            output_type = output_info.type.tensor_type.elem_type
            print(f"   - 出力{i+1}: {output_shape}, 型: {onnx.TensorProto.DataType.Name(output_type)}")
        
        # ノード分析
        node_types = {}
        for node in model.graph.node:
            node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
        
        print(f"   - ノード構成: {dict(sorted(node_types.items()))}")
        
        # 分類器か回帰器かの判定
        has_argmax = 'ArgMax' in node_types
        has_softmax = 'Softmax' in node_types
        has_sigmoid = 'Sigmoid' in node_types
        
        model_type = "分類器" if (has_argmax or has_softmax) else "回帰器" if has_sigmoid else "不明"
        print(f"   - 推定型: {model_type}")
        
        # 3. 実際の推論テスト
        print("\n3️⃣ 推論テスト...")
        
        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"✅ ONNXRuntime情報:")
        print(f"   - 入力名: {input_name}")
        print(f"   - 出力名: {output_names}")
        
        # テスト用入力データ（44次元想定）
        expected_features = 44
        test_inputs = [
            np.random.normal(0, 1, (1, expected_features)).astype(np.float32),  # 標準正規分布
            np.random.uniform(-1, 1, (1, expected_features)).astype(np.float32),  # 均等分布
            np.zeros((1, expected_features), dtype=np.float32),  # ゼロ入力
            np.ones((1, expected_features), dtype=np.float32),   # 1入力
            np.random.normal(0, 0.5, (1, expected_features)).astype(np.float32),  # 小さな分散
        ]
        
        predictions = []
        
        for i, test_input in enumerate(test_inputs):
            try:
                outputs = session.run(output_names, {input_name: test_input})
                
                prediction_data = {
                    'input_id': i+1,
                    'input_stats': {
                        'min': float(test_input.min()),
                        'max': float(test_input.max()),
                        'mean': float(test_input.mean()),
                        'std': float(test_input.std())
                    },
                    'outputs': {}
                }
                
                for j, (output_name, output) in enumerate(zip(output_names, outputs)):
                    prediction_data['outputs'][output_name] = {
                        'shape': list(output.shape),
                        'dtype': str(output.dtype),
                        'values': output.tolist(),
                        'stats': {
                            'min': float(output.min()),
                            'max': float(output.max()),
                            'mean': float(output.mean()),
                            'std': float(output.std()) if output.size > 1 else 0.0
                        }
                    }
                
                predictions.append(prediction_data)
                
                print(f"   テスト{i+1}: ", end="")
                for output_name, output in zip(output_names, outputs):
                    print(f"{output_name}={output.flatten()[:3]} (型:{output.dtype})", end=" ")
                print()
                
            except Exception as e:
                print(f"   テスト{i+1}: エラー - {e}")
        
        # 4. 問題診断
        print("\n4️⃣ 問題診断...")
        
        if not predictions:
            print("❌ 全てのテストで推論失敗")
            return False
        
        # 出力の分析
        first_prediction = predictions[0]
        main_output_name = output_names[0]
        main_output = first_prediction['outputs'][main_output_name]
        
        print(f"✅ 診断結果:")
        print(f"   - 主出力型: {main_output['dtype']}")
        print(f"   - 主出力形状: {main_output['shape']}")
        
        # 分類器問題の確認
        is_classifier_output = 'int' in main_output['dtype'].lower()
        
        if is_classifier_output:
            print("   🚨 確認: 分類器出力（int型）")
            
            # 全ての予測が同じ値かチェック
            all_predictions_same = True
            first_values = predictions[0]['outputs'][main_output_name]['values']
            
            for pred in predictions[1:]:
                if pred['outputs'][main_output_name]['values'] != first_values:
                    all_predictions_same = False
                    break
            
            if all_predictions_same:
                print("   ❌ 全予測で同じ値 → 実質的に機能していない")
            else:
                print("   ✅ 予測値に多様性あり")
            
            # 5. 修復可能性の検証
            print("\n5️⃣ 修復可能性検証...")
            
            # ONNXモデルの最終層を確認
            print("   最終層ノード分析:")
            final_nodes = model.graph.node[-5:]  # 最後の5ノード
            for node in final_nodes:
                print(f"     - {node.op_type}: {node.input} → {node.output}")
            
            # 修復戦略の提案
            print("\n📋 修復戦略:")
            
            if has_argmax:
                print("   1. ArgMaxノード除去 → 確率値直接取得")
                print("   2. Softmax出力を回帰値として解釈")
                fix_difficulty = "中程度"
            elif 'int' in main_output['dtype'].lower():
                print("   1. 最終層をfloat出力に変更")
                print("   2. 閾値適用前の確率値取得")
                fix_difficulty = "困難"
            else:
                print("   1. 出力の後処理修正")
                fix_difficulty = "簡単"
            
            print(f"   修復難易度: {fix_difficulty}")
            
            # 6. 代替アプローチ
            print("\n6️⃣ 代替アプローチ...")
            
            # 複数出力がある場合の確認
            if len(output_names) > 1:
                print("   複数出力検出 - 確率出力を探索:")
                for name, pred in first_prediction['outputs'].items():
                    if 'float' in pred['dtype'].lower():
                        print(f"     ✅ {name}: float型出力発見 (値: {pred['values']})")
                        print("     → この出力を信頼度として使用可能")
                        return True
            
            # TreeEnsemble特有の処理
            if 'TreeEnsemble' in str(node_types):
                print("   TreeEnsemble検出:")
                print("     - 確率出力（probabilites）の存在確認")
                print("     - ノード再構成による回帰化")
            
            print(f"\n📊 総合判定:")
            if all_predictions_same:
                print("❌ 修復不推奨: 予測多様性なし")
                return False
            elif len(output_names) > 1:
                print("✅ 修復可能: 複数出力から確率値抽出")
                return True
            elif has_softmax or has_sigmoid:
                print("🟡 修復可能: 確率出力を回帰値として使用")
                return True
            else:
                print("❌ 修復困難: 根本的な構造変更が必要")
                return False
        
        else:
            print("   ✅ 既に回帰器出力（float型）")
            
            # 予測多様性の確認
            all_values = []
            for pred in predictions:
                values = pred['outputs'][main_output_name]['values']
                all_values.extend(values)
            
            variance = np.var(all_values)
            
            if variance > 0.001:
                print(f"   ✅ 予測多様性良好 (分散: {variance:.6f})")
                return True
            else:
                print(f"   ⚠️ 予測多様性不足 (分散: {variance:.6f})")
                return False
        
    except Exception as e:
        print(f"❌ 診断エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    try:
        result = asyncio.run(test_v31_improved_model())
        
        print("\n" + "="*80)
        if result:
            print("🎉 v3.1_improved修復可能性: ✅ 高い")
            print("推奨アクション: 修復実装に進む")
        else:
            print("❌ v3.1_improved修復可能性: 低い") 
            print("推奨アクション: 別のアプローチを検討")
        print("="*80)
        
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()