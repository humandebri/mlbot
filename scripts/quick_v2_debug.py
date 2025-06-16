#!/usr/bin/env python3
"""
v2.0モデルの問題を診断するクイックテスト
"""
import os
import sys
import onnxruntime as ort
import numpy as np
import json
from pathlib import Path

# プロジェクトのルートパスを追加
sys.path.append(str(Path(__file__).parent.parent))

def test_v2_model():
    """v2.0モデルをテストしてAUC 0.5の原因を特定"""
    
    model_path = Path(__file__).parent.parent / "models" / "v2.0" / "model.onnx"
    metadata_path = Path(__file__).parent.parent / "models" / "v2.0" / "metadata.json"
    
    if not model_path.exists():
        print(f"モデルファイルが見つかりません: {model_path}")
        return
    
    # メタデータを読み込み
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"モデル情報:")
    print(f"  タイプ: {metadata.get('model_type', 'unknown')}")
    print(f"  特徴量数: {metadata.get('feature_count', 0)}")
    print(f"  訓練日時: {metadata.get('training_date', 'unknown')}")
    
    # ONNXモデルをロード
    try:
        session = ort.InferenceSession(str(model_path))
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"\nONNXモデル情報:")
        print(f"  入力名: {input_names}")
        print(f"  出力名: {output_names}")
        
        # 入力の形状を確認
        for inp in session.get_inputs():
            print(f"  入力形状 '{inp.name}': {inp.shape}")
        
        # 出力の形状を確認
        for out in session.get_outputs():
            print(f"  出力形状 '{out.name}': {out.shape}")
        
    except Exception as e:
        print(f"ONNXモデルのロードに失敗: {e}")
        return
    
    # テストデータを生成（156特徴量）
    feature_count = metadata.get('feature_count', 156)
    test_samples = 100
    
    # 正常な範囲の値でテストデータを生成
    np.random.seed(42)
    test_data = np.random.randn(test_samples, feature_count).astype(np.float32)
    
    print(f"\n予測テスト:")
    print(f"  テストサンプル数: {test_samples}")
    print(f"  特徴量数: {feature_count}")
    print(f"  データ範囲: [{test_data.min():.4f}, {test_data.max():.4f}]")
    
    try:
        # 予測実行
        predictions = session.run(output_names, {input_names[0]: test_data})[0]
        
        print(f"\n予測結果:")
        print(f"  予測値の形状: {predictions.shape}")
        print(f"  予測値の範囲: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"  予測値の平均: {predictions.mean():.6f}")
        print(f"  予測値の標準偏差: {predictions.std():.6f}")
        print(f"  ユニークな予測値数: {len(np.unique(predictions))}")
        
        # 分布を確認
        unique_values = np.unique(predictions)
        if len(unique_values) <= 10:
            print(f"  全ユニーク値: {unique_values}")
        
        # すべての予測が同じ値かチェック
        if len(unique_values) == 1:
            print(f"  ⚠️  警告: すべての予測値が同じ ({unique_values[0]:.6f})")
            print("  これがAUC 0.5000の原因です")
        
        # 予測値の詳細統計
        print(f"\n詳細統計:")
        print(f"  25%分位: {np.percentile(predictions, 25):.6f}")
        print(f"  50%分位 (中央値): {np.percentile(predictions, 50):.6f}")
        print(f"  75%分位: {np.percentile(predictions, 75):.6f}")
        
    except Exception as e:
        print(f"予測の実行に失敗: {e}")
        return
    
    # 以前の高性能モデル（v1.0やsimple_ensemble）と比較の提案
    print(f"\n\n🔍 次のステップ:")
    print("1. 以前の高性能モデル（AUC 0.867）のコードを確認")
    print("2. v2.0訓練時のデータと特徴量生成プロセスを調査")
    print("3. 実際の市場データに基づく特徴量エンジニアリングを再実装")
    print("4. 適切な検証データでモデルを再訓練")

if __name__ == "__main__":
    test_v2_model()