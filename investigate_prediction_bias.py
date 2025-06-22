#!/usr/bin/env python3
"""
予測の偏りを調査する簡易スクリプト
"""

import numpy as np
import onnxruntime as ort
import json

def investigate_bias():
    """モデルの出力傾向を調査"""
    
    print("🔍 モデルの予測バイアスを調査...\n")
    
    # 1. モデルとスケーラーを読み込み
    model_path = "models/v3.1_improved/model.onnx"
    scaler_path = "models/v3.1_improved/manual_scaler.json"
    
    session = ort.InferenceSession(model_path)
    
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    
    mean = np.array(scaler_params['means'])
    std = np.array(scaler_params['stds'])
    
    print(f"📊 スケーラー情報:")
    print(f"  特徴量数: {len(mean)}")
    print(f"  平均値の範囲: [{np.min(mean):.4f}, {np.max(mean):.4f}]")
    print(f"  標準偏差の範囲: [{np.min(std):.4f}, {np.max(std):.4f}]")
    
    # stdがゼロまたは極小の特徴量を確認
    zero_std = np.sum(std == 0)
    tiny_std = np.sum((std > 0) & (std < 1e-6))
    print(f"  stdがゼロの特徴量: {zero_std}個")
    print(f"  stdが極小(<1e-6)の特徴量: {tiny_std}個")
    
    # 2. 様々な入力でモデルをテスト
    print("\n\n📈 モデル予測テスト:")
    
    test_cases = [
        ("ゼロベクトル", np.zeros(44)),
        ("平均値", mean),
        ("平均値 + 1std", mean + std),
        ("平均値 - 1std", mean - std),
        ("ランダム（正規分布）", np.random.randn(44)),
        ("ランダム（一様分布）", np.random.uniform(-3, 3, 44)),
    ]
    
    predictions = []
    
    for name, features in test_cases:
        # 正規化
        std_safe = np.where(std == 0, 1.0, std)
        normalized = (features - mean) / std_safe
        
        # NaN/Infチェック
        nan_count = np.isnan(normalized).sum()
        inf_count = np.isinf(normalized).sum()
        
        # 予測
        input_data = normalized.reshape(1, -1).astype(np.float32)
        outputs = session.run(None, {'float_input': input_data})
        
        if len(outputs) > 1 and isinstance(outputs[1], list) and len(outputs[1]) > 0:
            prob_dict = outputs[1][0]
            prediction = prob_dict.get(1, 0.5)
        else:
            prediction = float(outputs[0][0])
        
        predictions.append(prediction)
        
        print(f"\n{name}:")
        print(f"  NaN/Inf: {nan_count}/{inf_count}")
        print(f"  予測値: {prediction:.6f}")
        print(f"  方向: {'BUY' if prediction > 0.5 else 'SELL'}")
    
    # 3. 予測値の統計
    print("\n\n📊 予測値の統計:")
    predictions = np.array(predictions)
    print(f"  最小値: {np.min(predictions):.6f}")
    print(f"  最大値: {np.max(predictions):.6f}")
    print(f"  平均値: {np.mean(predictions):.6f}")
    print(f"  標準偏差: {np.std(predictions):.6f}")
    print(f"  変動幅: {np.max(predictions) - np.min(predictions):.6f}")
    
    # 4. 特定の特徴量の影響を調査
    print("\n\n🔬 特徴量の影響度調査:")
    base_features = mean.copy()
    base_normalized = (base_features - mean) / std_safe
    base_pred = session.run(None, {'input': base_normalized.reshape(1, -1).astype(np.float32)})
    
    if len(base_pred) > 1 and isinstance(base_pred[1], list):
        base_value = base_pred[1][0].get(1, 0.5)
    else:
        base_value = float(base_pred[0][0])
    
    print(f"ベースライン予測: {base_value:.6f}")
    
    # 各特徴量を個別に変更して影響を確認
    influences = []
    for i in range(44):
        if std[i] == 0:
            continue
            
        # +2stdの変更
        test_features = base_features.copy()
        test_features[i] += 2 * std[i]
        test_normalized = (test_features - mean) / std_safe
        test_pred = session.run(None, {'input': test_normalized.reshape(1, -1).astype(np.float32)})
        
        if len(test_pred) > 1 and isinstance(test_pred[1], list):
            test_value = test_pred[1][0].get(1, 0.5)
        else:
            test_value = float(test_pred[0][0])
        
        influence = abs(test_value - base_value)
        influences.append((i, influence))
    
    # 影響度の大きい特徴量トップ5
    influences.sort(key=lambda x: x[1], reverse=True)
    print("\n影響度の大きい特徴量トップ5:")
    for idx, (feat_idx, influence) in enumerate(influences[:5]):
        print(f"  {idx+1}. 特徴量{feat_idx}: 影響度 {influence:.6f}")
    
    # 5. 結論
    print("\n\n💡 分析結果:")
    if np.max(predictions) - np.min(predictions) < 0.1:
        print("❌ モデルの出力範囲が極端に狭い（< 0.1）")
        print("   → モデルが適切に学習されていない可能性")
    
    if all(p < 0.5 for p in predictions):
        print("❌ すべてのテストケースでSELL予測")
        print("   → モデルに強いバイアスがある")
    
    if zero_std > 0 or tiny_std > 0:
        print(f"⚠️  標準偏差が異常な特徴量が存在（ゼロ: {zero_std}個, 極小: {tiny_std}個）")
        print("   → スケーラーの計算に問題がある可能性")

if __name__ == "__main__":
    investigate_bias()