#!/usr/bin/env python3
"""
特徴量の分布とモデル予測の偏りを調査
"""

import numpy as np
import onnxruntime as ort
import json
import redis
from improved_feature_generator import ImprovedFeatureGeneratorEnhanced
import asyncio
from datetime import datetime

async def analyze_predictions():
    """予測の偏りを分析"""
    
    # 1. Feature Generatorを初期化
    print("🔍 Feature Generatorを初期化...")
    feature_gen = ImprovedFeatureGeneratorEnhanced(
        db_path="data/historical_data.duckdb",
        enable_redis=True
    )
    
    # 2. モデルとスケーラーを読み込み
    print("\n📊 モデルとスケーラーを読み込み...")
    model_path = "models/v3.1_improved/model.onnx"
    scaler_path = "models/v3.1_improved/manual_scaler.json"
    
    session = ort.InferenceSession(model_path)
    
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    
    # 3. 各シンボルで特徴量を生成して予測
    symbols = ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']
    
    for symbol in symbols:
        print(f"\n=== {symbol} ===")
        
        # 特徴量を生成
        features = await feature_gen.get_features(symbol)
        
        if features is None:
            print(f"❌ {symbol}の特徴量生成失敗")
            continue
        
        # 特徴量の統計を表示
        features_array = np.array(features)
        print(f"特徴量数: {len(features)}")
        print(f"特徴量の統計:")
        print(f"  最小値: {np.min(features_array):.4f}")
        print(f"  最大値: {np.max(features_array):.4f}")
        print(f"  平均値: {np.mean(features_array):.4f}")
        print(f"  標準偏差: {np.std(features_array):.4f}")
        
        # 正規化前の特徴量のサンプル
        print(f"\n正規化前の特徴量サンプル（最初の10個）:")
        print(features_array[:10])
        
        # 正規化
        mean = np.array(scaler_params['mean'])
        std = np.array(scaler_params['std'])
        
        # ゼロ除算を避ける
        std_safe = np.where(std == 0, 1.0, std)
        normalized_features = (features_array - mean) / std_safe
        
        print(f"\n正規化後の特徴量統計:")
        print(f"  最小値: {np.min(normalized_features):.4f}")
        print(f"  最大値: {np.max(normalized_features):.4f}")
        print(f"  平均値: {np.mean(normalized_features):.4f}")
        print(f"  標準偏差: {np.std(normalized_features):.4f}")
        
        # 異常な値をチェック
        nan_count = np.isnan(normalized_features).sum()
        inf_count = np.isinf(normalized_features).sum()
        print(f"\n異常値チェック:")
        print(f"  NaN数: {nan_count}")
        print(f"  Inf数: {inf_count}")
        
        # モデル予測
        input_data = normalized_features.reshape(1, -1).astype(np.float32)
        outputs = session.run(None, {'input': input_data})
        
        print(f"\nモデル出力:")
        print(f"  outputs数: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  output[{i}] shape: {output.shape if hasattr(output, 'shape') else type(output)}")
            print(f"  output[{i}] value: {output}")
        
        # 予測値を抽出
        if len(outputs) > 1 and isinstance(outputs[1], list) and len(outputs[1]) > 0:
            prob_dict = outputs[1][0]
            prediction = prob_dict.get(1, 0.5)
        else:
            prediction = float(outputs[0][0])
        
        print(f"\n最終予測値: {prediction:.4f}")
        
    # 4. スケーラーパラメータの分析
    print("\n\n📊 スケーラーパラメータの分析:")
    print(f"mean長さ: {len(mean)}")
    print(f"std長さ: {len(std)}")
    
    # ゼロのstdをチェック
    zero_std_indices = np.where(std == 0)[0]
    print(f"\nstdがゼロの特徴量インデックス: {zero_std_indices}")
    print(f"ゼロstdの数: {len(zero_std_indices)}")
    
    # 極端に小さいstdをチェック
    small_std_indices = np.where((std > 0) & (std < 0.0001))[0]
    print(f"\n極端に小さいstd (<0.0001) のインデックス: {small_std_indices}")
    print(f"小さいstdの数: {len(small_std_indices)}")

if __name__ == "__main__":
    asyncio.run(analyze_predictions())