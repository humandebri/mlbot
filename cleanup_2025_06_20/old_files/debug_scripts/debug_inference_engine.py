#!/usr/bin/env python3
"""
InferenceEngine完全デバッグ - 全プロセス可視化
"""

import sys
import os
import numpy as np
import json
import asyncio
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def complete_inference_debug():
    """InferenceEngineの完全デバッグと全プロセス可視化"""
    
    print("🔍 InferenceEngine完全デバッグ開始")
    print("=" * 80)
    
    # 1. 環境確認
    print("\n1. 環境・ファイル確認")
    print("-" * 40)
    
    model_path = "models/v3.1_improved/model.onnx"
    scaler_path = "models/v3.1_improved/manual_scaler.json"
    
    print(f"Model exists: {os.path.exists(model_path)}")
    print(f"Scaler exists: {os.path.exists(scaler_path)}")
    
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path)
        print(f"Model size: {model_size:,} bytes")
    
    # 2. InferenceEngineを直接テスト
    print("\n2. InferenceEngine直接テスト")
    print("-" * 40)
    
    try:
        from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
        
        # 設定作成
        config = InferenceConfig(
            model_path=model_path,
            enable_thompson_sampling=False,
            confidence_threshold=0.1,
            providers=["CPUExecutionProvider"]
        )
        
        # エンジン初期化
        engine = InferenceEngine(config)
        print("✅ InferenceEngine初期化成功")
        
        # モデル読み込み
        engine.load_model(model_path)
        print("✅ モデル読み込み成功")
        
        # 3. 技術的指標生成
        print("\n3. 技術的指標生成")
        print("-" * 40)
        
        from src.feature_hub.technical_indicators import TechnicalIndicatorEngine
        from src.ml_pipeline.feature_adapter_44 import FeatureAdapter44
        
        tech_engine = TechnicalIndicatorEngine()
        adapter = FeatureAdapter44()
        
        # 複数の価格データで履歴構築
        prices = [
            (106000, 106500, 105500, 106250, 1000000),
            (106250, 106800, 105800, 106600, 1100000),
            (106600, 107000, 106200, 106800, 1200000),
            (106800, 107200, 106400, 107000, 1300000),
            (107000, 107500, 106700, 107200, 1400000),
        ]
        
        for i, (open_p, high, low, close, volume) in enumerate(prices):
            tech_engine.update_price_data("BTCUSDT", open_p, high, low, close, volume)
        
        features = tech_engine.get_latest_features("BTCUSDT")
        print(f"✅ 技術的指標生成: {len(features)}個")
        
        # 重要な指標値確認
        key_features = ["returns", "vol_20", "rsi_14", "macd", "bb_position_20"]
        print("\n重要指標値:")
        for key in key_features:
            value = features.get(key, "MISSING")
            print(f"  {key}: {value}")
        
        # 4. 特徴量変換
        print("\n4. 特徴量変換（44次元）")
        print("-" * 40)
        
        adapted = adapter.adapt(features)
        print(f"✅ 変換完了: {adapted.shape}")
        print(f"非ゼロ値: {np.count_nonzero(adapted)}/44")
        print(f"値範囲: [{np.min(adapted):.3f}, {np.max(adapted):.3f}]")
        
        # 5. InferenceEngine予測
        print("\n5. InferenceEngine予測（完全トレース）")
        print("-" * 40)
        
        # Dictで渡す（実際のシステムと同じ）
        feature_dict = {f"feature_{i}": float(adapted[i]) for i in range(len(adapted))}
        
        print("📥 InferenceEngineに送信...")
        print(f"  特徴量タイプ: {type(feature_dict)}")
        print(f"  特徴量数: {len(feature_dict)}")
        
        # 予測実行
        result = engine.predict(feature_dict, return_confidence=True, use_cache=False)
        
        print("📤 InferenceEngine返り値（RAW）:")
        print(f"  返り値タイプ: {type(result)}")
        print(f"  キー一覧: {list(result.keys()) if isinstance(result, dict) else 'NOT_DICT'}")
        
        for key, value in result.items():
            print(f"  {key}: {type(value)} = {value}")
        
        # 6. 返り値詳細分析
        print("\n6. 返り値詳細分析")
        print("-" * 40)
        
        # 各フィールドを詳しく確認
        predictions = result.get("predictions")
        raw_predictions = result.get("raw_predictions") 
        confidence_scores = result.get("confidence_scores")
        
        print(f"predictions: {type(predictions)} = {predictions}")
        print(f"raw_predictions: {type(raw_predictions)} = {raw_predictions}")
        print(f"confidence_scores: {type(confidence_scores)} = {confidence_scores}")
        
        # 7. 手動でONNX実行（比較用）
        print("\n7. 手動ONNX実行（比較用）")
        print("-" * 40)
        
        import onnxruntime as ort
        
        # スケーラー読み込み
        with open(scaler_path, 'r') as f:
            scaler_data = json.load(f)
        
        means = np.array(scaler_data['means'])
        stds = np.array(scaler_data['stds'])
        
        # 正規化
        normalized = (adapted - means) / stds
        normalized = np.clip(normalized, -5, 5)
        
        print(f"正規化後範囲: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
        
        # 直接ONNX実行
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        onnx_result = session.run([output_name], {input_name: normalized.reshape(1, -1).astype(np.float32)})
        direct_output = onnx_result[0][0]
        
        print(f"直接ONNX出力: {direct_output}")
        print(f"出力タイプ: {type(direct_output)}")
        
        # 8. 問題診断
        print("\n8. 問題診断")
        print("-" * 40)
        
        issues = []
        
        if direct_output == 0:
            issues.append("🚨 ONNX直接実行で0が返される - モデル自体に問題")
        
        if predictions is None:
            issues.append("🚨 InferenceEngine.predictionsがNone")
        
        if raw_predictions is None:
            issues.append("🚨 InferenceEngine.raw_predictionsがNone") 
        
        if confidence_scores is None:
            issues.append("🚨 InferenceEngine.confidence_scoresがNone")
        
        # InferenceEngineと直接ONNXの比較
        if raw_predictions is not None and len(raw_predictions) > 0:
            ie_output = raw_predictions[0]
            if abs(float(ie_output) - float(direct_output)) > 0.0001:
                issues.append(f"🚨 InferenceEngine({ie_output}) vs 直接ONNX({direct_output})の不一致")
        
        if issues:
            print("❌ 発見された問題:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("✅ 大きな問題は発見されませんでした")
        
        # 9. 修正提案
        print("\n9. 修正提案")
        print("-" * 40)
        
        if direct_output == 0:
            print("💡 モデル修復方法:")
            print("  - 異なるモデルファイルを試す")
            print("  - 特徴量の正規化方法を確認")
            print("  - モデル訓練時のデータ形式を確認")
        
        if raw_predictions is None or len(raw_predictions) == 0:
            print("💡 InferenceEngine修復方法:")
            print("  - InferenceEngine.predictメソッドの返り値構造を確認")
            print("  - confidence計算ロジックを確認")
        
        # 10. 代替案
        print("\n10. 代替シグナル生成（技術的指標ベース）")
        print("-" * 40)
        
        rsi = features.get("rsi_14", 50)
        macd = features.get("macd", 0)
        vol = features.get("vol_20", 0.01)
        bb_pos = features.get("bb_position_20", 0)
        
        print(f"RSI: {rsi:.2f}")
        print(f"MACD: {macd:.2f}")
        print(f"Volatility: {vol:.4f}")
        print(f"BB Position: {bb_pos:.2f}")
        
        # 技術的指標ベースのシグナル
        signal_strength = 0
        signal_direction = "HOLD"
        
        if rsi > 70:
            signal_strength += 0.3
            signal_direction = "SELL"
            print("  🔴 RSI過買い → SELL信号")
        elif rsi < 30:
            signal_strength += 0.3 
            signal_direction = "BUY"
            print("  🟢 RSI過売り → BUY信号")
        
        if abs(macd) > 50:
            signal_strength += 0.2
            if macd > 0 and signal_direction != "SELL":
                signal_direction = "BUY"
                print("  🟢 MACD強気 → BUY信号")
            elif macd < 0 and signal_direction != "BUY":
                signal_direction = "SELL"
                print("  🔴 MACD弱気 → SELL信号")
        
        if vol > 0.02:
            signal_strength += 0.2
            print("  ⚡ 高ボラティリティ → 信号強度向上")
        
        if signal_strength > 0.4:
            print(f"\n🎯 技術的指標シグナル: {signal_direction} (強度: {signal_strength:.2f})")
        else:
            print(f"\n⏸️  技術的指標シグナル: {signal_direction} (強度: {signal_strength:.2f} - 弱い)")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("完全デバッグ終了")

if __name__ == "__main__":
    asyncio.run(complete_inference_debug())