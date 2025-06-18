#!/usr/bin/env python3
"""
高性能復元モデル統合テスト
- 新しいbalanced_restored_26dモデルの読み込み確認
- FeatureAdapter26との互換性確認
- 予測機能の動作確認
"""

import sys
import asyncio
from pathlib import Path
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_model_integration():
    """統合テスト実行"""
    print("="*70)
    print("🧪 高性能復元モデル統合テスト")
    print("="*70)
    
    try:
        # 1. モデルファイル存在確認
        print("\n1️⃣ モデルファイル確認中...")
        
        model_path = Path("models/balanced_restored_26d/model.onnx")
        scaler_path = Path("models/balanced_restored_26d/scaler.pkl")
        metadata_path = Path("models/balanced_restored_26d/metadata.json")
        
        if not model_path.exists():
            print(f"❌ モデルファイルが見つかりません: {model_path}")
            return False
        
        if not scaler_path.exists():
            print(f"❌ スケーラーファイルが見つかりません: {scaler_path}")
            return False
        
        if not metadata_path.exists():
            print(f"❌ メタデータファイルが見つかりません: {metadata_path}")
            return False
        
        print(f"✅ モデルファイル確認完了:")
        print(f"   - {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
        print(f"   - {scaler_path} ({scaler_path.stat().st_size / 1024:.1f} KB)")
        print(f"   - {metadata_path} ({metadata_path.stat().st_size / 1024:.1f} KB)")
        
        # 2. メタデータ確認
        print("\n2️⃣ メタデータ確認中...")
        
        import json
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"✅ メタデータ:")
        print(f"   - モデル型: {metadata['model_type']}")
        print(f"   - バージョン: {metadata['model_version']}")
        print(f"   - 特徴量数: {metadata['feature_count']}")
        print(f"   - AUC: {metadata['performance']['auc_mean']:.3f}")
        print(f"   - ONNX変換: {metadata['onnx_converted']}")
        
        # 3. InferenceEngine初期化テスト
        print("\n3️⃣ InferenceEngine初期化テスト...")
        
        from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
        
        config = InferenceConfig(
            model_path=str(model_path),
            preprocessor_path=str(scaler_path),
            max_inference_time_ms=100.0,
            batch_size=32,
            providers=["CPUExecutionProvider"]
        )
        
        engine = InferenceEngine(config)
        engine.load_model(str(model_path))
        
        print(f"✅ InferenceEngine初期化成功")
        print(f"   - モデルパス: {model_path}")
        print(f"   - 入力次元: 26次元期待")
        
        # 4. FeatureAdapterRestored26Dテスト
        print("\n4️⃣ FeatureAdapterRestored26Dテスト...")
        
        from src.ml_pipeline.feature_adapter_restored_26d import FeatureAdapterRestored26D
        
        adapter = FeatureAdapterRestored26D()
        
        # サンプル特徴量（多次元）
        sample_features = {
            'close': 70000.0,
            'open': 69500.0,
            'high': 70200.0,
            'low': 69300.0,
            'volume': 1500000.0,
            'returns': 0.0072,
            'log_returns': 0.0071,
            'hl_ratio': 0.0013,
            'oc_ratio': 0.0072,
            'return_1': 0.0024,
            'return_3': 0.0051,
            'return_5': 0.0089,
            'return_10': 0.0156,
            'vol_5': 0.0145,
            'vol_10': 0.0142,
            'vol_20': 0.0138,
            'rsi': 65.4,
            'bb_position': 0.67,
            'volume_ratio': 1.34,
            'momentum_3': 0.0051,
            'momentum_5': 0.0089,
            # Extra features (should be ignored)
            'extra_feature_1': 123.45,
            'extra_feature_2': 678.90,
            'irrelevant_data': 'ignored'
        }
        
        adapted_features = adapter.adapt(sample_features)
        stats = adapter.get_adaptation_stats(sample_features)
        
        print(f"✅ FeatureAdapterRestored26D動作確認:")
        print(f"   - 入力特徴量数: {len(sample_features)}")
        print(f"   - 出力次元: {adapted_features.shape}")
        print(f"   - マッチ率: {stats['match_rate']:.1%}")
        print(f"   - 適合特徴量: {stats['matched_features']}/{stats['target_features']}")
        print(f"   - 導出特徴量: {stats.get('derived_features', 0)}")
        
        if adapted_features.shape[0] != 26:
            print(f"❌ 出力次元エラー: 期待26次元、実際{adapted_features.shape[0]}次元")
            return False
        
        # 5. 予測実行テスト
        print("\n5️⃣ 予測実行テスト...")
        
        # FeatureAdapterRestored26Dで変換した特徴量を使用
        feature_array = adapted_features.reshape(1, -1).astype(np.float32)
        
        prediction = engine.predict(feature_array)
        
        print(f"✅ 予測実行成功:")
        print(f"   - 入力形状: {feature_array.shape}")
        print(f"   - 予測型: {type(prediction)}")
        
        # InferenceEngineは辞書形式で結果を返す
        if isinstance(prediction, dict):
            pred_array = prediction.get('predictions', [])
            confidence_scores = prediction.get('confidence_scores', [])
            inference_time = prediction.get('inference_time_ms', 0)
            
            if len(pred_array) > 0:
                pred_value = float(pred_array[0])
                confidence = float(confidence_scores[0]) if len(confidence_scores) > 0 else 0.5
                
                print(f"   - 予測値: {pred_value:.4f}")
                print(f"   - 信頼度: {confidence:.1%}")
                print(f"   - 推論時間: {inference_time:.2f}ms")
                
                if confidence > 0.8:
                    print(f"   🎯 高信頼度シグナル (>80%)")
                elif confidence > 0.6:
                    print(f"   📊 中信頼度シグナル (>60%)")
                else:
                    print(f"   📋 低信頼度 (<60%)")
            else:
                print("   ❌ 予測値が空です")
        else:
            print(f"   ⚠️ 予期しない予測形式: {prediction}")
        
        # 6. 統合テスト（DynamicTradingCoordinator風）
        print("\n6️⃣ 統合シミュレーション...")
        
        # 複数の特徴量パターンでテスト
        test_cases = [
            {"close": 70000, "returns": 0.005, "vol_20": 0.015, "rsi": 70},  # 上昇トレンド
            {"close": 69000, "returns": -0.003, "vol_20": 0.020, "rsi": 30}, # 下降トレンド
            {"close": 69500, "returns": 0.001, "vol_20": 0.010, "rsi": 50},  # レンジ
        ]
        
        scenarios = ["上昇トレンド", "下降トレンド", "レンジ相場"]
        
        for i, (test_case, scenario) in enumerate(zip(test_cases, scenarios)):
            # 基本特徴量から完全な特徴量セットを構築
            full_features = {**sample_features, **test_case}
            
            adapted = adapter.adapt(full_features)
            input_array = adapted.reshape(1, -1).astype(np.float32)
            pred = engine.predict(input_array)
            
            if isinstance(pred, dict):
                pred_array = pred.get('predictions', [])
                confidence_scores = pred.get('confidence_scores', [])
                pred_val = float(pred_array[0]) if len(pred_array) > 0 else 0.0
                confidence = float(confidence_scores[0]) if len(confidence_scores) > 0 else 0.5
            else:
                pred_val = pred[0] if isinstance(pred, (list, np.ndarray)) else pred
                confidence = 1 / (1 + np.exp(-5 * (pred_val - 0.5)))
            
            print(f"   {i+1}. {scenario}: 予測={pred_val:.3f}, 信頼度={confidence:.1%}")
        
        # 7. 最終確認
        print("\n7️⃣ 最終確認...")
        
        success_checks = [
            model_path.exists(),
            metadata['feature_count'] == 26,
            metadata['onnx_converted'] == True,
            adapted_features.shape[0] == 26,
            stats['match_rate'] > 0.5,
            prediction is not None
        ]
        
        if all(success_checks):
            print("✅ 全テストPASS！高性能復元モデル統合完了")
            print(f"\n🎯 パフォーマンス期待値:")
            print(f"   - AUC: {metadata['performance']['auc_mean']:.3f} (目標0.867の{metadata['performance']['auc_mean']/0.867:.1%})")
            print(f"   - 現行システム比: +{(metadata['performance']['auc_mean']-0.700)/0.700:.1%}改善")
            print(f"   - デプロイ準備: ✅ 完了")
            
            return True
        else:
            print("❌ 一部テスト失敗")
            return False
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    try:
        result = asyncio.run(test_model_integration())
        
        if result:
            print("\n" + "="*70)
            print("🎉 高性能復元モデル統合テスト成功")
            print("📝 次のステップ:")
            print("1. EC2での動作確認")
            print("2. ライブ取引での性能検証")
            print("3. 信頼度閾値の最適化（推奨75-85%）")
            print("="*70)
        else:
            print("\n❌ 統合テスト失敗。問題を修正してから再実行してください。")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ テスト中断")
    except Exception as e:
        print(f"\nエラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()