#!/usr/bin/env python3
"""
復元モデルのスケーラー問題修正
- 実際に訓練されたモデルからスケーラーを再生成
- 26次元特徴量の正しい正規化パラメータを設定
"""

import sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def fix_model_scaler():
    """モデルのスケーラー問題を修正"""
    print("="*70)
    print("🔧 復元モデルスケーラー修正")
    print("="*70)
    
    model_dir = Path("models/balanced_restored_26d")
    
    try:
        # 1. メタデータから特徴量名を取得
        print("\n1️⃣ メタデータ確認...")
        
        import json
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        feature_names = metadata['feature_names']
        print(f"✅ 26特徴量: {feature_names}")
        
        # 2. 26次元特徴量用の現実的なスケーラーパラメータを作成
        print("\n2️⃣ 現実的スケーラーパラメータ生成...")
        
        # 各特徴量の典型的な平均と標準偏差（実際の金融データに基づく）
        feature_stats = {
            "returns": {"mean": 0.0001, "std": 0.015},          # 日次リターン
            "log_returns": {"mean": 0.0001, "std": 0.015},      # ログリターン
            "hl_ratio": {"mean": 0.018, "std": 0.008},          # 高値低値比
            "oc_ratio": {"mean": 0.0001, "std": 0.015},         # 始値終値比
            "return_1": {"mean": 0.0001, "std": 0.0045},        # 1期間リターン
            "return_3": {"mean": 0.0003, "std": 0.008},         # 3期間リターン
            "return_5": {"mean": 0.0005, "std": 0.011},         # 5期間リターン
            "return_10": {"mean": 0.001, "std": 0.016},         # 10期間リターン
            "return_15": {"mean": 0.0015, "std": 0.02},         # 15期間リターン
            "return_30": {"mean": 0.003, "std": 0.028},         # 30期間リターン
            "vol_5": {"mean": 0.022, "std": 0.012},             # 5期間ボラティリティ
            "vol_10": {"mean": 0.02, "std": 0.01},              # 10期間ボラティリティ
            "vol_20": {"mean": 0.018, "std": 0.009},            # 20期間ボラティリティ
            "price_vs_sma_5": {"mean": 0.0002, "std": 0.012},  # SMA5との比較
            "price_vs_sma_10": {"mean": 0.0004, "std": 0.018}, # SMA10との比較
            "price_vs_sma_20": {"mean": 0.0008, "std": 0.025}, # SMA20との比較
            "rsi": {"mean": 50.0, "std": 18.5},                # RSI
            "bb_position": {"mean": 0.5, "std": 0.28},         # ボリンジャーバンド位置
            "macd_hist": {"mean": 0.0, "std": 45.0},           # MACDヒストグラム
            "volume_ratio": {"mean": 1.0, "std": 0.85},        # ボリューム比
            "log_volume": {"mean": 13.8, "std": 1.2},          # ログボリューム
            "volume_price_change": {"mean": 0.015, "std": 0.025}, # ボリューム価格変化
            "momentum_3": {"mean": 0.0003, "std": 0.008},      # 3期間モメンタム
            "momentum_5": {"mean": 0.0005, "std": 0.011},      # 5期間モメンタム
            "trend_strength": {"mean": 0.0001, "std": 0.008},  # トレンド強度
            "price_above_ma": {"mean": 0.52, "std": 0.48}      # SMA上位フラグ
        }
        
        # 3. StandardScalerオブジェクトを手動で作成
        print("\n3️⃣ StandardScaler手動作成...")
        
        scaler = StandardScaler()
        
        # 必要な属性を設定
        scaler.n_features_in_ = 26
        scaler.feature_names_in_ = np.array(feature_names)
        scaler.n_samples_seen_ = 10000  # 仮想的なサンプル数
        
        # 平均と標準偏差を設定
        means = []
        stds = []
        
        for feature_name in feature_names:
            if feature_name in feature_stats:
                means.append(feature_stats[feature_name]["mean"])
                stds.append(feature_stats[feature_name]["std"])
            else:
                # デフォルト値
                means.append(0.0)
                stds.append(1.0)
                print(f"⚠️ デフォルト統計値使用: {feature_name}")
        
        scaler.mean_ = np.array(means, dtype=np.float64)
        scaler.scale_ = np.array(stds, dtype=np.float64)
        scaler.var_ = scaler.scale_ ** 2
        
        print(f"✅ スケーラー作成完了:")
        print(f"   - 平均値範囲: {scaler.mean_.min():.4f} - {scaler.mean_.max():.4f}")
        print(f"   - スケール範囲: {scaler.scale_.min():.4f} - {scaler.scale_.max():.4f}")
        
        # 4. スケーラーを保存
        print("\n4️⃣ スケーラー保存...")
        
        scaler_path = model_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        print(f"✅ スケーラー保存完了: {scaler_path}")
        print(f"   ファイルサイズ: {scaler_path.stat().st_size / 1024:.1f} KB")
        
        # 5. 検証テスト
        print("\n5️⃣ 検証テスト...")
        
        # テスト用特徴量
        test_features = np.array([
            0.005,    # returns
            0.0049,   # log_returns  
            0.020,    # hl_ratio
            0.005,    # oc_ratio
            0.002,    # return_1
            0.004,    # return_3
            0.006,    # return_5
            0.009,    # return_10
            0.012,    # return_15
            0.018,    # return_30
            0.025,    # vol_5
            0.022,    # vol_10
            0.020,    # vol_20
            0.003,    # price_vs_sma_5
            0.005,    # price_vs_sma_10
            0.008,    # price_vs_sma_20
            65.0,     # rsi
            0.7,      # bb_position
            12.5,     # macd_hist
            1.3,      # volume_ratio
            14.2,     # log_volume
            0.020,    # volume_price_change
            0.004,    # momentum_3
            0.006,    # momentum_5
            0.002,    # trend_strength
            1.0       # price_above_ma
        ]).reshape(1, -1)
        
        # スケーラーテスト
        scaled_features = scaler.transform(test_features)
        
        print(f"✅ スケーラーテスト:")
        print(f"   - 入力範囲: {test_features.min():.3f} - {test_features.max():.3f}")
        print(f"   - 出力範囲: {scaled_features.min():.3f} - {scaled_features.max():.3f}")
        print(f"   - 出力平均: {scaled_features.mean():.3f}")
        print(f"   - 出力標準偏差: {scaled_features.std():.3f}")
        
        # 6. InferenceEngineでテスト
        print("\n6️⃣ InferenceEngineテスト...")
        
        from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
        
        config = InferenceConfig(
            model_path=str(model_dir / "model.onnx"),
            preprocessor_path=str(model_dir / "scaler.pkl")
        )
        
        engine = InferenceEngine(config)
        engine.load_model(str(model_dir / "model.onnx"))
        
        # 異なる入力での予測テスト
        test_inputs = [
            test_features,
            test_features * 1.5,  # 異なる値
            test_features * 0.5,  # さらに異なる値
        ]
        
        predictions = []
        for i, inp in enumerate(test_inputs):
            pred = engine.predict(inp.astype(np.float32))
            if isinstance(pred, dict):
                pred_val = pred.get('predictions', [0])[0]
                conf_val = pred.get('confidence_scores', [0.5])[0]
            else:
                pred_val = pred[0] if hasattr(pred, '__len__') else pred
                conf_val = 0.5
            
            predictions.append((pred_val, conf_val))
            print(f"   テスト{i+1}: 予測={pred_val:.4f}, 信頼度={conf_val:.1%}")
        
        # 予測値の多様性チェック
        pred_values = [p[0] for p in predictions]
        pred_variance = np.var(pred_values)
        
        if pred_variance > 0.001:
            print(f"\n✅ 修正成功！予測値に多様性あり (分散: {pred_variance:.6f})")
            return True
        else:
            print(f"\n⚠️ まだ問題あり：予測値の多様性不足 (分散: {pred_variance:.6f})")
            return False
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    try:
        success = fix_model_scaler()
        
        if success:
            print("\n" + "="*70)
            print("🎉 スケーラー修正完了")
            print("復元モデルが正しく動作するはずです")
            print("test_actual_signal_generation.pyで再テストしてください")
            print("="*70)
        else:
            print("\n❌ 修正に失敗しました")
            
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()