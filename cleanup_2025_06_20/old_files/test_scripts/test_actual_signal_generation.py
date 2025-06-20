#!/usr/bin/env python3
"""
実際のシグナル生成テスト
- 復元モデルが実際に異なる予測値を返すかテスト
- 多様な市場条件での動作確認
- 現実的なシグナル生成頻度の検証
"""

import sys
import asyncio
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_real_signal_generation():
    """実際のシグナル生成能力をテスト"""
    print("="*70)
    print("🔬 実際のシグナル生成テスト")
    print("="*70)
    
    try:
        # 1. モデルとアダプター初期化
        print("\n1️⃣ モデル初期化...")
        
        from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig
        from src.ml_pipeline.feature_adapter_restored_26d import FeatureAdapterRestored26D
        
        config = InferenceConfig(
            model_path="models/balanced_restored_26d/model.onnx",
            preprocessor_path="models/balanced_restored_26d/scaler.pkl"
        )
        
        engine = InferenceEngine(config)
        engine.load_model("models/balanced_restored_26d/model.onnx")
        adapter = FeatureAdapterRestored26D()
        
        print("✅ モデル・アダプター初期化完了")
        
        # 2. 多様な市場条件シミュレーション
        print("\n2️⃣ 多様な市場条件でのテスト...")
        
        # より現実的で多様な市場シナリオ
        market_scenarios = [
            {
                "name": "強い上昇トレンド",
                "returns": 0.025,      # 2.5%上昇
                "vol_20": 0.035,       # 高ボラティリティ
                "rsi": 75,             # 買われ過ぎ
                "volume_ratio": 2.5,   # 高出来高
                "bb_position": 0.9,    # 上部バンド付近
                "trend_strength": 0.015 # 強いトレンド
            },
            {
                "name": "急激な下落",
                "returns": -0.031,     # 3.1%下落
                "vol_20": 0.045,       # 非常に高いボラティリティ
                "rsi": 25,             # 売られ過ぎ
                "volume_ratio": 3.2,   # パニック売り
                "bb_position": 0.1,    # 下部バンド付近
                "trend_strength": -0.022
            },
            {
                "name": "レンジ相場（安定）",
                "returns": 0.002,      # わずかな上昇
                "vol_20": 0.012,       # 低ボラティリティ
                "rsi": 52,             # 中立
                "volume_ratio": 0.8,   # 低出来高
                "bb_position": 0.45,   # 中央付近
                "trend_strength": 0.001
            },
            {
                "name": "ボラティリティ爆発",
                "returns": 0.008,      # 小さな上昇
                "vol_20": 0.055,       # 極めて高いボラティリティ
                "rsi": 65,             # やや強気
                "volume_ratio": 4.1,   # 極めて高い出来高
                "bb_position": 0.75,   # 上部寄り
                "trend_strength": 0.005
            },
            {
                "name": "反転シグナル",
                "returns": -0.015,     # 下落
                "vol_20": 0.028,       # 中程度のボラティリティ  
                "rsi": 35,             # 売られ過ぎに近い
                "volume_ratio": 1.8,   # やや高出来高
                "bb_position": 0.25,   # 下部寄り
                "trend_strength": -0.008
            },
            {
                "name": "微妙な上昇",
                "returns": 0.006,      # 小さな上昇
                "vol_20": 0.018,       # やや低ボラティリティ
                "rsi": 58,             # やや強気
                "volume_ratio": 1.2,   # 平均的出来高
                "bb_position": 0.62,   # やや上部
                "trend_strength": 0.004
            }
        ]
        
        # ベース特徴量（BTC価格ベース）
        base_features = {
            'close': 70000.0,
            'open': 69800.0,
            'high': 70500.0,
            'low': 69200.0,
            'volume': 1200000.0,
            'log_returns': 0.0,
            'hl_ratio': 0.0186,
            'oc_ratio': 0.0029,
            'return_1': 0.003,
            'return_3': 0.008,
            'return_5': 0.012,
            'return_10': 0.015,
            'return_15': 0.018,
            'return_30': 0.025,
            'vol_5': 0.022,
            'vol_10': 0.019,
            'price_vs_sma_5': 0.002,
            'price_vs_sma_10': 0.005,
            'price_vs_sma_20': 0.008,
            'macd_hist': 15.2,
            'log_volume': 13.996,
            'volume_price_change': 0.025,
            'momentum_3': 0.008,
            'momentum_5': 0.012,
            'price_above_ma': 1.0
        }
        
        predictions = []
        
        for scenario in market_scenarios:
            # シナリオに応じて特徴量を調整
            test_features = base_features.copy()
            test_features.update(scenario)
            
            # 削除: nameキー（特徴量ではない）
            scenario_name = test_features.pop('name')
            
            # 特徴量を適応・予測
            adapted = adapter.adapt(test_features)
            input_array = adapted.reshape(1, -1).astype(np.float32)
            pred_result = engine.predict(input_array)
            
            if isinstance(pred_result, dict):
                pred_value = float(pred_result.get('predictions', [0])[0])
                confidence = float(pred_result.get('confidence_scores', [0.5])[0])
            else:
                pred_value = float(pred_result[0]) if hasattr(pred_result, '__len__') else float(pred_result)
                confidence = 1 / (1 + np.exp(-5 * (pred_value - 0.5)))
            
            predictions.append({
                'scenario': scenario_name,
                'prediction': pred_value,
                'confidence': confidence,
                'key_features': {
                    'returns': test_features['returns'],
                    'vol_20': test_features['vol_20'],
                    'rsi': test_features['rsi'],
                    'volume_ratio': test_features['volume_ratio']
                }
            })
            
            # シグナル判定
            signal_strength = "🔴 低" if confidence < 0.6 else "🟡 中" if confidence < 0.8 else "🟢 高"
            direction = "📈 BUY" if pred_value > 0.05 else "📉 SELL" if pred_value < -0.05 else "⏸️ HOLD"
            
            print(f"   {scenario_name}:")
            print(f"     予測値: {pred_value:.4f}, 信頼度: {confidence:.1%} {signal_strength}")
            print(f"     判定: {direction}")
        
        # 3. 予測値の分散確認
        print("\n3️⃣ 予測値分散分析...")
        
        pred_values = [p['prediction'] for p in predictions]
        conf_values = [p['confidence'] for p in predictions]
        
        print(f"✅ 予測値統計:")
        print(f"   - 最小値: {min(pred_values):.4f}")
        print(f"   - 最大値: {max(pred_values):.4f}")
        print(f"   - 平均値: {np.mean(pred_values):.4f}")
        print(f"   - 標準偏差: {np.std(pred_values):.4f}")
        print(f"   - 信頼度範囲: {min(conf_values):.1%} - {max(conf_values):.1%}")
        
        # 4. シグナル生成頻度分析
        print("\n4️⃣ シグナル生成頻度分析...")
        
        high_confidence_signals = sum(1 for c in conf_values if c > 0.8)
        medium_confidence_signals = sum(1 for c in conf_values if 0.6 <= c <= 0.8)
        tradeable_signals = sum(1 for p in pred_values if abs(p) > 0.03)  # 3%以上の予測
        
        print(f"📊 シグナル頻度 (6シナリオ中):")
        print(f"   - 高信頼度(>80%): {high_confidence_signals} ({high_confidence_signals/len(predictions):.1%})")
        print(f"   - 中信頼度(60-80%): {medium_confidence_signals} ({medium_confidence_signals/len(predictions):.1%})")
        print(f"   - 取引可能シグナル: {tradeable_signals} ({tradeable_signals/len(predictions):.1%})")
        
        # 5. 時系列データでのテスト
        print("\n5️⃣ 時系列データシミュレーション...")
        
        # 24時間の5分足データをシミュレート（288本）
        time_series_predictions = []
        base_price = 70000.0
        
        for i in range(24):  # 24時間分の代表データ
            # 時間帯による市場特性変化
            if 8 <= i <= 16:  # アジア・欧州時間
                vol_multiplier = 1.2
                volume_multiplier = 1.4
            elif 14 <= i <= 22:  # 欧州・米国時間
                vol_multiplier = 1.5
                volume_multiplier = 1.8
            else:  # 夜間
                vol_multiplier = 0.8
                volume_multiplier = 0.6
            
            # ランダムな市場変動
            price_change = np.random.normal(0, 0.015) * vol_multiplier
            volume_change = np.random.lognormal(0, 0.3) * volume_multiplier
            
            hourly_features = base_features.copy()
            hourly_features.update({
                'returns': price_change,
                'vol_20': 0.015 * vol_multiplier,
                'rsi': 50 + np.random.normal(0, 15),
                'volume_ratio': volume_change,
                'bb_position': 0.5 + np.random.normal(0, 0.2)
            })
            
            # RSIを0-100範囲に制限
            hourly_features['rsi'] = np.clip(hourly_features['rsi'], 0, 100)
            hourly_features['bb_position'] = np.clip(hourly_features['bb_position'], 0, 1)
            
            adapted = adapter.adapt(hourly_features)
            input_array = adapted.reshape(1, -1).astype(np.float32)
            pred_result = engine.predict(input_array)
            
            if isinstance(pred_result, dict):
                pred_value = float(pred_result.get('predictions', [0])[0])
                confidence = float(pred_result.get('confidence_scores', [0.5])[0])
            else:
                pred_value = float(pred_result[0]) if hasattr(pred_result, '__len__') else float(pred_result)
                confidence = 1 / (1 + np.exp(-5 * (pred_value - 0.5)))
            
            time_series_predictions.append({
                'hour': i,
                'prediction': pred_value,
                'confidence': confidence
            })
        
        # 時系列統計
        ts_pred_values = [p['prediction'] for p in time_series_predictions]
        ts_conf_values = [p['confidence'] for p in time_series_predictions]
        
        daily_high_conf_signals = sum(1 for c in ts_conf_values if c > 0.75)
        
        print(f"📈 24時間シミュレーション結果:")
        print(f"   - 予測値範囲: {min(ts_pred_values):.4f} - {max(ts_pred_values):.4f}")
        print(f"   - 信頼度範囲: {min(ts_conf_values):.1%} - {max(ts_conf_values):.1%}")
        print(f"   - 高信頼度シグナル(>75%): {daily_high_conf_signals}/24時間 ({daily_high_conf_signals/24:.1%})")
        
        # 6. 最終判定
        print("\n6️⃣ 実用性評価...")
        
        # 予測値の多様性チェック
        pred_variance = np.var(pred_values)
        conf_variance = np.var(conf_values)
        
        print(f"📊 モデル性能評価:")
        print(f"   - 予測値分散: {pred_variance:.6f} (目標 > 0.001)")
        print(f"   - 信頼度分散: {conf_variance:.6f} (目標 > 0.01)")
        
        # 実用性判定
        is_diverse = pred_variance > 0.001
        has_high_conf = any(c > 0.8 for c in conf_values)
        reasonable_signal_rate = daily_high_conf_signals <= 12  # 1日12回以下
        
        if is_diverse and has_high_conf and reasonable_signal_rate:
            print("\n✅ 実用性評価: 合格")
            print("   復元モデルは実際にシグナルを生成可能")
            print("   多様な市場条件に対応")
            print("   適切な頻度でシグナル生成")
            result = True
        else:
            print("\n❌ 実用性評価: 問題あり")
            if not is_diverse:
                print("   ⚠️ 予測値の多様性不足")
            if not has_high_conf:
                print("   ⚠️ 高信頼度シグナルなし")
            if not reasonable_signal_rate:
                print("   ⚠️ シグナル頻度過多")
            result = False
        
        return result
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    try:
        result = asyncio.run(test_real_signal_generation())
        
        if result:
            print("\n" + "="*70)
            print("🎉 実際のシグナル生成確認完了")
            print("復元モデルは実用的なシグナル生成能力を持っています")
            print("="*70)
        else:
            print("\n❌ シグナル生成に問題があります")
            print("モデルまたは特徴量処理の見直しが必要です")
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()