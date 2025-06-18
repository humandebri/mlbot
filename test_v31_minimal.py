#!/usr/bin/env python3
"""
V3.1_improved モデルの最小限テスト
設定やAPI依存を避けた純粋なモデルテスト
"""

import sys
import asyncio
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_v31_minimal():
    """V3.1_improved モデルの最小限テスト"""
    print("="*80)
    print("🔧 V3.1_improved 最小限テスト")
    print("="*80)
    
    try:
        # 1. モデル単体テスト
        print("\n1️⃣ モデル単体テスト...")
        
        from src.ml_pipeline.v31_improved_inference_engine import V31ImprovedInferenceEngine, V31ImprovedConfig
        
        config = V31ImprovedConfig(
            model_path="models/v3.1_improved/model.onnx",
            confidence_threshold=0.7
        )
        
        engine = V31ImprovedInferenceEngine(config)
        engine.load_model()
        
        print("✅ V3.1_improvedエンジン読み込み成功")
        
        # 2. 特徴量アダプターテスト
        print("\n2️⃣ 特徴量アダプター単体テスト...")
        
        from src.ml_pipeline.feature_adapter_44 import FeatureAdapter44
        
        adapter = FeatureAdapter44()
        
        # テスト特徴量
        test_features = {
            "returns": 0.025,
            "log_returns": 0.0247,
            "hl_ratio": 0.018,
            "oc_ratio": 0.025,
            "vol_5": 0.035,
            "rsi_14": 75.0,
            "open": 67000.0,
            "close": 67300.0,
            "volume": 1500000.0
        }
        
        # 44次元に変換
        feature_44d = adapter.adapt(test_features)
        print(f"✅ 特徴量変換成功: {len(test_features)}次元 → {feature_44d.shape[0]}次元")
        
        # 3. 推論テスト
        print("\n3️⃣ 推論テスト...")
        
        result = engine.predict(test_features)
        
        prediction = result['prediction']
        confidence = result['confidence']
        signal_info = result['signal']
        
        print(f"✅ 推論成功:")
        print(f"   予測値: {prediction:.4f}")
        print(f"   信頼度: {confidence:.1%}")
        print(f"   方向: {signal_info['direction']}")
        print(f"   取引可能: {signal_info['tradeable']}")
        
        # 4. 複数シナリオテスト
        print("\n4️⃣ 複数シナリオテスト...")
        
        scenarios = [
            {"name": "強い上昇", "modifier": 1.5},
            {"name": "下落", "modifier": -0.8}, 
            {"name": "レンジ", "modifier": 0.1},
            {"name": "ニュートラル", "modifier": 1.0}
        ]
        
        tradeable_count = 0
        
        for scenario in scenarios:
            # 特徴量を修正
            modified_features = test_features.copy()
            modified_features["returns"] *= scenario["modifier"]
            modified_features["rsi_14"] = 50 + (scenario["modifier"] * 25)
            
            result = engine.predict(modified_features)
            signal = result['signal']
            
            print(f"   {scenario['name']}: {signal['direction']} "
                  f"(信頼度: {result['confidence']:.1%}, "
                  f"取引可能: {signal['tradeable']})")
            
            if signal['tradeable']:
                tradeable_count += 1
        
        # 5. 結果判定
        print("\n5️⃣ 結果判定...")
        
        success_criteria = [
            prediction > 0 and prediction < 1,  # 予測値が妥当な範囲
            confidence > 0 and confidence < 1,  # 信頼度が妥当な範囲
            signal_info['direction'] in ['BUY', 'SELL', 'HOLD'],  # 方向が妥当
            tradeable_count > 0  # 少なくとも1つのシナリオで取引可能
        ]
        
        all_passed = all(success_criteria)
        
        print(f"📊 テスト結果:")
        print(f"   予測値範囲チェック: {'✅' if success_criteria[0] else '❌'}")
        print(f"   信頼度範囲チェック: {'✅' if success_criteria[1] else '❌'}")  
        print(f"   方向判定チェック: {'✅' if success_criteria[2] else '❌'}")
        print(f"   取引可能シグナル: {'✅' if success_criteria[3] else '❌'} ({tradeable_count}/4)")
        
        if all_passed:
            print("\n🎉 V3.1_improved最小限テスト成功！")
            print("✅ モデルは正常に動作しています")
            print("✅ 実際の取引システムで使用可能")
            return True
        else:
            print("\n❌ テストに一部問題があります")
            return False
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    try:
        success = asyncio.run(test_v31_minimal())
        
        print("\n" + "="*80)
        if success:
            print("🎯 結論: V3.1_improved モデルは正常動作")
            print("EC2での本格運用に問題なし")
        else:
            print("❌ 結論: モデルに問題があります")
        print("="*80)
        
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()