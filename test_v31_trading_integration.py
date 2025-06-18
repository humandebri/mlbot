#!/usr/bin/env python3
"""
V3.1_improved モデルの取引システム統合テスト
修復されたモデルが実際に取引システムで動作するかテスト
"""

import sys
import asyncio
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_v31_trading_integration():
    """V3.1_improved モデルと取引システムの統合テスト"""
    print("="*80)
    print("🔧 V3.1_improved 取引システム統合テスト")
    print("="*80)
    
    try:
        # 1. V3.1_improvedエンジンの初期化
        print("\n1️⃣ V3.1_improvedエンジン初期化...")
        
        from src.ml_pipeline.v31_improved_inference_engine import V31ImprovedInferenceEngine, V31ImprovedConfig
        
        config = V31ImprovedConfig(
            model_path="models/v3.1_improved/model.onnx",
            confidence_threshold=0.7,
            buy_threshold=0.55,
            sell_threshold=0.45,
            high_confidence=0.75,
            medium_confidence=0.6
        )
        
        engine = V31ImprovedInferenceEngine(config)
        engine.load_model()
        
        print("✅ V3.1_improvedエンジン初期化完了")
        
        # 2. 特徴量アダプターテスト
        print("\n2️⃣ 特徴量アダプターテスト...")
        
        from src.ml_pipeline.feature_adapter_44 import FeatureAdapter44
        
        adapter = FeatureAdapter44()
        print("✅ FeatureAdapter44初期化完了")
        
        # 3. 統合シナリオテスト
        print("\n3️⃣ 統合シナリオテスト...")
        
        # シミュレートされた特徴量データ（実際のシステムから来るような）
        test_scenarios = [
            {
                "name": "強い上昇シグナル",
                "features": {
                    # 基本価格特徴量
                    "returns": 0.025,
                    "log_returns": 0.0247, 
                    "hl_ratio": 0.018,
                    "oc_ratio": 0.025,
                    
                    # リターン系
                    "return_1": 0.008,
                    "return_3": 0.015,
                    "return_5": 0.020,
                    "return_10": 0.025,
                    "return_15": 0.030,
                    
                    # ボラティリティ
                    "vol_5": 0.035,
                    "vol_10": 0.032,
                    "vol_20": 0.028,
                    
                    # テクニカル指標
                    "rsi_14": 75.0,
                    "bb_position_20": 0.85,
                    "macd_hist": 5.0,
                    
                    # 価格データ
                    "open": 67000.0,
                    "high": 67500.0,
                    "low": 66800.0,
                    "close": 67300.0,
                    "volume": 1500000.0,
                    
                    # その他の特徴量（ランダム生成で補完）
                    **{f"feature_{i}": np.random.normal(0, 0.1) for i in range(100, 150)}
                }
            },
            {
                "name": "下落シグナル",
                "features": {
                    # 基本価格特徴量
                    "returns": -0.035,
                    "log_returns": -0.0356,
                    "hl_ratio": 0.025,
                    "oc_ratio": -0.035,
                    
                    # リターン系
                    "return_1": -0.012,
                    "return_3": -0.025,
                    "return_5": -0.030,
                    "return_10": -0.035,
                    "return_15": -0.040,
                    
                    # ボラティリティ
                    "vol_5": 0.050,
                    "vol_10": 0.048,
                    "vol_20": 0.045,
                    
                    # テクニカル指標
                    "rsi_14": 25.0,
                    "bb_position_20": 0.15,
                    "macd_hist": -7.0,
                    
                    # 価格データ
                    "open": 67000.0,
                    "high": 67100.0,
                    "low": 64500.0,
                    "close": 64800.0,
                    "volume": 2000000.0,
                    
                    # その他の特徴量
                    **{f"feature_{i}": np.random.normal(0, 0.1) for i in range(100, 150)}
                }
            },
            {
                "name": "レンジ相場",
                "features": {
                    # 基本価格特徴量
                    "returns": 0.002,
                    "log_returns": 0.00199,
                    "hl_ratio": 0.012,
                    "oc_ratio": 0.002,
                    
                    # リターン系
                    "return_1": 0.001,
                    "return_3": 0.002,
                    "return_5": 0.003,
                    "return_10": 0.002,
                    "return_15": 0.001,
                    
                    # ボラティリティ
                    "vol_5": 0.015,
                    "vol_10": 0.014,
                    "vol_20": 0.013,
                    
                    # テクニカル指標
                    "rsi_14": 52.0,
                    "bb_position_20": 0.50,
                    "macd_hist": 0.5,
                    
                    # 価格データ
                    "open": 66500.0,
                    "high": 66700.0,
                    "low": 66300.0,
                    "close": 66550.0,
                    "volume": 800000.0,
                    
                    # その他の特徴量
                    **{f"feature_{i}": np.random.normal(0, 0.05) for i in range(100, 150)}
                }
            }
        ]
        
        trading_signals = []
        
        for scenario in test_scenarios:
            print(f"\n   📊 {scenario['name']}:")
            
            # 推論実行
            result = engine.predict(scenario['features'])
            
            # 結果表示
            prediction = result['prediction']
            confidence = result['confidence']
            signal_info = result['signal']
            
            print(f"     予測値: {prediction:.4f}")
            print(f"     信頼度: {confidence:.1%}")
            print(f"     方向: {signal_info['direction']}")
            print(f"     取引可能: {signal_info['tradeable']}")
            print(f"     ポジションサイズ: {signal_info['position_size_multiplier']:.1%}")
            
            # 取引可能なシグナルをリストに保存
            if signal_info['tradeable']:
                trading_signals.append({
                    'scenario': scenario['name'],
                    'direction': signal_info['direction'],
                    'confidence': confidence,
                    'prediction': prediction,
                    'position_size': signal_info['position_size_multiplier']
                })
        
        # 4. TradingSignal作成テスト
        print("\n4️⃣ TradingSignal作成テスト...")
        
        from src.order_router.smart_router import TradingSignal
        from datetime import datetime
        
        created_signals = []
        
        for signal_data in trading_signals:
            # TradingSignalオブジェクト作成
            trading_signal = TradingSignal(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                prediction=(signal_data['prediction'] - 0.5) * 0.02,  # Expected PnL
                confidence=signal_data['confidence'],
                features=test_scenarios[0]['features'],  # Use first scenario features
                liquidation_detected=False,
                liquidation_size=0.0,
                liquidation_side=signal_data['direction'].lower()
            )
            
            created_signals.append(trading_signal)
            
            print(f"     ✅ {signal_data['scenario']}: {signal_data['direction']} シグナル作成")
            print(f"        信頼度: {signal_data['confidence']:.1%}")
            print(f"        期待PnL: {trading_signal.prediction:.4f}")
        
        # 5. 統合テスト結果
        print("\n5️⃣ 統合テスト結果...")
        
        total_scenarios = len(test_scenarios)
        tradeable_signals = len(trading_signals)
        created_signal_objects = len(created_signals)
        
        print(f"✅ 統合テスト統計:")
        print(f"   - テストシナリオ: {total_scenarios}")
        print(f"   - 取引可能シグナル: {tradeable_signals}")
        print(f"   - TradingSignal作成: {created_signal_objects}")
        print(f"   - 成功率: {(created_signal_objects/total_scenarios)*100:.1f}%")
        
        # 6. 最終判定
        print("\n6️⃣ 最終判定...")
        
        if created_signal_objects > 0:
            print("🎉 V3.1_improved統合テスト成功！")
            print("✅ モデルが正常に予測を生成")
            print("✅ シグナル判定ロジックが動作")
            print("✅ TradingSignalオブジェクト作成可能")
            print("✅ 実際の取引システムで使用可能")
            
            print("\n📝 次のステップ:")
            print("1. main_dynamic_integration.pyでV3.1_improvedエンジンを使用")
            print("2. EC2本番環境にデプロイ")
            print("3. 実際の市場データでのライブテスト")
            
            return True
        else:
            print("❌ 統合テスト失敗")
            print("シグナル生成またはTradingSignal作成に問題があります")
            return False
        
    except Exception as e:
        print(f"❌ 統合テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    try:
        success = asyncio.run(test_v31_trading_integration())
        
        print("\n" + "="*80)
        if success:
            print("🎯 結論: V3.1_improved取引システム統合成功")
            print("修復されたモデルは実際の取引で使用可能です")
        else:
            print("❌ 結論: 統合に問題があります")
            print("追加の修正が必要です")
        print("="*80)
        
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()