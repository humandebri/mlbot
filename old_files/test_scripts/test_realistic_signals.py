#!/usr/bin/env python3
"""
現実的な価格変動でシグナル生成をテスト
"""

import sys
import os
import numpy as np
import json
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_hub.technical_indicators import TechnicalIndicatorEngine
from src.ml_pipeline.feature_adapter_44 import FeatureAdapter44

def generate_realistic_price_data():
    """現実的な価格変動データを生成"""
    
    print("📈 現実的な価格変動データを生成中...")
    
    # 初期価格
    base_price = 106000
    prices = []
    
    # 50個の価格データを生成（ランダムウォーク + トレンド）
    for i in range(50):
        # ランダムな変動（-0.5% ~ +0.5%）
        change_pct = np.random.normal(0, 0.002)  # 平均0、標準偏差0.2%
        
        # 時々大きな動き（5%の確率で大きな変動）
        if np.random.random() < 0.05:
            change_pct *= 5  # 大きな変動
        
        new_price = base_price * (1 + change_pct)
        
        # OHLC生成
        open_price = base_price
        close_price = new_price
        
        # High/Low計算（0.1%のランダム幅）
        spread = new_price * 0.001 * np.random.uniform(0.5, 2.0)
        high = max(open_price, close_price) + spread * np.random.uniform(0, 1)
        low = min(open_price, close_price) - spread * np.random.uniform(0, 1)
        
        # Volume（10万〜200万の範囲）
        volume = np.random.uniform(100000, 2000000)
        
        prices.append({
            'open': open_price,
            'high': high,
            'low': low, 
            'close': close_price,
            'volume': volume,
            'timestamp': datetime.now() + timedelta(minutes=i)
        })
        
        base_price = new_price
    
    print(f"   ✅ {len(prices)}個の価格データを生成")
    print(f"   価格範囲: ${prices[0]['close']:.0f} ~ ${prices[-1]['close']:.0f}")
    
    return prices

def test_signal_generation():
    """現実的なデータでシグナル生成をテスト"""
    
    print("🎯 シグナル生成テスト開始")
    print("=" * 60)
    
    # 1. 現実的な価格データ生成
    price_data = generate_realistic_price_data()
    
    # 2. 技術的指標エンジン初期化
    tech_engine = TechnicalIndicatorEngine()
    adapter = FeatureAdapter44()
    
    print("\n2. 価格データを順次処理中...")
    
    for i, data in enumerate(price_data):
        features = tech_engine.update_price_data(
            "BTCUSDT",
            data['open'],
            data['high'], 
            data['low'],
            data['close'],
            data['volume']
        )
        
        if i % 10 == 0:
            print(f"   データ {i+1}/{len(price_data)} 処理完了")
    
    # 3. 最終的な技術的指標を確認
    final_features = tech_engine.get_latest_features("BTCUSDT")
    
    print("\n3. 生成された技術的指標:")
    key_indicators = {
        "returns": final_features.get("returns", 0),
        "vol_20": final_features.get("vol_20", 0),
        "rsi_14": final_features.get("rsi_14", 0),
        "macd": final_features.get("macd", 0),
        "price_vs_sma_20": final_features.get("price_vs_sma_20", 0),
        "bb_position_20": final_features.get("bb_position_20", 0),
        "trend_strength_long": final_features.get("trend_strength_long", 0)
    }
    
    for name, value in key_indicators.items():
        status = "✅" if abs(value) > 0.001 else "❌"
        print(f"   {status} {name}: {value:.6f}")
    
    # 4. 特徴量アダプターでテスト
    print("\n4. 特徴量変換テスト...")
    adapted_features = adapter.adapt(final_features)
    
    print(f"   ✅ 変換後形状: {adapted_features.shape}")
    print(f"   非ゼロ値の数: {np.count_nonzero(adapted_features)}/44")
    print(f"   値の範囲: {np.min(adapted_features):.3f} ~ {np.max(adapted_features):.3f}")
    
    # 5. 手動でONNXテスト
    print("\n5. ONNXモデル直接テスト...")
    
    try:
        import onnxruntime as ort
        
        # スケーラー読み込み
        scaler_path = "models/v3.1_improved/manual_scaler.json"
        with open(scaler_path, 'r') as f:
            scaler_data = json.load(f)
        
        means = np.array(scaler_data['means'])
        stds = np.array(scaler_data['stds'])
        
        # 正規化
        normalized = (adapted_features - means) / stds
        normalized = np.clip(normalized, -5, 5)
        
        # モデル実行
        model_path = "models/v3.1_improved/model.onnx"
        session = ort.InferenceSession(model_path)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        prediction = session.run([output_name], {input_name: normalized.reshape(1, -1).astype(np.float32)})
        raw_output = prediction[0][0]
        
        print(f"   ✅ モデル出力: {raw_output:.6f}")
        
        # 信頼度計算
        confidence = abs(raw_output - 0.5) * 2
        expected_pnl = (raw_output - 0.5) * 0.02
        
        print(f"   信頼度: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"   期待PnL: {expected_pnl:.6f} ({expected_pnl*100:.4f}%)")
        
        # 6. シグナル判定
        print("\n6. シグナル判定:")
        
        # 現在の閾値
        conf_threshold = 0.7
        pnl_threshold = 0.001
        
        print(f"   現在の閾値: 信頼度>{conf_threshold*100}%, PnL>{pnl_threshold*100}%")
        
        if confidence > conf_threshold and abs(expected_pnl) > pnl_threshold:
            print("   🎯 シグナル生成: YES")
        else:
            print("   ❌ シグナル生成: NO")
            
            # より低い閾値でテスト
            low_conf = 0.1
            low_pnl = 0.0001
            
            print(f"\n   低い閾値でテスト: 信頼度>{low_conf*100}%, PnL>{low_pnl*100}%")
            
            if confidence > low_conf and abs(expected_pnl) > low_pnl:
                print("   ✅ 低い閾値ではシグナル生成可能")
            else:
                print("   ❌ 低い閾値でもシグナル生成不可")
        
        # 7. 推奨修正
        print("\n7. 推奨修正:")
        
        if raw_output == 0:
            print("   🚨 モデル出力が0 - モデル修復が必要")
        elif 0.45 <= raw_output <= 0.55:
            print("   ⚠️  モデル出力が中立 - より極端な市場条件が必要")
        elif confidence < 0.1:
            print("   💡 信頼度が低い - 閾値を5%以下に下げる")
        elif abs(expected_pnl) < 0.0001:
            print("   💡 期待PnLが小さい - 閾値を0.01%以下に下げる")
            
    except Exception as e:
        print(f"   ❌ ONNXテストエラー: {e}")
    
    print("\n" + "=" * 60)
    print("テスト完了！")

if __name__ == "__main__":
    test_signal_generation()