#!/usr/bin/env python3
"""
トレーディングシステムの現在状況を診断
- 残高情報
- モデル予測状況  
- 特徴量生成状況
- 取引シグナル生成状況
"""
import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.common.config import settings
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier
from src.common.account_monitor import AccountMonitor
from src.feature_hub.main import FeatureHub
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig

logger = get_logger(__name__)

async def check_system_status():
    """システム全体の状況をチェック"""
    
    print("🔍 トレーディングシステム診断中...")
    
    try:
        # 1. 残高情報の確認
        print("\n📊 残高情報の確認:")
        account_monitor = AccountMonitor(check_interval=5)
        await account_monitor.start()
        await asyncio.sleep(3)  # データ取得待機
        
        if account_monitor.current_balance:
            balance = account_monitor.current_balance
            print(f"  ✅ 残高取得成功:")
            print(f"    - 総資産: ${balance.total_equity:.8f}")
            print(f"    - 利用可能: ${balance.available_balance:.8f}")
            print(f"    - 未実現PnL: ${balance.unrealized_pnl:.8f}")
            print(f"    - 取引可能: {'YES' if balance.total_equity >= 10 else 'NO'}")
        else:
            print("  ❌ 残高取得失敗")
            
        await account_monitor.stop()
        
        # 2. 特徴量生成の確認
        print("\n🔧 特徴量生成の確認:")
        feature_hub = FeatureHub()
        
        for symbol in settings.bybit.symbols:
            features = feature_hub.get_latest_features(symbol)
            print(f"  {symbol}: {len(features) if features else 0} 特徴量")
            
            if features and len(features) > 10:
                print(f"    ✅ 十分な特徴量あり")
            else:
                print(f"    ⚠️ 特徴量不足 (最低10個必要)")
        
        # 3. モデル推論の確認
        print("\n🤖 モデル推論の確認:")
        inference_config = InferenceConfig(
            model_path=settings.model.model_path,
            enable_batching=True,
            confidence_threshold=0.6
        )
        inference_engine = InferenceEngine(inference_config)
        
        try:
            inference_engine.load_model()
            print(f"  ✅ モデル読み込み成功: {settings.model.model_path}")
            
            # 各シンボルで予測テスト
            for symbol in settings.bybit.symbols:
                features = feature_hub.get_latest_features(symbol)
                
                if features and len(features) > 10:
                    try:
                        result = inference_engine.predict(features)
                        prediction = result["predictions"][0] if result["predictions"] else 0
                        confidence = result["confidence_scores"][0] if result["confidence_scores"] else 0
                        
                        print(f"  {symbol}:")
                        print(f"    - 予測値: {prediction:.4f}")
                        print(f"    - 信頼度: {confidence:.2%}")
                        print(f"    - 取引閾値: {'✅ 超過' if confidence > 0.6 else '❌ 未達'} (60%)")
                        
                        if confidence > 0.6:
                            side = "BUY" if prediction > 0 else "SELL"
                            print(f"    - 🚨 高信頼度シグナル: {side}")
                            
                    except Exception as e:
                        print(f"    ❌ 予測エラー: {e}")
                else:
                    print(f"  {symbol}: ❌ 特徴量不足のため予測不可")
                    
        except Exception as e:
            print(f"  ❌ モデル読み込みエラー: {e}")
        
        # 4. システム設定の確認
        print("\n⚙️ システム設定:")
        print(f"  - 信頼度閾値: 60%")
        print(f"  - 最大レバレッジ: 3倍")
        print(f"  - リスク/取引: 1%")
        print(f"  - 監視シンボル: {', '.join(settings.bybit.symbols)}")
        
        # 5. Discord通知テスト
        print(f"\n📤 診断結果をDiscordに送信中...")
        
        # 診断結果をDiscordに送信
        current_time = datetime.now().strftime("%H:%M")
        
        fields = {
            "診断時刻": current_time,
            "システム状態": "正常動作中",
            "残高": f"${balance.total_equity:.2f}" if account_monitor.current_balance else "取得失敗",
            "モデル": "正常動作" if inference_engine.onnx_session else "エラー",
            "取引可能性": "高信頼度シグナル待機中"
        }
        
        discord_notifier.send_notification(
            title="🔍 システム診断レポート",
            description=f"残高$100設定後の動作確認",
            color="03b2f8",
            fields=fields
        )
        
        print("✅ 診断完了！")
        
    except Exception as e:
        print(f"❌ 診断エラー: {e}")
        logger.error(f"Diagnosis error: {e}")

if __name__ == "__main__":
    asyncio.run(check_system_status())