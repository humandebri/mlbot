#!/bin/bash

echo "🔧 モデルバイアス補正を適用..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

# バイアス補正コードを作成
cat > bias_correction_patch.py << 'EOF'
# simple_improved_bot_with_trading_fixed.pyに追加するバイアス補正コード

import numpy as np
from collections import deque

class BiasCorrector:
    """予測値のバイアスを動的に補正"""
    
    def __init__(self, window_size=1000, initial_offset=0.15):
        self.predictions = deque(maxlen=window_size)
        self.initial_offset = initial_offset
        
    def add_prediction(self, pred):
        """予測値を記録"""
        self.predictions.append(pred)
        
    def get_bias_offset(self):
        """現在のバイアスオフセットを計算"""
        if len(self.predictions) < 100:
            return self.initial_offset
        
        mean_pred = np.mean(list(self.predictions))
        # 平均が0.5になるようオフセットを計算
        offset = 0.5 - mean_pred
        # 極端な補正を避ける
        return np.clip(offset, -0.3, 0.3)
        
    def correct_prediction(self, raw_pred):
        """予測値を補正"""
        offset = self.get_bias_offset()
        
        # シグモイド関数でスムーズに調整
        # より急峻な変換で中央値付近の感度を上げる
        adjusted = raw_pred + offset
        corrected = 1 / (1 + np.exp(-8 * (adjusted - 0.5)))
        
        return float(np.clip(corrected, 0.0, 1.0))
    
    def get_stats(self):
        """統計情報を取得"""
        if len(self.predictions) == 0:
            return {}
        
        preds = list(self.predictions)
        return {
            'count': len(preds),
            'mean': np.mean(preds),
            'std': np.std(preds),
            'min': np.min(preds),
            'max': np.max(preds),
            'offset': self.get_bias_offset()
        }
EOF

ssh -i $KEY_PATH $EC2_HOST << 'REMOTE_EOF'
cd /home/ubuntu/mlbot

# 1. 現在のボットを停止
echo "🛑 現在のボットを停止..."
pkill -f simple_improved_bot_with_trading_fixed.py
sleep 3

# 2. バックアップを作成
echo "💾 バックアップを作成..."
cp simple_improved_bot_with_trading_fixed.py simple_improved_bot_with_trading_fixed.py.backup_$(date +%Y%m%d_%H%M%S)

# 3. バイアス補正を組み込んだ新しいボットを作成
echo "📝 バイアス補正版ボットを作成..."
cat > simple_improved_bot_bias_corrected.py << 'EOF'
#!/usr/bin/env python3
"""
ML取引ボット（バイアス補正版）
v3.1_improvedモデルのSELLバイアスを動的に補正
"""

import asyncio
import os
import sys
import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import json
import traceback
import aiohttp
import websockets
import redis.asyncio as redis
import onnxruntime as ort
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from src.utils.logger import Logger
from src.common.bybit_client import BybitRESTClient, BybitWebSocketClient

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Improved Feature Generatorを使用
from improved_feature_generator_readonly import ImprovedFeatureGeneratorReadOnly as ImprovedFeatureGenerator

class BiasCorrector:
    """予測値のバイアスを動的に補正"""
    
    def __init__(self, window_size=1000, initial_offset=0.15):
        self.predictions = deque(maxlen=window_size)
        self.initial_offset = initial_offset
        self.correction_enabled = True  # 補正の有効/無効を制御
        
    def add_prediction(self, pred):
        """予測値を記録"""
        self.predictions.append(pred)
        
    def get_bias_offset(self):
        """現在のバイアスオフセットを計算"""
        if len(self.predictions) < 100:
            return self.initial_offset
        
        mean_pred = np.mean(list(self.predictions))
        # 平均が0.5になるようオフセットを計算
        offset = 0.5 - mean_pred
        # 極端な補正を避ける
        return np.clip(offset, -0.3, 0.3)
        
    def correct_prediction(self, raw_pred):
        """予測値を補正"""
        if not self.correction_enabled:
            return raw_pred
            
        offset = self.get_bias_offset()
        
        # シグモイド関数でスムーズに調整
        # より急峻な変換で中央値付近の感度を上げる
        adjusted = raw_pred + offset
        corrected = 1 / (1 + np.exp(-8 * (adjusted - 0.5)))
        
        return float(np.clip(corrected, 0.0, 1.0))
    
    def get_stats(self):
        """統計情報を取得"""
        if len(self.predictions) == 0:
            return {}
        
        preds = list(self.predictions)
        buy_count = sum(1 for p in preds if p > 0.5)
        sell_count = len(preds) - buy_count
        
        return {
            'count': len(preds),
            'mean': np.mean(preds),
            'std': np.std(preds),
            'min': np.min(preds),
            'max': np.max(preds),
            'offset': self.get_bias_offset(),
            'buy_ratio': buy_count / len(preds) if len(preds) > 0 else 0,
            'sell_ratio': sell_count / len(preds) if len(preds) > 0 else 0
        }

class MLTradingBot:
    def __init__(self, testnet: bool = False):
        self.testnet = testnet
        self.logger = Logger()
        
        # 取引設定
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']
        self.confidence_threshold = 0.50  # 信頼度閾値
        self.position_size_pct = 0.02  # アカウントバランスの2%
        self.max_positions = 3
        self.min_confidence = 0.50
        
        # バイアス補正器
        self.bias_corrector = BiasCorrector(window_size=1000, initial_offset=0.15)
        
        # Discord設定
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK")
        self.last_signal_time = {}
        self.signal_cooldown = 300  # 5分のクールダウン
        
        # Redis接続
        self.redis_client = None
        
        # モデルとスケーラー
        model_path = os.getenv("MODEL__MODEL_PATH", "models/v3.1_improved/model.onnx")
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # スケーラーパラメータを読み込み
        scaler_path = "models/v3.1_improved/manual_scaler.json"
        with open(scaler_path, 'r') as f:
            self.scaler_params = json.load(f)
        
        # Feature Generator
        self.feature_generator = ImprovedFeatureGenerator()
        
        # カウンター
        self.prediction_count = 0
        self.signal_count = 0
        self.prediction_history = []
        
        # REST client
        self.rest_client = None
        
        # アカウント情報
        self.current_balance = None
        self.open_positions = {}
        
        # 時間レポート用
        self.last_report_time = datetime.utcnow()
        self.hourly_predictions = {symbol: [] for symbol in self.symbols}
        
    async def initialize(self):
        """初期化処理"""
        try:
            # Redis接続
            self.redis_client = await redis.from_url(
                'redis://localhost:6379',
                decode_responses=False
            )
            
            # REST client初期化
            self.rest_client = BybitRESTClient(testnet=self.testnet)
            await self.rest_client.__aenter__()
            
            # 初期バランス取得
            await self.update_balance()
            
            logger.info("ML取引ボット（バイアス補正版）初期化完了")
            
        except Exception as e:
            logger.error(f"初期化エラー: {e}")
            raise
    
    async def update_balance(self):
        """アカウントバランスを更新"""
        try:
            account_info = await self.rest_client.get_account_info()
            if account_info:
                self.current_balance = float(account_info.get('totalEquity', 0))
                logger.info(f"アカウントバランス更新: ${self.current_balance:.2f}")
        except Exception as e:
            logger.error(f"バランス更新エラー: {e}")
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """特徴量を正規化"""
        mean = np.array(self.scaler_params['means'])
        std = np.array(self.scaler_params['stds'])
        
        # ゼロ除算を避ける
        std_safe = np.where(std == 0, 1.0, std)
        
        normalized = (features - mean) / std_safe
        return normalized
    
    async def predict(self, symbol: str) -> Optional[Dict[str, Any]]:
        """価格の動きを予測"""
        try:
            # 特徴量を生成
            features = await self.feature_generator.get_features(symbol)
            if features is None:
                return None
            
            # 正規化
            normalized = self.normalize_features(np.array(features))
            
            # 予測
            input_data = normalized.reshape(1, -1).astype(np.float32)
            outputs = self.session.run(None, {'float_input': input_data})
            
            # 予測値を抽出
            if len(outputs) > 1 and isinstance(outputs[1], list) and len(outputs[1]) > 0:
                prob_dict = outputs[1][0]
                raw_prediction = prob_dict.get(1, 0.5)
            else:
                raw_prediction = float(outputs[0][0])
            
            # バイアス補正を適用
            self.bias_corrector.add_prediction(raw_prediction)
            prediction = self.bias_corrector.correct_prediction(raw_prediction)
            
            # 信頼度を計算
            confidence = abs(prediction - 0.5) * 2
            
            # 予測回数を更新
            self.prediction_count += 1
            
            result = {
                'symbol': symbol,
                'raw_prediction': raw_prediction,
                'corrected_prediction': prediction,
                'confidence': confidence,
                'direction': 'BUY' if prediction > 0.5 else 'SELL',
                'timestamp': datetime.utcnow()
            }
            
            # 履歴に追加
            self.prediction_history.append({
                'symbol': symbol,
                'prediction': prediction,
                'raw_prediction': raw_prediction,
                'confidence': confidence,
                'timestamp': datetime.utcnow()
            })
            
            # 時間別統計用に保存
            self.hourly_predictions[symbol].append(prediction)
            
            # 履歴のサイズを制限
            if len(self.prediction_history) > 2000:
                self.prediction_history = self.prediction_history[-2000:]
            
            return result
            
        except Exception as e:
            logger.error(f"{symbol}の予測エラー: {e}")
            logger.error(traceback.format_exc())
            return None
    
    async def should_send_signal(self, symbol: str, confidence: float) -> bool:
        """シグナルを送信すべきか判断"""
        if confidence < self.confidence_threshold:
            return False
        
        # クールダウンチェック
        now = datetime.utcnow()
        if symbol in self.last_signal_time:
            elapsed = (now - self.last_signal_time[symbol]).total_seconds()
            if elapsed < self.signal_cooldown:
                return False
        
        return True
    
    async def send_discord_signal(self, prediction: Dict[str, Any]):
        """Discord通知を送信"""
        if not self.discord_webhook:
            return
        
        try:
            symbol = prediction['symbol']
            direction = prediction['direction']
            confidence = prediction['confidence']
            raw_pred = prediction['raw_prediction']
            corrected_pred = prediction['corrected_prediction']
            
            # 現在の価格を取得
            ticker = await self.rest_client.get_ticker(symbol)
            if not ticker:
                return
            
            current_price = float(ticker.get('lastPrice', 0))
            
            # メッセージを作成（バイアス補正情報を含む）
            color = 0x00ff00 if direction == 'BUY' else 0xff0000
            
            embed = {
                "title": f"🎯 ML Signal #{self.signal_count + 1} - {symbol}",
                "description": f"方向: **{direction}**",
                "color": color,
                "fields": [
                    {"name": "信頼度", "value": f"{confidence*100:.1f}%", "inline": True},
                    {"name": "価格", "value": f"${current_price:,.2f}", "inline": True},
                    {"name": "予測値", "value": f"Raw: {raw_pred:.4f}\nCorrected: {corrected_pred:.4f}", "inline": True},
                    {"name": "バイアス補正", "value": f"Offset: {self.bias_corrector.get_bias_offset():.4f}", "inline": True}
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # 統計情報を追加
            stats = self.bias_corrector.get_stats()
            if stats:
                embed["fields"].append({
                    "name": "予測統計",
                    "value": f"Buy/Sell比: {stats['buy_ratio']:.1%}/{stats['sell_ratio']:.1%}",
                    "inline": False
                })
            
            async with aiohttp.ClientSession() as session:
                await session.post(self.discord_webhook, json={"embeds": [embed]})
            
            self.last_signal_time[symbol] = datetime.utcnow()
            self.signal_count += 1
            logger.info(f"Discord通知送信: {symbol} {direction} (信頼度: {confidence*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"Discord通知エラー: {e}")
    
    async def execute_trade(self, prediction: Dict[str, Any]):
        """取引を実行"""
        try:
            symbol = prediction['symbol']
            direction = prediction['direction']
            confidence = prediction['confidence']
            
            # ポジションサイズを計算
            if not self.current_balance:
                await self.update_balance()
            
            position_value = self.current_balance * self.position_size_pct
            
            # 現在の価格を取得
            ticker = await self.rest_client.get_ticker(symbol)
            if not ticker:
                return
            
            current_price = float(ticker.get('lastPrice', 0))
            
            # 数量を計算
            qty = position_value / current_price
            
            # 最小取引単位に丸める
            if symbol == 'BTCUSDT':
                qty = round(qty, 3)
            elif symbol == 'ETHUSDT':
                qty = round(qty, 2)
            elif symbol == 'ICPUSDT':
                qty = round(qty, 0)
            
            # 注文を送信
            side = 'Buy' if direction == 'BUY' else 'Sell'
            
            order = await self.rest_client.place_order(
                symbol=symbol,
                side=side,
                order_type='Market',
                qty=qty,
                reduce_only=False
            )
            
            if order:
                logger.info(f"注文実行成功: {symbol} {side} {qty} @ {current_price}")
                await self.send_discord_signal(prediction)
            else:
                logger.error(f"注文実行失敗: {symbol} {side}")
                
        except Exception as e:
            logger.error(f"取引実行エラー: {e}")
            logger.error(traceback.format_exc())
    
    async def monitor_positions(self):
        """ポジションを監視"""
        try:
            positions = await self.rest_client.get_open_positions()
            if positions:
                for pos in positions:
                    symbol = pos.get('symbol')
                    size = float(pos.get('size', 0))
                    side = pos.get('side')
                    unrealized_pnl = float(pos.get('unrealisedPnl', 0))
                    
                    logger.info(f"Position {symbol}: {size} {side}, P&L: ${unrealized_pnl:.2f}")
                    
                    # ストップロス/テイクプロフィット管理（将来実装）
                    
        except Exception as e:
            logger.error(f"ポジション監視エラー: {e}")
    
    async def send_hourly_report(self):
        """時間別レポートを送信"""
        if not self.discord_webhook:
            return
        
        try:
            # バランスを更新
            await self.update_balance()
            
            # ポジション情報を取得
            positions = await self.rest_client.get_open_positions()
            total_unrealized_pnl = 0
            position_details = []
            
            if positions:
                for pos in positions:
                    unrealized_pnl = float(pos.get('unrealisedPnl', 0))
                    total_unrealized_pnl += unrealized_pnl
                    position_details.append(f"{pos.get('symbol')}: ${unrealized_pnl:.2f}")
            
            # 統計情報を計算
            stats = self.bias_corrector.get_stats()
            
            # 各シンボルの統計
            symbol_stats = []
            for symbol in self.symbols:
                if self.hourly_predictions[symbol]:
                    preds = self.hourly_predictions[symbol]
                    buy_count = sum(1 for p in preds if p > 0.5)
                    sell_count = len(preds) - buy_count
                    avg_pred = np.mean(preds)
                    
                    # 現在の価格を取得
                    ticker = await self.rest_client.get_ticker(symbol)
                    price = float(ticker.get('lastPrice', 0)) if ticker else 0
                    
                    symbol_stats.append({
                        "name": symbol,
                        "value": f"Price: ${price:,.2f}\nAvg Pred: {avg_pred:.4f}\nBuy/Sell: {buy_count}/{sell_count}",
                        "inline": True
                    })
            
            # メッセージを作成
            message = {
                "embeds": [{
                    "title": "📊 時間レポート（バイアス補正版）",
                    "description": f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                    "color": 0x0099ff,
                    "fields": [
                        {
                            "name": "💰 アカウント情報",
                            "value": f"• 残高: ${self.current_balance:.2f}\n• ポジション数: {len(positions) if positions else 0}\n• 未実現損益: ${total_unrealized_pnl:.2f}",
                            "inline": True
                        },
                        {
                            "name": "📈 取引統計",
                            "value": f"• 予測回数: {self.prediction_count}\n• シグナル数: {self.signal_count}",
                            "inline": True
                        },
                        {
                            "name": "🔧 バイアス補正",
                            "value": f"• オフセット: {stats.get('offset', 0):.4f}\n• Buy/Sell比: {stats.get('buy_ratio', 0):.1%}/{stats.get('sell_ratio', 0):.1%}",
                            "inline": True
                        }
                    ] + symbol_stats
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                await session.post(self.discord_webhook, json=message)
            
            # 時間別予測をリセット
            self.hourly_predictions = {symbol: [] for symbol in self.symbols}
            
            logger.info("時間別レポート送信完了")
            
        except Exception as e:
            logger.error(f"時間別レポート送信エラー: {e}")
    
    async def run(self):
        """メインループ"""
        await self.initialize()
        
        prediction_interval = 10  # 10秒ごとに予測
        position_check_interval = 60  # 1分ごとにポジションチェック
        report_interval = 3600  # 1時間ごとにレポート
        
        last_position_check = datetime.utcnow()
        
        try:
            while True:
                try:
                    # 各シンボルで予測
                    for symbol in self.symbols:
                        prediction = await self.predict(symbol)
                        
                        if prediction:
                            # ログ出力（バイアス補正情報を含む）
                            logger.info(
                                f"📊 {symbol}: raw={prediction['raw_prediction']:.4f}, "
                                f"corrected={prediction['corrected_prediction']:.4f}, "
                                f"conf={prediction['confidence']*100:.2f}%, "
                                f"dir={prediction['direction']}, "
                                f"offset={self.bias_corrector.get_bias_offset():.4f}"
                            )
                            
                            # 取引シグナルチェック
                            if await self.should_send_signal(symbol, prediction['confidence']):
                                await self.execute_trade(prediction)
                    
                    # ポジション監視
                    now = datetime.utcnow()
                    if (now - last_position_check).total_seconds() >= position_check_interval:
                        await self.monitor_positions()
                        last_position_check = now
                    
                    # 時間別レポート
                    if (now - self.last_report_time).total_seconds() >= report_interval:
                        await self.send_hourly_report()
                        self.last_report_time = now
                    
                    # 次の予測まで待機
                    await asyncio.sleep(prediction_interval)
                    
                except Exception as e:
                    logger.error(f"メインループエラー: {e}")
                    logger.error(traceback.format_exc())
                    await asyncio.sleep(30)
                    
        except KeyboardInterrupt:
            logger.info("シャットダウン中...")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """クリーンアップ処理"""
        try:
            if self.rest_client:
                await self.rest_client.__aexit__(None, None, None)
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.feature_generator.close()
            
            logger.info("クリーンアップ完了")
            
        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Trading Bot with Bias Correction')
    parser.add_argument('--testnet', action='store_true', help='Use testnet')
    args = parser.parse_args()
    
    bot = MLTradingBot(testnet=args.testnet)
    asyncio.run(bot.run())
EOF

# 4. 新しいボットを起動
echo "🚀 バイアス補正版ボットを起動..."
export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq'
nohup python3 simple_improved_bot_bias_corrected.py > logs/mlbot_bias_corrected_$(date +%Y%m%d_%H%M%S).log 2>&1 &
NEW_PID=$!

echo "✅ バイアス補正版ボット起動完了 (PID: $NEW_PID)"

# 5. 動作確認
sleep 5
echo ""
echo "📊 動作確認:"
tail -20 logs/mlbot_bias_corrected_*.log | grep -E "(corrected=|offset=|Buy/Sell)"

echo ""
echo "✅ バイアス補正適用完了！"
echo ""
echo "📈 期待される効果:"
echo "  - 予測値が0.5を中心により均等に分布"
echo "  - Buy/Sell比率の改善"
echo "  - より多様な取引シグナル"

REMOTE_EOF

echo ""
echo "✅ 完了！"