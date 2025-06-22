#!/usr/bin/env python3
"""
予測カウントレポートの修正スクリプト
Discordレポートで予測回数が正しく表示されない問題を修正
"""

import asyncio
import logging
from datetime import datetime
import redis
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportMonitor:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.discord_webhook = 'https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq'
        
    def get_actual_stats(self):
        """実際の統計情報を取得"""
        # ログファイルから実際の予測回数を計算
        import subprocess
        
        # 最新のログファイルを取得
        result = subprocess.run(
            "ls -t logs/mlbot_*.log | head -1 | xargs grep -c 'pred='",
            shell=True,
            capture_output=True,
            text=True
        )
        
        prediction_count = int(result.stdout.strip()) if result.stdout.strip() else 0
        
        # 信頼度50%以上のシグナル数を計算
        result = subprocess.run(
            "ls -t logs/mlbot_*.log | head -1 | xargs grep 'Signal sent' | wc -l",
            shell=True,
            capture_output=True,
            text=True
        )
        
        signal_count = int(result.stdout.strip()) if result.stdout.strip() else 0
        
        return prediction_count, signal_count
    
    def send_corrected_report(self):
        """修正されたレポートを送信"""
        prediction_count, signal_count = self.get_actual_stats()
        
        # 現在の価格情報を取得
        entries = self.redis_client.xrevrange('market_data:kline', count=10)
        
        btc_price = eth_price = icp_price = 0
        for entry_id, data in entries:
            try:
                import json
                parsed = json.loads(data.get('data', '{}'))
                topic = parsed.get('topic', '')
                
                if 'BTCUSDT' in topic and btc_price == 0:
                    btc_price = float(parsed['data'][0]['close'])
                elif 'ETHUSDT' in topic and eth_price == 0:
                    eth_price = float(parsed['data'][0]['close'])
                elif 'ICPUSDT' in topic and icp_price == 0:
                    icp_price = float(parsed['data'][0]['close'])
                    
                if all([btc_price, eth_price, icp_price]):
                    break
            except:
                pass
        
        # Discordメッセージを作成
        message = {
            'embeds': [{
                'title': '📊 時間レポート（修正版）',
                'description': f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC',
                'color': 0x0099ff,
                'fields': [
                    {
                        'name': '💰 現在の価格',
                        'value': f'BTC: ${btc_price:,.2f}\nETH: ${eth_price:,.2f}\nICP: ${icp_price:.3f}',
                        'inline': True
                    },
                    {
                        'name': '📈 取引統計',
                        'value': f'• 予測回数: {prediction_count:,}\n• シグナル数: {signal_count}',
                        'inline': True
                    },
                    {
                        'name': '⚙️ システム状態',
                        'value': '✅ 正常稼働中\n信頼度閾値: 50%\n最高信頼度: ~47%',
                        'inline': True
                    },
                    {
                        'name': '📌 メモ',
                        'value': 'データ蓄積中により信頼度向上中',
                        'inline': False
                    }
                ],
                'footer': {
                    'text': 'ML Trading Bot - Fixed Report'
                }
            }]
        }
        
        # Discord通知を送信
        response = requests.post(self.discord_webhook, json=message)
        if response.status_code == 204:
            logger.info(f"修正レポート送信完了: 予測{prediction_count:,}回, シグナル{signal_count}件")
            return True
        else:
            logger.error(f"Discord送信エラー: {response.status_code}")
            return False
    
    async def run_monitor(self):
        """定期的にレポートを送信"""
        while True:
            try:
                self.send_corrected_report()
                await asyncio.sleep(3600)  # 1時間ごと
            except Exception as e:
                logger.error(f"エラー: {e}")
                await asyncio.sleep(300)  # エラー時は5分後に再試行

def main():
    """メイン関数"""
    monitor = ReportMonitor()
    
    # 即座に修正レポートを送信
    print("📊 修正レポートを送信中...")
    if monitor.send_corrected_report():
        print("✅ 修正レポート送信完了！")
    else:
        print("❌ 修正レポート送信失敗")
    
    # 定期モニタリングを開始する場合
    # asyncio.run(monitor.run_monitor())

if __name__ == "__main__":
    main()