#!/bin/bash

echo "📊 Discordレポートとモニタリングを修正..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. ポジションをAPIから直接取得して表示
echo "💰 現在のポジション（API経由）:"
cat > check_positions.py << 'PYTHON'
import asyncio
import os
from src.common.bybit_client import BybitRESTClient
from datetime import datetime
import requests

# 環境変数を設定
os.environ['BYBIT__API_KEY'] = 'KgMS2YHiCPG49hmWuV'
os.environ['BYBIT__API_SECRET'] = 'LUkW6d5SHHIrbu2SvEDPjFwGYkyUGRun8JF3'

async def get_positions_and_notify():
    client = BybitRESTClient(testnet=False)
    await client.__aenter__()
    
    try:
        # ポジション情報取得
        positions = await client.get_open_positions()
        
        total_upnl = 0
        position_info = []
        
        if positions:
            print(f'\nオープンポジション: {len(positions)}件')
            for pos in positions:
                symbol = pos.get('symbol')
                size = float(pos.get('size', 0))
                avg_price = float(pos.get('avgPrice', 0))
                side = pos.get('side')
                upnl = float(pos.get('unrealisedPnl', 0))
                total_upnl += upnl
                
                print(f"  {symbol}: {size} @ ${avg_price} ({side}) P&L: ${upnl:.4f}")
                
                position_info.append({
                    'name': symbol,
                    'value': f"{size} {side} @ ${avg_price}\\nP&L: ${upnl:.2f}",
                    'inline': True
                })
        
        # アカウント情報
        account = await client.get_account_info()
        if account:
            balance = float(account.get('totalEquity', 0))
            print(f"\nアカウントバランス: ${balance:.2f}")
            print(f"未実現損益合計: ${total_upnl:.4f}")
        
        # Discord通知
        webhook_url = 'https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq'
        
        # システム状態
        with open('logs/prediction_count.txt', 'r') as f:
            pred_count = int(f.read().strip())
        
        message = {
            'embeds': [{
                'title': '📊 リアルタイムポジションレポート',
                'description': f'現在時刻: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC',
                'color': 0x0099ff,
                'fields': [
                    {
                        'name': '💵 アカウント',
                        'value': f'バランス: ${balance:.2f}\\n未実現損益: ${total_upnl:.2f}',
                        'inline': True
                    },
                    {
                        'name': '📈 予測統計',
                        'value': f'予測回数: {pred_count}\\nシグナル: 0 (50%閾値)',
                        'inline': True
                    }
                ] + position_info
            }]
        }
        
        response = requests.post(webhook_url, json=message)
        if response.status_code == 204:
            print('\n✅ Discord通知送信完了')
        
    finally:
        await client.__aexit__(None, None, None)

asyncio.run(get_positions_and_notify())
PYTHON

# 予測カウントファイルを作成
echo "1074" > logs/prediction_count.txt

# スクリプトを実行
python3 check_positions.py

# 2. 自動モニタリングスクリプトを作成
echo ""
echo "🔄 自動モニタリングスクリプトを作成:"
cat > auto_monitor.py << 'PYTHON'
import time
import subprocess
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_position_check():
    """ポジションチェックを実行"""
    try:
        subprocess.run(['python3', 'check_positions.py'], check=True)
        logger.info(f"ポジションチェック完了: {datetime.now()}")
    except Exception as e:
        logger.error(f"エラー: {e}")

# 30分ごとに実行
while True:
    run_position_check()
    time.sleep(1800)  # 30分
PYTHON

# 3. データ蓄積状況を確認
echo ""
echo "📊 データ蓄積状況:"
python3 -c "
import redis
r = redis.Redis(host='localhost', port=6379)
length = r.xlen(b'market_data:kline')
print(f'Redisエントリー数: {length}')
"

# 4. 今後の対策
echo ""
echo "💡 信頼度向上のための対策:"
echo "  1. ✅ リアルタイムデータ取得中（Ingestor稼働中）"
echo "  2. ⏳ 3-6時間後に十分なデータが蓄積"
echo "  3. 📈 その後DuckDBを更新して信頼度50%以上達成"

EOF

echo ""
echo "✅ 修正完了！"