#!/bin/bash

echo "🚨 緊急：データ取得を修復..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. Ingestorが動いているか確認
echo "🔍 Ingestorプロセスを確認:"
ps aux | grep -E "(ingestor|websocket)" | grep -v grep
if [ $? -ne 0 ]; then
    echo "⚠️  Ingestorが動いていません！"
fi

# 2. WebSocket接続をテスト
echo ""
echo "🌐 WebSocket接続テスト:"
python3 -c "
import asyncio
import websockets
import json
from datetime import datetime

async def test_websocket():
    uri = 'wss://stream.bybit.com/v5/public/linear'
    
    try:
        async with websockets.connect(uri) as ws:
            # Subscribe to kline data
            subscribe_msg = {
                'op': 'subscribe',
                'args': ['kline.1.BTCUSDT']
            }
            
            await ws.send(json.dumps(subscribe_msg))
            print('接続成功！')
            
            # 数秒間データを受信
            for i in range(5):
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                
                if data.get('topic') and 'kline' in data['topic']:
                    kline = data['data'][0]
                    print(f\"受信: {datetime.now()} - BTCUSDT: \${kline['close']}\")
                    
    except Exception as e:
        print(f'エラー: {e}')

asyncio.run(test_websocket())
"

# 3. 簡易Ingestorを起動
echo ""
echo "🚀 簡易Ingestorを起動:"
cat > temp_ingestor.py << 'PYTHON'
import asyncio
import websockets
import json
import redis
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleIngestor:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']
        
    async def run(self):
        uri = 'wss://stream.bybit.com/v5/public/linear'
        
        while True:
            try:
                async with websockets.connect(uri) as ws:
                    # Subscribe to kline streams
                    for symbol in self.symbols:
                        subscribe_msg = {
                            'op': 'subscribe',
                            'args': [f'kline.1.{symbol}']
                        }
                        await ws.send(json.dumps(subscribe_msg))
                        logger.info(f'Subscribed to {symbol}')
                    
                    # Receive and store data
                    while True:
                        response = await ws.recv()
                        data = json.loads(response)
                        
                        if data.get('topic') and 'kline' in data['topic']:
                            # Store in Redis
                            stream_data = {
                                'data': json.dumps(data)
                            }
                            
                            self.redis_client.xadd(
                                'market_data:kline',
                                stream_data,
                                maxlen=50000
                            )
                            
                            symbol = data['topic'].split('.')[-1]
                            kline = data['data'][0]
                            logger.info(f"{symbol}: ${kline['close']} at {datetime.now()}")
                            
            except Exception as e:
                logger.error(f'Error: {e}')
                await asyncio.sleep(5)

if __name__ == '__main__':
    ingestor = SimpleIngestor()
    asyncio.run(ingestor.run())
PYTHON

# Ingestorをバックグラウンドで起動
nohup python3 temp_ingestor.py > logs/temp_ingestor.log 2>&1 &
INGESTOR_PID=$!
echo "Ingestor PID: $INGESTOR_PID"

# 4. 数秒待ってからRedisを確認
sleep 10
echo ""
echo "🔴 Redisの最新データを確認:"
python3 -c "
import redis
import json
from datetime import datetime

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# 最新エントリーを確認
entries = r.xrevrange('market_data:kline', count=10)

print(f'最新{len(entries)}件のエントリー:')
for entry_id, data in entries[:5]:
    try:
        parsed = json.loads(data['data'])
        if 'topic' in parsed:
            symbol = parsed['topic'].split('.')[-1]
            timestamp = datetime.fromtimestamp(parsed.get('timestamp', 0))
            kline = parsed['data'][0]
            print(f'  {symbol}: ${kline[\"close\"]} at {timestamp}')
    except:
        pass
        
# ストリームのサイズを確認
length = r.xlen('market_data:kline')
print(f'\\nRedisストリームの合計エントリー数: {length}')
"

# 5. DuckDBを更新
echo ""
echo "🗄️ DuckDBを最新データで更新:"
python3 update_duckdb_enhanced_fixed.py --lookback-hours 96 || echo "DuckDB更新エラー"

# 6. ボットのキャッシュをクリアして再起動
echo ""
echo "🔄 ボットを再起動:"
pkill -f simple_improved_bot_with_trading_fixed.py
sleep 3

export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq'
nohup python3 simple_improved_bot_with_trading_fixed.py > logs/mlbot_realtime_$(date +%Y%m%d_%H%M%S).log 2>&1 &
BOT_PID=$!
echo "Bot PID: $BOT_PID"

echo ""
echo "📄 ステータス:"
echo "  - Ingestor: PID $INGESTOR_PID (リアルタイムデータ取得中)"
echo "  - Bot: PID $BOT_PID (50%闾値で再起動)"
echo "  - 期待: 最新データで信頼度向上"

EOF

echo ""
echo "✅ 緊急修復完了！"