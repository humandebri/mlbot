#!/bin/bash

echo "🔍 Discordレポートの問題を調査..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. レポート生成ログを確認
echo "📄 最近の時間別レポート:"
grep -A20 -B5 "send_hourly_report" logs/mlbot_*.log 2>/dev/null | tail -100

# 2. ポジションモニターの詳細ログ
echo ""
echo "💰 ポジションモニターの詳細:"
grep -E "(Position:|symbol|size|avgPrice|unrealisedPnl)" logs/mlbot_*.log 2>/dev/null | tail -30

# 3. シグナルカウントを確認
echo ""
echo "🎯 シグナルカウント確認:"
grep -E "self.signal_count" logs/mlbot_*.log 2>/dev/null | tail -20

# 4. ポジション情報を直接取得
echo ""
echo "🔄 現在のポジションをAPIで確認:"
python3 -c "
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

# 環境変数を直接設定
os.environ['BYBIT__API_KEY'] = 'KgMS2YHiCPG49hmWuV'
os.environ['BYBIT__API_SECRET'] = 'LUkW6d5SHHIrbu2SvEDPjFwGYkyUGRun8JF3'

async def check_positions():
    import sys
    sys.path.insert(0, '.')
    from src.common.bybit_client import BybitRESTClient
    
    client = BybitRESTClient(testnet=False)
    await client.__aenter__()
    
    try:
        # ポジション情報取得
        positions = await client.get_open_positions()
        
        print('ポジション情報:')
        if positions:
            for pos in positions:
                print(f\"  {pos.get('symbol')}: {pos.get('size')} units\")
                print(f\"    エントリー: ${pos.get('avgPrice')}\")
                print(f\"    未実現損益: ${pos.get('unrealisedPnl', 0):.4f}\")
                print(f\"    サイド: {pos.get('side')}\")
        else:
            print('  ポジションなし')
            
        # アカウント情報
        account = await client.get_account_info()
        if account:
            print(f\"\nアカウント情報:\")
            print(f\"  バランス: ${account.get('totalEquity', 0):.2f}\")
            print(f\"  未実現損益合計: ${account.get('totalPerpUPL', 0):.4f}\")
        
    finally:
        await client.__aexit__(None, None, None)

asyncio.run(check_positions())
"

# 5. send_hourly_reportメソッドを確認
echo ""
echo "📝 send_hourly_reportメソッドの実装:"
grep -A30 "def send_hourly_report" simple_improved_bot_with_trading_fixed.py | head -35

# 6. データベースからのポジション取得を確認
echo ""
echo "🗾 データベースポジション取得エラー:"
grep -E "(get_open_positions_from_db|database\.py)" logs/mlbot_*.log 2>/dev/null | tail -20

EOF

echo ""
echo "✅ 調査完了！"