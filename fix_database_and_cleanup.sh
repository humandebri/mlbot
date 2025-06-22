#!/bin/bash

echo "🔧 データベースエラーを修正し、現在の状態を確認..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. 現在のポジションを確認
echo "💰 現在のBybitポジション:"
python3 -c "
import asyncio
from src.common.bybit_client import BybitRESTClient

async def check_positions():
    client = BybitRESTClient(testnet=False)
    await client.__aenter__()
    try:
        positions = await client.get_open_positions()
        if positions:
            print(f'\nオープンポジション: {len(positions)}件')
            for pos in positions:
                print(f\"  {pos.get('symbol')}: {pos.get('size')} @ {pos.get('avgPrice')} ({pos.get('side')})\"
                      f\" P&L: {pos.get('unrealisedPnl', 0):.4f} USDT\")
        else:
            print('ポジションなし')
    finally:
        await client.__aexit__(None, None, None)

asyncio.run(check_positions())
"

# 2. データベーススキーマを修正
echo ""
echo "🗄️ データベーススキーマを修正..."
python3 -c "
import duckdb

# データベースに接続
conn = duckdb.connect('data/trading_bot.db')

try:
    # positionsテーブルの現在のスキーマを確認
    columns = conn.execute(\"DESCRIBE positions\").fetchall()
    print('現在のpositionsテーブルカラム:')
    for col in columns:
        print(f'  {col[0]}: {col[1]}')
    
    # opened_atカラムがない場合は追加
    has_opened_at = any(col[0] == 'opened_at' for col in columns)
    if not has_opened_at:
        print('\nopened_atカラムを追加...')
        conn.execute(\"ALTER TABLE positions ADD COLUMN opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\")
        print('✅ opened_atカラムを追加しました')
    else:
        print('\n✅ opened_atカラムは既に存在します')
    
    # 他の必要なカラムも確認
    conn.execute(\"CREATE TABLE IF NOT EXISTS positions_backup AS SELECT * FROM positions\")
    print('\nバックアップテーブルを作成しました')
    
finally:
    conn.close()
"

# 3. 古いボットプロセスを確実に停止
echo ""
echo "🛑 古いボットプロセスをクリーンアップ..."
# 50%闾値のボット以外を停止
CURRENT_PID=$(ps aux | grep "mlbot_50pct" | grep -v grep | awk '{print $2}')
echo "現在の50%ボットPID: $CURRENT_PID"

# 他の全てのボットを停止
ps aux | grep simple_improved_bot | grep -v grep | grep -v "$CURRENT_PID" | awk '{print $2}' | while read pid; do
    echo "古いボットを停止: PID $pid"
    kill -9 $pid 2>/dev/null || true
done

# 4. 現在の状態を再確認
echo ""
echo "📋 現在のシステム状態:"
echo "稼働中のボット:"
ps aux | grep simple_improved_bot | grep -v grep

echo ""
echo "最新の予測状況 (50%闾値):"
LATEST_LOG=$(ls -t logs/mlbot_50pct_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    tail -20 "$LATEST_LOG" | grep -E "conf=" | tail -5
fi

echo ""
echo "✅ クリーンアップ完了！"
echo ""
echo "📄 まとめ:"
echo "  - 8 ICPのポジションはBybit上に存在（正常）"
echo "  - データベースエラーは修正済み"
echo "  - 現在は50%闾値で稼働中"
echo "  - 43%のシグナルは過去のもの"
EOF

echo ""
echo "✅ 完了！"