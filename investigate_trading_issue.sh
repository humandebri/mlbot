#!/bin/bash

echo "🔍 取引エラーとシグナル問題を調査..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. 現在稼働中のボットを確認
echo "🤖 稼働中のボットプロセス:"
ps aux | grep simple_improved_bot | grep -v grep
echo ""

# 2. 現在の信頼度閾値を確認
echo "📊 現在の信頼度閾値設定:"
grep "self.min_confidence =" simple_improved_bot_with_trading_fixed.py | head -2
echo ""

# 3. 最近のシグナル送信履歴を確認
echo "📨 最近のDiscordシグナル送信:"
find logs -name "*.log" -mtime -1 -exec grep -l "ML Signal" {} \; | while read log; do
    echo "File: $log"
    grep -A2 -B2 "ML Signal" "$log" | tail -20
    echo "---"
done | tail -50
echo ""

# 4. データベースエラーを確認
echo "❌ データベースエラー:"
find logs -name "*.log" -mtime -1 -exec grep -l "TransactionContext Error" {} \; | while read log; do
    echo "File: $log"
    grep -A5 -B5 "TransactionContext Error" "$log" | tail -30
    echo "---"
done
echo ""

# 5. 実際の取引実行を確認
echo "💰 実際の取引実行:"
find logs -name "*.log" -mtime -1 -exec grep -E "(Executing.*order|order_result|orderId)" {} \; | tail -20
echo ""

# 6. 複数のボットが稼働していないか確認
echo "⚠️ 複数ボット確認:"
PID_COUNT=$(ps aux | grep simple_improved_bot | grep -v grep | wc -l)
if [ "$PID_COUNT" -gt 1 ]; then
    echo "警告: $PID_COUNT 個のボットが稼働中！"
    ps aux | grep simple_improved_bot | grep -v grep
else
    echo "正常: 1個のボットのみ稼働中"
fi
echo ""

# 7. 古いボットプロセスの確認
echo "🕐 ボットプロセスの起動時間:"
ps aux | grep -E "(simple_improved|mlbot)" | grep -v grep | awk '{print $2, $9, $11}'
echo ""

# 8. 最新ログの信頼度設定を確認
echo "📋 最新ログの信頼度:"
LATEST_LOG=$(ls -t logs/mlbot_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Latest log: $LATEST_LOG"
    grep -E "(min_confidence|conf=4[3-9]|conf=5[0-9])" "$LATEST_LOG" | tail -10
fi
echo ""

# 9. データベース接続状態を確認
echo "🗄️ DuckDBファイルの状態:"
ls -la data/historical_data.duckdb*
lsof data/historical_data.duckdb 2>/dev/null || echo "No processes accessing DuckDB"
echo ""

# 10. トランザクションエラーの詳細
echo "🔍 トランザクションエラーの詳細分析:"
grep -B10 -A10 "TransactionContext Error" logs/mlbot_fixed_*.log 2>/dev/null | tail -50

EOF

echo ""
echo "✅ 調査完了！"