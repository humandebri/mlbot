#!/bin/bash

echo "🎯 信頼度閾値を50%に戻します..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. 現在のボットを停止
echo "⏹️  現在のボットを停止中..."
pkill -f simple_improved_bot_with_trading_fixed.py || true
sleep 5

# 2. 信頼度閾値を50%に戻す
echo ""
echo "📝 信頼度閾値を50%に変更..."
sed -i 's/self.min_confidence = 0.[0-9]\+/self.min_confidence = 0.50/g' simple_improved_bot_with_trading_fixed.py

# 変更を確認
echo "変更確認:"
grep "self.min_confidence =" simple_improved_bot_with_trading_fixed.py | head -2

# 3. ボットを再起動
echo ""
echo "🚀 50%閾値でボットを再起動..."
export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq'
nohup python3 simple_improved_bot_with_trading_fixed.py > logs/mlbot_50pct_$(date +%Y%m%d_%H%M%S).log 2>&1 &
BOT_PID=$!
echo "Bot PID: $BOT_PID"

# 4. 初期状態を確認
echo ""
echo "⏰ ボットの初期化を待機中..."
sleep 20

# 5. 現在の予測状況を確認
echo ""
echo "📊 50%閾値での予測状況:"
LATEST_LOG=$(ls -t logs/mlbot_50pct_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo "最新の予測値:"
    tail -100 "$LATEST_LOG" | grep -E "conf=" | tail -15
    
    echo ""
    echo "🎯 50%以上の信頼度チェック:"
    HIGH_CONF=$(tail -200 "$LATEST_LOG" | grep -E "conf=[5-9][0-9]\.[0-9]+%" | wc -l)
    
    if [ "$HIGH_CONF" -gt 0 ]; then
        echo "✅ 50%以上の予測が見つかりました！"
        tail -200 "$LATEST_LOG" | grep -E "conf=[5-9][0-9]\.[0-9]+%" | tail -5
    else
        echo "⚠️  現在50%以上の予測はありません"
        echo "最高信頼度:"
        tail -200 "$LATEST_LOG" | grep -oE "conf=[0-9]+\.[0-9]+%" | sort -t= -k2 -nr | head -5
    fi
fi

# 6. 自動更新の状態確認
echo ""
echo "🔄 自動更新ステータス:"
AUTO_PID=$(ps aux | grep auto_update_duckdb.py | grep -v grep | awk '{print $2}')
if [ -n "$AUTO_PID" ]; then
    echo "✅ 自動更新は稼働中 (PID: $AUTO_PID)"
else
    echo "⚠️  自動更新が停止しています"
fi

# 7. サマリー
echo ""
echo "📄 設定サマリー:"
echo "  - 信頼度閾値: 50% (本来の設定に復元)"
echo "  - Bot PID: $BOT_PID"
echo "  - シグナル生成: 50%以上の信頼度が必要"
echo "  - 現在の最高信頼度: ~43%"
echo ""
echo "💡 次のステップ:"
echo "  1. DuckDBのデータを最新化して信頼度向上"
echo "  2. より長い履歴期間の使用を検討"
echo "  3. 特徴量生成の最適化"
EOF

echo ""
echo "✅ 信頼度閾値を50%に戻しました！"