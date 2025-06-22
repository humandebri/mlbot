#!/bin/bash

echo "🔧 ボットのレポート機能を修正..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. 現在のボットを停止
echo "🛑 現在のボットを停止..."
pkill -f simple_improved_bot_with_trading_fixed.py
sleep 3

# 2. コードを修正
echo "📝 コードを修正..."
cp simple_improved_bot_with_trading_fixed.py simple_improved_bot_with_trading_fixed.py.bak

# sedで修正（予測回数をself.prediction_countに変更）
sed -i 's/f"• 予測回数: {len(recent_preds)}"/f"• 予測回数: {self.prediction_count}"/g' simple_improved_bot_with_trading_fixed.py

# 3. ボットを再起動
echo "🚀 ボットを再起動..."
export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq'
nohup python3 simple_improved_bot_with_trading_fixed.py > logs/mlbot_fixed_$(date +%Y%m%d_%H%M%S).log 2>&1 &
NEW_PID=$!

echo "✅ ボット再起動完了 (PID: $NEW_PID)"

# 4. 修正の確認
echo ""
echo "📄 修正内容の確認:"
grep -n "予測回数" simple_improved_bot_with_trading_fixed.py | grep prediction_count

echo ""
echo "✅ 修正完了！次回のレポートから正しい予測回数が表示されます"

EOF

echo ""
echo "✅ 完了！"