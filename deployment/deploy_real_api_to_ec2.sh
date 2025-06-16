#!/bin/bash
# EC2に実際のAPI統合をデプロイ

EC2_USER="ubuntu"
EC2_HOST="13.212.91.54"
EC2_KEY="~/.ssh/mlbot-key-1749802416.pem"

echo "🚀 EC2に実際のBybit API統合をデプロイ..."

# 1. ファイルをEC2に転送
echo "📤 ファイルを転送中..."
scp -i $EC2_KEY trading_with_real_api.py $EC2_USER@$EC2_HOST:/home/ubuntu/mlbot/
scp -i $EC2_KEY src/common/config.py $EC2_USER@$EC2_HOST:/home/ubuntu/mlbot/src/common/
scp -i $EC2_KEY src/common/account_monitor.py $EC2_USER@$EC2_HOST:/home/ubuntu/mlbot/src/common/

# 2. EC2で実行
ssh -i $EC2_KEY $EC2_USER@$EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 既存のプロセスを停止
echo "🛑 既存のプロセスを停止..."
pkill -f "trading_with_balance_fixed" || true
sleep 3

# 新しいシステムを起動
echo "🚀 実際のAPI統合システムを起動..."
nohup python3 trading_with_real_api.py > real_api_trading.log 2>&1 &

sleep 5

# 起動確認
echo -e "\n✅ 起動確認:"
ps aux | grep python | grep real_api | grep -v grep

# ログ確認
echo -e "\n📄 初期ログ:"
tail -30 real_api_trading.log | grep -E "balance|equity|API|kelly" | tail -15

echo -e "\n✅ EC2で実際のAPI統合が動作開始！"
EOF

echo ""
echo "🎉 EC2デプロイ完了！"
echo "📊 ログ監視: ssh -i $EC2_KEY $EC2_USER@$EC2_HOST 'tail -f /home/ubuntu/mlbot/real_api_trading.log | grep -E \"balance|position|kelly\"'"