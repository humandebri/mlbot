#!/bin/bash
# 取引システムの現在の状態を確認

EC2_USER="ubuntu"
EC2_HOST="13.212.91.54"
EC2_KEY="~/.ssh/mlbot-key-1749802416.pem"

echo "🔍 取引システムの状態確認..."

ssh -i $EC2_KEY $EC2_USER@$EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

echo "📊 実行中のプロセス:"
echo "=================="
ps aux | grep python | grep -E "(unified|trading|normalized)" | grep -v grep | awk '{print $2, $11, $12}'

echo -e "\n🔧 レバレッジ設定:"
echo "=================="
grep -n "max_leverage" src/order_router/risk_manager.py | head -3

echo -e "\n📈 最新の取引活動（直近100行から）:"
echo "=================="
tail -100 /home/ubuntu/mlbot/unified_trading.log | grep -E "signal|position|order|trade" -i | tail -10

echo -e "\n🚨 エラーログ:"
echo "=================="
tail -100 /home/ubuntu/mlbot/unified_trading.log | grep -E "ERROR|error|Exception" | tail -5

echo -e "\n📊 システムメトリクス:"
echo "=================="
tail -200 /home/ubuntu/mlbot/unified_trading.log | grep -E "msg/s|features/s|predictions" | tail -5

echo -e "\n🔔 Discord通知:"
echo "=================="
tail -100 /home/ubuntu/mlbot/unified_trading.log | grep -i "discord" | tail -5

# FeatureHubの状態確認
echo -e "\n🧮 FeatureHub状態:"
echo "=================="
grep -i "featurehub\|feature hub" /home/ubuntu/mlbot/unified_trading.log | tail -5 || echo "FeatureHubログが見つかりません"

# Trading loopの状態確認  
echo -e "\n🔄 Trading Loop状態:"
echo "=================="
grep -i "trading loop\|trading_loop" /home/ubuntu/mlbot/unified_trading.log | tail -5 || echo "Trading loopログが見つかりません"

EOF