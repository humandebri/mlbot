#!/bin/bash
# 正規化統合済み最終版システムのデプロイスクリプト

EC2_USER="ubuntu"
EC2_HOST="13.212.91.54"
EC2_KEY="~/.ssh/mlbot-key-1749802416.pem"
PROJECT_DIR="/home/ubuntu/mlbot"

echo "🚀 正規化統合済み最終版システムのデプロイ開始..."

# 1. 重要ファイルをEC2に転送
echo "📦 ファイル転送中..."

# 手動スケーラーJSONファイル
scp -i $EC2_KEY models/v3.1_improved/manual_scaler.json $EC2_USER@$EC2_HOST:$PROJECT_DIR/models/v3.1_improved/

# 最終版Pythonスクリプト
scp -i $EC2_KEY final_normalized_trading_system.py $EC2_USER@$EC2_HOST:$PROJECT_DIR/
scp -i $EC2_KEY fix_scaler_and_normalization.py $EC2_USER@$EC2_HOST:$PROJECT_DIR/

echo "✅ ファイル転送完了"

# 2. EC2上で最終版システムを実行
echo "🔧 EC2上で最終版システムを起動..."

ssh -i $EC2_KEY $EC2_USER@$EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 既存のプロセスを停止
echo "🛑 既存プロセスを停止中..."
pkill -f "python.*mlbot" || true
pkill -f "python.*trading" || true

# 少し待機
sleep 2

# 最終版システムを起動
echo "🚀 正規化統合済み最終版システムを起動..."
nohup python3 final_normalized_trading_system.py > final_system.log 2>&1 &

# プロセス確認
sleep 3
ps aux | grep -E "python.*final_normalized" | grep -v grep

# ログの最初の部分を表示
echo "📋 システムログ:"
tail -n 20 final_system.log

echo "✅ デプロイ完了"
EOF

echo "🎉 正規化統合済み最終版システムのデプロイが完了しました！"
echo "📊 Discord通知を確認してください"