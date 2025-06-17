#!/bin/bash
# EC2に最新コードをデプロイして24時間稼働させる

EC2_HOST="ubuntu@13.212.91.54"
SSH_KEY="~/.ssh/mlbot-key-1749802416.pem"

echo "🚀 EC2にトレーディングシステムをデプロイ中..."

# 1. 最新コードを同期
echo "📦 コードを同期中..."
rsync -avz --exclude='.env' --exclude='__pycache__' --exclude='.venv' \
    -e "ssh -i $SSH_KEY" \
    . $EC2_HOST:/home/ubuntu/mlbot/

# 2. 環境変数を更新
echo "🔧 環境設定中..."
ssh -i $SSH_KEY $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# .envファイルの確認と更新
if [ ! -f .env ]; then
    cp .env.example .env
fi

# APIキーの確認
grep -q "BYBIT__API_KEY" .env || echo "⚠️ APIキーの設定が必要です"

# 本番環境設定
sed -i 's/BYBIT__TESTNET=true/BYBIT__TESTNET=false/g' .env
sed -i 's/ENVIRONMENT=development/ENVIRONMENT=production/g' .env

echo "✅ 環境設定完了"
EOF

# 3. Dockerコンテナで起動
echo "🐳 Dockerコンテナを起動中..."
ssh -i $SSH_KEY $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 既存のコンテナを停止
docker-compose down

# Dockerイメージを再ビルド
docker-compose build

# バックグラウンドで起動
docker-compose up -d

# ログ確認
echo "📊 起動状態:"
docker-compose ps
echo ""
echo "📝 最新ログ:"
docker-compose logs --tail=20
EOF

echo "✅ デプロイ完了！"
echo "🌐 EC2で24時間稼働中: http://13.212.91.54:8080/system/health"
echo "📊 ログ確認: ssh -i $SSH_KEY $EC2_HOST 'cd /home/ubuntu/mlbot && docker-compose logs -f'"