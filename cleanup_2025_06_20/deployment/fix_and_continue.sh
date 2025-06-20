#!/bin/bash
# デプロイ修復・継続スクリプト

set -e

echo "🔧 デプロイ修復・継続"
echo "===================="

# カラー定義
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 接続情報読み込み
if [ ! -f "mlbot_connection_info_500.txt" ]; then
    echo "❌ 接続情報ファイルが見つかりません"
    exit 1
fi

PUBLIC_IP=$(grep "パブリックIP:" mlbot_connection_info_500.txt | cut -d' ' -f2)
KEY_FILE=$(grep "SSHキー:" mlbot_connection_info_500.txt | cut -d' ' -f2)

echo "接続先: $PUBLIC_IP"

# 1. ディレクトリ作成と初期化確認
echo -e "\n${BLUE}1. サーバー初期化確認${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'INIT_CHECK'
echo "🔍 初期化状態確認中..."

# Gitクローン確認
if [ ! -d "/home/ubuntu/mlbot" ]; then
    echo "📥 プロジェクトをクローン中..."
    cd /home/ubuntu
    git clone https://github.com/humandebri/mlbot.git
    chown -R ubuntu:ubuntu mlbot
fi

# ディレクトリ作成
cd /home/ubuntu/mlbot
mkdir -p models data logs
echo "✅ ディレクトリ作成完了"

# 権限確認
ls -la
INIT_CHECK

# 2. モデルファイルアップロード（リトライ）
echo -e "\n${BLUE}2. モデルファイルアップロード${NC}"

echo "モデルファイルを再アップロード中..."
scp -o StrictHostKeyChecking=no -i $KEY_FILE ../models/fast_nn_final.pth ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/models/
scp -o StrictHostKeyChecking=no -i $KEY_FILE ../models/fast_nn_scaler.pkl ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/models/
echo -e "${GREEN}✅ モデルファイルアップロード成功${NC}"

# requirements.txtもアップロード
scp -o StrictHostKeyChecking=no -i $KEY_FILE ../requirements.txt ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/
echo -e "${GREEN}✅ requirements.txtアップロード成功${NC}"

# 3. セットアップ継続
echo -e "\n${BLUE}3. セットアップ継続${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'CONTINUE_SETUP'
set -e

cd /home/ubuntu/mlbot

# セットアップスクリプト実行
if [ -f "../setup_mlbot_500.sh" ]; then
    echo "🔧 セットアップスクリプト実行中..."
    bash ../setup_mlbot_500.sh
    echo "✅ 環境設定完了"
fi

# Secrets読み込み
echo "🔑 環境変数を読み込み中..."
source load_secrets.sh

# 環境変数確認
if [ -z "$BYBIT_API_KEY" ]; then
    echo "⚠️  APIキーが読み込まれていません"
else
    echo "✅ APIキー確認: ${BYBIT_API_KEY:0:8}..."
fi

if [ -z "$DISCORD_WEBHOOK" ]; then
    echo "⚠️  Discord Webhookが設定されていません"
else
    echo "✅ Discord確認: ${DISCORD_WEBHOOK:0:30}..."
fi

# Dockerインストール確認
if ! command -v docker &> /dev/null; then
    echo "🐳 Dockerをインストール中..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    newgrp docker
fi

# Docker Composeインストール確認
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Docker Composeをインストール中..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

echo "✅ Docker環境準備完了"
CONTINUE_SETUP

# 4. Docker起動
echo -e "\n${BLUE}4. Dockerコンテナ起動${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'DOCKER_START'
cd /home/ubuntu/mlbot

# 環境変数を再読み込み
source load_secrets.sh

# Dockerビルド・起動
echo "🐳 Dockerコンテナをビルド中..."
sudo docker-compose up --build -d

# 起動確認
sleep 20
echo "📊 コンテナ状態:"
sudo docker-compose ps

# ログ確認
echo "📝 最新ログ:"
sudo docker-compose logs --tail 30 mlbot
DOCKER_START

# 5. 最終確認
echo -e "\n${BLUE}5. 最終確認${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'FINAL'
cd /home/ubuntu/mlbot

echo "🏥 ヘルスチェック..."
for i in {1..5}; do
    if curl -s http://localhost:8080/health; then
        echo "✅ ヘルスチェック成功"
        break
    else
        echo "試行 $i/5..."
        sleep 10
    fi
done

echo ""
echo "📊 システム状態:"
sudo docker-compose ps
echo ""
echo "💾 ディスク使用量:"
df -h /
echo ""
echo "🧠 メモリ使用量:"
free -h
FINAL

echo -e "\n${GREEN}✅ デプロイ修復完了！${NC}"
echo ""
echo "📱 Discord通知を確認してください"
echo "📊 ログ監視: ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo "   → sudo docker-compose logs -f mlbot"
echo ""
echo "🎯 設定内容:"
echo "   初期資金: \$500"
echo "   対象通貨: ICPUSDT"
echo "   ポジションサイズ: 5%"
echo "   レバレッジ: 2倍"