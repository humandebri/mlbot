#!/bin/bash
# GitHubが公開されたので再デプロイ

set -e

echo "🚀 再デプロイ（GitHub公開版）"
echo "============================="

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

# 1. 既存ディレクトリクリーンアップとGitクローン
echo -e "\n${BLUE}1. GitHubから最新コードを取得${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'GIT_CLONE'
echo "🧹 既存ディレクトリをクリーンアップ..."
cd /home/ubuntu
rm -rf mlbot

echo "📥 GitHubからクローン中..."
git clone https://github.com/humandebri/mlbot.git
cd mlbot

echo "✅ クローン成功！"
ls -la

# ディレクトリ作成
mkdir -p models data logs
chown -R ubuntu:ubuntu /home/ubuntu/mlbot
GIT_CLONE

# 2. モデルファイルアップロード
echo -e "\n${BLUE}2. モデルファイルアップロード${NC}"

echo "📤 モデルファイル転送中..."
scp -o StrictHostKeyChecking=no -i $KEY_FILE ../models/fast_nn_final.pth ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/models/
scp -o StrictHostKeyChecking=no -i $KEY_FILE ../models/fast_nn_scaler.pkl ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/models/
echo -e "${GREEN}✅ モデルファイル転送完了${NC}"

# 3. セットアップ実行
echo -e "\n${BLUE}3. セットアップ実行${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'SETUP'
cd /home/ubuntu/mlbot

# セットアップスクリプト実行
if [ -f "../setup_mlbot_500.sh" ]; then
    echo "🔧 $500版セットアップ実行中..."
    bash ../setup_mlbot_500.sh
fi

# 環境変数読み込み
source load_secrets.sh

# 確認
echo "✅ 環境変数確認:"
echo "   API Key: ${BYBIT_API_KEY:0:8}..."
echo "   Discord: ${DISCORD_WEBHOOK:0:30}..."

# Docker環境確認
if ! command -v docker &> /dev/null; then
    echo "🐳 Dockerインストール中..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    echo "⚠️  Dockerグループ追加のため再ログインが必要です"
fi
SETUP

# 4. Docker起動
echo -e "\n${BLUE}4. Dockerコンテナ起動${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'DOCKER_START'
cd /home/ubuntu/mlbot

# 環境変数を再読み込み
source load_secrets.sh

# Dockerビルド・起動（初回はsudo必要）
echo "🐳 Dockerイメージをビルド中..."
sudo -E docker-compose up --build -d

# 起動待機
echo "⏳ 起動待機中..."
sleep 30

# 状態確認
echo "📊 コンテナ状態:"
sudo docker-compose ps

echo "📝 最新ログ:"
sudo docker-compose logs --tail 50 mlbot

# ヘルスチェック
echo "🏥 ヘルスチェック:"
curl -s http://localhost:8080/health || echo "ヘルスチェックエンドポイントは未実装"
DOCKER_START

# 5. 最終確認
echo -e "\n${BLUE}5. 最終確認${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'FINAL_CHECK'
cd /home/ubuntu/mlbot

echo "📂 ファイル構成:"
ls -la models/

echo ""
echo "🐳 実行中のコンテナ:"
sudo docker ps

echo ""
echo "💾 リソース使用状況:"
df -h /
free -h

echo ""
echo "🔍 プロセス確認:"
ps aux | grep -E "docker|mlbot" | grep -v grep || true
FINAL_CHECK

echo -e "\n${GREEN}🎉 再デプロイ完了！${NC}"
echo ""
echo "📱 Discord通知を確認してください"
echo ""
echo "🔧 管理コマンド:"
echo "  SSH接続: ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo "  ログ確認: sudo docker-compose logs -f mlbot"
echo "  再起動: sudo docker-compose restart mlbot"
echo "  停止: sudo docker-compose down"
echo ""
echo "💰 設定:"
echo "  初期資金: \$500"
echo "  通貨ペア: ICPUSDT"
echo "  レバレッジ: 2倍"