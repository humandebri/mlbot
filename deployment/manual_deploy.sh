#!/bin/bash
# 手動デプロイスクリプト（Git不要）

set -e

echo "🔧 手動デプロイ（Gitクローン不要）"
echo "=================================="

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

# 1. ディレクトリ構造を手動作成
echo -e "\n${BLUE}1. ディレクトリ構造作成${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'CREATE_DIRS'
echo "📁 ディレクトリ構造を作成中..."

# mlbotディレクトリ作成
mkdir -p /home/ubuntu/mlbot/{models,data,logs,src/common,src/system,scripts}

# 権限設定
chown -R ubuntu:ubuntu /home/ubuntu/mlbot

echo "✅ ディレクトリ作成完了"
ls -la /home/ubuntu/mlbot/
CREATE_DIRS

# 2. 必要なファイルをアップロード
echo -e "\n${BLUE}2. ファイルアップロード${NC}"

# モデルファイル
echo "📤 モデルファイル..."
scp -o StrictHostKeyChecking=no -i $KEY_FILE ../models/fast_nn_final.pth ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/models/
scp -o StrictHostKeyChecking=no -i $KEY_FILE ../models/fast_nn_scaler.pkl ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/models/

# requirements.txt
echo "📤 requirements.txt..."
scp -o StrictHostKeyChecking=no -i $KEY_FILE ../requirements.txt ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/

# Dockerfile
echo "📤 Dockerfile..."
scp -o StrictHostKeyChecking=no -i $KEY_FILE ../Dockerfile ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/

# 主要なソースファイルをアップロード
echo "📤 ソースファイル..."
# src/common配下
for file in ../src/common/*.py; do
    if [ -f "$file" ]; then
        scp -o StrictHostKeyChecking=no -i $KEY_FILE "$file" ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/src/common/
    fi
done

# src/system/main.py（存在する場合）
if [ -f "../src/system/main.py" ]; then
    scp -o StrictHostKeyChecking=no -i $KEY_FILE ../src/system/main.py ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/src/system/
fi

# scriptsディレクトリから必要なファイル
if [ -f "../scripts/fast_nn_model.py" ]; then
    scp -o StrictHostKeyChecking=no -i $KEY_FILE ../scripts/fast_nn_model.py ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/scripts/
fi

echo -e "${GREEN}✅ ファイルアップロード完了${NC}"

# 3. 最小限のmain.pyを作成（存在しない場合）
echo -e "\n${BLUE}3. エントリーポイント作成${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'CREATE_MAIN'
cd /home/ubuntu/mlbot

# 最小限のmain.pyを作成
if [ ! -f "src/system/main.py" ]; then
    cat > src/system/main.py << 'MAIN_PY'
#!/usr/bin/env python3
"""
MLBot Main Entry Point
"""
import os
import asyncio
from datetime import datetime

print(f"🚀 MLBot Starting - {datetime.now()}")
print(f"📊 Initial Capital: $500")
print(f"🎯 Target: ICPUSDT")
print(f"⚙️  Environment: {os.getenv('ENVIRONMENT', 'development')}")

# Discord通知テスト
discord_webhook = os.getenv('DISCORD_WEBHOOK')
if discord_webhook:
    print(f"📱 Discord Webhook: {discord_webhook[:50]}...")
else:
    print("⚠️  Discord Webhook not configured")

# メインループ
async def main():
    print("✅ Bot is running...")
    while True:
        await asyncio.sleep(60)
        print(f"💓 Heartbeat - {datetime.now()}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped")
MAIN_PY
fi

# 最小限の__init__.pyを作成
touch src/__init__.py
touch src/common/__init__.py
touch src/system/__init__.py

echo "✅ エントリーポイント作成完了"
CREATE_MAIN

# 4. セットアップ継続
echo -e "\n${BLUE}4. セットアップ継続${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'CONTINUE_SETUP'
cd /home/ubuntu/mlbot

# セットアップスクリプト実行
if [ -f "../setup_mlbot_500.sh" ]; then
    echo "🔧 セットアップスクリプト実行中..."
    bash ../setup_mlbot_500.sh
fi

# Secrets読み込み確認
if [ -f "load_secrets.sh" ]; then
    source load_secrets.sh
    echo "✅ 環境変数読み込み完了"
fi

# Docker環境確認
if ! command -v docker &> /dev/null; then
    echo "🐳 Dockerインストール中..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
fi

echo "✅ セットアップ完了"
CONTINUE_SETUP

# 5. Docker起動（sudoで実行）
echo -e "\n${BLUE}5. Docker起動${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'DOCKER_START'
cd /home/ubuntu/mlbot

# 環境変数を再読み込み
if [ -f "load_secrets.sh" ]; then
    source load_secrets.sh
    export BYBIT_API_KEY
    export BYBIT_API_SECRET
    export DISCORD_WEBHOOK
fi

# Docker起動（初回なのでsudo必要）
echo "🐳 Dockerコンテナ起動中..."
sudo -E docker-compose up --build -d

# 状態確認
sleep 15
echo "📊 コンテナ状態:"
sudo docker ps

echo "📝 ログ確認:"
sudo docker-compose logs --tail 20
DOCKER_START

echo -e "\n${GREEN}✅ 手動デプロイ完了！${NC}"
echo ""
echo "SSH接続: ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo "ログ確認: sudo docker-compose logs -f mlbot"
echo ""
echo "⚠️  注意: 最小限の構成でデプロイしました"
echo "完全な機能を使用するには、GitHubリポジトリの修正が必要です"