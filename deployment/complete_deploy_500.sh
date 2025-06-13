#!/bin/bash
# 完全自動デプロイスクリプト ($500版)

set -e

echo "🚀 MLBot 完全自動デプロイ ($500版)"
echo "=================================="

# カラー定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 前提条件チェック
echo -e "\n${BLUE}📋 前提条件チェック${NC}"

# モデルファイル確認
if [ ! -f "../models/fast_nn_final.pth" ] || [ ! -f "../models/fast_nn_scaler.pkl" ]; then
    echo -e "${RED}❌ モデルファイルが見つかりません${NC}"
    echo "必要なファイル:"
    echo "  - models/fast_nn_final.pth"
    echo "  - models/fast_nn_scaler.pkl"
    exit 1
fi
echo -e "${GREEN}✅ モデルファイル確認済み${NC}"

# requirements.txt確認
if [ ! -f "../requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txtが見つかりません${NC}"
    exit 1
fi
echo -e "${GREEN}✅ requirements.txt確認済み${NC}"

# Git状態確認
cd ..
if ! git status &> /dev/null; then
    echo -e "${YELLOW}⚠️  Gitリポジトリではありません。ローカルデプロイを実行します${NC}"
    LOCAL_DEPLOY=true
else
    echo -e "${GREEN}✅ Gitリポジトリ確認済み${NC}"
    LOCAL_DEPLOY=false
fi
cd deployment

# ステップ1: AWS環境作成
echo -e "\n${BLUE}🏗️  ステップ1: AWS環境作成${NC}"
./quick_aws_setup_500.sh

# 接続情報読み込み
if [ ! -f "mlbot_connection_info_500.txt" ]; then
    echo -e "${RED}❌ 接続情報ファイルが見つかりません${NC}"
    exit 1
fi

# 接続情報抽出
PUBLIC_IP=$(grep "パブリックIP:" mlbot_connection_info_500.txt | cut -d' ' -f2)
KEY_FILE=$(grep "SSHキー:" mlbot_connection_info_500.txt | cut -d' ' -f2)

echo -e "${GREEN}✅ 接続情報取得: IP=$PUBLIC_IP${NC}"

# ステップ2: ファイルアップロード
echo -e "\n${BLUE}📤 ステップ2: ファイルアップロード${NC}"

# SSH接続テスト
echo "SSH接続テスト中..."
for i in {1..5}; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $KEY_FILE ubuntu@$PUBLIC_IP "echo 'Connection OK'" &> /dev/null; then
        echo -e "${GREEN}✅ SSH接続成功${NC}"
        break
    else
        echo -e "${YELLOW}接続試行 $i/5...${NC}"
        sleep 10
    fi
done

# モデルファイルアップロード
echo "モデルファイルをアップロード中..."
scp -o StrictHostKeyChecking=no -i $KEY_FILE ../models/fast_nn_final.pth ubuntu@$PUBLIC_IP:~/mlbot/models/
scp -o StrictHostKeyChecking=no -i $KEY_FILE ../models/fast_nn_scaler.pkl ubuntu@$PUBLIC_IP:~/mlbot/models/
echo -e "${GREEN}✅ モデルファイルアップロード完了${NC}"

# ローカルデプロイの場合、追加ファイルをアップロード
if [ "$LOCAL_DEPLOY" = true ]; then
    echo "ローカルファイルをアップロード中..."
    scp -o StrictHostKeyChecking=no -i $KEY_FILE ../requirements.txt ubuntu@$PUBLIC_IP:~/mlbot/
    # その他必要なファイルもアップロード
fi

# ステップ3: サーバーセットアップ
echo -e "\n${BLUE}⚙️  ステップ3: サーバーセットアップ${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'REMOTE'
set -e

cd ~/mlbot

echo "🔧 サーバーセットアップ開始..."

# セットアップスクリプト実行
if [ -f "../setup_mlbot_500.sh" ]; then
    bash ../setup_mlbot_500.sh
    echo "✅ 環境設定完了"
else
    echo "❌ セットアップスクリプトが見つかりません"
    exit 1
fi

# Secrets読み込み
source load_secrets.sh

# 環境変数確認
if [ -z "$BYBIT_API_KEY" ]; then
    echo "❌ APIキーが読み込まれていません"
    exit 1
fi
echo "✅ 環境変数確認済み"

# Dockerビルド・起動
echo "🐳 Dockerコンテナをビルド・起動中..."
docker-compose up --build -d

# 起動確認
sleep 30
if docker-compose ps | grep -q "Up"; then
    echo "✅ コンテナ起動成功"
else
    echo "❌ コンテナ起動失敗"
    docker-compose logs
    exit 1
fi

# ヘルスチェック
echo "🏥 ヘルスチェック中..."
for i in {1..10}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ ヘルスチェック成功"
        break
    else
        echo "ヘルスチェック試行 $i/10..."
        sleep 10
    fi
done

echo "🎉 デプロイ完了！"
REMOTE

# ステップ4: 最終確認
echo -e "\n${BLUE}✅ ステップ4: 最終確認${NC}"

# リモートでの動作確認
echo "最終動作確認中..."
ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'FINAL_CHECK'
cd ~/mlbot

echo "=== コンテナ状態 ==="
docker-compose ps

echo "=== 最新ログ ==="
docker-compose logs --tail 20 mlbot

echo "=== ディスク使用量 ==="
df -h

echo "=== メモリ使用量 ==="
free -h
FINAL_CHECK

# 成功メッセージ
echo -e "\n${GREEN}🎉 完全デプロイ成功！${NC}"
echo ""
echo "📊 接続情報:"
echo "   SSH: ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo "   ログ: docker-compose logs -f mlbot"
echo "   停止: docker-compose down"
echo ""
echo "💰 設定内容:"
echo "   初期資金: $500"
echo "   対象通貨: ICPUSDT"
echo "   ポジションサイズ: 5%"
echo "   レバレッジ: 2倍"
echo ""
echo "📱 Discord通知を確認してください！"
echo ""
echo "詳細は mlbot_connection_info_500.txt を参照"