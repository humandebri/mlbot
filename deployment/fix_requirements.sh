#!/bin/bash
# requirements.txt修正と再ビルド

set -e

echo "🔧 requirements.txt修正と再ビルド"
echo "================================"

# カラー定義
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 接続情報
PUBLIC_IP="13.212.91.54"
KEY_FILE="~/.ssh/mlbot-key-1749802416.pem"

echo "接続先: $PUBLIC_IP"

# 1. requirements.txtを修正
echo -e "\n${BLUE}1. requirements.txt修正${NC}"

# 正しいrequirements.txtを作成
cat > ../requirements_fixed.txt << 'REQ'
# Core dependencies
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
torch>=2.0.0
joblib>=1.3.0

# API and networking
aiohttp>=3.8.0
websockets>=11.0.0
requests>=2.31.0
discord-webhook>=1.3.0

# Data processing
duckdb>=0.8.0
redis>=4.5.0

# Machine learning
lightgbm>=4.0.0
catboost>=1.2.0
optuna>=3.3.0
shap>=0.42.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
fastapi>=0.100.0
uvicorn>=0.23.0

# Monitoring
prometheus-client>=0.17.0
rich>=13.5.0

# Japanese font support
japanize-matplotlib>=1.1.3

# AWS
boto3>=1.28.0

# Data science
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Statistics
statsmodels>=0.14.0
numba>=0.57.0
REQ

# アップロード
echo "📤 修正したrequirements.txtをアップロード..."
scp -o StrictHostKeyChecking=no -i $KEY_FILE ../requirements_fixed.txt ubuntu@$PUBLIC_IP:/home/ubuntu/mlbot/requirements.txt

# 2. サーバーで再ビルド
echo -e "\n${BLUE}2. Docker再ビルド${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'REBUILD'
cd /home/ubuntu/mlbot

# requirements.txt確認
echo "📋 requirements.txt確認:"
tail -5 requirements.txt

# 環境変数読み込み
source load_secrets.sh

# クリーンビルド
echo "🧹 既存のコンテナ・イメージを削除..."
sudo docker-compose down
sudo docker system prune -f

# 再ビルド
echo "🐳 Docker再ビルド中..."
sudo -E docker-compose up --build -d

# 起動待機
echo "⏳ 起動待機中..."
sleep 30

# 状態確認
echo "📊 コンテナ状態:"
sudo docker-compose ps

echo "📝 最新ログ:"
sudo docker-compose logs --tail 50 mlbot

# 実行中のコンテナ確認
echo "🐳 実行中のコンテナ:"
sudo docker ps
REBUILD

# 3. Discord通知確認
echo -e "\n${BLUE}3. 最終確認${NC}"

ssh -o StrictHostKeyChecking=no -i $KEY_FILE ubuntu@$PUBLIC_IP << 'FINAL'
cd /home/ubuntu/mlbot

# プロセス確認
echo "🔍 MLBotプロセス確認:"
sudo docker-compose logs mlbot | tail -20 | grep -E "Starting|Discord|API|Error" || echo "ログ確認中..."

# リソース状況
echo ""
echo "💾 システムリソース:"
free -h
df -h /

echo ""
echo "🌐 ネットワーク接続:"
sudo docker-compose exec mlbot ping -c 1 api.bybit.com || echo "ネットワークテスト"
FINAL

echo -e "\n${GREEN}✅ 修正完了！${NC}"
echo ""
echo "📱 Discord通知を確認してください"
echo "🔧 ログ監視: ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo "   → sudo docker-compose logs -f mlbot"