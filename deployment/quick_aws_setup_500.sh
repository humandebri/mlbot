#!/bin/bash
# $500版 クイックAWSセットアップスクリプト

set -e

echo "========================================"
echo "MLBot AWS クイックセットアップ ($500版)"
echo "========================================"

# カラー定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

# 設定確認
echo -e "\n${YELLOW}以下の情報が必要です:${NC}"
echo "1. Bybit本番APIキー"
echo "2. Bybit本番APIシークレット"
echo "3. Discord Webhook URL（オプション）"
echo ""
read -p "準備できましたか？ (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# AWS CLI確認
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLIがインストールされていません${NC}"
    echo "インストール方法:"
    echo "  brew install awscli"
    exit 1
fi

# AWS認証確認
echo -e "\n${GREEN}AWS認証情報を確認中...${NC}"
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}AWS認証が設定されていません${NC}"
    echo "以下を実行してください:"
    echo "  aws configure"
    exit 1
fi

# リージョン選択
echo -e "\n${GREEN}リージョンを選択してください:${NC}"
echo "1) ap-northeast-1 (東京) - 推奨"
echo "2) ap-southeast-1 (シンガポール)"
echo "3) us-east-1 (バージニア)"
read -p "選択 (1-3): " region_choice

case $region_choice in
    1) REGION="ap-northeast-1"; AMI_ID="ami-0d52744d6551d851e" ;;
    2) REGION="ap-southeast-1"; AMI_ID="ami-0df7a207adb9748c7" ;;
    3) REGION="us-east-1"; AMI_ID="ami-0557a15b87f6559cf" ;;
    *) echo "無効な選択"; exit 1 ;;
esac

# API情報入力
echo -e "\n${GREEN}Bybit API情報を入力してください:${NC}"
read -p "API Key: " BYBIT_API_KEY
read -s -p "API Secret: " BYBIT_API_SECRET
echo

echo -e "\n${GREEN}Discord Webhook URL (スキップする場合はEnter):${NC}"
read -p "URL: " DISCORD_WEBHOOK

# Secrets Manager に保存
echo -e "\n${GREEN}機密情報をAWS Secrets Managerに保存中...${NC}"

# 既存のシークレットを削除（存在する場合）
aws secretsmanager delete-secret --secret-id mlbot/bybit-api-key --force-delete-without-recovery --region $REGION 2>/dev/null || true
aws secretsmanager delete-secret --secret-id mlbot/discord-webhook --force-delete-without-recovery --region $REGION 2>/dev/null || true

sleep 2

# 新規作成
aws secretsmanager create-secret \
    --name mlbot/bybit-api-key \
    --secret-string "{\"api_key\":\"$BYBIT_API_KEY\",\"api_secret\":\"$BYBIT_API_SECRET\"}" \
    --region $REGION

if [ ! -z "$DISCORD_WEBHOOK" ]; then
    aws secretsmanager create-secret \
        --name mlbot/discord-webhook \
        --secret-string "{\"webhook_url\":\"$DISCORD_WEBHOOK\"}" \
        --region $REGION
fi

echo -e "${GREEN}✅ 機密情報を保存しました${NC}"

# EC2インスタンス作成
echo -e "\n${GREEN}EC2インスタンスを作成中...${NC}"

# キーペア作成
KEY_NAME="mlbot-key-$(date +%s)"
aws ec2 create-key-pair \
    --key-name $KEY_NAME \
    --region $REGION \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/${KEY_NAME}.pem
chmod 600 ~/.ssh/${KEY_NAME}.pem

# セキュリティグループ作成
VPC_ID=$(aws ec2 describe-vpcs --region $REGION --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 create-security-group \
    --group-name "mlbot-sg-$(date +%s)" \
    --description "MLBot Trading System" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# ルール追加
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# ユーザーデータスクリプト
USER_DATA=$(cat << 'EOF'
#!/bin/bash
apt-get update
apt-get install -y docker.io docker-compose git awscli jq

# Dockerグループに追加
usermod -aG docker ubuntu

# プロジェクトをクローン
cd /home/ubuntu
git clone https://github.com/humandebri/mlbot.git
chown -R ubuntu:ubuntu mlbot

# ディレクトリ作成
cd mlbot
mkdir -p data logs models
EOF
)

# Wait for instance profile to be ready
echo -e "${GREEN}IAMインスタンスプロファイルの準備を待っています...${NC}"
sleep 30

# インスタンス起動
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type t3.medium \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --region $REGION \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=MLBot-Production-500}]" \
    --iam-instance-profile Name=mlbot-instance-profile \
    --user-data "$USER_DATA" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}✅ インスタンス起動中: $INSTANCE_ID${NC}"

# 起動待機
echo -e "${YELLOW}インスタンスの起動を待っています...${NC}"
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# パブリックIP取得
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

# $500用セットアップスクリプト作成
cat > setup_mlbot_500.sh << 'SETUP'
#!/bin/bash
set -e

cd /home/ubuntu/mlbot

# $500用環境設定
cat > .env << 'ENV'
# Bybit API Configuration
BYBIT_API_KEY=will_be_loaded_from_secrets
BYBIT_API_SECRET=will_be_loaded_from_secrets
USE_TESTNET=false

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Discord Configuration
DISCORD_WEBHOOK=will_be_loaded_from_secrets

# System Configuration
LOG_LEVEL=INFO
ENVIRONMENT=production

# Trading Configuration - $500版
SYMBOLS=ICPUSDT
MIN_CONFIDENCE=0.65
MIN_EXPECTED_PNL=0.0015

# Risk Management - $500版
INITIAL_CAPITAL=500
MAX_POSITION_SIZE_USD=25
MAX_LEVERAGE=2
MAX_DAILY_LOSS_USD=25
MAX_DRAWDOWN_PCT=0.08
BASE_POSITION_SIZE_PCT=0.05

# Execution Configuration
USE_POST_ONLY=true
PRICE_BUFFER_PCT=0.0003
MAX_ORDER_AGE_SECONDS=180
AGGRESSIVE_FILL_TIMEOUT=20

# Account Monitoring
BALANCE_CHECK_INTERVAL=900
AUTO_COMPOUND=true
COMPOUND_FREQUENCY=daily

# Special Settings for Small Capital
MIN_ORDER_SIZE_USD=12
MAX_CONCURRENT_POSITIONS=2
TRADE_COOLDOWN_SECONDS=300
ENV

# Secrets取得スクリプト
cat > load_secrets.sh << 'SECRETS'
#!/bin/bash
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

# Bybit credentials
BYBIT_SECRET=$(aws secretsmanager get-secret-value --secret-id mlbot/bybit-api-key --region $REGION --query SecretString --output text 2>/dev/null || echo '{}')
if [ "$BYBIT_SECRET" != "{}" ]; then
    export BYBIT_API_KEY=$(echo $BYBIT_SECRET | jq -r .api_key)
    export BYBIT_API_SECRET=$(echo $BYBIT_SECRET | jq -r .api_secret)
fi

# Discord webhook
DISCORD_SECRET=$(aws secretsmanager get-secret-value --secret-id mlbot/discord-webhook --region $REGION --query SecretString --output text 2>/dev/null || echo '{}')
if [ "$DISCORD_SECRET" != "{}" ]; then
    export DISCORD_WEBHOOK=$(echo $DISCORD_SECRET | jq -r .webhook_url)
fi

echo "✅ Secrets loaded successfully"
echo "API Key: ${BYBIT_API_KEY:0:8}..."
echo "Discord: ${DISCORD_WEBHOOK:0:30}..."
SECRETS

chmod +x load_secrets.sh

# Docker Compose設定
cat > docker-compose.yml << 'COMPOSE'
version: '3.8'

services:
  mlbot:
    build: .
    container_name: mlbot-production-500
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - PYTHONUNBUFFERED=1
      - BYBIT_API_KEY=${BYBIT_API_KEY}
      - BYBIT_API_SECRET=${BYBIT_API_SECRET}
      - DISCORD_WEBHOOK=${DISCORD_WEBHOOK}
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    depends_on:
      - redis
    ports:
      - "8080:8080"

  redis:
    image: redis:7-alpine
    container_name: mlbot-redis-500
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 256mb
    volumes:
      - redis_data:/data

volumes:
  redis_data:
COMPOSE

echo "✅ $500用セットアップ完了！"
SETUP

# 接続情報を保存
cat > mlbot_connection_info_500.txt << INFO
======================================
MLBot AWS接続情報 ($500版)
======================================

インスタンスID: $INSTANCE_ID
パブリックIP: $PUBLIC_IP
リージョン: $REGION
SSHキー: ~/.ssh/${KEY_NAME}.pem

接続コマンド:
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP

セットアップ完了後:
1. モデルファイルをアップロード:
   scp -i ~/.ssh/${KEY_NAME}.pem models/fast_nn_final.pth ubuntu@$PUBLIC_IP:~/mlbot/models/
   scp -i ~/.ssh/${KEY_NAME}.pem models/fast_nn_scaler.pkl ubuntu@$PUBLIC_IP:~/mlbot/models/

2. サーバーでセットアップ実行:
   ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP
   cd mlbot
   bash ../setup_mlbot_500.sh
   source load_secrets.sh
   docker-compose up --build -d

3. ログ確認:
   docker-compose logs -f mlbot

4. 動作確認:
   curl http://localhost:8080/health

設定内容 ($500版):
- 初期資金: $500
- 対象通貨: ICPUSDT のみ
- ポジションサイズ: 5% ($25)
- レバレッジ: 2倍
- 残高チェック: 15分間隔

======================================
INFO

echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}$500版デプロイメント準備完了！${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
cat mlbot_connection_info_500.txt

# セットアップスクリプトをアップロード
echo -e "\n${YELLOW}セットアップスクリプトをアップロード中...${NC}"
sleep 15  # インスタンスの初期化を十分待つ

# アップロード試行（リトライ付き）
for i in {1..3}; do
    if scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ~/.ssh/${KEY_NAME}.pem setup_mlbot_500.sh ubuntu@$PUBLIC_IP:~/; then
        echo -e "${GREEN}✅ セットアップスクリプトアップロード成功${NC}"
        break
    else
        echo -e "${YELLOW}リトライ $i/3...${NC}"
        sleep 10
    fi
done

echo -e "\n${GREEN}✅ 完了！次のステップ:${NC}"
echo "1. モデルファイルをアップロード"
echo "2. SSHで接続してセットアップを実行"
echo ""
echo "詳細は mlbot_connection_info_500.txt を参照してください"