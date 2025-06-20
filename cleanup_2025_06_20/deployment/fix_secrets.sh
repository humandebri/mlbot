#!/bin/bash
# AWS Secrets Manager の削除予定Secretを修復

REGION="ap-southeast-1"

echo "🔧 Secrets Manager 修復スクリプト"
echo "================================="

# 色定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "\n${YELLOW}既存のSecretを強制削除します...${NC}"

# 既存のSecretを強制削除
echo "mlbot/bybit-api-key を強制削除中..."
aws secretsmanager delete-secret \
    --secret-id mlbot/bybit-api-key \
    --force-delete-without-recovery \
    --region $REGION 2>/dev/null || echo "Secret not found or already deleted"

echo "mlbot/discord-webhook を強制削除中..."
aws secretsmanager delete-secret \
    --secret-id mlbot/discord-webhook \
    --force-delete-without-recovery \
    --region $REGION 2>/dev/null || echo "Secret not found or already deleted"

echo -e "\n${GREEN}✅ 強制削除完了${NC}"
echo -e "\n${YELLOW}10秒待機中...${NC}"
sleep 10

echo -e "\n${GREEN}quick_aws_setup.sh を再実行してください:${NC}"
echo "./quick_aws_setup.sh"