#!/bin/bash
# AWS権限チェックスクリプト

echo "🔐 AWS権限チェック"
echo "===================="

# 色定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

# AWS CLI確認
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLIがインストールされていません${NC}"
    exit 1
fi

echo -e "${GREEN}✅ AWS CLI確認済み${NC}"

# 認証確認
echo -e "\n📊 認証情報確認中..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS認証が設定されていません${NC}"
    echo "実行してください: aws configure"
    exit 1
fi

USER_INFO=$(aws sts get-caller-identity)
echo -e "${GREEN}✅ 認証確認済み${NC}"
echo "$USER_INFO"

# EC2権限チェック
echo -e "\n🖥️  EC2権限チェック中..."
if aws ec2 describe-regions --region us-east-1 &> /dev/null; then
    echo -e "${GREEN}✅ EC2権限確認済み${NC}"
else
    echo -e "${RED}❌ EC2権限が不足しています${NC}"
    echo "必要な権限: AmazonEC2FullAccess"
fi

# Secrets Manager権限チェック
echo -e "\n🔒 Secrets Manager権限チェック中..."
if aws secretsmanager list-secrets --region us-east-1 &> /dev/null; then
    echo -e "${GREEN}✅ Secrets Manager権限確認済み${NC}"
else
    echo -e "${RED}❌ Secrets Manager権限が不足しています${NC}"
    echo "必要な権限: SecretsManagerReadWrite"
fi

# IAM権限チェック
echo -e "\n👤 IAM権限チェック中..."
if aws iam list-roles --max-items 1 &> /dev/null; then
    echo -e "${GREEN}✅ IAM権限確認済み${NC}"
else
    echo -e "${RED}❌ IAM権限が不足しています${NC}"
    echo "必要な権限: IAMFullAccess (デプロイ時のみ)"
fi

# 権限修正コマンド提示
echo -e "\n" + "=" * 50
echo "🔧 権限が不足している場合の修正方法"
echo "=" * 50

echo -e "\n${YELLOW}AWS Consoleでの設定:${NC}"
echo "1. https://console.aws.amazon.com/iam/"
echo "2. ユーザー → [あなたのユーザー名]"
echo "3. 許可を追加 → ポリシーを直接アタッチ"
echo "4. 以下のポリシーを追加:"
echo "   - AmazonEC2FullAccess"
echo "   - SecretsManagerReadWrite"
echo "   - IAMFullAccess"

echo -e "\n${YELLOW}CLI での設定:${NC}"
echo "export AWS_USERNAME=\"your-username\""
echo ""
echo "aws iam attach-user-policy \\"
echo "    --user-name \$AWS_USERNAME \\"
echo "    --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess"
echo ""
echo "aws iam attach-user-policy \\"
echo "    --user-name \$AWS_USERNAME \\"
echo "    --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite"
echo ""
echo "aws iam attach-user-policy \\"
echo "    --user-name \$AWS_USERNAME \\"
echo "    --policy-arn arn:aws:iam::aws:policy/IAMFullAccess"

echo -e "\n" + "=" * 50
echo -e "${GREEN}権限設定完了後、再度このスクリプトを実行してください${NC}"
echo -e "${GREEN}その後、デプロイスクリプトを実行できます:${NC}"
echo "./quick_aws_setup.sh"