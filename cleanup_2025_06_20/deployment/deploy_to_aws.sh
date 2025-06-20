#!/bin/bash
# AWS EC2へのデプロイスクリプト

set -e

echo "==================================="
echo "MLBot AWS デプロイメントスクリプト"
echo "==================================="

# 設定
INSTANCE_TYPE="t3.medium"
REGION="ap-northeast-1"
KEY_NAME="mlbot-key"
SECURITY_GROUP="mlbot-sg"
AMI_ID="ami-0d52744d6551d851e"  # Ubuntu 22.04 LTS

# 色付き出力
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# 1. キーペアの作成
echo -e "\n${GREEN}1. SSHキーペアを作成中...${NC}"
if ! aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION 2>/dev/null; then
    aws ec2 create-key-pair --key-name $KEY_NAME --region $REGION \
        --query 'KeyMaterial' --output text > ~/.ssh/mlbot-key.pem
    chmod 600 ~/.ssh/mlbot-key.pem
    echo "✅ キーペア作成完了: ~/.ssh/mlbot-key.pem"
else
    echo "✅ キーペアは既に存在します"
fi

# 2. セキュリティグループの作成
echo -e "\n${GREEN}2. セキュリティグループを作成中...${NC}"
if ! aws ec2 describe-security-groups --group-names $SECURITY_GROUP --region $REGION 2>/dev/null; then
    VPC_ID=$(aws ec2 describe-vpcs --region $REGION --query 'Vpcs[0].VpcId' --output text)
    
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SECURITY_GROUP \
        --description "MLBot Trading System Security Group" \
        --vpc-id $VPC_ID \
        --region $REGION \
        --query 'GroupId' --output text)
    
    # SSH許可
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $REGION
    
    echo "✅ セキュリティグループ作成完了: $SG_ID"
else
    echo "✅ セキュリティグループは既に存在します"
fi

# 3. EC2インスタンスの起動
echo -e "\n${GREEN}3. EC2インスタンスを起動中...${NC}"
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --region $REGION \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=MLBot-Production}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ インスタンス起動中: $INSTANCE_ID"

# インスタンスが起動するまで待機
echo "⏳ インスタンスの起動を待っています..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# パブリックIPを取得
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✅ インスタンス起動完了"
echo "   Public IP: $PUBLIC_IP"

# 4. 初期設定スクリプトの作成
cat > setup_server.sh << 'EOF'
#!/bin/bash
set -e

# システムアップデート
sudo apt-get update
sudo apt-get upgrade -y

# Dockerのインストール
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Docker Composeのインストール
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Gitのインストール
sudo apt-get install -y git

# プロジェクトのクローン
cd /home/ubuntu
git clone https://github.com/humandebri/mlbot.git
cd mlbot

# ディレクトリの作成
mkdir -p data logs models

echo "✅ サーバーセットアップ完了"
EOF

# 5. サーバーへの接続情報を表示
echo -e "\n${GREEN}==================================="
echo "デプロイメント完了！"
echo "===================================${NC}"
echo ""
echo "接続方法:"
echo "  ssh -i ~/.ssh/mlbot-key.pem ubuntu@$PUBLIC_IP"
echo ""
echo "次のステップ:"
echo "1. サーバーに接続"
echo "2. setup_server.shを実行"
echo "3. .envファイルを設定"
echo "4. docker-compose up -d で起動"
echo ""
echo "セットアップコマンド:"
echo "  scp -i ~/.ssh/mlbot-key.pem setup_server.sh ubuntu@$PUBLIC_IP:~/"
echo "  ssh -i ~/.ssh/mlbot-key.pem ubuntu@$PUBLIC_IP 'bash setup_server.sh'"

# 6. ElastiCache (Redis)の作成
echo -e "\n${GREEN}Redis (ElastiCache)を作成しますか？ (y/n)${NC}"
read -r CREATE_REDIS

if [[ $CREATE_REDIS == "y" ]]; then
    SUBNET_ID=$(aws ec2 describe-subnets --region $REGION --query 'Subnets[0].SubnetId' --output text)
    
    CACHE_SUBNET_GROUP=$(aws elasticache create-cache-subnet-group \
        --cache-subnet-group-name mlbot-subnet-group \
        --cache-subnet-group-description "MLBot Redis Subnet Group" \
        --subnet-ids $SUBNET_ID \
        --region $REGION \
        --query 'CacheSubnetGroup.CacheSubnetGroupName' \
        --output text 2>/dev/null || echo "mlbot-subnet-group")
    
    REDIS_ENDPOINT=$(aws elasticache create-cache-cluster \
        --cache-cluster-id mlbot-redis \
        --cache-node-type cache.t3.micro \
        --engine redis \
        --num-cache-nodes 1 \
        --cache-subnet-group-name $CACHE_SUBNET_GROUP \
        --region $REGION \
        --query 'CacheCluster.CacheNodes[0].Endpoint.Address' \
        --output text 2>/dev/null || echo "既存のRedisを使用")
    
    echo "✅ Redis作成完了: $REDIS_ENDPOINT"
fi

echo -e "\n${GREEN}デプロイメント完了！${NC}"