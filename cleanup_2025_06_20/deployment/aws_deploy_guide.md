# AWS デプロイメントガイド

## 前提条件チェック

1. **AWS CLIのインストール確認**
   ```bash
   aws --version
   ```
   
   インストールされていない場合:
   ```bash
   # macOS
   brew install awscli
   
   # または公式インストーラー
   curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
   sudo installer -pkg AWSCLIV2.pkg -target /
   ```

2. **AWS認証情報の設定**
   ```bash
   aws configure
   ```
   以下を入力:
   - AWS Access Key ID: [あなたのアクセスキー]
   - AWS Secret Access Key: [あなたのシークレットキー]
   - Default region name: ap-northeast-1
   - Default output format: json

## ステップ1: セキュリティ設定

### Secrets Managerで機密情報を保護

```bash
# Bybit APIキーを保存
aws secretsmanager create-secret \
    --name mlbot/bybit-api-key \
    --secret-string '{"api_key":"YOUR_API_KEY","api_secret":"YOUR_API_SECRET"}' \
    --region ap-northeast-1

# Discord Webhookを保存
aws secretsmanager create-secret \
    --name mlbot/discord-webhook \
    --secret-string '{"webhook_url":"YOUR_WEBHOOK_URL"}' \
    --region ap-northeast-1
```

## ステップ2: EC2インスタンスの作成

### 自動デプロイスクリプトの実行

```bash
cd /Users/0xhude/Desktop/mlbot/deployment
bash deploy_to_aws.sh
```

### または手動でEC2を作成

1. AWSコンソールにログイン
2. EC2ダッシュボードへ移動
3. 「インスタンスを起動」をクリック
4. 以下の設定:
   - AMI: Ubuntu Server 22.04 LTS
   - インスタンスタイプ: t3.medium
   - キーペア: 新規作成または既存のものを選択
   - ネットワーク設定:
     - VPC: デフォルト
     - サブネット: 任意のパブリックサブネット
     - パブリックIP: 有効化
   - セキュリティグループ:
     - SSH (22): あなたのIPから
     - カスタムTCP (8080): 0.0.0.0/0（ヘルスチェック用）
   - ストレージ: 100GB gp3

## ステップ3: サーバーの初期設定

### SSHで接続

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@[EC2のパブリックIP]
```

### 基本セットアップ

```bash
# システム更新
sudo apt update && sudo apt upgrade -y

# 必要なツールをインストール
sudo apt install -y git python3-pip python3-venv

# Dockerのインストール
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Docker Composeのインストール
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 再ログインして設定を反映
exit
```

再度SSHで接続:
```bash
ssh -i ~/.ssh/your-key.pem ubuntu@[EC2のパブリックIP]
```

## ステップ4: アプリケーションのデプロイ

### コードの取得と設定

```bash
# プロジェクトをクローン
cd ~
git clone https://github.com/humandebri/mlbot.git
cd mlbot

# ディレクトリ作成
mkdir -p data logs models

# 本番環境設定をコピー
cp .env.production .env

# .envファイルを編集
nano .env
# ここでAPIキーなどを設定
```

### AWS Secrets Managerから機密情報を取得

```bash
# AWS CLIをインストール
sudo apt install -y awscli

# シークレットを取得するスクリプトを作成
cat > get_secrets.sh << 'EOF'
#!/bin/bash

# Bybit credentials
BYBIT_SECRET=$(aws secretsmanager get-secret-value --secret-id mlbot/bybit-api-key --region ap-northeast-1 --query SecretString --output text)
export BYBIT_API_KEY=$(echo $BYBIT_SECRET | jq -r .api_key)
export BYBIT_API_SECRET=$(echo $BYBIT_SECRET | jq -r .api_secret)

# Discord webhook
DISCORD_SECRET=$(aws secretsmanager get-secret-value --secret-id mlbot/discord-webhook --region ap-northeast-1 --query SecretString --output text)
export DISCORD_WEBHOOK=$(echo $DISCORD_SECRET | jq -r .webhook_url)

echo "Secrets loaded successfully"
EOF

chmod +x get_secrets.sh
```

### モデルファイルのアップロード

ローカルから:
```bash
# モデルファイルをアップロード
scp -i ~/.ssh/your-key.pem /Users/0xhude/Desktop/mlbot/models/fast_nn_final.pth ubuntu@[EC2のIP]:~/mlbot/models/
scp -i ~/.ssh/your-key.pem /Users/0xhude/Desktop/mlbot/models/fast_nn_scaler.pkl ubuntu@[EC2のIP]:~/mlbot/models/
```

## ステップ5: アプリケーションの起動

### Dockerを使用した起動

```bash
cd ~/mlbot

# Dockerイメージをビルド
docker build -t mlbot:latest .

# シークレットを読み込んで起動
source get_secrets.sh
docker-compose -f docker-compose.production.yml up -d
```

### または直接Pythonで起動

```bash
# Python仮想環境の作成
python3 -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt

# シークレットを読み込む
source get_secrets.sh

# 起動
python src/system/main.py
```

## ステップ6: 監視とログ

### ログの確認

```bash
# Dockerの場合
docker-compose -f docker-compose.production.yml logs -f

# 直接実行の場合
tail -f logs/trading_bot.log
```

### システム状態の確認

```bash
# ヘルスチェック
curl http://localhost:8080/health

# メトリクス確認
curl http://localhost:8080/metrics
```

## ステップ7: 自動起動の設定

### systemdサービスの作成

```bash
sudo nano /etc/systemd/system/mlbot.service
```

以下を貼り付け:
```ini
[Unit]
Description=MLBot Trading System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/mlbot
EnvironmentFile=/home/ubuntu/mlbot/.env
ExecStartPre=/bin/bash -c 'source /home/ubuntu/mlbot/get_secrets.sh'
ExecStart=/usr/bin/docker-compose -f docker-compose.production.yml up
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

有効化:
```bash
sudo systemctl enable mlbot.service
sudo systemctl start mlbot.service
sudo systemctl status mlbot.service
```

## トラブルシューティング

### よくある問題

1. **メモリ不足**
   ```bash
   # スワップ追加
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **ディスク容量不足**
   ```bash
   # 不要なDockerイメージを削除
   docker system prune -a
   ```

3. **接続エラー**
   - セキュリティグループの設定を確認
   - Elastic IPを割り当てて固定IP化

## 次のステップ

1. CloudWatchアラームの設定
2. 自動バックアップの設定
3. 小額でのテスト運用（1週間）
4. パフォーマンスモニタリング
5. 本格運用開始