#!/usr/bin/env python3
"""
クイックデプロイメントスクリプト - DigitalOcean版
シンプルで低コストな本番環境構築
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# 色付き出力
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class QuickDeploy:
    """簡単デプロイメントツール"""
    
    def __init__(self):
        self.droplet_size = "s-2vcpu-4gb"  # $24/月
        self.region = "sgp1"  # シンガポール（低レイテンシ）
        self.image = "ubuntu-22-04-x64"
        
    def check_requirements(self):
        """必要なツールの確認"""
        print(f"\n{BLUE}1. 必要なツールを確認中...{RESET}")
        
        tools = {
            'docker': 'Docker',
            'git': 'Git',
            'ssh': 'SSH'
        }
        
        missing = []
        for cmd, name in tools.items():
            if subprocess.run(['which', cmd], capture_output=True).returncode != 0:
                missing.append(name)
        
        if missing:
            print(f"{RED}❌ 以下のツールをインストールしてください: {', '.join(missing)}{RESET}")
            return False
            
        print(f"{GREEN}✅ 必要なツールが揃っています{RESET}")
        return True
    
    def create_env_file(self):
        """本番用.envファイルの作成"""
        print(f"\n{BLUE}2. 環境設定ファイルを作成中...{RESET}")
        
        env_content = """# Bybit API Configuration
BYBIT_API_KEY=your_production_api_key_here
BYBIT_API_SECRET=your_production_api_secret_here
USE_TESTNET=false

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Discord Configuration
DISCORD_WEBHOOK=your_discord_webhook_url_here

# System Configuration
LOG_LEVEL=INFO
ENVIRONMENT=production

# Trading Configuration
SYMBOLS=BTCUSDT,ETHUSDT,ICPUSDT
MIN_CONFIDENCE=0.6
MIN_EXPECTED_PNL=0.001

# Risk Management
MAX_POSITION_SIZE_USD=10000
MAX_LEVERAGE=3
MAX_DAILY_LOSS_USD=500
MAX_DRAWDOWN_PCT=0.10
"""
        
        env_path = Path(".env.production")
        env_path.write_text(env_content)
        
        print(f"{YELLOW}⚠️  .env.productionファイルを編集してAPIキーを設定してください{RESET}")
        print(f"   ファイル: {env_path.absolute()}")
        return True
    
    def create_docker_compose(self):
        """シンプルなDocker Compose設定"""
        print(f"\n{BLUE}3. Docker Compose設定を作成中...{RESET}")
        
        compose_content = """version: '3.8'

services:
  mlbot:
    build: .
    container_name: mlbot-production
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - PYTHONUNBUFFERED=1
    env_file:
      - .env.production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    depends_on:
      - redis
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  redis:
    image: redis:7-alpine
    container_name: mlbot-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  redis_data:
"""
        
        compose_path = Path("docker-compose.production.yml")
        compose_path.write_text(compose_content)
        
        print(f"{GREEN}✅ Docker Compose設定作成完了{RESET}")
        return True
    
    def create_deployment_script(self):
        """サーバー上で実行するスクリプト"""
        print(f"\n{BLUE}4. デプロイメントスクリプトを作成中...{RESET}")
        
        script_content = """#!/bin/bash
set -e

echo "MLBot本番環境セットアップ開始..."

# システムアップデート
sudo apt-get update
sudo apt-get upgrade -y

# Dockerインストール
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Docker Composeインストール
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# プロジェクトセットアップ
cd /home/$USER
if [ ! -d "mlbot" ]; then
    git clone https://github.com/humandebri/mlbot.git
fi

cd mlbot
git pull origin main

# ディレクトリ作成
mkdir -p data logs models

# 起動
docker-compose -f docker-compose.production.yml up -d --build

echo "✅ セットアップ完了！"
echo "ログ確認: docker-compose -f docker-compose.production.yml logs -f"
"""
        
        script_path = Path("server_setup.sh")
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        print(f"{GREEN}✅ デプロイメントスクリプト作成完了{RESET}")
        return True
    
    def print_instructions(self):
        """手動デプロイ手順の表示"""
        print(f"\n{GREEN}{'='*60}{RESET}")
        print(f"{GREEN}デプロイメント準備完了！{RESET}")
        print(f"{GREEN}{'='*60}{RESET}")
        
        print(f"\n{BLUE}【推奨: DigitalOcean】{RESET}")
        print("1. https://www.digitalocean.com にアクセス")
        print("2. Dropletを作成:")
        print(f"   - Distribution: Ubuntu 22.04")
        print(f"   - Plan: Basic → Regular → {self.droplet_size} ($24/月)")
        print(f"   - Region: Singapore")
        print("   - Authentication: SSHキーを追加")
        
        print(f"\n{BLUE}【サーバーセットアップ】{RESET}")
        print("1. .env.productionを編集してAPIキーを設定")
        print("2. サーバーにファイルをアップロード:")
        print("   scp -r . root@YOUR_SERVER_IP:/root/mlbot/")
        print("3. サーバーに接続:")
        print("   ssh root@YOUR_SERVER_IP")
        print("4. セットアップスクリプトを実行:")
        print("   cd /root/mlbot && bash server_setup.sh")
        
        print(f"\n{BLUE}【AWS Lightsail代替案】{RESET}")
        print("AWS Lightsailも低コストでおすすめ:")
        print("- インスタンス: 2GB RAM ($20/月)")
        print("- 東京リージョンで低レイテンシ")
        print("- 簡単な管理画面")
        
        print(f"\n{YELLOW}【重要な注意事項】{RESET}")
        print("⚠️  本番APIキーは必ず環境変数で管理")
        print("⚠️  初期は少額で動作確認")
        print("⚠️  定期的なバックアップを設定")
        print("⚠️  Discord通知で監視")
        
        print(f"\n{GREEN}頑張ってください！🚀{RESET}")


def main():
    """メイン処理"""
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}MLBot クイックデプロイメント{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    deploy = QuickDeploy()
    
    # 必要なツールの確認
    if not deploy.check_requirements():
        return
    
    # ファイル作成
    deploy.create_env_file()
    deploy.create_docker_compose()
    deploy.create_deployment_script()
    
    # 手順の表示
    deploy.print_instructions()


if __name__ == "__main__":
    main()