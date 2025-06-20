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
