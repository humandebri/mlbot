#!/bin/bash
# EC2で即座に取引を開始する

EC2_HOST="ubuntu@13.212.91.54"
SSH_KEY="~/.ssh/mlbot-key-1749802416.pem"

echo "🚀 EC2で取引システムを即座に開始..."

# 必要なファイルを転送
echo "📦 必要なファイルを転送中..."
scp -i $SSH_KEY production_trading_system.py $EC2_HOST:/home/ubuntu/mlbot/
scp -i $SSH_KEY -r src/common/account_monitor.py $EC2_HOST:/home/ubuntu/mlbot/src/common/

# EC2で実行
ssh -i $SSH_KEY $EC2_HOST << 'SETUP'
cd /home/ubuntu/mlbot

echo "🔧 Python環境セットアップ中..."
# Python3.9以上がインストールされているか確認
python3 --version

# 仮想環境を作成
python3 -m venv venv
source venv/bin/activate

# 必要最小限のパッケージをインストール
pip install --upgrade pip
pip install -r requirements.txt

# .envファイルの作成（既存のものがない場合）
if [ ! -f .env ]; then
    cat > .env << 'ENV'
# Environment Configuration
ENVIRONMENT=production
DEBUG=false

# Bybit API Configuration
BYBIT__API_KEY=KgMS2YHiCPG49hmWuV
BYBIT__API_SECRET=LUkW6d5SHHIrbu2SvEDPjFwGYkyUGRun8JF3
BYBIT__TESTNET=false
BYBIT__SYMBOLS=["BTCUSDT", "ETHUSDT"]

# Discord
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq

# Redis Configuration
REDIS__HOST=localhost
REDIS__PORT=6379
REDIS__DB=0
ENV
fi

# tmuxセッションで起動
tmux kill-session -t trading 2>/dev/null || true
tmux new-session -d -s trading

tmux send-keys -t trading "cd /home/ubuntu/mlbot" C-m
tmux send-keys -t trading "source venv/bin/activate" C-m
tmux send-keys -t trading "export BYBIT__TESTNET=false" C-m
tmux send-keys -t trading "nohup python3 production_trading_system.py > trading.log 2>&1 &" C-m

echo "⏳ 起動中..."
sleep 5

# プロセス確認
ps aux | grep production_trading_system | grep -v grep

echo "✅ セットアップ完了！"
SETUP

echo ""
echo "✅ EC2で取引システムが稼働開始しました！"
echo "📊 ログを確認: ssh -i $SSH_KEY $EC2_HOST 'tail -f /home/ubuntu/mlbot/trading.log'"
echo "🔍 tmuxセッション: ssh -i $SSH_KEY $EC2_HOST 'tmux attach -t trading'"