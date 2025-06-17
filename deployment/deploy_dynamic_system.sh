#\!/bin/bash
set -e

echo "🚀 EC2への動的パラメータ版デプロイを開始"

# EC2接続情報
EC2_HOST="13.212.91.54"
EC2_USER="ubuntu"
SSH_KEY="~/.ssh/mlbot-key-1749802416.pem"

# 現在実行中のプロセスを停止
echo "📋 現在実行中のプロセスを停止中..."
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'ENDSSH'
# 既存のプロセスを停止
if pgrep -f "production_trading_system" > /dev/null; then
    echo "既存のトレーディングシステムを停止中..."
    pkill -f "production_trading_system" || true
    sleep 5
fi

# Dockerコンテナも停止
cd /home/ubuntu/mlbot
docker-compose down || true
ENDSSH

# 新しいファイルをアップロード
echo "📦 動的パラメータ版をアップロード中..."
scp -i "$SSH_KEY" production_trading_system_dynamic_fixed.py "$EC2_USER@$EC2_HOST:/home/ubuntu/mlbot/production_trading_system_dynamic.py"

# EC2で起動
echo "🎯 EC2で動的パラメータ版を起動中..."
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_HOST" << 'ENDSSH'
cd /home/ubuntu/mlbot

# 仮想環境を有効化
source .venv/bin/activate

# 環境変数を設定
export BYBIT__TESTNET=false
export ENVIRONMENT=production

# tmuxセッションで起動
tmux new-session -d -s trading "cd /home/ubuntu/mlbot && source .venv/bin/activate && python production_trading_system_dynamic.py 2>&1 | tee -a trading_dynamic.log"

echo "✅ 動的パラメータ版が起動しました"

# 起動確認
sleep 10
if pgrep -f "production_trading_system_dynamic" > /dev/null; then
    echo "✅ プロセスが正常に動作しています"
    ps aux | grep -E "(production_trading_system_dynamic|python)" | grep -v grep
else
    echo "❌ プロセスが起動していません。ログを確認してください："
    tail -20 trading_dynamic.log
fi
ENDSSH

echo "🎉 デプロイ完了"
