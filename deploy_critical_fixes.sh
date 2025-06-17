#!/bin/bash

# EC2への重要な修正のデプロイスクリプト

EC2_HOST="ubuntu@13.212.91.54"
SSH_KEY="~/.ssh/mlbot-key-1749802416.pem"
PROJECT_DIR="/home/ubuntu/mlbot"

echo "🚨 EC2への重要な修正をデプロイ中..."

# 1. 修正されたファイルをEC2に転送
echo "📤 修正ファイルの転送..."
files=(
    "src/common/bybit_client.py"
    "src/order_router/risk_manager.py"
    "src/order_router/order_executor.py"
    "src/ml_pipeline/inference_engine.py"
    "production_trading_system.py"
)

for file in "${files[@]}"; do
    echo "   - $file"
    scp -i "$SSH_KEY" "$file" "$EC2_HOST:$PROJECT_DIR/$file"
done

# 2. EC2で実行中のシステムを安全に停止
echo "🛑 既存システムの停止..."
ssh -i "$SSH_KEY" "$EC2_HOST" << 'EOF'
cd /home/ubuntu/mlbot

# 既存のプロセスを停止
echo "既存プロセスの検索..."
pids=$(pgrep -f "production_trading_system.py" || true)
if [ ! -z "$pids" ]; then
    echo "プロセスID: $pids を停止中..."
    kill -TERM $pids 2>/dev/null || true
    sleep 5
    # まだ残っていれば強制終了
    kill -KILL $pids 2>/dev/null || true
fi

echo "プロセスが停止しました"
EOF

# 3. EC2でシステムを再起動
echo "🚀 修正済みシステムの起動..."
ssh -i "$SSH_KEY" "$EC2_HOST" << 'EOF'
cd /home/ubuntu/mlbot

# tmuxセッションで起動
tmux new-session -d -s trading "cd /home/ubuntu/mlbot && source .venv/bin/activate && python production_trading_system.py 2>&1 | tee -a logs/production_$(date +%Y%m%d).log"

echo "✅ システムが起動しました"

# 起動確認
sleep 5
echo "プロセス確認:"
ps aux | grep -E "production_trading_system|python" | grep -v grep

# tmuxセッション確認
echo -e "\ntmuxセッション:"
tmux ls
EOF

echo "✅ デプロイ完了！"
echo ""
echo "📝 EC2にSSH接続してログを確認:"
echo "   ssh -i $SSH_KEY $EC2_HOST"
echo "   tmux attach -t trading"
echo ""
echo "🔍 ログファイル確認:"
echo "   tail -f /home/ubuntu/mlbot/logs/production_$(date +%Y%m%d).log"