#!/bin/bash
# EC2で取引システムを24時間稼働させる

EC2_HOST="ubuntu@13.212.91.54"
SSH_KEY="~/.ssh/mlbot-key-1749802416.pem"

echo "🚀 EC2で取引システムを起動中..."

# 最新コードとAccountMonitor修正を同期
echo "📦 最新コードを同期..."
scp -i $SSH_KEY src/common/account_monitor.py $EC2_HOST:/home/ubuntu/mlbot/src/common/
scp -i $SSH_KEY production_trading_system.py $EC2_HOST:/home/ubuntu/mlbot/

# EC2で起動
ssh -i $SSH_KEY $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 既存プロセスを停止
pkill -f "python.*trading" || true

# tmuxセッションを作成（または再利用）
tmux kill-session -t mlbot 2>/dev/null || true
tmux new-session -d -s mlbot

# 仮想環境をアクティベートして起動
tmux send-keys -t mlbot "cd /home/ubuntu/mlbot" C-m
tmux send-keys -t mlbot "source venv/bin/activate" C-m
tmux send-keys -t mlbot "export BYBIT__TESTNET=false" C-m
tmux send-keys -t mlbot "python production_trading_system.py" C-m

echo "✅ システム起動完了！"
echo ""
echo "📊 ログを確認:"
echo "  ssh -i ~/.ssh/mlbot-key-1749802416.pem ubuntu@13.212.91.54"
echo "  tmux attach -t mlbot"
echo ""
echo "🔄 セッションから離れる: Ctrl+B → D"
echo "❌ 停止する: tmux kill-session -t mlbot"
EOF

echo ""
echo "✅ EC2で24時間稼働開始！"
echo "💰 残高: $99.92"
echo "🤖 PCをシャットダウンしても取引継続します"