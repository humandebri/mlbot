#!/bin/bash

EC2_HOST="ubuntu@13.212.91.54"
EC2_KEY="/Users/0xhude/.ssh/mlbot-key-1749802416.pem"

echo "🚀 Deploying Simple Improved Bot..."

# Execute on remote
ssh -i "$EC2_KEY" "$EC2_HOST" 'bash -s' << 'ENDSSH'
cd /home/ubuntu/mlbot

# Clean up
echo "Cleaning up..."
pkill -f "simple_improved_bot" || true
tmux kill-session -t simple_bot 2>/dev/null || true

# Install dependencies
echo "Checking dependencies..."
python3 -m pip install --user aiohttp onnxruntime redis asyncio-redis 2>/dev/null

# Get Discord webhook from .env
export DISCORD_WEBHOOK=$(grep DISCORD_WEBHOOK .env | cut -d= -f2- | tr -d '"' | head -1)

# Start bot in tmux
echo "Starting bot..."
tmux new-session -d -s simple_bot bash -c '
cd /home/ubuntu/mlbot
export DISCORD_WEBHOOK=$(grep DISCORD_WEBHOOK .env | cut -d= -f2- | tr -d '"' | head -1)
echo "Starting Simple Improved Bot..."
echo "Time: $(date)"
echo "Discord webhook: ${DISCORD_WEBHOOK:0:50}..."
python3 simple_improved_bot.py 2>&1 | tee -a logs/simple_bot.log
'

# Check status
sleep 10
echo -e "\n=== Bot Status ==="
if ps aux | grep "simple_improved_bot" | grep -v grep; then
    echo "✅ Bot is running!"
else
    echo "❌ Bot not running. Checking logs..."
    tail -30 logs/simple_bot.log 2>/dev/null || tmux capture-pane -t simple_bot -p | tail -30
fi

echo -e "\n=== Active Sessions ==="
tmux ls
ENDSSH

echo -e "\n✅ Deployment complete!"
echo "Monitor: ssh -i $EC2_KEY $EC2_HOST 'tmux attach -t simple_bot'"
echo "Logs: ssh -i $EC2_KEY $EC2_HOST 'tail -f /home/ubuntu/mlbot/logs/simple_bot.log'"