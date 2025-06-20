#!/bin/bash

# Deploy fixed ML production bot to EC2

EC2_HOST="ubuntu@13.212.91.54"
SSH_KEY="$HOME/.ssh/mlbot-key-1749802416.pem"

echo "🚀 Deploying fixed ML production bot..."

# Copy the fixed bot file
echo "📦 Copying fixed bot file..."
scp -i $SSH_KEY working_ml_production_bot.py $EC2_HOST:~/mlbot/

# Deploy and restart
echo "🔄 Restarting bot on EC2..."
ssh -i $SSH_KEY $EC2_HOST << 'EOF'
cd ~/mlbot

# Stop any existing bot
echo "Stopping existing processes..."
pkill -f "working_ml_production_bot.py" || true
pkill -f "working_production_bot.py" || true

# Activate environment and start the fixed bot
echo "Starting fixed ML bot..."
source .venv/bin/activate
nohup python3 working_ml_production_bot.py > logs/ml_bot.log 2>&1 &

echo "Bot started. Checking status..."
sleep 3
ps aux | grep working_ml_production_bot | grep -v grep

echo "✅ Deployment complete!"
EOF

echo "📊 You can monitor logs with:"
echo "ssh -i $SSH_KEY $EC2_HOST 'tail -f ~/mlbot/logs/ml_bot.log'"