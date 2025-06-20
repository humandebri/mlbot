#!/bin/bash

# EC2 Configuration
EC2_HOST="ubuntu@13.212.91.54"
EC2_KEY="/Users/0xhude/.ssh/mlbot-key-1749802416.pem"

echo "🚀 Starting Improved ML Bot on EC2..."

# Execute startup commands on EC2
ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
cd /home/ubuntu/mlbot

# Clean up any existing processes
echo "Cleaning up existing processes..."
pkill -f "working_ml_production_bot_improved" || true
tmux kill-session -t mlbot_improved 2>/dev/null || true

# Load environment
echo "Loading environment..."
export $(cat .env | grep -v '^#' | xargs)

# Check if required files exist
echo "Checking files..."
if [ ! -f "working_ml_production_bot_improved.py" ]; then
    echo "ERROR: working_ml_production_bot_improved.py not found!"
    exit 1
fi

if [ ! -f "improved_feature_generator.py" ]; then
    echo "ERROR: improved_feature_generator.py not found!"
    exit 1
fi

if [ ! -f "data/historical_data.duckdb" ]; then
    echo "ERROR: data/historical_data.duckdb not found!"
    exit 1
fi

echo "All required files present."

# Start the bot
echo "Starting bot in tmux session..."
tmux new-session -d -s mlbot_improved bash -c '
cd /home/ubuntu/mlbot
export $(cat .env | grep -v "^#" | xargs)
echo "Starting Improved ML Production Bot..."
echo "Time: $(date)"
echo "Python: $(which python3)"
echo "Working Directory: $(pwd)"
python3 working_ml_production_bot_improved.py 2>&1 | tee -a logs/improved_bot.log
'

# Wait for startup
echo "Waiting for bot to start..."
sleep 10

# Check if running
echo -e "\n=== Bot Status ==="
if ps aux | grep "working_ml_production_bot_improved" | grep -v grep; then
    echo "✅ Bot is running!"
else
    echo "❌ Bot is not running. Checking logs..."
    tail -30 logs/improved_bot.log 2>/dev/null || echo "No log file yet"
    echo -e "\nChecking tmux output..."
    tmux capture-pane -t mlbot_improved -p | tail -30
fi

echo -e "\n=== Active tmux sessions ==="
tmux ls
ENDSSH

echo -e "\n✅ Startup script completed!"
echo ""
echo "📌 To monitor the bot:"
echo "   ssh -i $EC2_KEY $EC2_HOST"
echo "   tmux attach -t mlbot_improved"
echo ""
echo "📊 To check logs:"
echo "   ssh -i $EC2_KEY $EC2_HOST 'tail -f /home/ubuntu/mlbot/logs/improved_bot.log'"