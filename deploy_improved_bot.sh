#!/bin/bash
set -e

# Configuration
EC2_HOST="ubuntu@13.212.91.54"
EC2_KEY="/Users/0xhude/.ssh/mlbot-key-1749802416.pem"
EC2_DIR="/home/ubuntu/mlbot"
LOCAL_DIR="/Users/0xhude/Desktop/mlbot"

echo "🚀 Deploying Improved ML Production Bot to EC2..."

# Step 1: Stop existing processes
echo "📋 Step 1: Stopping existing processes..."
ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
# Kill any running Python mlbot processes
pkill -f "working_ml_production_bot" || true
pkill -f "production_trading_system" || true
pkill -f "main_dynamic_integration" || true
pkill -f "python.*mlbot" || true

# Kill zombie processes
ps aux | grep 'python.*<defunct>' | awk '{print $2}' | xargs -r kill -9 || true

# List tmux sessions
echo "Active tmux sessions:"
tmux ls 2>/dev/null || echo "No tmux sessions found"
ENDSSH

# Step 2: Create data directory if needed
echo "📋 Step 2: Ensuring data directory exists..."
ssh -i "$EC2_KEY" "$EC2_HOST" "mkdir -p $EC2_DIR/data"

# Step 3: Copy historical database
echo "📋 Step 3: Copying historical database (109MB)..."
scp -i "$EC2_KEY" "$LOCAL_DIR/data/historical_data.duckdb" "$EC2_HOST:$EC2_DIR/data/"

# Step 4: Copy improved scripts
echo "📋 Step 4: Copying improved bot scripts..."
scp -i "$EC2_KEY" "$LOCAL_DIR/improved_feature_generator.py" "$EC2_HOST:$EC2_DIR/"
scp -i "$EC2_KEY" "$LOCAL_DIR/working_ml_production_bot_improved.py" "$EC2_HOST:$EC2_DIR/"

# Step 5: Install any missing dependencies
echo "📋 Step 5: Checking dependencies..."
ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
cd /home/ubuntu/mlbot
source .venv/bin/activate

# Check if duckdb is installed
python -c "import duckdb" 2>/dev/null || pip install duckdb

# Check if pandas is installed  
python -c "import pandas" 2>/dev/null || pip install pandas

echo "Dependencies verified"
ENDSSH

# Step 6: Create startup script
echo "📋 Step 6: Creating startup script..."
ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
cd /home/ubuntu/mlbot

cat > start_improved_bot.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/mlbot
source .venv/bin/activate

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "Starting Improved ML Production Bot..."
echo "Model: v3.1_improved (44 features)"
echo "Feature Generator: Using real historical data"
echo "Database: data/historical_data.duckdb"

# Run the improved bot
python working_ml_production_bot_improved.py
EOF

chmod +x start_improved_bot.sh
ENDSSH

# Step 7: Start the bot in tmux
echo "📋 Step 7: Starting the improved bot..."
ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
cd /home/ubuntu/mlbot

# Kill existing tmux session if exists
tmux kill-session -t mlbot_improved 2>/dev/null || true

# Create new tmux session and start bot
tmux new-session -d -s mlbot_improved './start_improved_bot.sh'

# Wait a moment
sleep 3

# Check if running
echo "Checking bot status..."
tmux ls
ps aux | grep "working_ml_production_bot_improved" | grep -v grep || echo "Bot process not yet visible"

# Show initial logs
echo -e "\n📄 Initial logs:"
tail -n 20 logs/trading.log 2>/dev/null || echo "No logs yet"
ENDSSH

echo "✅ Deployment complete!"
echo ""
echo "📌 To monitor the bot:"
echo "   ssh -i $EC2_KEY $EC2_HOST"
echo "   tmux attach -t mlbot_improved"
echo ""
echo "📊 To check logs:"
echo "   ssh -i $EC2_KEY $EC2_HOST 'tail -f /home/ubuntu/mlbot/logs/trading.log'"
echo ""
echo "🛑 To stop the bot:"
echo "   ssh -i $EC2_KEY $EC2_HOST 'tmux kill-session -t mlbot_improved'"