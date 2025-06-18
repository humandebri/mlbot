#!/bin/bash
# Deploy technical indicator fix to EC2

set -e

EC2_HOST="13.212.91.54"
EC2_USER="ubuntu"
KEY_PATH="$HOME/.ssh/mlbot-key-1749802416.pem"

echo "🚀 Deploying technical indicator fix to EC2..."

# 1. Copy updated files
echo "📦 Copying updated files..."

# Copy feature hub files
scp -i "$KEY_PATH" \
    src/feature_hub/main.py \
    "$EC2_USER@$EC2_HOST:~/mlbot/src/feature_hub/"

scp -i "$KEY_PATH" \
    src/feature_hub/technical_indicators.py \
    "$EC2_USER@$EC2_HOST:~/mlbot/src/feature_hub/"

# Copy integration files  
scp -i "$KEY_PATH" \
    src/integration/dynamic_trading_coordinator.py \
    "$EC2_USER@$EC2_HOST:~/mlbot/src/integration/"

# Copy ML pipeline files
scp -i "$KEY_PATH" \
    src/ml_pipeline/inference_engine.py \
    "$EC2_USER@$EC2_HOST:~/mlbot/src/ml_pipeline/"

# Copy common files
scp -i "$KEY_PATH" \
    src/common/bybit_client.py \
    "$EC2_USER@$EC2_HOST:~/mlbot/src/common/"

# Copy order router files  
scp -i "$KEY_PATH" \
    src/order_router/order_executor.py \
    "$EC2_USER@$EC2_HOST:~/mlbot/src/order_router/"

# Copy test scripts
scp -i "$KEY_PATH" \
    test_technical_indicators.py \
    test_realistic_signals.py \
    diagnose_model_prediction.py \
    "$EC2_USER@$EC2_HOST:~/mlbot/"

# 2. SSH to EC2 and restart the system
echo "🔄 Restarting trading system on EC2..."
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_HOST" << 'EOF'
cd ~/mlbot

# Kill existing process
echo "Stopping existing process..."
pkill -f "main_dynamic_integration.py" || true
sleep 2

# Test technical indicators (optional)
echo "Testing technical indicators..."
python3 test_technical_indicators.py

echo "Checking signal generation..."
python3 test_realistic_signals.py | tail -20

# Start system in tmux
echo "Starting system in tmux..."
tmux new-session -d -s trading 'python3 main_dynamic_integration.py 2>&1 | tee -a logs/trading_$(date +%Y%m%d).log'

# Check if started
sleep 5
ps aux | grep -v grep | grep "main_dynamic_integration.py" && echo "✅ System started successfully" || echo "❌ Failed to start"

echo "📊 Recent logs:"
tail -n 20 logs/trading_*.log | grep -E "(Technical|feature|prediction|confidence)"
EOF

echo "✅ Deployment complete!"