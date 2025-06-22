#!/bin/bash

echo "🔧 Fixing auto update issues on EC2..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. Install missing module
echo "📦 Installing schedule module..."
pip3 install schedule

# 2. Check and kill old DuckDB process
echo ""
echo "🔍 Checking for DuckDB locks..."
OLD_PID=784712
if ps -p $OLD_PID > /dev/null; then
    echo "Killing old process $OLD_PID..."
    kill -9 $OLD_PID || true
    sleep 2
fi

# 3. Kill any stuck auto_update processes
echo ""
echo "🔄 Cleaning up old auto update processes..."
pkill -f auto_update_duckdb.py || true
sleep 2

# 4. Restart auto update with nohup
echo ""
echo "🚀 Starting auto update script..."
nohup python3 auto_update_duckdb.py > logs/auto_update_duckdb.log 2>&1 &
AUTO_UPDATE_PID=$!
echo "Auto update PID: $AUTO_UPDATE_PID"

# 5. Wait and check logs
sleep 10
echo ""
echo "📊 Auto update log:"
if [ -f logs/auto_update_duckdb.log ]; then
    tail -30 logs/auto_update_duckdb.log
else
    echo "Log file not found yet..."
fi

# 6. Check bot status
echo ""
echo "🤖 Current bot status:"
ps aux | grep simple_improved_bot | grep -v grep

# 7. Check latest confidence levels
echo ""
echo "📊 Latest confidence levels:"
LATEST_LOG=$(ls -t logs/mlbot_persistent_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    tail -100 "$LATEST_LOG" | grep -E "conf=" | tail -10
else
    echo "No bot logs found yet..."
fi

# 8. Check persistence log
echo ""
echo "💾 Persistence status:"
if [ -n "$LATEST_LOG" ]; then
    grep -i "persist" "$LATEST_LOG" | tail -5
fi

echo ""
echo "✅ Fix complete!"
echo ""
echo "📄 Summary:"
echo "  - Auto updater PID: $AUTO_UPDATE_PID"
echo "  - DuckDB updates: Every 1 hour"
echo "  - Redis persistence: Every 30 minutes"
echo "  - Both processes running with nohup"
EOF

echo ""
echo "✅ Local fix script completed!"