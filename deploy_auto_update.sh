#!/bin/bash

echo "🚀 Deploying auto update and persistent feature generator to EC2..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

# Execute on EC2
ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# Create logs directory
mkdir -p logs

# 1. Initial DuckDB update
echo ""
echo "🔄 Running initial DuckDB update..."
python3 update_duckdb_enhanced.py --lookback-hours 72

# 2. Start auto update script with nohup
echo ""
echo "🚀 Starting auto update script with nohup..."
nohup python3 auto_update_duckdb.py > logs/auto_update_duckdb.log 2>&1 &
AUTO_UPDATE_PID=$!
echo "Auto update PID: $AUTO_UPDATE_PID"

# 3. Stop current bot
echo ""
echo "⏹️  Stopping current bot..."
pkill -f simple_improved_bot_with_trading_fixed.py || true
sleep 3

# 4. Update bot to use persistent feature generator
echo ""
echo "📝 Updating bot to use persistent feature generator..."
cp simple_improved_bot_with_trading_fixed.py simple_improved_bot_with_trading_fixed.py.backup_persistent

# Update imports
if grep -q "improved_feature_generator_enhanced" simple_improved_bot_with_trading_fixed.py; then
    sed -i 's/from improved_feature_generator_enhanced import ImprovedFeatureGeneratorEnhanced as ImprovedFeatureGenerator/from improved_feature_generator_persistent import ImprovedFeatureGeneratorPersistent as ImprovedFeatureGenerator/g' simple_improved_bot_with_trading_fixed.py
    echo "Updated enhanced import"
fi

if grep -q "from improved_feature_generator import" simple_improved_bot_with_trading_fixed.py; then
    sed -i 's/from improved_feature_generator import ImprovedFeatureGenerator/from improved_feature_generator_persistent import ImprovedFeatureGeneratorPersistent as ImprovedFeatureGenerator/g' simple_improved_bot_with_trading_fixed.py
    echo "Updated standard import"
fi

# 5. Start persistent bot with nohup
echo ""
echo "🚀 Starting bot with persistent feature generator..."
export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq'
nohup python3 simple_improved_bot_with_trading_fixed.py > logs/mlbot_persistent_$(date +%Y%m%d_%H%M%S).log 2>&1 &
BOT_PID=$!
echo "Bot PID: $BOT_PID"

# 6. Verify processes
echo ""
echo "📋 Verifying running processes..."
sleep 5
ps aux | grep -E "(auto_update_duckdb|simple_improved_bot)" | grep -v grep

# 7. Check initial logs
echo ""
echo "📊 Auto update log (checking if started):" 
if [ -f logs/auto_update_duckdb.log ]; then
    tail -20 logs/auto_update_duckdb.log
else
    echo "Waiting for log file..."
fi

echo ""
echo "📊 Bot confidence levels (initial check):"
sleep 10
ls -t logs/mlbot_persistent_*.log | head -1 | xargs tail -50 | grep -E "(conf=|Signal sent|Started)" | tail -10

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📌 Important information:"
echo "  - Auto updater PID: $AUTO_UPDATE_PID (running with nohup)"
echo "  - Bot PID: $BOT_PID (running with nohup)"
echo "  - DuckDB will be updated every hour automatically"
echo "  - Redis data will be persisted every 30 minutes"
echo ""
echo "📋 Monitor with:"
echo "  ssh -i ~/.ssh/mlbot-key-1749802416.pem ubuntu@13.212.91.54"
echo "  tail -f /home/ubuntu/mlbot/logs/auto_update_duckdb.log"
echo "  tail -f /home/ubuntu/mlbot/logs/mlbot_persistent_*.log | grep conf="
EOF

echo ""
echo "✅ Local deployment script completed!"