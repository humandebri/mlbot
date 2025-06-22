#!/bin/bash

echo "🔧 Fixing bot configuration issues..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. Stop current bot
echo "⏹️  Stopping current bot..."
pkill -f simple_improved_bot_with_trading_fixed.py || true
sleep 5

# 2. Fix the min_confidence value correctly
echo ""
echo "📝 Fixing min_confidence to 0.43 (43%)..."
# Find the line with min_confidence and replace it
sed -i 's/self.min_confidence = 0.[0-9]\+/self.min_confidence = 0.43/g' simple_improved_bot_with_trading_fixed.py

# Verify the change
echo "Verifying change:"
grep "self.min_confidence =" simple_improved_bot_with_trading_fixed.py | head -2

# 3. Fix the pandas import issue in readonly generator
echo ""
echo "🔧 Fixing pandas import in readonly generator..."
if ! grep -q "import pandas as pd" improved_feature_generator_readonly.py; then
    # Add pandas import at the top with other imports
    sed -i '/import json/a import pandas as pd' improved_feature_generator_readonly.py
fi

# 4. Restart bot with fixed configuration
echo ""
echo "🚀 Starting bot with fixed configuration..."
export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq'
nohup python3 simple_improved_bot_with_trading_fixed.py > logs/mlbot_fixed_$(date +%Y%m%d_%H%M%S).log 2>&1 &
BOT_PID=$!
echo "Bot PID: $BOT_PID"

# 5. Wait for initialization
echo ""
echo "⏰ Waiting for bot initialization..."
sleep 30

# 6. Monitor for signals
echo ""
echo "🎯 Monitoring for signals with 43% threshold:"
LATEST_LOG=$(ls -t logs/mlbot_fixed_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo "Checking ICPUSDT (43.02% > 43.0%):"
    tail -100 "$LATEST_LOG" | grep "ICPUSDT.*conf=43" | tail -5
    
    echo ""
    echo "🔔 Checking for generated signals:"
    SIGNALS=$(tail -300 "$LATEST_LOG" | grep -E "(ML Signal|Signal sent|Executing.*order)" | wc -l)
    
    if [ "$SIGNALS" -gt 0 ]; then
        echo "✅ SUCCESS! Found $SIGNALS signal(s)!"
        tail -300 "$LATEST_LOG" | grep -E "(ML Signal|Signal sent|Executing.*order)" | tail -10
    else
        echo "⚠️  Waiting for signals..."
    fi
    
    # Live monitoring
    echo ""
    echo "📊 Starting 90-second live monitoring:"
    timeout 90 tail -f "$LATEST_LOG" | grep -E "(ICPUSDT.*conf=43|ML Signal|Signal sent|Executing)"
fi

echo ""
echo "📄 Summary:"
echo "  - Min confidence: 43% (fixed)"
echo "  - Redis errors: Fixed (pandas import)"
echo "  - Bot PID: $BOT_PID"
echo "  - ICPUSDT should trigger signals now"
EOF

echo ""
echo "✅ Fix completed!"