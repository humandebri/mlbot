#!/bin/bash

echo "🎯 Adjusting confidence threshold temporarily to enable signals..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. Update bot configuration
echo ""
echo "📝 Adjusting confidence threshold from 45% to 43%..."
cp simple_improved_bot_with_trading_fixed.py simple_improved_bot_with_trading_fixed.py.backup_threshold

# Change min_confidence to 0.43
sed -i 's/self.min_confidence = 0.45/self.min_confidence = 0.43/g' simple_improved_bot_with_trading_fixed.py

# 2. Stop current bot
echo ""
echo "⏹️  Stopping current bot..."
pkill -f simple_improved_bot_with_trading_fixed.py || true
sleep 5

# 3. Restart bot with adjusted threshold
echo ""
echo "🚀 Starting bot with 43% confidence threshold..."
export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq'
nohup python3 simple_improved_bot_with_trading_fixed.py > logs/mlbot_adjusted_$(date +%Y%m%d_%H%M%S).log 2>&1 &
BOT_PID=$!
echo "Bot PID: $BOT_PID"

# 4. Monitor for signals
echo ""
echo "🔔 Waiting for bot to generate signals..."
sleep 30

# 5. Check for signals
echo ""
echo "🎯 Checking for ML signals:"
LATEST_LOG=$(ls -t logs/mlbot_adjusted_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    # Check predictions
    echo "Latest predictions:"
    tail -100 "$LATEST_LOG" | grep -E "conf=" | tail -10
    
    # Check for signals
    SIGNALS=$(tail -200 "$LATEST_LOG" | grep -E "(Signal sent|ML Signal|Executing)" | wc -l)
    if [ "$SIGNALS" -gt 0 ]; then
        echo ""
        echo "✅ SUCCESS! Found $SIGNALS signal(s)!"
        echo "Signal details:"
        tail -200 "$LATEST_LOG" | grep -E "(Signal sent|ML Signal|Executing)" | tail -10
    else
        echo ""
        echo "⚠️  No signals generated yet"
    fi
    
    # Check Discord notifications
    echo ""
    echo "💬 Discord notifications:"
    tail -200 "$LATEST_LOG" | grep -i discord | tail -5
fi

# 6. Summary
echo ""
echo "📄 Configuration:"
echo "  - Confidence threshold: 43% (adjusted from 45%)"
echo "  - Expected signals: SELL signals for BTC/ETH/ICP"
echo "  - Discord notifications: Enabled"
echo "  - Bot PID: $BOT_PID"
echo ""
echo "📈 Note: This is a temporary adjustment for testing"
echo "Goal is still to achieve 50%+ confidence with proper data"

# 7. Show real-time monitoring
echo ""
echo "📊 Real-time monitoring:"
echo "  tail -f $LATEST_LOG | grep -E '(Signal|conf=4[3-9])'"
EOF

echo ""
echo "✅ Threshold adjustment completed!"