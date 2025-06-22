#!/bin/bash

echo "🔄 Extending lookback period to improve confidence..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. Run lookback extension
echo ""
echo "🕰️ Extending lookback period from 60 to 120 days..."
python3 extend_lookback_period.py --days 120

# 2. Stop current bot
echo ""
echo "⏹️  Stopping current bot..."
pkill -f simple_improved_bot_with_trading_fixed.py || true
sleep 5

# 3. Restart bot with extended lookback
echo ""
echo "🚀 Starting bot with extended lookback period..."
export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq'
nohup python3 simple_improved_bot_with_trading_fixed.py > logs/mlbot_extended_$(date +%Y%m%d_%H%M%S).log 2>&1 &
BOT_PID=$!
echo "Bot PID: $BOT_PID"

# 4. Monitor initial predictions
echo ""
echo "🔍 Waiting for bot to initialize with extended lookback..."
sleep 30

# 5. Check confidence levels
echo ""
echo "🎯 Checking confidence levels with 120-day lookback:"
LATEST_LOG=$(ls -t logs/mlbot_extended_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Latest predictions:"
    tail -100 "$LATEST_LOG" | grep -E "conf=" | tail -20
    
    # Check for 50%+ confidence
    HIGH_CONF=$(tail -200 "$LATEST_LOG" | grep -E "conf=[5-9][0-9]\.[0-9]+%" | wc -l)
    if [ "$HIGH_CONF" -gt 0 ]; then
        echo ""
        echo "✅ SUCCESS! Found $HIGH_CONF predictions with 50%+ confidence!"
        echo "High confidence predictions:"
        tail -200 "$LATEST_LOG" | grep -E "conf=[5-9][0-9]\.[0-9]+%" | tail -10
    else
        echo ""
        echo "💡 Still below 50% - checking max confidence levels:"
        tail -200 "$LATEST_LOG" | grep -oE "conf=[0-9]+\.[0-9]+%" | sort -t= -k2 -nr | head -10
    fi
fi

# 6. Summary
echo ""
echo "📄 Configuration Summary:"
echo "  - Lookback period: 120 days (extended from 60)"
echo "  - Bot PID: $BOT_PID (running with nohup)"
echo "  - Feature generator: Read-only mode"
echo "  - Auto updater: Running separately"
echo ""
echo "🎯 Target: 50%+ confidence for signal generation"

# 7. Show monitoring commands
echo ""
echo "📊 Monitor with:"
echo "  tail -f $LATEST_LOG | grep -E 'conf=[5-9][0-9]'"
echo "  tail -f $LATEST_LOG | grep 'Signal sent'"
EOF

echo ""
echo "✅ Lookback extension completed!"