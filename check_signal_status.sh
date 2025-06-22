#!/bin/bash

echo "🔍 Checking detailed signal status..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. Check current bot status
echo "🤖 Bot Process Status:"
ps aux | grep simple_improved_bot | grep -v grep

# 2. Check latest log file
LATEST_LOG=$(ls -t logs/mlbot_adjusted_*.log 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
    LATEST_LOG=$(ls -t logs/mlbot_*.log 2>/dev/null | head -1)
fi

echo ""
echo "📄 Analyzing log: $LATEST_LOG"

# 3. Check for ICPUSDT signals (43.02% confidence)
echo ""
echo "🎯 ICPUSDT Predictions (should trigger at 43%):"
tail -300 "$LATEST_LOG" | grep "ICPUSDT.*conf=43" | tail -10

# 4. Check signal cooldown status
echo ""
echo "⏱️  Checking cooldown status:"
tail -500 "$LATEST_LOG" | grep -E "(last_signal_time|cooldown)" | tail -10

# 5. Check if signals are being evaluated
echo ""
echo "🔔 Signal evaluation logic:"
tail -500 "$LATEST_LOG" | grep -E "(confidence >= self.min_confidence|Signal sent|last_signal)" | tail -20

# 6. Check exact confidence values
echo ""
echo "📊 All confidence values in last 5 minutes:"
tail -500 "$LATEST_LOG" | grep -oE "conf=[0-9]+\.[0-9]+%" | sort -u

# 7. Check for any errors
echo ""
echo "⚠️  Recent errors:"
tail -500 "$LATEST_LOG" | grep -iE "(error|exception|failed)" | tail -10

# 8. Force check the min_confidence value
echo ""
echo "🔧 Verifying min_confidence in code:"
grep "self.min_confidence" simple_improved_bot_with_trading_fixed.py | head -5

# 9. Check Discord webhook status
echo ""
echo "💬 Discord webhook status:"
tail -500 "$LATEST_LOG" | grep -i "discord" | tail -5

# 10. Live monitoring for next signal
echo ""
echo "📈 Starting live monitoring for signals..."
echo "Watching for ICPUSDT signals (43.02% > 43.0% threshold)..."
timeout 60 tail -f "$LATEST_LOG" | grep -E "(ICPUSDT.*conf=43|Signal sent|Executing)"

echo ""
echo "💡 Analysis complete. If no signals:"
echo "  1. Cooldown may still be active (5 min)"
echo "  2. Exact comparison issue (43.02% vs 43.0%)"
echo "  3. Other trading conditions not met"
EOF

echo ""
echo "✅ Status check completed!"