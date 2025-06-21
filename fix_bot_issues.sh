#!/bin/bash

echo "🔧 Fixing bot issues on EC2..."

ssh -i ~/.ssh/mlbot-key-1749802416.pem ubuntu@13.212.91.54 << 'EOF'
cd /home/ubuntu/mlbot

# Kill existing bot
echo "Stopping current bot..."
pkill -f simple_improved_bot_with_trading_fixed.py || true
sleep 3

# Fix confidence threshold to 50%
echo "Setting confidence threshold to 50%..."
sed -i 's/self.min_confidence = 0.35/self.min_confidence = 0.50/g' simple_improved_bot_with_trading_fixed.py
sed -i 's/self.min_confidence = 0.65/self.min_confidence = 0.50/g' simple_improved_bot_with_trading_fixed.py

# Fix database transaction error
echo "Fixing database transaction error..."
python3 -c "
import re

with open('simple_improved_bot_with_trading_fixed.py', 'r') as f:
    content = f.read()

# Remove any transaction rollback calls
content = re.sub(r'conn\.rollback\(\)', 'pass  # Removed rollback', content)

# Ensure proper connection handling
if 'get_duckdb_connection()' in content and 'conn.close()' not in content:
    # Add proper connection closing
    content = re.sub(
        r'(save_trade\([^)]+\))',
        r'\1\n            if hasattr(self, \"_db_conn\") and self._db_conn:\n                self._db_conn.close()\n                self._db_conn = None',
        content
    )

with open('simple_improved_bot_with_trading_fixed.py', 'w') as f:
    f.write(content)

print('✅ Fixed database handling')
"

# Also check if we're correctly checking the cooldown
echo "Checking signal cooldown logic..."
grep -n "last_signal_time\|signal_cooldown\|self.signal_count" simple_improved_bot_with_trading_fixed.py | head -20

# Start bot in tmux for persistence
echo ""
echo "Starting bot in tmux session..."
tmux kill-session -t mlbot50 2>/dev/null || true
tmux new-session -d -s mlbot50 "export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq' && python3 simple_improved_bot_with_trading_fixed.py"

sleep 5

# Check if running
if tmux ls | grep mlbot50 > /dev/null; then
    echo "✅ Bot started in tmux session 'mlbot50'"
    echo ""
    echo "📋 Monitoring initial predictions..."
    sleep 15
    tmux capture-pane -t mlbot50 -p | tail -30 | grep -E "(pred=|Signal|confidence|ERROR)"
else
    echo "❌ Failed to start tmux session"
fi

echo ""
echo "To attach to session: tmux attach -t mlbot50"
EOF