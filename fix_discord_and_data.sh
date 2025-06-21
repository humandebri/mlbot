#!/bin/bash

echo "🔧 Fixing Discord notifications and data issues..."

ssh -i ~/.ssh/mlbot-key-1749802416.pem ubuntu@13.212.91.54 << 'EOF'
cd /home/ubuntu/mlbot

# Stop current bot
tmux send-keys -t mlbot C-c 2>/dev/null || true
sleep 3

# First, update the historical data end date in original feature generator
echo "📅 Updating feature generator to use latest data..."
python3 -c "
import re

with open('improved_feature_generator.py', 'r') as f:
    content = f.read()

# Replace hardcoded date with dynamic date
content = re.sub(
    r'end_date = datetime\(2025, 6, 11, 15, 0, 0\)',
    '''# Get latest date from database or use current time
            end_date = datetime.utcnow()''',
    content
)

# Also ensure we use the correct table names
content = re.sub(
    r'table_name = f\"klines_\{symbol\.lower\(\)\}\"',
    '''# Try both naming conventions
                for table_name in [f\"kline_{symbol}\", f\"klines_{symbol.lower()}\"]:
                    try:
                        query = f\"\"\"
                        SELECT 
                            CASE 
                                WHEN open_time > 1e12 THEN datetime(open_time/1000, 'unixepoch')
                                ELSE datetime(open_time, 'unixepoch')
                            END as timestamp,
                            open,
                            high,
                            low,
                            close,
                            volume,
                            turnover
                        FROM {table_name}
                        WHERE datetime(open_time/1000, 'unixepoch') >= '{start_date.isoformat()}'
                            AND datetime(open_time/1000, 'unixepoch') <= '{end_date.isoformat()}'
                        ORDER BY open_time ASC
                        \"\"\"
                        df = self.conn.execute(query).df()
                        if len(df) > 0:
                            break
                    except:
                        continue''',
    content
)

with open('improved_feature_generator.py', 'w') as f:
    f.write(content)

print('✅ Feature generator updated to use latest data')
"

# Check Discord notification code
echo ""
echo "🔍 Checking Discord notification implementation..."
grep -n "discord_notifier.send_notification" simple_improved_bot_with_trading_fixed.py | head -5

# Verify the signal sending logic
echo ""
echo "🔍 Checking signal condition logic..."
grep -A 10 -B 5 "if confidence >= self.min_confidence:" simple_improved_bot_with_trading_fixed.py | head -20

# Restart bot with original feature generator
echo ""
echo "🚀 Restarting bot with original feature generator (50% threshold)..."
tmux send-keys -t mlbot "export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq' && python3 simple_improved_bot_with_trading_fixed.py" Enter

echo "✅ Bot restarted"

# Monitor for 1 minute
sleep 60

echo ""
echo "📊 Checking predictions and signals:"
tmux capture-pane -t mlbot -p | tail -100 | grep -E "(pred=|Signal sent|ML Signal|Discord notification sent)" | tail -20

echo ""
echo "🎯 Summary:"
echo "1. Updated feature generator to use latest data dynamically"
echo "2. Bot uses original feature generator (better confidence levels)"
echo "3. 50% confidence threshold maintained"
echo "4. Monitoring for Discord notifications..."

# Check if any positions were taken
echo ""
echo "💼 Checking for open positions:"
tmux capture-pane -t mlbot -p | grep -E "(Order executed|注文実行|position opened)" | tail -5
EOF