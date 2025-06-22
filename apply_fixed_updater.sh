#!/bin/bash

echo "🔧 Applying fixed DuckDB updater..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. Run the fixed updater
echo ""
echo "🔄 Running fixed DuckDB updater..."
python3 update_duckdb_enhanced_fixed.py --lookback-hours 72

# 2. Update auto_update_duckdb.py to use the fixed version
echo ""
echo "📝 Updating auto update script to use fixed version..."
sed -i 's/update_duckdb_enhanced.py/update_duckdb_enhanced_fixed.py/g' auto_update_duckdb.py

# 3. Restart auto updater
echo ""
echo "🔄 Restarting auto updater with fixed version..."
pkill -f auto_update_duckdb.py || true
sleep 2
nohup python3 auto_update_duckdb.py > logs/auto_update_duckdb.log 2>&1 &
AUTO_PID=$!
echo "Auto updater PID: $AUTO_PID"

# 4. Check bot confidence levels with updated data
echo ""
echo "📊 Checking bot confidence with updated data..."
sleep 20
LATEST_LOG=$(ls -t logs/mlbot_readonly_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Latest confidence levels:"
    tail -100 "$LATEST_LOG" | grep -E "conf=" | tail -15
fi

# 5. Verify DuckDB data
echo ""
echo "📊 Verifying DuckDB data:"
python3 -c "
import duckdb
conn = duckdb.connect('data/historical_data.duckdb', read_only=True)
try:
    result = conn.execute(\"\"\"\n        SELECT 
            symbol,
            COUNT(*) as count,
            to_timestamp(MAX(open_time/1000)) as latest
        FROM all_klines
        GROUP BY symbol
    \"\"\").fetchall()
    for row in result:
        print(f'{row[0]}: {row[1]:,} records, latest: {row[2]}')
except Exception as e:
    print(f'Error: {e}')
conn.close()
"

# 6. Check if confidence improved
echo ""
echo "🎯 Checking if confidence improved to 50%+..."
if [ -n "$LATEST_LOG" ]; then
    HIGH_CONF=$(tail -200 "$LATEST_LOG" | grep -E "conf=[5-9][0-9]\.[0-9]+%" | wc -l)
    if [ "$HIGH_CONF" -gt 0 ]; then
        echo "✅ Found $HIGH_CONF predictions with 50%+ confidence!"
        tail -200 "$LATEST_LOG" | grep -E "conf=[5-9][0-9]\.[0-9]+%" | tail -5
    else
        echo "⚠️  No 50%+ confidence predictions yet"
        echo "Current max confidence:"
        tail -200 "$LATEST_LOG" | grep -oE "conf=[0-9]+\.[0-9]+%" | sort -t= -k2 -nr | head -5
    fi
fi

echo ""
echo "✅ Fixed updater applied!"
echo ""
echo "📄 Next steps:"
echo "  1. Monitor for confidence improvement"
echo "  2. If still < 50%, run: python extend_lookback_period.py --days 120"
echo "  3. Check logs: tail -f logs/auto_update_duckdb.log"
EOF

echo ""
echo "✅ Local script completed!"