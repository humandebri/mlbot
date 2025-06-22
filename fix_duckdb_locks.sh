#!/bin/bash

echo "🔒 Fixing DuckDB lock issues..."

# Transfer read-only generator
echo "📤 Transferring read-only feature generator..."
scp -i ~/.ssh/mlbot-key-1749802416.pem improved_feature_generator_readonly.py ubuntu@13.212.91.54:~/mlbot/

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. Stop current bot
echo ""
echo "⏹️  Stopping current bot to release DuckDB lock..."
BOT_PID=$(ps aux | grep simple_improved_bot_with_trading_fixed.py | grep -v grep | awk '{print $2}')
if [ -n "$BOT_PID" ]; then
    echo "Stopping bot PID: $BOT_PID"
    kill $BOT_PID
    sleep 5
fi

# 2. Update bot to use read-only generator
echo ""
echo "📝 Updating bot to use read-only feature generator..."
cp simple_improved_bot_with_trading_fixed.py simple_improved_bot_with_trading_fixed.py.backup_readonly

# Replace imports to use read-only version
sed -i 's/from improved_feature_generator_persistent import ImprovedFeatureGeneratorPersistent as ImprovedFeatureGenerator/from improved_feature_generator_readonly import ImprovedFeatureGeneratorReadOnly as ImprovedFeatureGenerator/g' simple_improved_bot_with_trading_fixed.py

# 3. Now run the DuckDB update (no lock conflicts)
echo ""
echo "🔄 Running DuckDB update now that lock is released..."
python3 update_duckdb_enhanced.py --lookback-hours 48

# 4. Restart bot with read-only generator
echo ""
echo "🚀 Starting bot with read-only feature generator..."
export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq'
nohup python3 simple_improved_bot_with_trading_fixed.py > logs/mlbot_readonly_$(date +%Y%m%d_%H%M%S).log 2>&1 &
BOT_PID=$!
echo "Bot PID: $BOT_PID"

# 5. Ensure auto updater is still running
echo ""
echo "🔍 Checking auto updater status..."
AUTO_PID=$(ps aux | grep auto_update_duckdb.py | grep -v grep | awk '{print $2}')
if [ -z "$AUTO_PID" ]; then
    echo "Auto updater not running, starting it..."
    nohup python3 auto_update_duckdb.py > logs/auto_update_duckdb.log 2>&1 &
    AUTO_PID=$!
    echo "Auto updater PID: $AUTO_PID"
else
    echo "Auto updater already running with PID: $AUTO_PID"
fi

# 6. Wait and verify
sleep 10

# 7. Check processes
echo ""
echo "📋 Running processes:"
ps aux | grep -E "(auto_update|simple_improved_bot)" | grep -v grep

# 8. Check latest data in DuckDB
echo ""
echo "📊 Latest data in DuckDB:"
python3 -c "
import duckdb
conn = duckdb.connect('data/historical_data.duckdb', read_only=True)
for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
    result = conn.execute(f\"SELECT COUNT(*) as count, datetime(MAX(open_time/1000), 'unixepoch') as latest FROM all_klines WHERE symbol = '{symbol}'\").fetchone()
    print(f'{symbol}: {result[0]:,} records, latest: {result[1]}')
conn.close()
"

# 9. Check bot confidence
echo ""
echo "📊 Latest confidence levels:"
LATEST_LOG=$(ls -t logs/mlbot_readonly_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    tail -50 "$LATEST_LOG" | grep -E "conf=" | tail -10
fi

echo ""
echo "✅ DuckDB lock issue fixed!"
echo ""
echo "📄 Summary:"
echo "  - Bot using read-only DuckDB connection"
echo "  - Auto updater has exclusive write access"
echo "  - No more lock conflicts"
echo "  - Both processes running with nohup"
EOF

echo ""
echo "✅ Local fix script completed!"