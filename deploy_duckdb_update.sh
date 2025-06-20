#!/bin/bash

echo "🚀 Deploying DuckDB update script to EC2..."

# EC2 details
EC2_HOST="ec2-user@13.212.91.54"
EC2_KEY="~/.ssh/mlbot-key-*.pem"

# First, copy the update script
echo "📤 Copying update script..."
scp -i $EC2_KEY update_duckdb_complete.py $EC2_HOST:~/mlbot/

# SSH into EC2 and run the update
echo "🔄 Running DuckDB update on EC2..."
ssh -i $EC2_KEY $EC2_HOST << 'EOF'
cd ~/mlbot

# Stop the bot temporarily
echo "⏸️  Stopping trading bot..."
pkill -f "simple_improved_bot_with_trading_fixed.py" || true

# Backup current database
echo "💾 Backing up current database..."
cp data/historical_data.duckdb data/historical_data.duckdb.backup.$(date +%Y%m%d_%H%M%S)

# Run the update
echo "🔄 Running DuckDB update..."
python3 update_duckdb_complete.py

# Show results
echo "📊 Update complete! Checking database statistics..."
python3 -c "
import duckdb
conn = duckdb.connect('data/historical_data.duckdb')
for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
    try:
        result = conn.execute(f'''
            SELECT 
                COUNT(*) as records,
                datetime(MIN(open_time/1000), 'unixepoch') as earliest,
                datetime(MAX(open_time/1000), 'unixepoch') as latest
            FROM kline_{symbol}
        ''').fetchone()
        print(f'{symbol}: {result[0]} records, {result[1]} to {result[2]}')
    except:
        print(f'{symbol}: No data')
conn.close()
"

# Restart the bot
echo "🚀 Restarting trading bot..."
cd ~/mlbot
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq"
nohup python3 simple_improved_bot_with_trading_fixed.py > logs/bot.log 2>&1 &

echo "✅ Update complete and bot restarted!"
EOF

echo "✅ Deployment complete!"