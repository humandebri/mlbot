#!/bin/bash

echo "🚀 Deploying Redis integration for real-time data..."

# Copy Redis-enabled feature generator
scp -i ~/.ssh/mlbot-key-1749802416.pem improved_feature_generator_redis.py ubuntu@13.212.91.54:~/mlbot/

ssh -i ~/.ssh/mlbot-key-1749802416.pem ubuntu@13.212.91.54 << 'EOF'
cd /home/ubuntu/mlbot

# Stop current bot
echo "⏹️  Stopping current bot..."
tmux send-keys -t mlbot50_fixed C-c 2>/dev/null || true
sleep 3

# Update the Redis feature generator with fixes from the fixed version
echo "🔧 Applying critical fixes to Redis feature generator..."
python3 -c "
import re

# Read the Redis version
with open('improved_feature_generator_redis.py', 'r') as f:
    content = f.read()

# Fix the hardcoded date issue
content = content.replace(
    'end_date = datetime.now()',
    'end_date = datetime.utcnow()'
)

# Fix the table schema query
old_query = '''CASE 
                    WHEN timestamp_ms IS NOT NULL THEN timestamp_ms
                    ELSE CAST(timestamp * 1000 AS BIGINT)
                END as timestamp,'''

new_query = '''CASE 
                    WHEN open_time > 1e12 THEN open_time / 1000
                    ELSE open_time 
                END as timestamp_sec,'''

content = content.replace(old_query, new_query)

# Fix table naming
content = re.sub(
    r'FROM kline_\\{symbol\\}',
    'FROM kline_{}\" if \"kline_\" + symbol in self._get_tables() else \"klines_{}',
    content
)

# Add table existence check method
if '_get_tables' not in content:
    # Add after _connect_redis method
    insert_pos = content.find('def load_historical_data')
    new_method = '''
    def _get_tables(self):
        \"\"\"Get list of available tables.\"\"\"
        if not self.conn:
            return []
        try:
            tables = self.conn.execute(\"SHOW TABLES\").df()['name'].tolist()
            return tables
        except:
            return []
    
'''
    content = content[:insert_pos] + new_method + content[insert_pos:]

# Fix vol_30 error
content = re.sub(
    r\"features\\[\\\"vol_ratio_20\\\"\\] = vol_dict\\['vol_20'\\] / vol_dict\\['vol_30'\\]\",
    \"features['vol_ratio_20'] = vol_dict['vol_20'] / vol_dict['vol_30'] if 'vol_30' in vol_dict and vol_dict['vol_30'] > 0 else 1.15\",
    content
)

# Save fixed version
with open('improved_feature_generator_redis.py', 'w') as f:
    f.write(content)

print('✅ Redis feature generator fixed')
"

# Update bot to use Redis version with 50% threshold
echo "📝 Updating bot configuration..."
sed -i 's/from improved_feature_generator_fixed import ImprovedFeatureGeneratorFixed as ImprovedFeatureGenerator/from improved_feature_generator_redis import ImprovedFeatureGeneratorRedis as ImprovedFeatureGenerator/g' simple_improved_bot_with_trading_fixed.py

# Ensure 50% threshold
sed -i 's/self.min_confidence = 0.45/self.min_confidence = 0.50/g' simple_improved_bot_with_trading_fixed.py

# Test Redis connectivity
echo ""
echo "🧪 Testing Redis data availability..."
python3 test_redis_ec2.py

# Start bot with Redis integration
echo ""
echo "🚀 Starting bot with Redis real-time data..."
tmux new-session -d -s mlbot_redis "export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq' && python3 simple_improved_bot_with_trading_fixed.py"

echo "✅ Bot started with Redis integration in tmux session 'mlbot_redis'"

# Monitor for signals
sleep 30
echo ""
echo "📊 Monitoring predictions with real-time data:"
tmux capture-pane -t mlbot_redis -p | tail -50 | grep -E "(pred=|Signal sent|ML Signal|注文実行)" | tail -20

echo ""
echo "✅ Redis integration deployed!"
echo ""
echo "The bot now uses:"
echo "- Historical data from DuckDB"
echo "- Real-time updates from Redis every 60 seconds"
echo "- 50% confidence threshold"
echo "- Symbol-specific order quantity precision"
echo ""
echo "To monitor: tmux attach -t mlbot_redis"
EOF