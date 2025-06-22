#!/bin/bash

echo "🚀 Deploying enhanced feature generator to EC2..."

# Copy enhanced generator
scp -i ~/.ssh/mlbot-key-1749802416.pem improved_feature_generator_enhanced.py ubuntu@13.212.91.54:~/mlbot/

ssh -i ~/.ssh/mlbot-key-1749802416.pem ubuntu@13.212.91.54 << 'EOF'
cd /home/ubuntu/mlbot

# Stop current bot
echo "⏹️  Stopping current bot..."
pkill -f simple_improved_bot_with_trading_fixed.py || true
sleep 3

# Backup current bot
cp simple_improved_bot_with_trading_fixed.py simple_improved_bot_with_trading_fixed.py.backup

# Update bot to use enhanced generator
echo "📝 Updating bot to use enhanced feature generator..."
sed -i 's/from improved_feature_generator_redis import ImprovedFeatureGeneratorRedis as ImprovedFeatureGenerator/from improved_feature_generator_enhanced import ImprovedFeatureGeneratorEnhanced as ImprovedFeatureGenerator/g' simple_improved_bot_with_trading_fixed.py
sed -i 's/from improved_feature_generator import ImprovedFeatureGenerator/from improved_feature_generator_enhanced import ImprovedFeatureGeneratorEnhanced as ImprovedFeatureGenerator/g' simple_improved_bot_with_trading_fixed.py

# Ensure 50% threshold
sed -i 's/self.min_confidence = 0.45/self.min_confidence = 0.50/g' simple_improved_bot_with_trading_fixed.py

# Test the enhanced generator
echo ""
echo "🧪 Testing enhanced feature generator..."
python3 -c "
from improved_feature_generator_enhanced import ImprovedFeatureGeneratorEnhanced
import asyncio

async def test():
    gen = ImprovedFeatureGeneratorEnhanced(enable_redis=True)
    
    # Test each symbol
    for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
        gen.update_historical_cache(symbol)
        
        # Check data freshness
        if symbol in gen.historical_data and not gen.historical_data[symbol].empty:
            latest = gen.historical_data[symbol].index[-1]
            print(f'{symbol}: {len(gen.historical_data[symbol])} records, latest: {latest}')
    
    # Test feature generation
    test_ticker = {
        'lastPrice': '100000',
        'volume24h': '1000000',
        'highPrice24h': '101000',
        'lowPrice24h': '99000',
        'prevPrice24h': '99500'
    }
    
    features = gen.generate_features(test_ticker, 'BTCUSDT')
    print(f'✅ Generated {len(features)} features')
    
    gen.close()

asyncio.run(test())
"

# Start bot with enhanced generator
echo ""
echo "🚀 Starting bot with enhanced feature generator..."
tmux new-session -d -s mlbot_enhanced "export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq' && python3 simple_improved_bot_with_trading_fixed.py 2>&1 | tee -a logs/mlbot_enhanced_$(date +%Y%m%d_%H%M%S).log"

echo "✅ Bot started in tmux session 'mlbot_enhanced'"

# Monitor for high confidence predictions
sleep 40
echo ""
echo "📊 Monitoring predictions with enhanced generator:"
tmux capture-pane -t mlbot_enhanced -p | tail -100 | grep -E "(pred=|Signal sent|ML Signal)" | tail -20

echo ""
echo "🎯 Looking for confidence levels:"
tmux capture-pane -t mlbot_enhanced -p | tail -100 | grep -oE "conf=[0-9]+\.[0-9]+%" | tail -10

echo ""
echo "✅ Enhanced feature generator deployed!"
echo ""
echo "Expected improvements:"
echo "- Higher confidence levels (50%+ like original)"
echo "- Dynamic data updates (no fixed date)"
echo "- Optional Redis integration for real-time data"
echo "- 5-minute cache refresh"
echo ""
echo "To monitor: tmux attach -t mlbot_enhanced"
EOF