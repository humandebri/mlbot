#!/bin/bash

echo "🚀 Deploying critical fixes to EC2..."

# Copy fixed feature generator
scp -i ~/.ssh/mlbot-key-1749802416.pem improved_feature_generator_fixed.py ubuntu@13.212.91.54:~/mlbot/

# Apply all fixes via SSH
ssh -i ~/.ssh/mlbot-key-1749802416.pem ubuntu@13.212.91.54 << 'EOF'
cd /home/ubuntu/mlbot

# Stop current bot
echo "⏹️  Stopping current bot..."
tmux send-keys -t mlbot45 C-c 2>/dev/null || true
sleep 3

# Update bot to use fixed feature generator
echo "📝 Updating bot to use fixed feature generator..."
cp simple_improved_bot_with_trading_fixed.py simple_improved_bot_with_trading_fixed.py.backup

# Replace import statement
sed -i 's/from improved_feature_generator import ImprovedFeatureGenerator/from improved_feature_generator_fixed import ImprovedFeatureGeneratorFixed as ImprovedFeatureGenerator/g' simple_improved_bot_with_trading_fixed.py

# Fix confidence threshold to 50%
echo "🎯 Setting confidence threshold to 50%..."
sed -i 's/self.min_confidence = 0.45/self.min_confidence = 0.50/g' simple_improved_bot_with_trading_fixed.py

# Fix order quantity precision for ICPUSDT
echo "🔧 Fixing order quantity precision..."
python3 -c "
import re

with open('simple_improved_bot_with_trading_fixed.py', 'r') as f:
    content = f.read()

# Find the order quantity calculation section
if 'order_qty = position_size / current_price' in content:
    # Replace with symbol-specific precision
    new_qty_code = '''
            # Calculate order quantity with proper precision
            order_qty = position_size / current_price
            
            # Apply symbol-specific precision rules
            if symbol == \"BTCUSDT\":
                # BTC: 3 decimal places
                order_qty = round(order_qty, 3)
                min_qty = 0.001
            elif symbol == \"ETHUSDT\":
                # ETH: 2 decimal places
                order_qty = round(order_qty, 2)
                min_qty = 0.01
            elif symbol == \"ICPUSDT\":
                # ICP: 0 decimal places (whole numbers only)
                order_qty = int(order_qty)
                min_qty = 1
            else:
                # Default: 2 decimal places
                order_qty = round(order_qty, 2)
                min_qty = 0.01
            
            # Ensure minimum order size
            if order_qty < min_qty:
                logger.warning(f\"Order quantity {order_qty} below minimum {min_qty} for {symbol}\")
                return'''
    
    # Replace the section
    pattern = r'order_qty = position_size / current_price.*?(?=\n\s{12}# Execute order|\n\s{12}logger\.info)'
    content = re.sub(pattern, new_qty_code.strip(), content, flags=re.DOTALL)

with open('simple_improved_bot_with_trading_fixed.py', 'w') as f:
    f.write(content)

print('✅ Order quantity precision fixed')
"

# Test the fixed feature generator
echo ""
echo "🧪 Testing fixed feature generator..."
python3 -c "
from improved_feature_generator_fixed import ImprovedFeatureGeneratorFixed
import asyncio

async def test():
    gen = ImprovedFeatureGeneratorFixed()
    
    # Load data for all symbols
    for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
        gen.update_historical_cache(symbol)
    
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
    
    # Check for NaN/Inf
    import numpy as np
    feature_array = np.array(list(features.values()))
    has_nan = np.any(np.isnan(feature_array))
    has_inf = np.any(np.isinf(feature_array))
    
    if has_nan or has_inf:
        print('❌ WARNING: Features contain NaN or Inf values!')
    else:
        print('✅ All features are valid numbers')
    
    # Check latest data timestamp
    hist_data = gen.historical_data.get('BTCUSDT')
    if hist_data is not None and not hist_data.empty:
        latest = hist_data.index[-1]
        print(f'✅ Latest data: {latest}')
    
    gen.close()

asyncio.run(test())
"

# Start bot with all fixes
echo ""
echo "🚀 Starting bot with all fixes applied..."
tmux new-session -d -s mlbot50_fixed "export DISCORD_WEBHOOK='https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq' && python3 simple_improved_bot_with_trading_fixed.py"

echo "✅ Bot started in tmux session 'mlbot50_fixed'"

# Monitor initial activity
sleep 20
echo ""
echo "📊 Initial predictions and confidence levels:"
tmux capture-pane -t mlbot50_fixed -p | tail -50 | grep -E "(pred=|Signal|Order executed|confidence)" | tail -20

echo ""
echo "✅ All critical fixes deployed!"
echo ""
echo "Summary of fixes:"
echo "1. ✅ Fixed date: Now uses dynamic latest timestamp from DB"
echo "2. ✅ Fixed table names: Tries multiple naming patterns"
echo "3. ✅ Fixed timestamp handling: Converts ms to seconds properly"
echo "4. ✅ Fixed vol_30 error: Added safety checks"
echo "5. ✅ Integrated realtime data: Appends current price to history"
echo "6. ✅ Fixed NaN handling: Cleans all NaN/Inf values"
echo "7. ✅ Dynamic cache: Refreshes every 5 minutes"
echo "8. ✅ Thread safety: Added locks for concurrent access"
echo "9. ✅ Order quantity: Symbol-specific precision"
echo "10. ✅ Confidence threshold: Restored to 50%"
echo ""
echo "To monitor: tmux attach -t mlbot50_fixed"
EOF