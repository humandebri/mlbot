#!/bin/bash

echo "🔧 レポート問題の修正と信頼度向上方法の調査..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. 現在のポジションをシンプルに確認
echo "💰 現在のポジション監視ログ:"
grep "Position ICPUSDT" logs/mlbot_*.log 2>/dev/null | tail -5

# 2. レポートの生成実際の値を確認
echo ""
echo "📄 最近の時間レポート内容:"
grep -A15 "時間レポート" logs/mlbot_*.log 2>/dev/null | tail -30

# 3. prediction_historyが古いデータを含んでいるか確認
echo ""
echo "🕰️ prediction_historyの問題確認:"
echo "預測履歴のサイズ:"
grep "Keep only last 2000 predictions" logs/mlbot_*.log 2>/dev/null | tail -5

# 4. 信頼度向上のためのDuckDBデータギャップを詳細に確認
echo ""
echo "🗄️ DuckDBデータギャップの詳細:"
python3 -c "
import duckdb
from datetime import datetime, timedelta

try:
    conn = duckdb.connect('data/historical_data.duckdb', read_only=True)
    
    print('各シンボルのデータギャップ分析:')
    print('-' * 60)
    
    for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
        # all_klinesテーブルを確認
        try:
            result = conn.execute(f\"\"\"\n                SELECT 
                    COUNT(*) as count,
                    to_timestamp(MIN(open_time/1000)) as min_date,
                    to_timestamp(MAX(open_time/1000)) as max_date,
                    (MAX(open_time) - MIN(open_time)) / (1000.0 * 60 * 60 * 24) as days_span
                FROM all_klines
                WHERE symbol = '{symbol}'
            \"\"\").fetchone()
            
            if result and result[0] > 0:
                print(f'\\n{symbol}:')
                print(f'  レコード数: {result[0]:,}')
                print(f'  期間: {result[1]} 〜 {result[2]}')
                print(f'  日数: {result[3]:.1f}日')
                
                # ギャップ分析
                current_time = datetime.utcnow()
                latest_time = result[2]
                gap_days = (current_time - latest_time).total_seconds() / (60 * 60 * 24)
                print(f'  現在とのギャップ: {gap_days:.1f}日')
                
                # データ密度確認
                expected_records = result[3] * 1440  # 1分足の場合
                density = (result[0] / expected_records) * 100 if expected_records > 0 else 0
                print(f'  データ密度: {density:.1f}%')
                
        except Exception as e:
            print(f'{symbol}: エラー - {e}')
    
    # 古いテーブルの確認
    print('\\n\\n他の履歴テーブル:')
    print('-' * 60)
    
    tables = conn.execute(\"SHOW TABLES\").fetchall()
    for table in tables:
        table_name = table[0]
        if 'kline' in table_name and table_name != 'all_klines':
            try:
                count = conn.execute(f\"SELECT COUNT(*) FROM {table_name}\").fetchone()[0]
                date_range = conn.execute(f\"\"\"\n                    SELECT 
                        to_timestamp(MIN(open_time/1000)) as min_date,
                        to_timestamp(MAX(open_time/1000)) as max_date
                    FROM {table_name}
                \"\"\").fetchone()
                
                if count > 0:
                    print(f'\\n{table_name}:')
                    print(f'  レコード数: {count:,}')
                    print(f'  期間: {date_range[0]} 〜 {date_range[1]}')
            except:
                pass
                
finally:
    conn.close()
"

# 5. Redisの最新データを確認
echo ""
echo "🔴 Redisの最新データ状況:"
python3 -c "
import redis
import json
from datetime import datetime

try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    # ストリームから最新データを取得
    entries = r.xrevrange('market_data:kline', count=100)
    
    if entries:
        symbols_data = {}
        for entry_id, data in entries:
            try:
                parsed = json.loads(data.get('data', '{}'))
                topic = parsed.get('topic', '')
                
                for symbol in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
                    if symbol in topic and symbol not in symbols_data:
                        timestamp = datetime.fromtimestamp(parsed.get('timestamp', 0))
                        symbols_data[symbol] = timestamp
                        
            except:
                pass
        
        print('Redis最新データ:')
        for symbol, timestamp in symbols_data.items():
            print(f'  {symbol}: {timestamp}')
    else:
        print('Redisにデータがありません')
except Exception as e:
    print(f'Redisエラー: {e}')
"

# 6. 信頼度向上の具体的な方法
echo ""
echo "💡 信頼度50%以上を達成する方法:"
echo ""
echo "1. DuckDBデータの最新化:"
echo "   - 現在: 2025-06-19までのデータ"
echo "   - 必要: 2025-06-21までのデータ追加"
echo "   - 実行: python3 update_duckdb_enhanced_fixed.py"
echo ""
echo "2. 履歴期間の延長:"
echo "   - 現在: 60日間"
echo "   - 推奨: 120-180日間"
echo "   - 効果: 技術指標の精度向上"
echo ""
echo "3. 特徴量計算の最適化:"
echo "   - ボリュームデータの正規化"
echo "   - マーケットレジームの精緻な判定"
echo "   - RSI/MACDのパラメータ調整"

# 7. 具体的な実行コマンド
echo ""
echo "🚀 実行コマンド:"
echo "# 1. ボットを停止"
echo "pkill -f simple_improved_bot_with_trading_fixed.py"
echo ""
echo "# 2. DuckDBを最新化"
echo "python3 update_duckdb_enhanced_fixed.py --lookback-hours 96"
echo ""
echo "# 3. lookback期間を延長"
echo "python3 extend_lookback_period.py --days 150"
echo ""
echo "# 4. ボットを再起動"
echo "nohup python3 simple_improved_bot_with_trading_fixed.py > logs/mlbot_enhanced_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"

EOF

echo ""
echo "✅ 調査完了！"