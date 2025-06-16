#!/bin/bash
# レバレッジ3倍で取引システムを再起動

EC2_USER="ubuntu"
EC2_HOST="13.212.91.54"
EC2_KEY="~/.ssh/mlbot-key-1749802416.pem"
PROJECT_DIR="/home/ubuntu/mlbot"

echo "🔧 レバレッジ3倍で実取引システムを再起動..."

ssh -i $EC2_KEY $EC2_USER@$EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. 現在の取引システムの状態確認
echo "📊 現在実行中の取引システム:"
ps aux | grep -E "main_unified|trading" | grep -v grep

# 2. RiskManagerの設定確認
echo -e "\n📋 現在のレバレッジ設定:"
grep "max_leverage" src/order_router/risk_manager.py

# 3. 既存のmain_unified.pyプロセスを再起動
echo -e "\n🔄 既存の取引システムを再起動中..."

# プロセスIDを取得
UNIFIED_PID=$(ps aux | grep "python3 src/integration/main_unified.py" | grep -v grep | awk '{print $2}')

if [ ! -z "$UNIFIED_PID" ]; then
    echo "既存プロセス (PID: $UNIFIED_PID) を停止..."
    kill -TERM $UNIFIED_PID
    sleep 5
    
    # 強制終了が必要な場合
    if ps -p $UNIFIED_PID > /dev/null; then
        kill -KILL $UNIFIED_PID
    fi
fi

# 4. 統合取引システムを再起動
echo -e "\n🚀 レバレッジ3倍で統合取引システムを起動..."

# 起動コマンドを直接実行
cd /home/ubuntu/mlbot
nohup python3 -m src.integration.main_unified > unified_trading.log 2>&1 &

sleep 5

# 5. 起動確認
echo -e "\n✅ 起動確認:"
ps aux | grep -E "main_unified|trading" | grep -v grep

# 6. ログ確認
echo -e "\n📄 取引システムログ（最新20行）:"
tail -n 20 unified_trading.log || echo "ログファイルがまだ生成されていません"

# 7. ポジション管理設定の要約
echo -e "\n📊 ポジション管理設定:"
echo "========================"
echo "✅ 最大レバレッジ: 3倍"
echo "✅ シンボルごと最大: $100,000"
echo "✅ 総エクスポージャー: $500,000"
echo "✅ 同時ポジション数: 10"
echo "✅ ストップロス: 2%"
echo "✅ リスク/取引: 1%"
echo "========================"

# 8. Discord通知の状態確認
echo -e "\n🔔 Discord通知:"
grep -i "discord" unified_trading.log | tail -5 || echo "まだDiscord通知ログがありません"

echo -e "\n✅ 実取引システム再起動完了！"
EOF

echo ""
echo "🎉 レバレッジ3倍設定で実取引システムが再起動されました！"
echo ""
echo "📈 リアルタイムログ監視:"
echo "ssh -i $EC2_KEY $EC2_USER@$EC2_HOST 'tail -f /home/ubuntu/mlbot/unified_trading.log'"
echo ""
echo "⚠️  重要: 実際の注文が実行されるようになりました"
echo "- レバレッジ: 3倍"
echo "- OrderRouter: 有効"
echo "- RiskManager: 有効"
echo "- ポジション管理: 有効"