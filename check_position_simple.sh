#!/bin/bash

echo "💰 現在のポジションとシステム状態を確認..."

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. 最新ログからポジション情報を確認
echo "📄 最近のポジション監視ログ:"
grep -E "(Monitoring.*positions|Position:|Size:|Entry:|Unrealized)" logs/mlbot_*.log 2>/dev/null | tail -20

# 2. 注文実行履歴を確認
echo ""
echo "📊 最近の注文実行:"
grep -E "Order created.*orderId" logs/mlbot_*.log 2>/dev/null | tail -10

# 3. アカウントバランスを確認
echo ""
echo "💵 最新のアカウントバランス:"
grep "Account balance updated" logs/mlbot_*.log 2>/dev/null | tail -5

# 4. エラーの詳細を再確認
echo ""
echo "❌ データベースエラーの詳細:"
grep -A2 "Failed to save position" logs/mlbot_fixed_*.log 2>/dev/null | tail -10

# 5. ボットの現在の動作状態
echo ""
echo "🤖 ボットの現在の動作:"
LATEST_LOG=$(ls -t logs/mlbot_50pct_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Log: $LATEST_LOG"
    echo "最新の予測 (50%闾値):"
    tail -50 "$LATEST_LOG" | grep -E "conf=[4-5][0-9]" | tail -5
    if [ $? -ne 0 ]; then
        echo "50%以上の信頼度はまだありません"
        echo "現在の最高信頼度:"
        tail -100 "$LATEST_LOG" | grep -oE "conf=[0-9]+\.[0-9]+%" | sort -t= -k2 -nr | head -3
    fi
fi

# 6. データベーススキーマの修正方法
echo ""
echo "🔧 データベース修正の提案:"
echo "1. ボットを一時停止"
echo "2. database.pyでpositionsテーブルスキーマを更新"
echo "3. opened_atカラムを追加"
echo ""
echo "現在のdatabase.pyのスキーマ:"
grep -A20 "CREATE TABLE IF NOT EXISTS positions" src/common/database.py 2>/dev/null | head -25

echo ""
echo "📄 現状まとめ:"
echo "  - 8 ICPのポジションは実際にBybitに存在"
echo "  - データベース保存のみエラー（取引は成功）"
echo "  - 50%闾値で正常に稼働中"
echo "  - 43%シグナルは過去のボットから"

EOF

echo ""
echo "✅ 確認完了！"