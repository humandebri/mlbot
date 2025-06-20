#!/bin/bash
# Deploy trading system with daily report functionality

echo "==================================================
日次レポート機能付き取引システムのデプロイ
==================================================

📊 追加機能:
  - 毎日9:00 AM JST (00:00 UTC) に日次レポート送信
  - 1時間ごとの残高更新（既存）
  - 取引実績の自動集計
  - パフォーマンス分析とコメント

"

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "🔧 仮想環境をアクティベート中..."
    source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || {
        echo "❌ Error: Virtual environment not found"
        exit 1
    }
fi

# Check Python version
echo "📦 Python version: $(python --version)"
echo ""

# Stop existing processes
echo "🛑 既存のプロセスを停止中..."
pkill -f "trading_with_real_api" 2>/dev/null
pkill -f "trading_with_daily_report" 2>/dev/null
sleep 2

# Install required dependency (pytz for timezone)
echo "📦 依存関係をインストール中..."
pip install pytz > /dev/null 2>&1

# Start the new system
echo "🚀 日次レポート機能付き取引システムを起動中..."
LOG_FILE="logs/daily_report_system_$(date +%Y%m%d_%H%M%S).log"
nohup python trading_with_daily_report.py > "$LOG_FILE" 2>&1 &
PID=$!

sleep 5

# Check if process is running
if ps -p $PID > /dev/null; then
    echo "✅ システムが正常に起動しました (PID: $PID)"
    echo "📝 ログファイル: $LOG_FILE"
else
    echo "❌ システムの起動に失敗しました"
    echo "ログを確認してください:"
    tail -20 "$LOG_FILE"
    exit 1
fi

echo ""
echo "📊 日次レポートのスケジュール:"
echo "================================"
echo "• 送信時刻: 毎日 09:00 AM JST"
echo "• 内容:"
echo "  - 24時間の残高推移"
echo "  - 取引実績と勝率"
echo "  - 生成されたシグナル数"
echo "  - リスク指標（最大DD、最大ポジション）"
echo "  - パフォーマンスコメント"
echo ""
echo "📱 Discord通知:"
echo "• 日次レポート: 毎日 09:00 AM JST"
echo "• 残高更新: 1時間ごと"
echo "• 取引シグナル: リアルタイム"
echo ""
echo "🔍 ログ監視:"
echo "tail -f $LOG_FILE"
echo ""
echo "🛑 システム停止:"
echo "kill $PID"
echo ""
echo "✅ デプロイ完了！"