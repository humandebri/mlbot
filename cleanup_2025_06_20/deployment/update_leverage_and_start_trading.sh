#!/bin/bash
# レバレッジ3倍設定と実取引システムの起動

EC2_USER="ubuntu"
EC2_HOST="13.212.91.54"
EC2_KEY="~/.ssh/mlbot-key-1749802416.pem"
PROJECT_DIR="/home/ubuntu/mlbot"

echo "🔧 レバレッジ3倍設定と実取引システムの更新..."

ssh -i $EC2_KEY $EC2_USER@$EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. RiskManagerのレバレッジを3倍に更新
echo "📝 RiskManagerのレバレッジ設定を更新中..."
sed -i 's/max_leverage: float = 10.0/max_leverage: float = 3.0/' src/order_router/risk_manager.py

# 設定確認
echo "✅ 更新後の設定:"
grep "max_leverage" src/order_router/risk_manager.py

# 2. 既存のシグナル生成のみのプロセスを停止
echo "🛑 シグナル生成のみのプロセスを停止中..."
pkill -f "run_production_normalized.py" || true

# 3. 取引実行機能付き統合システムの起動準備
cat > start_trading_system.py << 'PYTHON'
#!/usr/bin/env python3
"""
レバレッジ3倍、実取引実行機能付き統合システム
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import asyncio
from src.integration.main_unified import main

if __name__ == "__main__":
    print("🚀 実取引実行システム起動（レバレッジ3倍）")
    print("✅ RiskManager統合")
    print("✅ OrderRouter統合")
    print("✅ ポジション管理機能")
    print("✅ 自動取引実行")
    asyncio.run(main())
PYTHON

# 4. 統合システムの設定確認
echo "📋 現在のリスク管理設定:"
echo "========================"
grep -A 20 "class RiskConfig" src/order_router/risk_manager.py | head -25

# 5. 本番取引システムを起動
echo "🚀 実取引実行システムを起動中..."
nohup python3 start_trading_system.py > trading_system.log 2>&1 &

sleep 5

# 6. プロセス確認
echo "📊 実行中のプロセス:"
ps aux | grep python | grep -E "(trading_system|main_unified)" | grep -v grep

# 7. ログ確認
echo "📄 システムログ（最初の20行）:"
tail -n 20 trading_system.log

echo "✅ レバレッジ3倍設定完了、実取引システム起動完了"
EOF

echo "🎉 設定完了！"
echo ""
echo "📊 ポジション管理機能:"
echo "- 最大レバレッジ: 3倍"
echo "- シンボルごとの最大ポジション: $100,000"
echo "- 総エクスポージャー上限: $500,000"
echo "- 最大同時ポジション数: 10"
echo "- リスク管理: RiskManager統合"
echo "- 取引実行: OrderRouter統合"
echo ""
echo "📈 監視方法:"
echo "ssh -i $EC2_KEY $EC2_USER@$EC2_HOST 'tail -f /home/ubuntu/mlbot/trading_system.log'"