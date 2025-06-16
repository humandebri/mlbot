#!/bin/bash
# クリーンな3倍レバレッジ取引システムの起動

EC2_USER="ubuntu"
EC2_HOST="13.212.91.54"
EC2_KEY="~/.ssh/mlbot-key-1749802416.pem"

echo "🧹 クリーンな取引システムを起動（レバレッジ3倍）..."

ssh -i $EC2_KEY $EC2_USER@$EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. 全ての既存プロセスを停止
echo "🛑 全ての既存プロセスを停止..."
pkill -f "python.*main_" || true
pkill -f "python.*mlbot" || true
pkill -f "python.*trading" || true
sleep 3

# 2. シグナル生成のみのシステムも念のため停止
pkill -f "run_production_normalized" || true

# 3. 実際に取引を実行するシステムを作成
cat > start_real_trading.py << 'PYTHON'
#!/usr/bin/env python3
"""
実取引実行システム（レバレッジ3倍、全機能統合）
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import asyncio
from src.integration.trading_coordinator import TradingCoordinator
from src.feature_hub.main import FeatureHub  
from src.ingestor.main import BybitIngestor
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier

logger = get_logger(__name__)

async def main():
    """統合取引システムのメイン関数"""
    logger.info("🚀 実取引システム起動（レバレッジ3倍）")
    
    # Discord通知
    discord_notifier.send_system_status(
        "real_trading_start",
        "🚀 **実取引システム起動** 🚀\\n\\n" +
        "✅ レバレッジ: 3倍\\n" +
        "✅ ポジション管理: 有効\\n" +
        "✅ 自動取引実行: 有効\\n" +
        "✅ リスク管理: 有効\\n\\n" +
        "📊 設定:\\n" +
        "- 最大ポジション/シンボル: $100,000\\n" +
        "- 総エクスポージャー: $500,000\\n" +
        "- ストップロス: 2%\\n" +
        "- 最大同時ポジション: 10"
    )
    
    # コンポーネント初期化
    ingestor = BybitIngestor()
    feature_hub = FeatureHub()
    coordinator = TradingCoordinator()
    
    # 起動
    await ingestor.start()
    await feature_hub.start()
    await coordinator.start()
    
    logger.info("✅ 全コンポーネント起動完了")
    
    # 実行継続
    try:
        while True:
            await asyncio.sleep(60)
            # ステータスレポート
            metrics = coordinator.get_metrics()
            logger.info(f"📊 システムメトリクス: {metrics}")
    except KeyboardInterrupt:
        logger.info("🛑 システム停止中...")
        await coordinator.stop()
        await feature_hub.stop()
        await ingestor.stop()

if __name__ == "__main__":
    asyncio.run(main())
PYTHON

# 4. システムを起動
echo "🚀 実取引システムを起動中..."
nohup python3 start_real_trading.py > real_trading.log 2>&1 &

sleep 5

# 5. 起動確認
echo -e "\n✅ 起動確認:"
ps aux | grep python | grep -E "real_trading|trading" | grep -v grep

# 6. ログ確認
echo -e "\n📄 システムログ:"
tail -20 real_trading.log || echo "起動中..."

echo -e "\n✅ 実取引システム起動完了！"
echo "📊 レバレッジ: 3倍"
echo "🔧 ポジション管理: 有効"
echo "💰 実際の取引: 実行中"
EOF

echo ""
echo "🎉 クリーンな取引システムが起動されました！"
echo "📈 ログ監視: ssh -i $EC2_KEY $EC2_USER@$EC2_HOST 'tail -f /home/ubuntu/mlbot/real_trading.log'"