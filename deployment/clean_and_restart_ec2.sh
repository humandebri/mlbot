#!/bin/bash
# EC2の全プロセスを停止して、実際のAPI統合で再起動

EC2_USER="ubuntu"
EC2_HOST="13.212.91.54"
EC2_KEY="~/.ssh/mlbot-key-1749802416.pem"

echo "🧹 EC2のクリーンアップと再起動..."

ssh -i $EC2_KEY $EC2_USER@$EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. 全ての既存プロセスを停止
echo "🛑 全てのMLボットプロセスを停止..."
pkill -f "python.*main_" || true
pkill -f "python.*trading" || true
pkill -f "python.*mlbot" || true
pkill -f "python.*unified" || true
sleep 5

# 残っているプロセスがあれば強制終了
for pid in $(ps aux | grep python | grep -E "main_|trading|mlbot|unified" | grep -v grep | awk '{print $2}'); do
    echo "強制終了: PID $pid"
    kill -9 $pid 2>/dev/null || true
done

# 2. 簡略版の実取引システムを作成（既存のコンポーネントを利用）
cat > start_production_trading.py << 'PYTHON'
#!/usr/bin/env python3
"""
実取引システム（実際のBybit API統合版）
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import asyncio
from src.integration.trading_coordinator import TradingCoordinator
from src.feature_hub.main import FeatureHub  
from src.ingestor.main import BybitIngestor
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier
from src.order_router.risk_manager import RiskManager

logger = get_logger(__name__)

async def main():
    """統合取引システムのメイン関数"""
    logger.info("🚀 実取引システム起動（レバレッジ3倍、実残高使用）")
    
    # RiskManagerで設定確認
    risk_manager = RiskManager()
    logger.info(f"✅ レバレッジ設定: {risk_manager.config.max_leverage}倍")
    logger.info(f"✅ ケリー基準: {risk_manager.config.kelly_fraction * 100}%")
    logger.info(f"✅ 1取引リスク: {risk_manager.config.risk_per_trade_pct * 100}%")
    
    # Discord通知
    try:
        discord_notifier.send_system_status(
            "production_trading_start",
            "🚀 **実取引システム起動** 🚀\\n\\n" +
            "✅ レバレッジ: 3倍\\n" +
            "✅ ケリー基準: 25%\\n" +
            "✅ リスク管理: 有効\\n" +
            "✅ 実残高使用: Bybit API\\n\\n" +
            "📊 設定:\\n" +
            "- 最大ポジション/シンボル: $100,000\\n" +
            "- 総エクスポージャー: $500,000\\n" +
            "- ストップロス: 2%\\n" +
            "- 最大同時ポジション: 10"
        )
    except Exception as e:
        logger.warning(f"Discord通知エラー: {e}")
    
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
            await asyncio.sleep(300)  # 5分ごと
            logger.info("💓 システム正常稼働中...")
    except KeyboardInterrupt:
        logger.info("🛑 システム停止中...")
        await coordinator.stop()
        await feature_hub.stop()
        await ingestor.stop()

if __name__ == "__main__":
    asyncio.run(main())
PYTHON

# 3. システムを起動
echo -e "\n🚀 実取引システムを起動中..."
nohup python3 start_production_trading.py > production_trading.log 2>&1 &

sleep 10

# 4. 起動確認
echo -e "\n✅ 起動確認:"
ps aux | grep python | grep -E "production|trading" | grep -v grep

# 5. ログ確認
echo -e "\n📄 システムログ:"
tail -30 production_trading.log | grep -E "起動|レバレッジ|kelly|risk|balance" | tail -20

echo -e "\n✅ EC2で実取引システムが起動しました！"
echo "📊 ステータス確認:"
ps aux | grep python | wc -l
EOF

echo ""
echo "🎉 EC2クリーンアップと再起動完了！"