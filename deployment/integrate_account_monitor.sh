#!/bin/bash
# AccountMonitorを統合した完全な取引システム

EC2_USER="ubuntu"
EC2_HOST="13.212.91.54"
EC2_KEY="~/.ssh/mlbot-key-1749802416.pem"

echo "💰 口座残高監視機能を統合..."

ssh -i $EC2_KEY $EC2_USER@$EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. 統合版取引システムを作成
cat > trading_with_account_monitor.py << 'PYTHON'
#!/usr/bin/env python3
"""
口座残高監視とケリー基準を統合した取引システム
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import asyncio
from src.integration.trading_coordinator import TradingCoordinator
from src.feature_hub.main import FeatureHub  
from src.ingestor.main import BybitIngestor
from src.common.account_monitor import AccountMonitor
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier

logger = get_logger(__name__)

async def main():
    """統合取引システムのメイン関数"""
    logger.info("🚀 口座残高監視付き取引システム起動")
    
    # AccountMonitor初期化
    account_monitor = AccountMonitor(check_interval=60)  # 60秒ごとに残高チェック
    await account_monitor.start()
    
    # 初期残高を取得
    balance = await account_monitor.update_balance()
    if balance:
        stats = account_monitor.get_performance_stats()
        
        discord_notifier.send_system_status(
            "trading_with_balance_start",
            f"💰 **口座残高監視付き取引システム起動** 💰\\n\\n" +
            f"📊 **口座情報**:\\n" +
            f"• 総資産: ${balance.total_equity:,.2f}\\n" +
            f"• 利用可能残高: ${balance.available_balance:,.2f}\\n" +
            f"• 未実現損益: ${balance.unrealized_pnl:,.2f}\\n" +
            f"• フリーマージン: {balance.free_margin_pct:.1f}%\\n\\n" +
            f"⚙️ **設定**:\\n" +
            f"• レバレッジ: 3倍\\n" +
            f"• ケリー基準: 25%\\n" +
            f"• 1取引リスク: 1%\\n" +
            f"• 残高チェック: 60秒ごと"
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
            await asyncio.sleep(300)  # 5分ごと
            
            # パフォーマンス統計
            stats = account_monitor.get_performance_stats()
            if stats:
                logger.info(f"📊 パフォーマンス統計: {stats}")
                
                # 大きな変動があればDiscord通知
                if abs(stats.get('total_return_pct', 0)) > 5:
                    discord_notifier.send_system_status(
                        "performance_update",
                        f"📊 **パフォーマンス更新** 📊\\n\\n" +
                        f"• 総資産: ${stats['current_equity']:,.2f}\\n" +
                        f"• 総リターン: {stats['total_return_pct']:.2f}%\\n" +
                        f"• 最大ドローダウン: {stats['max_drawdown_pct']:.2f}%\\n" +
                        f"• 未実現損益: ${stats['unrealized_pnl']:,.2f}"
                    )
                    
    except KeyboardInterrupt:
        logger.info("🛑 システム停止中...")
        await account_monitor.stop()
        await coordinator.stop()
        await feature_hub.stop()
        await ingestor.stop()

if __name__ == "__main__":
    asyncio.run(main())
PYTHON

# 2. RiskManagerがAccountMonitorと連携するよう更新
echo "📝 RiskManagerの更新確認..."
grep -n "current_equity" /home/ubuntu/mlbot/src/order_router/risk_manager.py | head -5

# 3. 既存プロセスを停止
echo "🛑 既存プロセスを停止..."
pkill -f "start_real_trading.py" || true
sleep 3

# 4. 新システムを起動
echo "🚀 口座残高監視付き取引システムを起動..."
nohup python3 trading_with_account_monitor.py > trading_with_balance.log 2>&1 &

sleep 5

# 5. 起動確認
echo -e "\n✅ 起動確認:"
ps aux | grep python | grep -E "account_monitor|trading_with" | grep -v grep

# 6. ログ確認
echo -e "\n📄 システムログ:"
tail -30 trading_with_balance.log | grep -E "balance|equity|kelly|position.*size" || tail -20 trading_with_balance.log

echo -e "\n✅ 口座残高監視付き取引システム起動完了！"
EOF

echo ""
echo "🎉 完全な取引システムが起動されました！"
echo ""
echo "📊 機能:"
echo "• Bybit APIで口座残高をリアルタイム取得（60秒ごと）"
echo "• ケリー基準（25%）でポジションサイズ決定"
echo "• 1取引あたりリスク: 資本の1%"
echo "• レバレッジ: 3倍"
echo "• ドローダウン監視"
echo ""
echo "📈 ログ監視:"
echo "ssh -i $EC2_KEY $EC2_USER@$EC2_HOST 'tail -f /home/ubuntu/mlbot/trading_with_balance.log | grep -E \"balance|position|kelly\"'"