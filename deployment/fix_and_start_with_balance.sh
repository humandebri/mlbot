#!/bin/bash
# 設定エラーを修正して口座残高監視付き取引システムを起動

EC2_USER="ubuntu"
EC2_HOST="13.212.91.54"
EC2_KEY="~/.ssh/mlbot-key-1749802416.pem"

echo "🔧 設定エラーを修正して再起動..."

ssh -i $EC2_KEY $EC2_USER@$EC2_HOST << 'EOF'
cd /home/ubuntu/mlbot

# 1. 修正版取引システムを作成
cat > trading_with_balance_fixed.py << 'PYTHON'
#!/usr/bin/env python3
"""
口座残高監視とケリー基準を統合した取引システム（修正版）
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

# 簡易的な残高監視クラス
class SimpleBalanceMonitor:
    def __init__(self):
        self.current_balance = 100000.0  # 初期残高（仮）
        logger.info("Simple balance monitor initialized")
    
    async def get_balance(self):
        # TODO: 実際のBybit API実装
        return self.current_balance

async def main():
    """統合取引システムのメイン関数"""
    logger.info("🚀 口座残高監視付き取引システム起動（修正版）")
    
    # 簡易残高監視初期化
    balance_monitor = SimpleBalanceMonitor()
    initial_balance = await balance_monitor.get_balance()
    
    # RiskManagerの確認
    risk_manager = RiskManager()
    logger.info(f"💰 初期残高: ${initial_balance:,.2f}")
    logger.info(f"📊 ケリー基準: {risk_manager.config.kelly_fraction}")
    logger.info(f"📊 1取引リスク: {risk_manager.config.risk_per_trade_pct * 100}%")
    logger.info(f"📊 最大レバレッジ: {risk_manager.config.max_leverage}倍")
    
    discord_notifier.send_system_status(
        "trading_with_balance_start",
        f"💰 **口座残高監視付き取引システム起動** 💰\\n\\n" +
        f"📊 **設定確認**:\\n" +
        f"• 初期残高: ${initial_balance:,.2f}（仮）\\n" +
        f"• レバレッジ: {risk_manager.config.max_leverage}倍\\n" +
        f"• ケリー基準: {risk_manager.config.kelly_fraction * 100}%\\n" +
        f"• 1取引リスク: {risk_manager.config.risk_per_trade_pct * 100}%\\n" +
        f"• 最大ポジション/シンボル: ${risk_manager.config.max_position_size_usd:,.0f}\\n" +
        f"• 総エクスポージャー上限: ${risk_manager.config.max_total_exposure_usd:,.0f}\\n\\n" +
        f"⚠️ 注: 実際の残高はBybit APIから取得予定"
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
            
            # 残高チェック（仮）
            current_balance = await balance_monitor.get_balance()
            logger.info(f"💰 現在の残高: ${current_balance:,.2f}")
            
            # ポジションサイズ計算例
            # モデルの信頼度60%、エントリー価格$100、ストップロス$98の場合
            example_size = risk_manager.calculate_position_size(
                symbol="BTCUSDT",
                entry_price=100000.0,
                stop_loss_price=98000.0,  # 2%ストップロス
                confidence=0.60
            )
            logger.info(f"📊 ポジションサイズ例（BTC）: {example_size:.4f} BTC")
                    
    except KeyboardInterrupt:
        logger.info("🛑 システム停止中...")
        await coordinator.stop()
        await feature_hub.stop()
        await ingestor.stop()

if __name__ == "__main__":
    asyncio.run(main())
PYTHON

# 2. 新システムを起動
echo "🚀 修正版取引システムを起動..."
nohup python3 trading_with_balance_fixed.py > trading_balance_fixed.log 2>&1 &

sleep 5

# 3. 起動確認
echo -e "\n✅ 起動確認:"
ps aux | grep python | grep -E "balance_fixed|trading" | grep -v grep

# 4. ログ確認
echo -e "\n📄 システムログ:"
tail -30 trading_balance_fixed.log

# 5. ケリー基準の計算例を表示
echo -e "\n📊 ケリー基準の動作説明:"
echo "=========================="
echo "• ポジションサイズ = (資本 × リスク%) / (エントリー価格 - ストップロス価格)"
echo "• ケリー調整 = 基本サイズ × 信頼度係数 × 0.25（フラクショナルケリー）"
echo "• 例: $100,000資本、1%リスク、2%ストップロスの場合"
echo "  基本サイズ = $1,000 / $2,000 = 0.5 BTC"
echo "  60%信頼度でケリー調整後 = 0.5 × 0.6 × 0.25 = 0.075 BTC"
echo "=========================="

echo -e "\n✅ システム起動完了！"
EOF

echo ""
echo "🎉 修正版の取引システムが起動されました！"
echo ""
echo "⚠️ 注意: 現在は簡易版の残高監視です。"
echo "実際のBybit API統合は別途実装が必要です。"
echo ""
echo "📈 ログ監視:"
echo "ssh -i $EC2_KEY $EC2_USER@$EC2_HOST 'tail -f /home/ubuntu/mlbot/trading_balance_fixed.log'"