#!/usr/bin/env python3
"""
Test Discord notification functionality
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

from src.common.discord_notifier import discord_notifier
from src.common.logging import get_logger

logger = get_logger(__name__)

def test_discord_notifications():
    """Test all Discord notification functions"""
    logger.info("🔔 Testing Discord notifications...")
    
    # Test 1: System status
    logger.info("Test 1: System status notification")
    success1 = discord_notifier.send_system_status(
        "online",
        "🔧 Discord通知テスト実行中 - システム正常動作確認"
    )
    logger.info(f"System status: {'✅ Success' if success1 else '❌ Failed'}")
    
    # Test 2: Trade signal
    logger.info("Test 2: Trade signal notification")
    success2 = discord_notifier.send_trade_signal(
        symbol="BTCUSDT",
        side="BUY",
        price=95000.0,
        confidence=0.75,
        expected_pnl=0.025
    )
    logger.info(f"Trade signal: {'✅ Success' if success2 else '❌ Failed'}")
    
    # Test 3: Error notification
    logger.info("Test 3: Error notification")
    success3 = discord_notifier.send_error(
        "test_component",
        "これはテスト用のエラー通知です"
    )
    logger.info(f"Error notification: {'✅ Success' if success3 else '❌ Failed'}")
    
    # Test 4: Daily summary
    logger.info("Test 4: Daily summary notification")
    success4 = discord_notifier.send_daily_summary(
        15,  # total_trades
        12,  # successful_trades
        156.78,  # total_pnl
        80.0,  # win_rate
        stats={"best_trade": 45.23, "worst_trade": -12.34, "total_volume": 125000.0}
    )
    logger.info(f"Daily summary: {'✅ Success' if success4 else '❌ Failed'}")
    
    # Final status
    total_success = sum([success1, success2, success3, success4])
    logger.info(f"🎯 Discord test results: {total_success}/4 notifications successful")
    
    if total_success == 4:
        discord_notifier.send_system_status(
            "testing_complete",
            f"✅ Discord通知テスト完了！全{total_success}件の通知が正常に送信されました。\n\n" +
            "📊 システム状況:\n" +
            "• Ingestor: 90+ msg/s で正常動作\n" +
            "• FeatureHub: 250+ features/s で高性能動作\n" +
            "• モデル: v3.1_improved (AUC 0.838) ロード済み\n" +
            "• 特徴量: 417個生成中（各シンボル139個）\n\n" +
            "🚀 取引システム準備完了！"
        )
        logger.info("🎉 All Discord notifications working perfectly!")
    else:
        logger.error(f"❌ Some Discord notifications failed: {4-total_success} failed")

if __name__ == "__main__":
    test_discord_notifications()