#!/usr/bin/env python3
"""
Test Discord notification functionality.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.discord_trading_bot import DiscordTradingBot


def test_discord_webhook():
    """Test Discord webhook functionality."""
    
    # Check for webhook URL
    webhook_url = os.getenv('DISCORD_WEBHOOK')
    if not webhook_url:
        print("❌ DISCORD_WEBHOOK環境変数が設定されていません")
        print("\n📝 設定方法:")
        print("1. .envファイルを開く")
        print("2. 最後の行のコメントを外して、WebhookのURLを設定:")
        print("   DISCORD_WEBHOOK=https://discord.com/api/webhooks/...")
        print("\nDiscord Webhookの取得方法:")
        print("1. Discordサーバーの設定を開く")
        print("2. 「連携サービス」→「ウェブフック」を選択")
        print("3. 「新しいウェブフック」を作成")
        print("4. URLをコピーして.envファイルに貼り付け")
        return
    
    print("✅ DISCORD_WEBHOOK設定を検出しました")
    print("📡 テストメッセージを送信中...")
    
    try:
        # Create bot instance
        bot = DiscordTradingBot()
        
        # Send test message
        success = bot.send_discord_message(
            title="🎉 MLBot接続テスト成功！",
            description="Discord通知が正常に動作しています",
            color=0x00ff00,
            fields=[
                {
                    'name': '📊 システム情報',
                    'value': 'MLBot Trading System v1.0',
                    'inline': True
                },
                {
                    'name': '💰 戦略',
                    'value': 'レバレッジ3倍\n月次収益目標: 4.16%',
                    'inline': True
                },
                {
                    'name': '🤖 機能',
                    'value': '• デイリーレポート\n• 取引アラート\n• エラー通知',
                    'inline': False
                }
            ]
        )
        
        if success:
            print("✅ テストメッセージの送信に成功しました！")
            print("📱 Discordチャンネルを確認してください")
            
            # Try to load model
            print("\n🔧 モデルのロードテスト...")
            if bot.load_model():
                print("✅ モデルのロードに成功しました")
                
                # Send sample daily report
                print("\n📊 サンプルレポートを送信中...")
                bot.send_daily_report()
                print("✅ サンプルレポートを送信しました")
            else:
                print("❌ モデルのロードに失敗しました")
                print("   （モデルファイルが存在しない可能性があります）")
        else:
            print("❌ テストメッセージの送信に失敗しました")
            print("   Webhook URLが正しいか確認してください")
            
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")


if __name__ == "__main__":
    print("="*60)
    print("🔧 Discord通知テスト")
    print("="*60)
    test_discord_webhook()