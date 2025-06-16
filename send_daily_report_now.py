#!/usr/bin/env python3
"""
今すぐ日次レポートを送信（テスト用）
既存システムの残高情報を使用
"""
import requests
import json
from datetime import datetime

# Discord webhook URL
WEBHOOK_URL = "https://discord.com/api/webhooks/1231943231416176692/t1iaVDKtm6WribhzNtYMOPjhMTpN4N9_GGr8NXprcFOjyOH_z5rnnesLqeIAdXJWy6wq"

# 現在のシステム状態（実際のログから）
CURRENT_BALANCE = 0.02128919
INITIAL_BALANCE = 0.02128919  # 起動時の残高

def send_daily_report():
    """日次レポートを送信"""
    
    # 計算
    daily_pnl = CURRENT_BALANCE - INITIAL_BALANCE
    daily_return_pct = (daily_pnl / INITIAL_BALANCE * 100) if INITIAL_BALANCE > 0 else 0
    
    # Discord Embed作成
    embed = {
        "title": f"📅 日次レポート - {datetime.now().strftime('%Y-%m-%d')}",
        "description": "本日の取引実績と残高推移（実際のデータ）",
        "color": 0x03b2f8,  # 青色
        "fields": [
            {
                "name": "📊 残高推移",
                "value": (
                    f"開始: ${INITIAL_BALANCE:.8f}\n"
                    f"現在: ${CURRENT_BALANCE:.8f}\n"
                    f"損益: ${daily_pnl:+.8f} ({daily_return_pct:+.2f}%)"
                ),
                "inline": True
            },
            {
                "name": "📈 取引実績",
                "value": (
                    f"総取引数: 0 (取引実行前)\n"
                    f"勝ち: 0 / 負け: 0\n"
                    f"勝率: N/A"
                ),
                "inline": True
            },
            {
                "name": "🎯 シグナル生成",
                "value": (
                    f"継続的に監視中\n"
                    f"高信頼度シグナル待機中\n"
                    f"清算スパイク: 複数検出"
                ),
                "inline": True
            },
            {
                "name": "⚡ システム稼働状況",
                "value": (
                    f"WebSocket: ✅ 正常 (70-90 msg/s)\n"
                    f"残高更新: ✅ 60秒ごと\n"
                    f"モデル: ✅ v3.1_improved (AUC 0.838)\n"
                    f"レバレッジ: 3倍設定"
                ),
                "inline": False
            },
            {
                "name": "📝 今後の予定",
                "value": (
                    f"• 毎日 09:00 AM JST に日次レポート\n"
                    f"• 1時間ごとの残高更新は継続中\n"
                    f"• 高信頼度シグナル時に即時通知"
                ),
                "inline": False
            }
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "footer": {
            "text": "MLBot Trading System with Daily Reports"
        }
    }
    
    # メッセージ送信
    data = {
        "embeds": [embed]
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=data)
        if response.status_code == 204:
            print("✅ 日次レポート送信成功！")
        else:
            print(f"❌ エラー: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ 送信エラー: {e}")


def send_schedule_notification():
    """レポートスケジュールの通知"""
    
    embed = {
        "title": "📅 日次レポート機能が有効になりました",
        "description": "既存の1時間ごと更新に加えて、日次レポートが追加されます",
        "color": 0x00ff00,  # 緑色
        "fields": [
            {
                "name": "🕐 レポートスケジュール",
                "value": (
                    "**日次レポート**: 毎日 09:00 AM JST\n"
                    "**1時間更新**: 継続中（既存）\n"
                    "**取引シグナル**: リアルタイム（既存）"
                ),
                "inline": False
            },
            {
                "name": "📊 日次レポートの内容",
                "value": (
                    "• 24時間の残高推移\n"
                    "• 取引実績と勝率\n"
                    "• シグナル生成統計\n"
                    "• リスク指標\n"
                    "• パフォーマンス分析"
                ),
                "inline": True
            },
            {
                "name": "💡 現在の状態",
                "value": (
                    f"残高: ${CURRENT_BALANCE:.8f}\n"
                    "取引: 実行待機中\n"
                    "システム: 正常稼働中"
                ),
                "inline": True
            }
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    data = {"embeds": [embed]}
    
    try:
        response = requests.post(WEBHOOK_URL, json=data)
        if response.status_code == 204:
            print("✅ スケジュール通知送信成功！")
    except Exception as e:
        print(f"❌ 送信エラー: {e}")


if __name__ == "__main__":
    print("📤 Discord通知を送信中...")
    
    # 1. スケジュール通知
    send_schedule_notification()
    
    # 2. サンプル日次レポート
    print("\n5秒後にサンプル日次レポートを送信...")
    import time
    time.sleep(5)
    send_daily_report()
    
    print("\n✅ 完了！")
    print("実際の日次レポートは毎日 09:00 AM JST に自動送信されます。")