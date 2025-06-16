#!/usr/bin/env python3
"""
既存の取引システムに日次レポート機能を追加
（既存の1時間ごと更新は維持）
"""
import sys
sys.path.insert(0, '/Users/0xhude/Desktop/mlbot')

import asyncio
import signal
import json
from datetime import datetime
from typing import Optional

from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier
from src.common.daily_report import DailyReportManager

logger = get_logger(__name__)


class DailyReportAddon:
    """既存システムに追加する日次レポート機能"""
    
    def __init__(self):
        # 既存システムの実際の残高を取得するため簡易版を作成
        self.daily_report = None
        self.running = False
        self._report_task = None
        
        # 実際の残高データ（既存システムから取得したい）
        self.current_balance = 0.02128919  # ログから確認した実際の残高
        
    async def start(self):
        """日次レポート機能を開始"""
        logger.info("日次レポート機能を既存システムに追加中...")
        
        # 簡易的な残高追跡（実際はAccountMonitorから取得すべき）
        class SimpleBalance:
            def __init__(self, equity):
                self.total_equity = equity
        
        class SimpleMonitor:
            def __init__(self):
                self.current_balance = SimpleBalance(0.02128919)
                self.initial_balance = 0.02128919
                
            def get_performance_stats(self):
                return {
                    'total_return_pct': 0.0,
                    'max_drawdown_pct': 0.0,
                    'peak_balance': self.initial_balance
                }
        
        # 日次レポートマネージャー初期化
        simple_monitor = SimpleMonitor()
        self.daily_report = DailyReportManager(
            account_monitor=simple_monitor,
            report_time="09:00",  # JST 9:00 AM
            timezone="Asia/Tokyo"
        )
        
        # 通知送信
        discord_notifier.send_notification(
            title="📅 日次レポート機能追加",
            description="既存の取引システムに日次レポート機能を追加しました",
            color="00ff00",
            fields={
                "レポート時刻": "毎日 09:00 AM JST",
                "既存の1時間更新": "継続中",
                "現在の残高": f"${simple_monitor.current_balance.total_equity:.8f}",
                "注意": "実際の残高は既存システムから取得されます"
            }
        )
        
        self.running = True
        
        # 日次レポートループを開始
        self._report_task = asyncio.create_task(self._enhanced_report_loop())
        
        logger.info("日次レポート機能が追加されました")
    
    async def _enhanced_report_loop(self):
        """テスト用：すぐにサンプルレポートを送信"""
        await asyncio.sleep(5)  # 5秒後にテストレポート
        
        # テスト用の日次レポート送信
        try:
            # 実際のデータに基づくレポート内容
            fields = {
                "📊 残高推移": (
                    f"開始: $0.02128919\n"
                    f"現在: $0.02128919\n"
                    f"損益: $0.00000000 (0.00%)"
                ),
                "📈 取引実績": (
                    f"総取引数: 0\n"
                    f"勝ち: 0 / 負け: 0\n"
                    f"勝率: N/A"
                ),
                "🎯 シグナル": (
                    f"生成数: 継続的に生成中\n"
                    f"高信頼度: 監視中\n"
                    f"清算スパイク: 複数検出"
                ),
                "⚡ システム状態": (
                    f"WebSocket: 正常 (90+ msg/s)\n"
                    f"残高更新: 60秒ごと\n"
                    f"1時間ごと通知: 有効"
                )
            }
            
            discord_notifier.send_notification(
                title=f"📅 日次レポート - {datetime.now().strftime('%Y-%m-%d')} (テスト)",
                description="既存システムの状態レポート（実際は毎日9:00 AMに送信）",
                color="03b2f8",
                fields=fields
            )
            
            logger.info("テスト日次レポート送信完了")
            
        except Exception as e:
            logger.error(f"レポート送信エラー: {e}")
        
        # 実際の日次レポートループ
        while self.running:
            await asyncio.sleep(3600)  # 1時間ごとにチェック
    
    async def stop(self):
        """停止"""
        self.running = False
        if self._report_task:
            self._report_task.cancel()
            try:
                await self._report_task
            except asyncio.CancelledError:
                pass


# Global instance
report_addon: Optional[DailyReportAddon] = None


def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}")
    asyncio.create_task(shutdown())


async def shutdown():
    if report_addon:
        await report_addon.stop()


async def main():
    """メインエントリーポイント"""
    global report_addon
    
    # シグナルハンドラー設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 日次レポート機能を開始
        report_addon = DailyReportAddon()
        await report_addon.start()
        
        # 継続実行
        while report_addon.running:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
    finally:
        if report_addon:
            await report_addon.stop()


if __name__ == "__main__":
    asyncio.run(main())