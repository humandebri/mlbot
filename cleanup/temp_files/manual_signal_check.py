#!/usr/bin/env python3
"""
手動で取引シグナルをチェック
"""
import asyncio
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.common.config import settings
from src.common.discord_notifier import discord_notifier

async def manual_signal():
    """手動でシグナルを送信"""
    
    # 実際の残高
    balance = 99.92
    
    # 仮想的な高信頼度シグナル
    fields = {
        "Symbol": "BTCUSDT",
        "Side": "BUY",
        "Price": "$64,825.50",
        "Confidence": "62.5%",
        "Expected PnL": "+1.2%",
        "Account Balance": f"${balance:.2f}",
        "Position Size": f"${balance * 0.2:.2f}",  # 20% of equity
        "Status": "システム稼働確認用テストシグナル"
    }
    
    discord_notifier.send_notification(
        title="🚨 システム動作確認",
        description="残高$100での取引準備完了",
        color="00ff00",
        fields=fields
    )
    
    print("✅ 通知送信完了！")
    print(f"💰 残高: ${balance}")
    print("⚠️  実際の取引シグナルは市場条件と信頼度60%以上で生成されます")

if __name__ == "__main__":
    asyncio.run(manual_signal())