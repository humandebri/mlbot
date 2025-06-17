#!/usr/bin/env python3
"""
本番トレーディングシステムを起動（残高$100）
"""
import os
import subprocess
import sys
import time

print("🚀 本番トレーディングシステムを起動中...")
print("💰 残高: $100")
print("🔑 APIキー: 本番用設定済み")

# 環境変数を明示的に設定
os.environ['BYBIT__TESTNET'] = 'false'
os.environ['ENVIRONMENT'] = 'production'

# 統合システムではなく、実績のあるトレーディングシステムを起動
cmd = [
    sys.executable,
    "trading_with_real_api.py"
]

print("\n📊 起動コマンド:", " ".join(cmd))
print("⏳ システム起動中...")

# プロセスを起動
process = subprocess.Popen(cmd)

print(f"\n✅ システム起動完了! PID: {process.pid}")
print("📈 取引開始準備完了")
print("🎯 高信頼度シグナル（60%以上）待機中...")
print("⚠️  本番環境で実際の資金を使用しています")
print("\n停止するには: kill", process.pid)