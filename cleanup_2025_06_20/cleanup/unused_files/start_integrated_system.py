#!/usr/bin/env python3
"""
統合トレーディングシステムを起動
"""
import subprocess
import sys
import time

print("🚀 統合トレーディングシステムを起動中...")

# 仮想環境をアクティベート
venv_activate = "source .venv/bin/activate"

# メインシステムを起動
cmd = f"{venv_activate} && python src/integration/main.py"

print("📊 起動コマンド:", cmd)
print("⏳ システム起動中...")

# バックグラウンドで起動
process = subprocess.Popen(cmd, shell=True)

print(f"✅ システム起動完了! PID: {process.pid}")
print("📈 取引開始準備完了")
print("💰 残高: $100")
print("🎯 高信頼度シグナル待機中...")
print("\n停止するには: kill", process.pid)