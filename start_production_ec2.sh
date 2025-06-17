#!/bin/bash

# EC2でProduction Trading Systemを起動するスクリプト

echo "🚀 Production Trading System起動中..."

# 仮想環境をアクティベート
source venv/bin/activate || source .venv/bin/activate

# Pythonバージョン確認
echo "Python version: $(python --version)"

# 既存のプロセスを確認
echo "既存のプロセスを確認中..."
pgrep -f production_trading_system.py && echo "⚠️ 既にシステムが実行中です" && exit 1

# システムを起動
echo "システムを起動します..."
python production_trading_system.py 2>&1 | tee -a logs/production_$(date +%Y%m%d_%H%M%S).log