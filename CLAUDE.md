# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 基本ルール
- 応答は日本語で行うこと
- 既存のファイルの修正を行う際は新しく別にファイルを作るのではなく可能な限り既存のファイルを更新すること
- ライブラリは最新のものを使用するように気をつけること
- デモデータを可能な限り挿入しないこと
- 更新を行った際はどの様な変更を行なったのか @CLAUDE.md に追記して下さい

## プロジェクト概要

**MLBot** - Bybitの清算フィードデータをリアルタイムで分析し、機械学習による期待値予測に基づいて自動取引を行うシステム。

- リアルタイム清算データ分析（WebSocket経由）
- 44次元の特徴量を使用したONNXモデル（AUC 0.838）
- Discord通知機能
- 動的リスク管理
- EC2での24時間稼働

## よく使うコマンド

### 環境セットアップ
```bash
# Python環境（Poetry非推奨、venv使用）
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ローカル開発
```bash
# Redis起動（必須）
docker-compose up -d redis

# DuckDB履歴データ確認
python check_db_tables.py
python check_latest_dates.py

# 改良版MLボット起動（履歴データ使用）
python working_ml_production_bot_improved.py

# 簡易版ボット（設定依存なし）
python simple_improved_bot_fixed.py
```

### EC2デプロイ・運用
```bash
# SSH接続
ssh -i ~/.ssh/mlbot-key-1749802416.pem ubuntu@13.212.91.54
cd /home/ubuntu/mlbot

# tmuxでボット起動
tmux new -s mlbot
python3 simple_improved_bot_fixed.py

# ログ確認
tail -f logs/improved_fixed.log | grep -E "(pred=|Signal sent)"

# tmuxセッション確認
tmux ls
tmux attach -t mlbot
```

### テスト・デバッグ
```bash
# 特徴量生成テスト
python test_improved_features.py

# モデル出力確認
python check_model_output.py

# Discord通知テスト
python test_discord_webhook.py

# APIクレデンシャル確認
python test_api_credentials.py
```

## アーキテクチャ概要

### 現在の実装状態
- **統合システム**: 単一プロセスで全サービスを管理（Docker廃止）
- **メインエントリー**: `main_dynamic_integration.py`または`simple_improved_bot_fixed.py`
- **EC2運用**: tmuxセッションで直接Python実行

### コアコンポーネント

#### データフロー
```
Bybit WebSocket → Ingestor → Redis Streams → FeatureHub → ML Inference → OrderRouter → Bybit API
                     ↓                           ↓
                  DuckDB                    Discord通知
```

#### 重要モジュール
- **improved_feature_generator.py**: 履歴データから実際の技術指標を計算（近似値排除）
- **src/ml_pipeline/inference_engine.py**: ONNX推論エンジン（dual output対応）
- **src/feature_hub/price_features.py**: 基本価格特徴量生成
- **src/common/bybit_client.py**: Bybit API統合

### モデル情報
- **現行モデル**: `models/v3.1_improved/model.onnx` (44次元入力)
- **特徴量変換**: 156次元→44次元（FeatureAdapter44）
- **正規化**: manual_scaler.json使用

## よくある問題と解決策

### モデル予測が0.0000になる
```python
# ONNXモデルのdual output対応
if len(outputs) > 1 and isinstance(outputs[1], list):
    prob_dict = outputs[1][0]
    prediction = prob_dict.get(1, 0.5)
```

### 履歴データが読み込めない
- DuckDBテーブル名: `all_klines`または`klines_btcusdt`等
- 最新データ: 2025-06-11まで（要更新）

### Discord通知が届かない
- 環境変数確認: `DISCORD_WEBHOOK`
- クールダウン: 5分/シンボル
- 信頼度閾値: 65%以上

### EC2でのエラー
- pydantic設定エラー → `simple_improved_bot_fixed.py`使用
- モデルファイル確認: `/home/ubuntu/mlbot/models/v3.1_improved/`

## 重要ドキュメント
- **SYSTEM_ARCHITECTURE.md**: ファイル間の依存関係
- **REPAIR_PLAN.md**: 段階的修正手順
- **README.md**: プロジェクト概要と取引戦略

## 環境変数
```bash
# .env必須項目
DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
BYBIT_API_KEY=...
BYBIT_API_SECRET=...
BYBIT__TESTNET=false
MODEL__MODEL_PATH=models/v3.1_improved/model.onnx
```

## 変更履歴

### 2025/06/20
- **improved_feature_generator.py作成**
  - DuckDB履歴データから実際の技術指標を計算
  - 近似値とランダム値を完全排除
  - RSI、MACD、ボリンジャーバンド等の正確な計算実装

- **simple_improved_bot_fixed.py作成**
  - pydantic設定依存を排除したシンプル版
  - ONNXモデルのdual output正しく処理
  - EC2で安定動作確認

- **EC2デプロイ完了**
  - 履歴データベース（109MB）転送
  - tmuxセッション（mlbot_improved_fixed）で24時間稼働
  - 予測値正常化（0.0000問題解決）

### 2025/06/19
- **重大な問題発見**
  - モデル予測が常に0.0000を返す問題
  - 原因: ONNXモデルの2つの出力（class labels + probability dict）のうち、class labelsのみ使用
  - 修正: probability dictから確率値を抽出するよう変更

### 2025/06/18
- **緊急修復作業実施（Phase 1完了）**
  - FeatureHub初期化不完全の修正
  - PriceFeatureEngineのlatest_features属性追加
  - 統合テスト全項目PASS確認