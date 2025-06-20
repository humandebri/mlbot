# ディレクトリ整理サマリー

## 整理日時
2025-06-20

## 整理後の推奨ディレクトリ構造

### ルートディレクトリ
```
mlbot/
├── .env                                    # 環境変数設定
├── requirements.txt                        # 依存関係
├── CLAUDE.md                              # プロジェクトガイド
├── README.md                              # プロジェクト概要
├── main_dynamic_integration.py            # メイン統合スクリプト
├── simple_improved_bot_fixed.py           # シンプル版ボット（EC2で稼働中）
├── simple_improved_bot_with_trading_fixed.py  # 取引機能付きボット
├── improved_feature_generator.py          # 特徴量生成器
├── ml_feature_generator.py                # ML特徴量生成
├── check_db_*.py                          # DBチェックツール（3ファイル）
├── check_model_output.py                  # モデル出力確認
├── simple_discord_test.py                 # Discord通知テスト
```

### 重要ディレクトリ
```
├── src/                    # ソースコード
├── models/                 # モデルファイル
├── logs/                   # ログファイル
├── data/                   # データファイル
├── scripts/                # ユーティリティスクリプト
├── config/                 # 設定ファイル
├── notebooks/              # Jupyterノートブック
```

### 整理済みディレクトリ
```
├── old_files/              # バックアップ（今回作成）
│   ├── test_scripts/       # テストスクリプト（28ファイル）
│   ├── debug_scripts/      # デバッグスクリプト
│   ├── fix_scripts/        # 修正・パッチスクリプト
│   ├── docker_files/       # Docker関連（使用停止）
│   └── unused_bots/        # 使用されていないボット
├── cleanup/                # 以前の整理ファイル
```

## 削除推奨項目

### 1. 仮想環境（オプション）
- `.venv_test` (1.6GB) - テスト用、削除可能
- `.venv_tf` (1.6GB) - TensorFlow用、使用していなければ削除可能
- `.venv` (1.7GB) - メイン環境、保持推奨

### 2. 不要なディレクトリ
- `docker/` - Docker使用停止のため削除可能
- `k8s/` - Kubernetes設定、使用していなければ削除可能
- `deployment/` - 古いデプロイメント設定
- `catboost_info/` - CatBoostの一時ファイル
- `__pycache__/` - Pythonキャッシュ、削除可能

## 保持すべき重要ファイル

### 現在稼働中
- `simple_improved_bot_fixed.py` - EC2で稼働中のメインボット
- `simple_improved_bot_with_trading_fixed.py` - 取引機能付き版

### データベース関連
- `check_db_*.py` - DuckDBデータ確認用
- `improved_feature_generator.py` - 履歴データから特徴量生成

### モデル関連
- `models/v3.1_improved/` - 現行モデル
- `check_model_output.py` - モデル出力テスト

## 推奨アクション

1. **即実行可能**
   - `old_files/`内のファイルを確認後、不要なものは削除
   - `__pycache__`ディレクトリの削除
   - `catboost_info/`の削除

2. **確認後実行**
   - `.venv_test`と`.venv_tf`の削除（使用していない場合）
   - `docker/`と`k8s/`ディレクトリの削除
   - `cleanup/`内の古いファイルの削除

3. **EC2での作業**
   - 同様の整理をEC2インスタンスでも実施
   - ログファイルのローテーション設定
   - 定期的なクリーンアップスクリプトの設定

## 削除コマンド例

```bash
# キャッシュ削除
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# catboost_info削除
rm -rf catboost_info/

# 不要な仮想環境削除（確認後）
rm -rf .venv_test .venv_tf

# Docker関連削除（確認後）
rm -rf docker/ k8s/ deployment/

# 古いログファイル削除（30日以上前）
find logs -name "*.log" -mtime +30 -delete
```