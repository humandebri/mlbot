# 自動更新実装ガイド

## 現状の問題点

1. **DuckDBは手動更新が必要**
   - Redisからのデータはメモリキャッシュのみ
   - DuckDBへの永続化なし

2. **Botは部分的にリアルタイムデータを使用**
   - 5分ごとにRedisから更新（メモリ内）
   - 基本データはDuckDBの古いデータに依存

## 解決策

### オプション1: 定期的な自動更新スクリプト

```bash
# auto_update_duckdb.pyを使用
python auto_update_duckdb.py
```

**特徴**:
- 1時間ごとにDuckDBを自動更新
- 既存のシステムに影響なし
- 別プロセスで実行

**tmuxで実行**:
```bash
tmux new -s auto_updater
python auto_update_duckdb.py
```

### オプション2: 永続化機能付きFeature Generator

```python
# improved_feature_generator_persistent.pyを使用
from improved_feature_generator_persistent import ImprovedFeatureGeneratorPersistent

# Botで使用
self.feature_generator = ImprovedFeatureGeneratorPersistent(
    enable_persistence=True,
    persistence_interval=1800  # 30分ごと
)
```

**特徴**:
- Redisデータを自動的にDuckDBに保存
- 30分ごとに永続化
- Bot内で完結

### オプション3: 両方の併用（推奨）

1. **短期的対応**: auto_update_duckdb.pyを実行
2. **長期的対応**: persistent feature generatorに移行

## EC2への適用手順

### 1. 自動更新スクリプトの導入
```bash
# ファイル転送
scp -i ~/.ssh/mlbot-key-1749802416.pem auto_update_duckdb.py ubuntu@13.212.91.54:~/mlbot/

# EC2で実行
ssh -i ~/.ssh/mlbot-key-1749802416.pem ubuntu@13.212.91.54
cd /home/ubuntu/mlbot
tmux new -s auto_updater
python3 auto_update_duckdb.py
```

### 2. Persistent Feature Generatorへの移行
```bash
# ファイル転送
scp -i ~/.ssh/mlbot-key-1749802416.pem improved_feature_generator_persistent.py ubuntu@13.212.91.54:~/mlbot/

# Botの更新
ssh -i ~/.ssh/mlbot-key-1749802416.pem ubuntu@13.212.91.54
cd /home/ubuntu/mlbot

# Botファイルを編集
vim simple_improved_bot_with_trading_fixed.py
# import文を変更:
# from improved_feature_generator_enhanced import ImprovedFeatureGeneratorEnhanced as ImprovedFeatureGenerator
# ↓
# from improved_feature_generator_persistent import ImprovedFeatureGeneratorPersistent as ImprovedFeatureGenerator

# Bot再起動
pkill -f simple_improved_bot_with_trading_fixed.py
tmux new -s mlbot_persistent "python3 simple_improved_bot_with_trading_fixed.py"
```

## 期待される効果

1. **データの最新性**: 常に最新のデータでML予測
2. **信頼度向上**: より正確な技術指標計算
3. **運用の自動化**: 手動更新不要
4. **データ永続性**: システム再起動後も最新データ保持

## モニタリング

```bash
# 自動更新ログ確認
tail -f logs/auto_update_duckdb.log

# DuckDB最新データ確認
python check_latest_dates.py

# Bot信頼度確認
tmux attach -t mlbot_persistent
```