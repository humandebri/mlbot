# 信頼度50%以上達成プラン

## 調査結果サマリー

### 現状の問題
1. **データギャップ**: DuckDBは2025-06-19 08:31までのデータしかない（6月19-21日分が欠落）
2. **信頼度の低さ**: 現在43-49%（50%閾値未満）
3. **履歴期間**: デフォルト60日のlookbackでは不十分

### 利用可能なリソース
- **Redis**: 10,006件のリアルタイムエントリ
- **古いテーブル**: klines_btcusdt等に2021年からの240万件のデータ
- **all_klinesテーブル**: 13,381件（統合が必要）

## 解決策

### 1. DuckDB更新スクリプト実行
```bash
# データベースを最新化
python update_duckdb_enhanced.py --lookback-hours 72
```

このスクリプトは：
- Redisから最新データを取得してギャップを埋める
- 古い履歴テーブル（klines_btcusdt等）を統合
- all_klinesテーブルに全データを集約

### 2. Lookback期間の延長
```bash
# 60日から120日に延長
python extend_lookback_period.py --days 120
```

これにより：
- より多くの履歴データを使用して特徴量を計算
- 技術指標（RSI、MACD等）の精度向上
- モデルの予測信頼度が向上

### 3. EC2への適用
```bash
# 更新されたデータベースをEC2に転送
scp -i ~/.ssh/mlbot-key-1749802416.pem data/historical_data.duckdb ubuntu@13.212.91.54:~/mlbot/data/

# 更新されたfeature generatorを転送
scp -i ~/.ssh/mlbot-key-1749802416.pem improved_feature_generator_enhanced.py ubuntu@13.212.91.54:~/mlbot/

# EC2でボット再起動
ssh -i ~/.ssh/mlbot-key-1749802416.pem ubuntu@13.212.91.54
cd /home/ubuntu/mlbot
pkill -f simple_improved_bot_with_trading_fixed.py
tmux new-session -d -s mlbot_high_conf "python3 simple_improved_bot_with_trading_fixed.py"
```

## 期待される結果

1. **信頼度向上**: 43-49% → 50-65%
2. **シグナル生成**: 50%閾値を超えてBUY/SELLシグナルが送信される
3. **データ継続性**: Redisからの自動更新で最新性を維持

## 実行順序

1. まず`update_duckdb_enhanced.py`を実行してデータベースを更新
2. 次に`extend_lookback_period.py`でlookback期間を延長
3. 最後にEC2に適用してボットを再起動

## モニタリング

```bash
# EC2で信頼度を確認
tmux attach -t mlbot_high_conf
# または
tail -f logs/mlbot_enhanced_*.log | grep -E "conf=[0-9]+\.[0-9]+%"
```

信頼度が50%を超えることを確認してください。