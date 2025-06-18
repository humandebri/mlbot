# 🚨 MLBot緊急修正計画書

作成日: 2025/06/17
作成者: Claude

## 📊 現状分析

### システムの状態
- **見かけ**: 正常動作（ログが流れている）
- **実際**: 完全に機能停止（取引実行不可能）
- **原因**: 複数の基本的な実装ミス

### 問題の深刻度評価

| レベル | 問題数 | 影響 |
|--------|--------|------|
| 🔴 致命的 | 4件 | システム機能停止 |
| 🟡 重大 | 1件 | 部分機能不全 |
| 🟢 軽微 | 2件 | 最適化必要 |

## 🎯 修正計画

### Phase 0: 緊急停止（即座実行）
```bash
# EC2で現在動作中のシステムを停止
ssh -i ~/.ssh/mlbot-key-*.pem ubuntu@13.212.91.54
tmux kill-session -t trading_production
```

### Phase 1: 基本機能の修復（1-2時間）

#### 1.1 ローカルでの修正作業
- [ ] `simple_service_manager_fixed.py`をsrc/integration/にコピー
- [ ] `price_features_fixed.py`をsrc/feature_hub/にコピー  
- [ ] `bybit_client_positions_fix.py`の内容をbybit_client.pyに統合
- [ ] main_complete_working_patched.pyの最終確認

#### 1.2 統合テスト（ローカル）
```bash
# Redisが起動していることを確認
docker-compose up -d redis

# テスト実行
python main_complete_working_patched.py
```

確認項目：
- [ ] FeatureHub初期化完了メッセージ
- [ ] 特徴量カウント > 0
- [ ] ML予測実行ログ
- [ ] エラーなし（最低5分間）

### Phase 2: EC2デプロイ（30分）

#### 2.1 ファイル転送
```bash
# 修正済みファイルをEC2に転送
scp -i ~/.ssh/mlbot-key-*.pem \
  main_complete_working_patched.py \
  simple_service_manager_fixed.py \
  price_features_fixed.py \
  ubuntu@13.212.91.54:/home/ubuntu/mlbot/
```

#### 2.2 EC2での適用
```bash
# バックアップ作成
cd /home/ubuntu/mlbot
mkdir -p backup/20250617
cp -r src/ backup/20250617/

# 修正ファイル適用
cp simple_service_manager_fixed.py src/integration/simple_service_manager.py
cp price_features_fixed.py src/feature_hub/price_features.py

# bybit_client.pyの手動修正（positions fix適用）
```

#### 2.3 システム再起動
```bash
# tmuxで起動
tmux new -s trading
python main_complete_working_patched.py

# デタッチ: Ctrl+B → D
```

### Phase 3: 動作確認（1時間）

#### 3.1 ログ監視
```bash
# 別セッションで監視
tmux attach -t trading
```

確認チェックリスト：
- [ ] データ受信: "Ingestor performance stats" > 50 msg/s
- [ ] 特徴量生成: "Feature counts" > 0（全シンボル）
- [ ] ML予測: "Prediction for BTCUSDT" ログ出現
- [ ] Discord通知: "Trading Signal Generated" 受信

#### 3.2 問題があった場合
1. ログのエラーメッセージを確認
2. SYSTEM_ARCHITECTURE.mdを参照して問題箇所を特定
3. 該当ファイルを修正
4. Phase 2から再実行

### Phase 4: 最適化（1週間以内）

#### 4.1 パフォーマンス改善
- [ ] モデル再訓練（AUC向上）
- [ ] 特徴量エンジニアリング改善
- [ ] レイテンシ最適化

#### 4.2 監視強化
- [ ] Grafanaダッシュボード構築
- [ ] アラート設定
- [ ] 自動復旧機能

#### 4.3 コード品質改善
- [ ] ユニットテスト追加
- [ ] 統合テスト自動化
- [ ] CI/CDパイプライン

## 📋 修正後の確認項目

### 必須確認（Phase 3で実施）
- [ ] 30分以上の安定稼働
- [ ] 最低1つのDiscordシグナル生成
- [ ] エラーログなし

### 成功の定義
- [ ] 24時間で最低10回の取引実行
- [ ] システムダウンタイムなし
- [ ] 予期しないエラーなし

## ⚠️ 注意事項

1. **修正作業は必ずバックアップを取ってから実施**
2. **SYSTEM_ARCHITECTURE.mdを常に参照・更新**
3. **問題発生時は即座にCLAUDE.mdに記録**

## 🔄 フォローアップ

- 毎日: ログ確認とDiscord通知チェック
- 週次: パフォーマンス分析とモデル評価
- 月次: システム全体のレビューと改善計画

---

この計画に従って、システムを**完全に機能する状態**に修復します。