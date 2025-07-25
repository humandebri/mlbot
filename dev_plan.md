# Bybit Liquidation-Driven Trading Bot 開発計画書

## 1. プロジェクト概要

### 1.1 ビジョン
Bybitの清算フィードデータをリアルタイムで分析し、機械学習による期待値予測に基づいて自動的に限価注文を発注する高頻度取引ボットを開発する。

### 1.2 主要目標
- **収益性**: 手数料控除後の期待収益 E[ret|x,δ,T] - fee > 0 を実現
- **低レイテンシ**: エンドツーエンドの処理時間 < 15ms
- **スケーラビリティ**: 複数通貨ペアの同時処理に対応
- **堅牢性**: 24/7稼働with自動リカバリ機能

### 1.3 成功指標
- バックテストSharpe比 > 2.0
- 実運用月次収益率 > 5%
- システム稼働率 > 99.9%
- 最大ドローダウン < 10%

## 2. 技術スタック選定

### 2.1 コア技術
| コンポーネント | 技術選定 | 理由 |
|---|---|---|
| 言語 | Python 3.11+ | ML生態系の成熟度、asyncio対応 |
| Webフレームワーク | FastAPI | 高性能、自動ドキュメント生成 |
| WebSocket | websockets / aiohttp | 非同期処理、低レイテンシ |
| キューイング | Redis Streams | 低レイテンシ、永続性オプション |
| 時系列DB | DuckDB + Parquet | 高速分析クエリ、効率的圧縮 |
| MLフレームワーク | CatBoost → ONNX | 勾配ブースティングの最高性能、高速推論 |
| コンテナ | Docker + Compose | 環境統一、スケーラビリティ |
| オーケストレーション | Kubernetes (本番) | 自動スケーリング、自己修復 |
| CI/CD | GitHub Actions + ArgoCD | GitOps、自動デプロイ |
| 監視 | Prometheus + Grafana | メトリクス収集、可視化 |
| ログ | ELK Stack | 集中ログ管理、分析 |

### 2.2 Python主要ライブラリ
```toml
[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.109.0"
websockets = "^12.0"
redis = "^5.0.1"
duckdb = "^0.10.0"
pandas = "^2.2.0"
numpy = "^1.26.3"
catboost = "^1.2.2"
onnxruntime = "^1.17.0"
pydantic = "^2.5.3"
httpx = "^0.26.0"
structlog = "^24.1.0"
prometheus-client = "^0.19.0"
```

## 3. システムアーキテクチャ

### 3.1 マイクロサービス構成
```
┌─────────────────────────────────────────────────────────────┐
│                        External Services                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Bybit WS API│  │Bybit REST API│  │External Data API │   │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────────┘   │
└─────────┼─────────────────┼─────────────────┼───────────────┘
          │                 │                 │
┌─────────▼─────────────────▼─────────────────▼───────────────┐
│                     Ingestion Layer                          │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ WS Ingestor │  │REST Collector│  │ Macro Collector │   │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────────┘   │
└─────────┼─────────────────┼─────────────────┼───────────────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │
                    ┌───────▼────────┐
                    │ Redis Streams  │
                    └───────┬────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                    Processing Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Feature Hub  │  │Model Server  │  │  Order Router    │   │
│  └──────┬───────┘  └──────▲───────┘  └──────┬───────────┘   │
└─────────┼──────────────────┼─────────────────┼───────────────┘
          │                  │                 │
    ┌─────▼──────┐    ┌──────┴──────┐   ┌─────▼────────┐
    │  DuckDB    │    │    ONNX     │   │ Bybit Order  │
    │  Storage   │    │   Runtime   │   │     API      │
    └────────────┘    └─────────────┘   └──────────────┘
```

### 3.2 データフロー設計
1. **リアルタイムパス** (< 15ms)
   - WebSocket → Ingestor → Redis → FeatureHub → Model → Order
   
2. **バッチパス** (hourly)
   - Redis → DuckDB → Parquet snapshots
   
3. **再学習パス** (weekly)
   - DuckDB → Feature Engineering → CatBoost → ONNX → Model Server

## 4. 実装フェーズ

### Phase 1: 基盤構築 (2週間)
- [ ] プロジェクト構造とCI/CD設定
- [ ] Docker環境構築
- [ ] Redis/DuckDB接続設定
- [ ] ログ・監視基盤
- [ ] Bybit WebSocket接続モジュール

### Phase 2: データ収集層 (2週間)
- [ ] WebSocket Ingestor実装
  - [ ] kline, orderbook, trades購読
  - [ ] 清算フィード購読
  - [ ] 再接続・エラーハンドリング
- [ ] REST Collector実装
  - [ ] OI/Funding定期取得
  - [ ] レート制限管理
- [ ] データ永続化
  - [ ] Redis Streamへの書き込み
  - [ ] Parquetスナップショット

### Phase 3: 特徴量エンジニアリング (2週間)
- [ ] FeatureHub実装
  - [ ] マイクロ流動性指標
  - [ ] ボラティリティ・モメンタム指標
  - [ ] 清算スパイク指標
  - [ ] 時間コンテキスト特徴
- [ ] リアルタイム計算パイプライン
- [ ] 特徴量キャッシュ戦略

### Phase 4: ML モデリング (3週間)
- [ ] オフラインラベル生成
  - [ ] 複数(δ,T)組み合わせでのラベリング
  - [ ] 手数料・スリッページ考慮
- [ ] CatBoostモデル学習
  - [ ] ハイパーパラメータ最適化
  - [ ] 交差検証戦略
  - [ ] SHAP分析
- [ ] ONNX変換・最適化
- [ ] Model Server実装

### Phase 5: 執行システム (2週間)
- [ ] Order Router実装
  - [ ] 注文サイズ計算
  - [ ] リスク管理ロジック
  - [ ] 注文執行・監視
- [ ] Position Manager
  - [ ] ポジション追跡
  - [ ] PnL計算
  - [ ] 緊急停止機能

### Phase 6: 動的最適化 (2週間)
- [ ] Thompson Samplingモジュール
  - [ ] β分布パラメータ管理
  - [ ] 探索・活用バランス
- [ ] オンライン学習パイプライン
- [ ] A/Bテストフレームワーク

### Phase 7: バックテスト・検証 (2週間)
- [ ] ティックレベルバックテスター
- [ ] 約定シミュレーション
- [ ] パフォーマンス分析ツール
- [ ] リスク指標計算

### Phase 8: 本番準備 (3週間)
- [ ] Testnet統合テスト
- [ ] 本番環境構築
- [ ] 監視・アラート設定
- [ ] 運用手順書作成
- [ ] 段階的ロールアウト計画

## 5. テスト戦略

### 5.1 テストレベル
1. **ユニットテスト** (カバレッジ > 80%)
   - 個別関数・クラスのテスト
   - モック活用による独立性確保

2. **統合テスト**
   - サービス間通信
   - データフロー検証
   - レイテンシ測定

3. **システムテスト**
   - エンドツーエンドシナリオ
   - 負荷テスト
   - 障害回復テスト

4. **受入テスト**
   - Testnet実績評価
   - リスク指標確認

### 5.2 パフォーマンステスト
- WebSocketメッセージ処理: < 1ms
- Redis読み書き: < 1ms
- 特徴量計算: < 5ms
- モデル推論: < 1ms
- 注文送信: < 10ms

## 6. リスク管理

### 6.1 技術リスク
| リスク | 影響 | 対策 |
|---|---|---|
| WebSocket切断 | データロス | 自動再接続、バッファリング |
| レート制限 | 注文失敗 | リクエスト管理、バッチ処理 |
| モデル劣化 | 収益低下 | 定期再学習、A/Bテスト |
| レイテンシ増大 | 機会損失 | プロファイリング、最適化 |

### 6.2 市場リスク
- **最大ポジションサイズ**: 資産の10%
- **最大ドローダウン**: 5%で警告、10%で停止
- **清算価格バッファ**: 20%以上維持
- **相関リスク**: 複数通貨ペアの相関監視

### 6.3 運用リスク
- **サーキットブレーカー**: 異常検知時の自動停止
- **デッドマンスイッチ**: 定期的なハートビート確認
- **バックアップ**: 設定・モデルの定期バックアップ
- **アクセス管理**: 最小権限原則、監査ログ

## 7. セキュリティ設計

### 7.1 API認証
- APIキーの暗号化保存（Hashicorp Vault）
- IP制限・ホワイトリスト
- 読み取り専用キーの活用

### 7.2 内部通信
- サービス間のmTLS
- Redis認証・暗号化
- ネットワークセグメンテーション

### 7.3 監査・コンプライアンス
- 全取引の記録保持
- 異常取引の検知・アラート
- 定期的なセキュリティ監査

## 8. 運用・監視

### 8.1 メトリクス
- **システムメトリクス**: CPU、メモリ、ネットワーク
- **アプリケーションメトリクス**: レイテンシ、スループット
- **ビジネスメトリクス**: PnL、約定率、ポジション

### 8.2 アラート設定
- P1: システム停止、大幅損失
- P2: パフォーマンス劣化、接続エラー
- P3: 定期タスク失敗、リソース警告

### 8.3 ダッシュボード
- リアルタイムPnL
- ポジション状況
- システムヘルス
- 市場状況（清算量、ボラティリティ）

## 9. スケーラビリティ計画

### 9.1 水平スケーリング
- Ingestor: 通貨ペアごとにインスタンス
- FeatureHub: シャーディングによる分散
- Model Server: ロードバランサー経由

### 9.2 垂直スケーリング
- Redis: クラスター構成
- DuckDB: パーティショニング
- ONNX: GPU活用

## 10. プロジェクトタイムライン

```
Week 1-2:   基盤構築
Week 3-4:   データ収集層
Week 5-6:   特徴量エンジニアリング
Week 7-9:   MLモデリング
Week 10-11: 執行システム
Week 12-13: 動的最適化
Week 14-15: バックテスト・検証
Week 16-18: 本番準備・デプロイ
```

総期間: 18週間（約4.5ヶ月）

## 11. 成功の鍵

1. **インクリメンタル開発**: 各フェーズで動作確認
2. **早期検証**: Testnetでの継続的テスト
3. **データ品質**: 欠損・異常値の適切な処理
4. **レイテンシ最適化**: 継続的なプロファイリング
5. **リスク管理**: 保守的なパラメータから開始

## 12. 次のアクション

1. 開発環境のセットアップ
2. Bybit APIドキュメントの詳細確認
3. 初期データ収集スクリプトの作成
4. プロトタイプ実装の開始