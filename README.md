# Bybit Liquidation-Driven Trading Bot

**高頻度取引ボット** - Bybitの清算フィードデータをリアルタイムで分析し、機械学習による期待値予測に基づいて自動的に限価注文を発注します。

## 🎯 プロジェクト概要

このプロジェクトは、Bybitの清算情報を活用した機械学習ベースの自動取引システムです。清算カスケードを予測し、価格の一時的な急落（wick）を狙って利益を上げることを目的としています。

### 主要機能

- **リアルタイム清算データ分析**: 500ms更新の清算フィードを監視
- **機械学習予測**: CatBoost → ONNX による期待値計算（<1ms）
- **動的パラメータ最適化**: Thompson Samplingによるδ（価格オフセット）とT（ルックアヘッド期間）の自動調整
- **マイクロサービス構成**: Docker + Kubernetesでクラウドネイティブ
- **包括的リスク管理**: サーキットブレーカー、ポジション制限、緊急停止機能

## 🏗️ システムアーキテクチャ

```
┌─────────────┐   WebSocket   ┌─────────────┐   Redis Streams   ┌──────────────┐
│ Bybit API   │──────────────►│  Ingestor   │──────────────────►│ FeatureHub   │
└─────────────┘               └─────────────┘                   └──────┬───────┘
                                                                        │
┌─────────────┐               ┌─────────────┐   ONNX Inference  ┌──────▼───────┐
│ Order API   │◄──────────────┤Order Router │◄──────────────────┤Model Server  │
└─────────────┘               └─────────────┘                   └──────────────┘
```

## 🚀 クイックスタート

### 前提条件

- Python 3.13+
- Docker & Docker Compose
- Redis（開発環境では自動起動）

### 1. 環境セットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd mlbot

# Python 3.13仮想環境を作成
python3.13 -m venv .venv
source .venv/bin/activate

# 依存関係をインストール
pip install fastapi "uvicorn[standard]" websockets redis duckdb pandas numpy structlog prometheus-client pydantic pydantic-settings httpx aiohttp psutil

# 環境設定をコピー
cp .env.example .env
```

### 2. Bybit API設定

`.env`ファイルを編集してBybit APIキーを設定：

```env
BYBIT__API_KEY=your_api_key_here
BYBIT__API_SECRET=your_api_secret_here
BYBIT__TESTNET=true  # 開発時はtrue
```

### 3. アプリケーション起動

#### 開発モード（単一プロセス）

```bash
source .venv/bin/activate
python -m src.main
```

#### プロダクションモード（Docker Compose）

```bash
docker-compose up --build
```

### 4. 動作確認

- **ヘルスチェック**: http://localhost:8080/health
- **メトリクス**: http://localhost:8080/metrics
- **Grafana ダッシュボード**: http://localhost:3000 (admin/admin)

## 📊 監視とメトリクス

### 主要メトリクス

- **システムメトリクス**: CPU、メモリ、ディスク使用率
- **アプリケーションメトリクス**: レイテンシ、スループット、エラー率
- **取引メトリクス**: PnL、約定率、ポジション状況

### アラート設定

- **P1**: システム停止、大幅損失（5σ超）
- **P2**: パフォーマンス劣化、接続エラー
- **P3**: リソース警告、定期タスク失敗

## 🛡️ リスク管理

### 自動制御機能

- **最大ポジションサイズ**: 資産の10%
- **最大ドローダウン**: 5%で警告、10%で緊急停止
- **清算価格バッファ**: 20%以上維持
- **注文タイムアウト**: 300秒で自動キャンセル

### 手動制御

```bash
# 緊急停止
curl -X POST http://localhost:8002/emergency-stop

# ポジション確認
curl http://localhost:8002/positions

# パフォーマンス確認
curl http://localhost:8002/performance
```

## 🔧 開発

### プロジェクト構造

```
mlbot/
├── src/
│   ├── common/          # 共通ユーティリティ
│   ├── ingestor/        # WebSocketデータ収集
│   ├── feature_hub/     # 特徴量エンジニアリング
│   ├── model_server/    # ML推論サーバー
│   └── order_router/    # 注文執行エンジン
├── tests/               # テストスイート
├── k8s/                 # Kubernetes設定
├── config/              # 設定ファイル
└── notebooks/           # データ分析用Jupyter
```

### テスト実行

```bash
# ユニットテスト
python -m pytest tests/unit/

# 統合テスト
python -m pytest tests/integration/

# 全テスト実行
python -m pytest
```

### コード品質

```bash
# フォーマット
black src/ tests/

# リント
ruff check src/ tests/

# 型チェック
mypy src/
```

## 📈 パフォーマンス目標

| 指標 | 目標値 |
|------|--------|
| エンドツーエンドレイテンシ | < 15ms |
| WebSocket処理 | < 1ms |
| モデル推論 | < 1ms |
| 特徴量計算 | < 5ms |
| システム稼働率 | > 99.9% |

## 🚀 デプロイメント

### Kubernetes

```bash
# Namespace作成
kubectl apply -f k8s/base/namespace.yaml

# 設定適用
kubectl apply -f k8s/base/

# サービス確認
kubectl get pods -n mlbot
```

### クラウドプロバイダー

- **AWS**: EKS + RDS + ElastiCache
- **GCP**: GKE + Cloud SQL + Memorystore
- **Azure**: AKS + Azure Database + Cache

## 📝 設定オプション

### 主要設定項目

| 設定項目 | デフォルト | 説明 |
|----------|------------|------|
| `TRADING__MAX_POSITION_PCT` | 0.10 | 最大ポジションサイズ（資産比） |
| `MODEL__DELTA_VALUES` | [0.02, 0.05, 0.10] | 価格オフセット候補 |
| `MODEL__LOOKAHEAD_WINDOWS` | [60, 300, 900] | ルックアヘッド期間（秒） |
| `BYBIT__SYMBOLS` | ["BTCUSDT", "ETHUSDT"] | 監視通貨ペア |

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## ⚠️ 免責事項

このソフトウェアは教育・研究目的で提供されています。実際の取引での使用は自己責任で行ってください。開発者は取引による損失について一切の責任を負いません。

## 📄 ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

---

**開発チーム**: ML Bot Team <team@mlbot.ai>  
**バージョン**: 0.1.0  
**最終更新**: 2025年6月10日