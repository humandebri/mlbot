# Bybit Liquidation-Driven Trading Bot

**高頻度取引ボット** - Bybitの清算フィードデータをリアルタイムで分析し、機械学習による期待値予測に基づいて自動的に限価注文を発注します。

## 🎯 プロジェクト概要

このプロジェクトは、Bybitの清算情報を活用した機械学習ベースの自動取引システムです。清算カスケードを予測し、価格の一時的な急落（wick）を狙って利益を上げることを目的としています。

### 動作原理

1. **清算データ収集**: WebSocketでリアルタイム清算フィードを監視
2. **カスケード検出**: Z-scoreベースのアルゴリズムで大規模清算を検出
3. **ML予測**: 50以上の特徴量から期待収益を予測
4. **注文実行**: 複数レベルの限価注文を自動配置
5. **リスク管理**: ポジションサイズとドローダウンを厳格に管理

### 主要機能

- **リアルタイム清算データ分析**: 500ms更新の清算フィードを監視
- **機械学習予測**: CatBoost → ONNX による期待値計算（<1ms）
- **動的パラメータ最適化**: Thompson Samplingによるδ（価格オフセット）とT（ルックアヘッド期間）の自動調整
- **マイクロサービス構成**: Docker + Kubernetesでクラウドネイティブ
- **包括的リスク管理**: サーキットブレーカー、ポジション制限、緊急停止機能

## 📈 取引戦略の詳細

### 清算カスケード検出アルゴリズム

システムは以下のステップで清算カスケードを検出します：

```python
# 1. 清算量の異常値検出
liquidation_z_score = (current_liq_volume - rolling_mean) / rolling_std

# 2. 清算スパイクの検出
if liquidation_z_score > threshold:
    # 大規模清算イベント検出
    cascade_detected = True

# 3. 価格インパクトの予測
expected_price_drop = ML_model.predict(features)
```

### 限価注文戦略

清算カスケード検出時、システムは複数レベルの限価注文を配置：

```
現在価格: $100,000
│
├─ レベル1: $99,800 (0.2% below) - 10% of position
├─ レベル2: $99,600 (0.4% below) - 20% of position  
├─ レベル3: $99,400 (0.6% below) - 30% of position
└─ レベル4: $99,200 (0.8% below) - 40% of position
```

各レベルは動的に調整され、市場状況に応じて最適化されます。

## 🧠 機械学習システム

### 特徴量エンジニアリング

システムは4つのカテゴリで50以上の特徴量を計算：

#### 1. **マイクロ流動性指標** (`micro_liquidity.py`)
- **Bid-Ask Spread**: 流動性の指標
- **Order Book Imbalance**: 買い/売り圧力の不均衡
- **Volume at Best**: 最良気配の注文量
- **Depth Skew**: オーダーブックの偏り

#### 2. **ボラティリティ・モメンタム指標** (`volatility_momentum.py`)
- **Realized Volatility**: 実現ボラティリティ（1分、5分、15分）
- **Price Momentum**: 価格の勢い
- **Volume Momentum**: 出来高の勢い
- **Volatility of Volatility**: ボラティリティの変動性

#### 3. **清算特徴量** (`liquidation_features.py`)
- **Liquidation Z-score**: 清算量の異常度
- **Liquidation Velocity**: 清算の加速度
- **Long/Short Ratio**: ロング/ショート清算比率
- **Cumulative Impact**: 累積価格影響
- 🆕 **Size Distribution**: 歪度、尖度、バイモーダリティ
- 🆕 **Asymmetry Metrics**: ロング/ショート非対称性
- 🆕 **Clustering Analysis**: 清算クラスタリング検出

#### 4. **時間コンテキスト** (`time_context.py`)
- **Hour of Day**: 時間帯効果
- **Day of Week**: 曜日効果
- **Market Session**: アジア/欧州/米国セッション
- **Event Proximity**: 重要イベントまでの時間

#### 🆕 5. **高度な市場特徴量** (`advanced_features.py`)
- **Open Interest Dynamics**: OI変化率、速度、加速度
- **Taker/Maker Flow**: 攻撃的取引の検出と分析
- **Order Flow Imbalance**: マイクロストラクチャー分析
- **Microprice Deviation**: Volume-weighted価格偏差

### モデルアーキテクチャ

```
CatBoost (訓練時)
    ↓
ONNX変換 (最適化)
    ↓
ONNXRuntime (推論時)
```

- **訓練**: CatBoost勾配ブースティング
- **最適化**: ONNX形式に変換してサイズ削減
- **推論**: ONNXRuntimeで<1ms推論を実現

### ラベル生成（期待収益計算）

```python
expPNL = (future_price / entry_price - 1) - fee - slippage

# 複数の(δ, T)組み合わせで計算
for delta in [0.02, 0.05, 0.10]:  # 価格オフセット
    for T in [60, 300, 900]:       # ルックアヘッド期間（秒）
        label = calculate_expPNL(delta, T)
```

### 🆕 高度な機械学習技術（v1.1.0）

#### 分数次差分（Fractional Differentiation）
時系列データの定常性を保ちながら、情報を最大限に保持する技術：

```python
from src.ml_pipeline.fractional_diff import FractionalDifferentiator

# 最適なdパラメータを自動探索
frac_diff = FractionalDifferentiator()
optimal_d = frac_diff.find_optimal_d(price_series)
stationary_series = frac_diff.transform(price_series)
```

#### 適応的特徴量ビニング
過学習を防ぎ、ノイズを削減する離散化技術：

```python
from src.ml_pipeline.feature_binning import AdaptiveBinner

# 最適なビン数を自動決定
binner = AdaptiveBinner()
optimal_bins = binner.find_optimal_bins(X, y, features)
X_binned = binner.fit_transform(X, features)
```

#### メタラベリング
予測精度を大幅に向上させる二段階予測フレームワーク：

```python
from src.ml_pipeline.meta_labeling import MetaLabeler

# Primary model: 方向予測
# Meta model: 取引するかどうかとサイズ
meta_labeler = MetaLabeler()
signal = meta_labeler.generate_trading_signal(features, primary_prediction)
```

## 🏗️ システムアーキテクチャ詳細

### マイクロサービス構成

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

### コンポーネント詳細

#### 1. **Ingestor** (データ収集層)
- **役割**: WebSocketデータの収集とRedis Streamsへの配信
- **処理内容**:
  - Kline (1秒足ローソク足)
  - OrderBook (25段階の板情報)
  - Trades (約定履歴)
  - Liquidations (清算情報) ← 最重要
- **パフォーマンス**: <1ms でRedisに配信

#### 2. **Feature Hub** (特徴量計算層)
- **役割**: リアルタイム特徴量計算
- **処理内容**:
  - ローリングウィンドウでの統計量計算
  - 複数時間軸での集計
  - 正規化とスケーリング
- **最適化**: NumPy/Pandasのベクトル化演算

#### 3. **Model Server** (推論層)
- **役割**: ML予測の提供
- **エンドポイント**:
  - `/predict` - 単一予測
  - `/predict/batch` - バッチ予測
  - `/model/reload` - モデル再読み込み
- **キャッシング**: 同一特徴量の予測結果を短期キャッシュ

#### 4. **Order Router** (執行層)
- **役割**: 注文の実行と管理
- **機能**:
  - スマートオーダールーティング
  - ポジション管理
  - リスク制御
  - PnL追跡

## 🔄 データフロー詳細

### リアルタイムパス (<15ms)

```
1. WebSocket Message受信 (0.5ms)
   ↓
2. JSON Parse & Validation (0.2ms)
   ↓
3. Redis Stream書き込み (0.8ms)
   ↓
4. Feature Hub読み取り (0.5ms)
   ↓
5. 特徴量計算 (3-5ms)
   ↓
6. Model Server推論 (0.8ms)
   ↓
7. Order Router判断 (1ms)
   ↓
8. 注文送信 (5-8ms)
```

### バッチ処理パス

```
Redis Streams → DuckDB (毎時)
   - パフォーマンス分析
   - バックテスト用データ
   - モデル再学習用データセット
```

## 💡 Thompson Sampling詳細

### 動的パラメータ最適化

システムは以下のパラメータを自動最適化：

```python
# ベータ分布でモデル化
success_count[δ, T] ~ Beta(α, β)

# 各ラウンドでサンプリング
for each trading_round:
    # 各(δ, T)組み合わせからサンプル
    samples = {
        (δ, T): beta.rvs(α[δ,T], β[δ,T])
        for δ, T in parameter_space
    }
    
    # 最大期待値のパラメータを選択
    best_params = argmax(samples)
    
    # 実行と結果観察
    result = execute_trade(best_params)
    
    # パラメータ更新
    if result > 0:
        α[best_params] += 1
    else:
        β[best_params] += 1
```

### 探索vs活用のバランス

- **初期段階**: 高い探索率で様々なパラメータを試行
- **収束段階**: 成功率の高いパラメータに集中
- **再探索**: 市場状況変化時に探索率を上昇

## 🛡️ リスク管理システム詳細

### 多層防御アーキテクチャ

#### レベル1: ポジションサイズ制限
```python
max_position_size = min(
    account_balance * 0.10,      # 資産の10%
    daily_loss_limit - daily_pnl, # 日次損失限度額まで
    kelly_criterion_size         # Kelly基準サイズ
)
```

#### レベル2: ドローダウン管理
```python
current_drawdown = (peak_equity - current_equity) / peak_equity

if current_drawdown > 0.05:
    send_alert("Drawdown warning: {:.2%}".format(current_drawdown))
    reduce_position_size(0.5)  # ポジションサイズ50%削減

if current_drawdown > 0.10:
    emergency_stop()  # 全ポジションクローズ
    halt_trading()    # 取引停止
```

#### レベル3: サーキットブレーカー
```python
# 短期損失チェック（5分間）
if losses_5min > account_balance * 0.02:
    pause_trading(minutes=30)

# 異常取引検出
if order_frequency > 100_per_minute:
    block_new_orders()
    investigate_anomaly()
```

### ポジション管理

```python
class PositionManager:
    def calculate_position_size(self, signal_strength, volatility):
        # Kelly基準
        kelly_fraction = (expected_return * signal_strength) / variance
        
        # ボラティリティ調整
        vol_adjusted_size = base_size / (1 + volatility * vol_multiplier)
        
        # 最終サイズ
        return min(kelly_fraction, vol_adjusted_size, max_position_size)
```

## 🚀 クイックスタート

### 前提条件

- Python 3.13+
- Docker & Docker Compose  
- Redis 7.0+
- 8GB+ RAM推奨

### 1. 環境セットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd mlbot

# Python 3.13仮想環境を作成
python3.13 -m venv .venv
source .venv/bin/activate

# 依存関係をインストール
pip install -r requirements.txt

# 環境設定をコピー
cp .env.example .env
```

### 2. Bybit API設定

`.env`ファイルを編集してBybit APIキーを設定：

```env
# Bybit API Configuration
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
USE_TESTNET=true  # 本番環境ではfalseに設定

# Trading Symbols  
SYMBOLS=BTCUSDT,ETHUSDT,ICPUSDT

# Risk Limits
MAX_POSITION_SIZE_USD=100000
MAX_LEVERAGE=10
MAX_DAILY_LOSS_USD=10000
```

### 3. システム起動

#### 開発モード（統合システム）

```bash
# Redis起動（別ターミナル）
docker-compose up redis

# システム起動
python scripts/start_system.py --testnet

# モニタリングダッシュボード付き起動
python scripts/start_system.py --testnet --dashboard
```

#### プロダクションモード（完全Docker）

```bash
# 全サービス起動
docker-compose up --build

# スケーリング（必要に応じて）
docker-compose up --scale ingestor=3 --scale model_server=2
```

### 4. 動作確認

```bash
# システムステータス確認
python scripts/check_status.py

# ヘルスチェック
curl http://localhost:8080/system/health

# 現在のポジション
curl http://localhost:8080/trading/positions

# パフォーマンス統計
curl http://localhost:8080/trading/performance
```

## 📊 監視とメトリクス

### リアルタイムダッシュボード

```
╭─────────────────── LIQUIDATION TRADING BOT ───────────────────╮
│ Status: ● RUNNING  │ Uptime: 4h 32m │ CPU: 12% │ Mem: 2.3GB  │
├────────────────────────────────────────────────────────────────┤
│ TRADING METRICS                                                │
│ ├─ Total P&L: $1,234.56 ▲                                     │
│ ├─ Win Rate: 67.3%                                            │
│ ├─ Active Positions: 3                                        │
│ └─ Daily Volume: $45,678                                      │
├────────────────────────────────────────────────────────────────┤
│ SYSTEM HEALTH                                                  │
│ ├─ WebSocket: ✓ Connected (12ms ping)                         │
│ ├─ Redis: ✓ Healthy (0.3ms latency)                          │
│ ├─ Model Server: ✓ Running (0.8ms inference)                 │
│ └─ Order Router: ✓ Active                                    │
├────────────────────────────────────────────────────────────────┤
│ RECENT TRADES                                                  │
│ 12:34:56 BTCUSDT BUY  0.01 @ $65,432 FILLED                  │
│ 12:33:12 ETHUSDT SELL 0.10 @ $3,456  FILLED                  │
│ 12:31:45 ICPUSDT BUY  5.00 @ $45.67  PARTIAL                 │
╰────────────────────────────────────────────────────────────────╯
```

### Prometheusメトリクス

主要メトリクス：
- `mlbot_messages_received_total` - 受信メッセージ数
- `mlbot_orders_placed_total` - 発注数
- `mlbot_orders_filled_total` - 約定数
- `mlbot_prediction_latency_seconds` - 予測レイテンシ
- `mlbot_position_pnl_dollars` - ポジション損益

### アラート設定例

```yaml
# Prometheus Alert Rules
groups:
  - name: mlbot_alerts
    rules:
      - alert: HighDrawdown
        expr: mlbot_drawdown_percent > 0.05
        for: 1m
        annotations:
          summary: "High drawdown detected: {{ $value | humanizePercentage }}"
          
      - alert: LowWinRate
        expr: rate(mlbot_wins_total[1h]) / rate(mlbot_trades_total[1h]) < 0.4
        for: 5m
        annotations:
          summary: "Win rate below 40%"
```

## 🧪 バックテスト

### バックテストの実行

```python
from src.ml_pipeline.backtester import Backtester

# バックテスター初期化
backtester = Backtester(
    start_date="2024-01-01",
    end_date="2024-12-31",
    initial_capital=10000,
    fee_rate=0.00055  # Bybit taker fee
)

# 戦略実行
results = backtester.run(
    model_path="models/v1.0/model.onnx",
    data_path="data/historical/2024.parquet"
)

# 結果分析
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
```

### パフォーマンス評価指標

- **Sharpe Ratio**: リスク調整後リターン
- **Sortino Ratio**: 下方リスクのみ考慮
- **Calmar Ratio**: リターン/最大ドローダウン
- **Information Ratio**: アクティブリターン/トラッキングエラー

## 🔧 高度な設定

### カスタム特徴量の追加

```python
# src/feature_hub/custom_features.py
class CustomFeatureCalculator:
    def calculate(self, market_data: Dict) -> Dict[str, float]:
        features = {}
        
        # カスタム指標例：価格加速度
        prices = market_data['prices']
        velocity = np.diff(prices)
        acceleration = np.diff(velocity)
        features['price_acceleration'] = acceleration[-1]
        
        # カスタム指標例：板の厚み比率
        bid_depth = sum(market_data['bids'].values())
        ask_depth = sum(market_data['asks'].values())
        features['depth_ratio'] = bid_depth / (bid_depth + ask_depth)
        
        return features
```

### モデルのカスタマイズ

```python
# カスタムCatBoostパラメータ
custom_params = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 5.0,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'min_data_in_leaf': 20,
    'random_seed': 42,
    'task_type': 'GPU',  # GPU使用時
    'devices': '0'       # GPU ID
}
```

### リスクパラメータの調整

```python
# 保守的な設定
CONSERVATIVE_CONFIG = {
    'max_position_pct': 0.05,      # 5%に制限
    'max_drawdown_pct': 0.03,      # 3%で警告
    'min_confidence': 0.8,         # 80%以上の確信度
    'order_levels': 2,             # 2レベルのみ
}

# アグレッシブな設定
AGGRESSIVE_CONFIG = {
    'max_position_pct': 0.20,      # 20%まで許可
    'max_drawdown_pct': 0.15,      # 15%まで許容
    'min_confidence': 0.5,         # 50%以上で取引
    'order_levels': 6,             # 6レベル配置
}
```

## 🐛 トラブルシューティング

### よくある問題と解決方法

#### 1. WebSocket接続エラー

```bash
ERROR: WebSocket connection failed: 504 Gateway Timeout
```

**解決方法**:
- Bybit APIステータスを確認: https://status.bybit.com
- ファイアウォール設定を確認
- 正しいWebSocket URLを使用しているか確認

#### 2. Redis接続エラー

```bash
ERROR: Could not connect to Redis at localhost:6379
```

**解決方法**:
```bash
# Redisが起動しているか確認
docker ps | grep redis

# 起動していない場合
docker-compose up -d redis
```

#### 3. モデル推論エラー

```bash
ERROR: ONNX Runtime error: Invalid input shape
```

**解決方法**:
- 特徴量の数が訓練時と一致しているか確認
- モデルファイルが破損していないか確認
- 正しいモデルバージョンを使用しているか確認

#### 4. 注文拒否エラー

```bash
ERROR: Order rejected: Insufficient margin
```

**解決方法**:
- アカウント残高を確認
- レバレッジ設定を確認
- ポジションサイズ計算ロジックを確認

### ログレベルの調整

```bash
# 詳細デバッグログ
export LOG_LEVEL=DEBUG
python scripts/start_system.py

# 本番環境（エラーのみ）
export LOG_LEVEL=ERROR
```

### パフォーマンスプロファイリング

```python
# cProfileを使用
python -m cProfile -o profile.stats src/main.py

# 結果を表示
import pstats
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative').print_stats(20)
```

## ❓ FAQ

### Q: 必要な最小資金は？
A: テストネットでは制限なし。本番環境では最低$1,000を推奨（より大きな資金でより安定した運用が可能）。

### Q: VPSは必要ですか？
A: 低レイテンシが重要なため、取引所に近いVPS（東京、シンガポール）の使用を強く推奨します。

### Q: 複数の取引所に対応していますか？
A: 現在はBybitのみ対応。アーキテクチャは拡張可能な設計のため、他の取引所への対応も可能です。

### Q: バックテストの精度は？
A: ティックレベルのデータを使用し、手数料・スリッページを考慮しているため、高い精度を実現しています。

### Q: モデルの再学習頻度は？
A: デフォルトでは週次。市場環境の変化に応じて調整可能です。

### Q: 停電やネットワーク障害時の対応は？
A: 全ポジションに停止価格を設定。デッドマンスイッチにより、接続断時は自動的に全ポジションをクローズします。

## 📈 パフォーマンス最適化

### レイテンシ削減のヒント

1. **地理的最適化**
   - 取引所に近いVPSを使用（ping < 10ms）
   - 専用ネットワーク回線の検討

2. **コード最適化**
   - NumPyベクトル化演算の活用
   - 不要なメモリコピーの削減
   - asyncioによる並行処理

3. **システム最適化**
   - CPU周波数ガバナーを'performance'に設定
   - ネットワークバッファサイズの調整
   - Redisの永続化を無効化（開発環境）

### スケーリング戦略

```yaml
# Horizontal Pod Autoscaler設定
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
    - type: Pods
      pods:
        metric:
          name: inference_latency_p99
        target:
          type: AverageValue
          averageValue: "2"  # 2ms
```

## 🚀 プロダクション移行チェックリスト

- [ ] APIキーの権限を最小限に設定（取引のみ、出金不可）
- [ ] 本番用の設定ファイルを作成
- [ ] バックアップとリカバリ手順を文書化
- [ ] 監視アラートの受信先を設定
- [ ] セキュリティ監査の実施
- [ ] 負荷テストの完了
- [ ] インシデント対応手順の策定
- [ ] 取引ログの法的要件確認
- [ ] 税務処理の自動化設定
- [ ] 緊急連絡体制の確立

## 🤝 コントリビューション

### 開発環境のセットアップ

```bash
# 開発用依存関係をインストール
pip install -r requirements-dev.txt

# pre-commitフックを設定
pre-commit install

# テストを実行
pytest -v

# カバレッジレポート
pytest --cov=src --cov-report=html
```

### コーディング規約

- PEP 8準拠（Black/Ruffで自動整形）
- 型ヒント必須（mypyでチェック）
- Docstring必須（Google Style）
- テストカバレッジ80%以上

### プルリクエストのガイドライン

1. 機能ブランチを作成
2. テストを追加/更新
3. ドキュメントを更新
4. `make lint`でコード品質チェック
5. レビューを依頼

## ⚠️ 免責事項

このソフトウェアは教育・研究目的で提供されています。実際の取引での使用は自己責任で行ってください。

- 過去のパフォーマンスは将来の結果を保証しません
- 暗号資産取引は高リスクです
- 投資額以上の損失が発生する可能性があります
- 税務上の義務を確認してください

開発者は取引による損失について一切の責任を負いません。

## 📄 ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 📞 サポート

- **ドキュメント**: https://docs.mlbot.ai
- **Discord**: https://discord.gg/mlbot
- **Email**: support@mlbot.ai
- **Issue Tracker**: GitHub Issues

---

**開発チーム**: ML Bot Team  
**バージョン**: 1.1.0  
**最終更新**: 2025年6月11日

<div align="center">
  <strong>⚡ Built for Speed | 🛡️ Designed for Safety | 📈 Optimized for Profit</strong>
</div>