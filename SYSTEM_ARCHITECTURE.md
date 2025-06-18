# MLBot System Architecture
作成日: 2025/06/17  
更新日: 2025/06/18

## 📁 ファイル構造と依存関係

### 🎯 エントリーポイント

#### main_dynamic_integration.py
- **役割**: 統合システムのメインエントリーポイント
- **依存関係**:
  - SimpleServiceManager (src/integration/)
  - DynamicSystemConfig
  - BybitRESTClient, AccountMonitor
  - OrderRouter, TradingCoordinator
- **状態**: ✅ 動作中（Dockerは使用しない）

### 📦 Core Services

#### 1. ingestor/main.py (BybitIngestor)
- **役割**: Bybit WebSocketからのデータ取り込み
- **依存関係**: 
  - BybitWebSocketClient
  - RedisStreams (データ出力)
  - DataArchiver (DuckDB保存)
- **既知の問題**: 
  - ✅ connection_timeout設定エラー → 修正済み（ハードコード60秒）

#### 2. feature_hub/main.py (FeatureHub) 
- **役割**: リアルタイム特徴量生成
- **依存関係**:
  - PriceFeatureEngine → ✅ latest_features属性追加で修正
  - MicroLiquidityEngine, VolatilityMomentumEngine
  - LiquidationFeatureEngine → ✅ FeatureEngineError→FeatureError修正
  - TimeContextEngine, AdvancedFeatureAggregator
  - RedisStreams (データ入出力)
- **既知の問題**: 
  - ✅ 初期化不完全 → SimpleServiceManagerで修正
  - ✅ _initialize_feature_engines()非async化

#### 3. order_router/main.py (OrderRouter)
- **役割**: 注文ルーティングと実行
- **依存関係**:
  - RiskManager, PositionManager
  - OrderExecutor, SmartRouter
  - BybitRESTClient
- **既知の問題**:
  - ✅ initializeメソッド不在 → __init__で初期化済み

### 🔧 共通モジュール (src/common/)

#### bybit_client.py
- **クラス**: BybitWebSocketClient, BybitRESTClient
- **依存関係**: websockets, aiohttp
- **既知の問題**:
  - ✅ get_open_positions()のNoneTypeエラー → sessionチェック追加

#### simple_service_manager.py (src/integration/)
- **役割**: サービスの起動・停止管理
- **既知の問題**:
  - ✅ FeatureHub初期化不完全 → バックグラウンドタスク起動実装
  - ✅ Ingestorのstart()がブロッキング → asyncio.create_task()で解決

### 📊 データフロー

```
[Bybit WebSocket]
    ↓ (kline, orderbook, trades, liquidation)
[Ingestor]
    ↓ (Redis Streams)
[FeatureHub]
    ↓ (Features - 142 per symbol)
[TradingCoordinator]
    ↓ (Signals)
[OrderRouter]
    ↓ (Orders)
[Bybit REST API]
```

### 🚨 修正履歴（2025/06/18）

1. **PriceFeatureEngine**
   - latest_features属性の初期化追加
   - update_trade_featuresメソッドのシグネチャ修正

2. **LiquidationFeatureEngine**
   - FeatureEngineError → FeatureError例外インポート修正

3. **SimpleServiceManager**
   - Ingestorを非同期タスクとして起動
   - FeatureHubの完全な初期化（4つのバックグラウンドタスク起動）

4. **BybitWebSocketClient**
   - connection_timeout値のハードコード（60秒）

5. **FeatureHub**
   - _initialize_feature_engines()を非async化（Python 3.13対応）

### ✅ テスト結果

- quick_feature_test.py: **PASSED**
  - BTCUSDT: 142 features ✅
  - ETHUSDT: 142 features ✅
  - 全サービス正常起動・停止 ✅

### 📝 重要事項

- Docker/Docker Composeは使用しない（直接Python実行）
- Python 3.13環境（.venv）で動作
- Redis、DuckDBは外部プロセスとして稼働
- テストは tests/integration/ に集約