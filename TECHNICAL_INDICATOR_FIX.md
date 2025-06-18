# 技術的指標統合による根本的解決

## 問題
MLモデル（v3.1_improved）が常に0を返す（confidence=0.0, expected_pnl=0.0）

## 原因
- モデルは44個の技術的指標を期待（RSI、MACD、ボリンジャーバンド等）
- システムは186個のマイクロストラクチャ特徴量を生成
- FeatureAdapter44が対応する特徴量を見つけられず、全て0にデフォルト

## 実装した解決策

### 1. TechnicalIndicatorEngine の作成
`src/feature_hub/technical_indicators.py`
- 44個の技術的指標を正確に計算
- 価格履歴の管理（最大100期間）
- 全ての指標の実装：
  - 基本リターン（1, 3, 5, 10, 20期間）
  - ボラティリティ（5, 10, 20, 30期間）
  - 移動平均との乖離（SMA 5, 10, 20, 30）
  - 指数移動平均との乖離（EMA 5, 12）
  - MACD、RSI（14, 21）、ボリンジャーバンド
  - ボリューム比率、モメンタム指標
  - 市場レジーム検出（高/低ボラティリティ、トレンド）
  - 時間特徴量（hour_sin, hour_cos, is_weekend）

### 2. FeatureHub の修正
`src/feature_hub/main.py`
- TechnicalIndicatorEngineの初期化を追加
- `_update_kline_features`でOHLCVデータから技術的指標を計算
- 計算された44指標を特徴量キャッシュにマージ

### 3. DynamicTradingCoordinator の修正
`src/integration/dynamic_trading_coordinator.py`
- `_prepare_features_for_model`をFeatureAdapter44を使用するよう修正
- ハードコードされた特徴量名リストを削除
- 適応統計のロギングを追加
- 応急処置のランダムシグナル生成を削除

### 4. 技術的改善
- 時間特徴量の動的計算（プレースホルダーではなく実際の時刻から）
- 価格履歴が不足時のデフォルト値処理
- エラーハンドリングの強化

## 期待される結果
- MLモデルが期待する44個の技術的指標を正確に受け取る
- モデルが実際の予測値（非ゼロ）を返す
- 取引シグナルが正常に生成される

## テスト方法
```bash
python test_technical_indicators.py
```

## デプロイ方法
```bash
chmod +x deploy_technical_fix.sh
./deploy_technical_fix.sh
```

## 確認ポイント
1. 特徴量生成数が44になっているか
2. FeatureAdapter44のマッチ率が高いか（80%以上が理想）
3. MLモデルの予測が非ゼロ値を返すか
4. 取引シグナルが生成されるか