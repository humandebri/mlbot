# ML Bot モデル再訓練計画

## 現状の問題
- **v3.1_improvedモデル**: すべての予測が0.20-0.46（SELL側）に偏っている
- **原因**: 訓練データの不均衡、クラスバランシング未実施
- **影響**: 取引シグナルが生成されない（Buy機会を逃す）

## 再訓練の全体計画

### Phase 1: データ準備とクラスバランス改善（1-2日）

#### 1.1 データ収集と前処理
```python
# data_preparation.py
class DataPreparation:
    def __init__(self):
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'ICPUSDT', 'SOLUSDT', 'AVAXUSDT']
        self.lookback_days = 180  # 6ヶ月分のデータ
        
    def collect_liquidation_data(self):
        """清算データの収集（Buy/Sell両方）"""
        # 清算データのバランスを確認
        # 上昇相場と下落相場の両方を含める
        
    def create_balanced_dataset(self):
        """クラスバランシングの実施"""
        # 1. アンダーサンプリング
        # 2. SMOTE（オーバーサンプリング）
        # 3. クラス重み付け
```

#### 1.2 特徴量エンジニアリング改善
```python
# 新たに追加する特徴量（現在の44次元に追加）
additional_features = [
    "buy_liquidation_ratio",     # Buy清算の比率
    "sell_liquidation_ratio",    # Sell清算の比率
    "liquidation_imbalance",     # 清算の偏り指標
    "market_sentiment_score",    # 市場センチメント
    "funding_rate",              # 資金調達率
    "open_interest_change",      # 建玉変化率
    "volume_imbalance",          # 売買ボリューム不均衡
    "order_book_imbalance",      # オーダーブックの偏り
]
```

### Phase 2: モデルアーキテクチャ設計（1日）

#### 2.1 アンサンブルアプローチ
```python
class EnsembleModel:
    """複数のモデルを組み合わせてバイアスを軽減"""
    
    def __init__(self):
        self.models = {
            'lstm': self.build_lstm_model(),
            'transformer': self.build_transformer_model(),
            'gradient_boost': self.build_xgboost_model(),
            'neural_net': self.build_nn_model()
        }
    
    def build_lstm_model(self):
        """時系列パターンを学習するLSTM"""
        model = tf.keras.Sequential([
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        return model
    
    def build_transformer_model(self):
        """Attention機構を使用したTransformer"""
        # Multi-head attentionで長期依存関係を捉える
```

#### 2.2 損失関数の改善
```python
def focal_loss(gamma=2.0, alpha=0.25):
    """クラス不均衡に強いFocal Loss"""
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Focal loss計算
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - pt, gamma)
        
        # クラス重み付け
        alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        
        # 最終損失
        loss = -alpha_weight * focal_weight * tf.log(pt)
        return tf.reduce_mean(loss)
    
    return loss
```

### Phase 3: 訓練とバリデーション（2-3日）

#### 3.1 訓練戦略
```python
class ModelTrainer:
    def __init__(self):
        self.batch_size = 512
        self.epochs = 100
        self.early_stopping = EarlyStopping(patience=10)
        
    def train_with_validation(self, X_train, y_train, X_val, y_val):
        """時系列を考慮したWalk-forward validation"""
        # 1. 時系列分割（過去→未来）
        # 2. クラスバランスを各分割で確認
        # 3. 各エポックでBuy/Sell予測比率を監視
        
    def monitor_predictions(self, model, X_test):
        """予測の偏りをリアルタイム監視"""
        predictions = model.predict(X_test)
        buy_ratio = (predictions > 0.5).mean()
        
        if buy_ratio < 0.3 or buy_ratio > 0.7:
            logger.warning(f"予測バイアス検出: Buy比率 {buy_ratio:.2%}")
```

#### 3.2 ハイパーパラメータ最適化
```python
# Optuna使用
def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1e-2),
        'dropout_rate': trial.suggest_uniform('dropout', 0.1, 0.5),
        'lstm_units': trial.suggest_int('lstm_units', 64, 256),
        'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024]),
        'class_weight_ratio': trial.suggest_uniform('class_weight', 0.3, 0.7)
    }
    
    # モデル訓練
    model = build_model(params)
    
    # Buy/Sellバランスを評価指標に含める
    score = evaluate_model(model, X_val, y_val)
    balance_penalty = abs(0.5 - model.predict(X_val).mean())
    
    return score - balance_penalty * 0.5
```

### Phase 4: 検証と本番デプロイ（1-2日）

#### 4.1 厳密な検証
```python
class ModelValidator:
    def validate_model(self, model, test_data):
        """包括的なモデル検証"""
        
        # 1. 予測分布の確認
        predictions = model.predict(test_data)
        assert predictions.min() > 0.1, "予測が下限に偏っている"
        assert predictions.max() < 0.9, "予測が上限に偏っている"
        assert 0.4 < predictions.mean() < 0.6, "予測の平均が偏っている"
        
        # 2. 時間帯別の検証
        for hour in range(24):
            hour_preds = predictions[test_data['hour'] == hour]
            buy_ratio = (hour_preds > 0.5).mean()
            assert 0.3 < buy_ratio < 0.7, f"時間帯{hour}で偏り検出"
        
        # 3. 市場状況別の検証
        bull_market_preds = predictions[test_data['trend'] > 0]
        bear_market_preds = predictions[test_data['trend'] < 0]
        
        return {
            'overall_balance': predictions.mean(),
            'buy_sell_ratio': (predictions > 0.5).mean(),
            'distribution_std': predictions.std(),
            'bull_market_buy_ratio': (bull_market_preds > 0.5).mean(),
            'bear_market_sell_ratio': (bear_market_preds < 0.5).mean()
        }
```

#### 4.2 A/Bテスト戦略
```python
# 段階的デプロイ
deployment_stages = [
    {"name": "test", "allocation": 0.1, "duration": "24h"},
    {"name": "pilot", "allocation": 0.3, "duration": "3d"},
    {"name": "rollout", "allocation": 0.7, "duration": "7d"},
    {"name": "full", "allocation": 1.0, "duration": "ongoing"}
]
```

## タイムライン

| Phase | 期間 | 成果物 |
|-------|------|--------|
| Phase 1 | 1-2日 | バランスの取れたデータセット、拡張特徴量 |
| Phase 2 | 1日 | 新モデルアーキテクチャ、損失関数 |
| Phase 3 | 2-3日 | 訓練済みモデル（複数候補） |
| Phase 4 | 1-2日 | 検証済み本番モデル、デプロイメント |

**合計: 5-8日**

## 成功指標

1. **予測バランス**: Buy/Sell比率が40-60%の範囲
2. **精度**: AUC 0.85以上（現在0.838）
3. **安定性**: 24時間の予測分散が0.15以下
4. **収益性**: バックテストで月利5%以上

## リスクと対策

| リスク | 対策 |
|--------|------|
| 過学習 | Dropout、正則化、早期停止 |
| データリーク | 厳密な時系列分割 |
| 新たなバイアス | 継続的監視、A/Bテスト |
| 市場変化 | オンライン学習の実装 |

## 実装開始コマンド

```bash
# 1. 新しいブランチを作成
git checkout -b model-retraining-v4

# 2. データ準備スクリプトを実行
python scripts/prepare_balanced_dataset.py

# 3. モデル訓練を開始
python scripts/train_ensemble_model.py --config configs/model_v4.yaml

# 4. 検証とバックテスト
python scripts/validate_model.py --model models/v4_candidate/
```