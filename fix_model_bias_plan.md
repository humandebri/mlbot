# モデルバイアス修正計画

## 問題の概要
現在のv3.1_improvedモデルは強いSELLバイアスを持っています：
- すべての予測が0.2〜0.46の範囲（0.5未満）
- Buy/Sell比率: 0/358（100% SELL）

## 原因分析
1. **訓練データの不均衡**
   - 元のデータセットがSELL側に偏っていた可能性
   - 清算データの性質上、下落時の清算が多い？

2. **モデル訓練の問題**
   - クラス不均衡への対処不足
   - 損失関数の設定ミス

## 短期的対策（即座に実行可能）

### 1. 予測値の後処理による調整
```python
# バイアス補正関数
def correct_prediction_bias(raw_prediction, bias_offset=0.15):
    """
    生の予測値を補正してバランスを改善
    """
    # シグモイド関数でスムーズに調整
    corrected = 1 / (1 + np.exp(-10 * (raw_prediction + bias_offset - 0.5)))
    return corrected
```

### 2. 信頼度閾値の調整
```python
# 現在のバイアスを考慮した閾値設定
# 0.46が最大値なので、0.45以上を高信頼度とする
ADJUSTED_CONFIDENCE_THRESHOLD = 0.40  # 元の0.50から調整
```

### 3. 動的バイアス検出と調整
```python
class BiasCorrector:
    def __init__(self, window_size=1000):
        self.predictions = deque(maxlen=window_size)
        
    def add_prediction(self, pred):
        self.predictions.append(pred)
        
    def get_bias_offset(self):
        if len(self.predictions) < 100:
            return 0.15  # デフォルト
        
        mean_pred = np.mean(self.predictions)
        # 平均が0.5になるようオフセットを計算
        return 0.5 - mean_pred
        
    def correct_prediction(self, raw_pred):
        offset = self.get_bias_offset()
        return np.clip(raw_pred + offset, 0.0, 1.0)
```

## 中期的対策

### 1. モデルの再訓練
- クラスバランシングを実施
- focal lossやclass weightsを使用
- SMOTE等のオーバーサンプリング技術を適用

### 2. アンサンブルアプローチ
- 複数のモデルを組み合わせてバイアスを軽減
- 異なる期間のデータで訓練したモデルを統合

### 3. 特徴量エンジニアリング
- BUY/SELL両方向に感度の高い特徴量を追加
- 市場のブル/ベア状態を明示的に特徴量化

## 実装スクリプト

以下のスクリプトで即座にバイアス補正を適用できます：

```bash
# fix_model_bias.py
```