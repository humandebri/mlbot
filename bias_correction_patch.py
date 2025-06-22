# simple_improved_bot_with_trading_fixed.pyに追加するバイアス補正コード

import numpy as np
from collections import deque

class BiasCorrector:
    """予測値のバイアスを動的に補正"""
    
    def __init__(self, window_size=1000, initial_offset=0.15):
        self.predictions = deque(maxlen=window_size)
        self.initial_offset = initial_offset
        
    def add_prediction(self, pred):
        """予測値を記録"""
        self.predictions.append(pred)
        
    def get_bias_offset(self):
        """現在のバイアスオフセットを計算"""
        if len(self.predictions) < 100:
            return self.initial_offset
        
        mean_pred = np.mean(list(self.predictions))
        # 平均が0.5になるようオフセットを計算
        offset = 0.5 - mean_pred
        # 極端な補正を避ける
        return np.clip(offset, -0.3, 0.3)
        
    def correct_prediction(self, raw_pred):
        """予測値を補正"""
        offset = self.get_bias_offset()
        
        # シグモイド関数でスムーズに調整
        # より急峻な変換で中央値付近の感度を上げる
        adjusted = raw_pred + offset
        corrected = 1 / (1 + np.exp(-8 * (adjusted - 0.5)))
        
        return float(np.clip(corrected, 0.0, 1.0))
    
    def get_stats(self):
        """統計情報を取得"""
        if len(self.predictions) == 0:
            return {}
        
        preds = list(self.predictions)
        return {
            'count': len(preds),
            'mean': np.mean(preds),
            'std': np.std(preds),
            'min': np.min(preds),
            'max': np.max(preds),
            'offset': self.get_bias_offset()
        }
