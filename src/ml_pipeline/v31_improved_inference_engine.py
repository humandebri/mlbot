"""
V3.1_improved専用推論エンジン
TreeEnsembleClassifierの確率出力を回帰値として解釈
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass

import onnxruntime as ort

from ..common.logging import get_logger
from .feature_adapter_44 import FeatureAdapter44

logger = get_logger(__name__)


@dataclass
class V31ImprovedConfig:
    """V3.1_improved推論エンジン設定"""
    
    model_path: str = "models/v3.1_improved/model.onnx"
    confidence_threshold: float = 0.65  # 65%以上で取引実行（調整）
    
    # 予測値閾値
    buy_threshold: float = 0.55   # 55%以上でBUY
    sell_threshold: float = 0.45  # 45%以下でSELL
    
    # 信頼度レベル
    high_confidence: float = 0.75  # 75%以上で高信頼度
    medium_confidence: float = 0.6  # 60%以上で中信頼度


class V31ImprovedInferenceEngine:
    """
    V3.1_improvedモデル専用推論エンジン
    TreeEnsembleClassifierの確率出力を正しく処理
    """
    
    def __init__(self, config: Optional[V31ImprovedConfig] = None):
        """初期化"""
        self.config = config or V31ImprovedConfig()
        self.onnx_session = None
        self.input_name = None
        self.feature_adapter = FeatureAdapter44()
        
        logger.info("V3.1_improved推論エンジン初期化完了")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """モデル読み込み"""
        model_path = model_path or self.config.model_path
        
        try:
            # ONNX Runtime設定（最適化）
            session_options = ort.SessionOptions()
            session_options.enable_cpu_mem_arena = False
            session_options.enable_mem_pattern = False
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # モデル読み込み
            self.onnx_session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=['CPUExecutionProvider']
            )
            
            # 入力名取得
            self.input_name = self.onnx_session.get_inputs()[0].name
            
            # ウォームアップ
            self._warmup_model()
            
            logger.info("V3.1_improvedモデル読み込み完了", model_path=model_path)
            
        except Exception as e:
            logger.error("モデル読み込み失敗", exception=e, model_path=model_path)
            raise
    
    def predict(self, features: Union[Dict[str, float], np.ndarray]) -> Dict[str, Any]:
        """
        予測実行
        
        Args:
            features: 特徴量（辞書またはnumpy配列）
            
        Returns:
            予測結果辞書
        """
        if self.onnx_session is None:
            raise ValueError("モデルが読み込まれていません。load_model()を先に実行してください。")
        
        start_time = time.perf_counter()
        
        try:
            # 特徴量を44次元に変換
            if isinstance(features, dict):
                feature_array = self.feature_adapter.adapt(features)
            else:
                # 既にnumpy配列の場合
                feature_array = features
                if feature_array.shape[-1] != 44:
                    # 44次元でない場合は変換
                    feature_dict = {f"feature_{i}": float(feature_array.flat[i]) 
                                  for i in range(min(len(feature_array.flat), 156))}
                    feature_array = self.feature_adapter.adapt(feature_dict)
            
            # 形状を(1, 44)に調整
            if len(feature_array.shape) == 1:
                feature_array = feature_array.reshape(1, -1)
            
            feature_array = feature_array.astype(np.float32)
            
            # 推論実行
            outputs = self.onnx_session.run(None, {self.input_name: feature_array})
            
            # 出力解析
            label_output = outputs[0][0]  # int64ラベル
            probability_output = outputs[1][0]  # 確率辞書
            
            # 確率辞書から値を抽出
            prob_class_0 = probability_output.get(0, 0.5)  # クラス0の確率
            prob_class_1 = probability_output.get(1, 0.5)  # クラス1の確率
            
            # 回帰値として解釈（クラス1の確率を使用）
            prediction_value = float(prob_class_1)
            
            # 信頼度計算（より確信が高いほど信頼度が高い）
            confidence = float(max(prob_class_0, prob_class_1))
            
            # 推論時間
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            # シグナル判定
            signal_info = self._generate_signal(prediction_value, confidence)
            
            result = {
                'prediction': prediction_value,
                'confidence': confidence,
                'raw_label': int(label_output),
                'probabilities': {
                    'class_0': float(prob_class_0),
                    'class_1': float(prob_class_1)
                },
                'signal': signal_info,
                'inference_time_ms': inference_time_ms,
                'model_info': {
                    'input_shape': feature_array.shape,
                    'model_type': 'v3.1_improved_classifier'
                }
            }
            
            return result
            
        except Exception as e:
            logger.error("予測エラー", exception=e)
            return {
                'prediction': 0.0,
                'confidence': 0.5,
                'error': str(e),
                'raw_label': 0,
                'probabilities': {'class_0': 0.5, 'class_1': 0.5},
                'signal': {'direction': 'HOLD', 'strength': 'low', 'tradeable': False},
                'inference_time_ms': (time.perf_counter() - start_time) * 1000
            }
    
    def _generate_signal(self, prediction: float, confidence: float) -> Dict[str, Any]:
        """シグナル生成"""
        
        # 方向判定
        if prediction >= self.config.buy_threshold:
            direction = 'BUY'
            if prediction >= 0.7:
                direction_strength = 'strong'
            else:
                direction_strength = 'moderate'
        elif prediction <= self.config.sell_threshold:
            direction = 'SELL'
            if prediction <= 0.3:
                direction_strength = 'strong'
            else:
                direction_strength = 'moderate'
        else:
            direction = 'HOLD'
            direction_strength = 'neutral'
        
        # 信頼度レベル
        if confidence >= self.config.high_confidence:
            confidence_level = 'high'
        elif confidence >= self.config.medium_confidence:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        # 取引実行可能性（より積極的な設定）
        tradeable = (
            direction != 'HOLD' and 
            confidence >= 0.6  # 60%で取引実行（テスト用に緩和）
        )
        
        # 推奨ポジションサイズ（信頼度ベース）
        if tradeable:
            if confidence >= 0.8:
                position_size_multiplier = 1.0
            elif confidence >= 0.75:
                position_size_multiplier = 0.8
            elif confidence >= 0.7:
                position_size_multiplier = 0.6
            else:
                position_size_multiplier = 0.4
        else:
            position_size_multiplier = 0.0
        
        return {
            'direction': direction,
            'strength': direction_strength,
            'confidence_level': confidence_level,
            'tradeable': tradeable,
            'position_size_multiplier': position_size_multiplier,
            'prediction_value': prediction,
            'confidence_value': confidence
        }
    
    def _warmup_model(self) -> None:
        """モデルウォームアップ"""
        try:
            # ダミー44次元特徴量
            dummy_features = np.random.randn(44).astype(np.float32).reshape(1, -1)
            
            # ウォームアップ推論
            for _ in range(3):
                _ = self.onnx_session.run(None, {self.input_name: dummy_features})
            
            logger.info("V3.1_improvedモデルウォームアップ完了")
            
        except Exception as e:
            logger.warning("モデルウォームアップ失敗", exception=e)
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        if not self.onnx_session:
            return {"status": "not_loaded"}
        
        input_info = self.onnx_session.get_inputs()[0]
        output_info = self.onnx_session.get_outputs()
        
        return {
            "status": "loaded",
            "model_path": self.config.model_path,
            "input_shape": input_info.shape,
            "input_type": input_info.type,
            "output_count": len(output_info),
            "output_names": [o.name for o in output_info],
            "providers": self.onnx_session.get_providers(),
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "buy_threshold": self.config.buy_threshold,
                "sell_threshold": self.config.sell_threshold,
                "high_confidence": self.config.high_confidence,
                "medium_confidence": self.config.medium_confidence
            }
        }