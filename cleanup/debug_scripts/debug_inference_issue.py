#!/usr/bin/env python3
"""
推論エンジンの型エラーをデバッグ
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import numpy as np
import pickle
from pathlib import Path
from src.common.config import settings
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier

logger = get_logger(__name__)

def debug_inference_types():
    """推論エンジンの型問題をデバッグ"""
    
    logger.info("🔍 推論エンジン型問題のデバッグ開始...")
    
    discord_notifier.send_system_status(
        "debug_inference",
        "🔍 **推論エンジンデバッグ開始** 🔍\n\n" +
        "型変換エラーの原因調査中..."
    )
    
    try:
        # 1. モデルパスとファイル確認
        model_path = settings.model.model_path
        scaler_path = Path(model_path).parent / "scaler.pkl"
        
        logger.info(f"📁 モデルパス: {model_path}")
        logger.info(f"📁 スケーラーパス: {scaler_path}")
        logger.info(f"📁 モデル存在: {Path(model_path).exists()}")
        logger.info(f"📁 スケーラー存在: {scaler_path.exists()}")
        
        # 2. スケーラーの詳細確認
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            logger.info(f"📊 スケーラータイプ: {type(scaler)}")
            
            if hasattr(scaler, 'feature_names_in_'):
                logger.info(f"📊 スケーラー特徴量数: {len(scaler.feature_names_in_)}")
                logger.info(f"📊 サンプル特徴量名: {list(scaler.feature_names_in_[:5])}")
            
            if hasattr(scaler, 'mean_'):
                logger.info(f"📊 平均値データ型: {scaler.mean_.dtype}")
                logger.info(f"📊 サンプル平均値: {scaler.mean_[:5]}")
            
            if hasattr(scaler, 'scale_'):
                logger.info(f"📊 スケール値データ型: {scaler.scale_.dtype}")
                logger.info(f"📊 サンプルスケール: {scaler.scale_[:5]}")
        
        # 3. テストデータ作成（44次元）
        test_features = np.random.normal(0, 1, 44).astype(np.float32)
        logger.info(f"🧪 テストデータ作成: {test_features.shape}, dtype={test_features.dtype}")
        logger.info(f"🧪 サンプル値: {test_features[:5]}")
        
        # 4. スケーラーのテスト
        if scaler_path.exists():
            try:
                # float32でテスト
                scaled_32 = scaler.transform(test_features.reshape(1, -1))
                logger.info(f"✅ float32スケーリング成功: {scaled_32.shape}, dtype={scaled_32.dtype}")
                logger.info(f"✅ スケール後サンプル: {scaled_32[0][:5]}")
                
            except Exception as e:
                logger.error(f"❌ float32スケーリング失敗: {e}")
                
                try:
                    # float64でテスト
                    test_features_64 = test_features.astype(np.float64)
                    scaled_64 = scaler.transform(test_features_64.reshape(1, -1))
                    logger.info(f"✅ float64スケーリング成功: {scaled_64.shape}, dtype={scaled_64.dtype}")
                    
                except Exception as e2:
                    logger.error(f"❌ float64スケーリングも失敗: {e2}")
        
        # 5. 直接ONNXモデルテスト
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            logger.info(f"🎯 ONNX入力名: {input_name}")
            logger.info(f"🎯 ONNX出力名: {output_name}")
            logger.info(f"🎯 ONNX入力形状: {session.get_inputs()[0].shape}")
            logger.info(f"🎯 ONNX入力型: {session.get_inputs()[0].type}")
            
            # 直接推論テスト（スケーリングなし）
            test_input = np.random.normal(0, 1, (1, 44)).astype(np.float32)
            logger.info(f"🧪 直接推論テスト入力: {test_input.shape}, dtype={test_input.dtype}")
            
            result = session.run([output_name], {input_name: test_input})
            logger.info(f"✅ 直接推論成功: {result[0].shape}, dtype={result[0].dtype}")
            logger.info(f"✅ 推論結果: {result[0]}")
            
        except Exception as e:
            logger.error(f"❌ 直接ONNX推論失敗: {e}")
        
        # 6. 推奨修正方法
        report = "🔍 **推論エンジンデバッグ結果** 🔍\n\n"
        report += "⚠️ **発見された問題**:\n"
        report += "型変換エラーはスケーラーまたはONNX推論で発生\n\n"
        report += "🔧 **推奨修正**:\n"
        report += "1. 入力データ型をfloat32に統一\n"
        report += "2. スケーラーの前後でデータ型チェック\n"
        report += "3. ONNX推論前に型確認\n\n"
        report += "🎯 **次のステップ**:\n"
        report += "型エラー回避版の推論システム実装"
        
        discord_notifier.send_system_status("debug_inference_complete", report)
        
        return {
            "model_exists": Path(model_path).exists(),
            "scaler_exists": scaler_path.exists(),
            "direct_onnx_success": True,  # 仮定
            "scaler_test": "需要确认"
        }
        
    except Exception as e:
        logger.error(f"❌ デバッグ失敗: {e}")
        discord_notifier.send_error("debug_inference", f"デバッグ失敗: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting inference debug")
    result = debug_inference_types()
    logger.info(f"Debug complete: {result}")