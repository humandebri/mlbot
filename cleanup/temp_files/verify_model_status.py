#!/usr/bin/env python3
"""
現在のモデル状況を詳細検証
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import os
import json
import pickle
from pathlib import Path
from src.common.config import settings
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier
from src.ml_pipeline.inference_engine import InferenceEngine, InferenceConfig

logger = get_logger(__name__)

def verify_model_status():
    """現在のモデル状況を詳細検証"""
    
    logger.info("🔍 モデル状況の詳細検証開始...")
    
    discord_notifier.send_system_status(
        "model_verification",
        "🔍 **モデル状況検証開始** 🔍\n\n" +
        "デモデータ混入問題の調査中..."
    )
    
    verification_results = {}
    
    try:
        # 1. 設定確認
        model_path = settings.model.model_path
        logger.info(f"📋 設定されたモデルパス: {model_path}")
        verification_results["config_path"] = model_path
        
        # 2. ファイル存在確認
        model_exists = os.path.exists(model_path)
        logger.info(f"📁 モデルファイル存在: {model_exists}")
        verification_results["model_exists"] = model_exists
        
        if model_exists:
            model_size = os.path.getsize(model_path)
            logger.info(f"📊 モデルサイズ: {model_size:,} bytes")
            verification_results["model_size"] = model_size
        
        # 3. メタデータ確認
        metadata_path = Path(model_path).parent / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"📜 メタデータ: {metadata}")
            verification_results["metadata"] = metadata
        
        # 4. 利用可能モデル一覧
        models_dir = Path("models")
        available_models = {}
        
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                model_onnx = model_dir / "model.onnx"
                if model_onnx.exists():
                    size = model_onnx.stat().st_size
                    available_models[model_dir.name] = {
                        "size": size,
                        "path": str(model_onnx)
                    }
                    
                    # メタデータがあれば読み込み
                    meta_path = model_dir / "metadata.json"
                    if meta_path.exists():
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        available_models[model_dir.name]["metadata"] = meta
        
        logger.info(f"📂 利用可能モデル: {list(available_models.keys())}")
        verification_results["available_models"] = available_models
        
        # 5. 推論エンジンテスト
        try:
            inference_config = InferenceConfig(
                model_path=model_path,
                confidence_threshold=0.5
            )
            inference_engine = InferenceEngine(inference_config)
            inference_engine.load_model()
            
            logger.info("✅ 推論エンジン初期化成功")
            verification_results["inference_engine"] = "success"
            
            # モデル詳細取得
            if hasattr(inference_engine, 'onnx_session'):
                input_shape = inference_engine.onnx_session.get_inputs()[0].shape
                output_shape = inference_engine.onnx_session.get_outputs()[0].shape
                logger.info(f"🎯 モデル入力形状: {input_shape}")
                logger.info(f"🎯 モデル出力形状: {output_shape}")
                verification_results["model_shapes"] = {
                    "input": input_shape,
                    "output": output_shape
                }
            
        except Exception as e:
            logger.error(f"❌ 推論エンジンエラー: {e}")
            verification_results["inference_engine"] = f"error: {e}"
        
        # 6. 報告書生成
        report = "🔍 **モデル状況検証結果** 🔍\n\n"
        
        # 現在のモデル
        current_model_name = Path(model_path).parent.name
        report += f"📋 **現在のモデル**: {current_model_name}\n"
        report += f"📁 **存在**: {'✅' if model_exists else '❌'}\n"
        
        if model_exists:
            report += f"📊 **サイズ**: {model_size/1024/1024:.1f}MB\n"
            
            if "metadata" in verification_results:
                meta = verification_results["metadata"]
                auc = meta.get("performance", {}).get("auc", "不明")
                features = meta.get("features", "不明")
                report += f"🎯 **AUC**: {auc}\n"
                report += f"🔢 **特徴量数**: {features}\n"
        
        # 利用可能モデル比較
        report += f"\n📂 **利用可能モデル**:\n"
        for model_name, model_info in available_models.items():
            size_mb = model_info["size"] / 1024 / 1024
            
            meta = model_info.get("metadata", {})
            auc = meta.get("performance", {}).get("auc", "不明")
            
            status = "🟢 現在" if model_name == current_model_name else "⭐"
            report += f"{status} **{model_name}**: {size_mb:.1f}MB, AUC {auc}\n"
        
        # 推奨事項
        if current_model_name == "v2.0":
            report += f"\n⚠️ **警告**: v2.0モデルはランダムデータで訓練\n"
            report += f"📈 **推奨**: v3.1_improvedに切り替え（AUC 0.838）"
        elif current_model_name == "v3.1_improved":
            report += f"\n✅ **良好**: 高性能モデル使用中"
        
        # Discord通知
        if current_model_name == "v3.1_improved":
            discord_notifier.send_system_status("model_verification_good", report)
        else:
            discord_notifier.send_error("model_verification", report)
        
        return verification_results
        
    except Exception as e:
        logger.error(f"❌ 検証失敗: {e}")
        discord_notifier.send_error("model_verification", f"検証失敗: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting model verification")
    result = verify_model_status()
    logger.info("Verification complete")