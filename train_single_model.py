#!/usr/bin/env python3
"""
単一モデル（XGBoost）の訓練スクリプト
アンサンブルの代わりに高速で安定したXGBoostのみを使用
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os
import logging
from datetime import datetime
import json

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_xgboost_model(X_train, y_train, X_val, y_val):
    """XGBoostモデルの訓練"""
    logger.info("Training XGBoost model...")
    
    # パラメータ設定
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',  # M3での高速化
        'device': 'cpu',
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.01,
        'reg_lambda': 1,
        'scale_pos_weight': 1.2,  # 不均衡データ対策
        'seed': 42
    }
    
    # データセット作成
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # 評価セット
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    # コールバック設定
    callbacks = [
        xgb.callback.EarlyStopping(rounds=50, metric_name='val-auc', maximize=True),
        xgb.callback.EvaluationMonitor(period=10)
    ]
    
    # モデル訓練
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=evals,
        callbacks=callbacks,
        verbose_eval=True
    )
    
    # 予測
    val_pred = model.predict(dval)
    val_auc = roc_auc_score(y_val, val_pred)
    
    logger.info(f"Validation AUC: {val_auc:.4f}")
    
    # 特徴量重要度
    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("\nTop 10 important features:")
    for feat, score in sorted_importance[:10]:
        logger.info(f"  {feat}: {score:.4f}")
    
    return model, val_auc


def evaluate_model(model, X_test, y_test):
    """モデルの評価"""
    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)
    
    # バイナリ予測
    y_pred = (predictions >= 0.5).astype(int)
    
    # メトリクス計算
    test_auc = roc_auc_score(y_test, predictions)
    
    logger.info(f"\nTest Set Performance:")
    logger.info(f"AUC: {test_auc:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # 予測分布
    logger.info(f"\nPrediction distribution:")
    logger.info(f"Mean: {predictions.mean():.4f}")
    logger.info(f"Std: {predictions.std():.4f}")
    logger.info(f"Min: {predictions.min():.4f}")
    logger.info(f"Max: {predictions.max():.4f}")
    
    # Buy/Sell比率
    buy_ratio = (predictions >= 0.5).mean()
    logger.info(f"\nBuy ratio: {buy_ratio:.2%}")
    
    return test_auc, predictions


def save_model(model, output_dir="models/v4_xgboost"):
    """モデルの保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # XGBoostネイティブ形式
    model_path = os.path.join(output_dir, "model.json")
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # メタデータ保存
    metadata = {
        "training_date": datetime.now().isoformat(),
        "model_type": "xgboost",
        "feature_dim": 52,
        "framework_version": xgb.__version__
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_path


def main():
    """メイン処理"""
    # データセット読み込み
    logger.info("Loading balanced dataset...")
    data = np.load('data/balanced_dataset_v4_full.npz')
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    logger.info(f"Train Buy ratio: {y_train.mean():.2%}")
    logger.info(f"Val Buy ratio: {y_val.mean():.2%}")
    logger.info(f"Test Buy ratio: {y_test.mean():.2%}")
    
    # モデル訓練
    model, val_auc = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # テストセット評価
    test_auc, predictions = evaluate_model(model, X_test, y_test)
    
    # モデル保存
    model_path = save_model(model)
    
    # サマリー
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Model: XGBoost")
    logger.info(f"Validation AUC: {val_auc:.4f}")
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()