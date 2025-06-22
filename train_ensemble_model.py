#!/usr/bin/env python3
"""
アンサンブル深層学習モデルの訓練スクリプト
Buy/Sellバイアスを解決するための複数モデルの組み合わせ
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import json
import os
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU設定
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")


class FocalLoss(tf.keras.losses.Loss):
    """クラス不均衡に強いFocal Loss"""
    
    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Focal loss計算
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # クラス重み付け
        alpha_weight = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        # 交差エントロピー
        ce = -tf.math.log(p_t)
        
        # 最終損失
        loss = alpha_weight * focal_weight * ce
        
        return tf.reduce_mean(loss)


class EnsembleModelTrainer:
    """アンサンブルモデルの訓練"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        self.models = {}
        self.histories = {}
        self.predictions = {}
        
    def get_default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'lstm': {
                'units': [128, 64],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 512,
                'epochs': 100
            },
            'transformer': {
                'n_heads': 4,
                'd_model': 64,
                'n_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 512,
                'epochs': 100
            },
            'xgboost': {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 1.0  # バランス調整
            },
            'neural_net': {
                'layers': [256, 128, 64],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 512,
                'epochs': 100
            },
            'ensemble': {
                'weights': [0.3, 0.3, 0.2, 0.2],  # LSTM, Transformer, XGBoost, NN
                'threshold_optimization': True
            }
        }
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """LSTMモデルを構築"""
        config = self.config['lstm']
        
        model = models.Sequential([
            # 入力を3Dに変換（batch, timesteps, features）
            layers.Reshape((1, input_shape[0]), input_shape=(input_shape[0],)),
            
            # LSTM層
            layers.LSTM(config['units'][0], return_sequences=True, 
                       kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(config['dropout']),
            
            layers.LSTM(config['units'][1], 
                       kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(config['dropout']),
            
            # 出力層
            layers.Dense(32, activation='relu'),
            layers.Dropout(config['dropout']/2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss=FocalLoss(gamma=2.0, alpha=0.25),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def build_transformer_model(self, input_shape: Tuple[int]) -> tf.keras.Model:
        """Transformerモデルを構築"""
        config = self.config['transformer']
        
        # Multi-head attention層
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Attention
            x = layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(inputs, inputs)
            x = layers.Dropout(dropout)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            res = x + inputs
            
            # Feed Forward
            x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
            x = layers.Dropout(dropout)(x)
            x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            return x + res
        
        # モデル構築
        inputs = layers.Input(shape=(input_shape[0],))
        x = layers.Reshape((1, input_shape[0]))(inputs)
        
        # Positional encoding
        positions = tf.range(start=0, limit=1, delta=1)
        position_embedding = layers.Embedding(
            input_dim=1, output_dim=config['d_model']
        )(positions)
        
        x = layers.Dense(config['d_model'])(x)
        x = x + position_embedding
        
        # Transformer blocks
        for _ in range(config['n_layers']):
            x = transformer_encoder(
                x, 
                head_size=config['d_model']//config['n_heads'],
                num_heads=config['n_heads'],
                ff_dim=config['d_model']*4,
                dropout=config['dropout']
            )
        
        # 出力層
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(config['dropout'])(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss=FocalLoss(gamma=2.0, alpha=0.25),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def build_neural_net_model(self, input_shape: Tuple[int]) -> tf.keras.Model:
        """通常のニューラルネットワークモデル"""
        config = self.config['neural_net']
        
        model = models.Sequential()
        
        # 入力層
        model.add(layers.Input(shape=(input_shape[0],)))
        
        # 隠れ層
        for i, units in enumerate(config['layers']):
            model.add(layers.Dense(units, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(config['dropout']))
        
        # 出力層
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss=FocalLoss(gamma=2.0, alpha=0.25),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def create_prediction_monitor_callback(self):
        """予測の偏りを監視するコールバック"""
        class PredictionMonitor(tf.keras.callbacks.Callback):
            def __init__(self, X_val, y_val):
                super().__init__()
                self.X_val = X_val
                self.y_val = y_val
                self.history = {'epoch': [], 'buy_ratio': [], 'val_auc': []}
            
            def on_epoch_end(self, epoch, logs=None):
                predictions = self.model.predict(self.X_val, verbose=0)
                buy_ratio = (predictions > 0.5).mean()
                val_auc = roc_auc_score(self.y_val, predictions)
                
                self.history['epoch'].append(epoch)
                self.history['buy_ratio'].append(buy_ratio)
                self.history['val_auc'].append(val_auc)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Buy ratio = {buy_ratio:.2%}, Val AUC = {val_auc:.4f}")
                
                # 極端な偏りを検出
                if buy_ratio < 0.2 or buy_ratio > 0.8:
                    logger.warning(f"⚠️  Extreme bias detected! Buy ratio: {buy_ratio:.2%}")
        
        return PredictionMonitor
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray):
        """すべてのモデルを訓練"""
        logger.info("Starting ensemble model training...")
        
        # 1. LSTMモデル
        logger.info("\n1. Training LSTM model...")
        lstm_model = self.build_lstm_model((X_train.shape[1],))
        
        monitor = self.create_prediction_monitor_callback()(X_val, y_val)
        early_stop = callbacks.EarlyStopping(patience=15, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        
        history_lstm = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['lstm']['epochs'],
            batch_size=self.config['lstm']['batch_size'],
            callbacks=[monitor, early_stop, reduce_lr],
            verbose=1
        )
        
        self.models['lstm'] = lstm_model
        self.histories['lstm'] = history_lstm
        
        # 2. Transformerモデル
        logger.info("\n2. Training Transformer model...")
        transformer_model = self.build_transformer_model((X_train.shape[1],))
        
        monitor = self.create_prediction_monitor_callback()(X_val, y_val)
        
        history_transformer = transformer_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['transformer']['epochs'],
            batch_size=self.config['transformer']['batch_size'],
            callbacks=[monitor, early_stop, reduce_lr],
            verbose=1
        )
        
        self.models['transformer'] = transformer_model
        self.histories['transformer'] = history_transformer
        
        # 3. XGBoostモデル
        logger.info("\n3. Training XGBoost model...")
        xgb_config = self.config['xgboost']
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            subsample=xgb_config['subsample'],
            colsample_bytree=xgb_config['colsample_bytree'],
            scale_pos_weight=xgb_config['scale_pos_weight'],
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=100
        )
        
        self.models['xgboost'] = xgb_model
        
        # 4. ニューラルネットワークモデル
        logger.info("\n4. Training Neural Network model...")
        nn_model = self.build_neural_net_model((X_train.shape[1],))
        
        monitor = self.create_prediction_monitor_callback()(X_val, y_val)
        
        history_nn = nn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['neural_net']['epochs'],
            batch_size=self.config['neural_net']['batch_size'],
            callbacks=[monitor, early_stop, reduce_lr],
            verbose=1
        )
        
        self.models['neural_net'] = nn_model
        self.histories['neural_net'] = history_nn
        
        logger.info("\nAll models trained successfully!")
    
    def evaluate_individual_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """個別モデルの評価"""
        logger.info("\nEvaluating individual models...")
        
        results = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                # XGBoost
                predictions = model.predict_proba(X_test)[:, 1]
            else:
                # Keras models
                predictions = model.predict(X_test).flatten()
            
            # メトリクス計算
            auc = roc_auc_score(y_test, predictions)
            buy_ratio = (predictions > 0.5).mean()
            
            # 予測分布
            pred_stats = {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'q25': float(np.percentile(predictions, 25)),
                'q50': float(np.percentile(predictions, 50)),
                'q75': float(np.percentile(predictions, 75))
            }
            
            results[name] = {
                'auc': auc,
                'buy_ratio': buy_ratio,
                'prediction_stats': pred_stats
            }
            
            self.predictions[name] = predictions
            
            logger.info(f"{name}: AUC={auc:.4f}, Buy ratio={buy_ratio:.2%}")
            logger.info(f"  Prediction range: [{pred_stats['min']:.3f}, {pred_stats['max']:.3f}]")
        
        return results
    
    def create_ensemble_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """アンサンブル予測を作成"""
        logger.info("\nCreating ensemble predictions...")
        
        # 重み付け平均
        weights = self.config['ensemble']['weights']
        ensemble_pred = np.zeros(len(X_test))
        
        model_names = ['lstm', 'transformer', 'xgboost', 'neural_net']
        
        for i, name in enumerate(model_names):
            ensemble_pred += self.predictions[name] * weights[i]
        
        # 閾値最適化（オプション）
        if self.config['ensemble']['threshold_optimization']:
            # 各モデルの予測を標準化
            for name in model_names:
                pred = self.predictions[name]
                mean, std = pred.mean(), pred.std()
                self.predictions[f'{name}_normalized'] = (pred - mean) / (std + 1e-8)
            
            # 標準化後の重み付け平均
            ensemble_normalized = np.zeros(len(X_test))
            for i, name in enumerate(model_names):
                ensemble_normalized += self.predictions[f'{name}_normalized'] * weights[i]
            
            # シグモイド変換で0-1に戻す
            ensemble_pred = 1 / (1 + np.exp(-ensemble_normalized))
        
        return ensemble_pred
    
    def analyze_ensemble_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """アンサンブルモデルの性能を分析"""
        ensemble_pred = self.create_ensemble_predictions(X_test)
        
        # メトリクス
        auc = roc_auc_score(y_test, ensemble_pred)
        buy_ratio = (ensemble_pred > 0.5).mean()
        
        # 混同行列
        y_pred_binary = (ensemble_pred > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_binary)
        
        # 詳細な統計
        results = {
            'auc': auc,
            'buy_ratio': buy_ratio,
            'confusion_matrix': cm.tolist(),
            'prediction_stats': {
                'mean': float(np.mean(ensemble_pred)),
                'std': float(np.std(ensemble_pred)),
                'min': float(np.min(ensemble_pred)),
                'max': float(np.max(ensemble_pred))
            },
            'classification_report': classification_report(y_test, y_pred_binary, output_dict=True)
        }
        
        logger.info(f"\nEnsemble Performance:")
        logger.info(f"AUC: {auc:.4f}")
        logger.info(f"Buy ratio: {buy_ratio:.2%}")
        logger.info(f"Prediction range: [{results['prediction_stats']['min']:.3f}, {results['prediction_stats']['max']:.3f}]")
        
        return results
    
    def plot_results(self, save_path: str = "models/v4_ensemble/"):
        """結果をプロット"""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 予測分布のヒストグラム
        plt.figure(figsize=(15, 10))
        
        for i, (name, predictions) in enumerate(self.predictions.items()):
            if '_normalized' in name:
                continue
                
            plt.subplot(2, 3, i+1)
            plt.hist(predictions, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(0.5, color='red', linestyle='--', label='Decision boundary')
            plt.title(f'{name.upper()} Predictions')
            plt.xlabel('Prediction Value')
            plt.ylabel('Frequency')
            buy_ratio = (predictions > 0.5).mean()
            plt.text(0.05, 0.95, f'Buy ratio: {buy_ratio:.2%}', 
                    transform=plt.gca().transAxes, verticalalignment='top')
        
        # アンサンブル予測
        ensemble_pred = self.create_ensemble_predictions(None)  # X_testは不要
        plt.subplot(2, 3, 6)
        plt.hist(ensemble_pred, bins=50, alpha=0.7, edgecolor='black', color='green')
        plt.axvline(0.5, color='red', linestyle='--', label='Decision boundary')
        plt.title('ENSEMBLE Predictions')
        plt.xlabel('Prediction Value')
        plt.ylabel('Frequency')
        buy_ratio = (ensemble_pred > 0.5).mean()
        plt.text(0.05, 0.95, f'Buy ratio: {buy_ratio:.2%}', 
                transform=plt.gca().transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'prediction_distributions.png'))
        plt.close()
        
        # 2. 訓練履歴（Kerasモデルのみ）
        plt.figure(figsize=(15, 5))
        
        for i, (name, history) in enumerate(self.histories.items()):
            if history is None:
                continue
                
            plt.subplot(1, 3, i+1)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title(f'{name.upper()} Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'))
        plt.close()
        
        logger.info(f"Plots saved to {save_path}")
    
    def save_models(self, save_path: str = "models/v4_ensemble/"):
        """モデルを保存"""
        os.makedirs(save_path, exist_ok=True)
        
        # Kerasモデル
        for name in ['lstm', 'transformer', 'neural_net']:
            if name in self.models:
                model_path = os.path.join(save_path, f'{name}_model.h5')
                self.models[name].save(model_path)
                logger.info(f"Saved {name} model to {model_path}")
        
        # XGBoostモデル
        if 'xgboost' in self.models:
            import joblib
            xgb_path = os.path.join(save_path, 'xgboost_model.pkl')
            joblib.dump(self.models['xgboost'], xgb_path)
            logger.info(f"Saved XGBoost model to {xgb_path}")
        
        # 設定とメタデータ
        metadata = {
            'config': self.config,
            'created_at': datetime.now().isoformat(),
            'model_names': list(self.models.keys()),
            'ensemble_weights': self.config['ensemble']['weights']
        }
        
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("All models saved successfully!")


def main():
    """メイン処理"""
    # データセット読み込み
    logger.info("Loading balanced dataset...")
    data = np.load('data/balanced_dataset_v4.npz')
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # 検証セット作成（訓練データの20%）
    val_split = int(len(X_train) * 0.8)
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]
    
    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    logger.info(f"Train Buy ratio: {y_train.mean():.2%}")
    logger.info(f"Val Buy ratio: {y_val.mean():.2%}")
    logger.info(f"Test Buy ratio: {y_test.mean():.2%}")
    
    # トレーナー初期化
    trainer = EnsembleModelTrainer()
    
    # モデル訓練
    trainer.train_models(X_train, y_train, X_val, y_val)
    
    # 評価
    individual_results = trainer.evaluate_individual_models(X_test, y_test)
    ensemble_results = trainer.analyze_ensemble_performance(X_test, y_test)
    
    # 結果をプロット
    trainer.plot_results()
    
    # モデル保存
    trainer.save_models()
    
    # 最終サマリー
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print("\nIndividual Model Performance:")
    for name, result in individual_results.items():
        print(f"  {name}: AUC={result['auc']:.4f}, Buy ratio={result['buy_ratio']:.2%}")
    
    print(f"\nEnsemble Performance:")
    print(f"  AUC: {ensemble_results['auc']:.4f}")
    print(f"  Buy ratio: {ensemble_results['buy_ratio']:.2%}")
    print(f"  Prediction range: [{ensemble_results['prediction_stats']['min']:.3f}, {ensemble_results['prediction_stats']['max']:.3f}]")
    
    print("\n✅ Model training complete! Models saved to models/v4_ensemble/")
    print("="*60)


if __name__ == "__main__":
    main()