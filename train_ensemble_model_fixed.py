#!/usr/bin/env python3
"""
アンサンブル深層学習モデルの訓練スクリプト（修正版）
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
from sklearn.preprocessing import StandardScaler
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


class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    """クラス重み付きバイナリ交差エントロピー"""
    
    def __init__(self, pos_weight=1.0, name='weighted_bce'):
        super().__init__(name=name)
        self.pos_weight = pos_weight
    
    def call(self, y_true, y_pred):
        # 安定性のためのクリッピング
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # 重み付きバイナリ交差エントロピー
        bce = y_true * tf.math.log(y_pred) * self.pos_weight + (1 - y_true) * tf.math.log(1 - y_pred)
        return -tf.reduce_mean(bce)


class EnsembleModelTrainer:
    """アンサンブルモデルの訓練"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        self.models = {}
        self.histories = {}
        self.predictions = {}
        self.scaler = StandardScaler()
        self.pos_weight = 1.0  # 後で計算
        
    def get_default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'mlp': {  # LSTMをMLPに変更
                'units': [256, 128, 64],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 128,
                'epochs': 100
            },
            'transformer': {
                'n_heads': 4,
                'd_model': 64,
                'n_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 128,
                'epochs': 100
            },
            'xgboost': {
                'n_estimators': 1000,
                'max_depth': 8,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': None  # 自動計算
            },
            'neural_net': {
                'layers': [256, 128, 64],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 128,
                'epochs': 100
            },
            'ensemble': {
                'weights': [0.25, 0.25, 0.25, 0.25],  # 均等重み
                'threshold_optimization': True
            }
        }
    
    def build_mlp_model(self, input_shape: Tuple[int]) -> tf.keras.Model:
        """MLPモデルを構築（LSTMの代替）"""
        config = self.config['mlp']
        
        model = models.Sequential([
            layers.Input(shape=(input_shape[0],)),
            
            # 入力層の正規化
            layers.BatchNormalization(),
            
            # 隠れ層
            layers.Dense(config['units'][0], activation='relu',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Dropout(config['dropout']),
            
            layers.Dense(config['units'][1], activation='relu',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Dropout(config['dropout']),
            
            layers.Dense(config['units'][2], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(config['dropout']),
            
            # 出力層
            layers.Dense(32, activation='relu'),
            layers.Dropout(config['dropout']/2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss=WeightedBinaryCrossentropy(pos_weight=self.pos_weight),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def build_transformer_model(self, input_shape: Tuple[int]) -> tf.keras.Model:
        """Transformerモデルを構築（修正版）"""
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
            x = layers.Dense(ff_dim, activation="relu")(res)
            x = layers.Dropout(dropout)(x)
            x = layers.Dense(inputs.shape[-1])(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            return x + res
        
        # モデル構築
        inputs = layers.Input(shape=(input_shape[0],))
        x = layers.BatchNormalization()(inputs)
        x = layers.Reshape((1, input_shape[0]))(x)
        
        # Linear projection
        x = layers.Dense(config['d_model'])(x)
        
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
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(config['dropout'])(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss=WeightedBinaryCrossentropy(pos_weight=self.pos_weight),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def build_neural_net_model(self, input_shape: Tuple[int]) -> tf.keras.Model:
        """通常のニューラルネットワークモデル（修正版）"""
        config = self.config['neural_net']
        
        model = models.Sequential()
        
        # 入力層
        model.add(layers.Input(shape=(input_shape[0],)))
        model.add(layers.BatchNormalization())
        
        # 隠れ層
        for i, units in enumerate(config['layers']):
            model.add(layers.Dense(units, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(config['dropout']))
        
        # 出力層
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss=WeightedBinaryCrossentropy(pos_weight=self.pos_weight),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def create_prediction_monitor_callback(self):
        """予測の偏りを監視するコールバック"""
        class PredictionMonitor(tf.keras.callbacks.Callback):
            def __init__(self, X_val, y_val):
                super().__init__()
                self.X_val = X_val[:1000]  # サンプリングして高速化
                self.y_val = y_val[:1000]
                self.history = {'epoch': [], 'buy_ratio': [], 'val_auc': []}
            
            def on_epoch_end(self, epoch, logs=None):
                predictions = self.model.predict(self.X_val, verbose=0)
                buy_ratio = (predictions > 0.5).mean()
                
                if hasattr(self.y_val, 'values'):
                    y_true = self.y_val.values
                else:
                    y_true = self.y_val
                    
                val_auc = roc_auc_score(y_true, predictions)
                
                self.history['epoch'].append(epoch)
                self.history['buy_ratio'].append(buy_ratio)
                self.history['val_auc'].append(val_auc)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Buy ratio = {buy_ratio:.2%}, Val AUC = {val_auc:.4f}")
                
                # 極端な偏りを検出
                if buy_ratio < 0.1 or buy_ratio > 0.9:
                    logger.warning(f"⚠️  Extreme bias detected! Buy ratio: {buy_ratio:.2%}")
        
        return PredictionMonitor
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray):
        """すべてのモデルを訓練"""
        logger.info("Starting ensemble model training...")
        
        # クラス重みの計算
        self.pos_weight = float((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5))
        logger.info(f"Calculated pos_weight: {self.pos_weight:.2f}")
        
        # データの正規化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 1. MLPモデル（LSTMの代替）
        logger.info("\n1. Training MLP model...")
        mlp_model = self.build_mlp_model((X_train.shape[1],))
        
        monitor = self.create_prediction_monitor_callback()(X_val_scaled, y_val)
        early_stop = callbacks.EarlyStopping(
            patience=20, 
            restore_best_weights=True,
            monitor='val_auc',
            mode='max'
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            patience=10, 
            factor=0.5,
            monitor='val_auc',
            mode='max',
            min_lr=1e-6
        )
        
        history_mlp = mlp_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.config['mlp']['epochs'],
            batch_size=self.config['mlp']['batch_size'],
            callbacks=[monitor, early_stop, reduce_lr],
            class_weight={0: 1.0, 1: self.pos_weight},  # クラス重み追加
            verbose=1
        )
        
        self.models['mlp'] = mlp_model
        self.histories['mlp'] = history_mlp
        
        # 2. Transformerモデル
        logger.info("\n2. Training Transformer model...")
        transformer_model = self.build_transformer_model((X_train.shape[1],))
        
        monitor = self.create_prediction_monitor_callback()(X_val_scaled, y_val)
        early_stop = callbacks.EarlyStopping(
            patience=20, 
            restore_best_weights=True,
            monitor='val_auc',
            mode='max'
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            patience=10, 
            factor=0.5,
            monitor='val_auc',
            mode='max',
            min_lr=1e-6
        )
        
        history_transformer = transformer_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.config['transformer']['epochs'],
            batch_size=self.config['transformer']['batch_size'],
            callbacks=[monitor, early_stop, reduce_lr],
            class_weight={0: 1.0, 1: self.pos_weight},
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
            scale_pos_weight=self.pos_weight,
            use_label_encoder=False,
            eval_metric='auc',
            early_stopping_rounds=50,
            random_state=42
        )
        
        xgb_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=True
        )
        
        self.models['xgboost'] = xgb_model
        
        # 4. ニューラルネットワークモデル
        logger.info("\n4. Training Neural Network model...")
        nn_model = self.build_neural_net_model((X_train.shape[1],))
        
        monitor = self.create_prediction_monitor_callback()(X_val_scaled, y_val)
        early_stop = callbacks.EarlyStopping(
            patience=20, 
            restore_best_weights=True,
            monitor='val_auc',
            mode='max'
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            patience=10, 
            factor=0.5,
            monitor='val_auc',
            mode='max',
            min_lr=1e-6
        )
        
        history_nn = nn_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.config['neural_net']['epochs'],
            batch_size=self.config['neural_net']['batch_size'],
            callbacks=[monitor, early_stop, reduce_lr],
            class_weight={0: 1.0, 1: self.pos_weight},
            verbose=1
        )
        
        self.models['neural_net'] = nn_model
        self.histories['neural_net'] = history_nn
        
        logger.info("\nAll models trained successfully!")
    
    def evaluate_individual_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """個別モデルの評価"""
        logger.info("\nEvaluating individual models...")
        
        # テストデータも正規化
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                # XGBoost
                predictions = model.predict_proba(X_test_scaled)[:, 1]
            else:
                # Keras models
                predictions = model.predict(X_test_scaled).flatten()
            
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
    
    def create_ensemble_predictions(self, X_test: np.ndarray = None) -> np.ndarray:
        """アンサンブル予測を作成"""
        logger.info("\nCreating ensemble predictions...")
        
        # 重み付け平均
        weights = self.config['ensemble']['weights']
        
        # 最初の予測で長さを取得
        first_pred = next(iter(self.predictions.values()))
        ensemble_pred = np.zeros(len(first_pred))
        
        model_names = ['mlp', 'transformer', 'xgboost', 'neural_net']
        
        for i, name in enumerate(model_names):
            if name in self.predictions:
                ensemble_pred += self.predictions[name] * weights[i]
        
        # 正規化（重みの合計で割る）
        total_weight = sum(weights[i] for i, name in enumerate(model_names) if name in self.predictions)
        ensemble_pred /= total_weight
        
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
    
    def plot_results(self, save_path: str = "models/v4_ensemble_fixed/"):
        """結果をプロット"""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 予測分布のヒストグラム
        plt.figure(figsize=(15, 10))
        
        plot_idx = 1
        for name, predictions in self.predictions.items():
            if '_normalized' in name:
                continue
                
            plt.subplot(2, 3, plot_idx)
            plt.hist(predictions, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(0.5, color='red', linestyle='--', label='Decision boundary')
            plt.title(f'{name.upper()} Predictions')
            plt.xlabel('Prediction Value')
            plt.ylabel('Frequency')
            buy_ratio = (predictions > 0.5).mean()
            plt.text(0.05, 0.95, f'Buy ratio: {buy_ratio:.2%}', 
                    transform=plt.gca().transAxes, verticalalignment='top')
            plot_idx += 1
        
        # アンサンブル予測
        ensemble_pred = self.create_ensemble_predictions()
        plt.subplot(2, 3, plot_idx)
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
        
        plot_idx = 1
        for name, history in self.histories.items():
            if history is None:
                continue
                
            plt.subplot(1, 3, plot_idx)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title(f'{name.upper()} Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'))
        plt.close()
        
        logger.info(f"Plots saved to {save_path}")
    
    def save_models(self, save_path: str = "models/v4_ensemble_fixed/"):
        """モデルを保存"""
        os.makedirs(save_path, exist_ok=True)
        
        # Kerasモデル
        for name in ['mlp', 'transformer', 'neural_net']:
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
        
        # スケーラーを保存
        scaler_path = os.path.join(save_path, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # 設定とメタデータ
        metadata = {
            'config': self.config,
            'created_at': datetime.now().isoformat(),
            'model_names': list(self.models.keys()),
            'ensemble_weights': self.config['ensemble']['weights'],
            'pos_weight': self.pos_weight,
            'scaler_params': {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
        }
        
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("All models saved successfully!")


def main():
    """メイン処理"""
    # データセット読み込み
    logger.info("Loading balanced dataset...")
    data = np.load('data/balanced_dataset_v4_full.npz')
    
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
    
    print("\n✅ Model training complete! Models saved to models/v4_ensemble_fixed/")
    print("="*60)


if __name__ == "__main__":
    main()