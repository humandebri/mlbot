#!/usr/bin/env python3
"""
改善された高性能モデル
- 適切なラベル生成（より長い時間軸と現実的な閾値）
- 不均衡データ処理
- 以前の成功事例の設定を活用
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Tuple
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ImprovedHighPerformanceModel:
    """改善された高性能モデル"""
    
    def __init__(self, profit_threshold: float = 0.003):  # より現実的な閾値
        self.profit_threshold = profit_threshold
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
    def load_market_data(self, symbol: str = "BTCUSDT", limit: int = 50000) -> pd.DataFrame:
        """より多くのデータを読み込み"""
        print(f"📊 {symbol}の市場データを読み込み中（最大{limit:,}件）...")
        
        try:
            conn = duckdb.connect("data/historical_data.duckdb", read_only=True)
            
            query = f"""
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM klines_{symbol.lower()}
            WHERE timestamp >= '2024-01-01'
            ORDER BY timestamp ASC
            LIMIT {limit}
            """
            
            data = conn.execute(query).df()
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
            
            conn.close()
            print(f"✅ {len(data):,}件のレコードを読み込み完了")
            print(f"期間: {data.index.min()} から {data.index.max()}")
            
            return data
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return pd.DataFrame()

    def create_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """包括的な特徴量生成（以前の成功事例に基づく）"""
        print("🔧 包括的特徴量を生成中...")
        
        df = data.copy()
        
        # 基本価格特徴量
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['oc_ratio'] = (df['close'] - df['open']) / df['open']
        
        # マルチタイムフレームリターン（より多くの期間）
        for period in [1, 2, 3, 5, 10, 15, 20, 30, 60]:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # ボラティリティ特徴量
        for window in [5, 10, 15, 20, 30, 60]:
            df[f'vol_{window}'] = df['returns'].rolling(window).std()
            df[f'vol_ratio_{window}'] = df[f'vol_{window}'] / df[f'vol_{window}'].rolling(window*2).mean()
        
        # 移動平均とトレンド
        for ma in [5, 10, 15, 20, 30, 50]:
            df[f'sma_{ma}'] = df['close'].rolling(ma).mean()
            df[f'price_vs_sma_{ma}'] = (df['close'] - df[f'sma_{ma}']) / df[f'sma_{ma}']
        
        # 指数移動平均
        for ema in [5, 12, 26]:
            df[f'ema_{ema}'] = df['close'].ewm(span=ema).mean()
            df[f'price_vs_ema_{ema}'] = (df['close'] - df[f'ema_{ema}']) / df[f'ema_{ema}']
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI (複数期間)
        for rsi_period in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
        
        # ボリンジャーバンド（複数期間）
        for bb_period in [20, 50]:
            bb_middle = df['close'].rolling(bb_period).mean()
            bb_std = df['close'].rolling(bb_period).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            df[f'bb_position_{bb_period}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
            df[f'bb_width_{bb_period}'] = (bb_upper - bb_lower) / bb_middle
        
        # ボリューム特徴量
        for vol_ma in [10, 20, 50]:
            df[f'volume_ma_{vol_ma}'] = df['volume'].rolling(vol_ma).mean()
            df[f'volume_ratio_{vol_ma}'] = df['volume'] / df[f'volume_ma_{vol_ma}']
        
        df['log_volume'] = np.log(df['volume'] + 1)
        df['volume_price_trend'] = df['volume_ratio_20'] * df['returns']
        
        # モメンタム指標
        for momentum_period in [3, 5, 10, 20]:
            df[f'momentum_{momentum_period}'] = df['close'].pct_change(momentum_period)
        
        # 価格位置指標
        for lookback in [20, 50, 100]:
            df[f'price_percentile_{lookback}'] = df['close'].rolling(lookback).rank(pct=True)
        
        # トレンド強度
        df['trend_strength_short'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
        df['trend_strength_long'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        
        # 市場レジーム指標
        df['high_vol_regime'] = (df['vol_20'] > df['vol_20'].rolling(100).quantile(0.8)).astype(int)
        df['low_vol_regime'] = (df['vol_20'] < df['vol_20'].rolling(100).quantile(0.2)).astype(int)
        df['trending_market'] = (abs(df['trend_strength_short']) > df['trend_strength_short'].rolling(50).quantile(0.7)).astype(int)
        
        # 時間特徴量
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 最も重要な特徴量を選択（以前の成功事例に基づく）
        important_features = [
            'returns', 'log_returns', 'hl_ratio', 'oc_ratio',
            'return_1', 'return_3', 'return_5', 'return_10', 'return_20',
            'vol_5', 'vol_10', 'vol_20', 'vol_30',
            'vol_ratio_10', 'vol_ratio_20',
            'price_vs_sma_5', 'price_vs_sma_10', 'price_vs_sma_20', 'price_vs_sma_30',
            'price_vs_ema_5', 'price_vs_ema_12',
            'macd', 'macd_hist',
            'rsi_14', 'rsi_21',
            'bb_position_20', 'bb_width_20',
            'volume_ratio_10', 'volume_ratio_20',
            'log_volume', 'volume_price_trend',
            'momentum_3', 'momentum_5', 'momentum_10',
            'price_percentile_20', 'price_percentile_50',
            'trend_strength_short', 'trend_strength_long',
            'high_vol_regime', 'low_vol_regime', 'trending_market',
            'hour_sin', 'hour_cos', 'is_weekend'
        ]
        
        # 存在する特徴量のみを使用
        available_features = [f for f in important_features if f in df.columns]
        self.feature_names = available_features
        
        print(f"✅ {len(available_features)}個の重要特徴量を選択")
        
        return df

    def create_balanced_labels(self, data: pd.DataFrame, horizons: List[int] = [10, 15, 20]) -> pd.Series:
        """バランスの取れたラベル生成"""
        print(f"🎯 複数時間軸でのラベル生成中...")
        
        transaction_cost = 0.0008  # より現実的
        
        labels = []
        
        for horizon in horizons:
            future_return = data['close'].pct_change(horizon).shift(-horizon)
            
            # 方向性重視のラベル
            long_profit = future_return - transaction_cost
            short_profit = -future_return - transaction_cost
            
            # より現実的な閾値
            threshold = self.profit_threshold
            profitable = ((long_profit > threshold) | (short_profit > threshold))
            
            labels.append(profitable)
        
        # 複数時間軸での合意
        combined_labels = pd.concat(labels, axis=1).any(axis=1).astype(int)
        
        print(f"✅ ラベル生成完了")
        print(f"   陽性率: {combined_labels.mean():.2%}")
        print(f"   各時間軸の陽性率: {[f'{h}min: {l.mean():.2%}' for h, l in zip(horizons, labels)]}")
        
        return combined_labels

    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """訓練データを準備"""
        print("🗂️ 訓練データを準備中...")
        
        y = self.create_balanced_labels(data)
        X = data[self.feature_names].copy()
        
        # 有効なサンプルのみ使用
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # 欠損値処理
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)
        
        # データの品質チェック
        print(f"✅ 訓練データ準備完了")
        print(f"   サンプル数: {len(X):,}")
        print(f"   特徴量数: {len(X.columns)}")
        print(f"   陽性率: {y.mean():.2%}")
        print(f"   期間: {X.index.min()} から {X.index.max()}")
        
        return X, y

    def train_balanced_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """バランス調整されたモデル訓練"""
        print("🤖 バランス調整モデル訓練中...")
        
        # クラス重みを計算
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"クラス重み: {class_weight_dict}")
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=50,
                min_child_samples=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost': cb.CatBoostClassifier(
                iterations=500,
                learning_rate=0.03,
                depth=8,
                class_weights=class_weight_dict,
                random_seed=42,
                verbose=False
            )
        }
        
        cv_results = {}
        tscv = TimeSeriesSplit(n_splits=3, test_size=int(len(X)*0.2))
        
        for name, model in models.items():
            print(f"🔄 {name}を訓練中...")
            
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=1)
            
            model.fit(X, y)
            self.models[name] = model
            
            cv_results[name] = {
                'auc_mean': cv_scores.mean(),
                'auc_std': cv_scores.std()
            }
            
            print(f"   AUC: {cv_results[name]['auc_mean']:.3f}±{cv_results[name]['auc_std']:.3f}")
        
        # 加重アンサンブル作成
        print("🔄 加重アンサンブル作成中...")
        
        # 性能に基づく重み
        weights = [cv_results[name]['auc_mean'] for name in models.keys()]
        ensemble_models = [(name, model) for name, model in self.models.items()]
        
        ensemble = VotingClassifier(
            estimators=ensemble_models, 
            voting='soft'
        )
        ensemble.fit(X, y)
        self.models['ensemble'] = ensemble
        
        ensemble_cv_scores = cross_val_score(ensemble, X, y, cv=tscv, scoring='roc_auc', n_jobs=1)
        cv_results['ensemble'] = {
            'auc_mean': ensemble_cv_scores.mean(),
            'auc_std': ensemble_cv_scores.std()
        }
        
        print(f"   アンサンブルAUC: {cv_results['ensemble']['auc_mean']:.3f}±{cv_results['ensemble']['auc_std']:.3f}")
        
        return cv_results

    def comprehensive_evaluation(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """包括的な評価"""
        print("📊 包括的評価実行中...")
        
        best_model_name = max(self.models.keys(), 
                             key=lambda k: self.models[k].__class__.__name__ != 'NoneType' and
                             hasattr(self.models[k], 'predict_proba'))
        
        best_model = self.models.get('ensemble', self.models.get('catboost', self.models.get('random_forest')))
        
        y_proba = best_model.predict_proba(X)[:, 1]
        
        # 閾値別評価
        evaluation_results = {}
        
        for threshold in np.arange(0.3, 0.95, 0.05):
            threshold = round(threshold, 2)
            signals = (y_proba > threshold)
            
            if signals.sum() > 0:
                accuracy = y[signals].mean()
                signal_count = signals.sum()
                
                # 取引頻度計算
                total_hours = (X.index.max() - X.index.min()).total_seconds() / 3600
                signals_per_day = signal_count / (total_hours / 24)
                
                evaluation_results[threshold] = {
                    'signal_count': int(signal_count),
                    'accuracy': float(accuracy),
                    'signals_per_day': float(signals_per_day),
                    'precision': float(accuracy),
                    'expected_daily_trades': max(0, int(signals_per_day))
                }
            else:
                evaluation_results[threshold] = {
                    'signal_count': 0,
                    'accuracy': 0,
                    'signals_per_day': 0,
                    'precision': 0,
                    'expected_daily_trades': 0
                }
        
        return evaluation_results

    def save_improved_model(self, cv_results: Dict, model_dir: str = "models/improved_high_performance"):
        """改善されたモデルを保存"""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 全モデルを保存
        for name, model in self.models.items():
            model_file = model_path / f"{name}_model.pkl"
            joblib.dump(model, model_file)
            print(f"💾 {name}モデルを保存")
        
        # メタデータ
        import json
        from datetime import datetime
        
        best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]['auc_mean'])
        
        metadata = {
            "model_type": "improved_high_performance",
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "training_date": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_version": "3.1_improved",
            "best_model": best_model_name,
            "performance": cv_results,
            "config": {
                "profit_threshold": self.profit_threshold,
                "transaction_cost": 0.0008,
                "horizons": [10, 15, 20],
                "approach": "balanced_multi_horizon"
            }
        }
        
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata


def main():
    """メイン実行"""
    print("="*80)
    print("🚀 改善された高性能モデル訓練")
    print("="*80)
    
    model = ImprovedHighPerformanceModel(profit_threshold=0.003)
    
    # 1. データ読み込み
    data = model.load_market_data(limit=50000)
    if len(data) < 5000:
        print("❌ データ不足")
        return
    
    # 2. 特徴量生成
    data_with_features = model.create_comprehensive_features(data)
    
    # 3. 訓練データ準備
    X, y = model.prepare_training_data(data_with_features)
    if len(X) < 1000:
        print("❌ 訓練データ不足")
        return
    
    # 4. モデル訓練
    cv_results = model.train_balanced_models(X, y)
    
    # 5. 包括的評価
    evaluation_results = model.comprehensive_evaluation(X, y)
    
    # 6. モデル保存
    metadata = model.save_improved_model(cv_results)
    
    # 結果表示
    print("\n" + "="*80)
    print("📊 改善された高性能モデル結果")
    print("="*80)
    
    print(f"\n🔢 データ統計:")
    print(f"   サンプル数: {len(X):,}")
    print(f"   特徴量数: {len(X.columns)}")
    print(f"   陽性率: {y.mean():.2%}")
    print(f"   期間: {X.index.min().strftime('%Y-%m-%d')} から {X.index.max().strftime('%Y-%m-%d')}")
    
    print(f"\n🎯 モデル性能:")
    for model_name, scores in cv_results.items():
        status = "🏆" if scores['auc_mean'] == max(s['auc_mean'] for s in cv_results.values()) else "  "
        print(f"   {status} {model_name.upper():15}: AUC = {scores['auc_mean']:.3f}±{scores['auc_std']:.3f}")
    
    best_auc = max(cv_results[k]['auc_mean'] for k in cv_results.keys())
    
    print(f"\n📈 閾値別取引シミュレーション:")
    optimal_thresholds = []
    for threshold, result in evaluation_results.items():
        if result['signal_count'] > 0 and result['accuracy'] > 0.6:
            optimal_thresholds.append((threshold, result))
    
    # 上位5つの閾値を表示
    optimal_thresholds.sort(key=lambda x: x[1]['accuracy'], reverse=True)
    for threshold, result in optimal_thresholds[:5]:
        print(f"   閾値 {threshold}: {result['signal_count']}回取引, "
              f"精度 {result['accuracy']:.1%}, "
              f"1日 {result['expected_daily_trades']}回")
    
    # 最終評価
    if best_auc >= 0.75:
        print(f"\n✅ 優秀な性能 (AUC {best_auc:.3f}) - デプロイ推奨")
        deployment_status = "✅ デプロイ準備完了"
    elif best_auc >= 0.65:
        print(f"\n🟨 良好な性能 (AUC {best_auc:.3f}) - 条件付きで使用可能")
        deployment_status = "🟨 慎重にデプロイ"
    else:
        print(f"\n❌ 性能不足 (AUC {best_auc:.3f}) - さらなる改善必要")
        deployment_status = "❌ デプロイ不推奨"
    
    print(f"\n🎯 デプロイメント推奨: {deployment_status}")
    print(f"💾 保存場所: models/improved_high_performance/")
    
    return model, cv_results, evaluation_results


if __name__ == "__main__":
    model, cv_results, evaluation_results = main()