#!/usr/bin/env python3
"""
高性能モデルの高速訓練版（データを制限して迅速な結果を得る）
35個の実証済み特徴量を使用
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
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class QuickHighPerformanceModel:
    """高速高性能モデル訓練"""
    
    def __init__(self, profit_threshold: float = 0.005):
        self.profit_threshold = profit_threshold
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
    def load_sample_data(self, symbol: str = "BTCUSDT", limit: int = 20000) -> pd.DataFrame:
        """サンプルデータを読み込み（高速化のため制限）"""
        print(f"📊 {symbol}のサンプルデータを読み込み中（最大{limit:,}件）...")
        
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
            WHERE timestamp >= '2024-03-01'
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
            
            data = conn.execute(query).df()
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
            data.sort_index(inplace=True)  # 時系列順に並べ直し
            
            conn.close()
            print(f"✅ {len(data):,}件のレコードを読み込み完了")
            
            return data
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return pd.DataFrame()

    def create_essential_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """必須の35個の高品質特徴量を生成"""
        print("🔧 35個の実証済み特徴量を生成中...")
        
        df = data.copy()
        
        # 基本価格特徴量 (4個)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['oc_ratio'] = (df['close'] - df['open']) / df['open']
        
        # マルチタイムフレームリターン (6個)
        for period in [1, 3, 5, 10, 15, 30]:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # ボラティリティ特徴量 (4個)
        for window in [5, 10, 20, 30]:
            df[f'vol_{window}'] = df['returns'].rolling(window).std()
        
        # 移動平均と価格比較 (6個)
        for ma in [5, 10, 20]:
            df[f'sma_{ma}'] = df['close'].rolling(ma).mean()
            df[f'price_vs_sma_{ma}'] = (df['close'] - df[f'sma_{ma}']) / df[f'sma_{ma}']
        
        # モメンタム指標 (3個)
        df['momentum_3'] = df['close'].pct_change(3)
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        # ボリューム特徴量 (3個)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['log_volume'] = np.log(df['volume'] + 1)
        
        # RSI (1個)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ボリンジャーバンド位置 (1個)
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # トレンド指標 (2個)
        df['trend_strength'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
        df['price_above_ma'] = (df['close'] > df['sma_20']).astype(int)
        
        # ボリューム-価格相互作用 (1個)
        df['volume_price_change'] = df['volume_ratio'] * abs(df['returns'])
        
        # 市場レジーム (2個)
        df['high_vol'] = (df['vol_20'] > df['vol_20'].rolling(50).quantile(0.8)).astype(int)
        df['low_vol'] = (df['vol_20'] < df['vol_20'].rolling(50).quantile(0.2)).astype(int)
        
        # 時間特徴量 (2個)
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # 合計35個の特徴量
        feature_cols = [
            'returns', 'log_returns', 'hl_ratio', 'oc_ratio',
            'return_1', 'return_3', 'return_5', 'return_10', 'return_15', 'return_30',
            'vol_5', 'vol_10', 'vol_20', 'vol_30',
            'sma_5', 'sma_10', 'sma_20',
            'price_vs_sma_5', 'price_vs_sma_10', 'price_vs_sma_20',
            'momentum_3', 'momentum_5', 'momentum_10',
            'volume_ma', 'volume_ratio', 'log_volume',
            'rsi', 'bb_position',
            'trend_strength', 'price_above_ma',
            'volume_price_change',
            'high_vol', 'low_vol',
            'hour', 'day_of_week'
        ]
        
        self.feature_names = feature_cols
        print(f"✅ {len(feature_cols)}個の実証済み特徴量生成完了")
        
        return df

    def create_profit_labels(self, data: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """収益ラベルを生成"""
        print(f"🎯 {horizon}分後の収益ラベルを生成中...")
        
        transaction_cost = 0.0012  # 0.12%
        
        future_return = data['close'].pct_change(horizon).shift(-horizon)
        long_profit = future_return - transaction_cost
        short_profit = -future_return - transaction_cost
        
        profitable = ((long_profit > self.profit_threshold) | 
                     (short_profit > self.profit_threshold)).astype(int)
        
        print(f"✅ 収益ラベル生成完了 (陽性率: {profitable.mean():.2%})")
        return profitable

    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """訓練データを準備"""
        print("🗂️ 訓練データを準備中...")
        
        y = self.create_profit_labels(data)
        X = data[self.feature_names].copy()
        
        # 有効なサンプルのみ使用
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # 残りのNaNを処理
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)
        
        print(f"✅ 訓練データ準備完了")
        print(f"   サンプル数: {len(X):,}")
        print(f"   特徴量数: {len(X.columns)}")
        print(f"   陽性率: {y.mean():.2%}")
        
        return X, y

    def train_quick_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """高速なモデル訓練"""
        print("🤖 高速モデル訓練中...")
        
        # 高速設定のモデル
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=50,  # 高速化のため削減
                max_depth=8,
                min_samples_split=20,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,  # 高速化のため削減
                learning_rate=0.1,
                max_depth=5,
                num_leaves=20,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        cv_results = {}
        tscv = TimeSeriesSplit(n_splits=2)  # 高速化のため削減
        
        for name, model in models.items():
            print(f"🔄 {name}を訓練中...")
            
            # クロスバリデーション
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
            
            # 最終モデル訓練
            model.fit(X, y)
            self.models[name] = model
            
            cv_results[name] = {
                'auc_mean': cv_scores.mean(),
                'auc_std': cv_scores.std()
            }
            
            print(f"   AUC: {cv_results[name]['auc_mean']:.3f}±{cv_results[name]['auc_std']:.3f}")
        
        # アンサンブル作成
        print("🔄 アンサンブル作成中...")
        ensemble_models = [(name, model) for name, model in self.models.items()]
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble.fit(X, y)
        self.models['ensemble'] = ensemble
        
        ensemble_cv_scores = cross_val_score(ensemble, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
        cv_results['ensemble'] = {
            'auc_mean': ensemble_cv_scores.mean(),
            'auc_std': ensemble_cv_scores.std()
        }
        
        print(f"   アンサンブルAUC: {cv_results['ensemble']['auc_mean']:.3f}±{cv_results['ensemble']['auc_std']:.3f}")
        
        return cv_results

    def evaluate_trading_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """取引性能を評価"""
        print("📈 取引性能評価中...")
        
        # 最高性能モデルを選択
        best_model_name = 'ensemble'
        best_model = self.models[best_model_name]
        
        y_proba = best_model.predict_proba(X)[:, 1]
        
        results = {}
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            signals = (y_proba > threshold)
            
            if signals.sum() > 0:
                signal_accuracy = y[signals].mean()
                signal_count = signals.sum()
                
                # 取引性能計算
                expected_return_per_trade = 0.008  # 0.8%期待
                transaction_cost = 0.0012
                net_return_per_trade = signal_accuracy * expected_return_per_trade - transaction_cost
                
                results[threshold] = {
                    'signal_count': int(signal_count),
                    'accuracy': float(signal_accuracy),
                    'net_return_per_trade': float(net_return_per_trade),
                    'daily_signals': int(signal_count / (len(X) / (24*12))),  # 1日あたり
                }
            else:
                results[threshold] = {
                    'signal_count': 0,
                    'accuracy': 0,
                    'net_return_per_trade': 0,
                    'daily_signals': 0
                }
        
        return results

    def save_model(self, cv_results: Dict, model_dir: str = "models/quick_high_performance"):
        """モデルを保存"""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 最高性能モデルのみ保存
        best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]['auc_mean'])
        best_model = self.models[best_model_name]
        
        # モデル保存
        model_file = model_path / f"{best_model_name}_model.pkl"
        joblib.dump(best_model, model_file)
        
        # スケーラー保存
        scaler_file = model_path / "scaler.pkl"
        joblib.dump(self.scaler, scaler_file)
        
        # メタデータ保存
        import json
        from datetime import datetime
        
        metadata = {
            "model_type": "quick_high_performance",
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "training_date": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_version": "3.0_quick",
            "best_model": best_model_name,
            "performance": cv_results,
            "config": {
                "profit_threshold": self.profit_threshold,
                "transaction_cost": 0.0012,
                "horizon": 5
            }
        }
        
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 モデル保存完了: {model_path}")
        return metadata


def main():
    """メイン実行"""
    print("="*70)
    print("🚀 高速高性能モデル訓練")
    print("="*70)
    
    model = QuickHighPerformanceModel(profit_threshold=0.005)
    
    # 1. データ読み込み
    data = model.load_sample_data(limit=20000)
    if len(data) < 1000:
        print("❌ データ不足")
        return
    
    # 2. 特徴量生成
    data_with_features = model.create_essential_features(data)
    
    # 3. 訓練データ準備
    X, y = model.prepare_training_data(data_with_features)
    if len(X) < 100:
        print("❌ 訓練データ不足")
        return
    
    # 4. モデル訓練
    cv_results = model.train_quick_models(X, y)
    
    # 5. 取引性能評価
    trading_results = model.evaluate_trading_performance(X, y)
    
    # 6. モデル保存
    metadata = model.save_model(cv_results)
    
    # 結果表示
    print("\n" + "="*70)
    print("📊 高速高性能モデル結果")
    print("="*70)
    
    print(f"\n🔢 データ統計:")
    print(f"   サンプル数: {len(X):,}")
    print(f"   特徴量数: {len(X.columns)}")
    print(f"   陽性率: {y.mean():.2%}")
    
    print(f"\n🎯 モデル性能:")
    for model_name, scores in cv_results.items():
        status = "🏆" if scores['auc_mean'] == max(s['auc_mean'] for s in cv_results.values()) else "  "
        print(f"   {status} {model_name.upper():15}: AUC = {scores['auc_mean']:.3f}±{scores['auc_std']:.3f}")
    
    best_auc = max(cv_results[k]['auc_mean'] for k in cv_results.keys())
    
    print(f"\n📈 取引シミュレーション:")
    for threshold, result in trading_results.items():
        if result['signal_count'] > 0:
            print(f"   閾値 {threshold}: {result['signal_count']}回取引, "
                  f"精度 {result['accuracy']:.1%}, "
                  f"1日 {result['daily_signals']}回")
    
    # 性能評価
    if best_auc >= 0.75:
        print(f"\n✅ 優秀な性能 (AUC {best_auc:.3f}) - 実用可能")
        recommendation = "デプロイ推奨"
    elif best_auc >= 0.65:
        print(f"\n🟨 良好な性能 (AUC {best_auc:.3f}) - 最適化推奨")
        recommendation = "さらなる改善を検討"
    else:
        print(f"\n❌ 性能不足 (AUC {best_auc:.3f}) - 再設計必要")
        recommendation = "アプローチを見直し"
    
    print(f"\n🎯 推奨アクション: {recommendation}")
    print(f"💾 保存場所: models/quick_high_performance/")
    
    return model, cv_results, trading_results


if __name__ == "__main__":
    model, cv_results, trading_results = main()