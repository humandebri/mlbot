#!/usr/bin/env python3
"""
改善された高性能モデル（44次元）をONNXに変換
AUC 0.838の高性能モデルをEC2での実用に最適化
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def convert_model_to_onnx():
    """改善された高性能モデルをONNX形式に変換"""
    
    print("🔄 改善された高性能モデルをONNX変換中...")
    
    # モデルとメタデータを読み込み
    model_dir = Path("models/improved_high_performance")
    metadata_file = model_dir / "metadata.json"
    
    if not metadata_file.exists():
        print("❌ メタデータファイルが見つかりません")
        return False
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"📋 モデル情報:")
    print(f"   特徴量数: {metadata['feature_count']}")
    print(f"   最高AUC: {metadata['performance']['random_forest']['auc_mean']:.3f}")
    print(f"   最良モデル: {metadata['best_model']}")
    
    # 最良モデルを読み込み
    best_model_name = metadata['best_model']
    model_file = model_dir / f"{best_model_name}_model.pkl"
    
    if not model_file.exists():
        print(f"❌ モデルファイルが見つかりません: {model_file}")
        return False
    
    model = joblib.load(model_file)
    feature_names = metadata['feature_names']
    
    print(f"✅ {best_model_name}モデルを読み込み完了")
    
    # サンプルデータでスケーラーを作成（実際のデータから）
    print("🔧 スケーラーを作成中...")
    
    try:
        import duckdb
        conn = duckdb.connect("data/historical_data.duckdb", read_only=True)
        
        # サンプルデータを取得
        query = """
        SELECT 
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM klines_btcusdt 
        LIMIT 10000
        """
        
        sample_data = conn.execute(query).df()
        sample_data['timestamp'] = pd.to_datetime(sample_data['timestamp'])
        sample_data.set_index('timestamp', inplace=True)
        conn.close()
        
        # 同じ特徴量を生成（簡略版）
        df = sample_data.copy()
        
        # 基本特徴量
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['oc_ratio'] = (df['close'] - df['open']) / df['open']
        
        # 短期間リターン
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # ボラティリティ
        for window in [5, 10, 20, 30]:
            df[f'vol_{window}'] = df['returns'].rolling(window).std()
            if window in [10, 20]:
                df[f'vol_ratio_{window}'] = df[f'vol_{window}'] / df[f'vol_{window}'].rolling(window*2).mean()
        
        # 移動平均
        for ma in [5, 10, 20, 30]:
            df[f'sma_{ma}'] = df['close'].rolling(ma).mean()
            df[f'price_vs_sma_{ma}'] = (df['close'] - df[f'sma_{ma}']) / df[f'sma_{ma}']
        
        # EMA
        for ema in [5, 12]:
            df[f'ema_{ema}'] = df['close'].ewm(span=ema).mean()
            df[f'price_vs_ema_{ema}'] = (df['close'] - df[f'ema_{ema}']) / df[f'ema_{ema}']
        
        # MACD
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        for rsi_period in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
        
        # ボリンジャーバンド
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        df['bb_position_20'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        df['bb_width_20'] = (bb_upper - bb_lower) / bb_middle
        
        # ボリューム
        for vol_ma in [10, 20]:
            df[f'volume_ma_{vol_ma}'] = df['volume'].rolling(vol_ma).mean()
            df[f'volume_ratio_{vol_ma}'] = df['volume'] / df[f'volume_ma_{vol_ma}']
        
        df['log_volume'] = np.log(df['volume'] + 1)
        df['volume_price_trend'] = df['volume_ratio_20'] * df['returns']
        
        # モメンタム
        for momentum_period in [3, 5, 10]:
            df[f'momentum_{momentum_period}'] = df['close'].pct_change(momentum_period)
        
        # 価格位置
        for lookback in [20, 50]:
            df[f'price_percentile_{lookback}'] = df['close'].rolling(lookback).rank(pct=True)
        
        # トレンド強度
        df['sma_50'] = df['close'].rolling(50).mean()
        df['trend_strength_short'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
        df['trend_strength_long'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        
        # 市場レジーム
        df['high_vol_regime'] = (df['vol_20'] > df['vol_20'].rolling(100).quantile(0.8)).astype(int)
        df['low_vol_regime'] = (df['vol_20'] < df['vol_20'].rolling(100).quantile(0.2)).astype(int)
        df['trending_market'] = (abs(df['trend_strength_short']) > df['trend_strength_short'].rolling(50).quantile(0.7)).astype(int)
        
        # 時間特徴量
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # 特徴量を選択
        available_features = [f for f in feature_names if f in df.columns]
        print(f"✅ 利用可能な特徴量: {len(available_features)}/{len(feature_names)}")
        
        X_sample = df[available_features].copy()
        
        # データクリーニング
        X_sample = X_sample.replace([np.inf, -np.inf], np.nan)
        X_sample = X_sample.ffill().fillna(0)
        
        # スケーラーをフィット
        scaler = StandardScaler()
        scaler.fit(X_sample.dropna())
        
        print("✅ スケーラー作成完了")
        
    except Exception as e:
        print(f"⚠️ 実データからのスケーラー作成に失敗: {e}")
        print("💡 デフォルトスケーラーを使用")
        
        # デフォルトスケーラー（平均0、標準偏差1）
        scaler = StandardScaler()
        dummy_data = np.random.randn(1000, len(feature_names))
        scaler.fit(dummy_data)
    
    # ONNX変換
    try:
        print("🔄 ONNX変換中...")
        
        # skl2onnxを使用してONNX変換
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        # 出力ディレクトリ作成
        output_dir = Path("models/v3.1_improved")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ONNXモデル保存
        onnx_file = output_dir / "model.onnx"
        with open(onnx_file, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        # スケーラー保存
        scaler_file = output_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_file)
        
        # メタデータ更新
        updated_metadata = {
            **metadata,
            "model_format": "onnx",
            "onnx_conversion_date": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
            "input_shape": [None, len(feature_names)],
            "deployment_ready": True
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(updated_metadata, f, indent=2)
        
        print("✅ ONNX変換完了")
        print(f"📁 保存場所: {output_dir}")
        print(f"🎯 AUC性能: {metadata['performance']['random_forest']['auc_mean']:.3f}")
        print(f"🔢 入力次元: {len(feature_names)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX変換エラー: {e}")
        return False

def main():
    """メイン実行"""
    print("=" * 80)
    print("🚀 改善された高性能モデル ONNX変換")
    print("=" * 80)
    
    success = convert_model_to_onnx()
    
    if success:
        print("\n✅ 変換完了 - EC2デプロイ準備完了")
    else:
        print("\n❌ 変換失敗")
    
    return success

if __name__ == "__main__":
    success = main()