#!/usr/bin/env python3
"""
バランスの取れたデータセットを準備するスクリプト
クラス不均衡を解決し、Buy/Sell両方のパターンを適切に学習できるようにする
"""

import numpy as np
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BalancedDatasetPreparator:
    """バランスの取れたデータセットを作成"""
    
    def __init__(self, db_path: str = "data/historical_data.duckdb"):
        self.db_path = db_path
        self.conn = None
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'ICPUSDT', 'SOLUSDT', 'AVAXUSDT']
        self.lookback_days = 180  # 6ヶ月分
        
        # 拡張特徴量の定義
        self.feature_columns = [
            # 既存の44特徴量
            "returns", "log_returns", "hl_ratio", "oc_ratio", 
            "return_1", "return_3", "return_5", "return_10", "return_20",
            "vol_5", "vol_10", "vol_20", "vol_30",
            "vol_ratio_10", "vol_ratio_20",
            "price_vs_sma_5", "price_vs_sma_10", "price_vs_sma_20", "price_vs_sma_30",
            "price_vs_ema_5", "price_vs_ema_12",
            "macd", "macd_hist",
            "rsi_14", "rsi_21",
            "bb_position_20", "bb_width_20",
            "volume_ratio_10", "volume_ratio_20",
            "log_volume",
            "volume_price_trend",
            "momentum_3", "momentum_5", "momentum_10",
            "price_percentile_20", "price_percentile_50",
            "trend_strength_short", "trend_strength_long",
            "high_vol_regime", "low_vol_regime", "trending_market",
            "hour_sin", "hour_cos", "is_weekend",
            
            # 新規追加特徴量（8個）
            "buy_pressure",          # 買い圧力指標
            "sell_pressure",         # 売り圧力指標
            "volume_imbalance",      # ボリューム不均衡
            "price_acceleration",    # 価格加速度
            "volatility_regime",     # ボラティリティレジーム
            "trend_consistency",     # トレンド一貫性
            "market_efficiency",     # 市場効率性
            "liquidity_score"        # 流動性スコア
        ]
        
        self._connect_db()
    
    def _connect_db(self):
        """データベース接続"""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def load_raw_data(self) -> pd.DataFrame:
        """生データを読み込み"""
        all_data = []
        
        for symbol in self.symbols:
            logger.info(f"Loading data for {symbol}...")
            
            query = f"""
            SELECT 
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                turnover
            FROM all_klines
            WHERE symbol = '{symbol}'
                AND timestamp >= CURRENT_DATE - INTERVAL '{self.lookback_days}' DAY
            ORDER BY timestamp ASC
            """
            
            try:
                df = self.conn.execute(query).df()
                if len(df) > 0:
                    df['symbol'] = symbol
                    all_data.append(df)
                    logger.info(f"  Loaded {len(df)} records for {symbol}")
                else:
                    logger.warning(f"  No data found for {symbol}")
            except Exception as e:
                logger.error(f"  Error loading {symbol}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            logger.info(f"Total records loaded: {len(combined_df)}")
            return combined_df
        else:
            raise ValueError("No data loaded from database")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量エンジニアリング（拡張版）"""
        logger.info("Engineering features...")
        
        # シンボルごとに特徴量を計算
        feature_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df = symbol_df.sort_values('timestamp')
            
            # 基本的な価格特徴量
            symbol_df['returns'] = symbol_df['close'].pct_change()
            symbol_df['log_returns'] = np.log(symbol_df['close'] / symbol_df['close'].shift(1))
            symbol_df['hl_ratio'] = (symbol_df['high'] - symbol_df['low']) / symbol_df['close']
            symbol_df['oc_ratio'] = (symbol_df['close'] - symbol_df['open']) / symbol_df['close']
            
            # 多期間リターン
            for period in [1, 3, 5, 10, 20]:
                symbol_df[f'return_{period}'] = symbol_df['close'].pct_change(period)
            
            # ボラティリティ
            for period in [5, 10, 20, 30]:
                symbol_df[f'vol_{period}'] = symbol_df['returns'].rolling(period).std()
            
            # ボラティリティ比率
            symbol_df['vol_ratio_10'] = symbol_df['vol_10'] / symbol_df['vol_20']
            symbol_df['vol_ratio_20'] = symbol_df['vol_20'] / symbol_df['vol_30']
            
            # 移動平均
            for period in [5, 10, 20, 30]:
                sma = symbol_df['close'].rolling(period).mean()
                symbol_df[f'price_vs_sma_{period}'] = symbol_df['close'] / sma
            
            # EMA
            for period in [5, 12]:
                ema = symbol_df['close'].ewm(span=period).mean()
                symbol_df[f'price_vs_ema_{period}'] = symbol_df['close'] / ema
            
            # MACD
            ema_12 = symbol_df['close'].ewm(span=12).mean()
            ema_26 = symbol_df['close'].ewm(span=26).mean()
            symbol_df['macd'] = ema_12 - ema_26
            symbol_df['macd_signal'] = symbol_df['macd'].ewm(span=9).mean()
            symbol_df['macd_hist'] = symbol_df['macd'] - symbol_df['macd_signal']
            
            # RSI
            for period in [14, 21]:
                delta = symbol_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss
                symbol_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma_20 = symbol_df['close'].rolling(20).mean()
            std_20 = symbol_df['close'].rolling(20).std()
            symbol_df['bb_upper'] = sma_20 + 2 * std_20
            symbol_df['bb_lower'] = sma_20 - 2 * std_20
            symbol_df['bb_position_20'] = (symbol_df['close'] - symbol_df['bb_lower']) / (symbol_df['bb_upper'] - symbol_df['bb_lower'])
            symbol_df['bb_width_20'] = (symbol_df['bb_upper'] - symbol_df['bb_lower']) / sma_20
            
            # ボリューム特徴量
            symbol_df['volume_ratio_10'] = symbol_df['volume'] / symbol_df['volume'].rolling(10).mean()
            symbol_df['volume_ratio_20'] = symbol_df['volume'] / symbol_df['volume'].rolling(20).mean()
            symbol_df['log_volume'] = np.log(symbol_df['volume'] + 1)
            
            # Volume Price Trend
            symbol_df['volume_price_trend'] = (symbol_df['volume'] * symbol_df['returns']).cumsum()
            
            # モメンタム
            for period in [3, 5, 10]:
                symbol_df[f'momentum_{period}'] = symbol_df['close'].pct_change(period)
            
            # 価格パーセンタイル
            symbol_df['price_percentile_20'] = symbol_df['close'].rolling(20).apply(lambda x: (x.iloc[-1] > x).sum() / len(x))
            symbol_df['price_percentile_50'] = symbol_df['close'].rolling(50).apply(lambda x: (x.iloc[-1] > x).sum() / len(x))
            
            # トレンド強度
            symbol_df['trend_strength_short'] = abs(symbol_df['returns'].rolling(5).mean())
            symbol_df['trend_strength_long'] = abs(symbol_df['returns'].rolling(20).mean())
            
            # マーケットレジーム
            recent_vol = symbol_df['returns'].rolling(20).std()
            symbol_df['high_vol_regime'] = (recent_vol > recent_vol.quantile(0.7)).astype(float)
            symbol_df['low_vol_regime'] = (recent_vol < recent_vol.quantile(0.3)).astype(float)
            symbol_df['trending_market'] = (abs(symbol_df['returns'].rolling(10).mean()) > 0.001).astype(float)
            
            # 時間特徴量
            symbol_df['hour'] = symbol_df['timestamp'].dt.hour
            symbol_df['hour_sin'] = np.sin(2 * np.pi * symbol_df['hour'] / 24)
            symbol_df['hour_cos'] = np.cos(2 * np.pi * symbol_df['hour'] / 24)
            symbol_df['is_weekend'] = (symbol_df['timestamp'].dt.weekday >= 5).astype(float)
            
            # === 新規追加特徴量 ===
            
            # Buy/Sell圧力（価格の上昇/下落の勢い）
            symbol_df['buy_pressure'] = ((symbol_df['close'] - symbol_df['low']) / (symbol_df['high'] - symbol_df['low'])).rolling(5).mean()
            symbol_df['sell_pressure'] = ((symbol_df['high'] - symbol_df['close']) / (symbol_df['high'] - symbol_df['low'])).rolling(5).mean()
            
            # ボリューム不均衡（上昇時と下落時のボリューム差）
            up_volume = symbol_df.loc[symbol_df['returns'] > 0, 'volume'].rolling(10).sum()
            down_volume = symbol_df.loc[symbol_df['returns'] < 0, 'volume'].rolling(10).sum()
            symbol_df['volume_imbalance'] = (up_volume - down_volume) / (up_volume + down_volume)
            symbol_df['volume_imbalance'] = symbol_df['volume_imbalance'].fillna(0)
            
            # 価格加速度（モメンタムの変化率）
            symbol_df['price_acceleration'] = symbol_df['momentum_5'].diff()
            
            # ボラティリティレジーム（現在のボラティリティの相対的位置）
            vol_percentile = symbol_df['vol_20'].rolling(50).rank(pct=True)
            symbol_df['volatility_regime'] = vol_percentile.iloc[-1] if len(vol_percentile) > 0 else 0.5
            
            # トレンド一貫性（方向性の安定度）
            returns_sign = np.sign(symbol_df['returns'])
            symbol_df['trend_consistency'] = returns_sign.rolling(10).mean()
            
            # 市場効率性（ランダムウォークからの乖離）
            symbol_df['market_efficiency'] = abs(symbol_df['returns'].rolling(20).sum()) / symbol_df['returns'].rolling(20).std()
            
            # 流動性スコア（ボリュームとスプレッドから算出）
            symbol_df['liquidity_score'] = symbol_df['volume'] / (symbol_df['high'] - symbol_df['low'])
            symbol_df['liquidity_score'] = np.log(symbol_df['liquidity_score'] + 1)
            
            # ターゲット変数（1分後の価格方向）
            symbol_df['future_return'] = symbol_df['close'].shift(-1) / symbol_df['close'] - 1
            symbol_df['target'] = (symbol_df['future_return'] > 0).astype(int)
            
            feature_dfs.append(symbol_df)
        
        # 結合
        feature_df = pd.concat(feature_dfs, ignore_index=True)
        
        # 欠損値処理
        feature_df = feature_df.fillna(method='ffill').fillna(0)
        
        # 無限値の処理
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Feature engineering complete. Shape: {feature_df.shape}")
        
        return feature_df
    
    def analyze_class_balance(self, df: pd.DataFrame) -> Dict:
        """クラスバランスを分析"""
        logger.info("Analyzing class balance...")
        
        target_counts = df['target'].value_counts()
        balance_ratio = target_counts[1] / len(df) if 1 in target_counts else 0
        
        analysis = {
            'total_samples': len(df),
            'buy_samples': int(target_counts.get(1, 0)),
            'sell_samples': int(target_counts.get(0, 0)),
            'buy_ratio': balance_ratio,
            'sell_ratio': 1 - balance_ratio,
            'is_balanced': 0.4 <= balance_ratio <= 0.6
        }
        
        # シンボル別の分析
        symbol_balance = {}
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            symbol_target = symbol_df['target'].value_counts()
            symbol_ratio = symbol_target.get(1, 0) / len(symbol_df)
            symbol_balance[symbol] = {
                'buy_ratio': symbol_ratio,
                'sell_ratio': 1 - symbol_ratio,
                'total': len(symbol_df)
            }
        
        analysis['symbol_balance'] = symbol_balance
        
        logger.info(f"Overall balance - Buy: {analysis['buy_ratio']:.2%}, Sell: {analysis['sell_ratio']:.2%}")
        for symbol, balance in symbol_balance.items():
            logger.info(f"  {symbol} - Buy: {balance['buy_ratio']:.2%}, Sell: {balance['sell_ratio']:.2%}")
        
        return analysis
    
    def apply_balancing_techniques(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """複数のバランシング技術を適用"""
        logger.info("Applying balancing techniques...")
        
        # 元のクラス分布
        original_balance = np.mean(y)
        logger.info(f"Original Buy ratio: {original_balance:.2%}")
        
        # 1. SMOTE（オーバーサンプリング）でマイノリティクラスを増やす
        if original_balance < 0.4 or original_balance > 0.6:
            logger.info("Applying SMOTE...")
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            new_balance = np.mean(y_balanced)
            logger.info(f"After SMOTE - Buy ratio: {new_balance:.2%}, Samples: {len(y_balanced)}")
        else:
            X_balanced, y_balanced = X, y
        
        # 2. さらにアンダーサンプリングで完全にバランスを取る（オプション）
        if abs(0.5 - np.mean(y_balanced)) > 0.1:
            logger.info("Applying additional undersampling...")
            rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            X_final, y_final = rus.fit_resample(X_balanced, y_balanced)
            
            final_balance = np.mean(y_final)
            logger.info(f"After undersampling - Buy ratio: {final_balance:.2%}, Samples: {len(y_final)}")
        else:
            X_final, y_final = X_balanced, y_balanced
        
        return X_final, y_final
    
    def prepare_train_test_split(self, df: pd.DataFrame, test_ratio: float = 0.2) -> Dict:
        """時系列を考慮した訓練/テスト分割"""
        logger.info("Preparing train/test split...")
        
        # 時系列でソート
        df = df.sort_values('timestamp')
        
        # 分割点を計算
        split_idx = int(len(df) * (1 - test_ratio))
        
        # 訓練データとテストデータに分割
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # 特徴量とターゲットを分離
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values
        
        # 訓練データのみバランシング
        X_train_balanced, y_train_balanced = self.apply_balancing_techniques(X_train, y_train)
        
        logger.info(f"Train set: {len(X_train_balanced)} samples")
        logger.info(f"Test set: {len(X_test)} samples (original distribution maintained)")
        
        return {
            'X_train': X_train_balanced,
            'y_train': y_train_balanced,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_cols,
            'train_dates': (train_df['timestamp'].min(), train_df['timestamp'].max()),
            'test_dates': (test_df['timestamp'].min(), test_df['timestamp'].max())
        }
    
    def save_dataset(self, data: Dict, output_path: str = "data/balanced_dataset_v4.npz"):
        """データセットを保存"""
        logger.info(f"Saving dataset to {output_path}...")
        
        # NumPy形式で保存
        np.savez_compressed(
            output_path,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_test=data['X_test'],
            y_test=data['y_test']
        )
        
        # メタデータをJSON形式で保存
        metadata = {
            'feature_names': data['feature_names'],
            'train_dates': [str(d) for d in data['train_dates']],
            'test_dates': [str(d) for d in data['test_dates']],
            'train_samples': len(data['X_train']),
            'test_samples': len(data['X_test']),
            'train_buy_ratio': float(np.mean(data['y_train'])),
            'test_buy_ratio': float(np.mean(data['y_test'])),
            'n_features': len(data['feature_names']),
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = output_path.replace('.npz', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Dataset saved successfully!")
        
        # サマリーを表示
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Features: {metadata['n_features']}")
        print(f"Train samples: {metadata['train_samples']:,}")
        print(f"Test samples: {metadata['test_samples']:,}")
        print(f"Train Buy/Sell ratio: {metadata['train_buy_ratio']:.1%}/{100-metadata['train_buy_ratio']*100:.1%}")
        print(f"Test Buy/Sell ratio: {metadata['test_buy_ratio']:.1%}/{100-metadata['test_buy_ratio']*100:.1%}")
        print(f"Train period: {metadata['train_dates'][0]} to {metadata['train_dates'][1]}")
        print(f"Test period: {metadata['test_dates'][0]} to {metadata['test_dates'][1]}")
        print("="*60)
    
    def run(self):
        """メイン処理を実行"""
        try:
            # 1. データ読み込み
            raw_df = self.load_raw_data()
            
            # 2. 特徴量エンジニアリング
            feature_df = self.engineer_features(raw_df)
            
            # 3. クラスバランス分析
            balance_analysis = self.analyze_class_balance(feature_df)
            
            # 4. 訓練/テスト分割とバランシング
            dataset = self.prepare_train_test_split(feature_df)
            
            # 5. データセット保存
            self.save_dataset(dataset)
            
            logger.info("Dataset preparation complete!")
            
        except Exception as e:
            logger.error(f"Error in dataset preparation: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()


if __name__ == "__main__":
    # 出力ディレクトリの作成
    os.makedirs("data", exist_ok=True)
    
    # データセット準備を実行
    preparator = BalancedDatasetPreparator()
    preparator.run()