#!/usr/bin/env python3
"""
統合された履歴データから バランスの取れたデータセットを準備
4年分のデータから多様な市場状況を学習
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
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BalancedDatasetPreparatorV2:
    """4年分のデータを活用したバランスデータセット作成"""
    
    def __init__(self, db_path: str = "data/historical_data.duckdb"):
        self.db_path = db_path
        self.conn = None
        
        # 使用するシンボル（実データがある3つ）
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']
        
        # 拡張特徴量（52次元）
        self.feature_columns = [
            # === 既存の44特徴量 ===
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
            
            # === 新規追加特徴量（8個）===
            "buy_pressure",          # 買い圧力
            "sell_pressure",         # 売り圧力
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
    
    def analyze_market_cycles(self) -> Dict:
        """市場サイクルを分析（ブル/ベア/横ばい）"""
        logger.info("Analyzing market cycles...")
        
        cycles = {}
        
        for symbol in self.symbols:
            # 月次リターンを計算
            monthly_data = self.conn.execute(f"""
                SELECT 
                    DATE_TRUNC('month', timestamp) as month,
                    FIRST(open) as month_open,
                    LAST(close) as month_close,
                    AVG(volume) as avg_volume
                FROM all_klines
                WHERE symbol = '{symbol}'
                GROUP BY DATE_TRUNC('month', timestamp)
                ORDER BY month
            """).df()
            
            if len(monthly_data) > 0:
                monthly_data['monthly_return'] = (
                    monthly_data['month_close'] / monthly_data['month_open'] - 1
                )
                
                # 市場状態を分類
                monthly_data['market_state'] = pd.cut(
                    monthly_data['monthly_return'],
                    bins=[-np.inf, -0.05, 0.05, np.inf],
                    labels=['bear', 'neutral', 'bull']
                )
                
                # サイクル統計
                state_counts = monthly_data['market_state'].value_counts()
                
                cycles[symbol] = {
                    'total_months': len(monthly_data),
                    'bull_months': int(state_counts.get('bull', 0)),
                    'bear_months': int(state_counts.get('bear', 0)),
                    'neutral_months': int(state_counts.get('neutral', 0)),
                    'avg_monthly_return': float(monthly_data['monthly_return'].mean()),
                    'volatility': float(monthly_data['monthly_return'].std())
                }
                
                logger.info(f"{symbol}: Bull {cycles[symbol]['bull_months']}m, "
                          f"Bear {cycles[symbol]['bear_months']}m, "
                          f"Neutral {cycles[symbol]['neutral_months']}m")
        
        return cycles
    
    def load_stratified_data(self) -> pd.DataFrame:
        """層化サンプリングでデータを読み込み"""
        logger.info("Loading data with stratified sampling...")
        
        all_data = []
        
        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")
            
            # データ総数を確認
            total_count = self.conn.execute(f"""
                SELECT COUNT(*) FROM all_klines WHERE symbol = '{symbol}'
            """).fetchone()[0]
            
            logger.info(f"  Total records: {total_count:,}")
            
            # サンプリング戦略：各年から均等にサンプル
            yearly_samples = []
            
            for year in [2021, 2022, 2023, 2024, 2025]:
                # 年別データ
                year_query = f"""
                SELECT *
                FROM all_klines
                WHERE symbol = '{symbol}'
                    AND YEAR(timestamp) = {year}
                ORDER BY timestamp
                """
                
                year_df = self.conn.execute(year_query).df()
                
                if len(year_df) > 0:
                    # 各年から最大50万レコードをサンプル（メモリ考慮）
                    if len(year_df) > 500000:
                        # 均等間隔でサンプリング
                        indices = np.linspace(0, len(year_df)-1, 500000, dtype=int)
                        year_df = year_df.iloc[indices]
                    
                    yearly_samples.append(year_df)
                    logger.info(f"    Year {year}: {len(year_df):,} records sampled")
            
            # 年別データを結合
            if yearly_samples:
                symbol_df = pd.concat(yearly_samples, ignore_index=True)
                symbol_df = symbol_df.sort_values('timestamp')
                all_data.append(symbol_df)
        
        # 全シンボルのデータを結合
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        
        logger.info(f"Total loaded records: {len(combined_df):,}")
        
        return combined_df
    
    def engineer_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """最適化された特徴量エンジニアリング"""
        logger.info("Engineering features (optimized for large dataset)...")
        
        feature_dfs = []
        
        # プログレスバー付きでシンボルごとに処理
        for symbol in tqdm(df['symbol'].unique(), desc="Processing symbols"):
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df = symbol_df.sort_values('timestamp')
            
            # 基本的な価格特徴量
            symbol_df['returns'] = symbol_df['close'].pct_change()
            symbol_df['log_returns'] = np.log(symbol_df['close'] / symbol_df['close'].shift(1))
            symbol_df['hl_ratio'] = (symbol_df['high'] - symbol_df['low']) / symbol_df['close']
            symbol_df['oc_ratio'] = (symbol_df['close'] - symbol_df['open']) / symbol_df['close']
            
            # 多期間リターン（ベクトル化）
            for period in [1, 3, 5, 10, 20]:
                symbol_df[f'return_{period}'] = symbol_df['close'].pct_change(period)
            
            # ボラティリティ（ベクトル化）
            returns = symbol_df['returns'].values
            for period in [5, 10, 20, 30]:
                symbol_df[f'vol_{period}'] = pd.Series(returns).rolling(period).std()
            
            # ボラティリティ比率
            symbol_df['vol_ratio_10'] = symbol_df['vol_10'] / symbol_df['vol_20']
            symbol_df['vol_ratio_20'] = symbol_df['vol_20'] / symbol_df['vol_30']
            
            # 移動平均（ベクトル化）
            close_prices = symbol_df['close']
            for period in [5, 10, 20, 30]:
                sma = close_prices.rolling(period).mean()
                symbol_df[f'price_vs_sma_{period}'] = close_prices / sma
            
            # EMA
            for period in [5, 12]:
                ema = close_prices.ewm(span=period, adjust=False).mean()
                symbol_df[f'price_vs_ema_{period}'] = close_prices / ema
            
            # MACD（ベクトル化）
            ema_12 = close_prices.ewm(span=12, adjust=False).mean()
            ema_26 = close_prices.ewm(span=26, adjust=False).mean()
            symbol_df['macd'] = ema_12 - ema_26
            symbol_df['macd_signal'] = symbol_df['macd'].ewm(span=9, adjust=False).mean()
            symbol_df['macd_hist'] = symbol_df['macd'] - symbol_df['macd_signal']
            
            # RSI（ベクトル化実装）
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            for period in [14, 21]:
                avg_gain = gain.rolling(period).mean()
                avg_loss = loss.rolling(period).mean()
                rs = avg_gain / avg_loss
                symbol_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma_20 = close_prices.rolling(20).mean()
            std_20 = close_prices.rolling(20).std()
            upper_band = sma_20 + 2 * std_20
            lower_band = sma_20 - 2 * std_20
            symbol_df['bb_position_20'] = (close_prices - lower_band) / (upper_band - lower_band)
            symbol_df['bb_width_20'] = (upper_band - lower_band) / sma_20
            
            # ボリューム特徴量
            volumes = symbol_df['volume']
            symbol_df['volume_ratio_10'] = volumes / volumes.rolling(10).mean()
            symbol_df['volume_ratio_20'] = volumes / volumes.rolling(20).mean()
            symbol_df['log_volume'] = np.log(volumes + 1)
            
            # Volume Price Trend
            symbol_df['volume_price_trend'] = (volumes * symbol_df['returns']).cumsum()
            
            # モメンタム
            for period in [3, 5, 10]:
                symbol_df[f'momentum_{period}'] = close_prices.pct_change(period)
            
            # 価格パーセンタイル（最適化版）
            for period in [20, 50]:
                rolling_window = close_prices.rolling(period)
                symbol_df[f'price_percentile_{period}'] = rolling_window.apply(
                    lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) == period else np.nan
                )
            
            # トレンド強度
            symbol_df['trend_strength_short'] = abs(symbol_df['returns'].rolling(5).mean())
            symbol_df['trend_strength_long'] = abs(symbol_df['returns'].rolling(20).mean())
            
            # マーケットレジーム
            recent_vol = symbol_df['returns'].rolling(20).std()
            vol_threshold_high = recent_vol.quantile(0.7)
            vol_threshold_low = recent_vol.quantile(0.3)
            symbol_df['high_vol_regime'] = (recent_vol > vol_threshold_high).astype(float)
            symbol_df['low_vol_regime'] = (recent_vol < vol_threshold_low).astype(float)
            symbol_df['trending_market'] = (abs(symbol_df['returns'].rolling(10).mean()) > 0.001).astype(float)
            
            # 時間特徴量
            symbol_df['hour'] = symbol_df['timestamp'].dt.hour
            hour_angle = 2 * np.pi * symbol_df['hour'] / 24
            symbol_df['hour_sin'] = np.sin(hour_angle)
            symbol_df['hour_cos'] = np.cos(hour_angle)
            symbol_df['is_weekend'] = (symbol_df['timestamp'].dt.weekday >= 5).astype(float)
            
            # === 新規特徴量 ===
            
            # Buy/Sell圧力
            price_range = symbol_df['high'] - symbol_df['low']
            symbol_df['buy_pressure'] = ((symbol_df['close'] - symbol_df['low']) / price_range).rolling(5).mean()
            symbol_df['sell_pressure'] = ((symbol_df['high'] - symbol_df['close']) / price_range).rolling(5).mean()
            
            # ボリューム不均衡
            up_mask = symbol_df['returns'] > 0
            down_mask = symbol_df['returns'] < 0
            up_volume = symbol_df.loc[up_mask, 'volume'].rolling(10).sum()
            down_volume = symbol_df.loc[down_mask, 'volume'].rolling(10).sum()
            
            # 全インデックスに拡張
            up_volume = up_volume.reindex(symbol_df.index, method='ffill')
            down_volume = down_volume.reindex(symbol_df.index, method='ffill')
            
            total_volume = up_volume + down_volume
            symbol_df['volume_imbalance'] = np.where(
                total_volume > 0,
                (up_volume - down_volume) / total_volume,
                0
            )
            
            # 価格加速度
            symbol_df['price_acceleration'] = symbol_df['momentum_5'].diff()
            
            # ボラティリティレジーム
            vol_rank = symbol_df['vol_20'].rolling(50).rank(pct=True)
            symbol_df['volatility_regime'] = vol_rank
            
            # トレンド一貫性
            returns_sign = np.sign(symbol_df['returns'])
            symbol_df['trend_consistency'] = returns_sign.rolling(10).mean()
            
            # 市場効率性
            abs_return_sum = abs(symbol_df['returns'].rolling(20).sum())
            return_std = symbol_df['returns'].rolling(20).std()
            symbol_df['market_efficiency'] = np.where(
                return_std > 0,
                abs_return_sum / return_std,
                0
            )
            
            # 流動性スコア
            symbol_df['liquidity_score'] = np.log(volumes / price_range + 1)
            
            # ターゲット変数
            symbol_df['future_return'] = symbol_df['close'].shift(-1) / symbol_df['close'] - 1
            symbol_df['target'] = (symbol_df['future_return'] > 0).astype(int)
            
            feature_dfs.append(symbol_df)
        
        # 結合
        feature_df = pd.concat(feature_dfs, ignore_index=True)
        
        # 欠損値処理（前方補完後、ゼロ埋め）
        feature_df = feature_df.fillna(method='ffill').fillna(0)
        
        # 無限値の処理
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Feature engineering complete. Shape: {feature_df.shape}")
        
        return feature_df
    
    def create_temporal_splits(self, df: pd.DataFrame) -> List[Dict]:
        """時系列を考慮した複数の訓練/検証セット作成"""
        logger.info("Creating temporal splits...")
        
        # 時系列でソート
        df = df.sort_values('timestamp')
        
        # Walk-forward分割（3つの期間）
        splits = []
        
        # Split 1: 2021-2022年訓練、2023年検証
        split1_train = df[df['timestamp'] < '2023-01-01']
        split1_val = df[(df['timestamp'] >= '2023-01-01') & (df['timestamp'] < '2024-01-01')]
        
        if len(split1_train) > 0 and len(split1_val) > 0:
            splits.append({
                'name': 'split_2021_2022',
                'train': split1_train,
                'val': split1_val
            })
        
        # Split 2: 2021-2023年訓練、2024年検証
        split2_train = df[df['timestamp'] < '2024-01-01']
        split2_val = df[(df['timestamp'] >= '2024-01-01') & (df['timestamp'] < '2025-01-01')]
        
        if len(split2_train) > 0 and len(split2_val) > 0:
            splits.append({
                'name': 'split_2021_2023',
                'train': split2_train,
                'val': split2_val
            })
        
        # Split 3: 2021-2024年訓練、2025年検証
        split3_train = df[df['timestamp'] < '2025-01-01']
        split3_val = df[df['timestamp'] >= '2025-01-01']
        
        if len(split3_train) > 0 and len(split3_val) > 0:
            splits.append({
                'name': 'split_2021_2024',
                'train': split3_train,
                'val': split3_val
            })
        
        # 各分割の統計を表示
        for split in splits:
            train_buy_ratio = split['train']['target'].mean()
            val_buy_ratio = split['val']['target'].mean()
            
            logger.info(f"\n{split['name']}:")
            logger.info(f"  Train: {len(split['train']):,} samples, Buy ratio: {train_buy_ratio:.2%}")
            logger.info(f"  Val: {len(split['val']):,} samples, Buy ratio: {val_buy_ratio:.2%}")
        
        return splits
    
    def apply_advanced_balancing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """高度なバランシング技術を適用"""
        logger.info("Applying advanced balancing...")
        
        original_ratio = np.mean(y)
        logger.info(f"Original Buy ratio: {original_ratio:.2%}")
        
        # 目標比率を計算（45-55%の範囲を目指す）
        target_ratio = 0.5
        
        if 0.45 <= original_ratio <= 0.55:
            logger.info("Data is already well balanced")
            return X, y
        
        # SMOTEでバランシング
        try:
            # サンプリング戦略を設定
            if original_ratio < 0.45:
                # Buyが少ない場合
                sampling_strategy = {1: int(len(y) * target_ratio), 0: int(len(y) * (1-target_ratio))}
            else:
                # Sellが少ない場合
                sampling_strategy = {0: int(len(y) * (1-target_ratio)), 1: int(len(y) * target_ratio)}
            
            smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            new_ratio = np.mean(y_balanced)
            logger.info(f"After SMOTE - Buy ratio: {new_ratio:.2%}, Samples: {len(y_balanced):,}")
            
            # さらに微調整が必要な場合
            if abs(new_ratio - 0.5) > 0.05:
                rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
                X_final, y_final = rus.fit_resample(X_balanced, y_balanced)
                
                final_ratio = np.mean(y_final)
                logger.info(f"After fine-tuning - Buy ratio: {final_ratio:.2%}, Samples: {len(y_final):,}")
                
                return X_final, y_final
            else:
                return X_balanced, y_balanced
                
        except Exception as e:
            logger.error(f"Balancing error: {e}")
            return X, y
    
    def prepare_final_dataset(self, df: pd.DataFrame) -> Dict:
        """最終的なデータセットを準備"""
        logger.info("Preparing final dataset...")
        
        # 最新の分割を使用（2021-2024訓練、2025検証）
        train_df = df[df['timestamp'] < '2025-01-01']
        test_df = df[df['timestamp'] >= '2025-01-01']
        
        # 特徴量とターゲットを分離
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values
        
        # 訓練データのバランシング
        X_train_balanced, y_train_balanced = self.apply_advanced_balancing(X_train, y_train)
        
        # 検証セット作成（バランス済み訓練データの20%）
        val_split = int(len(X_train_balanced) * 0.8)
        X_val = X_train_balanced[val_split:]
        y_val = y_train_balanced[val_split:]
        X_train_final = X_train_balanced[:val_split]
        y_train_final = y_train_balanced[:val_split]
        
        logger.info(f"\nFinal dataset:")
        logger.info(f"  Train: {len(X_train_final):,} samples, Buy ratio: {np.mean(y_train_final):.2%}")
        logger.info(f"  Val: {len(X_val):,} samples, Buy ratio: {np.mean(y_val):.2%}")
        logger.info(f"  Test: {len(X_test):,} samples, Buy ratio: {np.mean(y_test):.2%}")
        
        return {
            'X_train': X_train_final,
            'y_train': y_train_final,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_cols,
            'train_dates': (train_df['timestamp'].min(), train_df['timestamp'].max()),
            'test_dates': (test_df['timestamp'].min(), test_df['timestamp'].max())
        }
    
    def save_dataset(self, data: Dict, output_path: str = "data/balanced_dataset_v4_full.npz"):
        """データセットを保存"""
        logger.info(f"Saving dataset to {output_path}...")
        
        # NumPy形式で保存
        np.savez_compressed(
            output_path,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            X_test=data['X_test'],
            y_test=data['y_test']
        )
        
        # メタデータ
        metadata = {
            'feature_names': data['feature_names'],
            'n_features': len(data['feature_names']),
            'train_dates': [str(d) for d in data['train_dates']],
            'test_dates': [str(d) for d in data['test_dates']],
            'train_samples': len(data['X_train']),
            'val_samples': len(data['X_val']),
            'test_samples': len(data['X_test']),
            'train_buy_ratio': float(np.mean(data['y_train'])),
            'val_buy_ratio': float(np.mean(data['y_val'])),
            'test_buy_ratio': float(np.mean(data['y_test'])),
            'created_at': datetime.now().isoformat(),
            'data_source': '4+ years of historical data (2021-2025)'
        }
        
        metadata_path = output_path.replace('.npz', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # サマリー表示
        print("\n" + "="*60)
        print("DATASET PREPARATION COMPLETE")
        print("="*60)
        print(f"Features: {metadata['n_features']} dimensions")
        print(f"Train: {metadata['train_samples']:,} samples ({metadata['train_buy_ratio']:.1%} buy)")
        print(f"Val: {metadata['val_samples']:,} samples ({metadata['val_buy_ratio']:.1%} buy)")
        print(f"Test: {metadata['test_samples']:,} samples ({metadata['test_buy_ratio']:.1%} buy)")
        print(f"Training period: {metadata['train_dates'][0]} to {metadata['train_dates'][1]}")
        print(f"Test period: {metadata['test_dates'][0]} to {metadata['test_dates'][1]}")
        print("="*60)
        
        logger.info("Dataset saved successfully!")
    
    def run(self):
        """メイン処理"""
        try:
            # 1. 市場サイクル分析
            market_cycles = self.analyze_market_cycles()
            
            # 2. 層化サンプリングでデータ読み込み
            raw_df = self.load_stratified_data()
            
            # 3. 特徴量エンジニアリング
            feature_df = self.engineer_features_optimized(raw_df)
            
            # 4. 時系列分割の作成と分析
            temporal_splits = self.create_temporal_splits(feature_df)
            
            # 5. 最終データセット準備
            final_dataset = self.prepare_final_dataset(feature_df)
            
            # 6. データセット保存
            self.save_dataset(final_dataset)
            
            logger.info("\n✅ Dataset preparation complete!")
            
        except Exception as e:
            logger.error(f"Error in dataset preparation: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if self.conn:
                self.conn.close()


if __name__ == "__main__":
    # 出力ディレクトリ確認
    os.makedirs("data", exist_ok=True)
    
    # データセット準備を実行
    preparator = BalancedDatasetPreparatorV2()
    preparator.run()