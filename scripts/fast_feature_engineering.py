#!/usr/bin/env python3
"""
Fast and focused feature engineering for profitability prediction.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Tuple
import talib
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class FastFeatureEngineer:
    """Fast feature engineering focused on profitability."""
    
    def __init__(self):
        pass
        
    def load_sample_data(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Load smaller sample of data for faster processing."""
        logger.info(f"Loading sample data for {symbol}")
        
        conn = duckdb.connect("data/historical_data.duckdb")
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
          AND timestamp <= '2024-02-15'  -- 1.5 months for speed
        ORDER BY timestamp
        """
        
        data = conn.execute(query).df()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        conn.close()
        
        logger.info(f"Loaded {len(data)} records")
        return data
    
    def create_profit_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create optimized profit prediction targets."""
        
        # Future returns with transaction costs
        transaction_cost = 0.0012  # 0.12% round-trip
        
        for horizon in [3, 5, 10]:
            # Raw future returns
            data[f'future_return_{horizon}m'] = data['close'].pct_change(horizon).shift(-horizon)
            
            # Profitable trades (after costs)
            long_profit = data[f'future_return_{horizon}m'] - transaction_cost
            short_profit = -data[f'future_return_{horizon}m'] - transaction_cost
            
            # Binary profitability targets
            data[f'profitable_long_{horizon}m'] = (long_profit > 0.003).astype(int)  # >0.3% profit
            data[f'profitable_short_{horizon}m'] = (short_profit > 0.003).astype(int)
            data[f'profitable_either_{horizon}m'] = ((long_profit > 0.003) | (short_profit > 0.003)).astype(int)
            
            # Strong profit targets (better risk/reward)
            data[f'strong_profit_{horizon}m'] = ((long_profit > 0.008) | (short_profit > 0.008)).astype(int)
        
        return data
    
    def create_essential_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create most essential features for profit prediction."""
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        # Basic price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Multi-timeframe returns
        for period in [1, 3, 5, 10, 15, 30]:
            data[f'ret_{period}'] = data['close'].pct_change(period)
        
        # Volatility features
        for window in [5, 10, 20]:
            data[f'vol_{window}'] = data['returns'].rolling(window).std()
            data[f'vol_rank_{window}'] = data[f'vol_{window}'].rolling(window*2).rank(pct=True)
        
        # Essential technical indicators
        data['rsi'] = talib.RSI(close, timeperiod=14)
        data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
        data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
        
        # MACD
        data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(close)
        data['macd_bullish'] = (data['macd'] > data['macd_signal']).astype(int)
        
        # Bollinger Bands
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(close)
        data['bb_position'] = (close - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-8)
        data['bb_squeeze'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Moving averages
        data['sma_10'] = talib.SMA(close, timeperiod=10)
        data['sma_20'] = talib.SMA(close, timeperiod=20)
        data['ema_12'] = talib.EMA(close, timeperiod=12)
        
        # Price position relative to MAs
        data['price_vs_sma10'] = (data['close'] - data['sma_10']) / data['sma_10']
        data['price_vs_sma20'] = (data['close'] - data['sma_20']) / data['sma_20']
        
        # Trend strength
        data['sma_trend'] = (data['sma_10'] - data['sma_20']) / data['sma_20']
        
        # Volume features
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['volume_spike'] = (data['volume_ratio'] > 2).astype(int)
        
        # Volume-Price relationship
        data['vpt'] = (data['volume'] * data['returns']).rolling(10).sum()
        data['obv'] = talib.OBV(close, volume)
        data['obv_trend'] = data['obv'].pct_change(10)
        
        # Price action patterns
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['body_ratio'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
        
        # Gap analysis
        data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        data['gap_up'] = (data['gap'] > 0.002).astype(int)
        data['gap_down'] = (data['gap'] < -0.002).astype(int)
        
        # Momentum features
        for period in [3, 5, 10]:
            data[f'momentum_{period}'] = data['close'].pct_change(period)
            data[f'acceleration_{period}'] = data[f'momentum_{period}'] - data[f'momentum_{period}'].shift(period)
        
        # Market regime indicators
        data['high_vol_regime'] = (data['vol_20'] > data['vol_20'].rolling(50).quantile(0.8)).astype(int)
        data['trending_regime'] = (abs(data['sma_trend']) > 0.01).astype(int)
        
        # Mean reversion indicators
        data['mean_reversion_10'] = (data['close'] - data['sma_10']) / data['vol_10']
        data['mean_reversion_20'] = (data['close'] - data['sma_20']) / data['vol_20']
        
        # Temporal features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['is_market_hours'] = ((data['hour'] >= 9) & (data['hour'] <= 16)).astype(int)
        
        return data
    
    def analyze_feature_importance(self, data: pd.DataFrame) -> Dict:
        """Analyze which features predict profitability best."""
        
        logger.info("Analyzing feature importance for profitability")
        
        # Get feature columns
        feature_cols = [col for col in data.columns if not col.startswith(('future_', 'profitable_', 'strong_'))]
        feature_cols = [col for col in feature_cols if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        results = {}
        
        # Test different profit targets
        targets = ['profitable_either_3m', 'profitable_either_5m', 'profitable_either_10m', 'strong_profit_5m']
        
        for target in targets:
            if target not in data.columns:
                continue
                
            logger.info(f"Analyzing features for {target}")
            
            # Prepare data
            X = data[feature_cols].copy()
            y = data[target].copy()
            
            # Remove NaN rows
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 100:
                logger.warning(f"Insufficient data for {target}")
                continue
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # Calculate mutual information
            try:
                mi_scores = mutual_info_regression(X, y, random_state=42)
                
                # Calculate correlations
                correlations = []
                for col in X.columns:
                    try:
                        corr = abs(np.corrcoef(X[col], y)[0, 1])
                        correlations.append(corr if not np.isnan(corr) else 0)
                    except:
                        correlations.append(0)
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'mutual_info': mi_scores,
                    'correlation': correlations,
                    'combined_score': mi_scores + np.array(correlations)
                }).sort_values('combined_score', ascending=False)
                
                # Calculate baseline statistics
                positive_rate = y.mean()
                total_samples = len(y)
                
                results[target] = {
                    'importance_df': importance_df,
                    'positive_rate': positive_rate,
                    'total_samples': total_samples,
                    'top_features': importance_df.head(15)['feature'].tolist()
                }
                
                logger.info(f"{target}: {positive_rate:.1%} positive rate, {total_samples} samples")
                
            except Exception as e:
                logger.error(f"Error analyzing {target}: {e}")
                continue
        
        return results
    
    def create_optimized_feature_set(self, data: pd.DataFrame, analysis_results: Dict) -> pd.DataFrame:
        """Create optimized feature set based on analysis."""
        
        # Get all top features across targets
        all_top_features = set()
        for target, results in analysis_results.items():
            all_top_features.update(results['top_features'])
        
        # Add essential features
        essential_features = [
            'returns', 'ret_1', 'ret_3', 'ret_5', 'ret_10',
            'vol_5', 'vol_10', 'vol_20',
            'rsi', 'macd_hist', 'bb_position',
            'price_vs_sma10', 'price_vs_sma20', 'sma_trend',
            'volume_ratio', 'vpt', 'obv_trend',
            'momentum_3', 'momentum_5', 'momentum_10',
            'high_vol_regime', 'trending_regime'
        ]
        
        # Combine and filter existing features
        selected_features = list(all_top_features.union(essential_features))
        selected_features = [f for f in selected_features if f in data.columns]
        
        logger.info(f"Selected {len(selected_features)} optimized features")
        
        return data[selected_features + [col for col in data.columns if col.startswith(('profitable_', 'strong_', 'future_'))]].copy()
    
    def run_fast_engineering(self, symbol: str = "BTCUSDT") -> Tuple[pd.DataFrame, Dict]:
        """Run fast feature engineering pipeline."""
        
        logger.info(f"Starting fast feature engineering for {symbol}")
        
        # Load data
        data = self.load_sample_data(symbol)
        
        # Create targets
        logger.info("Creating profit targets...")
        data = self.create_profit_targets(data)
        
        # Create features
        logger.info("Creating essential features...")
        data = self.create_essential_features(data)
        
        # Remove initial NaN rows
        data = data.dropna()
        
        logger.info(f"Created {len(data.columns)} features with {len(data)} valid samples")
        
        # Analyze feature importance
        analysis_results = self.analyze_feature_importance(data)
        
        # Create optimized feature set
        optimized_data = self.create_optimized_feature_set(data, analysis_results)
        
        return optimized_data, analysis_results


def main():
    """Run fast feature engineering."""
    
    engineer = FastFeatureEngineer()
    
    # Run engineering
    data, analysis_results = engineer.run_fast_engineering()
    
    # Print results
    print("\n" + "="*80)
    print("収益性予測特徴量分析結果")
    print("="*80)
    
    for target, results in analysis_results.items():
        print(f"\n🎯 ターゲット: {target}")
        print(f"   陽性率: {results['positive_rate']:.1%}")
        print(f"   サンプル数: {results['total_samples']:,}")
        print(f"   トップ特徴量:")
        
        for i, (_, row) in enumerate(results['importance_df'].head(8).iterrows(), 1):
            print(f"     {i}. {row['feature']:<25} "
                  f"(MI: {row['mutual_info']:.3f}, Corr: {row['correlation']:.3f})")
    
    # Plot feature importance for main target
    if 'profitable_either_5m' in analysis_results:
        results = analysis_results['profitable_either_5m']
        top_features = results['importance_df'].head(15)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Mutual information
        ax1.barh(range(len(top_features)), top_features['mutual_info'], color='skyblue', alpha=0.7)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('Mutual Information Score')
        ax1.set_title('特徴量重要度 (Mutual Information)')
        ax1.grid(True, alpha=0.3)
        
        # Correlation
        ax2.barh(range(len(top_features)), top_features['correlation'], color='lightcoral', alpha=0.7)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature'])
        ax2.set_xlabel('Absolute Correlation')
        ax2.set_title('特徴量重要度 (Correlation)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Save results
    data_path = Path("data/fast_features.pkl")
    data.to_pickle(data_path)
    
    analysis_path = Path("data/fast_feature_analysis.pkl")
    pd.to_pickle(analysis_results, analysis_path)
    
    print(f"\n💾 保存完了:")
    print(f"   特徴量データ: {data_path}")
    print(f"   分析結果: {analysis_path}")
    print(f"   最適化特徴量数: {len([c for c in data.columns if not c.startswith(('profitable_', 'strong_', 'future_'))])}")
    print(f"   データ行数: {len(data):,}")
    
    return data, analysis_results


if __name__ == "__main__":
    main()