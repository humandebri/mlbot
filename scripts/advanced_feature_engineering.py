#!/usr/bin/env python3
"""
Advanced feature engineering for profitability-focused trading model.

This script explores and creates features that are specifically designed 
to predict profitable trading opportunities rather than just volatility events.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Tuple
import talib
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class AdvancedFeatureEngineer:
    """Advanced feature engineering for profitable trading prediction."""
    
    def __init__(self):
        self.feature_importance = {}
        self.scaler = StandardScaler()
        
    def load_data(self, symbol: str = "BTCUSDT", start_date: str = "2024-01-01", 
                  end_date: str = "2024-03-31") -> pd.DataFrame:
        """Load price data with volume."""
        logger.info(f"Loading data for {symbol}")
        
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
        WHERE timestamp >= '{start_date}'
          AND timestamp <= '{end_date}'
        ORDER BY timestamp
        """
        
        data = conn.execute(query).df()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        conn.close()
        
        logger.info(f"Loaded {len(data)} records for {symbol}")
        return data
    
    def create_target_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create multiple target variables for profitability prediction."""
        
        # Forward returns at different horizons
        for horizon in [1, 3, 5, 10, 15, 30]:
            data[f'future_return_{horizon}m'] = data['close'].pct_change(horizon).shift(-horizon)
        
        # Profitable trade targets (considering transaction costs)
        transaction_cost = 0.0012  # 0.12% round-trip cost
        
        for horizon in [1, 3, 5, 10, 15]:
            # Long profitability
            long_return = data[f'future_return_{horizon}m'] - transaction_cost
            data[f'profitable_long_{horizon}m'] = (long_return > 0.002).astype(int)  # >0.2% profit
            
            # Short profitability  
            short_return = -data[f'future_return_{horizon}m'] - transaction_cost
            data[f'profitable_short_{horizon}m'] = (short_return > 0.002).astype(int)
            
            # Either direction profitability
            data[f'profitable_either_{horizon}m'] = (
                (long_return > 0.002) | (short_return > 0.002)
            ).astype(int)
            
            # Strong moves (for better risk/reward)
            data[f'strong_move_{horizon}m'] = (
                abs(data[f'future_return_{horizon}m']) > 0.005
            ).astype(int)
        
        # Volatility-adjusted returns (Sharpe-like)
        for horizon in [5, 10, 15]:
            vol = data['close'].pct_change().rolling(horizon*2).std()
            data[f'sharpe_future_{horizon}m'] = data[f'future_return_{horizon}m'] / (vol + 1e-8)
        
        return data
    
    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive price-based features."""
        
        # Basic price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['oc_ratio'] = (data['close'] - data['open']) / data['open']
        
        # Multi-timeframe price changes
        for period in [1, 3, 5, 10, 15, 30, 60, 120]:
            data[f'price_change_{period}'] = data['close'].pct_change(period)
            data[f'high_change_{period}'] = data['high'].pct_change(period)
            data[f'low_change_{period}'] = data['low'].pct_change(period)
        
        # Price momentum and acceleration
        for period in [5, 10, 15, 30]:
            data[f'momentum_{period}'] = data['close'].pct_change(period)
            data[f'acceleration_{period}'] = data[f'momentum_{period}'] - data[f'momentum_{period}'].shift(period)
            
        # Price volatility measures
        for window in [5, 10, 15, 30, 60]:
            data[f'volatility_{window}'] = data['returns'].rolling(window).std()
            data[f'volatility_rank_{window}'] = data[f'volatility_{window}'].rolling(window*2).rank(pct=True)
            
        # Gap analysis
        data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        data['gap_filled'] = ((data['low'] <= data['close'].shift(1)) & (data['gap'] > 0)) | \
                            ((data['high'] >= data['close'].shift(1)) & (data['gap'] < 0))
        
        return data
    
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis indicators."""
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        # Trend indicators
        data['sma_10'] = talib.SMA(close, timeperiod=10)
        data['sma_20'] = talib.SMA(close, timeperiod=20)
        data['sma_50'] = talib.SMA(close, timeperiod=50)
        data['ema_12'] = talib.EMA(close, timeperiod=12)
        data['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # MACD
        data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(close)
        
        # Bollinger Bands
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(close)
        data['bb_position'] = (close - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        data['bb_squeeze'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # RSI and Stochastic
        data['rsi'] = talib.RSI(close)
        data['stoch_k'], data['stoch_d'] = talib.STOCH(high, low, close)
        
        # Williams %R
        data['williams_r'] = talib.WILLR(high, low, close)
        
        # Commodity Channel Index
        data['cci'] = talib.CCI(high, low, close)
        
        # Average Directional Index
        data['adx'] = talib.ADX(high, low, close)
        data['plus_di'] = talib.PLUS_DI(high, low, close)
        data['minus_di'] = talib.MINUS_DI(high, low, close)
        
        # Parabolic SAR
        data['sar'] = talib.SAR(high, low)
        
        # Aroon
        data['aroon_up'], data['aroon_down'] = talib.AROON(high, low)
        
        return data
    
    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        
        volume = data['volume'].values
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Volume moving averages
        for window in [5, 10, 20, 50]:
            data[f'volume_ma_{window}'] = data['volume'].rolling(window).mean()
            data[f'volume_ratio_{window}'] = data['volume'] / data[f'volume_ma_{window}']
            
        # Volume Rate of Change
        for period in [1, 5, 10]:
            data[f'volume_roc_{period}'] = data['volume'].pct_change(period)
        
        # On Balance Volume
        data['obv'] = talib.OBV(close, volume)
        data['obv_ma'] = data['obv'].rolling(20).mean()
        data['obv_divergence'] = data['obv'] - data['obv_ma']
        
        # Volume Price Trend
        data['vpt'] = talib.AD(high, low, close, volume)
        
        # Accumulation/Distribution Line
        data['ad_line'] = talib.AD(high, low, close, volume)
        
        # Money Flow Index
        data['mfi'] = talib.MFI(high, low, close, volume)
        
        # Volume-weighted features
        data['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
        data['vwap_distance'] = (data['close'] - data['vwap']) / data['vwap']
        
        # Volume concentration
        data['volume_std'] = data['volume'].rolling(20).std()
        data['volume_spike'] = data['volume'] > (data['volume'].rolling(20).mean() + 2 * data['volume_std'])
        
        return data
    
    def create_market_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features."""
        
        # Spread proxies
        data['hl_spread'] = (data['high'] - data['low']) / data['close']
        data['oc_spread'] = abs(data['open'] - data['close']) / data['close']
        
        # Price efficiency measures
        for window in [5, 10, 20]:
            data[f'price_efficiency_{window}'] = abs(data['close'].pct_change(window)) / \
                                               data['returns'].rolling(window).apply(lambda x: abs(x).sum())
        
        # Realized volatility vs implied volatility proxy
        data['realized_vol'] = data['returns'].rolling(20).std() * np.sqrt(1440)  # 1440 minutes per day
        data['intraday_vol'] = data['hl_spread'].rolling(20).mean()
        data['vol_ratio'] = data['realized_vol'] / (data['intraday_vol'] + 1e-8)
        
        # Price impact measures
        data['price_impact'] = abs(data['returns']) / (data['volume'] / data['volume'].rolling(20).mean())
        
        # Liquidity measures
        data['amihud_illiq'] = abs(data['returns']) / (data['volume'] * data['close'])
        
        # Microstructure noise
        data['microstructure_noise'] = data['returns'].rolling(5).apply(
            lambda x: x.autocorr() if len(x.dropna()) > 1 else 0
        )
        
        return data
    
    def create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market regime and state features."""
        
        # Volatility regimes
        vol_20 = data['returns'].rolling(20).std()
        vol_q25 = vol_20.rolling(100).quantile(0.25)
        vol_q75 = vol_20.rolling(100).quantile(0.75)
        
        data['vol_regime'] = 1  # Normal
        data.loc[vol_20 < vol_q25, 'vol_regime'] = 0  # Low vol
        data.loc[vol_20 > vol_q75, 'vol_regime'] = 2  # High vol
        
        # Trend regimes
        sma_slope = data['sma_20'].pct_change(10)
        data['trend_regime'] = 0  # Sideways
        data.loc[sma_slope > 0.01, 'trend_regime'] = 1  # Uptrend
        data.loc[sma_slope < -0.01, 'trend_regime'] = -1  # Downtrend
        
        # Market state features
        data['distance_from_sma20'] = (data['close'] - data['sma_20']) / data['sma_20']
        data['distance_from_sma50'] = (data['close'] - data['sma_50']) / data['sma_50']
        
        # Regime persistence
        data['vol_regime_duration'] = data.groupby((data['vol_regime'] != data['vol_regime'].shift()).cumsum()).cumcount()
        data['trend_regime_duration'] = data.groupby((data['trend_regime'] != data['trend_regime'].shift()).cumsum()).cumcount()
        
        return data
    
    def create_cross_asset_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create cross-asset correlation features (when multiple assets available)."""
        
        # For now, create autocorrelation features as proxy
        for lag in [1, 5, 10, 20]:
            data[f'return_autocorr_{lag}'] = data['returns'].rolling(50).apply(
                lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag else 0
            )
        
        # Lead-lag relationships with volume
        data['price_volume_corr'] = data['returns'].rolling(20).corr(data['volume'].pct_change())
        
        return data
    
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        
        # Hour of day effects
        data['hour'] = data.index.hour
        data['is_market_open_us'] = ((data['hour'] >= 9) & (data['hour'] <= 16)).astype(int)
        data['is_market_open_asia'] = ((data['hour'] >= 0) & (data['hour'] <= 8)).astype(int)
        
        # Day of week effects
        data['day_of_week'] = data.index.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Monthly effects
        data['day_of_month'] = data.index.day
        data['is_month_end'] = (data['day_of_month'] >= 28).astype(int)
        
        return data
    
    def select_features_for_profitability(self, data: pd.DataFrame, 
                                        target_col: str = 'profitable_either_5m') -> Tuple[List[str], pd.DataFrame]:
        """Select features most predictive of profitability."""
        
        # Get feature columns (exclude targets and metadata)
        feature_cols = [col for col in data.columns if not col.startswith(('future_return', 'profitable', 'strong_move', 'sharpe_future'))]
        
        # Prepare data
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        # Remove rows with NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            logger.warning("No valid data for feature selection")
            return [], pd.DataFrame()
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y)
        
        # Calculate correlation with target
        correlations = []
        for col in X.columns:
            try:
                corr = abs(np.corrcoef(X[col], y)[0, 1])
                correlations.append(corr if not np.isnan(corr) else 0)
            except:
                correlations.append(0)
        
        # Combine scores
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores,
            'correlation': correlations,
            'combined_score': mi_scores + np.array(correlations)
        }).sort_values('combined_score', ascending=False)
        
        # Select top features
        top_features = feature_importance.head(50)['feature'].tolist()
        
        logger.info(f"Selected {len(top_features)} features for {target_col}")
        
        return top_features, feature_importance
    
    def analyze_feature_profitability_relationship(self, data: pd.DataFrame) -> Dict:
        """Analyze which features are most predictive of profitability."""
        
        logger.info("Analyzing feature-profitability relationships")
        
        results = {}
        
        # Test different target variables
        target_variables = [
            'profitable_either_3m',
            'profitable_either_5m', 
            'profitable_either_10m',
            'strong_move_5m',
            'strong_move_10m'
        ]
        
        for target in target_variables:
            if target in data.columns:
                top_features, importance_df = self.select_features_for_profitability(data, target)
                results[target] = {
                    'top_features': top_features[:20],
                    'importance_df': importance_df.head(20)
                }
        
        return results
    
    def run_feature_engineering(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Run complete feature engineering pipeline."""
        
        logger.info(f"Starting advanced feature engineering for {symbol}")
        
        # Load data
        data = self.load_data(symbol)
        
        # Create target variables
        logger.info("Creating target variables...")
        data = self.create_target_variables(data)
        
        # Create feature groups
        logger.info("Creating price features...")
        data = self.create_price_features(data)
        
        logger.info("Creating technical indicators...")
        data = self.create_technical_indicators(data)
        
        logger.info("Creating volume features...")
        data = self.create_volume_features(data)
        
        logger.info("Creating market microstructure features...")
        data = self.create_market_microstructure_features(data)
        
        logger.info("Creating regime features...")
        data = self.create_regime_features(data)
        
        logger.info("Creating cross-asset features...")
        data = self.create_cross_asset_features(data)
        
        logger.info("Creating temporal features...")
        data = self.create_temporal_features(data)
        
        logger.info(f"Generated {len(data.columns)} total features")
        
        return data


def main():
    """Run feature engineering analysis."""
    
    engineer = AdvancedFeatureEngineer()
    
    # Generate features
    data = engineer.run_feature_engineering()
    
    # Analyze feature-profitability relationships
    analysis_results = engineer.analyze_feature_profitability_relationship(data)
    
    # Print results
    print("\n" + "="*80)
    print("特徴量-収益性分析結果")
    print("="*80)
    
    for target, results in analysis_results.items():
        print(f"\n🎯 ターゲット: {target}")
        print(f"トップ特徴量:")
        
        for i, (_, row) in enumerate(results['importance_df'].head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:<30} "
                  f"(MI: {row['mutual_info']:.4f}, Corr: {row['correlation']:.4f})")
    
    # Save feature data
    output_path = Path("data/advanced_features.pkl")
    data.to_pickle(output_path)
    logger.info(f"Advanced features saved to {output_path}")
    
    # Save analysis results
    results_path = Path("data/feature_analysis_results.pkl")
    pd.to_pickle(analysis_results, results_path)
    logger.info(f"Analysis results saved to {results_path}")
    
    print(f"\n💾 データ保存:")
    print(f"  特徴量データ: {output_path}")
    print(f"  分析結果: {results_path}")
    print(f"  総特徴量数: {len(data.columns)}")
    print(f"  データポイント数: {len(data)}")


if __name__ == "__main__":
    main()