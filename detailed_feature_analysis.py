#!/usr/bin/env python3
"""
詳細な特徴量処理状況の分析
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import asyncio
import redis
import numpy as np
import json
from src.common.config import settings
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier
from src.ml_pipeline.feature_adapter_44 import FeatureAdapter44

logger = get_logger(__name__)

def parse_numpy_string(value_str):
    """Parse numpy string representation back to float"""
    import re
    
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    if not isinstance(value_str, str):
        return float(value_str)
    
    patterns = [
        r'np\.float\d*\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)',
        r'numpy\.float\d*\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, value_str)
        if match:
            return float(match.group(1))
    
    try:
        return float(value_str)
    except ValueError:
        return 0.0

def get_latest_features_with_analysis(redis_client, symbol: str) -> dict:
    """特徴量取得と詳細分析"""
    
    r = redis_client
    possible_keys = [
        f"features:{symbol}:latest",
        f"features:{symbol}",
        f"{symbol}:features",
        f"features_{symbol}"
    ]
    
    for key in possible_keys:
        if r.exists(key):
            key_type = r.type(key)
            
            if key_type == 'stream':
                try:
                    entries = r.xrevrange(key, count=1)
                    if entries:
                        entry_id, fields = entries[0]
                        parsed_data = {}
                        
                        raw_count = 0
                        parsed_count = 0
                        error_count = 0
                        
                        for k, v in dict(fields).items():
                            raw_count += 1
                            try:
                                parsed_value = parse_numpy_string(v)
                                parsed_data[str(k)] = parsed_value
                                parsed_count += 1
                            except Exception as e:
                                error_count += 1
                                logger.debug(f"Parse error for {k}={v}: {e}")
                        
                        logger.info(f"📊 {symbol} ({key}): {raw_count}生→{parsed_count}解析成功, {error_count}エラー")
                        return parsed_data
                except Exception as e:
                    logger.error(f"Stream読み込みエラー {key}: {e}")
                    return {}
            
            elif key_type == 'hash':
                raw_data = r.hgetall(key)
                parsed_data = {}
                
                raw_count = 0
                parsed_count = 0
                error_count = 0
                
                for k, v in raw_data.items():
                    raw_count += 1
                    try:
                        parsed_value = parse_numpy_string(v)
                        parsed_data[str(k)] = parsed_value
                        parsed_count += 1
                    except Exception as e:
                        error_count += 1
                        logger.debug(f"Parse error for {k}={v}: {e}")
                
                logger.info(f"📊 {symbol} ({key}): {raw_count}生→{parsed_count}解析成功, {error_count}エラー")
                return parsed_data
    
    return {}

async def detailed_feature_analysis():
    """特徴量処理の詳細分析"""
    
    logger.info("🔍 特徴量処理状況の詳細分析開始...")
    
    discord_notifier.send_system_status(
        "feature_analysis",
        "🔍 **特徴量処理状況の詳細分析** 🔍\n\n" +
        "実際の処理品質を調査中..."
    )
    
    try:
        # Redis接続
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        if not redis_client.ping():
            raise Exception("Redis connection failed")
        
        logger.info("✅ Redis接続成功")
        
        # FeatureAdapter44初期化
        feature_adapter = FeatureAdapter44()
        logger.info("✅ FeatureAdapter44初期化成功")
        
        analysis_results = {}
        total_raw_features = 0
        total_parsed_features = 0
        adaptation_success = 0
        
        for symbol in settings.bybit.symbols:
            logger.info(f"\n🧪 {symbol}の詳細分析...")
            
            # 1. 生特徴量取得・解析
            features = get_latest_features_with_analysis(redis_client, symbol)
            raw_count = len(features)
            total_raw_features += raw_count
            
            if raw_count == 0:
                analysis_results[symbol] = {"error": "特徴量なし"}
                continue
            
            # 2. 特徴量の型・値の分析
            numeric_features = 0
            zero_features = 0
            nan_features = 0
            inf_features = 0
            value_ranges = []
            
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    numeric_features += 1
                    total_parsed_features += 1
                    
                    if value == 0:
                        zero_features += 1
                    elif np.isnan(value):
                        nan_features += 1
                    elif np.isinf(value):
                        inf_features += 1
                    else:
                        value_ranges.append(abs(value))
            
            # 3. FeatureAdapter44での変換テスト
            try:
                adapted_features = feature_adapter.adapt(features)
                adaptation_success += 1
                
                # 変換後の特徴量分析
                adapted_stats = {
                    "mean": float(np.mean(adapted_features)),
                    "std": float(np.std(adapted_features)),
                    "min": float(np.min(adapted_features)),
                    "max": float(np.max(adapted_features)),
                    "zeros": int(np.sum(adapted_features == 0)),
                    "nans": int(np.sum(np.isnan(adapted_features))),
                    "infs": int(np.sum(np.isinf(adapted_features)))
                }
                
                logger.info(f"  ✅ 変換成功: {raw_count}→44次元")
                logger.info(f"  📈 統計: mean={adapted_stats['mean']:.6f}, std={adapted_stats['std']:.6f}")
                logger.info(f"  📊 範囲: [{adapted_stats['min']:.6f}, {adapted_stats['max']:.6f}]")
                logger.info(f"  ⚠️ 問題値: zeros={adapted_stats['zeros']}, nans={adapted_stats['nans']}, infs={adapted_stats['infs']}")
                
                analysis_results[symbol] = {
                    "success": True,
                    "raw_features": raw_count,
                    "numeric_features": numeric_features,
                    "adapted_shape": adapted_features.shape,
                    "adapted_stats": adapted_stats,
                    "quality_issues": {
                        "zeros": zero_features,
                        "nans": nan_features,
                        "infs": inf_features
                    }
                }
                
                # 4. サンプル特徴量の確認
                if len(value_ranges) > 0:
                    sample_features = dict(list(features.items())[:5])
                    logger.info(f"  🔍 サンプル特徴量: {sample_features}")
                    
                    sample_adapted = adapted_features[:5]
                    logger.info(f"  🔄 変換後サンプル: {sample_adapted}")
                
            except Exception as e:
                logger.error(f"  ❌ 変換失敗: {e}")
                analysis_results[symbol] = {
                    "success": False,
                    "raw_features": raw_count,
                    "numeric_features": numeric_features,
                    "error": str(e)
                }
        
        # 5. 全体サマリー
        logger.info(f"\n📋 全体分析結果:")
        logger.info(f"  生特徴量総数: {total_raw_features}")
        logger.info(f"  解析成功数: {total_parsed_features}")
        logger.info(f"  変換成功シンボル: {adaptation_success}/3")
        
        # Discord報告
        success_rate = adaptation_success / 3 * 100
        
        report = f"🔍 **特徴量処理分析結果** 🔍\n\n"
        report += f"📊 **処理統計**:\n"
        report += f"• 生特徴量: {total_raw_features}個\n"
        report += f"• 解析成功: {total_parsed_features}個\n"
        report += f"• 変換成功率: {success_rate:.1f}%\n\n"
        
        for symbol, result in analysis_results.items():
            if result.get("success"):
                stats = result["adapted_stats"]
                issues = result["quality_issues"]
                report += f"✅ **{symbol}**: {result['raw_features']}→44次元\n"
                report += f"  • 統計: μ={stats['mean']:.4f}, σ={stats['std']:.4f}\n"
                report += f"  • 問題値: {stats['zeros']+stats['nans']+stats['infs']}個\n"
            else:
                report += f"❌ **{symbol}**: {result.get('error', '不明エラー')}\n"
        
        if adaptation_success == 3:
            report += "\n🚀 **全シンボル正常処理中**"
            discord_notifier.send_system_status("feature_analysis_success", report)
        else:
            report += f"\n⚠️ **{3-adaptation_success}シンボルで問題あり**"
            discord_notifier.send_error("feature_analysis", report)
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"❌ 分析失敗: {e}")
        discord_notifier.send_error("feature_analysis", f"分析失敗: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting detailed feature analysis")
    result = asyncio.run(detailed_feature_analysis())
    logger.info("Analysis complete")