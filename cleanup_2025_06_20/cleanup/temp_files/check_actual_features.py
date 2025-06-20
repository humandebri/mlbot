#!/usr/bin/env python3
"""
実際の特徴量名を確認
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import redis
import json
from src.common.config import settings
from src.common.logging import get_logger

logger = get_logger(__name__)

def parse_numpy_string(value_str):
    import re
    if isinstance(value_str, (int, float)):
        return float(value_str)
    if not isinstance(value_str, str):
        return float(value_str)
    
    patterns = [
        r'np\.float\d*\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, value_str)
        if match:
            return float(match.group(1))
    
    try:
        return float(value_str)
    except ValueError:
        return 0.0

def get_feature_names(redis_client, symbol: str):
    r = redis_client
    key = f"features:{symbol}:latest"
    
    if r.exists(key):
        key_type = r.type(key)
        
        if key_type == 'stream':
            entries = r.xrevrange(key, count=1)
            if entries:
                entry_id, fields = entries[0]
                return list(dict(fields).keys())
        elif key_type == 'hash':
            return list(r.hgetall(key).keys())
    
    return []

def main():
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    print("🔍 実際の特徴量名確認...")
    
    all_features = set()
    
    for symbol in settings.bybit.symbols:
        feature_names = get_feature_names(redis_client, symbol)
        all_features.update(feature_names)
        
        print(f"\n📊 {symbol}: {len(feature_names)}個の特徴量")
        print("上位20個:")
        for i, name in enumerate(sorted(feature_names)[:20]):
            print(f"  {i+1:2d}. {name}")
    
    print(f"\n📋 全体で{len(all_features)}種類の特徴量:")
    
    # カテゴリ別分類
    categories = {
        "price": [],
        "volume": [],
        "spread": [],
        "liquidity": [],
        "volatility": [],
        "time": [],
        "returns": [],
        "technical": [],
        "market": [],
        "other": []
    }
    
    for feature in sorted(all_features):
        feature_lower = feature.lower()
        
        if any(x in feature_lower for x in ['price', 'close', 'open', 'high', 'low']):
            categories["price"].append(feature)
        elif any(x in feature_lower for x in ['volume', 'vol']):
            categories["volume"].append(feature)
        elif any(x in feature_lower for x in ['spread', 'bid', 'ask']):
            categories["spread"].append(feature)
        elif any(x in feature_lower for x in ['size', 'depth', 'book']):
            categories["liquidity"].append(feature)
        elif any(x in feature_lower for x in ['volatility', 'std', 'var']):
            categories["volatility"].append(feature)
        elif any(x in feature_lower for x in ['time', 'hour', 'minute', 'second']):
            categories["time"].append(feature)
        elif any(x in feature_lower for x in ['return', 'pct', 'change']):
            categories["returns"].append(feature)
        elif any(x in feature_lower for x in ['rsi', 'macd', 'sma', 'ema', 'bb']):
            categories["technical"].append(feature)
        elif any(x in feature_lower for x in ['trend', 'regime', 'market']):
            categories["market"].append(feature)
        else:
            categories["other"].append(feature)
    
    for category, features in categories.items():
        if features:
            print(f"\n📈 {category.upper()} ({len(features)}個):")
            for feature in features[:10]:  # 上位10個
                print(f"  • {feature}")
            if len(features) > 10:
                print(f"  ... and {len(features)-10} more")

if __name__ == "__main__":
    main()