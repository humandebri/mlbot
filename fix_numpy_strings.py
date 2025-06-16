#!/usr/bin/env python3
"""
Fix numpy string parsing in feature data
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import re
import redis
import numpy as np
from src.common.config import settings
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier

logger = get_logger(__name__)

def parse_numpy_string(value_str):
    """Parse numpy string representation back to float"""
    
    # Handle various numpy string formats
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    if not isinstance(value_str, str):
        return float(value_str)
    
    # Remove numpy type wrapper
    # Patterns like 'np.float64(1.23)' or 'numpy.float64(1.23)'
    patterns = [
        r'np\.float\d*\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)',
        r'numpy\.float\d*\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)',
        r'float\d*\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, value_str)
        if match:
            return float(match.group(1))
    
    # Try direct float conversion
    try:
        return float(value_str)
    except ValueError:
        # If all else fails, try to extract any float-like number
        number_pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
        match = re.search(number_pattern, value_str)
        if match:
            return float(match.group())
        else:
            raise ValueError(f"Cannot parse numeric value from: {value_str}")

def get_latest_features_with_numpy_fix(redis_client, symbol: str) -> dict:
    """Fixed version that handles numpy strings properly"""
    
    r = redis_client
    
    # Try different key patterns
    possible_keys = [
        f"features:{symbol}:latest",
        f"features:{symbol}",
        f"{symbol}:features",
        f"features_{symbol}"
    ]
    
    for key in possible_keys:
        if r.exists(key):
            key_type = r.type(key)
            
            if key_type == 'hash':
                raw_data = r.hgetall(key)
                parsed_data = {}
                for k, v in raw_data.items():
                    try:
                        parsed_data[str(k)] = parse_numpy_string(v)
                    except ValueError as e:
                        logger.warning(f"Skipping unparseable feature {k}: {v} ({e})")
                        continue
                return parsed_data
            
            elif key_type == 'stream':
                try:
                    entries = r.xrevrange(key, count=1)
                    if entries:
                        entry_id, fields = entries[0]
                        parsed_data = {}
                        
                        for k, v in dict(fields).items():
                            try:
                                parsed_data[str(k)] = parse_numpy_string(v)
                            except ValueError as e:
                                logger.warning(f"Skipping unparseable feature {k}: {v} ({e})")
                                continue
                        
                        return parsed_data
                    else:
                        return {}
                except Exception as e:
                    logger.error(f"Error reading stream {key}: {e}")
                    return {}
    
    return {}

async def test_numpy_fix():
    """Test the numpy string parsing fix"""
    
    logger.info("🔧 Testing numpy string parsing fix...")
    
    discord_notifier.send_system_status(
        "numpy_fix_test",
        "🔧 Numpy文字列パース修正のテストを開始..."
    )
    
    try:
        # Test the parser function first
        test_cases = [
            "np.float64(-1.0958094368558438e-05)",
            "np.float64(3.101785077326803e-05)",
            "numpy.float64(123.456)",
            "1.23",
            "-456.789e-3",
            "0.0"
        ]
        
        logger.info("🧪 Testing parser function:")
        parser_success = 0
        
        for test_case in test_cases:
            try:
                parsed = parse_numpy_string(test_case)
                logger.info(f"  ✅ '{test_case}' → {parsed}")
                parser_success += 1
            except Exception as e:
                logger.error(f"  ❌ '{test_case}' → {e}")
        
        logger.info(f"Parser success rate: {parser_success}/{len(test_cases)}")
        
        # Test with real Redis data
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        if not redis_client.ping():
            raise Exception("Redis connection failed")
        
        logger.info("✅ Redis connected")
        
        # Test for each symbol
        test_results = {}
        total_features = 0
        successful_symbols = 0
        
        for symbol in settings.bybit.symbols:
            logger.info(f"🧪 Testing {symbol}...")
            
            try:
                features = get_latest_features_with_numpy_fix(redis_client, symbol)
                feature_count = len(features)
                total_features += feature_count
                
                if feature_count > 0:
                    successful_symbols += 1
                    
                    # Sample some values to verify parsing
                    sample_features = dict(list(features.items())[:5])
                    numeric_count = 0
                    
                    for k, v in sample_features.items():
                        if isinstance(v, (int, float)):
                            numeric_count += 1
                    
                    test_results[symbol] = {
                        "success": True,
                        "count": feature_count,
                        "sample": sample_features,
                        "numeric_ratio": f"{numeric_count}/{len(sample_features)}"
                    }
                    
                    logger.info(f"  ✅ {feature_count} features, {numeric_count}/{len(sample_features)} numeric")
                    
                else:
                    test_results[symbol] = {
                        "success": False,
                        "count": 0,
                        "error": "No features found"
                    }
                    logger.warning(f"  ❌ No features found")
            
            except Exception as e:
                logger.error(f"  ❌ Error testing {symbol}: {e}")
                test_results[symbol] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Generate report
        report = f"""
🔧 **Numpy文字列修正テスト結果** 🔧

🧪 **パーサーテスト**: {parser_success}/{len(test_cases)}
📊 **成功シンボル**: {successful_symbols}/3
📈 **総特徴量数**: {total_features}

"""
        
        for symbol, result in test_results.items():
            if result.get("success"):
                count = result.get("count", 0)
                numeric_ratio = result.get("numeric_ratio", "0/0")
                report += f"✅ {symbol}: {count}個 (数値: {numeric_ratio})\n"
            else:
                error = result.get("error", "Unknown error")
                report += f"❌ {symbol}: {error}\n"
        
        logger.info(report)
        
        # Send Discord notification
        if successful_symbols == 3 and total_features > 300:
            discord_notifier.send_system_status(
                "numpy_fix_success",
                f"✅ **Numpy修正完了** ✅\n\n" +
                f"パーサー: {parser_success}/{len(test_cases)}\n" +
                f"成功シンボル: {successful_symbols}/3\n" +
                f"総特徴量: {total_features}個\n\n" +
                "🚀 予測テスト準備完了"
            )
            
            # Save the fixed function
            logger.info("💾 Saving fixed feature access function...")
            
            fixed_code = '''
import re

def parse_numpy_string(value_str):
    """Parse numpy string representation back to float"""
    
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    if not isinstance(value_str, str):
        return float(value_str)
    
    # Remove numpy type wrapper
    patterns = [
        r'np\\.float\\d*\\(([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\\)',
        r'numpy\\.float\\d*\\(([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\\)',
        r'float\\d*\\(([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, value_str)
        if match:
            return float(match.group(1))
    
    try:
        return float(value_str)
    except ValueError:
        number_pattern = r'[-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
        match = re.search(number_pattern, value_str)
        if match:
            return float(match.group())
        else:
            raise ValueError(f"Cannot parse numeric value from: {value_str}")

def get_latest_features_fixed(redis_client, symbol: str) -> dict:
    """Fixed version that handles numpy strings properly"""
    
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
            
            if key_type == 'hash':
                raw_data = r.hgetall(key)
                parsed_data = {}
                for k, v in raw_data.items():
                    try:
                        parsed_data[str(k)] = parse_numpy_string(v)
                    except ValueError:
                        continue
                return parsed_data
            
            elif key_type == 'stream':
                try:
                    entries = r.xrevrange(key, count=1)
                    if entries:
                        entry_id, fields = entries[0]
                        parsed_data = {}
                        
                        for k, v in dict(fields).items():
                            try:
                                parsed_data[str(k)] = parse_numpy_string(v)
                            except ValueError:
                                continue
                        
                        return parsed_data
                    else:
                        return {}
                except Exception:
                    return {}
    
    return {}
'''
            
            with open('/home/ubuntu/mlbot/numpy_fixed_features.py', 'w') as f:
                f.write(fixed_code)
            
            logger.info("✅ Fixed feature access with numpy parsing saved")
            
        else:
            discord_notifier.send_error(
                "numpy_fix",
                f"Numpy修正部分成功:\n{successful_symbols}/3シンボル、{total_features}個"
            )
        
        return {
            "parser_success": parser_success,
            "total_test_cases": len(test_cases),
            "successful_symbols": successful_symbols,
            "total_features": total_features,
            "test_results": test_results
        }
        
    except Exception as e:
        logger.error(f"❌ Numpy fix test failed: {e}")
        discord_notifier.send_error("numpy_fix", f"修正テスト失敗: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import asyncio
    logger.info("Starting numpy string fix test")
    result = asyncio.run(test_numpy_fix())
    logger.info(f"Test complete: {result}")