#!/usr/bin/env python3
"""
Fix feature access issue by correcting Redis data type handling
"""
import sys
sys.path.insert(0, '/home/ubuntu/mlbot')

import redis
import json
import asyncio
from src.common.config import settings
from src.common.logging import get_logger
from src.common.discord_notifier import discord_notifier

logger = get_logger(__name__)

async def fix_feature_access():
    """Fix feature access by correcting Redis data handling"""
    
    logger.info("🔧 Starting feature access fix...")
    
    discord_notifier.send_system_status(
        "fixing",
        "🔧 特徴量アクセス問題の修正を開始します..."
    )
    
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        logger.info("✅ Connected to Redis")
        
        # Investigate actual Redis data structure
        all_keys = r.keys("*")
        feature_keys = [k for k in all_keys if 'features' in k]
        stream_keys = [k for k in all_keys if 'market_data' in k]
        
        logger.info(f"📊 Redis analysis:")
        logger.info(f"  Total keys: {len(all_keys)}")
        logger.info(f"  Feature keys: {len(feature_keys)}")
        logger.info(f"  Stream keys: {len(stream_keys)}")
        
        # Check each feature key type and content
        feature_data = {}
        for key in feature_keys:
            key_type = r.type(key)
            logger.info(f"🔍 {key}: type={key_type}")
            
            if key_type == 'stream':
                # Get latest entry from stream
                try:
                    entries = r.xrevrange(key, count=1)
                    if entries:
                        entry_id, fields = entries[0]
                        feature_data[key] = {
                            "type": "stream",
                            "latest_id": entry_id,
                            "field_count": len(fields),
                            "sample_fields": dict(list(fields.items())[:3])
                        }
                        logger.info(f"  ✅ Stream {key}: {len(fields)} fields, latest_id={entry_id}")
                    else:
                        feature_data[key] = {"type": "stream", "empty": True}
                        logger.warning(f"  ⚠️ Stream {key}: empty")
                
                except Exception as e:
                    logger.error(f"  ❌ Error reading stream {key}: {e}")
                    feature_data[key] = {"type": "stream", "error": str(e)}
            
            elif key_type == 'hash':
                # Get hash content
                try:
                    fields = r.hgetall(key)
                    feature_data[key] = {
                        "type": "hash",
                        "field_count": len(fields),
                        "sample_fields": dict(list(fields.items())[:3])
                    }
                    logger.info(f"  ✅ Hash {key}: {len(fields)} fields")
                
                except Exception as e:
                    logger.error(f"  ❌ Error reading hash {key}: {e}")
                    feature_data[key] = {"type": "hash", "error": str(e)}
            
            else:
                feature_data[key] = {"type": key_type, "unsupported": True}
                logger.warning(f"  ⚠️ Unsupported type {key}: {key_type}")
        
        # Now create a fixed feature access function
        def get_latest_features_fixed(symbol: str) -> dict:
            """Fixed version that handles both hash and stream types"""
            
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
                    logger.debug(f"Found key {key} with type {key_type}")
                    
                    if key_type == 'hash':
                        # Read as hash
                        return r.hgetall(key)
                    
                    elif key_type == 'stream':
                        # Read latest entry from stream
                        try:
                            entries = r.xrevrange(key, count=1)
                            if entries:
                                entry_id, fields = entries[0]
                                # Convert to regular dict
                                return dict(fields)
                            else:
                                logger.warning(f"Stream {key} is empty")
                                return {}
                        except Exception as e:
                            logger.error(f"Error reading stream {key}: {e}")
                            return {}
                    
                    else:
                        logger.warning(f"Unsupported key type {key_type} for key {key}")
                        return {}
            
            logger.warning(f"No feature data found for symbol {symbol}")
            return {}
        
        # Test the fixed function
        logger.info("🧪 Testing fixed feature access...")
        
        test_results = {}
        total_features = 0
        
        for symbol in settings.bybit.symbols:
            features = get_latest_features_fixed(symbol)
            feature_count = len(features)
            total_features += feature_count
            
            test_results[symbol] = {
                "count": feature_count,
                "accessible": feature_count > 0
            }
            
            if feature_count > 0:
                # Sample some feature values
                sample_features = dict(list(features.items())[:5])
                test_results[symbol]["sample"] = sample_features
                logger.info(f"✅ {symbol}: {feature_count} features accessible")
                
                # Try to convert to float to verify data quality
                numeric_count = 0
                for key, value in sample_features.items():
                    try:
                        float(value)
                        numeric_count += 1
                    except ValueError:
                        pass
                
                test_results[symbol]["numeric_features"] = numeric_count
                logger.info(f"  📊 {numeric_count}/{len(sample_features)} sample features are numeric")
                
            else:
                logger.warning(f"❌ {symbol}: No features accessible")
        
        # Generate fix report
        successful_symbols = sum(1 for result in test_results.values() if result["accessible"])
        
        report = f"""
🔧 **特徴量アクセス修正結果** 🔧

📊 **修正前の問題**:
• データ型不一致: Stream vs Hash
• アクセス失敗: 全シンボル0個

🛠️ **実装した修正**:
• 複数キーパターン対応
• Stream/Hash両方に対応
• エラーハンドリング強化

📈 **修正後の結果**:
• アクセス可能シンボル: {successful_symbols}/3
• 総特徴量数: {total_features}個
"""
        
        for symbol in settings.bybit.symbols:
            result = test_results[symbol]
            status = "✅" if result["accessible"] else "❌"
            count = result["count"]
            report += f"• {symbol}: {status} {count}個\n"
        
        logger.info(report)
        
        # Send Discord notification
        if successful_symbols == 3 and total_features > 0:
            discord_notifier.send_system_status(
                "fix_success",
                f"✅ **特徴量アクセス修正完了** ✅\n\n" +
                f"アクセス可能: {successful_symbols}/3シンボル\n" +
                f"総特徴量: {total_features}個\n" +
                f"データ型: Stream対応完了\n\n" +
                "🚀 次: 統合取引システム構築"
            )
            
            # Save the fixed function for use in main system
            logger.info("💾 Saving fixed feature access method...")
            
            # Write the fixed method to a file that can be imported
            fixed_code = '''
def get_latest_features_fixed(redis_client, symbol: str) -> dict:
    """Fixed version that handles both hash and stream types"""
    import redis
    
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
                # Read as hash
                return r.hgetall(key)
            
            elif key_type == 'stream':
                # Read latest entry from stream
                try:
                    entries = r.xrevrange(key, count=1)
                    if entries:
                        entry_id, fields = entries[0]
                        # Convert to regular dict
                        return dict(fields)
                    else:
                        return {}
                except Exception:
                    return {}
            
            else:
                return {}
    
    return {}
'''
            
            with open('/home/ubuntu/mlbot/fixed_feature_access.py', 'w') as f:
                f.write(fixed_code)
            
            logger.info("✅ Fixed feature access method saved")
            
        else:
            discord_notifier.send_error(
                "feature_fix",
                f"特徴量アクセス修正に部分的成功:\n{successful_symbols}/3シンボル、{total_features}個"
            )
        
        return {
            "redis_keys": feature_data,
            "test_results": test_results,
            "successful_symbols": successful_symbols,
            "total_features": total_features
        }
    
    except Exception as e:
        logger.error(f"❌ Feature fix failed: {e}")
        discord_notifier.send_error("feature_fix", f"修正失敗: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting feature access fix")
    result = asyncio.run(fix_feature_access())
    logger.info(f"Fix complete: {result}")