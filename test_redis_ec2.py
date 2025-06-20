#!/usr/bin/env python3
"""
Test Redis connectivity and data availability on EC2
"""

import redis
import json

def test_redis():
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Test connection
        r.ping()
        print("✅ Redis connection successful!")
        
        # Check kline stream
        stream_info = r.xinfo_stream('market_data:kline')
        print(f"\n📊 Kline stream info:")
        print(f"  - Length: {stream_info['length']} entries")
        print(f"  - First entry: {stream_info['first-entry'][0] if stream_info['first-entry'] else 'None'}")
        print(f"  - Last entry: {stream_info['last-entry'][0] if stream_info['last-entry'] else 'None'}")
        
        # Get some recent entries
        recent_entries = r.xrevrange('market_data:kline', count=10)
        
        print(f"\n📈 Recent kline entries:")
        symbols_found = set()
        for entry_id, data in recent_entries:
            try:
                parsed = json.loads(data.get('data', '{}'))
                topic = parsed.get('topic', '')
                
                # Extract symbol
                symbol = None
                for s in ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']:
                    if s in topic:
                        symbol = s
                        symbols_found.add(s)
                        break
                
                if symbol:
                    timestamp = parsed.get('timestamp', 0)
                    close = parsed.get('close', 0)
                    print(f"  - {symbol}: ${close:.2f} @ {entry_id}")
                    
            except:
                pass
        
        print(f"\n✅ Found data for symbols: {', '.join(sorted(symbols_found))}")
        
        # Check other streams
        for stream in ['market_data:orderbook', 'market_data:trades', 'market_data:liquidation']:
            try:
                info = r.xinfo_stream(stream)
                print(f"\n{stream}: {info['length']} entries")
            except:
                print(f"\n{stream}: Not found")
        
    except redis.ConnectionError:
        print("❌ Failed to connect to Redis!")
        print("Make sure Redis is running: sudo systemctl status redis")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_redis()