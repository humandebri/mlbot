#!/usr/bin/env python3
"""
既存の履歴データテーブルを統合し、最新データで補完するスクリプト
2-4年分の豊富なデータを活用してバランスの取れたデータセットを作成
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import redis
import json
import logging
from typing import Dict, List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDataMerger:
    """履歴データの統合と更新"""
    
    def __init__(self, db_path: str = "data/historical_data.duckdb"):
        self.db_path = db_path
        self.conn = None
        self.redis_client = None
        
        # 対象シンボル
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'ICPUSDT']
        
        # データ統計
        self.stats = {}
        
    def connect(self):
        """データベースとRedis接続"""
        try:
            # DuckDB接続（書き込み可能）
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"Connected to DuckDB: {self.db_path}")
            
            # Redis接続
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            logger.info("Connected to Redis")
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise
    
    def analyze_existing_data(self) -> Dict:
        """既存データの分析"""
        logger.info("Analyzing existing historical data...")
        
        analysis = {}
        
        for symbol in self.symbols:
            table_name = f"klines_{symbol.lower()}"
            
            try:
                # データ量と期間を確認
                query = f"""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as start_date,
                    MAX(timestamp) as end_date,
                    COUNT(DISTINCT DATE(timestamp)) as trading_days
                FROM {table_name}
                """
                
                result = self.conn.execute(query).fetchone()
                
                if result[0] > 0:
                    analysis[symbol] = {
                        'table_name': table_name,
                        'total_records': result[0],
                        'start_date': result[1],
                        'end_date': result[2],
                        'trading_days': result[3],
                        'years': (pd.to_datetime(result[2]) - pd.to_datetime(result[1])).days / 365.25
                    }
                    
                    # 年別のデータ分布を確認
                    year_dist = self.conn.execute(f"""
                        SELECT 
                            YEAR(timestamp) as year,
                            COUNT(*) as records
                        FROM {table_name}
                        GROUP BY YEAR(timestamp)
                        ORDER BY year
                    """).fetchall()
                    
                    analysis[symbol]['yearly_distribution'] = {
                        str(year): count for year, count in year_dist
                    }
                    
                    logger.info(f"{symbol}: {result[0]:,} records from {result[1]} to {result[2]} ({analysis[symbol]['years']:.1f} years)")
                    
            except Exception as e:
                logger.warning(f"Could not analyze {table_name}: {e}")
        
        self.stats['analysis'] = analysis
        return analysis
    
    def create_unified_table(self):
        """統合テーブルの作成"""
        logger.info("Creating unified all_klines table...")
        
        # 既存のall_klinesを削除（ビューまたはテーブル）
        try:
            self.conn.execute("DROP VIEW IF EXISTS all_klines")
        except:
            pass
        try:
            self.conn.execute("DROP TABLE IF EXISTS all_klines")
        except:
            pass
        
        # 新しいall_klinesテーブルを作成
        create_table_sql = """
        CREATE TABLE all_klines (
            timestamp TIMESTAMP,
            symbol VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            turnover DOUBLE,
            open_time BIGINT,
            PRIMARY KEY (symbol, timestamp)
        )
        """
        
        self.conn.execute(create_table_sql)
        logger.info("Created all_klines table")
    
    def merge_historical_data(self):
        """個別テーブルからデータを統合"""
        logger.info("Merging historical data from individual tables...")
        
        total_merged = 0
        
        for symbol in self.symbols:
            table_name = f"klines_{symbol.lower()}"
            
            try:
                # データを統合（重複を避けるためINSERT OR IGNORE）
                merge_sql = f"""
                INSERT OR IGNORE INTO all_klines 
                SELECT 
                    timestamp,
                    '{symbol}' as symbol,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    turnover,
                    EXTRACT(EPOCH FROM timestamp) * 1000 as open_time
                FROM {table_name}
                WHERE timestamp IS NOT NULL
                """
                
                result = self.conn.execute(merge_sql)
                rows_affected = result.fetchone()[0] if result else 0
                
                total_merged += rows_affected
                logger.info(f"Merged {rows_affected:,} records from {table_name}")
                
            except Exception as e:
                logger.error(f"Error merging {table_name}: {e}")
        
        logger.info(f"Total records merged: {total_merged:,}")
        
        # インデックスを作成
        logger.info("Creating indexes...")
        self.conn.execute("CREATE INDEX idx_all_klines_timestamp ON all_klines(timestamp)")
        self.conn.execute("CREATE INDEX idx_all_klines_symbol ON all_klines(symbol)")
        
        return total_merged
    
    def get_latest_timestamps(self) -> Dict[str, datetime]:
        """各シンボルの最新タイムスタンプを取得"""
        latest_times = {}
        
        for symbol in self.symbols:
            try:
                result = self.conn.execute(f"""
                    SELECT MAX(timestamp) 
                    FROM all_klines 
                    WHERE symbol = '{symbol}'
                """).fetchone()
                
                if result[0]:
                    latest_times[symbol] = pd.to_datetime(result[0])
                    logger.info(f"{symbol} latest data: {result[0]}")
                    
            except Exception as e:
                logger.error(f"Error getting latest timestamp for {symbol}: {e}")
        
        return latest_times
    
    def update_from_redis(self, lookback_hours: int = 24 * 14):  # 14日分
        """Redisから最新データを取得して更新"""
        logger.info(f"Updating from Redis (last {lookback_hours} hours)...")
        
        if not self.redis_client:
            logger.warning("Redis client not initialized")
            return
        
        # 最新タイムスタンプを取得
        latest_times = self.get_latest_timestamps()
        
        # Redisからデータを取得
        entries = self.redis_client.xrevrange('market_data:kline', count=100000)
        
        if not entries:
            logger.warning("No data in Redis")
            return
        
        # データを処理
        updates = {symbol: [] for symbol in self.symbols}
        
        for entry_id, data in entries:
            try:
                parsed = json.loads(data.get('data', '{}'))
                topic = parsed.get('topic', '')
                
                # シンボルを抽出
                symbol = None
                for sym in self.symbols:
                    if sym in topic:
                        symbol = sym
                        break
                
                if not symbol:
                    continue
                
                # タイムスタンプを確認
                timestamp = pd.to_datetime(parsed.get('ts', 0), unit='ms')
                
                # 最新データより新しい場合のみ追加
                if symbol in latest_times and timestamp > latest_times[symbol]:
                    kline_data = parsed.get('data', [{}])[0]
                    
                    updates[symbol].append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'open': float(kline_data.get('open', 0)),
                        'high': float(kline_data.get('high', 0)),
                        'low': float(kline_data.get('low', 0)),
                        'close': float(kline_data.get('close', 0)),
                        'volume': float(kline_data.get('volume', 0)),
                        'turnover': float(kline_data.get('turnover', 0)),
                        'open_time': int(timestamp.timestamp() * 1000)
                    })
                    
            except Exception as e:
                logger.debug(f"Error processing Redis entry: {e}")
        
        # データベースに挿入
        total_updated = 0
        for symbol, records in updates.items():
            if records:
                df = pd.DataFrame(records)
                df = df.drop_duplicates(subset=['timestamp'])
                
                # DuckDBに挿入
                self.conn.execute("BEGIN TRANSACTION")
                
                for _, row in df.iterrows():
                    try:
                        self.conn.execute("""
                            INSERT INTO all_klines 
                            (timestamp, symbol, open, high, low, close, volume, turnover, open_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            row['timestamp'], row['symbol'], row['open'], 
                            row['high'], row['low'], row['close'], 
                            row['volume'], row['turnover'], row['open_time']
                        ])
                        total_updated += 1
                    except Exception as e:
                        logger.debug(f"Skipping duplicate: {e}")
                
                self.conn.execute("COMMIT")
                logger.info(f"Updated {len(records)} records for {symbol}")
        
        logger.info(f"Total records updated from Redis: {total_updated}")
        
        return total_updated
    
    def verify_data_quality(self):
        """データ品質の検証"""
        logger.info("Verifying data quality...")
        
        quality_report = {}
        
        for symbol in self.symbols:
            # 基本統計
            stats = self.conn.execute(f"""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as start_date,
                    MAX(timestamp) as end_date,
                    AVG(close) as avg_price,
                    COUNT(DISTINCT DATE(timestamp)) as trading_days,
                    SUM(CASE WHEN close > open THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as bullish_pct
                FROM all_klines
                WHERE symbol = '{symbol}'
            """).fetchone()
            
            # ギャップ分析（1分足で1時間以上のギャップ）
            gaps = self.conn.execute(f"""
                WITH ordered_data AS (
                    SELECT 
                        timestamp,
                        LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
                    FROM all_klines
                    WHERE symbol = '{symbol}'
                )
                SELECT COUNT(*) as gap_count
                FROM ordered_data
                WHERE timestamp - prev_timestamp > INTERVAL '1 hour'
            """).fetchone()
            
            quality_report[symbol] = {
                'total_records': stats[0] if stats[0] else 0,
                'date_range': f"{stats[1]} to {stats[2]}",
                'average_price': float(stats[3]) if stats[3] else 0.0,
                'trading_days': stats[4] if stats[4] else 0,
                'bullish_percentage': float(stats[5]) if stats[5] else 0.0,
                'data_gaps': gaps[0] if gaps[0] else 0,
                'completeness': (stats[0] / (stats[4] * 1440)) * 100 if stats[0] and stats[4] else 0.0
            }
            
            logger.info(f"{symbol}: {stats[0]:,} records, {stats[5]:.1f}% bullish, {gaps[0]} gaps")
        
        self.stats['quality_report'] = quality_report
        return quality_report
    
    def generate_summary_report(self):
        """統合結果のサマリーレポート生成"""
        logger.info("\n" + "="*60)
        logger.info("DATA MERGE SUMMARY REPORT")
        logger.info("="*60)
        
        # 全体統計
        total_stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as symbols,
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest
            FROM all_klines
        """).fetchone()
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total Records: {total_stats[0]:,}")
        print(f"  Symbols: {total_stats[1]}")
        print(f"  Date Range: {total_stats[2]} to {total_stats[3]}")
        
        # シンボル別統計
        print(f"\n📈 Symbol Statistics:")
        for symbol, report in self.stats['quality_report'].items():
            print(f"\n  {symbol}:")
            print(f"    Records: {report['total_records']:,}")
            print(f"    Date Range: {report['date_range']}")
            print(f"    Bull/Bear Ratio: {report['bullish_percentage']:.1f}% / {100-report['bullish_percentage']:.1f}%")
            print(f"    Data Completeness: {report['completeness']:.1f}%")
            print(f"    Average Price: ${report['average_price']:.2f}")
        
        # データ分布
        print(f"\n📅 Yearly Distribution:")
        year_dist = self.conn.execute("""
            SELECT 
                YEAR(timestamp) as year,
                COUNT(*) as records,
                COUNT(DISTINCT symbol) as symbols
            FROM all_klines
            GROUP BY YEAR(timestamp)
            ORDER BY year
        """).fetchall()
        
        for year, records, symbols in year_dist:
            print(f"  {year}: {records:,} records ({symbols} symbols)")
        
        # メタデータを保存
        metadata = {
            'merge_date': datetime.now().isoformat(),
            'total_records': int(total_stats[0]),
            'symbols': self.symbols,
            'date_range': {
                'start': str(total_stats[2]),
                'end': str(total_stats[3])
            },
            'quality_report': self.stats['quality_report'],
            'yearly_distribution': {str(y): int(r) for y, r, _ in year_dist}
        }
        
        with open('data/merged_data_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Metadata saved to data/merged_data_metadata.json")
        print("="*60)
    
    def run(self):
        """メイン処理を実行"""
        try:
            # 1. 接続
            self.connect()
            
            # 2. 既存データ分析
            self.analyze_existing_data()
            
            # 3. 統合テーブル作成
            self.create_unified_table()
            
            # 4. データ統合
            self.merge_historical_data()
            
            # 5. 最新データで更新
            self.update_from_redis()
            
            # 6. データ品質検証
            self.verify_data_quality()
            
            # 7. サマリーレポート生成
            self.generate_summary_report()
            
            logger.info("\n✅ Data merge completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in data merge process: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()
            if self.redis_client:
                self.redis_client.close()


if __name__ == "__main__":
    # 出力ディレクトリ確認
    os.makedirs("data", exist_ok=True)
    
    # データ統合を実行
    merger = HistoricalDataMerger()
    merger.run()