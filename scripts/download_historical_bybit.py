#!/usr/bin/env python3
"""
Download historical data from Bybit for ML training.
"""

import asyncio
import aiohttp
import gzip
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging
from src.storage.duckdb_manager import DuckDBManager

setup_logging()
logger = get_logger(__name__)


class BybitHistoricalDownloader:
    """Download historical data from Bybit."""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        """
        Initialize downloader.
        
        Args:
            symbols: List of symbols to download (e.g., ["BTCUSDT", "ETHUSDT", "ICPUSDT"])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.symbols = symbols
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.base_url = "https://api.bybit.com"
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def download_klines(self, symbol: str, interval: str = "1") -> pd.DataFrame:
        """
        Download kline (candlestick) data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        
        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.base_url}/v5/market/kline"
        
        all_data = []
        end_time = int(self.end_date.timestamp() * 1000)
        start_time = int(self.start_date.timestamp() * 1000)
        
        logger.info(f"Downloading {symbol} klines from {self.start_date} to {self.end_date}")
        
        while end_time > start_time:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": 1000,  # Max limit
                "end": end_time
            }
            
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("retCode") == 0:
                            klines = data.get("result", {}).get("list", [])
                            
                            if not klines:
                                break
                            
                            all_data.extend(klines)
                            
                            # Update end_time to the oldest timestamp for next request
                            oldest_timestamp = int(klines[-1][0])
                            if oldest_timestamp <= end_time:
                                end_time = oldest_timestamp - 1
                            else:
                                break
                            
                            logger.debug(f"Downloaded {len(klines)} klines for {symbol}, earliest: {datetime.fromtimestamp(oldest_timestamp/1000)}")
                            
                            # Rate limiting
                            await asyncio.sleep(0.1)
                        else:
                            logger.error(f"API error: {data.get('retMsg')}")
                            break
                    else:
                        logger.error(f"HTTP error: {response.status}")
                        break
                        
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                break
        
        if all_data:
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit='ms')
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = df[col].astype(float)
            
            # Add symbol column
            df["symbol"] = symbol
            
            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Filter by date range
            df = df[(df["timestamp"] >= self.start_date) & (df["timestamp"] <= self.end_date)]
            
            logger.info(f"Downloaded {len(df)} klines for {symbol}")
            return df
        else:
            logger.warning(f"No data downloaded for {symbol}")
            return pd.DataFrame()
    
    async def download_trades(self, symbol: str, limit_per_request: int = 1000) -> pd.DataFrame:
        """Download trade data (limited history available)."""
        url = f"{self.base_url}/v5/market/recent-trade"
        
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": limit_per_request
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("retCode") == 0:
                        trades = data.get("result", {}).get("list", [])
                        
                        if trades:
                            df = pd.DataFrame(trades)
                            df["time"] = pd.to_datetime(df["time"].astype(int), unit='ms')
                            df["price"] = df["price"].astype(float)
                            df["size"] = df["size"].astype(float)
                            
                            logger.info(f"Downloaded {len(df)} recent trades for {symbol}")
                            return df
                        
        except Exception as e:
            logger.error(f"Error downloading trades for {symbol}: {e}")
        
        return pd.DataFrame()
    
    async def download_all_symbols(self) -> Dict[str, pd.DataFrame]:
        """Download data for all symbols."""
        results = {}
        
        for symbol in self.symbols:
            logger.info(f"Downloading data for {symbol}")
            
            # Download kline data
            kline_df = await self.download_klines(symbol)
            if not kline_df.empty:
                results[symbol] = kline_df
            
            # Small delay between symbols
            await asyncio.sleep(0.5)
        
        return results
    
    def save_to_duckdb(self, data: Dict[str, pd.DataFrame], db_path: str = "data/historical_data.duckdb"):
        """Save downloaded data to DuckDB."""
        manager = DuckDBManager(db_path=db_path)
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            table_name = f"klines_{symbol.lower()}"
            
            # Create table
            manager.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp TIMESTAMP,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    turnover DOUBLE,
                    symbol VARCHAR
                )
            """)
            
            # Insert data
            manager.conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            logger.info(f"Saved {len(df)} records for {symbol} to {table_name}")
        
        # Create a unified view
        all_tables = [f"klines_{s.lower()}" for s in self.symbols if s in data and not data[s].empty]
        if all_tables:
            union_query = " UNION ALL ".join([f"SELECT * FROM {t}" for t in all_tables])
            manager.conn.execute(f"CREATE OR REPLACE VIEW all_klines AS {union_query}")
            logger.info("Created unified view 'all_klines'")
        
        # Show summary
        result = manager.conn.execute("SELECT symbol, COUNT(*) as count, MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM all_klines GROUP BY symbol").df()
        print("\n📊 Data Summary:")
        print(result.to_string())
        
        manager.conn.close()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download historical data from Bybit")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "ICPUSDT"], help="Symbols to download")
    parser.add_argument("--start-date", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="1", help="Kline interval (1=1min, 5=5min, etc)")
    
    args = parser.parse_args()
    
    print(f"📥 Downloading historical data")
    print(f"   Symbols: {args.symbols}")
    print(f"   Period: {args.start_date} to {args.end_date}")
    print(f"   Interval: {args.interval} minute(s)")
    
    async with BybitHistoricalDownloader(args.symbols, args.start_date, args.end_date) as downloader:
        data = await downloader.download_all_symbols()
        
        if data:
            downloader.save_to_duckdb(data)
            print("\n✅ Download completed successfully!")
        else:
            print("\n❌ No data downloaded")


if __name__ == "__main__":
    asyncio.run(main())