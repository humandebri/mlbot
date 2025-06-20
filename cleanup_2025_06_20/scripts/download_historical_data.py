#!/usr/bin/env python3
"""
Download historical data from various sources.

Usage:
    python scripts/download_historical_data.py --symbol BTCUSDT --start 2022-01-01
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
import gzip
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class HistoricalDataDownloader:
    """Download historical data from multiple sources."""
    
    def __init__(self):
        self.data_dir = project_root / "data" / "historical"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def download_bybit_historical(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str = None
    ):
        """
        Download historical data from Bybit.
        
        Note: Bybit provides limited historical data via API.
        For extensive historical data, consider:
        1. Bybit Data Download Portal (manual download)
        2. Third-party data providers
        """
        logger.info(f"Downloading Bybit historical data for {symbol}")
        
        # Bybit REST API for klines (limited to recent data)
        base_url = "https://api.bybit.com/v5/market/kline"
        
        interval = "1"  # 1 minute bars
        limit = 1000    # Max per request
        
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date or datetime.now()).timestamp() * 1000)
        
        all_data = []
        current_end = end_ts
        
        while current_end > start_ts:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "start": start_ts,
                "end": current_end,
                "limit": limit
            }
            
            try:
                response = requests.get(base_url, params=params)
                data = response.json()
                
                if data["retCode"] == 0:
                    klines = data["result"]["list"]
                    if not klines:
                        break
                    
                    all_data.extend(klines)
                    
                    # Update end time for next batch
                    oldest_time = int(klines[-1][0])
                    if oldest_time == current_end:
                        break
                    current_end = oldest_time - 1
                    
                    logger.info(f"Downloaded {len(klines)} records, total: {len(all_data)}")
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                else:
                    logger.error(f"API error: {data}")
                    break
                    
            except Exception as e:
                logger.error(f"Download error: {e}")
                break
        
        if all_data:
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            
            # Convert string prices to float
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = df[col].astype(float)
            
            # Save to parquet
            output_file = self.data_dir / f"{symbol}_klines_{start_date}.parquet"
            df.to_parquet(output_file, index=False)
            
            logger.info(f"Saved {len(df)} records to {output_file}")
            return df
        
        return None
    
    async def download_binance_historical(
        self, 
        symbol: str, 
        start_date: str,
        data_type: str = "klines"
    ):
        """
        Download historical data from Binance Data.
        
        Binance provides extensive historical data:
        https://data.binance.vision/
        """
        logger.info(f"Downloading Binance historical data for {symbol}")
        
        # Convert symbol format (BTCUSDT stays the same)
        binance_symbol = symbol
        
        # Binance data download URL pattern
        base_url = "https://data.binance.vision/data/futures/um/daily"
        
        start = pd.Timestamp(start_date)
        end = pd.Timestamp.now()
        
        all_files = []
        current_date = start
        
        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            
            if data_type == "klines":
                # 1m klines
                url = f"{base_url}/klines/{binance_symbol}/1m/{binance_symbol}-1m-{date_str}.zip"
            elif data_type == "trades":
                url = f"{base_url}/trades/{binance_symbol}/{binance_symbol}-trades-{date_str}.zip"
            elif data_type == "liquidation":
                # Note: Liquidation data might have different format
                url = f"{base_url}/liquidationSnapshot/{binance_symbol}/{binance_symbol}-liquidationSnapshot-{date_str}.zip"
            
            try:
                logger.info(f"Downloading {date_str}...")
                response = requests.get(url, stream=True)
                
                if response.status_code == 200:
                    # Save zip file
                    zip_file = self.data_dir / f"{binance_symbol}_{data_type}_{date_str}.zip"
                    with open(zip_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    all_files.append(zip_file)
                    logger.info(f"Downloaded {zip_file.name}")
                else:
                    logger.debug(f"No data for {date_str}")
                    
            except Exception as e:
                logger.error(f"Error downloading {date_str}: {e}")
            
            # Move to next day
            current_date += timedelta(days=1)
            
            # Rate limiting
            await asyncio.sleep(0.5)
        
        # Process downloaded files
        if all_files:
            logger.info(f"Processing {len(all_files)} files...")
            df = await self._process_binance_files(all_files, data_type)
            
            if df is not None:
                output_file = self.data_dir / f"{symbol}_{data_type}_binance_{start_date}.parquet"
                df.to_parquet(output_file, index=False)
                logger.info(f"Saved {len(df)} records to {output_file}")
                
                # Clean up zip files
                for f in all_files:
                    f.unlink()
                
                return df
        
        return None
    
    async def _process_binance_files(self, files: list, data_type: str):
        """Process downloaded Binance zip files."""
        import zipfile
        
        all_data = []
        
        for zip_file in files:
            try:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    for filename in zf.namelist():
                        if filename.endswith('.csv'):
                            with zf.open(filename) as f:
                                df = pd.read_csv(f)
                                all_data.append(df)
            except Exception as e:
                logger.error(f"Error processing {zip_file}: {e}")
        
        if all_data:
            # Combine all DataFrames
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Process based on data type
            if data_type == "klines":
                # Rename columns to match our format
                combined_df.columns = [
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_volume",
                    "taker_buy_quote_volume", "ignore"
                ]
                combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], unit="ms")
                
            elif data_type == "trades":
                combined_df["timestamp"] = pd.to_datetime(combined_df["time"], unit="ms")
                
            return combined_df
        
        return None
    
    async def download_alternative_sources(self, symbol: str, start_date: str):
        """
        Alternative data sources for historical crypto data.
        """
        sources = {
            "CryptoDataDownload": "https://www.cryptodatadownload.com",
            "Kaggle": "https://www.kaggle.com/datasets",
            "CoinGecko": "https://www.coingecko.com/api",
            "CryptoCompare": "https://min-api.cryptocompare.com",
            "Kaiko": "https://www.kaiko.com",  # Premium
            "Tardis": "https://tardis.dev",     # Premium
        }
        
        logger.info("Alternative historical data sources:")
        for name, url in sources.items():
            logger.info(f"  - {name}: {url}")
        
        # Example: CryptoCompare API (free tier available)
        logger.info("\nExample: Downloading from CryptoCompare...")
        
        # Note: You need to register for a free API key
        api_key = "YOUR_CRYPTOCOMPARE_API_KEY"
        
        if api_key == "YOUR_CRYPTOCOMPARE_API_KEY":
            logger.warning("Please set a valid CryptoCompare API key")
            return None
        
        # ... Implementation for CryptoCompare download ...
        
    def estimate_data_size(self, days: int, symbols: int = 3):
        """Estimate storage requirements for historical data."""
        # Rough estimates
        klines_per_day = 24 * 60  # 1-minute bars
        trades_per_day = 100000   # Varies greatly
        orderbook_per_day = 500000  # Snapshots
        
        total_records = days * symbols * (klines_per_day + trades_per_day + orderbook_per_day)
        
        # Assume 100 bytes per record average
        size_gb = (total_records * 100) / (1024**3)
        
        logger.info(f"Estimated storage for {days} days, {symbols} symbols:")
        logger.info(f"  - Total records: {total_records:,}")
        logger.info(f"  - Storage size: ~{size_gb:.1f} GB")
        
        return size_gb


async def main():
    parser = argparse.ArgumentParser(description="Download historical crypto data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--source", choices=["bybit", "binance", "all"], default="bybit")
    parser.add_argument("--estimate-only", action="store_true", help="Only estimate data size")
    
    args = parser.parse_args()
    
    downloader = HistoricalDataDownloader()
    
    if args.estimate_only:
        # Estimate data size
        days = (pd.Timestamp.now() - pd.Timestamp(args.start)).days
        downloader.estimate_data_size(days)
        return
    
    # Download data
    if args.source in ["bybit", "all"]:
        await downloader.download_bybit_historical(args.symbol, args.start, args.end)
    
    if args.source in ["binance", "all"]:
        await downloader.download_binance_historical(args.symbol, args.start, "klines")
        # Also download liquidation data if available
        await downloader.download_binance_historical(args.symbol, args.start, "liquidation")
    
    # Show alternative sources
    await downloader.download_alternative_sources(args.symbol, args.start)


if __name__ == "__main__":
    asyncio.run(main())