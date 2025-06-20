#!/usr/bin/env python3
"""
Script to check collected data statistics and quality.

Usage:
    python scripts/check_data.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.logging import get_logger, setup_logging
from src.storage.duckdb_manager import DuckDBManager

setup_logging()
logger = get_logger(__name__)


async def check_data():
    """Check collected data statistics and quality."""
    storage_manager = DuckDBManager()
    
    try:
        # Get overall summary
        summary = await storage_manager.get_data_summary()
        
        if not summary or summary.get("total_records", 0) == 0:
            logger.warning("No data found in storage!")
            return
        
        logger.info("=" * 60)
        logger.info("DATA COLLECTION SUMMARY")
        logger.info("=" * 60)
        
        # Basic statistics
        logger.info(f"Total records: {summary['total_records']:,}")
        logger.info(f"Symbols: {', '.join(summary['symbols'])}")
        logger.info(f"Time range: {summary['time_range']['start']} to {summary['time_range']['end']}")
        
        # Data types breakdown
        logger.info("\nData Types:")
        for data_type, count in summary['data_types'].items():
            logger.info(f"  - {data_type}: {count:,} records")
        
        # Per symbol statistics
        logger.info("\nPer Symbol Statistics:")
        for symbol in summary['symbols']:
            symbol_data = await storage_manager.get_symbol_statistics(symbol)
            
            logger.info(f"\n{symbol}:")
            logger.info(f"  - Klines: {symbol_data.get('klines', 0):,}")
            logger.info(f"  - Trades: {symbol_data.get('trades', 0):,}")
            logger.info(f"  - Orderbook updates: {symbol_data.get('orderbook', 0):,}")
            logger.info(f"  - Liquidations: {symbol_data.get('liquidations', 0):,}")
            
            # Liquidation statistics
            if symbol_data.get('liquidations', 0) > 0:
                liq_stats = await storage_manager.get_liquidation_statistics(symbol)
                logger.info(f"  - Liquidation Stats:")
                logger.info(f"    * Total volume: ${liq_stats['total_volume']:,.2f}")
                logger.info(f"    * Avg size: ${liq_stats['avg_size']:,.2f}")
                logger.info(f"    * Max size: ${liq_stats['max_size']:,.2f}")
                logger.info(f"    * Long/Short ratio: {liq_stats['long_ratio']:.2%}/{liq_stats['short_ratio']:.2%}")
        
        # Data quality checks
        logger.info("\n" + "=" * 60)
        logger.info("DATA QUALITY CHECKS")
        logger.info("=" * 60)
        
        quality_issues = []
        
        # Check for gaps in time series
        for symbol in summary['symbols']:
            gaps = await storage_manager.check_time_gaps(symbol)
            if gaps:
                quality_issues.append(f"{symbol}: {len(gaps)} time gaps detected")
                for gap in gaps[:3]:  # Show first 3 gaps
                    logger.warning(f"  - Gap: {gap['duration']} seconds at {gap['timestamp']}")
        
        # Check for data anomalies
        anomalies = await storage_manager.check_data_anomalies()
        if anomalies:
            quality_issues.extend(anomalies)
        
        if quality_issues:
            logger.warning(f"\nFound {len(quality_issues)} quality issues:")
            for issue in quality_issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✓ No quality issues detected!")
        
        # Feature readiness check
        logger.info("\n" + "=" * 60)
        logger.info("ML READINESS CHECK")
        logger.info("=" * 60)
        
        # Check if we have enough data for ML
        min_records_for_ml = 10000  # Minimum records needed
        min_liquidations = 100      # Minimum liquidation events
        
        ml_ready = True
        
        if summary['total_records'] < min_records_for_ml:
            logger.warning(f"❌ Insufficient data: {summary['total_records']:,} < {min_records_for_ml:,} records")
            ml_ready = False
        else:
            logger.info(f"✓ Sufficient records: {summary['total_records']:,}")
        
        total_liquidations = sum(summary['data_types'].get('liquidation', 0) for _ in summary['symbols'])
        if total_liquidations < min_liquidations:
            logger.warning(f"❌ Insufficient liquidations: {total_liquidations} < {min_liquidations}")
            ml_ready = False
        else:
            logger.info(f"✓ Sufficient liquidations: {total_liquidations}")
        
        # Check data recency
        latest_time = datetime.fromisoformat(summary['time_range']['end'])
        age_hours = (datetime.now() - latest_time).total_seconds() / 3600
        
        if age_hours > 24:
            logger.warning(f"⚠️  Data is {age_hours:.1f} hours old")
        else:
            logger.info(f"✓ Data is recent ({age_hours:.1f} hours old)")
        
        if ml_ready:
            logger.info("\n✅ Data is ready for ML training!")
            logger.info("Next steps:")
            logger.info("  1. Run feature engineering: python scripts/prepare_features.py")
            logger.info("  2. Train model: python scripts/train_model.py")
        else:
            logger.info("\n⚠️  Need more data for ML training")
            logger.info("Recommendations:")
            logger.info("  1. Collect more data: python scripts/collect_data.py --duration 48")
            logger.info("  2. Use mainnet for more liquidations: python scripts/collect_data.py --mainnet")
        
    except Exception as e:
        logger.error(f"Error checking data: {e}")
        raise
    finally:
        await storage_manager.close()


async def main():
    """Main entry point."""
    await check_data()


if __name__ == "__main__":
    asyncio.run(main())