#!/usr/bin/env python3
"""
Real data collection test script for system validation.

This script:
1. Connects to Bybit Testnet 
2. Collects real market data
3. Tests the entire data pipeline
4. Validates feature generation
5. Measures performance metrics
"""

import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import settings
from src.common.logging import setup_logging, get_logger, TradingLogger
from src.common.database import init_databases, close_databases
from src.common.monitoring import start_monitoring
from src.ingestor.main import BybitIngestor
from src.feature_hub.main import FeatureHub

# Setup logging
setup_logging()
logger = get_logger(__name__)
trading_logger = TradingLogger("test_data_collection")

class DataCollectionTest:
    """
    Comprehensive data collection and validation test.
    
    Tests:
    - Bybit API connectivity
    - WebSocket data streaming
    - Feature computation pipeline
    - Data archiving
    - Performance metrics
    """
    
    def __init__(self, test_duration_minutes: int = 30):
        """
        Initialize test environment.
        
        Args:
            test_duration_minutes: How long to run the test
        """
        self.test_duration = test_duration_minutes * 60  # Convert to seconds
        self.running = False
        self.start_time = None
        
        # Test results
        self.test_results = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "messages_received": 0,
            "features_computed": 0,
            "errors_encountered": 0,
            "performance_metrics": {},
            "data_quality": {},
            "success": False
        }
        
        # Components
        self.ingestor = None
        self.feature_hub = None
        self.health_checker = None
        self.metrics_collector = None
        
        # Monitoring data
        self.message_counts = {"kline": 0, "orderbook": 0, "trades": 0, "liquidation": 0}
        self.feature_counts = {}
        self.error_log = []
        
    async def run_test(self):
        """Run the comprehensive data collection test."""
        logger.info(
            "Starting data collection test",
            duration_minutes=self.test_duration / 60,
            symbols=settings.bybit.symbols,
            testnet=settings.bybit.testnet
        )
        
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Initialize test
            await self._initialize_test()
            
            # Run main test loop
            await self._run_test_loop()
            
            # Finalize test
            await self._finalize_test()
            
        except Exception as e:
            logger.error("Test failed with exception", exception=e)
            self.test_results["success"] = False
            self.test_results["errors_encountered"] += 1
            self.error_log.append(f"Critical error: {e}")
        
        finally:
            await self._cleanup()
    
    async def _initialize_test(self):
        """Initialize test environment and components."""
        logger.info("Initializing test environment")
        
        self.start_time = time.time()
        self.test_results["start_time"] = datetime.utcnow().isoformat()
        self.running = True
        
        try:
            # Initialize databases
            await init_databases()
            logger.info("✅ Databases initialized")
            
            # Start monitoring
            self.health_checker, self.metrics_collector = await start_monitoring()
            logger.info("✅ Monitoring started")
            
            # Verify Bybit connectivity
            await self._test_bybit_connectivity()
            
            # Initialize components
            self.ingestor = BybitIngestor()
            self.feature_hub = FeatureHub()
            
            logger.info("✅ Test environment initialized")
            
        except Exception as e:
            logger.error("Failed to initialize test environment", exception=e)
            raise
    
    async def _test_bybit_connectivity(self):
        """Test basic Bybit API connectivity."""
        logger.info("Testing Bybit API connectivity")
        
        try:
            from src.common.bybit_client import BybitRESTClient
            
            async with BybitRESTClient(testnet=True) as rest_client:
                # Test basic connectivity with OI data
                oi_data = await rest_client.get_open_interest(["BTCUSDT"])
                
                if oi_data:
                    logger.info("✅ Bybit REST API connectivity confirmed", oi_data=oi_data)
                else:
                    logger.warning("⚠️ Bybit REST API returned empty data")
                
                # Test funding rate
                funding_data = await rest_client.get_funding_rate(["BTCUSDT"])
                if funding_data:
                    logger.info("✅ Bybit funding rate API confirmed", funding_data=funding_data)
        
        except Exception as e:
            logger.error("Bybit connectivity test failed", exception=e)
            raise
    
    async def _run_test_loop(self):
        """Run the main test loop."""
        logger.info(f"Starting main test loop for {self.test_duration} seconds")
        
        # Start ingestor and feature hub
        ingestor_task = asyncio.create_task(self._run_ingestor())
        feature_hub_task = asyncio.create_task(self._run_feature_hub())
        monitor_task = asyncio.create_task(self._monitor_progress())
        
        try:
            # Wait for test duration or until stopped
            await asyncio.sleep(self.test_duration)
            
        except asyncio.CancelledError:
            logger.info("Test loop cancelled")
        
        finally:
            # Stop all tasks
            for task in [ingestor_task, feature_hub_task, monitor_task]:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def _run_ingestor(self):
        """Run the data ingestor."""
        try:
            logger.info("Starting data ingestor")
            await self.ingestor.start()
        except Exception as e:
            logger.error("Ingestor failed", exception=e)
            self.error_log.append(f"Ingestor error: {e}")
            self.test_results["errors_encountered"] += 1
    
    async def _run_feature_hub(self):
        """Run the feature hub."""
        try:
            # Wait a bit for ingestor to start producing data
            await asyncio.sleep(10)
            logger.info("Starting feature hub")
            await self.feature_hub.start()
        except Exception as e:
            logger.error("Feature hub failed", exception=e)
            self.error_log.append(f"Feature hub error: {e}")
            self.test_results["errors_encountered"] += 1
    
    async def _monitor_progress(self):
        """Monitor test progress and collect metrics."""
        logger.info("Starting progress monitoring")
        
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Collect current metrics
                await self._collect_progress_metrics()
                
                # Log progress
                elapsed = time.time() - self.start_time
                remaining = max(0, self.test_duration - elapsed)
                
                logger.info(
                    "Test progress update",
                    elapsed_minutes=elapsed / 60,
                    remaining_minutes=remaining / 60,
                    messages_received=self.test_results["messages_received"],
                    features_computed=self.test_results["features_computed"],
                    errors=self.test_results["errors_encountered"]
                )
                
            except Exception as e:
                logger.warning("Error in progress monitoring", exception=e)
                await asyncio.sleep(10)
    
    async def _collect_progress_metrics(self):
        """Collect current progress metrics."""
        try:
            # Get ingestor stats
            if self.ingestor:
                self.test_results["messages_received"] = getattr(self.ingestor, 'message_count', 0)
            
            # Get feature hub stats
            if self.feature_hub:
                self.test_results["features_computed"] = getattr(self.feature_hub, 'features_computed', 0)
                
                # Get feature summary
                feature_summary = self.feature_hub.get_feature_summary()
                self.feature_counts = feature_summary.get("feature_counts", {})
            
            # Get health status
            if self.health_checker:
                health_status = self.health_checker.get_status()
                self.test_results["system_health"] = health_status
        
        except Exception as e:
            logger.warning("Error collecting progress metrics", exception=e)
    
    async def _finalize_test(self):
        """Finalize test and collect final results."""
        logger.info("Finalizing test")
        
        self.running = False
        end_time = time.time()
        
        self.test_results.update({
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": end_time - self.start_time,
            "success": self.test_results["errors_encountered"] == 0
        })
        
        # Collect final metrics
        await self._collect_final_metrics()
        
        # Validate data quality
        await self._validate_data_quality()
        
        # Generate test report
        await self._generate_test_report()
    
    async def _collect_final_metrics(self):
        """Collect final performance metrics."""
        try:
            duration = self.test_results["duration_seconds"]
            
            # Calculate rates
            message_rate = self.test_results["messages_received"] / duration if duration > 0 else 0
            feature_rate = self.test_results["features_computed"] / duration if duration > 0 else 0
            
            self.test_results["performance_metrics"] = {
                "message_rate_per_second": message_rate,
                "feature_rate_per_second": feature_rate,
                "total_symbols_processed": len(self.feature_counts),
                "avg_features_per_symbol": sum(self.feature_counts.values()) / len(self.feature_counts) if self.feature_counts else 0,
                "error_rate": self.test_results["errors_encountered"] / duration if duration > 0 else 0
            }
            
            logger.info("Final performance metrics", **self.test_results["performance_metrics"])
            
        except Exception as e:
            logger.warning("Error collecting final metrics", exception=e)
    
    async def _validate_data_quality(self):
        """Validate the quality of collected data."""
        try:
            from src.common.database import get_redis_client
            
            redis_client = await get_redis_client()
            
            # Check Redis streams
            data_quality = {}
            
            for stream_type in ["kline", "orderbook", "trades", "liquidation"]:
                stream_name = f"market_data:{stream_type}"
                try:
                    info = await redis_client.xinfo_stream(stream_name)
                    data_quality[stream_type] = {
                        "length": info.get("length", 0),
                        "first_entry": info.get("first-entry", ["", {}])[0] if info.get("first-entry") else None,
                        "last_entry": info.get("last-entry", ["", {}])[0] if info.get("last-entry") else None
                    }
                except Exception as e:
                    logger.warning(f"Could not get info for stream {stream_name}: {e}")
                    data_quality[stream_type] = {"error": str(e)}
            
            # Check feature streams
            for symbol in settings.bybit.symbols:
                feature_stream = f"features:{symbol}:latest"
                try:
                    info = await redis_client.xinfo_stream(feature_stream)
                    data_quality[f"features_{symbol}"] = {
                        "length": info.get("length", 0),
                        "last_entry": info.get("last-entry", ["", {}])[0] if info.get("last-entry") else None
                    }
                except Exception as e:
                    logger.debug(f"Feature stream {feature_stream} may not exist yet: {e}")
            
            self.test_results["data_quality"] = data_quality
            logger.info("Data quality validation completed", data_quality=data_quality)
            
        except Exception as e:
            logger.warning("Error validating data quality", exception=e)
    
    async def _generate_test_report(self):
        """Generate comprehensive test report."""
        report = {
            "test_summary": {
                "success": self.test_results["success"],
                "duration_minutes": self.test_results["duration_seconds"] / 60,
                "messages_received": self.test_results["messages_received"],
                "features_computed": self.test_results["features_computed"],
                "errors_encountered": self.test_results["errors_encountered"]
            },
            "performance_metrics": self.test_results["performance_metrics"],
            "data_quality": self.test_results["data_quality"],
            "feature_counts": self.feature_counts,
            "error_log": self.error_log,
            "configuration": {
                "symbols": settings.bybit.symbols,
                "testnet": settings.bybit.testnet,
                "test_duration_minutes": self.test_duration / 60
            }
        }
        
        # Save report to file
        report_file = Path("data") / f"test_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Log summary
        logger.info(
            "✅ Test completed",
            success=report["test_summary"]["success"],
            duration_minutes=report["test_summary"]["duration_minutes"],
            message_rate=report["performance_metrics"].get("message_rate_per_second", 0),
            feature_rate=report["performance_metrics"].get("feature_rate_per_second", 0),
            report_file=str(report_file)
        )
        
        if report["test_summary"]["success"]:
            trading_logger.logger.info(
                "Data collection test PASSED",
                **report["test_summary"]
            )
        else:
            trading_logger.logger.error(
                "Data collection test FAILED",
                **report["test_summary"],
                errors=self.error_log
            )
    
    async def _cleanup(self):
        """Cleanup test environment."""
        logger.info("Cleaning up test environment")
        
        try:
            # Stop components
            if self.ingestor:
                await self.ingestor.stop()
            
            if self.feature_hub:
                await self.feature_hub.stop()
            
            # Stop monitoring
            if self.metrics_collector:
                await self.metrics_collector.stop()
            
            # Close databases
            await close_databases()
            
            logger.info("✅ Test cleanup completed")
            
        except Exception as e:
            logger.warning("Error during cleanup", exception=e)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, stopping test...")
        self.running = False


async def main():
    """Main entry point for data collection test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data collection test")
    parser.add_argument(
        "--duration", 
        type=int, 
        default=30,
        help="Test duration in minutes (default: 30)"
    )
    args = parser.parse_args()
    
    # Create and run test
    test = DataCollectionTest(test_duration_minutes=args.duration)
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())