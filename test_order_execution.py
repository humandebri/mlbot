#!/usr/bin/env python3
"""
Test order execution flow to identify where the issue occurs.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from order_router.smart_router import SmartRouter, TradingSignal, RoutingConfig
from order_router.order_executor import OrderExecutor, ExecutionConfig
from order_router.risk_manager import RiskManager, RiskConfig
from order_router.position_manager import PositionManager
from common.bybit_client import BybitRESTClient
from common.config import settings
from common.logging import get_logger

logger = get_logger(__name__)

async def test_order_execution():
    """Test the complete order execution flow."""
    
    print("=== Testing Order Execution Flow ===")
    
    # Initialize components
    print("1. Initializing components...")
    
    # Mock client for testing
    class MockBybitClient:
        async def create_order(self, **kwargs):
            print(f"MockBybitClient.create_order called with: {kwargs}")
            return {"orderId": "TEST_ORDER_123", "status": "NEW"}
        
        def get_ticker(self, symbol):
            return {"lastPrice": "50000", "bid": "49995", "ask": "50005"}
    
    try:
        mock_client = MockBybitClient()
        
        # Initialize components with mock client
        risk_manager = RiskManager(RiskConfig())
        position_manager = PositionManager()
        order_executor = OrderExecutor(mock_client, ExecutionConfig())
        smart_router = SmartRouter(order_executor, risk_manager, position_manager, RoutingConfig())
        
        print("✓ Components initialized")
        
        # Start components
        print("2. Starting components...")
        await order_executor.start()
        print("✓ Order executor started")
        
        # Create test signal
        print("3. Creating test signal...")
        signal = TradingSignal(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            prediction=0.002,  # 0.2% expected PnL
            confidence=0.75,   # 75% confidence
            features={"price": 50000.0, "volume": 1000.0},
            liquidation_detected=False,
            liquidation_size=0.0,
            liquidation_side=""
        )
        print(f"✓ Created signal: {signal.symbol}, pred={signal.prediction}, conf={signal.confidence}")
        
        # Process signal through smart router
        print("4. Processing signal through smart router...")
        position_id = await smart_router.process_signal(signal)
        
        if position_id:
            print(f"✓ Signal processed successfully! Position ID: {position_id}")
        else:
            print("✗ Signal processing failed - returned None")
            
            # Debug: Check routing stats
            stats = smart_router.get_routing_stats()
            print(f"Routing stats: {stats}")
            
        # Stop components
        print("5. Stopping components...")
        await order_executor.stop()
        print("✓ Order executor stopped")
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_order_execution())