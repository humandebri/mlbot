#!/usr/bin/env python3
"""
Account balance monitor test script.
Tests real-time balance tracking and position size adjustment.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def test_balance_monitor():
    """Test account balance monitoring (placeholder)."""
    print("=" * 60)
    print("Account Balance Monitor Test")
    print("=" * 60)
    print("⚠️  Real API test requires dependencies")
    print("Use production environment for full testing")


async def test_compound_calculation():
    """Test compound position size calculation."""
    
    print("\n" + "=" * 60)
    print("Compound Position Size Simulation")
    print("=" * 60)
    
    # Simulate balance growth
    initial_balance = 500
    monthly_return = 0.0416  # 4.16%
    position_pct = 0.05  # 5%
    
    print(f"Initial Balance: ${initial_balance}")
    print(f"Monthly Return: {monthly_return*100:.2f}%")
    print(f"Position Size: {position_pct*100:.1f}%")
    
    balance = initial_balance
    print(f"\nMonth | Balance | Position Size | Growth")
    print("-" * 45)
    
    for month in range(1, 13):
        # Apply monthly return
        balance *= (1 + monthly_return)
        position_size = balance * position_pct
        
        print(f"{month:5d} | ${balance:7.2f} | ${position_size:11.2f} | {(balance/initial_balance-1)*100:+5.1f}%")
    
    total_return = (balance - initial_balance) / initial_balance * 100
    print(f"\nFinal Result: ${balance:.2f} (+{total_return:.1f}%)")


if __name__ == "__main__":
    print("🤖 MLBot Balance Monitor Test")
    print("Requires valid Bybit API credentials in environment")
    print("")
    
    # Test compound calculation first (no API needed)
    asyncio.run(test_compound_calculation())
    
    # Ask user if they want to test real API
    response = input("\nTest real Bybit API balance monitoring? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(test_balance_monitor())
    else:
        print("Skipping API test")