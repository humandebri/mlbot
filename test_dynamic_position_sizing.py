#!/usr/bin/env python3
"""
Test dynamic position sizing with different account balances.
"""
import sys
sys.path.insert(0, '/Users/0xhude/Desktop/mlbot')

from src.order_router.dynamic_risk_config import DynamicRiskConfig
from src.order_router.enhanced_risk_manager import EnhancedRiskManager
from src.common.account_monitor import AccountBalance
from datetime import datetime
import json


def test_dynamic_sizing():
    """Test dynamic position sizing with various account sizes."""
    
    # Create dynamic config
    dynamic_config = DynamicRiskConfig()
    
    # Test with different account sizes
    test_balances = [
        100,       # $100 (minimum)
        1000,      # $1,000
        10000,     # $10,000
        50000,     # $50,000
        100000,    # $100,000
        500000,    # $500,000
        1000000,   # $1,000,000
    ]
    
    print("=" * 80)
    print("動的ポジション管理システムのデモンストレーション")
    print("=" * 80)
    print()
    
    for balance in test_balances:
        print(f"\n📊 口座残高: ${balance:,.2f}")
        print("-" * 60)
        
        # Calculate dynamic limits
        limits = dynamic_config.calculate_dynamic_limits(balance)
        
        # Display key metrics
        print(f"✅ 最大ポジションサイズ/シンボル: ${limits['max_position_size_usd']:,.2f} ({dynamic_config.max_position_pct_per_symbol * 100:.0f}%)")
        print(f"✅ 最大総エクスポージャー: ${limits['max_total_exposure_usd']:,.2f} ({dynamic_config.max_total_exposure_pct * 100:.0f}%)")
        print(f"✅ 1取引あたりリスク: ${limits['risk_per_trade_usd']:,.2f} ({dynamic_config.risk_per_trade_pct * 100:.0f}%)")
        print(f"✅ 最大同時ポジション数: {limits['max_positions']}")
        print(f"✅ 最大相関ポジション数: {limits['max_correlated_positions']}")
        print(f"✅ 日次損失上限: ${limits['max_daily_loss_usd']:,.2f} ({dynamic_config.max_daily_loss_pct * 100:.0f}%)")
        print(f"✅ サーキットブレーカー: ${limits['circuit_breaker_usd']:,.2f} ({dynamic_config.circuit_breaker_loss_pct * 100:.0f}%)")
        print(f"✅ 時間あたり最大取引数: {limits['max_trades_per_hour']}")
        print(f"✅ 1日あたり最大取引数: {limits['max_trades_per_day']}")
        
        # With 3x leverage
        print(f"\n💹 レバレッジ3倍での最大ポジション:")
        print(f"   • 最大レバレッジエクスポージャー: ${limits['max_leveraged_exposure']:,.2f}")
        print(f"   • 目標レバレッジエクスポージャー: ${limits['target_leveraged_exposure']:,.2f}")
    
    # Example position sizing calculation
    print("\n" + "=" * 80)
    print("ポジションサイズ計算の例（BTC @ $100,000）")
    print("=" * 80)
    
    btc_price = 100000
    stop_loss_pct = 0.02  # 2%
    
    for balance in [1000, 10000, 100000]:
        print(f"\n💰 口座残高: ${balance:,.2f}")
        
        # Create enhanced risk manager
        risk_manager = EnhancedRiskManager(dynamic_config)
        risk_manager.current_equity = balance
        
        # Test different confidence levels
        for confidence in [0.55, 0.60, 0.70]:
            stop_loss_price = btc_price * (1 - stop_loss_pct)
            
            position_size = risk_manager.calculate_position_size(
                symbol="BTCUSDT",
                entry_price=btc_price,
                stop_loss_price=stop_loss_price,
                confidence=confidence
            )
            
            position_value = position_size * btc_price
            position_pct = (position_value / balance * 100) if balance > 0 else 0
            
            print(f"  信頼度 {confidence*100:.0f}%: {position_size:.6f} BTC "
                  f"(${position_value:,.2f} = {position_pct:.1f}% of equity)")
    
    # Show adaptive behavior
    print("\n" + "=" * 80)
    print("市場状況に応じた調整")
    print("=" * 80)
    
    balance = 100000
    base_limits = dynamic_config.calculate_dynamic_limits(balance)
    
    # High volatility scenario
    high_vol_limits = dynamic_config.adjust_for_market_conditions(
        base_limits,
        market_volatility=0.30,  # 30% volatility (double normal)
        recent_performance=None
    )
    
    print(f"\n🌊 高ボラティリティ時（30% vs 通常15%）:")
    print(f"  通常時最大ポジション: ${base_limits['max_position_size_usd']:,.2f}")
    print(f"  調整後最大ポジション: ${high_vol_limits['max_position_size_usd']:,.2f}")
    print(f"  削減率: {(1 - high_vol_limits['max_position_size_usd'] / base_limits['max_position_size_usd']) * 100:.0f}%")
    
    # Poor performance scenario
    poor_perf_limits = dynamic_config.adjust_for_market_conditions(
        base_limits,
        market_volatility=0.15,
        recent_performance={
            'win_rate': 0.35,  # 35% win rate
            'current_drawdown_pct': 0.12  # 12% drawdown
        }
    )
    
    print(f"\n📉 パフォーマンス不調時（勝率35%、DD12%）:")
    print(f"  通常時最大ポジション: ${base_limits['max_position_size_usd']:,.2f}")
    print(f"  調整後最大ポジション: ${poor_perf_limits['max_position_size_usd']:,.2f}")
    print(f"  削減率: {(1 - poor_perf_limits['max_position_size_usd'] / base_limits['max_position_size_usd']) * 100:.0f}%")
    print(f"  最大ポジション数: {base_limits['max_positions']} → {poor_perf_limits['max_positions']}")
    
    # Excellent performance scenario
    good_perf_limits = dynamic_config.adjust_for_market_conditions(
        base_limits,
        market_volatility=0.15,
        recent_performance={
            'win_rate': 0.65,  # 65% win rate
            'current_drawdown_pct': 0.03  # 3% drawdown
        }
    )
    
    print(f"\n📈 好調時（勝率65%、DD3%）:")
    print(f"  通常時最大ポジション: ${base_limits['max_position_size_usd']:,.2f}")
    print(f"  調整後最大ポジション: ${good_perf_limits['max_position_size_usd']:,.2f}")
    print(f"  増加率: {(good_perf_limits['max_position_size_usd'] / base_limits['max_position_size_usd'] - 1) * 100:.0f}%")
    
    print("\n" + "=" * 80)
    print("✅ 動的ポジション管理システムの特徴:")
    print("- 口座残高に応じて自動的にポジションサイズを調整")
    print("- ケリー基準による最適なベットサイジング")
    print("- 市場ボラティリティに応じたリスク調整")
    print("- パフォーマンスに基づく適応的なサイジング")
    print("- 固定値ではなくパーセンテージベースの制限")
    print("=" * 80)


if __name__ == "__main__":
    test_dynamic_sizing()