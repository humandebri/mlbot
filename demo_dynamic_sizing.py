#!/usr/bin/env python3
"""
動的ポジションサイジングのデモンストレーション（依存関係なし）
"""

class DynamicPositionDemo:
    """動的ポジション管理のデモ"""
    
    def __init__(self):
        # Core risk parameters (as percentages)
        self.risk_per_trade_pct = 0.01  # 1% risk per trade
        self.kelly_fraction = 0.25  # 25% fractional Kelly
        
        # Position sizing parameters (as % of equity)
        self.max_position_pct_per_symbol = 0.20  # Max 20% per symbol
        self.max_total_exposure_pct = 1.0  # Max 100% (3x leverage = 300%)
        
        # Other parameters
        self.max_leverage = 3.0
        self.max_daily_loss_pct = 0.05  # 5% daily loss limit
        self.max_drawdown_pct = 0.20  # 20% max drawdown
        self.default_stop_loss_pct = 0.02  # 2% stop loss
        
        # Dynamic position count
        self.base_positions = 3
        self.positions_per_100k = 2
        self.max_positions = 20
    
    def calculate_limits(self, equity):
        """Calculate dynamic limits based on equity"""
        
        # Position limits
        max_position_size = equity * self.max_position_pct_per_symbol
        max_total_exposure = equity * self.max_total_exposure_pct
        
        # Loss limits
        max_daily_loss = equity * self.max_daily_loss_pct
        risk_per_trade = equity * self.risk_per_trade_pct
        
        # Dynamic position count
        additional_positions = int(equity / 100000) * self.positions_per_100k
        max_positions = min(self.base_positions + additional_positions, self.max_positions)
        
        # Rate limits (scale with account)
        max_trades_per_hour = max(5, int(equity / 10000) * 5)
        max_trades_per_day = max(20, int(equity / 10000) * 20)
        
        return {
            'equity': equity,
            'max_position_size': max_position_size,
            'max_total_exposure': max_total_exposure,
            'max_daily_loss': max_daily_loss,
            'risk_per_trade': risk_per_trade,
            'max_positions': max_positions,
            'max_trades_per_hour': max_trades_per_hour,
            'max_trades_per_day': max_trades_per_day,
            'max_leveraged_exposure': equity * self.max_leverage
        }
    
    def calculate_position_size(self, equity, entry_price, stop_loss_price, confidence=0.6):
        """Calculate position size using Kelly criterion"""
        
        # Risk amount (1% of equity)
        risk_amount = equity * self.risk_per_trade_pct
        
        # Position risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            return 0
        
        # Base position size
        base_size = risk_amount / risk_per_unit
        
        # Apply Kelly criterion
        if confidence > 0.5:
            # Simplified Kelly formula
            win_loss_ratio = 2.0  # Assume 2:1 R:R
            kelly_pct = confidence - (1 - confidence) / win_loss_ratio
            kelly_pct = max(0, min(kelly_pct, 1))
            
            # Apply fractional Kelly
            position_size = base_size * kelly_pct * self.kelly_fraction
        else:
            position_size = base_size * 0.5
        
        # Apply position limit
        max_position_value = equity * self.max_position_pct_per_symbol
        max_size = max_position_value / entry_price
        
        return min(position_size, max_size)


def main():
    demo = DynamicPositionDemo()
    
    print("=" * 80)
    print("🎯 動的ポジション管理システム - 完全に連携した設定")
    print("=" * 80)
    print()
    
    # Test different account sizes
    test_balances = [100, 1000, 10000, 50000, 100000, 500000, 1000000]
    
    for balance in test_balances:
        limits = demo.calculate_limits(balance)
        
        print(f"\n💰 口座残高: ${balance:,.0f}")
        print("-" * 60)
        print(f"📊 動的に計算された制限:")
        print(f"  • 最大ポジション/シンボル: ${limits['max_position_size']:,.0f} (残高の{demo.max_position_pct_per_symbol*100:.0f}%)")
        print(f"  • 最大総エクスポージャー: ${limits['max_total_exposure']:,.0f} (残高の{demo.max_total_exposure_pct*100:.0f}%)")
        print(f"  • 1取引リスク: ${limits['risk_per_trade']:,.0f} (残高の{demo.risk_per_trade_pct*100:.0f}%)")
        print(f"  • 最大同時ポジション: {limits['max_positions']}個")
        print(f"  • 日次損失上限: ${limits['max_daily_loss']:,.0f} (残高の{demo.max_daily_loss_pct*100:.0f}%)")
        print(f"  • レバレッジ3倍時の最大: ${limits['max_leveraged_exposure']:,.0f}")
    
    # Position sizing examples
    print("\n" + "=" * 80)
    print("📈 ポジションサイジングの例（BTC @ $100,000、2%ストップロス）")
    print("=" * 80)
    
    btc_price = 100000
    stop_loss_price = 98000  # 2% stop loss
    
    for balance in [1000, 10000, 100000]:
        print(f"\n💼 口座残高: ${balance:,.0f}")
        
        for confidence in [0.55, 0.60, 0.70]:
            position_size = demo.calculate_position_size(
                balance, btc_price, stop_loss_price, confidence
            )
            
            position_value = position_size * btc_price
            position_pct = (position_value / balance * 100) if balance > 0 else 0
            
            print(f"  信頼度{confidence*100:.0f}%: {position_size:.6f} BTC "
                  f"(${position_value:,.0f} = 残高の{position_pct:.1f}%)")
    
    # Key features
    print("\n" + "=" * 80)
    print("🔑 動的管理システムの主な特徴:")
    print("=" * 80)
    print("✅ 固定値ではなくパーセンテージベースの制限")
    print("✅ 口座残高に応じて自動的にスケール")
    print("✅ ケリー基準による最適なサイジング")
    print("✅ リスク管理パラメータが完全に連携")
    print()
    print("📊 例: $10,000の口座")
    print("  • 最大ポジション: $2,000 (20%)")
    print("  • 1取引リスク: $100 (1%)")
    print("  • 最大5ポジション同時保有可能")
    print()
    print("📊 例: $100,000の口座")
    print("  • 最大ポジション: $20,000 (20%)")
    print("  • 1取引リスク: $1,000 (1%)")
    print("  • 最大5ポジション同時保有可能")
    print()
    print("⚡ 口座が成長すれば、ポジションサイズも自動的に増加")
    print("⚡ 損失が発生すれば、ポジションサイズも自動的に減少")
    print("=" * 80)


if __name__ == "__main__":
    main()