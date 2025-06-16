"""
リスク管理設定の更新スクリプト
レバレッジを3倍に、ポジション管理を適切に設定
"""

RISK_CONFIG_UPDATE = """
@dataclass
class RiskConfig:
    \"\"\"Risk management configuration.\"\"\"
    
    # Position limits
    max_position_size_usd: float = 100000.0  # Maximum position size per symbol
    max_total_exposure_usd: float = 500000.0  # Maximum total exposure
    max_leverage: float = 3.0  # Maximum leverage allowed (3倍に変更)
    max_positions: int = 10  # Maximum number of concurrent positions
    
    # Loss limits
    max_daily_loss_usd: float = 10000.0  # Maximum daily loss
    max_drawdown_pct: float = 0.20  # Maximum drawdown (20%)
    stop_loss_pct: float = 0.02  # Default stop loss (2%)
    trailing_stop_pct: float = 0.015  # Trailing stop loss (1.5%)
    
    # Risk per trade
    risk_per_trade_pct: float = 0.01  # Risk 1% of capital per trade
    kelly_fraction: float = 0.25  # Fractional Kelly for position sizing
"""

print("Risk configuration update for 3x leverage:")
print("==========================================")
print(RISK_CONFIG_UPDATE)
print("\nポジション管理機能:")
print("- 最大レバレッジ: 3倍")
print("- シンボルごとの最大ポジション: $100,000")
print("- 総エクスポージャー上限: $500,000")
print("- 最大同時ポジション数: 10")
print("- デフォルトストップロス: 2%")
print("- トレーリングストップ: 1.5%")
print("- 1取引あたりのリスク: 資本の1%")