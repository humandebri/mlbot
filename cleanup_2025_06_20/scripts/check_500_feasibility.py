#!/usr/bin/env python3
"""
$500初期資金での実現可能性チェック
"""

# 設定
INITIAL_CAPITAL = 500  # $500
POSITION_SIZE_PCT = 0.035  # 3.5%
LEVERAGE = 3

# Bybit最小注文サイズ（2024年時点）
MIN_ORDER_SIZES = {
    "BTCUSDT": {"min_qty": 0.001, "price": 50000},  # 0.001 BTC = $50
    "ETHUSDT": {"min_qty": 0.01, "price": 3000},    # 0.01 ETH = $30
    "ICPUSDT": {"min_qty": 1.0, "price": 12}        # 1 ICP = $12
}

print("=" * 60)
print("$500初期資金での実現可能性分析")
print("=" * 60)

# 基本計算
base_position_size = INITIAL_CAPITAL * POSITION_SIZE_PCT
print(f"\n基本ポジションサイズ: ${base_position_size:.2f} (3.5%)")
print(f"レバレッジ考慮: ${base_position_size * LEVERAGE:.2f} (3倍)")

# 各通貨ペアの分析
print("\n各通貨ペアの最小注文要件:")
print("-" * 60)

feasible_pairs = []
for symbol, info in MIN_ORDER_SIZES.items():
    min_order_value = info["min_qty"] * info["price"]
    can_trade = base_position_size >= min_order_value
    
    print(f"{symbol}:")
    print(f"  最小数量: {info['min_qty']} units")
    print(f"  最小注文額: ${min_order_value:.2f}")
    print(f"  取引可能: {'✅ YES' if can_trade else '❌ NO'}")
    
    if can_trade:
        feasible_pairs.append(symbol)
    print()

# 推奨設定
print("\n【問題点】")
print(f"- BTCUSDTの最小注文額（$50）> ポジションサイズ（${base_position_size:.2f}）")
print("- 現在の設定では取引不可能")

print("\n【解決策】")

# 解決策1: ポジションサイズを増やす
min_required_pct = 50 / INITIAL_CAPITAL
print(f"\n1. ポジションサイズを{min_required_pct*100:.1f}%に増やす")
print(f"   - リスクが高くなる（1回の損失で-{min_required_pct*100:.1f}%）")
print(f"   - 最大ドローダウンが大きくなる")

# 解決策2: より安い通貨に集中
print(f"\n2. ICPUSDTのみで取引")
print(f"   - 最小注文額: $12（取引可能）")
print(f"   - ポジションサイズ: ${base_position_size:.2f}")
print(f"   - 分散投資ができない")

# 解決策3: 資金を増やす
min_capital_needed = 50 / 0.035
print(f"\n3. 初期資金を${min_capital_needed:.0f}以上に増やす")
print(f"   - 全通貨ペアで取引可能")
print(f"   - より安定した運用")

# 修正設定の提案
print("\n" + "=" * 60)
print("【推奨設定】$500での運用")
print("=" * 60)

# 安全な設定
safe_position_pct = 0.10  # 10%
safe_position_size = INITIAL_CAPITAL * safe_position_pct

print(f"\n設定1: 高リスク・高リターン")
print(f"- ポジションサイズ: 10% (${safe_position_size:.2f})")
print(f"- 対象通貨: BTCUSDT, ETHUSDT, ICPUSDT")
print(f"- 月次目標: 10-15%")
print(f"- リスク: 高（1取引で-10%の可能性）")

print(f"\n設定2: 低リスク・着実運用")
print(f"- ポジションサイズ: 5% ($25)")
print(f"- 対象通貨: ICPUSDT のみ")
print(f"- 月次目標: 3-5%")
print(f"- リスク: 中")

print(f"\n設定3: 最小リスクテスト")
print(f"- ポジションサイズ: 2.4% ($12)")
print(f"- 対象通貨: ICPUSDT のみ")
print(f"- 月次目標: 1-2%")
print(f"- リスク: 低（検証用）")

# シミュレーション
print("\n" + "=" * 60)
print("収益シミュレーション（$500スタート）")
print("=" * 60)

# 月4.16%の場合
monthly_return = 0.0416
capital = INITIAL_CAPITAL
print(f"\n標準設定（月4.16%）での推移:")
print(f"開始: ${capital:.2f}")

for month in [1, 3, 6, 12]:
    capital_future = INITIAL_CAPITAL * (1 + monthly_return) ** month
    profit = capital_future - INITIAL_CAPITAL
    print(f"{month:2d}ヶ月後: ${capital_future:.2f} (+${profit:.2f})")

# 保守的な月2%の場合
print(f"\n保守的設定（月2%）での推移:")
capital = INITIAL_CAPITAL
monthly_return_conservative = 0.02

for month in [1, 3, 6, 12]:
    capital_future = INITIAL_CAPITAL * (1 + monthly_return_conservative) ** month
    profit = capital_future - INITIAL_CAPITAL
    print(f"{month:2d}ヶ月後: ${capital_future:.2f} (+${profit:.2f})")

print("\n" + "=" * 60)
print("結論")
print("=" * 60)
print("✅ $500でも運用可能だが、設定調整が必要")
print("✅ ICPUSDTなら最小注文要件をクリア")
print("✅ 最初は低リスク設定でテスト推奨")
print("⚠️  BTCUSDTは$1,500以上推奨")