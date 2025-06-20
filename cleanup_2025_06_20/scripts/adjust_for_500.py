#!/usr/bin/env python3
"""
$500初期資金用の設定調整スクリプト
"""

def adjust_config_for_500():
    """$500用に設定を調整"""
    
    print("=" * 60)
    print("$500初期資金用設定調整")
    print("=" * 60)
    
    # オリジナル設定の保存
    original_config = {
        "symbols": ["BTCUSDT", "ETHUSDT", "ICPUSDT"],
        "position_size": 0.035,  # 3.5%
        "initial_capital": 10000,
        "max_leverage": 3
    }
    
    # $500用の調整
    adjustments = {
        "symbols": ["ICPUSDT"],  # ICPのみ
        "max_position_pct": 0.05,  # 5%
        "max_position_usd": 25,  # $25
        "initial_capital": 500,
        "max_leverage": 2,  # レバレッジ下げる
        "min_confidence": 0.65,  # より慎重
        "max_daily_loss": 25,  # $25
        "max_concurrent_positions": 2
    }
    
    print("\n【変更内容】")
    print(f"通貨ペア: {original_config['symbols']} → {adjustments['symbols']}")
    print(f"ポジションサイズ: {original_config['position_size']*100:.1f}% → {adjustments['max_position_pct']*100:.1f}%")
    print(f"最大レバレッジ: {original_config['max_leverage']}x → {adjustments['max_leverage']}x")
    print(f"初期資金: ${original_config['initial_capital']} → ${adjustments['initial_capital']}")
    
    # リスク計算
    print("\n【リスク分析】")
    position_size_usd = adjustments['initial_capital'] * adjustments['max_position_pct']
    print(f"1ポジションサイズ: ${position_size_usd:.2f}")
    print(f"レバレッジ考慮: ${position_size_usd * adjustments['max_leverage']:.2f}")
    print(f"最大同時ポジション: {adjustments['max_concurrent_positions']}")
    print(f"最大エクスポージャー: ${position_size_usd * adjustments['max_concurrent_positions']:.2f}")
    
    # 期待収益
    print("\n【期待収益（保守的見積もり）】")
    monthly_returns = [0.02, 0.03, 0.05]  # 2%, 3%, 5%
    
    for rate in monthly_returns:
        print(f"\n月利{rate*100:.0f}%の場合:")
        capital = adjustments['initial_capital']
        for month in [1, 3, 6, 12]:
            future_capital = capital * (1 + rate) ** month
            profit = future_capital - capital
            print(f"  {month:2d}ヶ月: ${future_capital:,.0f} (+${profit:,.0f})")
    
    # 推奨事項
    print("\n" + "=" * 60)
    print("【推奨事項】")
    print("=" * 60)
    print("1. 最初の1ヶ月は最小ポジション（$12-15）で練習")
    print("2. 勝率が50%を超えたら徐々にサイズアップ")
    print("3. 毎日の損益を記録し、週次で振り返り")
    print("4. $1,000到達後に他通貨ペア追加を検討")
    print("5. 感情的な取引を避ける（システムに従う）")
    
    # 設定ファイル出力
    print("\n【次のステップ】")
    print("1. .env.production.500 ファイルを使用")
    print("2. APIキーとDiscord URLを設定")
    print("3. 小額でテスト開始")
    
    return adjustments


def validate_minimum_orders():
    """最小注文要件の確認"""
    
    print("\n" + "=" * 60)
    print("最小注文要件チェック")
    print("=" * 60)
    
    # Bybitの最小注文要件（2024年）
    min_requirements = {
        "BTCUSDT": {"min_qty": 0.001, "min_notional": 5},
        "ETHUSDT": {"min_qty": 0.01, "min_notional": 5},
        "ICPUSDT": {"min_qty": 0.1, "min_notional": 5}
    }
    
    # 現在価格（概算）
    current_prices = {
        "BTCUSDT": 50000,
        "ETHUSDT": 3000,
        "ICPUSDT": 12
    }
    
    initial_capital = 500
    position_sizes = [0.024, 0.03, 0.05, 0.10]  # 2.4%, 3%, 5%, 10%
    
    print(f"初期資金: ${initial_capital}")
    print("\n各ポジションサイズでの取引可能性:")
    
    for size_pct in position_sizes:
        position_usd = initial_capital * size_pct
        print(f"\n{size_pct*100:.1f}% (${position_usd:.2f}):")
        
        for symbol, price in current_prices.items():
            min_qty = min_requirements[symbol]["min_qty"]
            min_usd = max(min_qty * price, min_requirements[symbol]["min_notional"])
            
            can_trade = position_usd >= min_usd
            status = "✅" if can_trade else "❌"
            
            print(f"  {symbol}: {status} (最小${min_usd:.2f})")


if __name__ == "__main__":
    adjustments = adjust_config_for_500()
    validate_minimum_orders()
    
    print("\n" + "=" * 60)
    print("設定調整完了")
    print("=" * 60)
    print("✅ .env.production.500 を使用してください")
    print("✅ 最初はICPUSDTのみで開始")
    print("✅ 慎重に資金を増やしていきましょう")