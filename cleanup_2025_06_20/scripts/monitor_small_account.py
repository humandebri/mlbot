#!/usr/bin/env python3
"""
小額アカウント（$500）の監視スクリプト
リスク管理と成長追跡に特化
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List

class SmallAccountMonitor:
    """小額アカウント専用モニター"""
    
    def __init__(self, initial_capital: float = 500):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.daily_pnl = []
        
    def calculate_metrics(self) -> Dict:
        """主要メトリクスの計算"""
        
        # 基本統計
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trades if t.get('pnl', 0) < 0])
        
        # 勝率
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 損益
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        total_return = total_pnl / self.initial_capital * 100
        
        # リスク指標
        if self.trades:
            max_loss = min(t.get('pnl', 0) for t in self.trades)
            max_drawdown_pct = abs(max_loss) / self.initial_capital * 100
        else:
            max_loss = 0
            max_drawdown_pct = 0
        
        return {
            'current_capital': self.current_capital,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'max_drawdown_pct': max_drawdown_pct,
            'total_pnl': total_pnl
        }
    
    def check_risk_limits(self) -> List[str]:
        """リスク限度のチェック"""
        warnings = []
        
        # 日次損失チェック（5%以上）
        today_pnl = sum(t.get('pnl', 0) for t in self.trades 
                       if t.get('date', '') == datetime.now().strftime('%Y-%m-%d'))
        if today_pnl < -self.initial_capital * 0.05:
            warnings.append(f"⚠️ 日次損失限度到達: ${today_pnl:.2f} (-{abs(today_pnl)/self.initial_capital*100:.1f}%)")
        
        # 総損失チェック（10%以上）
        total_loss = self.initial_capital - self.current_capital
        if total_loss > self.initial_capital * 0.10:
            warnings.append(f"🚨 総損失警告: ${total_loss:.2f} (-{total_loss/self.initial_capital*100:.1f}%)")
        
        # 連続損失チェック
        if len(self.trades) >= 3:
            last_3_trades = self.trades[-3:]
            if all(t.get('pnl', 0) < 0 for t in last_3_trades):
                warnings.append("⚠️ 3連続損失 - 戦略の見直しを検討")
        
        return warnings
    
    def growth_projection(self) -> Dict:
        """成長予測"""
        
        # 過去の実績から月利を計算
        if not self.trades:
            monthly_return = 0.02  # デフォルト2%
        else:
            days_traded = len(set(t.get('date', '') for t in self.trades))
            if days_traded > 0:
                daily_return = (self.current_capital - self.initial_capital) / self.initial_capital / days_traded
                monthly_return = daily_return * 30
            else:
                monthly_return = 0.02
        
        # 将来予測
        projections = {}
        capital = self.current_capital
        
        milestones = [600, 750, 1000, 1500, 2000]
        for milestone in milestones:
            if capital < milestone and monthly_return > 0:
                months_needed = 0
                temp_capital = capital
                while temp_capital < milestone and months_needed < 24:  # 最大2年
                    temp_capital *= (1 + monthly_return)
                    months_needed += 1
                
                projections[f"${milestone}"] = {
                    'months': months_needed,
                    'date': (datetime.now() + timedelta(days=months_needed*30)).strftime('%Y-%m')
                }
        
        return {
            'monthly_return_pct': monthly_return * 100,
            'milestones': projections
        }
    
    def print_report(self):
        """レポート出力"""
        
        print("=" * 60)
        print(f"小額アカウント監視レポート - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60)
        
        # 基本メトリクス
        metrics = self.calculate_metrics()
        print(f"\n📊 現在の状況:")
        print(f"  初期資金: ${self.initial_capital:.2f}")
        print(f"  現在資金: ${metrics['current_capital']:.2f}")
        print(f"  総収益: ${metrics['total_pnl']:.2f} ({metrics['total_return_pct']:+.2f}%)")
        print(f"  取引回数: {metrics['total_trades']}")
        print(f"  勝率: {metrics['win_rate']:.1f}%")
        print(f"  最大DD: {metrics['max_drawdown_pct']:.1f}%")
        
        # リスク警告
        warnings = self.check_risk_limits()
        if warnings:
            print(f"\n⚠️  リスク警告:")
            for warning in warnings:
                print(f"  {warning}")
        else:
            print(f"\n✅ リスク状態: 正常")
        
        # 成長予測
        projection = self.growth_projection()
        print(f"\n📈 成長予測 (月利 {projection['monthly_return_pct']:.1f}%):")
        for milestone, info in projection['milestones'].items():
            print(f"  {milestone} 到達: 約{info['months']}ヶ月後 ({info['date']})")
        
        # 推奨アクション
        print(f"\n💡 推奨アクション:")
        if metrics['current_capital'] < 450:  # 10%以上の損失
            print("  1. ポジションサイズを最小に縮小（$12-15）")
            print("  2. 勝率が改善するまで様子見")
            print("  3. 戦略の見直しを検討")
        elif metrics['current_capital'] < 600:
            print("  1. 現在の戦略を継続")
            print("  2. ICPUSDTに集中")
            print("  3. 慎重にポジションサイズ管理")
        elif metrics['current_capital'] < 1000:
            print("  1. ポジションサイズを段階的に増加")
            print("  2. ETHUSDTの追加を検討")
            print("  3. リスク管理を維持")
        else:
            print("  1. 全通貨ペアでの取引を開始")
            print("  2. 標準設定（3.5%ポジション）へ移行")
            print("  3. 複利効果を最大化")
        
        print("\n" + "=" * 60)


# デモ実行
if __name__ == "__main__":
    monitor = SmallAccountMonitor(initial_capital=500)
    
    # サンプルデータ
    monitor.current_capital = 525  # 5%の利益
    monitor.trades = [
        {'date': '2024-01-01', 'pnl': 5},
        {'date': '2024-01-02', 'pnl': -3},
        {'date': '2024-01-03', 'pnl': 8},
        {'date': '2024-01-04', 'pnl': 15},
    ]
    
    monitor.print_report()