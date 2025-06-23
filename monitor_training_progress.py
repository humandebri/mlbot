#!/usr/bin/env python3
"""
訓練進捗を監視するスクリプト
"""

import time
import os
import subprocess
from datetime import datetime


def get_latest_log():
    """最新のログファイルを取得"""
    logs_dir = "logs"
    files = [f for f in os.listdir(logs_dir) if f.startswith("ensemble_training_continued_")]
    if not files:
        return None
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(logs_dir, f)))
    return os.path.join(logs_dir, latest)


def monitor_progress():
    """進捗を監視"""
    print("📊 Training Progress Monitor")
    print("="*60)
    
    while True:
        # プロセス確認
        ps_output = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True
        ).stdout
        
        training_processes = [line for line in ps_output.split('\n') 
                            if 'train_ensemble_model.py' in line and 'grep' not in line]
        
        print(f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if training_processes:
            print("✅ Training is running")
            for proc in training_processes:
                parts = proc.split()
                cpu = parts[2]
                mem = parts[3]
                print(f"   CPU: {cpu}%, Memory: {mem}%")
        else:
            print("❌ Training process not found")
            break
        
        # 最新ログから進捗を取得
        log_file = get_latest_log()
        if log_file:
            # 最後の数行を取得
            tail_output = subprocess.run(
                ["tail", "-n", "20", log_file],
                capture_output=True,
                text=True
            ).stdout
            
            # エポック情報を抽出
            for line in tail_output.split('\n'):
                if 'Epoch' in line and '/' in line:
                    print(f"📈 {line.strip()}")
                elif 'accuracy' in line and 'auc' in line:
                    # 最新の精度情報
                    if '[0m' in line:
                        # プログレスバーの行から情報を抽出
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'accuracy:' in part:
                                acc = parts[i+1]
                            elif 'auc:' in part:
                                auc = parts[i+1]
                            elif 'loss:' in part:
                                loss = parts[i+1]
                        print(f"   Accuracy: {acc}, AUC: {auc}, Loss: {loss}")
                elif 'saved' in line.lower():
                    print(f"💾 {line.strip()}")
        
        # ディスク使用状況
        df_output = subprocess.run(
            ["df", "-h", "."],
            capture_output=True,
            text=True
        ).stdout.split('\n')[1]
        
        disk_parts = df_output.split()
        print(f"\n💾 Disk: {disk_parts[4]} used ({disk_parts[2]} / {disk_parts[1]})")
        
        print("\n" + "-"*60)
        print("Press Ctrl+C to stop monitoring...")
        
        # 30秒待機
        time.sleep(30)


if __name__ == "__main__":
    try:
        monitor_progress()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")