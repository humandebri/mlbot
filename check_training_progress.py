#!/usr/bin/env python3
"""
訓練進捗をリアルタイムで確認するスクリプト
"""

import os
import re
import time
from datetime import datetime
import subprocess

def get_latest_log_file():
    """最新のログファイルを取得"""
    log_dir = "logs"
    files = [f for f in os.listdir(log_dir) if f.startswith("fixed_training_")]
    if not files:
        return None
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))
    return os.path.join(log_dir, latest)

def parse_progress(line):
    """進捗行をパース"""
    # Epoch情報を抽出
    epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
    if epoch_match:
        return {
            'type': 'epoch',
            'current': int(epoch_match.group(1)),
            'total': int(epoch_match.group(2))
        }
    
    # ステップ情報を抽出
    step_match = re.search(r'\[1m\s*(\d+)/(\d+)\[0m', line)
    if step_match:
        # メトリクスも抽出
        accuracy = re.search(r'accuracy: ([\d.]+)', line)
        auc = re.search(r'auc: ([\d.]+)', line)
        loss = re.search(r'loss: ([\d.]+)', line)
        
        return {
            'type': 'step',
            'current': int(step_match.group(1)),
            'total': int(step_match.group(2)),
            'accuracy': float(accuracy.group(1)) if accuracy else None,
            'auc': float(auc.group(1)) if auc else None,
            'loss': float(loss.group(1)) if loss else None
        }
    
    # Buy ratio警告を抽出
    if "Buy ratio" in line and "WARNING" in line:
        ratio_match = re.search(r'Buy ratio: ([\d.]+)%', line)
        if ratio_match:
            return {
                'type': 'warning',
                'buy_ratio': float(ratio_match.group(1))
            }
    
    return None

def monitor_training():
    """訓練を監視"""
    log_file = get_latest_log_file()
    if not log_file:
        print("訓練ログが見つかりません")
        return
    
    print(f"📊 訓練進捗モニター")
    print(f"ログファイル: {log_file}")
    print("="*80)
    
    current_epoch = 0
    last_update = time.time()
    
    # tail -fでログを監視
    cmd = ['tail', '-f', log_file]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    
    try:
        for line in process.stdout:
            progress = parse_progress(line)
            if not progress:
                continue
            
            now = datetime.now().strftime("%H:%M:%S")
            
            if progress['type'] == 'epoch':
                current_epoch = progress['current']
                print(f"\n🔄 [{now}] エポック {current_epoch}/{progress['total']} 開始")
            
            elif progress['type'] == 'step' and time.time() - last_update > 5:  # 5秒ごとに更新
                percent = (progress['current'] / progress['total']) * 100
                print(f"📈 [{now}] エポック {current_epoch}: {percent:.1f}% " +
                      f"(acc={progress['accuracy']:.4f}, auc={progress['auc']:.4f}, loss={progress['loss']:.4f})")
                last_update = time.time()
            
            elif progress['type'] == 'warning':
                print(f"⚠️  [{now}] Buy比率警告: {progress['buy_ratio']:.2f}%")
            
            # エポック完了時のBuy ratio確認
            if "Epoch" in line and "Buy ratio" in line:
                print(f"✅ [{now}] {line.strip()}")
    
    except KeyboardInterrupt:
        print("\n\n監視を終了しました")
        process.terminate()

if __name__ == "__main__":
    monitor_training()