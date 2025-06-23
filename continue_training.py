#!/usr/bin/env python3
"""
訓練を最初から再開するスクリプト（中断された訓練の続き）
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """メイン処理"""
    logger.info("Continuing model training from the beginning...")
    logger.info("This will take several hours. The process will continue even if timeout occurs.")
    
    # ログファイル名
    log_file = f"logs/ensemble_training_continued_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Pythonコマンドを構築
    cmd = [
        sys.executable,  # 現在のPython実行ファイル
        "train_ensemble_model.py"
    ]
    
    # 出力をログファイルに記録しながら実行
    logger.info(f"Executing: {' '.join(cmd)}")
    logger.info(f"Log file: {log_file}")
    
    with open(log_file, 'w') as f:
        # ヘッダー情報を書き込み
        f.write(f"=== Ensemble Model Training Log ===\n")
        f.write(f"Started at: {datetime.now()}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("="*60 + "\n\n")
        f.flush()
        
        # プロセスを実行
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # リアルタイムで出力を処理
        for line in iter(process.stdout.readline, ''):
            if line:
                # ファイルに書き込み
                f.write(line)
                f.flush()
                
                # 重要な行は標準出力にも表示
                if any(keyword in line for keyword in ['Epoch', 'accuracy', 'AUC', 'saved', 'ERROR']):
                    print(line.strip())
        
        # プロセスの終了を待つ
        process.wait()
        
        # 終了情報を記録
        f.write(f"\n{'='*60}\n")
        f.write(f"Finished at: {datetime.now()}\n")
        f.write(f"Exit code: {process.returncode}\n")
    
    logger.info(f"Training completed. Exit code: {process.returncode}")
    logger.info(f"Full log saved to: {log_file}")
    
    return process.returncode

if __name__ == "__main__":
    exit(main())