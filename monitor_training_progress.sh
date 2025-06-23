#!/bin/bash

echo "📊 モデル訓練進捗モニター"
echo "=========================="
echo ""

EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

while true; do
    clear
    echo "📊 モデル訓練進捗モニター - $(date)"
    echo "=========================="
    echo ""
    
    # プロセス状況
    echo "🔄 実行中のプロセス:"
    ssh -i $KEY_PATH $EC2_HOST 'ps aux | grep -E "(prepare_balanced|train_ensemble)" | grep -v grep' | awk '{print $11, "- CPU:", $3"%", "MEM:", $4"%"}'
    
    echo ""
    echo "📄 最新ログ:"
    ssh -i $KEY_PATH $EC2_HOST 'cd /home/ubuntu/mlbot && tail -20 logs/model_training_*.log | grep -E "(INFO|Processing|samples|Buy ratio|ERROR|epoch|loss|AUC)" | tail -15'
    
    echo ""
    echo "💾 ディスク使用状況:"
    ssh -i $KEY_PATH $EC2_HOST 'df -h /home/ubuntu | tail -1'
    
    echo ""
    echo "🔄 30秒後に更新... (Ctrl+Cで終了)"
    sleep 30
done