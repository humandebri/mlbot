#!/bin/bash
# Start bot with nohup

cd /home/ubuntu/mlbot

# Kill existing tmux sessions
tmux kill-session -t mlbot_reports 2>/dev/null

# Start with nohup
nohup python3 simple_improved_bot_with_reports.py >> logs/nohup_bot.log 2>&1 &

# Save PID
echo $! > bot.pid

echo "Bot started with nohup. PID: $(cat bot.pid)"
echo "Logs: tail -f logs/nohup_bot.log"