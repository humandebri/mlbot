#!/bin/bash

# Real API Integration Deployment Script
# This script stops the placeholder system and starts the trading system with real Bybit API integration

set -e

echo "=================================================="
echo "Real Bybit API Integration Deployment"
echo "=================================================="
echo ""

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "📦 Python version: $PYTHON_VERSION"

# Stop current system
echo ""
echo "🛑 Stopping current placeholder system..."

# Try to stop process 521482 if it's still running
if ps -p 521482 > /dev/null 2>&1; then
    echo "Found process 521482, stopping..."
    kill -TERM 521482 2>/dev/null || true
    sleep 2
    # Force kill if still running
    if ps -p 521482 > /dev/null 2>&1; then
        kill -KILL 521482 2>/dev/null || true
    fi
fi

# Also look for any main_working_final.py processes
echo "Looking for any other trading bot processes..."
PIDS=$(ps aux | grep -E "(main_working_final|trading_with_real_api)" | grep -v grep | awk '{print $2}')

if [ ! -z "$PIDS" ]; then
    echo "Found trading bot processes: $PIDS"
    for PID in $PIDS; do
        echo "Stopping process $PID..."
        kill -TERM $PID 2>/dev/null || true
    done
    sleep 2
    # Force kill any remaining
    for PID in $PIDS; do
        if ps -p $PID > /dev/null 2>&1; then
            kill -KILL $PID 2>/dev/null || true
        fi
    done
fi

echo "✅ All previous processes stopped"

# Check .env file
echo ""
echo "🔍 Checking configuration..."
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please copy .env.example to .env and configure your API keys"
    exit 1
fi

# Check if API keys are configured
if grep -q "^BYBIT__API_KEY=your_api_key" .env || grep -q "^BYBIT__API_KEY=$" .env; then
    echo "❌ Bybit API key not configured in .env!"
    echo "Please set BYBIT__API_KEY and BYBIT__API_SECRET in .env file"
    exit 1
fi

# Verify the new trading file exists
if [ ! -f "trading_with_real_api.py" ]; then
    echo "❌ trading_with_real_api.py not found!"
    exit 1
fi

echo "✅ Configuration verified"

# Start the new system with real API integration
echo ""
echo "🚀 Starting Real API Trading System..."
echo "This system will:"
echo "  - Connect to real Bybit API"
echo "  - Retrieve actual account balance every 60 seconds"
echo "  - Use real balance for position sizing"
echo "  - Send Discord notifications with real balance info"
echo ""

# Create log directory if it doesn't exist
mkdir -p logs

# Export environment variable to ensure production mode
export BYBIT__TESTNET=false

# Start the system in background with nohup
LOG_FILE="logs/real_api_trading_$(date +%Y%m%d_%H%M%S).log"
nohup python trading_with_real_api.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "✅ Real API Trading System started with PID: $PID"
echo "📝 Log file: $LOG_FILE"

# Wait a moment for the system to start
sleep 5

# Check if process is still running
if ps -p $PID > /dev/null; then
    echo ""
    echo "✅ System is running successfully!"
    echo ""
    echo "📊 Monitoring commands:"
    echo "  - View logs: tail -f $LOG_FILE"
    echo "  - Check process: ps -p $PID"
    echo "  - Stop system: kill $PID"
    echo ""
    echo "💡 The system will:"
    echo "  1. Retrieve real account balance from Bybit API every 60 seconds"
    echo "  2. Use actual balance for Kelly criterion position sizing"
    echo "  3. Send Discord notifications with real balance information"
    echo "  4. Send hourly balance update notifications"
    echo ""
    echo "📱 Check your Discord for:"
    echo "  - Initial balance notification"
    echo "  - Trade signals with real position sizes"
    echo "  - Hourly balance updates"
    echo ""
    
    # Show initial log output
    echo "📋 Initial log output:"
    echo "------------------------"
    tail -n 20 "$LOG_FILE"
    
else
    echo ""
    echo "❌ System failed to start!"
    echo "Check the log file for errors: $LOG_FILE"
    echo ""
    echo "Last 50 lines of log:"
    tail -n 50 "$LOG_FILE"
    exit 1
fi

echo ""
echo "=================================================="
echo "Real API Integration Deployment Complete!"
echo "=================================================="