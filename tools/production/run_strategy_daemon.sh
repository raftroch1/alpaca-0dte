#!/bin/bash
"""
Strategy Daemon Runner - Keep Trading Strategy Running
=====================================================

This script runs your trading strategy as a background daemon.
It includes automatic restart, logging, and monitoring capabilities.
"""

# Configuration
STRATEGY_DIR="/Users/devops/Desktop/coding projects/windsurf/Alpaca_0dte/alpaca-0dte/strategies"
STRATEGY_FILE="live_ultra_aggressive_0dte.py"
LOG_DIR="/Users/devops/Desktop/coding projects/windsurf/Alpaca_0dte/alpaca-0dte/logs"
PID_FILE="$LOG_DIR/strategy.pid"
LOG_FILE="$LOG_DIR/strategy_daemon.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to start the strategy
start_strategy() {
    echo "$(date): Starting trading strategy..." >> "$LOG_FILE"
    
    cd "$STRATEGY_DIR"
    
    # Activate conda environment and run strategy
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate Alpaca_Options
    
    # Run strategy in background
    nohup python "$STRATEGY_FILE" >> "$LOG_FILE" 2>&1 &
    
    # Save PID
    echo $! > "$PID_FILE"
    
    echo "$(date): Strategy started with PID $(cat $PID_FILE)" >> "$LOG_FILE"
    echo "Strategy started! PID: $(cat $PID_FILE)"
    echo "Monitor logs: tail -f $LOG_FILE"
}

# Function to stop the strategy
stop_strategy() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "$(date): Stopping strategy with PID $PID..." >> "$LOG_FILE"
        kill "$PID" 2>/dev/null
        rm -f "$PID_FILE"
        echo "Strategy stopped."
    else
        echo "No PID file found. Strategy may not be running."
    fi
}

# Function to check strategy status
status_strategy() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Strategy is running with PID $PID"
            echo "Log file: $LOG_FILE"
            echo "Strategy log: $STRATEGY_DIR/conservative_0dte_live.log"
        else
            echo "PID file exists but process is not running. Cleaning up..."
            rm -f "$PID_FILE"
        fi
    else
        echo "Strategy is not running."
    fi
}

# Function to restart the strategy
restart_strategy() {
    echo "Restarting strategy..."
    stop_strategy
    sleep 2
    start_strategy
}

# Function to show logs
show_logs() {
    echo "=== Strategy Daemon Log ==="
    tail -50 "$LOG_FILE"
    echo ""
    echo "=== Strategy Application Log ==="
    tail -50 "$STRATEGY_DIR/conservative_0dte_live.log"
}

# Function to monitor logs in real-time
monitor_logs() {
    echo "Monitoring strategy logs (Ctrl+C to exit)..."
    tail -f "$LOG_FILE" "$STRATEGY_DIR/conservative_0dte_live.log"
}

# Main script logic
case "$1" in
    start)
        start_strategy
        ;;
    stop)
        stop_strategy
        ;;
    restart)
        restart_strategy
        ;;
    status)
        status_strategy
        ;;
    logs)
        show_logs
        ;;
    monitor)
        monitor_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|monitor}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the trading strategy in background"
        echo "  stop    - Stop the trading strategy"
        echo "  restart - Restart the trading strategy"
        echo "  status  - Check if strategy is running"
        echo "  logs    - Show recent logs"
        echo "  monitor - Monitor logs in real-time"
        echo ""
        echo "Examples:"
        echo "  ./run_strategy_daemon.sh start"
        echo "  ./run_strategy_daemon.sh monitor"
        echo "  ./run_strategy_daemon.sh stop"
        exit 1
        ;;
esac
