#!/usr/bin/env python3
"""
Signal Monitoring Script - Real-Time Strategy Activity Monitor
============================================================

This script monitors your live strategy and shows:
- Data retrieval status
- Signal generation activity  
- Trading decisions
- Risk management actions
- Performance metrics

Usage: python monitor_signals.py
"""

import os
import time
import subprocess
from datetime import datetime

def monitor_strategy_signals():
    """Monitor strategy for signal detection and trading activity"""
    
    print("🔍 LIVE STRATEGY SIGNAL MONITOR")
    print("=" * 60)
    print(f"Started monitoring at: {datetime.now()}")
    print("Watching for: Data retrieval, Signals, Trades, Risk management")
    print("Press Ctrl+C to stop monitoring\n")
    
    log_file = "strategies/conservative_0dte_live.log"
    
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        print("Make sure your strategy is running!")
        return
    
    # Get current log size to start monitoring from end
    try:
        with open(log_file, 'r') as f:
            f.seek(0, 2)  # Go to end of file
            current_pos = f.tell()
        
        print(f"✅ Monitoring log file: {log_file}")
        print(f"📊 Starting from position: {current_pos}")
        print("-" * 60)
        
        # Monitor for new log entries
        while True:
            try:
                with open(log_file, 'r') as f:
                    f.seek(current_pos)
                    new_lines = f.readlines()
                    current_pos = f.tell()
                
                # Process new log lines
                for line in new_lines:
                    line = line.strip()
                    if line:
                        # Highlight important events
                        if "SPY data" in line and "Retrieved" in line:
                            print(f"📊 DATA: {line}")
                        elif "signal" in line.lower() or "Signal" in line:
                            print(f"🎯 SIGNAL: {line}")
                        elif "trade" in line.lower() or "Trade" in line:
                            print(f"💰 TRADE: {line}")
                        elif "position" in line.lower() or "Position" in line:
                            print(f"📈 POSITION: {line}")
                        elif "ERROR" in line:
                            print(f"❌ ERROR: {line}")
                        elif "WARNING" in line:
                            print(f"⚠️  WARNING: {line}")
                        elif "INFO" in line and any(keyword in line for keyword in ["Starting", "Initialized", "Target", "Conservative"]):
                            print(f"ℹ️  INFO: {line}")
                        elif "RSI" in line or "MACD" in line or "SMA" in line:
                            print(f"📈 INDICATOR: {line}")
                        elif "P&L" in line or "profit" in line.lower() or "loss" in line.lower():
                            print(f"💵 P&L: {line}")
                
                time.sleep(2)  # Check every 2 seconds
                
            except KeyboardInterrupt:
                print("\n👋 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Error reading log: {e}")
                time.sleep(5)
                
    except Exception as e:
        print(f"❌ Error starting monitor: {e}")

def check_strategy_status():
    """Check if strategy is currently running"""
    
    print("\n🔍 STRATEGY STATUS CHECK")
    print("-" * 30)
    
    # Check if process is running
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'live_ultra_aggressive' in result.stdout:
            print("✅ Strategy process is running")
        else:
            print("❌ Strategy process not found")
            print("💡 Start strategy with: python strategies/live_ultra_aggressive_0dte.py")
    except Exception as e:
        print(f"❌ Error checking process: {e}")
    
    # Check recent log activity
    log_file = "strategies/conservative_0dte_live.log"
    if os.path.exists(log_file):
        try:
            # Get last few lines
            result = subprocess.run(['tail', '-5', log_file], capture_output=True, text=True)
            print(f"\n📋 Recent log activity:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line}")
        except Exception as e:
            print(f"❌ Error reading recent logs: {e}")
    else:
        print(f"❌ Log file not found: {log_file}")

def show_expected_signals():
    """Show what signals to expect from the strategy"""
    
    print("\n🎯 WHAT TO EXPECT - SIGNAL TYPES")
    print("-" * 40)
    print("📊 Data Signals:")
    print("   - 'Retrieved X SPY minute bars'")
    print("   - 'Latest price: $XXX.XX'")
    print("   - 'RSI: XX.X, MACD: X.XX'")
    print()
    print("🎯 Trading Signals:")
    print("   - 'BULLISH signal detected'")
    print("   - 'BEARISH signal detected'") 
    print("   - 'Signal strength: X.XX'")
    print()
    print("💰 Trading Actions:")
    print("   - 'Executing CALL trade'")
    print("   - 'Executing PUT trade'")
    print("   - 'Position size: X contracts'")
    print()
    print("🛡️ Risk Management:")
    print("   - 'Daily P&L: $XXX.XX'")
    print("   - 'Risk limit check: PASSED'")
    print("   - 'Closing position: PROFIT/LOSS'")

if __name__ == "__main__":
    try:
        check_strategy_status()
        show_expected_signals()
        print("\n" + "=" * 60)
        monitor_strategy_signals()
    except KeyboardInterrupt:
        print("\n👋 Signal monitoring stopped")
    except Exception as e:
        print(f"❌ Monitor error: {e}")
