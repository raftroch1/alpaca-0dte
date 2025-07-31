#!/usr/bin/env python3
"""
🚀 PHASE 4D: PAPER TRADING LAUNCHER
==================================

Quick launcher for Phase 4D optimized bull put spreads paper trading.
Run this script to start live paper trading with the proven profitable configuration.
"""

import sys
import os
from datetime import datetime

def check_environment():
    """Check if environment is properly configured"""
    print("🔧 Checking environment...")
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("❌ Missing .env file!")
        print("📝 Please create .env file with your Alpaca paper trading credentials:")
        print("   ALPACA_API_KEY=your_paper_api_key")
        print("   ALPACA_SECRET_KEY=your_paper_secret_key")
        print("🌐 Get credentials: https://app.alpaca.markets/paper/dashboard/overview")
        return False
    
    # Check packages
    try:
        import pandas
        import numpy  
        from alpaca.trading.client import TradingClient
        from dotenv import load_dotenv
        print("✅ All packages available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("📦 Install with: pip install pandas numpy alpaca-py python-dotenv")
        return False
    
    return True

def start_paper_trading():
    """Start paper trading session"""
    if not check_environment():
        return False
        
    print("\n🎯 PHASE 4D: OPTIMIZED BULL PUT SPREADS")
    print("🚀 Starting Paper Trading Session")
    print("=" * 50)
    print("📊 Strategy: 12pt spreads, 3 contracts, 75% targets")
    print("🎯 Daily Target: $365+ profit")
    print("📈 Expected Win Rate: 70%")
    print("⚠️ PAPER TRADING - No real money at risk")
    print("=" * 50)
    
    # Import and run the optimized strategy
    try:
        from phase4d_optimized_final_strategy import Phase4DOptimizedFinalStrategy
        
        strategy = Phase4DOptimizedFinalStrategy()
        
        # Run today's session
        today = datetime.now().strftime('%Y-%m-%d')
        print(f"\n🗓️ Running paper trading session for {today}")
        
        result = strategy.run_optimal_daily_session(today)
        
        print(f"\n📊 SESSION RESULTS:")
        print(f"   💰 P&L: ${result.get('total_pnl', 0):.2f}")
        print(f"   🔢 Trades: {result.get('trades', 0)}")
        print(f"   🎯 Target Achieved: {'✅' if result.get('target_achieved', False) else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running strategy: {e}")
        return False

if __name__ == "__main__":
    print("🎯 PHASE 4D PAPER TRADING LAUNCHER")
    print("=" * 40)
    
    success = start_paper_trading()
    
    if success:
        print("\n✅ Paper trading session completed!")
        print("📊 Check the results above")
        print("🔄 Run again for another session")
    else:
        print("\n❌ Setup incomplete")
        print("💡 Fix the issues above and try again")
    
    print("=" * 40)
