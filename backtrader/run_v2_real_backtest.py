#!/usr/bin/env python3
"""
Run V2-REAL Strategy Backtest with Real ThetaData - 6 MONTH COMPREHENSIVE TEST
"""

from high_frequency_0dte_v2_real import TrueHighFrequency0DTEStrategyV2Real
from datetime import datetime, timedelta

def main():
    print("🚀 Starting V2-REAL 6-Month Comprehensive Backtest")
    print("📊 Using ALL 128 trading days of cached 2024 ThetaData")
    print("🎯 Testing: Jan 2 - July 5, 2024 (6 months of real market data)")
    
    # Initialize strategy
    strategy = TrueHighFrequency0DTEStrategyV2Real()
    
    # Use ALL available 2024 cached data (128 trading days)
    start_date = "20240102"  # First available cached date
    end_date = "20240705"    # Last available cached date
    
    print(f"📅 Comprehensive backtest: {start_date} to {end_date}")
    print("💡 This will validate strategy across different market regimes")
    print("⚠️  If performance degrades significantly, live trading issues are systematic")
    
    # Run backtest
    results = strategy.run_backtest(start_date, end_date)
    
    print("✅ 6-Month Comprehensive Backtest Completed!")
    return results

if __name__ == "__main__":
    main()
