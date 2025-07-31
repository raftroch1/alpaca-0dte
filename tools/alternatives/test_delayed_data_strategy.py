#!/usr/bin/env python3
"""
Test Delayed Data Strategy - Works with Free Alpaca Data
=======================================================

This version uses delayed data (15-minute delay) which is free with Alpaca.
Good for testing strategy logic without real-time data subscription.
"""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

def test_delayed_data_approach():
    """Test strategy with delayed data"""
    
    print("🧪 TESTING DELAYED DATA APPROACH")
    print("=" * 50)
    
    load_dotenv()
    
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        
        # Initialize client
        data_client = StockHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )
        
        # Test with data from 30 minutes ago (should be available)
        print("🧪 Testing with 30-minute delayed data...")
        end_time = datetime.now() - timedelta(minutes=30)
        start_time = end_time - timedelta(minutes=60)
        
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time
        )
        
        bars = data_client.get_stock_bars(request)
        df = bars.df
        
        if not df.empty:
            print(f"✅ SUCCESS! Retrieved {len(df)} SPY bars with delayed data")
            print(f"   Latest price: ${df['close'].iloc[-1]:.2f}")
            print(f"   Data timestamp: {df.index[-1]}")
            print(f"   This approach works for strategy testing!")
            return True
        else:
            print("❌ Even delayed data is not available")
            return False
            
    except Exception as e:
        print(f"❌ Delayed data test failed: {e}")
        return False

def suggest_solutions():
    """Suggest practical solutions"""
    
    print("\n💡 PRACTICAL SOLUTIONS")
    print("=" * 50)
    
    print("1. 🎯 ENABLE ALPACA REAL-TIME DATA (Recommended)")
    print("   - Log into Alpaca dashboard")
    print("   - Subscribe to real-time market data (~$1-5/month)")
    print("   - Your strategy will work immediately")
    print("   - No code changes needed")
    
    print("\n2. 🔄 USE DELAYED DATA FOR TESTING")
    print("   - Modify strategy to use 15-30 minute delayed data")
    print("   - Free with Alpaca paper account")
    print("   - Good for testing strategy logic")
    print("   - Not suitable for real-time trading")
    
    print("\n3. 🚀 HYBRID APPROACH (Advanced)")
    print("   - Use ThetaData for options data")
    print("   - Use Alpaca for SPY data and execution")
    print("   - Best data quality for options trading")
    print("   - More complex setup")

if __name__ == "__main__":
    print("Testing solutions for Alpaca data access during live trading...\n")
    
    success = test_delayed_data_approach()
    suggest_solutions()
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATION")
    print("=" * 60)
    
    if success:
        print("✅ Delayed data works - you can test strategy logic")
        print("💰 For live trading: Enable real-time data in Alpaca dashboard")
        print("🎯 This is the simplest solution for pure Alpaca trading")
    else:
        print("❌ Data access issues persist")
        print("🔧 Consider contacting Alpaca support about data permissions")
    
    print("\n🚀 Your strategy is ready - it just needs data access!")
