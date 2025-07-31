#!/usr/bin/env python3
"""
Live Alpaca Data Test - Market Hours Data Verification
=====================================================

Test Alpaca data connection during market hours to identify the 403 issue.
This should work without ThetaData since Alpaca has its own data feed.
"""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_live_alpaca_data():
    """Test live Alpaca data during market hours"""
    
    print("🧪 LIVE ALPACA DATA TEST - MARKET HOURS")
    print("=" * 50)
    print(f"Current time: {datetime.now()}")
    print("Testing Alpaca data connection during market hours...")
    
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        
        # Initialize Alpaca data client
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            print("❌ API keys not found")
            return False
            
        print(f"✅ API Key: {api_key[:8]}...{api_key[-4:]}")
        
        # Create data client
        data_client = StockHistoricalDataClient(api_key, secret_key)
        print("✅ Data client initialized")
        
        # Test 1: Recent SPY data (last 10 minutes)
        print("\n🧪 Test 1: Recent SPY minute bars")
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=10)
        
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time
        )
        
        bars = data_client.get_stock_bars(request)
        df = bars.df
        
        if not df.empty:
            print(f"✅ Retrieved {len(df)} SPY bars")
            print(f"   Latest price: ${df['close'].iloc[-1]:.2f}")
            print(f"   Time range: {df.index[0]} to {df.index[-1]}")
        else:
            print("⚠️  No recent data available")
        
        # Test 2: Different time range (last hour)
        print("\n🧪 Test 2: Last hour SPY data")
        start_time = end_time - timedelta(hours=1)
        
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time
        )
        
        bars = data_client.get_stock_bars(request)
        df = bars.df
        
        if not df.empty:
            print(f"✅ Retrieved {len(df)} SPY bars (1 hour)")
            print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        else:
            print("⚠️  No hourly data available")
        
        # Test 3: Try different symbol
        print("\n🧪 Test 3: Testing with AAPL")
        request = StockBarsRequest(
            symbol_or_symbols="AAPL",
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=end_time - timedelta(minutes=5),
            end=end_time
        )
        
        bars = data_client.get_stock_bars(request)
        df = bars.df
        
        if not df.empty:
            print(f"✅ AAPL data works: {len(df)} bars")
            print(f"   Latest AAPL: ${df['close'].iloc[-1]:.2f}")
        else:
            print("⚠️  AAPL data also unavailable")
            
        return True
        
    except Exception as e:
        print(f"❌ Alpaca data test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Check if it's a 403 error specifically
        if "403" in str(e) or "forbidden" in str(e).lower():
            print("\n🔍 403 FORBIDDEN ERROR ANALYSIS:")
            print("   - This suggests an authentication or permissions issue")
            print("   - Your API keys might be for a different environment")
            print("   - Or there might be rate limiting during market hours")
            print("   - Let's check if this is a paper vs live account issue")
        
        return False

def test_alternative_approaches():
    """Test alternative data approaches"""
    
    print("\n🔧 ALTERNATIVE APPROACHES")
    print("=" * 50)
    
    # Test 1: Try with different base URL
    print("🧪 Test: Different API endpoint")
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        
        # Try with explicit paper URL
        data_client = StockHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            url_override="https://paper-api.alpaca.markets"
        )
        print("✅ Alternative endpoint client created")
        
    except Exception as e:
        print(f"❌ Alternative endpoint failed: {e}")
    
    # Test 2: Check account type
    print("\n🧪 Test: Account verification")
    try:
        from alpaca.trading.client import TradingClient
        
        trading_client = TradingClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )
        
        account = trading_client.get_account()
        print(f"✅ Account type: {account.status}")
        print(f"   Paper trading: {account.pattern_day_trader}")
        
    except Exception as e:
        print(f"❌ Account check failed: {e}")

if __name__ == "__main__":
    print("Testing Alpaca data during market hours...")
    print("This should work WITHOUT ThetaData for live paper trading!\n")
    
    success = test_live_alpaca_data()
    test_alternative_approaches()
    
    print("\n" + "=" * 60)
    print("🎯 CONCLUSION")
    print("=" * 60)
    
    if success:
        print("✅ Alpaca data is working - strategy should run fine!")
    else:
        print("❌ Alpaca data issue identified")
        print("   This explains why your strategy is getting 403 errors")
        print("   The issue is with Alpaca data access, not missing ThetaData")
        print("\n💡 NEXT STEPS:")
        print("   1. Check if API keys are for correct environment (paper vs live)")
        print("   2. Verify account permissions for real-time data")
        print("   3. Consider using ThetaData as backup data source")
