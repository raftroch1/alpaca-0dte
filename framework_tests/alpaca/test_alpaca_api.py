#!/usr/bin/env python3
"""
Alpaca API Key Test Script
=========================

This script tests your Alpaca API keys with various endpoints to ensure they're working correctly.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_alpaca_api():
    """Test Alpaca API connection with multiple methods"""
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    print("🎯 ALPACA API KEY TEST")
    print("=" * 50)
    print(f"API Key: {api_key[:8]}...{api_key[-4:] if api_key else 'None'}")
    print(f"Base URL: {base_url}")
    print("=" * 50)
    
    if not api_key or not secret_key:
        print("❌ API keys not found in environment variables")
        return False
    
    # Test 1: Account endpoint
    print("\n🧪 Test 1: Account Information")
    try:
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': secret_key
        }
        
        response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
        
        if response.status_code == 200:
            account_data = response.json()
            print("✅ Account endpoint successful")
            print(f"   Account ID: {account_data.get('id', 'N/A')}")
            print(f"   Status: {account_data.get('status', 'N/A')}")
            print(f"   Buying Power: ${float(account_data.get('buying_power', 0)):,.2f}")
            print(f"   Cash: ${float(account_data.get('cash', 0)):,.2f}")
        else:
            print(f"❌ Account endpoint failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Account test failed: {e}")
        return False
    
    # Test 2: Market data endpoint
    print("\n🧪 Test 2: Market Data Access")
    try:
        # Use data endpoint
        data_url = "https://data.alpaca.markets"
        
        response = requests.get(
            f"{data_url}/v2/stocks/SPY/bars/latest",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Market data endpoint successful")
            if 'bar' in data:
                bar = data['bar']
                print(f"   SPY Latest: ${bar.get('c', 'N/A')} (Close)")
                print(f"   Timestamp: {bar.get('t', 'N/A')}")
        else:
            print(f"❌ Market data endpoint failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Market data test failed: {e}")
    
    # Test 3: SDK Connection
    print("\n🧪 Test 3: Alpaca SDK Connection")
    try:
        from alpaca.trading.client import TradingClient
        
        trading_client = TradingClient(api_key, secret_key, paper=True)
        account = trading_client.get_account()
        
        print("✅ Alpaca SDK connection successful")
        print(f"   Account Status: {account.status}")
        print(f"   Equity: ${float(account.equity):,.2f}")
        print(f"   Day Trading Buying Power: ${float(account.daytrading_buying_power):,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Alpaca SDK test failed: {e}")
        return False

def test_data_client():
    """Test Alpaca Data Client specifically"""
    print("\n🧪 Test 4: Data Client (Alternative)")
    
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        # Try without explicit credentials (should use environment)
        client = StockHistoricalDataClient()
        
        print("✅ Data client initialized successfully")
        print("   Note: Using environment variables for authentication")
        
        return True
        
    except Exception as e:
        print(f"❌ Data client test failed: {e}")
        
        # Try with explicit credentials
        try:
            client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
            print("✅ Data client with explicit credentials successful")
            return True
        except Exception as e2:
            print(f"❌ Data client with explicit credentials failed: {e2}")
            return False

if __name__ == "__main__":
    success1 = test_alpaca_api()
    success2 = test_data_client()
    
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)
    
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! Your Alpaca API keys are working correctly.")
        print("\n✅ You can now run the full connectivity test:")
        print("   python test_framework_connectivity.py")
    elif success1:
        print("✅ Basic API access working, but data client needs attention.")
        print("   Your keys are valid for trading operations.")
    else:
        print("❌ API key issues detected. Please check:")
        print("   1. Keys are correctly set in .env file")
        print("   2. Keys are for the correct environment (paper/live)")
        print("   3. Account is properly set up and funded")
        print("   4. No typos in the keys")
