#!/usr/bin/env python3
"""
ALPACA CONNECTION DIAGNOSTIC TOOL
=================================

Tests Alpaca API connection and permissions
Helps identify the exact issue with 403 Forbidden errors
"""

import os
import sys
from dotenv import load_dotenv

# Add alpaca imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

def test_connection():
    """Test Alpaca connection and permissions"""
    print("🔧 ALPACA CONNECTION DIAGNOSTIC")
    print("=" * 50)
    
    # Load environment
    load_dotenv(override=True)
    
    # Get API keys
    api_key = os.getenv('ALPACA_API_KEY_ID')
    secret_key = os.getenv('ALPACA_API_SECRET_KEY')
    
    print(f"📋 API Key: {api_key[:10]}...{api_key[-4:] if api_key else 'NOT FOUND'}")
    print(f"📋 Secret Key: {'***FOUND***' if secret_key else 'NOT FOUND'}")
    print(f"📋 Key Type: {'Paper Trading' if api_key and api_key.startswith('PK') else 'Live Trading' if api_key else 'Unknown'}")
    
    if not api_key or not secret_key:
        print("❌ Missing API keys in environment")
        return False
    
    # Test 1: Trading Client Connection
    print("\n🔌 TEST 1: Trading Client Connection")
    try:
        trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True
        )
        print("✅ Trading client initialized")
        
        # Test account access
        try:
            account = trading_client.get_account()
            print(f"✅ Account access successful")
            print(f"   Account ID: {account.id}")
            print(f"   Status: {account.status}")
            print(f"   Trading Blocked: {account.trading_blocked}")
            print(f"   Options Trading Level: {account.options_trading_level}")
            print(f"   Account Equity: ${float(account.equity):,.2f}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            
            if account.trading_blocked:
                print("⚠️ WARNING: Trading is blocked on this account")
            
            if not account.options_trading_level or account.options_trading_level == 0:
                print("⚠️ WARNING: Options trading not enabled (Level 0)")
            else:
                print(f"✅ Options trading enabled (Level {account.options_trading_level})")
                
        except Exception as e:
            print(f"❌ Account access failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Trading client failed: {e}")
        return False
    
    # Test 2: Data Client Connection
    print("\n📊 TEST 2: Data Client Connection")
    try:
        data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key
        )
        print("✅ Data client initialized")
        
        # Test data access
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            request = StockLatestTradeRequest(symbol_or_symbols="SPY")
            response = data_client.get_stock_latest_trade(request)
            spy_price = float(response["SPY"].price)
            print(f"✅ Data access successful")
            print(f"   SPY Price: ${spy_price:.2f}")
            
        except Exception as e:
            print(f"❌ Data access failed: {e}")
            print("ℹ️ This might be due to market data subscription limits")
            
    except Exception as e:
        print(f"❌ Data client failed: {e}")
        return False
    
    # Test 3: Options Contract Search
    print("\n🎯 TEST 3: Options Contract Search")
    try:
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import ContractType
        from datetime import datetime
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        request = GetOptionContractsRequest(
            underlying_symbol="SPY",
            expiration_date=today,
            contract_type=ContractType.CALL,
            strike_price=550.0
        )
        
        contracts = trading_client.get_option_contracts(request)
        
        if contracts:
            print(f"✅ Found {len(contracts)} option contracts")
            print(f"   Example: {contracts[0].symbol}")
        else:
            print("⚠️ No option contracts found (might be weekend/holiday)")
            
    except Exception as e:
        print(f"❌ Options search failed: {e}")
        print("ℹ️ This might indicate options permissions issue")
    
    print("\n📝 DIAGNOSIS SUMMARY")
    print("=" * 50)
    print("✅ Connection tests complete")
    print("📋 If you see 403 Forbidden errors:")
    print("   1. Check if API keys are correct and active")
    print("   2. Verify options trading is enabled in Alpaca dashboard")
    print("   3. Ensure you're using Paper Trading keys (PK prefix)")
    print("   4. Check if account is funded and in good standing")
    
    return True

if __name__ == "__main__":
    test_connection() 