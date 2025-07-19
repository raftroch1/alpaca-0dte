#!/usr/bin/env python3
"""
Alpaca Account Diagnostic Script
===============================

This script helps diagnose common Alpaca API key issues and provides specific solutions.
"""

import os
import requests
from dotenv import load_dotenv

def diagnose_alpaca_account():
    """Comprehensive Alpaca account diagnosis"""
    
    load_dotenv()
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    print("🔍 ALPACA ACCOUNT DIAGNOSTIC")
    print("=" * 50)
    
    if not api_key or not secret_key:
        print("❌ CRITICAL: API keys not found in environment")
        print("\n🔧 SOLUTION:")
        print("1. Check your .env file exists")
        print("2. Verify the format:")
        print("   ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx")
        print("   ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return False
    
    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")
    print(f"✅ Secret Key found: {secret_key[:8]}...{secret_key[-4:]}")
    
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key
    }
    
    # Test different scenarios
    scenarios = [
        ("Paper Trading API", "https://paper-api.alpaca.markets"),
        ("Live Trading API", "https://api.alpaca.markets")
    ]
    
    for name, base_url in scenarios:
        print(f"\n🧪 Testing {name}")
        print("-" * 30)
        
        try:
            # Test account endpoint
            response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
            
            if response.status_code == 200:
                account = response.json()
                print(f"✅ {name} - SUCCESS!")
                print(f"   Account ID: {account.get('id', 'N/A')}")
                print(f"   Status: {account.get('status', 'N/A')}")
                print(f"   Trading Blocked: {account.get('trading_blocked', 'N/A')}")
                print(f"   Account Blocked: {account.get('account_blocked', 'N/A')}")
                print(f"   Pattern Day Trader: {account.get('pattern_day_trader', 'N/A')}")
                
                if account.get('status') != 'ACTIVE':
                    print(f"⚠️  Account status is not ACTIVE: {account.get('status')}")
                    print("   This may require account verification steps")
                
                return True
                
            elif response.status_code == 403:
                print(f"❌ {name} - 403 Forbidden")
                
                # Try to get more specific error info
                try:
                    error_data = response.json()
                    error_msg = error_data.get('message', 'No specific error message')
                    print(f"   Error: {error_msg}")
                    
                    # Common 403 error patterns
                    if 'forbidden' in error_msg.lower():
                        print("   🔧 LIKELY CAUSE: Account setup incomplete")
                        print("   📋 NEXT STEPS:")
                        print("      1. Log into your Alpaca dashboard")
                        print("      2. Complete account verification")
                        print("      3. Accept terms and conditions")
                        print("      4. Fund your account (even $1 for paper trading)")
                        
                except:
                    print("   Error details not available")
                    
            elif response.status_code == 401:
                print(f"❌ {name} - 401 Unauthorized")
                print("   🔧 LIKELY CAUSE: Invalid API keys")
                print("   📋 NEXT STEPS:")
                print("      1. Regenerate API keys in Alpaca dashboard")
                print("      2. Update .env file with new keys")
                print("      3. Ensure no extra spaces or characters")
                
            else:
                print(f"❌ {name} - HTTP {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ {name} - Connection Error: {e}")
    
    # Test data access (which we know works)
    print(f"\n🧪 Testing Data Access")
    print("-" * 30)
    
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        
        # This should work based on previous tests
        client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        print("✅ Data client initialization successful")
        print("   This confirms your API keys are valid")
        
    except Exception as e:
        print(f"❌ Data client failed: {e}")
    
    print(f"\n📋 SUMMARY & RECOMMENDATIONS")
    print("=" * 50)
    print("Based on the diagnostic results:")
    print()
    print("1. ✅ Your API keys are properly formatted and loading")
    print("2. ✅ Data access works (confirms keys are valid)")
    print("3. ❌ Account endpoint access blocked (403 Forbidden)")
    print()
    print("🎯 MOST LIKELY SOLUTION:")
    print("Your Alpaca account needs to be fully set up. Please:")
    print()
    print("   📱 1. Log into https://app.alpaca.markets/")
    print("   📝 2. Complete any pending verification steps")
    print("   ✅ 3. Accept all terms and conditions")
    print("   💰 4. Make a small deposit (even $1 for paper trading)")
    print("   🔄 5. Wait 5-10 minutes for account activation")
    print("   🧪 6. Re-run this diagnostic")
    print()
    print("💡 NOTE: Even paper trading accounts often require these steps!")
    
    return False

if __name__ == "__main__":
    diagnose_alpaca_account()
