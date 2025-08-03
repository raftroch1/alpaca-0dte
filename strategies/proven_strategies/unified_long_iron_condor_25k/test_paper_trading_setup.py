#!/usr/bin/env python3
"""
🧪 UNIFIED PAPER TRADING SETUP TEST
==================================

Test script to validate paper trading environment and configuration.
Run this before launching live paper trading to catch setup issues.

Author: Strategy Development Framework
Date: 2025-01-31
Version: v1.0
"""

import os
import sys
from dotenv import load_dotenv

def test_environment_setup():
    """Test environment variables and API keys"""
    print("🔍 Testing environment setup...")
    
    # Load environment variables
    load_dotenv()
    
    # Check API keys
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key:
        print("❌ ALPACA_API_KEY not found in environment")
        return False
    
    if not secret_key:
        print("❌ ALPACA_SECRET_KEY not found in environment")
        return False
    
    print(f"✅ ALPACA_API_KEY: {api_key[:8]}...")
    print(f"✅ ALPACA_SECRET_KEY: {secret_key[:8]}...")
    return True

def test_alpaca_imports():
    """Test Alpaca SDK imports"""
    print("\n📦 Testing Alpaca SDK imports...")
    
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOptionContractsRequest, OptionLegRequest
        from alpaca.trading.enums import OrderSide, OrderClass, ContractType
        from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
        print("✅ All Alpaca imports successful")
        return True
    except ImportError as e:
        print(f"❌ Alpaca import error: {e}")
        print("💡 Install with: pip install alpaca-py")
        return False

def test_alpaca_connection():
    """Test connection to Alpaca API"""
    print("\n🔌 Testing Alpaca API connection...")
    
    try:
        from alpaca.trading.client import TradingClient
        
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            print("⚠️ Skipping connection test - missing API keys")
            return False
        
        # Test connection
        client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True
        )
        
        account = client.get_account()
        print(f"✅ Connected to Alpaca Paper Trading")
        print(f"📊 Account Value: ${float(account.portfolio_value):,.2f}")
        print(f"💰 Buying Power: ${float(account.buying_power):,.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Alpaca connection failed: {e}")
        print("💡 Check your API keys and network connection")
        return False

def test_strategy_import():
    """Test strategy class import"""
    print("\n🎪 Testing strategy import...")
    
    try:
        from unified_long_condor_paper_trading import UnifiedLongCondorPaperTrading
        print("✅ Strategy class import successful")
        
        # Test instantiation (without running)
        strategy = UnifiedLongCondorPaperTrading()
        print("✅ Strategy instantiation successful")
        print(f"🎯 Target Daily P&L: ${strategy.target_daily_pnl}")
        print(f"🛡️ Max Daily Loss: ${strategy.max_daily_loss}")
        print(f"📊 Primary Contracts: {strategy.params['primary_base_contracts']}")
        return True
        
    except Exception as e:
        print(f"❌ Strategy import/instantiation failed: {e}")
        return False

def test_market_hours():
    """Test market hours functionality"""
    print("\n🕐 Testing market hours detection...")
    
    try:
        from unified_long_condor_paper_trading import UnifiedLongCondorPaperTrading
        from datetime import datetime
        
        strategy = UnifiedLongCondorPaperTrading()
        
        is_market_hours = strategy.is_market_hours()
        can_open_positions = strategy.can_open_new_positions()
        should_close = strategy.should_close_positions()
        
        current_time = datetime.now().strftime("%H:%M:%S")
        
        print(f"⏰ Current Time: {current_time}")
        print(f"📈 Market Hours: {'Yes' if is_market_hours else 'No'}")
        print(f"🆕 Can Open Positions: {'Yes' if can_open_positions else 'No'}")
        print(f"🔒 Should Close Positions: {'Yes' if should_close else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Market hours test failed: {e}")
        return False

def test_logs_directory():
    """Test logs directory creation"""
    print("\n📋 Testing logs directory...")
    
    try:
        os.makedirs("logs", exist_ok=True)
        
        if os.path.exists("logs") and os.path.isdir("logs"):
            print("✅ Logs directory ready")
            
            # Test write permissions
            test_file = "logs/test_write.tmp"
            with open(test_file, 'w') as f:
                f.write("test")
            
            os.remove(test_file)
            print("✅ Logs directory writable")
            return True
        else:
            print("❌ Failed to create logs directory")
            return False
            
    except Exception as e:
        print(f"❌ Logs directory test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🧪 UNIFIED PAPER TRADING SETUP VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Alpaca SDK Imports", test_alpaca_imports),
        ("Alpaca API Connection", test_alpaca_connection),
        ("Strategy Import", test_strategy_import),
        ("Market Hours Detection", test_market_hours),
        ("Logs Directory", test_logs_directory)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Ready for paper trading!")
        print("\n🚀 To start paper trading:")
        print("   python paper_trading_launcher.py --account 25k")
        return True
    else:
        print("⚠️ Some tests failed - fix issues before paper trading")
        print("\n💡 Common fixes:")
        print("   - Add API keys to .env file")
        print("   - Install alpaca-py: pip install alpaca-py")
        print("   - Check network connection")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)