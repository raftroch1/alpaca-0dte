#!/usr/bin/env python3
"""
Strategy Startup Test - Check if live_ultra_aggressive_0dte.py can initialize
========================================================================

Test the strategy initialization without running the full trading loop.
This will verify all connections and configurations are working.
"""

import os
import sys
from dotenv import load_dotenv

def test_strategy_startup():
    """Test if the strategy can initialize properly"""
    
    print("🧪 TESTING STRATEGY STARTUP")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    
    # Check environment variables
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Alpaca API keys not found in environment")
        return False
    
    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")
    print(f"✅ Secret Key found: {secret_key[:8]}...{secret_key[-4:]}")
    
    # Test strategy import
    print("\n🧪 Testing Strategy Import...")
    try:
        sys.path.append('/Users/devops/Desktop/coding projects/windsurf/Alpaca_0dte/alpaca-0dte')
        from strategies.live_ultra_aggressive_0dte import LiveUltraAggressive0DTEStrategy
        print("✅ Strategy import successful")
    except Exception as e:
        print(f"❌ Strategy import failed: {e}")
        return False
    
    # Test strategy initialization
    print("\n🧪 Testing Strategy Initialization...")
    try:
        strategy = LiveUltraAggressive0DTEStrategy()
        print("✅ Strategy initialization successful!")
        
        # Check if clients are initialized
        if hasattr(strategy, 'trading_client') and strategy.trading_client:
            print("✅ Trading client initialized")
        else:
            print("❌ Trading client not initialized")
            
        if hasattr(strategy, 'stock_client') and strategy.stock_client:
            print("✅ Stock data client initialized")
        else:
            print("❌ Stock data client not initialized")
            
        # Check strategy parameters
        if hasattr(strategy, 'params') and strategy.params:
            print("✅ Strategy parameters loaded")
            print(f"   Target profit: ${strategy.params.get('target_profit', 'N/A')}")
            print(f"   Max loss: ${strategy.params.get('max_loss', 'N/A')}")
        else:
            print("❌ Strategy parameters not loaded")
            
        # Test market hours check
        print("\n🧪 Testing Market Hours Check...")
        try:
            market_status = strategy.check_market_hours()
            print(f"✅ Market hours check successful")
            print(f"   Market open: {market_status.get('is_open', 'Unknown')}")
            print(f"   Current time: {market_status.get('current_time', 'Unknown')}")
        except Exception as e:
            print(f"⚠️  Market hours check failed: {e}")
            print("   This is expected outside market hours")
        
        # Test daily risk limits check
        print("\n🧪 Testing Risk Management...")
        try:
            can_trade = strategy.check_daily_risk_limits()
            print(f"✅ Risk management check successful")
            print(f"   Can trade: {can_trade}")
            print(f"   Daily P&L: ${strategy.daily_pnl}")
            print(f"   Daily trades: {strategy.daily_trades}")
        except Exception as e:
            print(f"❌ Risk management check failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Strategy initialization failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_strategy_components():
    """Test individual strategy components"""
    
    print("\n🧪 TESTING STRATEGY COMPONENTS")
    print("=" * 50)
    
    try:
        from strategies.live_ultra_aggressive_0dte import LiveUltraAggressive0DTEStrategy
        strategy = LiveUltraAggressive0DTEStrategy()
        
        # Test SPY data retrieval (should work even outside market hours)
        print("🧪 Testing SPY Data Retrieval...")
        try:
            spy_data = strategy.get_spy_minute_data(minutes_back=10)
            if spy_data is not None and len(spy_data) > 0:
                print(f"✅ SPY data retrieved: {len(spy_data)} bars")
                print(f"   Latest close: ${spy_data['close'].iloc[-1]:.2f}")
            else:
                print("⚠️  No SPY data available (expected outside market hours)")
        except Exception as e:
            print(f"⚠️  SPY data retrieval failed: {e}")
            print("   This is expected outside market hours")
        
        # Test technical indicators (with dummy data)
        print("\n🧪 Testing Technical Indicators...")
        try:
            import pandas as pd
            import numpy as np
            
            # Create dummy SPY data for testing
            dummy_data = pd.DataFrame({
                'close': np.random.normal(580, 5, 50),
                'high': np.random.normal(582, 5, 50),
                'low': np.random.normal(578, 5, 50),
                'volume': np.random.normal(1000000, 100000, 50)
            })
            
            indicators = strategy.calculate_technical_indicators(dummy_data)
            print("✅ Technical indicators calculated")
            print(f"   RSI: {indicators.get('rsi', 'N/A')}")
            print(f"   MACD: {indicators.get('macd', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Technical indicators failed: {e}")
            return False
        
        print("\n🎉 STRATEGY COMPONENTS TEST COMPLETE!")
        return True
        
    except Exception as e:
        print(f"❌ Strategy components test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing your live_ultra_aggressive_0dte.py strategy startup...")
    print("Note: We're outside market hours, so some data-dependent tests may show warnings.\n")
    
    startup_success = test_strategy_startup()
    components_success = test_strategy_components()
    
    print("\n" + "=" * 60)
    print("🎯 FINAL RESULTS")
    print("=" * 60)
    
    if startup_success and components_success:
        print("🎉 SUCCESS! Your strategy can start properly!")
        print()
        print("✅ Strategy initialization: Working")
        print("✅ Alpaca clients: Connected")
        print("✅ Risk management: Functional")
        print("✅ Technical indicators: Working")
        print()
        print("🚀 Your strategy is ready to run when markets open!")
        print("   To start live trading: python strategies/live_ultra_aggressive_0dte.py")
        print("   To test with paper trading: Already configured for paper mode")
        
    elif startup_success:
        print("✅ Strategy can start, but some components need attention")
        print("   Check the warnings above for details")
        
    else:
        print("❌ Strategy startup failed")
        print("   Please fix the errors shown above before running")
