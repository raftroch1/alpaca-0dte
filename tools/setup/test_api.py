#!/usr/bin/env python3
"""
Test Alpaca API connectivity and permissions
"""
import os
import sys
from dotenv import load_dotenv
from datetime import datetime

# Add alpaca imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import ContractType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

def main():
    print("🔧 Testing Alpaca API connectivity...")
    
    # Load environment variables
    env_path = os.path.join(os.path.dirname(os.getcwd()), '.env')
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ API keys not found in environment")
        return
    
    print(f"✅ API keys loaded (Key starts with: {api_key[:4]}...)")
    
    try:
        # Test trading client
        trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True
        )
        
        print("\n📊 Testing account access...")
        account = trading_client.get_account()
        print(f"✅ Account access successful")
        print(f"   Account ID: {account.id}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        
        print("\n🕐 Testing market clock...")
        try:
            clock = trading_client.get_clock()
            print(f"✅ Clock access successful")
            print(f"   Market is: {'OPEN' if clock.is_open else 'CLOSED'}")
            print(f"   Next open: {clock.next_open}")
        except Exception as e:
            print(f"❌ Clock access failed: {e}")
        
        print("\n📈 Testing stock data access...")
        try:
            stock_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key
            )
            
            request = StockBarsRequest(
                symbol_or_symbols=["SPY"],
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=datetime.now().replace(hour=9, minute=30),
                end=datetime.now()
            )
            
            bars = stock_client.get_stock_bars(request)
            df = bars.df
            print(f"✅ Stock data access successful")
            print(f"   Retrieved {len(df)} SPY bars")
            if not df.empty:
                latest_price = df['close'].iloc[-1]
                print(f"   Latest SPY price: ${latest_price:.2f}")
            
        except Exception as e:
            print(f"❌ Stock data access failed: {e}")
        
        print("\n🎯 Testing option contract discovery...")
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            request = GetOptionContractsRequest(
                underlying_symbol="SPY",
                expiration_date=today,
                contract_type=ContractType.CALL,
                strike_price_gte="570",
                strike_price_lte="590"
            )
            
            contracts = trading_client.get_option_contracts(request)
            print(f"✅ Option contract discovery successful")
            print(f"   Found {len(contracts)} 0DTE SPY call options")
            
            if contracts:
                sample = contracts[0]
                print(f"   Sample contract: {sample.symbol}")
                print(f"   Strike: ${sample.strike_price}")
                
        except Exception as e:
            print(f"❌ Option contract discovery failed: {e}")
            
        print("\n🎉 API connectivity test complete!")
        
    except Exception as e:
        print(f"❌ Trading client initialization failed: {e}")

if __name__ == "__main__":
    main()
