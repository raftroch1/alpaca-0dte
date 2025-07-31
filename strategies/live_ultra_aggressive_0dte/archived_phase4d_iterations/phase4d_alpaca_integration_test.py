#!/usr/bin/env python3
"""
Phase 4D: Alpaca Integration Test
Tests real Alpaca option pricing integration
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import gzip
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Alpaca clients
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame

class Phase4DAlpacaTest:
    def __init__(self):
        self.cache_dir = "../../thetadata/cached_data"
        
        # Initialize Alpaca option client
        try:
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key:
                print("⚠️ Missing Alpaca credentials - will test with fallback pricing")
                self.alpaca_option_client = None
            else:
                self.alpaca_option_client = OptionHistoricalDataClient(
                    api_key=api_key,
                    secret_key=secret_key
                )
                print("✅ Alpaca option client initialized successfully")
                
        except Exception as e:
            print(f"⚠️ Alpaca client error: {e}")
            self.alpaca_option_client = None
    
    def test_alpaca_option_pricing(self, date_str: str):
        print(f"\n🎯 Testing Alpaca option pricing for {date_str}")
        print("=" * 60)
        
        # Load SPY data to get current price
        try:
            cache_path = f"{self.cache_dir}/spy_bars/spy_bars_{date_str}.pkl.gz"
            with gzip.open(cache_path, 'rb') as f:
                spy_bars = pickle.load(f)
            
            current_spy = spy_bars['close'].iloc[0]
            print(f"📊 SPY Price: ${current_spy:.2f}")
            
        except Exception as e:
            print(f"❌ Error loading SPY data: {e}")
            return
        
        # Test bull put spread strikes
        short_strike = round(current_spy - 2.0)  # ~40 delta
        long_strike = short_strike - 12.0       # 12 points below
        
        print(f"📊 Testing Bull Put Spread:")
        print(f"   Short Put: ${short_strike:.0f}")
        print(f"   Long Put:  ${long_strike:.0f}")
        
        # Convert date for Alpaca API
        trade_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
        
        # Test real Alpaca pricing
        short_price = self.get_real_alpaca_price(short_strike, trade_date)
        long_price = self.get_real_alpaca_price(long_strike, trade_date)
        
        if short_price and long_price:
            net_credit = short_price - long_price
            max_profit = net_credit
            max_loss = 12.0 - net_credit
            risk_reward = max_loss / max_profit if max_profit > 0 else 0
            
            print(f"\n💰 REAL ALPACA PRICING:")
            print(f"   Short Put: ${short_price:.2f}")
            print(f"   Long Put:  ${long_price:.2f}")
            print(f"   Net Credit: ${net_credit:.2f}")
            print(f"   Max Profit: ${max_profit:.2f}")
            print(f"   Max Loss:   ${max_loss:.2f}")
            print(f"   Risk/Reward: {risk_reward:.1f}:1")
            
            # Position sizing (3 contracts)
            position_value = net_credit * 3 * 100
            max_position_profit = max_profit * 3 * 100
            max_position_loss = max_loss * 3 * 100
            
            print(f"\n📈 POSITION METRICS (3 contracts):")
            print(f"   Credit Received: ${position_value:.2f}")
            print(f"   Max Profit: ${max_position_profit:.2f}")
            print(f"   Max Loss: ${max_position_loss:.2f}")
            
            if net_credit >= 0.15 and risk_reward <= 20:
                print(f"✅ SPREAD QUALIFIES for trading")
            else:
                print(f"❌ Spread does not meet criteria (min $0.15 credit, max 20:1 R/R)")
        
        else:
            print("❌ Could not get real Alpaca pricing")
    
    def get_real_alpaca_price(self, strike: float, trade_date: str):
        if not self.alpaca_option_client:
            print(f"   Using fallback pricing for ${strike:.0f}P")
            return 0.50  # Fallback
        
        try:
            # Build Alpaca option symbol
            date_obj = datetime.strptime(trade_date, '%Y-%m-%d')
            exp_date = date_obj.strftime('%y%m%d')
            strike_str = f"{int(strike * 1000):08d}"
            alpaca_symbol = f"SPY{exp_date}P{strike_str}"
            
            print(f"   Requesting: {alpaca_symbol}")
            
            # Request option data
            request = OptionBarsRequest(
                symbol_or_symbols=[alpaca_symbol],
                timeframe=TimeFrame.Day,
                start=date_obj - timedelta(days=1),
                end=date_obj + timedelta(days=1)
            )
            
            option_data = self.alpaca_option_client.get_option_bars(request)
            
            if alpaca_symbol in option_data.data and len(option_data.data[alpaca_symbol]) > 0:
                bars = option_data.data[alpaca_symbol]
                
                for bar in bars:
                    if bar.timestamp.date() == date_obj.date():
                        price = float(bar.close)
                        print(f"   ✅ Real price: ${price:.2f}")
                        return price
                
                # Use latest if exact date not found
                if bars:
                    price = float(bars[-1].close)
                    print(f"   ✅ Latest price: ${price:.2f}")
                    return price
            
            print(f"   ❌ No data found for {alpaca_symbol}")
            return None
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None

if __name__ == "__main__":
    tester = Phase4DAlpacaTest()
    tester.test_alpaca_option_pricing('20240102')
