#!/usr/bin/env python3
"""
SPY Data Diagnostic Script
=========================

This script tests exactly what's happening with SPY data retrieval
in your live strategy to identify why no signals are being generated.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Add alpaca imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

def test_spy_data_retrieval():
    """Test SPY data retrieval exactly like the live strategy does"""
    
    print("🔍 SPY DATA DIAGNOSTIC TEST")
    print("=" * 50)
    
    # Load environment variables from current directory
    env_path = '.env'
    load_dotenv(dotenv_path=env_path)
    
    print(f"🔧 Loading environment from: {os.path.abspath(env_path)}")
    print(f"🔑 API Key found: {'✅' if os.getenv('ALPACA_API_KEY') else '❌'}")
    print(f"🔐 Secret Key found: {'✅' if os.getenv('ALPACA_SECRET_KEY') else '❌'}")
    
    # Initialize client exactly like the strategy
    try:
        stock_client = StockHistoricalDataClient(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY")
        )
        print("✅ Stock client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return
    
    # Test the exact same method as the strategy
    def get_spy_minute_data(minutes_back: int = 50):
        """Replicate the exact method from live strategy"""
        try:
            print(f"\n🔍 Testing SPY data retrieval ({minutes_back} minutes back)...")
            
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=minutes_back)
            
            print(f"📅 Time range: {start_time} to {end_time}")
            
            request = StockBarsRequest(
                symbol_or_symbols="SPY",
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start_time,
                end=end_time
            )
            
            print("📡 Making API request...")
            bars = stock_client.get_stock_bars(request)
            
            print("📊 Converting to DataFrame...")
            df = bars.df.reset_index()
            
            if df.empty:
                print("❌ No SPY data received - DataFrame is empty")
                return pd.DataFrame()
            
            print(f"✅ Retrieved {len(df)} SPY minute bars")
            print(f"📈 Latest price: ${df['close'].iloc[-1]:.2f}")
            print(f"📊 Data columns: {list(df.columns)}")
            print(f"🕐 Time range: {df.index[0]} to {df.index[-1]}")
            
            # Check if we have enough data for technical indicators
            sma_period = 15  # From strategy parameters
            if len(df) < sma_period:
                print(f"⚠️ Insufficient data: {len(df)} bars, need {sma_period}")
                return pd.DataFrame()
            
            print(f"✅ Sufficient data for technical analysis ({len(df)} >= {sma_period})")
            
            return df
            
        except Exception as e:
            print(f"❌ Failed to get SPY data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    # Test data retrieval
    spy_data = get_spy_minute_data(50)
    
    if not spy_data.empty:
        print(f"\n🎯 SIGNAL GENERATION TEST")
        print("-" * 30)
        
        # Test signal generation exactly like the strategy
        def generate_trading_signals(spy_data):
            """Replicate signal generation from strategy"""
            try:
                signals = []
                
                if len(spy_data) < 20:
                    print(f"❌ Not enough data for signals: {len(spy_data)} < 20")
                    return signals
                
                # Simple signal generation (placeholder)
                current_price = spy_data['close'].iloc[-1]
                
                # Conservative signal: only trade on strong momentum
                price_change = (current_price - spy_data['close'].iloc[-5]) / spy_data['close'].iloc[-5]
                
                print(f"📊 Current SPY price: ${current_price:.2f}")
                print(f"📈 5-minute price change: {price_change:.4f} ({price_change*100:.2f}%)")
                
                if abs(price_change) > 0.0005:  # 0.05% movement (much more sensitive)
                    signal_type = "CALL" if price_change > 0 else "PUT"
                    confidence = min(abs(price_change) * 100, 0.8)  # Cap confidence at 0.8
                    
                    signals.append({
                        'type': signal_type,
                        'confidence': confidence,
                        'spy_price': current_price,
                        'timestamp': datetime.now()
                    })
                    
                    print(f"🎯 SIGNAL DETECTED: {signal_type} (confidence: {confidence:.3f})")
                else:
                    print(f"⏸️ No signal: price change {abs(price_change)*100:.2f}% < 0.2% threshold")
                
                return signals
                
            except Exception as e:
                print(f"❌ Error generating signals: {e}")
                import traceback
                traceback.print_exc()
                return []
        
        # Test signal generation
        signals = generate_trading_signals(spy_data)
        
        if signals:
            print(f"\n✅ SUCCESS: Generated {len(signals)} trading signals!")
            for signal in signals:
                print(f"   📊 {signal['type']} signal at ${signal['spy_price']:.2f} (confidence: {signal['confidence']:.3f})")
        else:
            print(f"\n⏸️ No signals generated (market conditions don't meet criteria)")
    else:
        print(f"\n❌ FAILED: Cannot generate signals without SPY data")
    
    print(f"\n🎯 DIAGNOSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    test_spy_data_retrieval()
