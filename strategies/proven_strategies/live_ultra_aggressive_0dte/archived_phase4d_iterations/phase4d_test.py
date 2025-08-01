#!/usr/bin/env python3
"""
🎯 PHASE 4D: BULL PUT SPREADS - CACHED DATA TEST
================================================

Simple test with your cached ThetaData
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import gzip
import argparse
import logging
from typing import Dict, List, Optional, Tuple

class Phase4DTest:
    def __init__(self):
        self.cache_dir = "../../thetadata/cached_data"
        print(f"🎯 PHASE 4D: Testing with cached data")
        print(f"📁 Cache: {os.path.abspath(self.cache_dir)}")
    
    def load_cached_spy_data(self, date_str: str) -> Optional[pd.DataFrame]:
        try:
            cache_path = f"{self.cache_dir}/spy_bars/spy_bars_{date_str}.pkl.gz"
            
            if not os.path.exists(cache_path):
                print(f"❌ No cached SPY data for {date_str}")
                return None
            
            with gzip.open(cache_path, 'rb') as f:
                spy_bars = pickle.load(f)
            
            print(f"✅ Loaded {len(spy_bars)} cached SPY bars for {date_str}")
            print(f"📊 SPY Range: ${spy_bars['low'].min():.2f} - ${spy_bars['high'].max():.2f}")
            print(f"📊 Open: ${spy_bars['open'].iloc[0]:.2f}, Close: ${spy_bars['close'].iloc[-1]:.2f}")
            return spy_bars
            
        except Exception as e:
            print(f"❌ Error loading cached SPY data: {e}")
            return None
    
    def load_cached_option_data(self, date_str: str) -> Optional[Dict]:
        try:
            cache_path = f"{self.cache_dir}/option_chains/option_chains_{date_str}.pkl.gz"
            
            if not os.path.exists(cache_path):
                print(f"❌ No cached option data for {date_str}")
                return None
            
            with gzip.open(cache_path, 'rb') as f:
                option_data = pickle.load(f)
            
            puts = option_data.get('puts', {})
            calls = option_data.get('calls', {})
            spy_price = option_data.get('spy_price', 0)
            
            print(f"✅ Loaded cached option data for {date_str}")
            print(f"📊 SPY Price: ${spy_price:.2f}")
            print(f"📊 Put strikes: {len(puts)}, Call strikes: {len(calls)}")
            
            if puts:
                put_strikes = sorted(puts.keys())
                print(f"📊 Put range: ${put_strikes[0]:.0f} - ${put_strikes[-1]:.0f}")
                print(f"📊 Sample put prices: {dict(list(puts.items())[:3])}")
            
            return option_data
            
        except Exception as e:
            print(f"❌ Error loading cached option data: {e}")
            return None
    
    def test_bull_put_spread(self, date_str: str):
        print(f"\n🎯 Testing Phase 4D for {date_str}")
        print("=" * 50)
        
        # Load real cached data
        spy_bars = self.load_cached_spy_data(date_str)
        option_data = self.load_cached_option_data(date_str)
        
        if spy_bars is None:
            print("❌ Cannot test without SPY data")
            return
        
        # Get SPY price for strike selection
        current_spy = spy_bars['close'].iloc[0]
        
        # Calculate bull put spread strikes
        short_strike = round(current_spy - 2.0)  # ~40 delta
        long_strike = short_strike - 12.0       # 12 points below
        
        print(f"\n📊 BULL PUT SPREAD ANALYSIS:")
        print(f"   SPY Price: ${current_spy:.2f}")
        print(f"   Short Put: ${short_strike:.0f} (sell for income)")
        print(f"   Long Put:  ${long_strike:.0f} (buy for protection)")
        print(f"   Spread Width: ${short_strike - long_strike:.0f} points")
        
        # Get option prices from cached data
        if option_data and 'puts' in option_data:
            puts = option_data['puts']
            short_price = puts.get(int(short_strike), 1.00)
            long_price = puts.get(int(long_strike), 0.25)
            
            net_credit = short_price - long_price
            max_profit = net_credit
            max_loss = (short_strike - long_strike) - net_credit
            risk_reward = max_loss / max_profit if max_profit > 0 else 0
            
            print(f"\n💰 SPREAD PRICING (from cached data):")
            print(f"   Short Put Price: ${short_price:.2f}")
            print(f"   Long Put Price:  ${long_price:.2f}")
            print(f"   Net Credit:      ${net_credit:.2f}")
            print(f"   Max Profit:      ${max_profit:.2f}")
            print(f"   Max Loss:        ${max_loss:.2f}")
            print(f"   Risk/Reward:     {risk_reward:.1f}:1")
            
            # Test outcome with real SPY movement
            final_spy = spy_bars['close'].iloc[-1]
            spy_change = final_spy - current_spy
            spy_change_pct = (spy_change / current_spy) * 100
            
            print(f"\n📈 MARKET OUTCOME:")
            print(f"   Entry SPY: ${current_spy:.2f}")
            print(f"   Exit SPY:  ${final_spy:.2f}")
            print(f"   Movement:  ${spy_change:+.2f} ({spy_change_pct:+.1f}%)")
            
            # Bull put spread outcome
            if final_spy > short_strike:
                realized_pnl = max_profit
                outcome = "FULL PROFIT (SPY above short strike)"
            elif final_spy < long_strike:
                realized_pnl = -max_loss
                outcome = "MAX LOSS (SPY below long strike)"
            else:
                intrinsic = short_strike - final_spy
                realized_pnl = net_credit - intrinsic
                outcome = f"PARTIAL LOSS (SPY between strikes)"
            
            position_pnl = realized_pnl * 3 * 100  # 3 contracts
            
            print(f"\n🎯 SPREAD RESULT:")
            print(f"   Per Spread P&L: ${realized_pnl:.2f}")
            print(f"   Position P&L:   ${position_pnl:.2f} (3 contracts)")
            print(f"   Outcome: {outcome}")
            
            if position_pnl >= 365:
                print(f"✅ TARGET ACHIEVED! (${position_pnl:.2f} >= $365)")
            else:
                print(f"❌ Target missed (${position_pnl:.2f} < $365)")
        
        else:
            print("⚠️ No option data available for pricing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 4D Cached Data Test')
    parser.add_argument('--date', type=str, help='Date (YYYYMMDD)', default='20240102')
    
    args = parser.parse_args()
    
    tester = Phase4DTest()
    tester.test_bull_put_spread(args.date)
