#!/usr/bin/env python3
"""
Quick test of 25k scaling strategy on a single day
"""

from phase4d_balanced_25k_scale import Phase4DBalanced25k

def test_single_day():
    print("🔧 Testing 25K Scaling Strategy on Single Day")
    
    strategy = Phase4DBalanced25k()
    test_date = '20240301'  # Known working date
    
    print(f"📊 Running strategy for {test_date}")
    result = strategy.run_strategy(test_date)
    
    print("\n📋 RESULTS:")
    print(f"Date: {result.get('date', 'N/A')}")
    print(f"Trade Executed: {result.get('trade_executed', False)}")
    print(f"P&L: ${result.get('pnl', 0):.2f}")
    
    if 'trade_details' in result:
        print(f"Trade Details: {result['trade_details']}")
    
    if 'reason' in result:
        print(f"Filter Reason: {result['reason']}")
    
    return result

if __name__ == "__main__":
    test_single_day()