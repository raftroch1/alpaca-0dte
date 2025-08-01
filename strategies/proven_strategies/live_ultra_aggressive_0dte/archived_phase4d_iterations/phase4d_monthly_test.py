#!/usr/bin/env python3
"""
Phase 4D Monthly Test with Real Cached Data
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import gzip

class Phase4DMonthlyTest:
    def __init__(self):
        self.cache_dir = "../../thetadata/cached_data"
        print(f"🎯 PHASE 4D: Monthly Test with Cached Data")
        print(f"📁 Cache: {os.path.abspath(self.cache_dir)}")

    def test_single_day(self, date_str: str):
        try:
            # Load SPY data
            spy_path = f"{self.cache_dir}/spy_bars/spy_bars_{date_str}.pkl.gz"
            if not os.path.exists(spy_path):
                return None

            with gzip.open(spy_path, 'rb') as f:
                spy_bars = pickle.load(f)

            current_spy = spy_bars['close'].iloc[0]
            final_spy = spy_bars['close'].iloc[-1]

            # Bull put spread setup
            short_strike = round(current_spy - 2.0)
            long_strike = short_strike - 12.0

            # Simple option pricing (fallback)
            short_price = 1.00
            long_price = 0.25
            net_credit = short_price - long_price
            max_profit = net_credit
            max_loss = (short_strike - long_strike) - net_credit

            # Outcome
            if final_spy > short_strike:
                realized_pnl = max_profit
                outcome = "FULL_PROFIT"
            elif final_spy < long_strike:
                realized_pnl = -max_loss
                outcome = "MAX_LOSS"
            else:
                intrinsic = short_strike - final_spy
                realized_pnl = net_credit - intrinsic
                outcome = "PARTIAL_LOSS"

            position_pnl = realized_pnl * 3 * 100  # 3 contracts

            return {
                'date': date_str,
                'spy_entry': current_spy,
                'spy_exit': final_spy,
                'spy_change': final_spy - current_spy,
                'spy_change_pct': ((final_spy - current_spy) / current_spy) * 100,
                'short_strike': short_strike,
                'long_strike': long_strike,
                'net_credit': net_credit,
                'realized_pnl': realized_pnl,
                'position_pnl': position_pnl,
                'outcome': outcome,
                'is_winner': position_pnl > 0
            }

        except Exception as e:
            return None

    def run_monthly_test(self, start_date: str, end_date: str):
        print(f"\n🎯 Running Phase 4D from {start_date} to {end_date}")
        print("=" * 60)

        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        results = []
        current_date = start_dt

        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Trading days
                date_str = current_date.strftime('%Y%m%d')
                result = self.test_single_day(date_str)
                
                if result:
                    results.append(result)
                    print(f"📅 {date_str}: SPY ${result['spy_entry']:.2f}→${result['spy_exit']:.2f} ({result['spy_change_pct']:+.1f}%) → ${result['position_pnl']:+.0f} ({result['outcome']})")
                
            current_date += timedelta(days=1)

        if not results:
            print("❌ No results to analyze")
            return

        # Calculate summary statistics
        total_pnl = sum(r['position_pnl'] for r in results)
        winning_trades = len([r for r in results if r['is_winner']])
        total_trades = len(results)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_daily_pnl = total_pnl / total_trades if total_trades > 0 else 0
        profitable_days = len([r for r in results if r['position_pnl'] > 0])
        profitability_rate = profitable_days / total_trades if total_trades > 0 else 0

        print(f"\n🏆 PHASE 4D: MONTHLY RESULTS")
        print("=" * 40)
        print(f"📊 Days Tested: {total_trades}")
        print(f"💰 Total P&L: ${total_pnl:.2f}")
        print(f"📊 Avg Daily P&L: ${avg_daily_pnl:.2f}")
        print(f"📈 Winning Trades: {winning_trades}/{total_trades} ({win_rate:.1%})")
        print(f"📈 Profitable Days: {profitable_days}/{total_trades} ({profitability_rate:.1%})")
        print(f"🎯 Daily Target: $365.00")
        print(f"✅ Target Achievement: {'EXCEEDS' if avg_daily_pnl >= 365 else 'BELOW'}")

        # Best and worst days
        best_day = max(results, key=lambda x: x['position_pnl'])
        worst_day = min(results, key=lambda x: x['position_pnl'])
        
        print(f"\n📊 BEST DAY: {best_day['date']} → ${best_day['position_pnl']:+.2f}")
        print(f"📊 WORST DAY: {worst_day['date']} → ${worst_day['position_pnl']:+.2f}")

if __name__ == "__main__":
    tester = Phase4DMonthlyTest()
    tester.run_monthly_test('20240102', '20240112')  # First 2 weeks of Jan 2024
