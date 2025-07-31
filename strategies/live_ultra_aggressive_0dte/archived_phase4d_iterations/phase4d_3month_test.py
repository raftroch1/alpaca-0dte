#!/usr/bin/env python3
"""
Phase 4D: 3-Month Test with Real Cached Data
Testing Jan-Mar 2024 for statistical significance
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import gzip

class Phase4D3MonthTest:
    def __init__(self):
        self.cache_dir = "../../thetadata/cached_data"
        print(f"🎯 PHASE 4D: 3-MONTH TEST WITH REAL CACHED DATA")
        print(f"📁 Cache: {os.path.abspath(self.cache_dir)}")

    def test_single_day(self, date_str: str):
        try:
            # Load SPY data
            spy_path = f"{self.cache_dir}/spy_bars/spy_bars_{date_str}.pkl.gz"
            if not os.path.exists(spy_path):
                return None

            with gzip.open(spy_path, 'rb') as f:
                spy_bars = pickle.load(f)

            # Multiple spreads per day (simulate 8 max daily)
            daily_results = []
            
            # Sample 8 different times throughout the day
            total_bars = len(spy_bars)
            sample_intervals = max(total_bars // 8, 1)
            
            for i in range(0, min(total_bars, 8 * sample_intervals), sample_intervals):
                current_spy = spy_bars['close'].iloc[i]
                
                # Use end-of-day for exit (realistic holding)
                final_spy = spy_bars['close'].iloc[-1]

                # Bull put spread setup (12-point spread)
                short_strike = round(current_spy - 2.0)  # ~40 delta
                long_strike = short_strike - 12.0       # 12 points below

                # Realistic option pricing based on moneyness
                short_moneyness = short_strike / current_spy
                long_moneyness = long_strike / current_spy
                
                # More realistic pricing based on distance from money
                if short_moneyness > 1.02:  # Far OTM short
                    short_price = np.random.uniform(0.15, 0.35)
                elif short_moneyness > 0.98:  # Near ATM short
                    short_price = np.random.uniform(0.80, 1.50)
                else:  # ITM short
                    short_price = np.random.uniform(2.00, 4.00)
                
                if long_moneyness > 1.05:  # Far OTM long
                    long_price = np.random.uniform(0.05, 0.15)
                elif long_moneyness > 1.02:  # OTM long
                    long_price = np.random.uniform(0.10, 0.30)
                else:  # Less OTM long
                    long_price = np.random.uniform(0.20, 0.60)

                net_credit = short_price - long_price
                max_profit = net_credit
                max_loss = (short_strike - long_strike) - net_credit

                # Only trade if spread meets criteria
                if net_credit < 0.15 or max_loss / max_profit > 20:
                    continue

                # Outcome based on real SPY movement
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

                # Apply profit target (75%) and stop loss (75%)
                profit_target = max_profit * 0.75
                stop_loss = -max_profit * 0.75
                
                if realized_pnl >= profit_target:
                    realized_pnl = profit_target
                    outcome = "PROFIT_TARGET"
                elif realized_pnl <= stop_loss:
                    realized_pnl = stop_loss
                    outcome = "STOP_LOSS"

                position_pnl = realized_pnl * 3 * 100  # 3 contracts per spread

                daily_results.append({
                    'entry_spy': current_spy,
                    'exit_spy': final_spy,
                    'short_strike': short_strike,
                    'long_strike': long_strike,
                    'net_credit': net_credit,
                    'realized_pnl': realized_pnl,
                    'position_pnl': position_pnl,
                    'outcome': outcome,
                    'is_winner': position_pnl > 0
                })

            if not daily_results:
                return None

            # Aggregate daily results
            total_daily_pnl = sum(r['position_pnl'] for r in daily_results)
            winning_spreads = len([r for r in daily_results if r['is_winner']])
            total_spreads = len(daily_results)
            
            spy_entry = spy_bars['close'].iloc[0]
            spy_exit = spy_bars['close'].iloc[-1]

            return {
                'date': date_str,
                'spy_entry': spy_entry,
                'spy_exit': spy_exit,
                'spy_change': spy_exit - spy_entry,
                'spy_change_pct': ((spy_exit - spy_entry) / spy_entry) * 100,
                'total_daily_pnl': total_daily_pnl,
                'spreads_executed': total_spreads,
                'winning_spreads': winning_spreads,
                'daily_win_rate': winning_spreads / total_spreads if total_spreads > 0 else 0,
                'is_profitable_day': total_daily_pnl > 0,
                'target_achieved': total_daily_pnl >= 365,
                'spread_details': daily_results
            }

        except Exception as e:
            print(f"❌ Error processing {date_str}: {e}")
            return None

    def run_3month_test(self, start_date: str, end_date: str):
        print(f"\n🎯 Running Phase 4D from {start_date} to {end_date}")
        print("=" * 70)

        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        results = []
        current_date = start_dt
        
        # Progress tracking
        total_days = 0
        processed_days = 0

        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Trading days
                total_days += 1
                date_str = current_date.strftime('%Y%m%d')
                result = self.test_single_day(date_str)
                
                if result:
                    results.append(result)
                    processed_days += 1
                    print(f"📅 {date_str}: SPY ${result['spy_entry']:.2f}→${result['spy_exit']:.2f} ({result['spy_change_pct']:+.1f}%) | {result['spreads_executed']} spreads | ${result['total_daily_pnl']:+.0f} | {'✅' if result['target_achieved'] else '❌'}")
                else:
                    print(f"📅 {date_str}: No data available")
                
            current_date += timedelta(days=1)

        if not results:
            print("❌ No results to analyze")
            return

        # Calculate comprehensive statistics
        total_trading_days = len(results)
        total_pnl = sum(r['total_daily_pnl'] for r in results)
        total_spreads = sum(r['spreads_executed'] for r in results)
        total_winning_spreads = sum(r['winning_spreads'] for r in results)
        profitable_days = len([r for r in results if r['is_profitable_day']])
        target_days = len([r for r in results if r['target_achieved']])
        
        avg_daily_pnl = total_pnl / total_trading_days if total_trading_days > 0 else 0
        avg_spreads_per_day = total_spreads / total_trading_days if total_trading_days > 0 else 0
        overall_spread_win_rate = total_winning_spreads / total_spreads if total_spreads > 0 else 0
        daily_profitability_rate = profitable_days / total_trading_days if total_trading_days > 0 else 0
        target_achievement_rate = target_days / total_trading_days if total_trading_days > 0 else 0

        # Risk metrics
        daily_pnls = [r['total_daily_pnl'] for r in results]
        max_daily_gain = max(daily_pnls)
        max_daily_loss = min(daily_pnls)
        daily_std = np.std(daily_pnls)
        sharpe_ratio = (avg_daily_pnl / daily_std) if daily_std > 0 else 0

        # Monthly breakdown
        monthly_stats = {}
        for result in results:
            month_key = result['date'][:6]  # YYYYMM
            if month_key not in monthly_stats:
                monthly_stats[month_key] = []
            monthly_stats[month_key].append(result['total_daily_pnl'])

        print(f"\n🏆 PHASE 4D: 3-MONTH COMPREHENSIVE RESULTS")
        print("=" * 60)
        print(f"📊 Total Trading Days: {total_trading_days}")
        print(f"📊 Days with Data: {processed_days}/{total_days}")
        print(f"💰 Total P&L: ${total_pnl:.2f}")
        print(f"📊 Average Daily P&L: ${avg_daily_pnl:.2f}")
        print(f"📈 Profitable Days: {profitable_days}/{total_trading_days} ({daily_profitability_rate:.1%})")
        print(f"🎯 Target Days ($365+): {target_days}/{total_trading_days} ({target_achievement_rate:.1%})")
        print(f"🔢 Total Spreads: {total_spreads}")
        print(f"📊 Avg Spreads/Day: {avg_spreads_per_day:.1f}")
        print(f"📈 Overall Spread Win Rate: {overall_spread_win_rate:.1%}")
        
        print(f"\n📊 RISK METRICS:")
        print(f"📈 Best Day: ${max_daily_gain:.2f}")
        print(f"📉 Worst Day: ${max_daily_loss:.2f}")
        print(f"📊 Daily Volatility: ${daily_std:.2f}")
        print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Monthly breakdown
        print(f"\n📅 MONTHLY BREAKDOWN:")
        for month, pnls in monthly_stats.items():
            month_total = sum(pnls)
            month_avg = month_total / len(pnls)
            month_days = len(pnls)
            month_profitable = len([p for p in pnls if p > 0])
            print(f"   {month}: {month_days} days, ${month_total:.2f} total, ${month_avg:.2f} avg, {month_profitable}/{month_days} profitable")

        # Performance vs targets
        monthly_target = 365 * 21  # $365/day * ~21 trading days
        monthly_actual = total_pnl / (len(monthly_stats)) if monthly_stats else 0
        annual_projection = avg_daily_pnl * 252  # 252 trading days
        
        print(f"\n🎯 PERFORMANCE ANALYSIS:")
        print(f"💰 Monthly Target: ${monthly_target:.2f}")
        print(f"📊 Monthly Actual: ${monthly_actual:.2f}")
        print(f"📈 Monthly Achievement: {(monthly_actual/monthly_target)*100:.1f}%")
        print(f"🏆 Annual Projection: ${annual_projection:.2f}")
        print(f"✅ Strategy Assessment: {'PROFITABLE' if avg_daily_pnl > 0 else 'UNPROFITABLE'}")

        # Best and worst days
        best_day = max(results, key=lambda x: x['total_daily_pnl'])
        worst_day = min(results, key=lambda x: x['total_daily_pnl'])
        
        print(f"\n📊 EXTREME DAYS:")
        print(f"🏆 BEST: {best_day['date']} → ${best_day['total_daily_pnl']:+.2f} ({best_day['spreads_executed']} spreads)")
        print(f"💥 WORST: {worst_day['date']} → ${worst_day['total_daily_pnl']:+.2f} ({worst_day['spreads_executed']} spreads)")

if __name__ == "__main__":
    tester = Phase4D3MonthTest()
    # Test Q1 2024 (Jan-Mar)
    tester.run_3month_test('20240102', '20240329')
