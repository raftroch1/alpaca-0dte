#!/usr/bin/env python3
"""
🗓️ PHASE 4D: MONTHLY BULL PUT SPREADS TESTING - $300-500 DAILY TARGETS
======================================================================

Comprehensive monthly validation of the Phase 4D Bull Put Spreads strategy
targeting $300-500 daily profit on a $25K account.

TESTING FRAMEWORK:
✅ Full month of trading days (~20 days)
✅ Bull put spreads with enhanced position sizing
✅ Delta/IV-based selection criteria  
✅ High-frequency execution (20+ trades/day target)
✅ Statistical validation of daily profit targets
✅ Performance comparison vs previous phases

TARGET VALIDATION:
- Daily P&L: $300-500 (1.2-2% of $25K account)
- Win Rate: 70%+ (credit spreads advantage)
- Trade Frequency: 15-25 trades/day
- Monthly P&L: $6,000-10,000 (24-40% monthly return)
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append('/Users/devops/Desktop/coding projects/windsurf/Alpaca_0dte/alpaca-0dte')

from phase4d_final_profitability import Phase4DFinalProfitabilityStrategy


class Phase4DMonthlyRunner:
    """Monthly testing framework for Phase 4D bull put spreads strategy"""
    
    def __init__(self):
        """Initialize the monthly runner"""
        self.strategy = Phase4DFinalProfitabilityStrategy()
        
    def get_trading_days(self, year: int, month: int) -> List[str]:
        """Get list of trading days for a given month"""
        # Generate all days in the month
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            # Only include weekdays (Mon-Fri)
            if current_date.weekday() < 5:
                trading_days.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        return trading_days
    
    def run_monthly_backtest(self, year: int, month: int) -> Dict:
        """Run complete monthly backtest"""
        print(f"\n🎯 PHASE 4D MONTHLY BULL PUT SPREADS BACKTEST")
        print(f"📅 Testing Period: {year}-{month:02d}")
        print(f"🎯 Target: $300-500 daily profit with bull put spreads")
        print("=" * 60)
        
        # Get trading days
        trading_days = self.get_trading_days(year, month)
        print(f"📅 Trading Days: {len(trading_days)}")
        
        # Monthly results
        monthly_results = []
        total_pnl = 0.0
        total_trades = 0
        target_days = 0
        successful_days = 0
        
        # Daily results tracking
        daily_pnls = []
        daily_trades = []
        daily_win_rates = []
        
        # Run each trading day
        for i, date_str in enumerate(trading_days, 1):
            print(f"\n📊 Day {i}/{len(trading_days)}: {date_str}")
            
            try:
                # Run daily session
                daily_result = self.strategy.run_daily_session(date_str)
                monthly_results.append(daily_result)
                
                # Update totals
                daily_pnl = daily_result.get('total_pnl', 0.0)
                daily_trade_count = daily_result.get('trades', 0)
                daily_win_rate = daily_result.get('win_rate', 0.0)
                target_achieved = daily_result.get('target_achieved', False)
                
                total_pnl += daily_pnl
                total_trades += daily_trade_count
                if target_achieved:
                    target_days += 1
                if daily_trade_count > 0:
                    successful_days += 1
                
                # Track for analysis
                daily_pnls.append(daily_pnl)
                daily_trades.append(daily_trade_count)
                daily_win_rates.append(daily_win_rate)
                
                # Daily summary
                status = "🎯 TARGET MET" if target_achieved else "📉 Below Target"
                print(f"   💰 P&L: ${daily_pnl:.2f}")
                print(f"   📊 Trades: {daily_trade_count}")
                print(f"   📈 Win Rate: {daily_win_rate:.1%}")
                print(f"   {status}")
                
            except Exception as e:
                print(f"   ❌ Error on {date_str}: {e}")
                daily_result = {
                    'date': date_str,
                    'total_pnl': 0.0,
                    'trades': 0,
                    'win_rate': 0.0,
                    'target_achieved': False,
                    'strategy': 'Phase4D_BullPutSpreads'
                }
                monthly_results.append(daily_result)
                daily_pnls.append(0.0)
                daily_trades.append(0)
                daily_win_rates.append(0.0)
        
        # Calculate comprehensive metrics
        avg_daily_pnl = total_pnl / len(trading_days) if trading_days else 0
        avg_daily_trades = total_trades / len(trading_days) if trading_days else 0
        target_achievement_rate = target_days / len(trading_days) if trading_days else 0
        overall_win_rate = np.mean(daily_win_rates) if daily_win_rates else 0
        
        # Risk metrics
        daily_pnl_std = np.std(daily_pnls) if daily_pnls else 0
        sharpe_ratio = (avg_daily_pnl / daily_pnl_std) * np.sqrt(252) if daily_pnl_std > 0 else 0
        max_daily_loss = min(daily_pnls) if daily_pnls else 0
        max_daily_gain = max(daily_pnls) if daily_pnls else 0
        
        # Profit consistency
        profitable_days = len([pnl for pnl in daily_pnls if pnl > 0])
        profitability_rate = profitable_days / len(trading_days) if trading_days else 0
        
        # Generate final report
        monthly_summary = {
            # Core Performance
            'year': year,
            'month': month,
            'trading_days': len(trading_days),
            'total_pnl': total_pnl,
            'avg_daily_pnl': avg_daily_pnl,
            
            # Target Achievement
            'target_days': target_days,
            'target_achievement_rate': target_achievement_rate,
            'profitable_days': profitable_days,
            'profitability_rate': profitability_rate,
            
            # Trading Activity
            'total_trades': total_trades,
            'avg_daily_trades': avg_daily_trades,
            'successful_days': successful_days,
            'overall_win_rate': overall_win_rate,
            
            # Risk Metrics
            'daily_pnl_std': daily_pnl_std,
            'sharpe_ratio': sharpe_ratio,
            'max_daily_loss': max_daily_loss,
            'max_daily_gain': max_daily_gain,
            
            # Strategy Details
            'strategy': 'Phase4D_BullPutSpreads',
            'daily_results': monthly_results,
            'daily_pnls': daily_pnls,
            'daily_trades': daily_trades
        }
        
        return monthly_summary
    
    def print_comprehensive_report(self, results: Dict):
        """Print detailed monthly performance report"""
        print(f"\n" + "=" * 80)
        print(f"🎯 PHASE 4D BULL PUT SPREADS - MONTHLY PERFORMANCE REPORT")
        print(f"📅 Period: {results['year']}-{results['month']:02d}")
        print(f"=" * 80)
        
        # Core Performance
        print(f"\n💰 CORE PERFORMANCE:")
        print(f"   📅 Trading Days: {results['trading_days']}")
        print(f"   💰 Total P&L: ${results['total_pnl']:.2f}")
        print(f"   📊 Avg Daily P&L: ${results['avg_daily_pnl']:.2f}")
        monthly_return = (results['total_pnl'] / 25000) * 100  # $25K account
        print(f"   📈 Monthly Return: {monthly_return:.1f}%")
        
        # Target Achievement Analysis
        print(f"\n🎯 TARGET ACHIEVEMENT ($300-500 DAILY):")
        print(f"   🎯 Target Days: {results['target_days']}/{results['trading_days']}")
        print(f"   📊 Achievement Rate: {results['target_achievement_rate']:.1%}")
        print(f"   💚 Profitable Days: {results['profitable_days']}/{results['trading_days']}")
        print(f"   📈 Profitability Rate: {results['profitability_rate']:.1%}")
        
        # Trading Activity
        print(f"\n📊 TRADING ACTIVITY:")
        print(f"   🔢 Total Trades: {results['total_trades']}")
        print(f"   📊 Avg Daily Trades: {results['avg_daily_trades']:.1f}")
        print(f"   ✅ Active Trading Days: {results['successful_days']}/{results['trading_days']}")
        print(f"   📈 Overall Win Rate: {results['overall_win_rate']:.1%}")
        
        # Risk Analysis
        print(f"\n⚠️ RISK ANALYSIS:")
        print(f"   📊 Daily P&L Std Dev: ${results['daily_pnl_std']:.2f}")
        print(f"   📈 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   💔 Max Daily Loss: ${results['max_daily_loss']:.2f}")
        print(f"   💚 Max Daily Gain: ${results['max_daily_gain']:.2f}")
        
        # Strategy Assessment
        print(f"\n🔍 STRATEGY ASSESSMENT:")
        
        if results['target_achievement_rate'] >= 0.70:
            assessment = "🎯 EXCELLENT - Consistently hitting targets"
        elif results['target_achievement_rate'] >= 0.50:
            assessment = "✅ GOOD - Solid target achievement"
        elif results['target_achievement_rate'] >= 0.30:
            assessment = "⚠️ NEEDS IMPROVEMENT - Below target expectations"
        else:
            assessment = "❌ POOR - Significant optimization needed"
        
        print(f"   {assessment}")
        
        # Recommendations
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        
        if results['avg_daily_trades'] < 15:
            print(f"   🔄 Increase trade frequency (current: {results['avg_daily_trades']:.1f}, target: 15-25)")
        
        if results['overall_win_rate'] < 0.70:
            print(f"   📊 Improve trade selection (current: {results['overall_win_rate']:.1%}, target: 70%+)")
        
        if results['target_achievement_rate'] < 0.60:
            print(f"   💰 Consider increasing position size for daily targets")
        
        if results['sharpe_ratio'] < 1.5:
            print(f"   ⚠️ Improve risk-adjusted returns (current: {results['sharpe_ratio']:.2f})")
        
        # Final Verdict
        if (results['target_achievement_rate'] >= 0.60 and 
            results['profitability_rate'] >= 0.70 and
            results['total_pnl'] >= 6000):
            verdict = "🎯 STRATEGY READY FOR LIVE TRADING"
        elif results['total_pnl'] > 0 and results['profitability_rate'] >= 0.60:
            verdict = "✅ PROMISING - Minor optimizations needed"
        else:
            verdict = "❌ NEEDS MAJOR IMPROVEMENTS"
        
        print(f"\n🏆 FINAL VERDICT: {verdict}")
        print(f"=" * 80)


if __name__ == "__main__":
    # Run March 2024 comprehensive test
    runner = Phase4DMonthlyRunner()
    results = runner.run_monthly_backtest(2024, 3)
    runner.print_comprehensive_report(results) 