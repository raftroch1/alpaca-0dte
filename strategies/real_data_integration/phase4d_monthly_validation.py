#!/usr/bin/env python3
"""
🎯 PHASE 4D: MONTHLY VALIDATION RUNNER
=====================================

Tests the fixed Phase 4D strategy across multiple trading days to validate
profitability and identify any remaining issues.

Runs comprehensive testing to prove the strategy works with real market data.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import argparse
import logging

# Import our fixed Phase 4D strategy
from phase4d_standalone_profitable import Phase4DStandaloneProfitable

class Phase4DMonthlyValidator:
    """Monthly validation for Phase 4D fixed strategy"""
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        self.cache_dir = cache_dir
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(f'{__name__}.Validator')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def get_available_dates(self) -> list:
        """Get list of available trading dates from cached data"""
        spy_dir = os.path.join(self.cache_dir, "spy_bars")
        
        if not os.path.exists(spy_dir):
            self.logger.error(f"❌ SPY data directory not found: {spy_dir}")
            return []
        
        dates = []
        for filename in os.listdir(spy_dir):
            if filename.startswith("spy_bars_") and filename.endswith(".pkl.gz"):
                date_str = filename.replace("spy_bars_", "").replace(".pkl.gz", "")
                if len(date_str) == 8:  # YYYYMMDD format
                    dates.append(date_str)
        
        dates.sort()
        self.logger.info(f"✅ Found {len(dates)} available trading dates")
        return dates
    
    def run_monthly_validation(self, year: int = 2024, month: int = 3, max_days: int = 10) -> dict:
        """Run validation across multiple days in a month"""
        self.logger.info(f"🎯 Running monthly validation for {year}-{month:02d}")
        
        # Get available dates
        all_dates = self.get_available_dates()
        
        # Filter for target month
        month_prefix = f"{year}{month:02d}"
        month_dates = [d for d in all_dates if d.startswith(month_prefix)][:max_days]
        
        if not month_dates:
            self.logger.error(f"❌ No data found for {year}-{month:02d}")
            return {'error': f'No data for {year}-{month:02d}'}
        
        self.logger.info(f"📊 Testing {len(month_dates)} days in {year}-{month:02d}")
        
        # Initialize strategy
        strategy = Phase4DStandaloneProfitable(cache_dir=self.cache_dir)
        
        # Run backtests for each day
        results = []
        total_pnl = 0.0
        total_trades = 0
        profitable_days = 0
        
        for date_str in month_dates:
            self.logger.info(f"📈 Testing {date_str}")
            
            try:
                result = strategy.run_daily_backtest(date_str)
                
                if 'error' not in result:
                    results.append(result)
                    total_pnl += result['pnl']
                    total_trades += result['trades']
                    
                    if result['pnl'] > 0:
                        profitable_days += 1
                    
                    self.logger.info(f"   {date_str}: {result['trades']} trades, ${result['pnl']:.2f} P&L")
                else:
                    self.logger.warning(f"   {date_str}: Error - {result['error']}")
                    
            except Exception as e:
                self.logger.error(f"   {date_str}: Exception - {e}")
        
        # Calculate summary metrics
        if results:
            avg_daily_pnl = total_pnl / len(results)
            avg_trades_per_day = total_trades / len(results)
            profitable_day_rate = (profitable_days / len(results)) * 100
            
            # Calculate win rate across all trades
            all_trades = []
            for result in results:
                if 'trade_details' in result:
                    all_trades.extend(result['trade_details'])
            
            winning_trades = len([t for t in all_trades if t['total_pnl'] > 0])
            overall_win_rate = (winning_trades / len(all_trades) * 100) if all_trades else 0
            
            summary = {
                'month': f"{year}-{month:02d}",
                'days_tested': len(results),
                'total_pnl': total_pnl,
                'avg_daily_pnl': avg_daily_pnl,
                'total_trades': total_trades,
                'avg_trades_per_day': avg_trades_per_day,
                'profitable_days': profitable_days,
                'profitable_day_rate': profitable_day_rate,
                'overall_win_rate': overall_win_rate,
                'daily_results': results
            }
            
            self.logger.info(f"✅ Monthly validation complete:")
            self.logger.info(f"   Days tested: {len(results)}")
            self.logger.info(f"   Total P&L: ${total_pnl:.2f}")
            self.logger.info(f"   Avg daily P&L: ${avg_daily_pnl:.2f}")
            self.logger.info(f"   Total trades: {total_trades}")
            self.logger.info(f"   Profitable days: {profitable_days}/{len(results)} ({profitable_day_rate:.1f}%)")
            self.logger.info(f"   Overall win rate: {overall_win_rate:.1f}%")
            
            return summary
        else:
            return {'error': 'No successful backtests'}
    
    def diagnose_no_trades_issue(self, date_str: str) -> dict:
        """Diagnose why no trades are being executed"""
        self.logger.info(f"🔍 Diagnosing no-trades issue for {date_str}")
        
        strategy = Phase4DStandaloneProfitable(cache_dir=self.cache_dir)
        
        # Load SPY data
        spy_bars = strategy.load_cached_spy_data(date_str)
        if spy_bars is None:
            return {'issue': 'No SPY data'}
        
        # Check signal generation
        signals = strategy.generate_spread_signals(spy_bars)
        if not signals:
            return {'issue': 'No signals generated', 'spy_bars': len(spy_bars)}
        
        # Check spread finding
        signal = signals[0]
        spread = strategy.find_optimal_bull_put_spread(signal['spy_price'], date_str)
        if spread is None:
            return {'issue': 'No valid spreads found', 'signals': len(signals), 'spy_price': signal['spy_price']}
        
        return {
            'issue': 'Should have trades',
            'signals': len(signals),
            'spread_found': True,
            'spy_price': signal['spy_price'],
            'spread_credit': spread['net_credit']
        }

def main():
    parser = argparse.ArgumentParser(description='Phase 4D Monthly Validation')
    parser.add_argument('--year', type=int, default=2024, help='Year to test')
    parser.add_argument('--month', type=int, default=3, help='Month to test')
    parser.add_argument('--max-days', type=int, default=10, help='Max days to test')
    parser.add_argument('--diagnose', help='Diagnose specific date (YYYYMMDD)')
    parser.add_argument('--cache-dir', default='../../thetadata/cached_data', help='Cache directory')
    args = parser.parse_args()
    
    validator = Phase4DMonthlyValidator(cache_dir=args.cache_dir)
    
    if args.diagnose:
        result = validator.diagnose_no_trades_issue(args.diagnose)
        print(f"\n🔍 DIAGNOSIS for {args.diagnose}:")
        for key, value in result.items():
            print(f"   {key}: {value}")
    else:
        result = validator.run_monthly_validation(args.year, args.month, args.max_days)
        
        if 'error' not in result:
            print(f"\n🎯 PHASE 4D MONTHLY VALIDATION RESULTS:")
            print(f"   Month: {result['month']}")
            print(f"   Days Tested: {result['days_tested']}")
            print(f"   Total P&L: ${result['total_pnl']:.2f}")
            print(f"   Average Daily P&L: ${result['avg_daily_pnl']:.2f}")
            print(f"   Total Trades: {result['total_trades']}")
            print(f"   Average Trades/Day: {result['avg_trades_per_day']:.1f}")
            print(f"   Profitable Days: {result['profitable_days']}/{result['days_tested']} ({result['profitable_day_rate']:.1f}%)")
            print(f"   Overall Win Rate: {result['overall_win_rate']:.1f}%")
            
            if result['total_pnl'] > 0:
                print(f"✅ MONTHLY PROFIT: ${result['total_pnl']:.2f}")
            else:
                print(f"❌ Monthly loss: ${result['total_pnl']:.2f}")
                
            if result['total_trades'] == 0:
                print(f"⚠️  NO TRADES EXECUTED - May need parameter adjustment")
        else:
            print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    main()