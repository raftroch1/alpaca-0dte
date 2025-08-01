#!/usr/bin/env python3
"""
📊 PROPER UNIFIED STRATEGY BACKTEST - BULL PUT SPREADS
=====================================================

Professional backtest framework for our CORRECT unified dual strategy system using BULL PUT SPREADS.
EXACT same logic as previous unified system but with LIMITED RISK execution.

- Primary: Minimal scale balanced (2x volume, same signal quality) → BULL PUT SPREADS
- Counter: Focused counter strategy for filtered days → BULL PUT SPREADS
- Real SPY minute data from ThetaData cache
- Real Alpaca historical option prices  
- Realistic trading costs (commission, bid/ask, slippage)
- Professional performance metrics and risk analysis

🎯 GOAL: Validate that our PROPER dual BULL PUT SPREAD system:
- Maintains excellent performance from unified system
- Increases execution rate from 40% to 60%+ via counter strategy  
- Limited risk vs unlimited naked put risk
- Superior risk-adjusted returns with controlled max loss

Author: Strategy Development Framework
Date: 2025-08-01
Version: Proper Unified Bull Put Spreads Backtest v1.0
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import pickle
import gzip
from typing import Dict, List, Optional, Tuple
import argparse

# Import our proper unified bull put spread strategy
from proper_unified_system import ProperUnifiedSystemBullPutSpreads

class ProperUnifiedBacktest:
    """
    Comprehensive backtest framework for proper unified dual strategy
    Following the same professional standards as our balanced strategy backtest
    """
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        self.cache_dir = cache_dir
        self.setup_logging()
        
        # Initialize proper unified bull put spread strategy
        self.strategy = ProperUnifiedSystemBullPutSpreads(cache_dir=cache_dir)
        
        # Results tracking (following balanced strategy pattern)
        self.results = {
            'trades': [],
            'daily_results': [],
            'monthly_summary': {},
            'performance_metrics': {},
            'strategy_breakdown': {
                'primary_trades': 0,
                'counter_trades': 0,
                'total_trades': 0,
                'execution_rate': 0,
                'primary_pnl': 0,
                'counter_pnl': 0
            }
        }
        
        # Risk metrics tracking
        self.daily_pnl_series = []
        self.drawdown_series = []
        self.cumulative_pnl = 0
        self.peak_value = 0
        
    def setup_logging(self):
        """Setup logging following balanced strategy pattern"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """Get trading dates from cache directory (following existing pattern)"""
        trading_dates = []
        
        try:
            spy_bars_dir = os.path.join(self.cache_dir, "spy_bars")
            if not os.path.exists(spy_bars_dir):
                self.logger.error(f"❌ SPY bars directory not found: {spy_bars_dir}")
                return trading_dates
            
            # Get all available dates from cache
            for filename in sorted(os.listdir(spy_bars_dir)):
                if filename.startswith("spy_bars_") and filename.endswith(".pkl.gz"):
                    date_str = filename.replace("spy_bars_", "").replace(".pkl.gz", "")
                    
                    # Convert date formats for comparison (YYYYMMDD -> YYYY-MM-DD)
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    
                    # Check if date is in range
                    if start_date <= formatted_date <= end_date:
                        trading_dates.append(date_str)
            
            self.logger.info(f"📅 Found {len(trading_dates)} trading dates from {start_date} to {end_date}")
            return trading_dates
            
        except Exception as e:
            self.logger.error(f"❌ Error getting trading dates: {e}")
            return trading_dates
    
    def run_comprehensive_backtest(self, start_date: str, end_date: str) -> Dict:
        """
        Run comprehensive backtest following balanced strategy methodology
        """
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🚀 PROPER UNIFIED STRATEGY COMPREHENSIVE BACKTEST")
        self.logger.info(f"📊 Testing Period: {start_date} to {end_date}")
        self.logger.info(f"📈 Primary: Minimal Scale Balanced (2x volume, same quality)")
        self.logger.info(f"🛡️ Counter: Focused Counter Strategy")
        self.logger.info(f"🎯 Goal: Higher execution rate with maintained signal quality")
        self.logger.info(f"{'='*80}")
        
        # Get trading dates
        trading_dates = self.get_trading_dates(start_date, end_date)
        
        if not trading_dates:
            self.logger.error("❌ No trading dates found")
            return {'error': 'No trading dates available'}
        
        # Initialize tracking
        total_pnl = 0
        winning_trades = 0
        losing_trades = 0
        primary_count = 0
        counter_count = 0
        primary_pnl = 0
        counter_pnl = 0
        monthly_data = {}
        
        # Run backtest day by day
        for i, date_str in enumerate(trading_dates, 1):
            self.logger.info(f"\n📅 Day {i}/{len(trading_dates)}: {date_str}")
            
            # Run proper unified strategy for this day
            result = self.strategy.run_unified_strategy(date_str)
            
            # Process result
            daily_result = {
                'date': date_str,
                'trade_executed': False,
                'strategy_used': None,
                'pnl': 0,
                'spy_close': result.get('spy_close', 0)
            }
            
            if result.get('success'):
                # Successful trade
                trade = result['trade']
                strategy_used = result['strategy_used']
                
                daily_result.update({
                    'trade_executed': True,
                    'strategy_used': strategy_used,
                    'pnl': trade['final_pnl'],
                    'trade_details': trade
                })
                
                # Update counters
                total_pnl += trade['final_pnl']
                
                if trade['final_pnl'] > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
                
                if strategy_used == 'primary':
                    primary_count += 1
                    primary_pnl += trade['final_pnl']
                else:
                    counter_count += 1
                    counter_pnl += trade['final_pnl']
                
                # Store trade
                self.results['trades'].append(trade)
                
                self.logger.info(f"   ✅ {strategy_used.upper()}: ${trade['final_pnl']:.2f} ({trade['strategy']})")
                self.logger.info(f"   💰 Running Total: ${total_pnl:.2f}")
                
            else:
                # No trade executed
                reason = result.get('reason', 'Unknown')
                self.logger.info(f"   ❌ No trade: {reason}")
            
            # Store daily result
            self.results['daily_results'].append(daily_result)
            
            # Update risk tracking
            self.daily_pnl_series.append(daily_result['pnl'])
            self.cumulative_pnl += daily_result['pnl']
            
            if self.cumulative_pnl > self.peak_value:
                self.peak_value = self.cumulative_pnl
            
            current_drawdown = self.peak_value - self.cumulative_pnl
            self.drawdown_series.append(current_drawdown)
            
            # Monthly tracking
            month_key = date_str[:6]  # YYYYMM
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'trades': 0,
                    'primary_trades': 0,
                    'counter_trades': 0,
                    'pnl': 0,
                    'primary_pnl': 0,
                    'counter_pnl': 0,
                    'winning_trades': 0,
                    'days': 0
                }
            
            monthly_data[month_key]['days'] += 1
            if daily_result['trade_executed']:
                monthly_data[month_key]['trades'] += 1
                monthly_data[month_key]['pnl'] += daily_result['pnl']
                
                if daily_result['strategy_used'] == 'primary':
                    monthly_data[month_key]['primary_trades'] += 1
                    monthly_data[month_key]['primary_pnl'] += daily_result['pnl']
                else:
                    monthly_data[month_key]['counter_trades'] += 1
                    monthly_data[month_key]['counter_pnl'] += daily_result['pnl']
                
                if daily_result['pnl'] > 0:
                    monthly_data[month_key]['winning_trades'] += 1
        
        # Calculate final metrics
        total_trades = primary_count + counter_count
        execution_rate = (total_trades / len(trading_dates)) * 100 if trading_dates else 0
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Update results
        self.results['strategy_breakdown'].update({
            'primary_trades': primary_count,
            'counter_trades': counter_count,
            'total_trades': total_trades,
            'execution_rate': execution_rate,
            'primary_pnl': primary_pnl,
            'counter_pnl': counter_pnl
        })
        
        self.results['monthly_summary'] = monthly_data
        
        # Calculate comprehensive metrics
        self.calculate_performance_metrics(total_pnl, win_rate, execution_rate, len(trading_dates))
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        return self.results
    
    def calculate_performance_metrics(self, total_pnl: float, win_rate: float, execution_rate: float, total_days: int):
        """Calculate comprehensive performance metrics following balanced strategy pattern"""
        
        # Basic metrics
        total_trades = self.results['strategy_breakdown']['total_trades']
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        daily_returns = np.array(self.daily_pnl_series)
        volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
        
        # Sharpe ratio (assuming 1% risk-free rate)
        risk_free_rate = 0.01 / 252  # Daily risk-free rate
        excess_returns = daily_returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        # Drawdown metrics
        max_drawdown = max(self.drawdown_series) if self.drawdown_series else 0
        
        # Calmar ratio (annualized return / max drawdown)
        annualized_return = (total_pnl / total_days) * 252 if total_days > 0 else 0
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Profit factor
        winning_pnl = sum(pnl for pnl in daily_returns if pnl > 0)
        losing_pnl = abs(sum(pnl for pnl in daily_returns if pnl < 0))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
        
        self.results['performance_metrics'] = {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'execution_rate': execution_rate,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'profit_factor': profit_factor,
            'annualized_return': annualized_return
        }
    
    def generate_comprehensive_report(self):
        """Generate comprehensive backtest report following balanced strategy pattern"""
        
        metrics = self.results['performance_metrics']
        breakdown = self.results['strategy_breakdown']
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"📈 PROPER UNIFIED STRATEGY BACKTEST RESULTS")
        self.logger.info(f"{'='*80}")
        
        # Overall Performance
        self.logger.info(f"\n🎯 OVERALL PERFORMANCE:")
        self.logger.info(f"   💰 Total P&L: ${metrics['total_pnl']:.2f}")
        self.logger.info(f"   📊 Total Trades: {metrics['total_trades']}")
        self.logger.info(f"   📈 Execution Rate: {metrics['execution_rate']:.1f}%")
        self.logger.info(f"   🏆 Win Rate: {metrics['win_rate']:.1f}%")
        self.logger.info(f"   📊 Average Trade: ${metrics['avg_trade']:.2f}")
        
        # Strategy Breakdown
        self.logger.info(f"\n🔄 STRATEGY BREAKDOWN:")
        self.logger.info(f"   📈 Primary Trades: {breakdown['primary_trades']} ({breakdown['primary_trades']/metrics['total_trades']*100:.1f}%)")
        self.logger.info(f"   💰 Primary P&L: ${breakdown['primary_pnl']:.2f}")
        self.logger.info(f"   ��️ Counter Trades: {breakdown['counter_trades']} ({breakdown['counter_trades']/metrics['total_trades']*100:.1f}%)")
        self.logger.info(f"   💰 Counter P&L: ${breakdown['counter_pnl']:.2f}")
        
        # Risk Metrics
        self.logger.info(f"\n📊 RISK METRICS:")
        self.logger.info(f"   📈 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        self.logger.info(f"   📉 Max Drawdown: ${metrics['max_drawdown']:.2f}")
        self.logger.info(f"   📊 Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        self.logger.info(f"   🎯 Profit Factor: {metrics['profit_factor']:.2f}")
        self.logger.info(f"   📈 Annualized Return: {metrics['annualized_return']:.1f}%")
        
        # Monthly Performance
        self.generate_monthly_report()
        
        # Comparison vs Original Balanced
        self.generate_comparison_analysis()
        
        # Save results
        self.save_backtest_results()
    
    def generate_monthly_report(self):
        """Generate monthly performance breakdown"""
        
        self.logger.info(f"\n📅 MONTHLY PERFORMANCE:")
        
        for month, data in sorted(self.results['monthly_summary'].items()):
            month_str = f"{month[:4]}-{month[4:]}"
            execution_rate = (data['trades'] / data['days']) * 100
            win_rate = (data['winning_trades'] / data['trades']) * 100 if data['trades'] > 0 else 0
            
            self.logger.info(f"   📊 {month_str}: ${data['pnl']:>6.0f} | "
                           f"{data['trades']:>2} trades ({execution_rate:>4.1f}%) | "
                           f"{win_rate:>4.1f}% wins | "
                           f"P:{data['primary_trades']} (${data['primary_pnl']:>4.0f}) "
                           f"C:{data['counter_trades']} (${data['counter_pnl']:>4.0f})")
    
    def generate_comparison_analysis(self):
        """Generate comparison vs original balanced strategy"""
        self.logger.info(f"\n🔬 COMPARISON VS ORIGINAL BALANCED:")
        self.logger.info(f"   📊 Original Balanced: 46% execution, 82.5% wins, +$1,770")
        self.logger.info(f"   📈 Proper Unified: {self.results['performance_metrics']['execution_rate']:.1f}% execution, {self.results['performance_metrics']['win_rate']:.1f}% wins, ${self.results['performance_metrics']['total_pnl']:.0f}")
        
        # Calculate improvements
        exec_improvement = self.results['performance_metrics']['execution_rate'] - 46.0
        self.logger.info(f"   🎯 Execution Rate: +{exec_improvement:.1f}% improvement")
        
        # Expected performance based on 2x scaling
        expected_pnl = 1770 * 2 * (self.results['performance_metrics']['execution_rate'] / 100) / 0.46
        self.logger.info(f"   💰 Expected P&L (2x scaling): ~${expected_pnl:.0f}")
    
    def save_backtest_results(self):
        """Save comprehensive backtest results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save detailed results
            results_file = f"proper_unified_backtest_{timestamp}.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(self.results, f)
            
            # Save trades CSV
            if self.results['trades']:
                trades_df = pd.DataFrame(self.results['trades'])
                trades_csv = f"proper_unified_trades_{timestamp}.csv"
                trades_df.to_csv(trades_csv, index=False)
            
            # Save daily results CSV
            if self.results['daily_results']:
                daily_df = pd.DataFrame(self.results['daily_results'])
                daily_csv = f"proper_unified_daily_{timestamp}.csv"
                daily_df.to_csv(daily_csv, index=False)
            
            self.logger.info(f"\n💾 RESULTS SAVED:")
            self.logger.info(f"   📊 Detailed: {results_file}")
            if self.results['trades']:
                self.logger.info(f"   📈 Trades: {trades_csv}")
            if self.results['daily_results']:
                self.logger.info(f"   📅 Daily: {daily_csv}")
                
        except Exception as e:
            self.logger.error(f"❌ Error saving results: {e}")

def main():
    """Main execution for proper unified strategy backtest"""
    parser = argparse.ArgumentParser(description='Proper Unified Strategy Comprehensive Backtest')
    parser.add_argument('--start-date', default='20240301', help='Start date YYYYMMDD')
    parser.add_argument('--end-date', default='20240831', help='End date YYYYMMDD')
    parser.add_argument('--cache-dir', default='../../thetadata/cached_data', help='Cache directory')
    
    args = parser.parse_args()
    
    print(f"🚀 Proper Unified Strategy Comprehensive Backtest")
    print(f"📊 Period: {args.start_date} to {args.end_date}")
    print(f"📈 Primary: Minimal scale balanced (2x volume, same quality)")
    print(f"🛡️ Counter: Focused counter strategy")
    print(f"🎯 Goal: Higher execution rate with maintained signal quality")
    print(f"📁 Cache: {args.cache_dir}")
    
    # Run backtest
    backtest = ProperUnifiedBacktest(cache_dir=args.cache_dir)
    results = backtest.run_comprehensive_backtest(args.start_date, args.end_date)
    
    if 'error' in results:
        print(f"❌ Backtest failed: {results['error']}")
    else:
        metrics = results['performance_metrics']
        breakdown = results['strategy_breakdown']
        
        print(f"\n✅ PROPER UNIFIED BACKTEST COMPLETE!")
        print(f"💰 Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"📊 Execution Rate: {metrics['execution_rate']:.1f}%")
        print(f"🏆 Win Rate: {metrics['win_rate']:.1f}%")
        print(f"📈 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"🔄 Primary/Counter: {breakdown['primary_trades']}/{breakdown['counter_trades']}")
        print(f"💰 Primary/Counter P&L: ${breakdown['primary_pnl']:.0f}/${breakdown['counter_pnl']:.0f}")

if __name__ == "__main__":
    main()
