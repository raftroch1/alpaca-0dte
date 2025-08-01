#!/usr/bin/env python3
"""
🎯 PHASE 4D: BALANCED STRATEGY BACKTEST
=======================================

Comprehensive backtesting for the balanced Phase 4D strategy.
Tests the strategy that actually executes trades while maintaining disaster protection.

📊 COMPARISON TARGETS:
- Optimized: 0 trades, $0 return (too conservative)
- Balanced: Target ~50% execution rate with positive returns

Author: Strategy Development Framework
Date: 2025-01-30
Version: Phase 4D Balanced Backtest v1.0
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import pickle
import gzip
import argparse
from typing import Dict, List, Optional, Tuple

# Import the balanced strategy
from phase4d_balanced_strategy import Phase4DBalancedStrategy

class Phase4DBalancedBacktest:
    """Comprehensive backtesting for balanced Phase 4D strategy"""
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        self.cache_dir = cache_dir
        self.setup_logging()
        
        # Initialize balanced strategy
        self.strategy = Phase4DBalancedStrategy(cache_dir=cache_dir)
        
        # Results tracking
        self.results = {
            'trades': [],
            'daily_pnl': [],
            'equity_curve': [],
            'filtered_days': [],
            'metrics': {},
            'balanced_analysis': {}
        }
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """Get available trading dates"""
        available_dates = []
        
        spy_dir = os.path.join(self.cache_dir, "spy_bars")
        if os.path.exists(spy_dir):
            for file in os.listdir(spy_dir):
                if file.endswith('.pkl.gz') and 'spy_bars_' in file:
                    date_str = file.replace('spy_bars_', '').replace('.pkl.gz', '')
                    if start_date <= date_str <= end_date:
                        available_dates.append(date_str)
        
        available_dates.sort()
        self.logger.info(f"📅 Found {len(available_dates)} trading dates: {start_date} to {end_date}")
        return available_dates
    
    def apply_realistic_costs(self, gross_pnl: float, contracts: int) -> float:
        """Apply realistic trading costs"""
        commission = contracts * 0.65  # $0.65 per contract
        bid_ask_cost = contracts * 100 * 0.03  # $3 per contract bid/ask (tighter)
        slippage = abs(gross_pnl) * 0.005  # 0.5% slippage (improved)
        
        total_costs = commission + bid_ask_cost + slippage
        return gross_pnl - total_costs
    
    def run_balanced_backtest(self, start_date: str, end_date: str) -> Dict:
        """
        Run comprehensive balanced backtest
        """
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🎯 PHASE 4D BALANCED COMPREHENSIVE BACKTEST")
        self.logger.info(f"📅 Period: {start_date} to {end_date}")
        self.logger.info(f"🔧 Testing Balanced Trade Execution")
        self.logger.info(f"{'='*80}")
        
        # Get trading dates
        trading_dates = self.get_trading_dates(start_date, end_date)
        
        if not trading_dates:
            return {'error': 'No trading dates available'}
        
        # Reset results
        self.results = {
            'trades': [],
            'daily_pnl': [],
            'equity_curve': [],
            'filtered_days': [],
            'metrics': {},
            'balanced_analysis': {}
        }
        
        # Track balanced execution effectiveness
        execution_stats = {
            'disaster_filtered': 0,
            'premium_filtered': 0,
            'successful_trades': 0,
            'total_days': len(trading_dates),
            'itm_trades': 0,
            'otm_trades': 0
        }
        
        cumulative_pnl = 0
        
        for i, date_str in enumerate(trading_dates, 1):
            self.logger.info(f"📊 Processing {date_str} ({i}/{len(trading_dates)})")
            
            # Run balanced strategy
            day_result = self.strategy.run_single_day(date_str)
            
            if 'success' in day_result:
                trade_result = day_result['trade']
                
                # Apply realistic costs
                gross_pnl = trade_result['final_pnl']
                net_pnl = self.apply_realistic_costs(gross_pnl, trade_result['contracts'])
                
                # Enhanced trade result
                enhanced_trade = {
                    'date': date_str,
                    'strategy': trade_result['strategy'],
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'costs': gross_pnl - net_pnl,
                    'contracts': trade_result['contracts'],
                    'outcome': trade_result['outcome'],
                    'strike': trade_result['strike'],
                    'premium': trade_result['premium'],
                    'spy_close': trade_result['spy_close'],
                    'market_conditions': trade_result['market_conditions'],
                    'risk_reward_ratio': trade_result.get('risk_reward_ratio', 0),
                    'position_rationale': trade_result.get('position_rationale', 'N/A'),
                    'strike_type': 'ITM' if trade_result['strike'] > trade_result['spy_close'] else 'OTM'
                }
                
                self.results['trades'].append(enhanced_trade)
                cumulative_pnl += net_pnl
                
                self.results['daily_pnl'].append(net_pnl)
                self.results['equity_curve'].append(cumulative_pnl)
                
                execution_stats['successful_trades'] += 1
                
                # Track ITM vs OTM
                if enhanced_trade['strike_type'] == 'ITM':
                    execution_stats['itm_trades'] += 1
                else:
                    execution_stats['otm_trades'] += 1
                
                self.logger.info(f"   💰 P&L: ${net_pnl:.2f} | Total: ${cumulative_pnl:.2f}")
                
            elif 'no_trade' in day_result:
                self.results['daily_pnl'].append(0)
                self.results['equity_curve'].append(cumulative_pnl)
                
                # Record filtered day
                filtered_info = {
                    'date': date_str,
                    'spy_close': day_result.get('spy_close', 0),
                    'reason': 'Balanced filters active'
                }
                self.results['filtered_days'].append(filtered_info)
                
                self.logger.info(f"   📊 No trade: Balanced filters active")
                
            else:
                # Error case
                self.results['daily_pnl'].append(0)
                self.results['equity_curve'].append(cumulative_pnl)
                
                if 'error' in day_result:
                    self.logger.warning(f"   ❌ Error: {day_result['error']}")
        
        # Calculate comprehensive metrics
        self.results['metrics'] = self.calculate_balanced_metrics()
        self.results['balanced_analysis'] = execution_stats
        
        # Generate balanced analysis report
        self.generate_balanced_report()
        
        return {
            'success': True,
            'results': self.results,
            'summary': {
                'trading_days': len(trading_dates),
                'successful_trades': execution_stats['successful_trades'],
                'filtered_days': len(self.results['filtered_days']),
                'execution_rate': execution_stats['successful_trades'] / len(trading_dates) * 100,
                'total_return': self.results['metrics'].get('total_return', 0),
                'win_rate': self.results['metrics'].get('win_rate', 0),
                'sharpe_ratio': self.results['metrics'].get('sharpe_ratio', 0)
            }
        }
    
    def calculate_balanced_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not self.results['daily_pnl']:
            return {'error': 'No trading data available'}
        
        # Convert to pandas series
        pnl_series = pd.Series(self.results['daily_pnl'])
        
        # Basic metrics
        total_return = pnl_series.sum()
        total_trades = len([t for t in self.results['trades'] if t['net_pnl'] != 0])
        winning_trades = len([t for t in self.results['trades'] if t['net_pnl'] > 0])
        losing_trades = len([t for t in self.results['trades'] if t['net_pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Risk metrics
        if len(pnl_series) > 1 and pnl_series.std() > 0:
            returns_series = pnl_series.pct_change().dropna()
            if len(returns_series) > 0 and returns_series.std() > 0:
                sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            max_drawdown = (pnl_series.cumsum().expanding().max() - pnl_series.cumsum()).max()
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Profit factor
        gross_profit = sum([t['net_pnl'] for t in self.results['trades'] if t['net_pnl'] > 0])
        gross_loss = abs(sum([t['net_pnl'] for t in self.results['trades'] if t['net_pnl'] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average metrics
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def generate_balanced_report(self):
        """Generate detailed balanced analysis report"""
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"📈 BALANCED STRATEGY ANALYSIS REPORT")
        self.logger.info(f"{'='*80}")
        
        # Overall performance
        metrics = self.results['metrics']
        self.logger.info(f"💰 Total Return: ${metrics.get('total_return', 0):.2f}")
        self.logger.info(f"📊 Win Rate: {metrics.get('win_rate', 0):.1f}%")
        self.logger.info(f"📈 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        self.logger.info(f"🎯 Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        # Execution analysis
        total_trades = metrics.get('total_trades', 0)
        total_days = self.results['balanced_analysis']['total_days']
        execution_rate = (total_trades / total_days * 100) if total_days > 0 else 0
        
        self.logger.info(f"\n📋 BALANCED EXECUTION ANALYSIS:")
        self.logger.info(f"   Total Days: {total_days}")
        self.logger.info(f"   Trades Executed: {total_trades}")
        self.logger.info(f"   Execution Rate: {execution_rate:.1f}%")
        self.logger.info(f"   Days Filtered: {len(self.results['filtered_days'])}")
        
        # ITM vs OTM analysis
        itm_trades = self.results['balanced_analysis']['itm_trades']
        otm_trades = self.results['balanced_analysis']['otm_trades']
        
        if total_trades > 0:
            self.logger.info(f"\n🎯 STRIKE SELECTION ANALYSIS:")
            self.logger.info(f"   ITM Trades: {itm_trades} ({itm_trades/total_trades*100:.1f}%)")
            self.logger.info(f"   OTM Trades: {otm_trades} ({otm_trades/total_trades*100:.1f}%)")
        
        # Strategy comparison
        self.logger.info(f"\n📊 STRATEGY COMPARISON:")
        self.logger.info(f"   ✅ Balanced vs Original: ${metrics.get('total_return', 0):.2f} vs -$243.59")
        self.logger.info(f"   ✅ Balanced vs Optimized: {execution_rate:.1f}% vs 0% execution rate")
        
        # Save results
        self.save_balanced_results()
    
    def save_balanced_results(self):
        """Save detailed balanced results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save pickle file
            results_file = f"balanced_results_{timestamp}.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(self.results, f)
            
            # Save CSV file
            if self.results['trades']:
                trades_df = pd.DataFrame(self.results['trades'])
                csv_file = f"balanced_trades_{timestamp}.csv"
                trades_df.to_csv(csv_file, index=False)
                
                self.logger.info(f"💾 Balanced results saved: {results_file}")
                self.logger.info(f"📊 Trades CSV saved: {csv_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving results: {e}")

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Phase 4D Balanced Backtest')
    parser.add_argument('--start-date', default='20240301', help='Start date YYYYMMDD')
    parser.add_argument('--end-date', default='20240331', help='End date YYYYMMDD')
    parser.add_argument('--cache-dir', default='../../thetadata/cached_data', help='Cache directory')
    
    args = parser.parse_args()
    
    # Run balanced backtest
    backtester = Phase4DBalancedBacktest(cache_dir=args.cache_dir)
    results = backtester.run_balanced_backtest(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if 'success' in results:
        print(f"\n✅ Balanced backtest completed!")
        print(f"📊 Total Return: ${results['summary']['total_return']:.2f}")
        print(f"🎯 Win Rate: {results['summary']['win_rate']:.1f}%")
        print(f"📈 Execution Rate: {results['summary']['execution_rate']:.1f}%")
        print(f"📋 Trades: {results['summary']['successful_trades']} / {results['summary']['trading_days']} days")
    else:
        print(f"❌ Balanced backtest failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()