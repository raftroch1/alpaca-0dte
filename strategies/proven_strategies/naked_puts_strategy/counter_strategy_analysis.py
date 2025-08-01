#!/usr/bin/env python3
"""
🛡️ COUNTER STRATEGY ANALYSIS - ORIGINAL UNIFIED BACKTEST
=======================================================

Extract and analyze ONLY the counter strategy performance from our original 
unified backtest (naked puts + counter strategies). Scale for 25K account.

Goal: Understand capital efficiency and profitability of counter strategies in isolation.

Author: Strategy Development Framework
Date: 2025-08-01
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple

class CounterStrategyAnalysis:
    """
    Analyze counter strategy performance from unified backtest results
    """
    
    def __init__(self):
        self.setup_logging()
        
        # Results tracking
        self.counter_trades = []
        self.counter_daily_pnl = []
        self.analysis_results = {}
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_unified_backtest_data(self, trades_file: str, daily_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the unified backtest data from CSV files
        """
        try:
            trades_df = pd.read_csv(trades_file)
            daily_df = pd.read_csv(daily_file)
            
            self.logger.info(f"✅ Loaded {len(trades_df)} trades and {len(daily_df)} daily results")
            return trades_df, daily_df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {e}")
            return None, None
    
    def extract_counter_strategy_trades(self, trades_df: pd.DataFrame, daily_df: pd.DataFrame) -> Dict:
        """
        Extract ONLY counter strategy trades and performance
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🛡️ EXTRACTING COUNTER STRATEGY PERFORMANCE")
        self.logger.info(f"{'='*60}")
        
        # Identify counter strategy trades
        counter_strategies = ['closer_atm_bull_put_spread', 'bear_put_spread', 'short_call_supplement']
        
        # Filter trades for counter strategies only
        counter_trades_df = trades_df[trades_df['strategy'].isin(counter_strategies)].copy()
        
        # Filter daily results for counter strategy days only
        counter_daily_df = daily_df[daily_df['strategy_used'] == 'counter'].copy()
        
        self.logger.info(f"📊 COUNTER STRATEGY BREAKDOWN:")
        self.logger.info(f"   Total Unified Trades: {len(trades_df)}")
        self.logger.info(f"   Counter Strategy Trades: {len(counter_trades_df)}")
        self.logger.info(f"   Counter Strategy Days: {len(counter_daily_df)}")
        
        # Analyze counter strategy types
        strategy_breakdown = counter_trades_df['strategy'].value_counts()
        self.logger.info(f"\n📈 COUNTER STRATEGY TYPES:")
        for strategy, count in strategy_breakdown.items():
            avg_pnl = counter_trades_df[counter_trades_df['strategy'] == strategy]['final_pnl'].mean()
            self.logger.info(f"   {strategy}: {count} trades, Avg P&L: ${avg_pnl:.2f}")
        
        # Calculate counter strategy performance
        total_counter_pnl = counter_trades_df['final_pnl'].sum()
        counter_winning_trades = len(counter_trades_df[counter_trades_df['final_pnl'] > 0])
        counter_losing_trades = len(counter_trades_df[counter_trades_df['final_pnl'] < 0])
        counter_win_rate = (counter_winning_trades / len(counter_trades_df) * 100) if len(counter_trades_df) > 0 else 0
        
        # Calculate daily performance from counter days
        counter_daily_pnl = counter_daily_df['pnl'].sum()
        trading_days = len(daily_df[daily_df['trade_executed'] == True])
        counter_trading_days = len(counter_daily_df)
        counter_execution_rate = (counter_trading_days / trading_days * 100) if trading_days > 0 else 0
        
        # Time period analysis
        start_date = daily_df['date'].min()
        end_date = daily_df['date'].max()
        total_days = len(daily_df)
        
        self.logger.info(f"\n💰 COUNTER STRATEGY PERFORMANCE:")
        self.logger.info(f"   Period: {start_date} to {end_date} ({total_days} days)")
        self.logger.info(f"   Counter Trades: {len(counter_trades_df)}")
        self.logger.info(f"   Counter Days: {counter_trading_days}")
        self.logger.info(f"   Execution Rate: {counter_execution_rate:.1f}%")
        self.logger.info(f"   Win Rate: {counter_win_rate:.1f}%")
        self.logger.info(f"   Total P&L: ${total_counter_pnl:.2f}")
        self.logger.info(f"   Daily P&L: ${counter_daily_pnl:.2f}")
        self.logger.info(f"   Avg P&L per Trade: ${total_counter_pnl/len(counter_trades_df):.2f}")
        self.logger.info(f"   Avg P&L per Day: ${counter_daily_pnl/counter_trading_days:.2f}")
        
        return {
            'counter_trades_df': counter_trades_df,
            'counter_daily_df': counter_daily_df,
            'total_counter_pnl': total_counter_pnl,
            'counter_daily_pnl': counter_daily_pnl,
            'counter_trades': len(counter_trades_df),
            'counter_days': counter_trading_days,
            'execution_rate': counter_execution_rate,
            'win_rate': counter_win_rate,
            'avg_pnl_per_trade': total_counter_pnl/len(counter_trades_df) if len(counter_trades_df) > 0 else 0,
            'avg_pnl_per_day': counter_daily_pnl/counter_trading_days if counter_trading_days > 0 else 0,
            'period_days': total_days,
            'start_date': start_date,
            'end_date': end_date
        }
    
    def scale_for_25k_account(self, counter_results: Dict) -> Dict:
        """
        Scale counter strategy results for 25K account
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"💵 SCALING FOR 25K ACCOUNT")
        self.logger.info(f"{'='*60}")
        
        # Current counter strategy uses 1 contract per trade typically
        # For 25K account, we can scale based on capital efficiency
        
        # Analyze current capital usage from the trades
        counter_trades_df = counter_results['counter_trades_df']
        
        # Calculate average capital per trade (max risk)
        if 'max_risk' in counter_trades_df.columns:
            avg_capital_per_trade = counter_trades_df['max_risk'].mean()
        else:
            # Estimate based on strategy types - counter strategies are capital efficient
            avg_capital_per_trade = 150  # Conservative estimate for counter spreads
        
        # Calculate scaling factor for 25K account
        # Assume we want to use 10% of account per trade (conservative)
        target_capital_per_trade = 25000 * 0.10  # $2,500 per trade
        scaling_factor = target_capital_per_trade / avg_capital_per_trade
        
        # Apply scaling
        scaled_total_pnl = counter_results['total_counter_pnl'] * scaling_factor
        scaled_daily_pnl = counter_results['counter_daily_pnl'] * scaling_factor
        scaled_avg_per_trade = counter_results['avg_pnl_per_trade'] * scaling_factor
        scaled_avg_per_day = counter_results['avg_pnl_per_day'] * scaling_factor
        
        # Calculate 6-month projections
        days_per_month = 22  # Trading days
        period_months = counter_results['period_days'] / days_per_month
        six_month_days = 6 * days_per_month
        
        # Project to 6 months
        six_month_pnl = scaled_avg_per_day * counter_results['counter_days'] * (6 / period_months)
        
        self.logger.info(f"📊 SCALING ANALYSIS:")
        self.logger.info(f"   Current Avg Capital/Trade: ${avg_capital_per_trade:.2f}")
        self.logger.info(f"   Target Capital/Trade (25K): ${target_capital_per_trade:.2f}")
        self.logger.info(f"   Scaling Factor: {scaling_factor:.2f}x")
        
        self.logger.info(f"\n💰 SCALED COUNTER STRATEGY (25K Account):")
        self.logger.info(f"   Original Total P&L: ${counter_results['total_counter_pnl']:.2f}")
        self.logger.info(f"   Scaled Total P&L: ${scaled_total_pnl:.2f}")
        self.logger.info(f"   Scaled Avg/Trade: ${scaled_avg_per_trade:.2f}")
        self.logger.info(f"   Scaled Avg/Day: ${scaled_avg_per_day:.2f}")
        
        self.logger.info(f"\n🎯 6-MONTH PROJECTION (25K Account):")
        self.logger.info(f"   Period Analyzed: {period_months:.1f} months")
        self.logger.info(f"   Counter Days in Period: {counter_results['counter_days']}")
        self.logger.info(f"   Projected 6-Month P&L: ${six_month_pnl:.2f}")
        self.logger.info(f"   Monthly Average: ${six_month_pnl/6:.2f}")
        
        return {
            'original_results': counter_results,
            'scaling_factor': scaling_factor,
            'avg_capital_per_trade': avg_capital_per_trade,
            'target_capital_per_trade': target_capital_per_trade,
            'scaled_total_pnl': scaled_total_pnl,
            'scaled_avg_per_trade': scaled_avg_per_trade,
            'scaled_avg_per_day': scaled_avg_per_day,
            'six_month_projection': six_month_pnl,
            'monthly_average': six_month_pnl/6,
            'period_months': period_months
        }
    
    def analyze_capital_efficiency(self, counter_results: Dict, scaled_results: Dict) -> Dict:
        """
        Analyze capital efficiency vs other strategies
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"⚡ CAPITAL EFFICIENCY ANALYSIS")
        self.logger.info(f"{'='*60}")
        
        counter_trades_df = counter_results['counter_trades_df']
        
        # Analyze by strategy type
        efficiency_by_strategy = {}
        
        for strategy in counter_trades_df['strategy'].unique():
            strategy_trades = counter_trades_df[counter_trades_df['strategy'] == strategy]
            
            # Calculate efficiency metrics
            avg_pnl = strategy_trades['final_pnl'].mean()
            if 'max_risk' in strategy_trades.columns:
                avg_risk = strategy_trades['max_risk'].mean()
                roi_per_trade = (avg_pnl / avg_risk) * 100 if avg_risk > 0 else 0
            else:
                # Estimate risk based on strategy type
                if 'bear_put_spread' in strategy:
                    avg_risk = 100  # Typically $100 max cost
                elif 'closer_atm' in strategy:
                    avg_risk = 300  # Typically $300 max risk
                else:
                    avg_risk = 50   # Short calls have lower capital requirements
                roi_per_trade = (avg_pnl / avg_risk) * 100
            
            efficiency_by_strategy[strategy] = {
                'trades': len(strategy_trades),
                'avg_pnl': avg_pnl,
                'avg_risk': avg_risk,
                'roi_per_trade': roi_per_trade,
                'total_pnl': strategy_trades['final_pnl'].sum()
            }
            
            self.logger.info(f"📊 {strategy}:")
            self.logger.info(f"   Trades: {len(strategy_trades)}")
            self.logger.info(f"   Avg P&L: ${avg_pnl:.2f}")
            self.logger.info(f"   Avg Risk: ${avg_risk:.2f}")
            self.logger.info(f"   ROI/Trade: {roi_per_trade:.2f}%")
        
        # Compare to Iron Condor estimates
        iron_condor_risk = 150  # Estimated from our analysis
        iron_condor_daily_pnl = 2.44  # From our Iron Condor multi-trade backtest
        iron_condor_roi = (iron_condor_daily_pnl / iron_condor_risk) * 100
        
        counter_avg_risk = scaled_results['avg_capital_per_trade']
        counter_daily_pnl = scaled_results['scaled_avg_per_day']
        counter_roi = (counter_daily_pnl / counter_avg_risk) * 100 if counter_avg_risk > 0 else 0
        
        self.logger.info(f"\n🏆 EFFICIENCY COMPARISON:")
        self.logger.info(f"   Counter Strategy ROI: {counter_roi:.2f}%/day")
        self.logger.info(f"   Iron Condor ROI: {iron_condor_roi:.2f}%/day")
        self.logger.info(f"   Efficiency Advantage: {counter_roi - iron_condor_roi:.2f} percentage points")
        
        return {
            'efficiency_by_strategy': efficiency_by_strategy,
            'counter_roi': counter_roi,
            'iron_condor_roi': iron_condor_roi,
            'efficiency_advantage': counter_roi - iron_condor_roi
        }
    
    def run_analysis(self, trades_file: str = "proper_unified_trades_20250801_114942.csv", 
                    daily_file: str = "proper_unified_daily_20250801_114942.csv") -> Dict:
        """
        Run complete counter strategy analysis
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🛡️ COUNTER STRATEGY ANALYSIS - ORIGINAL UNIFIED BACKTEST")
        self.logger.info(f"🎯 Goal: Extract counter strategy performance for 25K account scaling")
        self.logger.info(f"{'='*80}")
        
        # Load data
        trades_df, daily_df = self.load_unified_backtest_data(trades_file, daily_file)
        if trades_df is None or daily_df is None:
            return {'error': 'Failed to load data'}
        
        # Extract counter strategy performance
        counter_results = self.extract_counter_strategy_trades(trades_df, daily_df)
        
        # Scale for 25K account
        scaled_results = self.scale_for_25k_account(counter_results)
        
        # Analyze capital efficiency
        efficiency_results = self.analyze_capital_efficiency(counter_results, scaled_results)
        
        # Final summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🎯 FINAL COUNTER STRATEGY SUMMARY (25K Account)")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"📊 6-Month Counter Strategy Projection:")
        self.logger.info(f"   Total Profit: ${scaled_results['six_month_projection']:.2f}")
        self.logger.info(f"   Monthly Average: ${scaled_results['monthly_average']:.2f}")
        self.logger.info(f"   Execution Rate: {counter_results['execution_rate']:.1f}%")
        self.logger.info(f"   Win Rate: {counter_results['win_rate']:.1f}%")
        self.logger.info(f"   Capital Efficiency: {efficiency_results['counter_roi']:.2f}% ROI/day")
        
        return {
            'counter_results': counter_results,
            'scaled_results': scaled_results,
            'efficiency_results': efficiency_results,
            'summary': {
                'six_month_profit': scaled_results['six_month_projection'],
                'monthly_average': scaled_results['monthly_average'],
                'execution_rate': counter_results['execution_rate'],
                'win_rate': counter_results['win_rate'],
                'roi_per_day': efficiency_results['counter_roi']
            }
        }

def main():
    """Run counter strategy analysis"""
    analyzer = CounterStrategyAnalysis()
    results = analyzer.run_analysis()
    
    if 'error' not in results:
        print(f"\n🎉 Analysis complete!")
        print(f"📈 6-Month Counter Strategy Profit (25K): ${results['summary']['six_month_profit']:.2f}")

if __name__ == "__main__":
    main()