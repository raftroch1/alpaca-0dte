#!/usr/bin/env python3
"""
🎯 PHASE 4D: REFINED OPTIMIZATION ENGINE - TARGETED PROFITABILITY DISCOVERY
==========================================================================

Refined optimization based on deep optimization insights:
✅ Focus on wider spreads (5-15 points) 
✅ Smaller position sizes (1-3 contracts)
✅ Higher profit targets (50-90%)
✅ Tighter stop losses (75-125%)
✅ Fixed win rate simulation for bull put spreads
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from itertools import product

# Add project root to path
sys.path.append('/Users/devops/Desktop/coding projects/windsurf/Alpaca_0dte/alpaca-0dte')

from phase4d_final_working_strategy import Phase4DFinalWorkingStrategy


class RefinedOptimizationConfig:
    """Refined configuration for optimization parameters"""
    
    def __init__(self, 
                 strike_width: float,
                 position_size: int,
                 profit_target_pct: float,
                 stop_loss_pct: float,
                 max_daily_trades: int,
                 base_win_rate: float = 0.65):
        self.strike_width = strike_width
        self.position_size = position_size
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_daily_trades = max_daily_trades
        self.base_win_rate = base_win_rate
        
    def __str__(self):
        return f"W{self.strike_width}_P{self.position_size}_PT{self.profit_target_pct:.0%}_SL{self.stop_loss_pct:.0%}_T{self.max_daily_trades}"


class Phase4DRefinedOptimization:
    """Refined optimization engine focusing on promising parameter ranges"""
    
    def __init__(self):
        """Initialize refined optimization engine"""
        self.base_strategy = Phase4DFinalWorkingStrategy()
        
        # Refined optimization space based on insights
        self.optimization_space = {
            'strike_widths': [5.0, 7.0, 10.0, 12.0, 15.0],     # Wider spreads performed better
            'position_sizes': [1, 2, 3],                        # Smaller sizes performed better
            'profit_targets': [0.50, 0.60, 0.75, 0.85],        # Higher targets performed better
            'stop_losses': [0.75, 1.00, 1.25],                 # Tighter stops performed better
            'daily_trade_limits': [8, 10, 12],                 # Focus on optimal range
            'base_win_rates': [0.60, 0.65, 0.70]               # Test different win rate assumptions
        }
        
        # Sample test dates
        self.sample_dates = [
            '2024-01-15', '2024-01-26', '2024-02-09', '2024-02-23',
            '2024-03-08', '2024-03-22', '2024-04-05', '2024-04-19',
            '2024-05-03', '2024-05-17', '2024-06-07', '2024-06-21'
        ]
        
    def run_refined_optimization_test(self, config: RefinedOptimizationConfig) -> Dict:
        """Run refined optimization test with improved simulation"""
        try:
            # Test across sample dates
            total_pnl = 0.0
            total_trades = 0
            total_credit = 0.0
            profitable_days = 0
            all_daily_pnls = []
            
            for date_str in self.sample_dates:
                daily_result = self._simulate_refined_day(date_str, config)
                
                daily_pnl = daily_result['total_pnl']
                total_pnl += daily_pnl
                total_trades += daily_result['trades']
                total_credit += daily_result['credit_collected']
                
                if daily_pnl > 0:
                    profitable_days += 1
                    
                all_daily_pnls.append(daily_pnl)
            
            # Calculate metrics
            avg_daily_pnl = total_pnl / len(self.sample_dates)
            profitability_rate = profitable_days / len(self.sample_dates)
            avg_trades_per_day = total_trades / len(self.sample_dates)
            
            # Risk metrics
            daily_std = np.std(all_daily_pnls)
            sharpe_ratio = (avg_daily_pnl / daily_std) * np.sqrt(252) if daily_std > 0 else 0
            max_daily_loss = min(all_daily_pnls)
            max_daily_gain = max(all_daily_pnls)
            
            # Target achievement
            target_days = len([pnl for pnl in all_daily_pnls if pnl >= 300])
            target_achievement_rate = target_days / len(self.sample_dates)
            
            # Enhanced profitability score
            profit_score = (
                (avg_daily_pnl / 400) * 50 +           # 50% weight on daily P&L vs $400 target
                profitability_rate * 25 +              # 25% weight on win rate
                target_achievement_rate * 15 +         # 15% weight on target achievement
                max(0, min(sharpe_ratio / 2, 1)) * 10  # 10% weight on risk-adjusted returns
            )
            
            return {
                'config': config,
                'total_pnl': total_pnl,
                'avg_daily_pnl': avg_daily_pnl,
                'profitable_days': profitable_days,
                'profitability_rate': profitability_rate,
                'total_trades': total_trades,
                'avg_trades_per_day': avg_trades_per_day,
                'total_credit': total_credit,
                'daily_std': daily_std,
                'sharpe_ratio': sharpe_ratio,
                'max_daily_loss': max_daily_loss,
                'max_daily_gain': max_daily_gain,
                'target_days': target_days,
                'target_achievement_rate': target_achievement_rate,
                'profit_score': profit_score,
                'is_profitable': total_pnl > 0
            }
            
        except Exception as e:
            return {
                'config': config,
                'total_pnl': -999999,
                'profit_score': -999,
                'is_profitable': False,
                'error': str(e)
            }
    
    def _simulate_refined_day(self, date_str: str, config: RefinedOptimizationConfig) -> Dict:
        """Simulate day with refined parameters and improved logic"""
        try:
            # Get market data
            spy_price = self.base_strategy.get_spy_price(date_str)
            expiration = pd.Timestamp(f"{date_str} 16:00:00", tz='America/New_York')
            options = self.base_strategy.generate_realistic_option_chain(spy_price, expiration)
            
            # Find optimal spread with target strike width
            spread = self._find_spread_with_target_width(options, spy_price, config.strike_width)
            
            if not spread:
                return {
                    'date': date_str,
                    'spy_price': spy_price,
                    'total_pnl': 0.0,
                    'trades': 0,
                    'credit_collected': 0.0
                }
            
            # Simulate trading session with refined parameters
            return self._simulate_refined_session(date_str, spy_price, spread, config)
            
        except Exception as e:
            return {
                'date': date_str,
                'spy_price': spy_price,
                'total_pnl': 0.0,
                'trades': 0,
                'credit_collected': 0.0,
                'error': str(e)
            }
    
    def _find_spread_with_target_width(self, options: List[Dict], spy_price: float, target_width: float) -> Dict:
        """Find spread with specific target width"""
        try:
            # Find short leg candidates (around -0.40 delta)
            short_candidates = [
                opt for opt in options
                if abs(opt['delta'] - (-0.40)) <= 0.08  # Slightly wider tolerance
            ]
            
            if not short_candidates:
                return None
                
            # Try each short candidate
            for short_option in short_candidates:
                target_long_strike = short_option['strike'] - target_width
                
                # Find closest long leg
                long_candidates = [
                    opt for opt in options
                    if opt['strike'] < short_option['strike']
                    and abs(opt['strike'] - target_long_strike) <= 2.0  # Within $2 of target
                ]
                
                if not long_candidates:
                    continue
                    
                # Select closest to target strike
                long_option = min(long_candidates, key=lambda x: abs(x['strike'] - target_long_strike))
                
                # Calculate spread metrics
                net_credit = short_option['bid'] - long_option['ask']
                spread_width = short_option['strike'] - long_option['strike']
                max_profit = net_credit
                max_loss = spread_width - net_credit
                
                # Validate spread
                if (net_credit >= 0.15 and  # Minimum meaningful credit
                    max_loss / max_profit <= 25.0 and  # Reasonable risk/reward
                    spread_width >= (target_width - 2.0) and  # Close to target width
                    spread_width <= (target_width + 2.0)):
                    
                    return {
                        'short_strike': short_option['strike'],
                        'long_strike': long_option['strike'],
                        'short_delta': short_option['delta'],
                        'long_delta': long_option['delta'],
                        'net_credit': net_credit,
                        'max_profit': max_profit,
                        'max_loss': max_loss,
                        'spread_width': spread_width
                    }
            
            return None
            
        except Exception:
            return None
    
    def _simulate_refined_session(self, date_str: str, spy_price: float, spread: Dict, config: RefinedOptimizationConfig) -> Dict:
        """Simulate trading session with refined win rate and exit logic"""
        daily_trades = config.max_daily_trades
        
        session_pnl = 0.0
        session_credit = 0.0
        
        # Use realistic win rate for bull put spreads
        win_rate = config.base_win_rate
        
        # Apply market condition adjustments
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        if date_obj.weekday() == 4:  # Friday (0DTE expiration)
            win_rate *= 0.95  # Slightly lower on expiration day
        elif date_obj.month in [1, 3]:  # Higher volatility months
            win_rate *= 0.90
        
        winners = int(daily_trades * win_rate)
        losers = daily_trades - winners
        
        # Calculate per-trade metrics
        credit_per_trade = spread['net_credit'] * config.position_size * 100
        max_profit_per_trade = spread['max_profit'] * config.position_size * 100
        
        # Simulate winners with refined profit taking
        for _ in range(winners):
            # Winners close at profit target or better
            profit_achieved = min(config.profit_target_pct, np.random.uniform(0.30, 0.90))
            trade_pnl = max_profit_per_trade * profit_achieved
            session_pnl += trade_pnl
            session_credit += credit_per_trade
        
        # Simulate losers with refined stop loss
        for _ in range(losers):
            # Losers hit stop loss (can vary slightly)
            loss_multiplier = config.stop_loss_pct * np.random.uniform(0.90, 1.10)
            trade_pnl = -max_profit_per_trade * loss_multiplier
            session_pnl += trade_pnl
            session_credit += credit_per_trade
        
        return {
            'date': date_str,
            'spy_price': spy_price,
            'total_pnl': session_pnl,
            'trades': daily_trades,
            'credit_collected': session_credit,
            'winners': winners,
            'losers': losers,
            'realized_win_rate': win_rate
        }
    
    def run_comprehensive_refined_optimization(self) -> List[Dict]:
        """Run comprehensive refined optimization"""
        print("🎯 PHASE 4D: REFINED OPTIMIZATION ENGINE")
        print("=" * 70)
        
        # Generate parameter combinations
        param_combinations = list(product(
            self.optimization_space['strike_widths'],
            self.optimization_space['position_sizes'],
            self.optimization_space['profit_targets'],
            self.optimization_space['stop_losses'],
            self.optimization_space['daily_trade_limits'],
            self.optimization_space['base_win_rates']
        ))
        
        total_combinations = len(param_combinations)
        print(f"🔬 Testing {total_combinations} refined parameter combinations")
        print(f"📅 Sample Size: {len(self.sample_dates)} representative trading days")
        print(f"🎯 Target: Discover profitable configuration for $300-500 daily profit")
        print("=" * 70)
        
        print(f"\n🚀 Beginning refined optimization of {total_combinations} configurations...\n")
        
        results = []
        profitable_configs = []
        
        for i, (width, size, profit, stop, trades, win_rate) in enumerate(param_combinations, 1):
            config = RefinedOptimizationConfig(width, size, profit, stop, trades, win_rate)
            
            if i % 10 == 0 or i == total_combinations:
                print(f"📊 Testing configuration {i}/{total_combinations}: {config}")
            
            result = self.run_refined_optimization_test(config)
            results.append(result)
            
            if result.get('is_profitable', False):
                profitable_configs.append(result)
        
        # Sort by profit score
        results.sort(key=lambda x: x.get('profit_score', -999), reverse=True)
        
        print(f"\n✅ Refined Optimization Complete!")
        print(f"📊 Total Configurations Tested: {total_combinations}")
        print(f"💚 Profitable Configurations Found: {len(profitable_configs)}")
        print(f"📈 Success Rate: {len(profitable_configs)/total_combinations:.1%}")
        
        return results
    
    def print_refined_summary(self, results: List[Dict]):
        """Print refined optimization summary"""
        print("\n" + "=" * 90)
        print("🎯 PHASE 4D: REFINED OPTIMIZATION RESULTS")
        print("=" * 90)
        
        top_10 = results[:10]
        profitable_results = [r for r in results if r.get('is_profitable', False)]
        
        print(f"\n🏆 TOP 10 REFINED CONFIGURATIONS:")
        print(f"{'Rank':<4} {'Configuration':<35} {'Daily P&L':<12} {'Win Rate':<10} {'Target Rate':<12} {'Score':<8}")
        print("-" * 90)
        
        for i, result in enumerate(top_10, 1):
            config = result['config']
            config_str = f"W{config.strike_width}|P{config.position_size}|PT{config.profit_target_pct:.0%}|SL{config.stop_loss_pct:.0%}|WR{config.base_win_rate:.0%}"
            
            print(f"{i:<4} {config_str:<35} ${result.get('avg_daily_pnl', 0):<11.2f} {result.get('profitability_rate', 0):<9.1%} {result.get('target_achievement_rate', 0):<11.1%} {result.get('profit_score', 0):<7.1f}")
        
        if profitable_results:
            best = profitable_results[0]
            
            print(f"\n🥇 OPTIMAL REFINED CONFIGURATION:")
            config = best['config']
            print(f"   📊 Strike Width: {config.strike_width} points")
            print(f"   📊 Position Size: {config.position_size} contracts")
            print(f"   📊 Profit Target: {config.profit_target_pct:.0%} of max profit")
            print(f"   📊 Stop Loss: {config.stop_loss_pct:.0%} of max profit")
            print(f"   📊 Daily Trades: {config.max_daily_trades}")
            print(f"   📊 Win Rate: {config.base_win_rate:.0%}")
            
            print(f"\n💰 OPTIMAL PERFORMANCE:")
            print(f"   💰 Avg Daily P&L: ${best['avg_daily_pnl']:.2f}")
            print(f"   📈 Profitability Rate: {best['profitability_rate']:.1%}")
            print(f"   🎯 Target Achievement: {best['target_achievement_rate']:.1%}")
            print(f"   📊 Avg Trades/Day: {best['avg_trades_per_day']:.1f}")
            print(f"   📈 Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            
            # Projections
            monthly_pnl = best['avg_daily_pnl'] * 21
            annual_return = (monthly_pnl * 12 / 25000) * 100
            
            print(f"\n📅 PROJECTIONS:")
            print(f"   📊 Monthly P&L: ${monthly_pnl:.2f}")
            print(f"   🏆 Annual Return: {annual_return:.1f}%")
            
            return best
        else:
            print(f"\n⚠️ NO PROFITABLE CONFIGURATIONS IN REFINED SPACE")
            print(f"💡 Best configuration: {top_10[0]['config']} with ${top_10[0]['avg_daily_pnl']:.2f} daily")
            
        print("=" * 90)
        return None


if __name__ == "__main__":
    # Run refined optimization
    engine = Phase4DRefinedOptimization()
    refined_results = engine.run_comprehensive_refined_optimization()
    optimal_config = engine.print_refined_summary(refined_results) 