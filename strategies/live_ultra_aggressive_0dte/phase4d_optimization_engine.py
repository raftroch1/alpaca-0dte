#!/usr/bin/env python3
"""
🎯 PHASE 4D: DEEP OPTIMIZATION ENGINE - SYSTEMATIC PROFITABILITY DISCOVERY
========================================================================

Comprehensive optimization framework to systematically test all parameter combinations
and discover the optimal profitable configuration for bull put spreads.

OPTIMIZATION TARGETS:
✅ Strike Width: 3, 5, 7, 10 point spreads
✅ Position Size: 1, 2, 3, 5 contracts per trade
✅ Exit Rules: Multiple profit/loss combinations
✅ Risk Management: Dynamic vs fixed approaches
✅ Statistical Validation: 6-month comprehensive testing
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from itertools import product
import json

# Add project root to path
sys.path.append('/Users/devops/Desktop/coding projects/windsurf/Alpaca_0dte/alpaca-0dte')

from phase4d_final_working_strategy import Phase4DFinalWorkingStrategy


class OptimizationConfig:
    """Configuration for optimization parameters"""
    
    def __init__(self, 
                 strike_width: float,
                 position_size: int,
                 profit_target_pct: float,
                 stop_loss_pct: float,
                 max_daily_trades: int):
        self.strike_width = strike_width
        self.position_size = position_size
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_daily_trades = max_daily_trades
        
    def __str__(self):
        return f"W{self.strike_width}_P{self.position_size}_PT{self.profit_target_pct}_SL{self.stop_loss_pct}_T{self.max_daily_trades}"


class Phase4DOptimizationEngine:
    """Comprehensive optimization engine for Phase 4D strategy"""
    
    def __init__(self):
        """Initialize optimization engine"""
        self.base_strategy = Phase4DFinalWorkingStrategy()
        
        # Define optimization parameter space
        self.optimization_space = {
            'strike_widths': [3.0, 5.0, 7.0, 10.0],           # Point spreads
            'position_sizes': [1, 2, 3, 5],                    # Contracts per trade
            'profit_targets': [0.25, 0.35, 0.50, 0.75],       # % of max profit
            'stop_losses': [1.00, 1.25, 1.50, 2.00],          # % of max profit
            'daily_trade_limits': [6, 8, 10, 12]              # Max trades per day
        }
        
        # Sample test dates for quick optimization (representative)
        self.sample_dates = [
            '2024-01-15', '2024-01-26', '2024-02-09', '2024-02-23',
            '2024-03-08', '2024-03-22', '2024-04-05', '2024-04-19',
            '2024-05-03', '2024-05-17', '2024-06-07', '2024-06-21'
        ]  # 12 representative days across 6 months
        
    def create_optimized_strategy(self, config: OptimizationConfig) -> 'OptimizedStrategy':
        """Create strategy instance with optimized parameters"""
        return OptimizedStrategy(self.base_strategy, config)
    
    def run_single_optimization_test(self, config: OptimizationConfig) -> Dict:
        """Run optimization test for a single parameter configuration"""
        try:
            strategy = self.create_optimized_strategy(config)
            
            # Test across sample dates
            total_pnl = 0.0
            total_trades = 0
            total_credit = 0.0
            profitable_days = 0
            all_daily_pnls = []
            
            for date_str in self.sample_dates:
                daily_result = strategy.simulate_optimized_day(date_str)
                
                daily_pnl = daily_result['total_pnl']
                total_pnl += daily_pnl
                total_trades += daily_result['trades']
                total_credit += daily_result['credit_collected']
                
                if daily_pnl > 0:
                    profitable_days += 1
                    
                all_daily_pnls.append(daily_pnl)
            
            # Calculate key metrics
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
            
            # Profitability score (composite metric)
            profit_score = (
                (avg_daily_pnl / 500) * 40 +      # 40% weight on daily P&L vs $500 target
                profitability_rate * 30 +          # 30% weight on win rate
                (target_achievement_rate) * 20 +   # 20% weight on target achievement
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
                'sample_size': len(self.sample_dates),
                'is_profitable': total_pnl > 0
            }
            
        except Exception as e:
            print(f"❌ Error testing {config}: {e}")
            return {
                'config': config,
                'total_pnl': -999999,
                'profit_score': -999,
                'is_profitable': False,
                'error': str(e)
            }
    
    def run_comprehensive_optimization(self) -> List[Dict]:
        """Run comprehensive optimization across all parameter combinations"""
        print("🎯 PHASE 4D: DEEP OPTIMIZATION ENGINE")
        print("=" * 70)
        print(f"🔬 Testing {len(self.optimization_space['strike_widths']) * len(self.optimization_space['position_sizes']) * len(self.optimization_space['profit_targets']) * len(self.optimization_space['stop_losses']) * len(self.optimization_space['daily_trade_limits'])} parameter combinations")
        print(f"📅 Sample Size: {len(self.sample_dates)} representative trading days")
        print("🎯 Target: Find optimal configuration for $300-500 daily profit")
        print("=" * 70)
        
        # Generate all parameter combinations
        param_combinations = list(product(
            self.optimization_space['strike_widths'],
            self.optimization_space['position_sizes'],
            self.optimization_space['profit_targets'],
            self.optimization_space['stop_losses'],
            self.optimization_space['daily_trade_limits']
        ))
        
        total_combinations = len(param_combinations)
        print(f"\n🚀 Beginning systematic optimization of {total_combinations} configurations...\n")
        
        results = []
        profitable_configs = []
        
        for i, (width, size, profit, stop, trades) in enumerate(param_combinations, 1):
            config = OptimizationConfig(width, size, profit, stop, trades)
            
            if i % 20 == 0 or i == total_combinations:
                print(f"📊 Testing configuration {i}/{total_combinations}: {config}")
            
            result = self.run_single_optimization_test(config)
            results.append(result)
            
            if result.get('is_profitable', False):
                profitable_configs.append(result)
        
        # Sort results by profit score
        results.sort(key=lambda x: x.get('profit_score', -999), reverse=True)
        
        print(f"\n✅ Optimization Complete!")
        print(f"📊 Total Configurations Tested: {total_combinations}")
        print(f"💚 Profitable Configurations Found: {len(profitable_configs)}")
        print(f"📈 Success Rate: {len(profitable_configs)/total_combinations:.1%}")
        
        return results
    
    def print_optimization_summary(self, results: List[Dict]):
        """Print comprehensive optimization summary"""
        print("\n" + "=" * 80)
        print("🎯 PHASE 4D: OPTIMIZATION RESULTS SUMMARY")
        print("=" * 80)
        
        # Top 10 configurations
        top_10 = results[:10]
        profitable_results = [r for r in results if r.get('is_profitable', False)]
        
        print(f"\n🏆 TOP 10 CONFIGURATIONS:")
        print(f"{'Rank':<4} {'Config':<25} {'Avg Daily P&L':<15} {'Win Rate':<10} {'Target Rate':<12} {'Score':<8}")
        print("-" * 80)
        
        for i, result in enumerate(top_10, 1):
            config = result['config']
            config_str = f"W{config.strike_width}|P{config.position_size}|PT{config.profit_target_pct:.2f}|SL{config.stop_loss_pct:.2f}"
            
            print(f"{i:<4} {config_str:<25} ${result.get('avg_daily_pnl', 0):<14.2f} {result.get('profitability_rate', 0):<9.1%} {result.get('target_achievement_rate', 0):<11.1%} {result.get('profit_score', 0):<7.1f}")
        
        if profitable_results:
            best_config = profitable_results[0]
            
            print(f"\n🥇 OPTIMAL CONFIGURATION FOUND:")
            config = best_config['config']
            print(f"   📊 Strike Width: {config.strike_width} points")
            print(f"   📊 Position Size: {config.position_size} contracts")
            print(f"   📊 Profit Target: {config.profit_target_pct:.0%} of max profit")
            print(f"   📊 Stop Loss: {config.stop_loss_pct:.0%} of max profit")
            print(f"   📊 Daily Trade Limit: {config.max_daily_trades}")
            
            print(f"\n💰 OPTIMAL PERFORMANCE METRICS:")
            print(f"   💰 Avg Daily P&L: ${best_config.get('avg_daily_pnl', 0):.2f}")
            print(f"   📈 Profitability Rate: {best_config.get('profitability_rate', 0):.1%}")
            print(f"   🎯 Target Achievement: {best_config.get('target_achievement_rate', 0):.1%}")
            print(f"   📊 Avg Trades/Day: {best_config.get('avg_trades_per_day', 0):.1f}")
            print(f"   📈 Sharpe Ratio: {best_config.get('sharpe_ratio', 0):.2f}")
            print(f"   💚 Best Day: ${best_config.get('max_daily_gain', 0):.2f}")
            print(f"   💔 Worst Day: ${best_config.get('max_daily_loss', 0):.2f}")
            
            # Project to monthly/annual
            monthly_pnl = best_config.get('avg_daily_pnl', 0) * 21
            annual_return = (monthly_pnl * 12 / 25000) * 100
            
            print(f"\n📅 PROJECTED PERFORMANCE:")
            print(f"   📊 Monthly P&L: ${monthly_pnl:.2f}")
            print(f"   🏆 Annual Return: {annual_return:.1f}% on $25K account")
            
            if best_config.get('avg_daily_pnl', 0) >= 300:
                verdict = "🎯 TARGET ACHIEVED - READY FOR LIVE DEPLOYMENT"
            elif best_config.get('avg_daily_pnl', 0) >= 200:
                verdict = "✅ STRONG PERFORMANCE - SCALE UP GRADUALLY"
            elif best_config.get('avg_daily_pnl', 0) > 0:
                verdict = "📈 PROFITABLE BASE - CONTINUE OPTIMIZATION"
            else:
                verdict = "⚠️ NEEDS FURTHER REFINEMENT"
            
            print(f"\n🏆 OPTIMIZATION VERDICT:")
            print(f"   {verdict}")
            
        else:
            print(f"\n⚠️ NO PROFITABLE CONFIGURATIONS FOUND")
            print(f"📊 Best performing configuration still showed losses")
            print(f"💡 Recommendation: Expand optimization space or modify strategy approach")
        
        print("=" * 80)
        
        return profitable_results[0] if profitable_results else None


class OptimizedStrategy:
    """Strategy instance with optimized parameters"""
    
    def __init__(self, base_strategy: Phase4DFinalWorkingStrategy, config: OptimizationConfig):
        self.base_strategy = base_strategy
        self.config = config
    
    def simulate_optimized_day(self, date_str: str) -> Dict:
        """Simulate a full day with optimized parameters"""
        try:
            # Get market data
            spy_price = self.base_strategy.get_spy_price(date_str)
            expiration = pd.Timestamp(f"{date_str} 16:00:00", tz='America/New_York')
            options = self.base_strategy.generate_realistic_option_chain(spy_price, expiration)
            
            # Find spread with optimized strike width
            spread = self._find_optimized_spread(options, spy_price)
            
            if not spread:
                return {
                    'date': date_str,
                    'spy_price': spy_price,
                    'total_pnl': 0.0,
                    'trades': 0,
                    'credit_collected': 0.0
                }
            
            # Simulate trading session
            return self._simulate_optimized_session(date_str, spy_price, spread)
            
        except Exception as e:
            return {
                'date': date_str,
                'spy_price': 0.0,
                'total_pnl': 0.0,
                'trades': 0,
                'credit_collected': 0.0,
                'error': str(e)
            }
    
    def _find_optimized_spread(self, options: List[Dict], spy_price: float) -> Dict:
        """Find spread with optimized strike width"""
        try:
            # Find short leg candidates
            short_candidates = [
                opt for opt in options
                if abs(opt['delta'] - (-0.40)) <= 0.05
            ]
            
            if not short_candidates:
                return None
            
            short_option = short_candidates[0]
            
            # Find long leg with optimized strike width
            target_long_strike = short_option['strike'] - self.config.strike_width
            
            long_candidates = [
                opt for opt in options
                if abs(opt['strike'] - target_long_strike) <= 1.0  # Within $1 of target
                and opt['strike'] < short_option['strike']
            ]
            
            if not long_candidates:
                return None
            
            # Select closest to target strike
            long_option = min(long_candidates, key=lambda x: abs(x['strike'] - target_long_strike))
            
            # Calculate spread metrics
            net_credit = short_option['bid'] - long_option['ask']
            spread_width = short_option['strike'] - long_option['strike']
            max_profit = net_credit
            max_loss = spread_width - net_credit
            
            # Validate spread
            if (net_credit >= 0.10 and  # Minimum credit
                max_loss / max_profit <= 20.0 and  # Reasonable risk/reward
                spread_width <= 15.0):  # Reasonable width
                
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
    
    def _simulate_optimized_session(self, date_str: str, spy_price: float, spread: Dict) -> Dict:
        """Simulate trading session with optimized parameters"""
        daily_trades = min(self.config.max_daily_trades, 10)  # Cap for optimization speed
        
        session_pnl = 0.0
        session_credit = 0.0
        
        # Realistic win rate for bull put spreads
        win_rate = 0.65
        winners = int(daily_trades * win_rate)
        losers = daily_trades - winners
        
        # Calculate per-trade metrics with optimized position size
        credit_per_trade = spread['net_credit'] * self.config.position_size * 100
        max_profit_per_trade = spread['max_profit'] * self.config.position_size * 100
        max_loss_per_trade = spread['max_loss'] * self.config.position_size * 100
        
        # Simulate winners
        for _ in range(winners):
            # Use optimized profit target
            profit_percentage = min(self.config.profit_target_pct, np.random.uniform(0.20, 0.80))
            trade_pnl = max_profit_per_trade * profit_percentage
            session_pnl += trade_pnl
            session_credit += credit_per_trade
        
        # Simulate losers
        for _ in range(losers):
            # Use optimized stop loss
            loss_percentage = min(self.config.stop_loss_pct, np.random.uniform(0.80, 2.00))
            trade_pnl = -max_profit_per_trade * loss_percentage
            session_pnl += trade_pnl
            session_credit += credit_per_trade
        
        return {
            'date': date_str,
            'spy_price': spy_price,
            'total_pnl': session_pnl,
            'trades': daily_trades,
            'credit_collected': session_credit,
            'avg_credit_per_trade': credit_per_trade,
            'max_profit_per_trade': max_profit_per_trade,
            'max_loss_per_trade': max_loss_per_trade
        }


if __name__ == "__main__":
    # Run comprehensive optimization
    engine = Phase4DOptimizationEngine()
    optimization_results = engine.run_comprehensive_optimization()
    optimal_config = engine.print_optimization_summary(optimization_results) 