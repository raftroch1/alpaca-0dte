#!/usr/bin/env python3
"""
🚀 PROPER UNIFIED SYSTEM
========================

What the user actually wanted:
1. Minimal scaling of balanced strategy (2x volume, keep signal quality)
2. Focus counter strategy on the ~54% of days balanced strategy doesn't trade
3. Intelligent switching between strategies

🎯 GOAL: Higher execution rate while maintaining excellent signal quality

Author: Strategy Development Framework
Date: 2025-01-31
Version: Proper Unified v1.0
"""

import os
import sys
import logging
from datetime import datetime
import argparse

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our strategies
from phase4d_balanced_minimal_scale import Phase4DBalancedMinimalScale
from focused_counter_strategy import FocusedCounterStrategy

class ProperUnifiedSystem:
    """
    Proper implementation of what the user wanted:
    - Minimal scale balanced (keep signal quality, 2x volume)  
    - Focused counter for filtered days
    """
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        self.cache_dir = cache_dir
        self.setup_logging()
        
        self.primary_strategy = Phase4DBalancedMinimalScale(cache_dir=self.cache_dir)
        self.counter_strategy = FocusedCounterStrategy(cache_dir=self.cache_dir)
        
        self.total_pnl = 0
        self.daily_pnl = 0
        self.last_run_date = None
        
    def setup_logging(self):
        """Setup logging for the unified system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def run_unified_strategy(self, date_str: str) -> dict:
        """
        Run the proper unified approach:
        1. Try minimal scale balanced (excellent signal, 2x volume)
        2. If filtered, try focused counter strategy
        """
        # Reset daily P&L if it's a new day
        if self.last_run_date != date_str:
            self.daily_pnl = 0
            self.last_run_date = date_str

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🚀 PROPER UNIFIED SYSTEM - {date_str}")
        self.logger.info(f"📈 Primary: Minimal Scale Balanced (2x volume, same quality)")
        self.logger.info(f"🛡️ Counter: Focused on filtered days")
        self.logger.info(f"💰 Daily P&L: ${self.daily_pnl:.2f}")
        self.logger.info(f"{'='*70}")

        trade_info = {
            'date': date_str, 
            'primary_trade': None, 
            'counter_trade': None, 
            'daily_pnl': 0,
            'strategy_used': None
        }

        # --- STEP 1: Try Primary Strategy (Minimal Scale Balanced) ---
        self.logger.info(f"\n🎯 STEP 1: Trying PRIMARY Strategy (Minimal Scale Balanced)")
        primary_result = self.primary_strategy.run_single_day(date_str)

        if primary_result.get('success'):
            # PRIMARY SUCCESS
            trade = primary_result['trade']
            self.logger.info(f"✅ PRIMARY SUCCESS: ${trade['final_pnl']:.2f} ({trade['strategy']})")
            self.daily_pnl += trade['final_pnl']
            trade_info.update({
                'success': True,
                'primary_trade': trade,
                'trade': trade,  # For compatibility
                'strategy_used': 'primary',
                'daily_pnl': self.daily_pnl,
                'spy_close': primary_result.get('spy_close')
            })
            
        else:
            # PRIMARY FILTERED - Try counter strategy
            filter_reason = primary_result.get('reason', 'Unknown filter')
            self.logger.info(f"❌ PRIMARY FILTERED: {filter_reason}")
            
            # --- STEP 2: Try Counter Strategy (Focused) ---
            self.logger.info(f"\n🛡️ STEP 2: Trying COUNTER Strategy (Focused)")
            counter_result = self.counter_strategy.run_counter_strategy(date_str, filter_reason)

            if counter_result.get('success'):
                # COUNTER SUCCESS
                trade = counter_result['trade']
                scenario = counter_result['scenario']
                self.logger.info(f"✅ COUNTER SUCCESS: ${trade['final_pnl']:.2f} ({trade['strategy']}, {scenario})")
                self.daily_pnl += trade['final_pnl']
                trade_info.update({
                    'success': True,
                    'counter_trade': trade,
                    'trade': trade,  # For compatibility
                    'strategy_used': 'counter',
                    'daily_pnl': self.daily_pnl,
                    'spy_close': counter_result.get('spy_close')
                })
                
            else:
                # BOTH STRATEGIES FILTERED
                counter_reason = counter_result.get('reason', 'Unknown')
                self.logger.info(f"❌ COUNTER FILTERED: {counter_reason}")
                self.logger.info(f"📊 NO TRADE: Both strategies filtered this day")
                
                trade_info.update({
                    'no_trade': True,
                    'reason': f"Primary: {filter_reason}, Counter: {counter_reason}",
                    'spy_close': primary_result.get('spy_close') or counter_result.get('spy_close')
                })
        
        trade_info['daily_pnl'] = self.daily_pnl
        self.logger.info(f"💰 Daily P&L: ${self.daily_pnl:.2f}")
        return trade_info

def main():
    """Main execution for the proper unified system."""
    parser = argparse.ArgumentParser(description='Proper Unified System')
    parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    parser.add_argument('--cache-dir', default='../../thetadata/cached_data', help='Cache directory')
    
    args = parser.parse_args()
    
    print(f"🚀 Proper Unified System")
    print(f"📈 Primary: Minimal scale balanced (2x volume, same quality)")
    print(f"🛡️ Counter: Focused on filtered days")
    print(f"🎯 Goal: Higher execution rate with excellent signal quality")
    print(f"📅 Date: {args.date}")
    
    system = ProperUnifiedSystem(cache_dir=args.cache_dir)
    result = system.run_unified_strategy(args.date)
    
    if result.get('success'):
        trade = result['trade']
        strategy_used = result['strategy_used']
        print(f"✅ {strategy_used.upper()} TRADE: ${trade['final_pnl']:.2f} ({trade['strategy']})")
        print(f"💰 Daily P&L: ${result['daily_pnl']:.2f}")
    elif result.get('no_trade'):
        print(f"📊 No trade executed: {result['reason']}")
    else:
        print(f"❌ Error in execution")

if __name__ == "__main__":
    main()
