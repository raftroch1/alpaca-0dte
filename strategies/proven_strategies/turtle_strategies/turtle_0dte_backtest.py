#!/usr/bin/env python3
"""
Turtle-Style 0DTE Strategy Backtest
==================================

This strategy extends the proven live_ultra_aggressive_0dte_backtest.py framework
with turtle-style improvements:

KEY IMPROVEMENTS:
✅ NO daily trade limits (removes 20-trade cap that leaves positions open)
✅ Turtle-style breakout signal generation (trend following)
✅ Dynamic profit/loss goals ($300-500 target, stop when reached)
✅ Better position management with proper closing logic

KEEPS ALL REALISTIC SIMULATION:
✅ Exact same simulate_trade function with realistic option pricing
✅ Time decay, volatility effects, and market dynamics
✅ Real ThetaData integration and SPY data access
✅ Variable P&L (no synthetic fixed amounts)

Target: $300-500 daily profit with improved risk management

Usage:
    python turtle_0dte_backtest.py --start_date 20240102 --end_date 20240705
    python turtle_0dte_backtest.py --date 20240315  # Single day test

Author: Strategy Development Framework
Date: 2025-01-29
Version: TURTLE v1.0 (extends LIVE BACKTEST v1.0)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Optional
import logging

# Import the existing proven backtest framework
from live_ultra_aggressive_0dte_backtest import LiveUltraAggressive0DTEBacktest

class Turtle0DTEBacktest(LiveUltraAggressive0DTEBacktest):
    """
    Turtle-style 0DTE backtest that extends the proven simulation framework
    with improved signal generation and NO trade limits
    """
    
    def __init__(self, cache_dir: str = "./thetadata/cached_data"):
        # Initialize with the proven framework
        super().__init__(cache_dir)
        
        # Override with turtle-style parameters
        self.params.update({
            # PROFIT/LOSS GOALS (NO TRADE LIMITS!)
            'daily_profit_target': 500.0,      # Stop when we hit $500 profit
            'max_daily_loss': 350.0,           # Stop when we hit $350 loss
            
            # TURTLE BREAKOUT DETECTION
            'breakout_periods': 20,             # Look back periods for breakout
            'breakout_threshold': 0.15,         # 0.15% minimum move for breakout (stronger than 0.1%)
            'volatility_periods': 10,           # Periods for volatility calculation
            
            # DYNAMIC POSITION SIZING
            'base_position_size': 2,            # Base contracts
            'max_position_size': 8,             # Max contracts per trade
            'volatility_scaling': True,         # Scale size based on volatility
            
            # IMPROVED RISK MANAGEMENT
            'profit_target_pct': 0.75,          # Take profit at 75% gain (vs 50% original)
            'stop_loss_pct': 0.40,              # Stop loss at 40% loss (vs 50% original)
            'max_position_time': 120,           # Max 2 hours per position
            
            # REMOVE TRADE LIMITS!
            # 'max_daily_trades': REMOVED!     # NO MORE 20-trade cap!
        })
        
        # Update logging to turtle-style
        self.logger.info("🐢 TURTLE 0DTE STRATEGY INITIALIZED")
        self.logger.info(f"🎯 Daily Profit Target: ${self.params['daily_profit_target']}")
        self.logger.info(f"🛑 Daily Loss Limit: ${self.params['max_daily_loss']}")
        self.logger.info("🚫 NO TRADE LIMITS - Trading until profit/loss goals met!")
    
    def calculate_volatility(self, prices: pd.Series) -> float:
        """Calculate turtle-style volatility using True Range method"""
        try:
            if len(prices) < self.params['volatility_periods']:
                return 0.02  # Default 2% volatility
                
            # True Range calculation (turtle method)
            high = prices.rolling(window=2).max()
            low = prices.rolling(window=2).min()
            close_prev = prices.shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.params['volatility_periods']).mean().iloc[-1]
            
            # Convert to percentage
            volatility = atr / prices.iloc[-1]
            return min(max(volatility, 0.005), 0.05)  # Cap between 0.5% and 5%
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility calculation error: {e}")
            return 0.02
    
    def detect_turtle_breakout(self, spy_data: pd.DataFrame) -> Dict:
        """Detect turtle-style breakouts (improved signal generation)"""
        try:
            if len(spy_data) < self.params['breakout_periods']:
                return {'type': 'NONE', 'strength': 0.0, 'confidence': 0.0}
            
            prices = spy_data['close']
            current_price = prices.iloc[-1]
            lookback_period = self.params['breakout_periods']
            
            # Calculate breakout levels (exclude current price to avoid look-ahead bias)
            historical_prices = prices.iloc[:-1]  # Exclude current price
            high_level = historical_prices.rolling(window=lookback_period).max().iloc[-1]
            low_level = historical_prices.rolling(window=lookback_period).min().iloc[-1]
            
            # Check for upside breakout
            if current_price > high_level:
                strength = (current_price - high_level) / high_level
                if strength >= self.params['breakout_threshold'] / 100:
                    return {
                        'type': 'CALL_BREAKOUT',
                        'strength': strength,
                        'confidence': min(strength * 10, 1.0),  # Scale to 0-1
                        'price_change_pct': strength * 100,
                        'level': high_level,
                        'direction': 'BULLISH'
                    }
            
            # Check for downside breakout
            elif current_price < low_level:
                strength = (low_level - current_price) / low_level
                if strength >= self.params['breakout_threshold'] / 100:
                    return {
                        'type': 'PUT_BREAKOUT',
                        'strength': strength,
                        'confidence': min(strength * 10, 1.0),  # Scale to 0-1
                        'price_change_pct': -strength * 100,  # Negative for downward move
                        'level': low_level,
                        'direction': 'BEARISH'
                    }
            
            return {'type': 'NONE', 'strength': 0.0, 'confidence': 0.0}
            
        except Exception as e:
            self.logger.warning(f"⚠️ Breakout detection error: {e}")
            return {'type': 'NONE', 'strength': 0.0, 'confidence': 0.0}
    
    def calculate_turtle_position_size(self, confidence: float, volatility: float) -> int:
        """Calculate turtle-style position size based on volatility and confidence"""
        try:
            base_size = self.params['base_position_size']
            
            if not self.params['volatility_scaling']:
                return base_size
            
            # Turtle-style: Reduce size in high volatility, increase in low volatility
            volatility_multiplier = 0.02 / max(volatility, 0.005)  # Target 2% volatility
            volatility_multiplier = min(max(volatility_multiplier, 0.5), 2.0)  # Cap 0.5x to 2x
            
            # Scale by signal confidence
            confidence_multiplier = 1.0 + confidence * 2.0  # Up to 3x for strong signals
            
            # Calculate final size
            position_size = int(base_size * volatility_multiplier * confidence_multiplier)
            position_size = min(position_size, self.params['max_position_size'])
            position_size = max(position_size, 1)  # At least 1 contract
            
            return position_size
            
        except Exception as e:
            self.logger.warning(f"⚠️ Position size calculation error: {e}")
            return self.params['base_position_size']
    
    def generate_signal(self, spy_data: pd.DataFrame, current_time: datetime) -> Optional[Dict]:
        """
        OVERRIDE: Generate turtle-style signals instead of original simple signals
        Returns signal in same format as original for compatibility with simulate_trade
        """
        try:
            # Detect turtle breakout
            breakout = self.detect_turtle_breakout(spy_data)
            if breakout['type'] == 'NONE':
                return None
            
            # Calculate market conditions
            prices = spy_data['close']
            volatility = self.calculate_volatility(prices)
            
            # Calculate dynamic position size
            position_size = self.calculate_turtle_position_size(breakout['confidence'], volatility)
            
            # Create signal in same format as original for compatibility
            signal = {
                'signal_type': breakout['type'],
                'type': breakout['type'],  # Add for compatibility with simulate_trade
                'confidence': breakout['confidence'],
                'price_change_pct': breakout['price_change_pct'],
                'position_size': position_size,
                'volatility': volatility,
                'entry_reason': f"TURTLE_{breakout['direction']}_BREAKOUT",
                'timestamp': current_time,
                'breakout_level': breakout.get('level', 0.0),
                'breakout_strength': breakout['strength'],
                'spy_price': 0.0  # Will be set when called
            }
            
            self.logger.debug(f"🐢 TURTLE SIGNAL: {signal['signal_type']} (confidence: {signal['confidence']:.3f}, strength: {breakout['strength']:.4f})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error generating turtle signal: {e}")
            return None
    
    def check_daily_risk_limits(self) -> bool:
        """
        OVERRIDE: Check profit/loss goals only (NO TRADE LIMITS!)
        """
        try:
            # Stop if we hit daily profit target
            if self.daily_pnl >= self.params['daily_profit_target']:
                self.logger.info(f"🎯 DAILY PROFIT TARGET REACHED: ${self.daily_pnl:.2f}")
                return False
                
            # Stop if we hit daily loss limit
            if self.daily_pnl <= -self.params['max_daily_loss']:
                self.logger.warning(f"🛑 DAILY LOSS LIMIT REACHED: ${self.daily_pnl:.2f}")
                return False
            
            # NO TRADE COUNT LIMITS! Keep trading until profit/loss goals met
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error checking profit/loss goals: {e}")
            return False
    
    def backtest_single_day(self, date_str: str) -> Dict:
        """
        OVERRIDE: Backtest single day with turtle-style logic and NO trade limits
        Uses all the proven simulation logic from parent class
        """
        try:
            self.logger.info(f"🐢 TURTLE BACKTEST: {date_str}")
            
            # Reset daily tracking
            self.daily_pnl = 0.0
            self.daily_trades = 0
            winning_trades_today = 0
            
            # Load data using parent class method (proven logic)
            data = self.load_cached_data(date_str)
            if not data or 'spy_bars' not in data or 'option_chain' not in data:
                self.logger.warning(f"⚠️ No data available for {date_str}")
                return {'date': date_str, 'pnl': 0.0, 'trades': 0, 'win_rate': 0.0}
            
            spy_data = data['spy_bars']
            option_chains = data['option_chain']
            
            # Convert date string to datetime for market hours
            trade_date = datetime.strptime(date_str, '%Y%m%d').date()
            market_open = datetime.combine(trade_date, datetime.min.time().replace(hour=9, minute=30))
            market_close = datetime.combine(trade_date, datetime.min.time().replace(hour=16, minute=0))
            
            # Process each minute during market hours
            for current_time in pd.date_range(market_open, market_close, freq='1min'):
                
                # Check profit/loss goals (NO TRADE LIMITS!)
                if not self.check_daily_risk_limits():
                    self.logger.info(f"🏁 Daily goal reached at {current_time.strftime('%H:%M')} - stopping trading")
                    break
                
                # Get SPY data up to current time for signal generation
                current_spy_data = spy_data[spy_data.index <= current_time]
                if len(current_spy_data) < self.params['breakout_periods']:
                    continue
                
                # Generate turtle signal (using overridden method)
                signal = self.generate_signal(current_spy_data, current_time)
                if signal is None:
                    continue
                
                # Get current SPY price using parent class method (proven logic)
                current_spy_price = self._get_spy_price_at_time(current_spy_data, current_time)
                if current_spy_price is None:
                    continue
                
                # Add SPY price to signal for compatibility
                signal['spy_price'] = current_spy_price
                
                # Find best option using parent class method (proven logic)
                option_type = 'CALL' if signal['signal_type'] in ['CALL_BREAKOUT', 'CALL'] else 'PUT'
                option_info = self.find_best_option(option_chains, current_spy_price, option_type)
                if option_info is None:
                    continue
                
                # Execute trade simulation using parent class method (EXACT SAME realistic logic)
                contracts = signal['position_size']
                trade_result = self.simulate_trade(signal, option_info, contracts, option_chains, spy_data, current_time)
                
                # The parent class simulate_trade always returns a result, check if it has valid data
                if trade_result and 'pnl' in trade_result:
                    self.daily_trades += 1
                    self.total_trades += 1
                    self.daily_pnl += trade_result['pnl']
                    
                    if trade_result['pnl'] > 0:
                        winning_trades_today += 1
                        self.winning_trades += 1
                        result_emoji = "📈 WIN"
                    else:
                        result_emoji = "📉 LOSS"
                    
                    self.logger.info(f"🐢 Trade #{self.daily_trades}: {result_emoji} - ${trade_result['pnl']:.2f} | Daily P&L: ${self.daily_pnl:.2f}")
                    self.logger.debug(f"   Signal: {signal['signal_type']} (confidence: {signal['confidence']:.3f})")
                    self.logger.debug(f"   Entry: ${trade_result['entry_price']:.2f} -> Exit: ${trade_result['exit_price']:.2f} ({trade_result['exit_reason']})")
            
            # Calculate daily results
            daily_win_rate = (winning_trades_today / max(self.daily_trades, 1)) * 100
            
            self.logger.info(f"✅ Day complete: {self.daily_trades} trades, ${self.daily_pnl:.2f} P&L, {daily_win_rate:.1f}% win rate")
            
            return {
                'date': date_str,
                'pnl': self.daily_pnl,
                'trades': self.daily_trades,
                'win_rate': daily_win_rate
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error backtesting {date_str}: {e}")
            return {'date': date_str, 'pnl': 0.0, 'trades': 0, 'win_rate': 0.0}
    
    def run_backtest(self, start_date: str, end_date: str):
        """
        OVERRIDE: Run turtle backtest with enhanced reporting
        Uses parent class logic for data loading and simulation
        """
        try:
            self.logger.info("🐢 TURTLE 0DTE STRATEGY BACKTEST STARTING")
            self.logger.info(f"📅 Period: {start_date} to {end_date}")
            self.logger.info(f"🎯 Daily Profit Target: ${self.params['daily_profit_target']}")
            self.logger.info(f"🛑 Daily Loss Limit: ${self.params['max_daily_loss']}")
            self.logger.info("🚫 NO TRADE LIMITS - Trading until profit/loss goals met!")
            
            # Use parent class method for getting trading dates (proven logic)
            import time as time_module
            start_time = time_module.time()
            
            # Get available dates from spy_bars subdirectory
            available_dates = []
            spy_bars_dir = os.path.join(self.cache_dir, 'spy_bars')
            if os.path.exists(spy_bars_dir):
                files = os.listdir(spy_bars_dir)
                for file in files:
                    if file.startswith('spy_bars_') and file.endswith('.pkl.gz'):
                        date_str = file.replace('spy_bars_', '').replace('.pkl.gz', '')
                        if start_date <= date_str <= end_date:
                            available_dates.append(date_str)
            
            available_dates.sort()
            self.logger.info(f"📊 Found {len(available_dates)} trading days")
            
            # Reset counters
            self.total_trades = 0
            self.winning_trades = 0
            self.daily_results = []
            
            # Backtest each day
            for date_str in available_dates:
                daily_result = self.backtest_single_day(date_str)
                self.daily_results.append(daily_result)
            
            # Calculate final results
            total_pnl = sum(day['pnl'] for day in self.daily_results)
            total_days = len(self.daily_results)
            avg_daily_pnl = total_pnl / max(total_days, 1)
            overall_win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
            profitable_days = sum(1 for day in self.daily_results if day['pnl'] > 0)
            profitable_day_rate = (profitable_days / max(total_days, 1)) * 100
            
            end_time = time_module.time()
            duration = end_time - start_time
            
            # Enhanced results reporting
            print("\n" + "="*80)
            print("🐢 TURTLE 0DTE STRATEGY - BACKTEST RESULTS")
            print("="*80)
            print(f"📅 Period: {start_date} to {end_date}")
            print(f"📊 Trading Days: {total_days}")
            print(f"⚡ Backtest Duration: {duration:.2f} seconds")
            print()
            print("💰 PERFORMANCE SUMMARY:")
            print(f"   Total P&L: ${total_pnl:.2f}")
            print(f"   Average Daily P&L: ${avg_daily_pnl:.2f}")
            print(f"   Max Daily Profit: ${max((day['pnl'] for day in self.daily_results), default=0):.2f}")
            print(f"   Max Daily Loss: ${min((day['pnl'] for day in self.daily_results), default=0):.2f}")
            print()
            print("📈 TRADE STATISTICS:")
            print(f"   Total Trades: {self.total_trades}")
            print(f"   Winning Trades: {self.winning_trades}")
            print(f"   Win Rate: {overall_win_rate:.1f}%")
            print(f"   Avg Trades/Day: {self.total_trades / max(total_days, 1):.1f}")
            print()
            print("📊 DAILY PERFORMANCE:")
            print(f"   Profitable Days: {profitable_days}/{total_days}")
            print(f"   Profitable Day Rate: {profitable_day_rate:.1f}%")
            print()
            print("🔍 TURTLE IMPROVEMENTS vs ORIGINAL:")
            print(f"   ✅ NO trade limits (removed 20-trade cap)")
            print(f"   ✅ Turtle-style breakout detection ({self.params['breakout_threshold']}% threshold)")
            print(f"   ✅ Dynamic position sizing (2-{self.params['max_position_size']} contracts)")
            print(f"   ✅ Profit/loss goal-based stopping (${self.params['daily_profit_target']}/${self.params['max_daily_loss']})")
            print(f"   ✅ Enhanced risk management ({self.params['profit_target_pct']:.0%} profit, {self.params['stop_loss_pct']:.0%} stop)")
            print("="*80)
            
            # Save results using parent class pattern
            import pickle
            results_dir = "backtrader/results"
            os.makedirs(results_dir, exist_ok=True)
            
            results_file = f"{results_dir}/turtle_0dte_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump({
                    'daily_results': self.daily_results,
                    'total_pnl': total_pnl,
                    'total_trades': self.total_trades,
                    'winning_trades': self.winning_trades,
                    'parameters': self.params,
                    'strategy_type': 'turtle_0dte'
                }, f)
            
            print(f"💾 Results saved to: {results_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Turtle backtest error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main backtest execution"""
    parser = argparse.ArgumentParser(description='Turtle 0DTE Strategy Backtest')
    parser.add_argument('--start_date', default='20240102', help='Start date (YYYYMMDD)')
    parser.add_argument('--end_date', default='20240705', help='End date (YYYYMMDD)')
    parser.add_argument('--date', help='Single date to backtest (YYYYMMDD)')
    
    args = parser.parse_args()
    
    backtest = Turtle0DTEBacktest()
    
    if args.date:
        # Single day backtest
        result = backtest.backtest_single_day(args.date)
        print(f"\n🐢 Single Day Result: {result}")
    else:
        # Multi-day backtest
        backtest.run_backtest(args.start_date, args.end_date)

if __name__ == "__main__":
    main()
