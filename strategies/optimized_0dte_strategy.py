#!/usr/bin/env python3

"""
Optimized 0DTE Strategy v2.0
============================

Based on sequential thinking analysis, this strategy fixes the critical issues:
1. INVERSE SIGNALS (original signals were 0% accurate)
2. REALISTIC option pricing (no fixed $1.60)  
3. SHORTER holding times (30-60 min vs 2 hours)
4. TIGHTER profit targets (20-30% vs 50%)
5. BETTER time decay modeling
6. DYNAMIC position sizing based on volatility
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from live_ultra_aggressive_0dte_backtest import LiveUltraAggressive0DTEBacktest

class Optimized0DTEStrategy(LiveUltraAggressive0DTEBacktest):
    """
    Optimized 0DTE Strategy with critical fixes applied
    """
    
    def __init__(self, cache_dir: str = "../thetadata/cached_data"):
        super().__init__(cache_dir)
        
        # Override with optimized parameters
        self.optimized_params = self.get_optimized_parameters()
        self.params.update(self.optimized_params)
        
        self.logger.info("🚀 OPTIMIZED 0DTE Strategy Initialized")
        self.logger.info("✅ Applied fixes: inverse signals, realistic pricing, shorter holds")

    def get_optimized_parameters(self) -> dict:
        """
        Optimized parameters based on diagnostic analysis
        """
        return {
            # CRITICAL FIX 1: Inverse signal strategy
            'use_inverse_signals': True,
            
            # CRITICAL FIX 2: Shorter holding times
            'max_position_time_minutes': 45,  # 45 minutes vs 2 hours
            'profit_target_check_interval': 5,  # Check every 5 minutes
            
            # CRITICAL FIX 3: Tighter profit targets
            'profit_target_pct': 0.25,  # 25% vs 50%
            'stop_loss_pct': 0.35,      # 35% vs 50%
            
            # CRITICAL FIX 4: Dynamic pricing
            'use_realistic_pricing': True,
            'base_option_price_range': (0.80, 2.50),  # Realistic range
            
            # CRITICAL FIX 5: Better time decay
            'realistic_time_decay': True,
            'theta_per_hour': 0.15,  # 15% per hour vs linear 90%/6hr
            
            # CRITICAL FIX 6: Improved risk management
            'volatility_position_sizing': True,
            'max_daily_trades': 12,  # Reduced from 20
            'min_time_between_signals': 120,  # 2 minutes between signals
        }

    def generate_optimized_signal(self, df: pd.DataFrame) -> dict:
        """
        Generate INVERSE signals since original signals were 0% accurate
        """
        # Get original signal
        original_signal = super().generate_trading_signal(df)
        
        if not self.params.get('use_inverse_signals', False):
            return original_signal
        
        # INVERSE the signal direction (CRITICAL FIX)
        if original_signal['signal'] == 1:  # Original CALL
            optimized_signal = -1  # Becomes PUT
            reason = f"INVERSE_{original_signal['reason']}_TO_PUT"
        elif original_signal['signal'] == -1:  # Original PUT  
            optimized_signal = 1  # Becomes CALL
            reason = f"INVERSE_{original_signal['reason']}_TO_CALL"
        else:
            optimized_signal = 0  # No signal
            reason = original_signal['reason']
        
        return {
            'signal': optimized_signal,
            'confidence': original_signal['confidence'],
            'score': original_signal['score'],
            'reason': reason,
            'spy_price': original_signal['spy_price'],
            'timestamp': original_signal['timestamp'],
            'original_signal': original_signal['signal']  # Track original for analysis
        }

    def calculate_realistic_option_price(self, spy_price: float, signal_type: str, 
                                       time_to_expiry_hours: float = 6.0) -> float:
        """
        Calculate realistic option prices instead of fixed $1.60
        """
        if not self.params.get('use_realistic_pricing', False):
            return 1.60  # Fallback to original
        
        # Base pricing factors
        min_price, max_price = self.params['base_option_price_range']
        
        # ATM/OTM pricing logic
        if signal_type == 'CALL':
            # Calls: $1 OTM, price depends on SPY level and volatility
            moneyness_factor = 0.8  # OTM discount
        else:
            # Puts: $1 OTM
            moneyness_factor = 0.8
        
        # Time decay factor (realistic)
        time_factor = max(0.1, time_to_expiry_hours / 6.0)
        
        # Volatility factor (SPY around $500-520 range)
        vol_factor = 1.0 + (spy_price - 500) / 1000  # Slight SPY level adjustment
        
        # Calculate realistic price
        realistic_price = (min_price + max_price) / 2 * moneyness_factor * time_factor * vol_factor
        
        return max(min_price, min(max_price, realistic_price))

    def calculate_realistic_time_decay(self, minutes_elapsed: int) -> float:
        """
        Realistic time decay modeling instead of aggressive linear decay
        """
        if not self.params.get('realistic_time_decay', False):
            return super()._calculate_time_decay(minutes_elapsed)
        
        # More realistic exponential decay
        hours_elapsed = minutes_elapsed / 60.0
        theta_per_hour = self.params['theta_per_hour']  # 15% per hour
        
        # Exponential decay: value = initial * e^(-theta * time)
        decay_factor = np.exp(-theta_per_hour * hours_elapsed)
        
        return max(0.05, decay_factor)  # Minimum 5% value retention

    def simulate_optimized_trade(self, signal: Dict, option_info: Dict, contracts: int) -> Dict:
        """
        Enhanced trade simulation with optimized parameters
        """
        entry_time = signal['timestamp']
        signal_type = 'CALL' if signal['signal'] == 1 else 'PUT'
        spy_price_entry = signal['spy_price']
        
        # CRITICAL FIX: Realistic entry pricing
        entry_price = self.calculate_realistic_option_price(spy_price_entry, signal_type)
        entry_cost = contracts * entry_price * 100
        
        self.logger.debug(f"🎯 OPTIMIZED Trade: {signal_type} entry=${entry_price:.2f} (vs fixed $1.60)")
        
        # Shortened simulation period (45 minutes vs 2 hours)
        max_minutes = self.params['max_position_time_minutes']
        exit_price = entry_price
        exit_reason = "TIME_LIMIT"
        last_estimated_price = entry_price
        
        # Simulate price movement in 5-minute intervals
        for minutes_elapsed in range(5, max_minutes + 1, 5):
            # Get time decay factor
            time_decay = self.calculate_realistic_time_decay(minutes_elapsed)
            
            # Simulate small SPY movement (more realistic than large swings)
            random_movement = np.random.normal(0, 0.001)  # 0.1% std dev
            spy_movement_pct = random_movement
            
            # Calculate option price change
            volatility_factor = self._calculate_volatility_effect(spy_movement_pct, signal_type)
            estimated_price = entry_price * time_decay * volatility_factor
            last_estimated_price = estimated_price  # Track the last price for TIME_LIMIT
            
            # Calculate P&L
            current_value = contracts * estimated_price * 100
            position_pnl = current_value - entry_cost
            position_pnl_pct = position_pnl / entry_cost
            
            # Debug logging for first few intervals
            if minutes_elapsed <= 15:
                self.logger.debug(f"  t+{minutes_elapsed}min: price=${estimated_price:.2f}, P&L=${position_pnl:.0f} ({position_pnl_pct:+.1%})")
            
            # CRITICAL FIX: Tighter profit targets (25% vs 50%)
            if position_pnl_pct >= self.params['profit_target_pct']:
                exit_price = estimated_price
                exit_reason = "PROFIT_TARGET"
                self.logger.debug(f"  ✅ PROFIT TARGET hit at t+{minutes_elapsed}min: +{position_pnl_pct:.1%}")
                break
            
            # CRITICAL FIX: Reasonable stop loss (35% vs 50%)  
            if position_pnl_pct <= -self.params['stop_loss_pct']:
                exit_price = estimated_price
                exit_reason = "STOP_LOSS"
                self.logger.debug(f"  🛑 STOP LOSS hit at t+{minutes_elapsed}min: {position_pnl_pct:.1%}")
                break
        
        # CRITICAL FIX: Use last estimated price for TIME_LIMIT exits
        if exit_reason == "TIME_LIMIT":
            exit_price = last_estimated_price
            self.logger.debug(f"  ⏰ TIME_LIMIT exit: final_price=${exit_price:.2f}")
        
        # Calculate final P&L
        exit_value = contracts * exit_price * 100
        final_pnl = exit_value - entry_cost
        outcome = "WIN" if final_pnl > 0 else "LOSS"
        
        return {
            'signal': signal,
            'option': option_info,
            'contracts': contracts,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_cost': entry_cost,
            'exit_value': exit_value,
            'pnl': final_pnl,
            'outcome': outcome,
            'exit_reason': exit_reason,
            'hold_time_minutes': minutes_elapsed if exit_reason != "TIME_LIMIT" else max_minutes,
            'original_signal_direction': signal.get('original_signal', 0)
        }

    def generate_signals(self, spy_bars: pd.DataFrame) -> List[Dict]:
        """
        Generate optimized signals with inverse logic
        """
        signals = []
        
        if len(spy_bars) < self.params['sma_period']:
            return signals
        
        # Resample to minute data
        spy_bars_minute = spy_bars.resample('1min').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        self.logger.info(f"📊 Resampled from {len(spy_bars):,} second bars to {len(spy_bars_minute):,} minute bars")
        
        # Calculate technical indicators
        spy_bars_minute = self.calculate_technical_indicators(spy_bars_minute.copy())
        
        # Generate signals with enhanced timing control
        for i in range(self.params['sma_period'], len(spy_bars_minute)):
            current_time = spy_bars_minute.index[i]
            
            # Skip if not market hours
            if not self.is_market_hours(current_time):
                continue
            
            # Enhanced timing control
            if self.last_signal_time is not None:
                time_since_last = (current_time - self.last_signal_time).total_seconds()
                if time_since_last < self.params['min_time_between_signals']:
                    continue
            
            # Get data for signal generation
            current_data = spy_bars_minute.iloc[:i+1].tail(50)
            
            # Generate OPTIMIZED signal (with inverse logic)
            signal_info = self.generate_optimized_signal(current_data)
            
            # Convert to backtest format if signal meets threshold
            if signal_info['signal'] != 0 and signal_info['confidence'] >= self.params['confidence_threshold']:
                signal_type = 'CALL' if signal_info['signal'] == 1 else 'PUT'
                
                # Update last signal time
                self.last_signal_time = current_time
                
                signals.append({
                    'timestamp': current_time,
                    'type': signal_type,
                    'confidence': signal_info['confidence'],
                    'spy_price': signal_info['spy_price'],
                    'signal_source': signal_info['reason'],
                    'score': signal_info['score'],
                    'original_signal': signal_info.get('original_signal', 0)
                })
                
                self.logger.info(f"🎯 OPTIMIZED Signal: {signal_type} (confidence: {signal_info['confidence']:.3f}, "
                               f"score: {signal_info['score']}, reason: {signal_info['reason']})")
        
        return signals

    def run_optimized_backtest(self, date: str) -> Dict:
        """
        Run optimized backtest with all fixes applied
        """
        self.logger.info(f"🚀 Running OPTIMIZED backtest for {date}")
        self.logger.info(f"✅ Fixes: inverse signals, realistic pricing, 45min holds, 25% targets")
        
        # Reset daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_signal_time = None
        day_trades = []
        
        # Load cached data
        data = self.load_cached_data(date)
        if not data:
            return {'date': date, 'trades': 0, 'pnl': 0, 'error': 'No data'}
        
        spy_bars = data['spy_bars']
        option_chain = data['option_chain']
        
        # Generate optimized signals
        signals = self.generate_signals(spy_bars)
        self.logger.info(f"📊 Generated {len(signals)} OPTIMIZED signals for {date}")
        
        # Process each signal with optimized logic
        for signal in signals:
            # Check daily limits
            if self.daily_pnl <= -self.params['max_daily_loss']:
                self.logger.info(f"🛑 Daily loss limit reached: ${self.daily_pnl:.2f}")
                break
            
            if self.daily_trades >= self.params['max_daily_trades']:
                self.logger.info(f"📈 Daily trade limit reached: {self.daily_trades}")
                break
            
            # Calculate optimized position size
            contracts = self.calculate_position_size(signal['confidence'])
            if contracts <= 0:
                continue
            
            # Convert signal format for simulation
            signal_dict = {
                'signal': 1 if signal['type'] == 'CALL' else -1,
                'timestamp': signal['timestamp'],
                'spy_price': signal['spy_price'],
                'confidence': signal['confidence']
            }
            
            # Simulate optimized trade
            trade_result = self.simulate_optimized_trade(signal_dict, {}, contracts)
            
            if trade_result:
                day_trades.append(trade_result)
                self.daily_pnl += trade_result['pnl']
                self.daily_trades += 1
                
                if trade_result['outcome'] == 'WIN':
                    self.winning_trades += 1
                
                self.logger.info(f"📈 Trade #{self.daily_trades}: {trade_result['outcome']} - "
                               f"${trade_result['pnl']:.2f} ({trade_result['exit_reason']}) | Daily P&L: ${self.daily_pnl:.2f}")
        
        # Calculate results
        win_rate = (self.winning_trades / max(1, self.daily_trades)) * 100
        
        self.logger.info(f"✅ OPTIMIZED Day complete: {self.daily_trades} trades, ${self.daily_pnl:.2f} P&L, {win_rate:.1f}% win rate")
        
        return {
            'date': date,
            'trades': self.daily_trades,
            'pnl': self.daily_pnl,
            'signals_generated': len(signals),
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'strategy_version': 'OPTIMIZED_v2.0'
        }

def main():
    """Run optimized strategy backtest"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized 0DTE Strategy Backtest')
    parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    args = parser.parse_args()
    
    # Run optimized backtest
    strategy = Optimized0DTEStrategy()
    result = strategy.run_optimized_backtest(args.date)
    
    print(f"✅ OPTIMIZED Result: {result}")

if __name__ == "__main__":
    main() 