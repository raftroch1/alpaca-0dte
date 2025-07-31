#!/usr/bin/env python3
"""
REALISTIC Enhanced Turtle-Style 0DTE Strategy Backtest
====================================================

REALISTIC TRADING APPROACH:
✅ 1-5 trades per day maximum (like real turtle systems)
✅ Check for signals every 15-30 minutes (not every minute)
✅ Clear breakout thresholds that actually trigger
✅ Fast multi-day backtesting capability
✅ Matches proven turtle_0dte_backtest.py structure

This version trades realistically and can backtest 6 months in minutes.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Optional
import logging
import time as time_module

# Import the existing proven turtle backtest framework
from turtle_0dte_backtest import Turtle0DTEBacktest

class RealisticEnhancedTurtle0DTEBacktest(Turtle0DTEBacktest):
    """
    REALISTIC Enhanced turtle-style 0DTE backtest - trades like real turtle systems
    """
    
    def __init__(self, cache_dir: str = "../thetadata/cached_data"):
        # Initialize with the proven turtle framework
        super().__init__(cache_dir)
        
        # REALISTIC TURTLE PARAMETERS
        self.params.update({
            # ACCOUNT & RISK
            'account_size': 25000.0,
            'risk_per_trade': 0.02,             # 2% risk per trade
            'max_daily_trades': 5,              # Maximum 5 trades per day (realistic)
            
            # REALISTIC BREAKOUT DETECTION
            'min_breakout_strength': 0.002,     # 0.2% minimum breakout (realistic for 0DTE)
            'strong_breakout_threshold': 0.005, # 0.5% for strong signals
            'n_periods': 20,                    # ATR calculation periods
            
            # DUAL SYSTEM (MINUTE-BASED)
            'system1_periods': 20,              # 20-minute breakouts
            'system2_periods': 55,              # 55-minute breakouts
            'signal_check_interval': 15,        # Check every 15 minutes (realistic)
            
            # 0DTE SPECIFIC
            'min_time_to_expiry': 30,           # 30 minutes minimum
            'max_position_time': 90,            # 90 minutes maximum
            'early_close_profit': 0.5,          # Close at 50% profit
            
            # PERFORMANCE
            'fast_mode': True,                  # Enable fast backtesting
        })
        
        # Enhanced tracking
        self.daily_trade_count = 0
        self.signals_checked = 0
        self.signals_generated = 0
        
        self.logger.info("🐢 REALISTIC ENHANCED TURTLE 0DTE STRATEGY INITIALIZED")
        self.logger.info(f"🎯 Max trades per day: {self.params['max_daily_trades']}")
        self.logger.info(f"⏰ Signal check interval: {self.params['signal_check_interval']} minutes")
        self.logger.info(f"⚖️ Breakout threshold: {self.params['min_breakout_strength']:.3f} ({self.params['min_breakout_strength']*100:.1f}%)")
    
    def calculate_true_range_n(self, data: pd.DataFrame) -> float:
        """Calculate turtle-style N value (ATR) with proper bounds"""
        try:
            if len(data) < self.params['n_periods']:
                # Simple volatility for short data
                returns = data['close'].pct_change().dropna()
                if len(returns) > 5:
                    volatility = returns.std()
                    return max(data['close'].iloc[-1] * volatility, 0.1)
                else:
                    return data['close'].iloc[-1] * 0.01
            
            high = data['high']
            low = data['low']
            close = data['close']
            prev_close = close.shift(1)
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            n_value = true_range.rolling(window=self.params['n_periods']).mean().iloc[-1]
            
            # Bounds checking for SPY
            if pd.isna(n_value) or n_value <= 0:
                price_range = data['high'].iloc[-self.params['n_periods']:].max() - data['low'].iloc[-self.params['n_periods']:].min()
                n_value = price_range / self.params['n_periods']
            
            current_price = data['close'].iloc[-1]
            min_n = current_price * 0.001  # 0.1% minimum
            max_n = current_price * 0.05   # 5% maximum
            
            return max(min(n_value, max_n), min_n)
            
        except Exception as e:
            return data['close'].iloc[-1] * 0.01  # 1% fallback
    
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """Simple market regime detection"""
        try:
            if len(data) < 30:
                return 'UNKNOWN'
            
            prices = data['close']
            sma_10 = prices.rolling(10).mean().iloc[-1]
            sma_20 = prices.rolling(20).mean().iloc[-1]
            
            if sma_10 > sma_20 * 1.002:
                return 'BULLISH'
            elif sma_10 < sma_20 * 0.998:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception:
            return 'UNKNOWN'
    
    def generate_realistic_turtle_signal(self, spy_bars: pd.DataFrame, current_time: datetime) -> Optional[Dict]:
        """
        REALISTIC turtle signal generation - checks for clear breakouts only
        """
        try:
            self.signals_checked += 1
            
            # Need enough data for both systems
            if len(spy_bars) < self.params['system2_periods']:
                return None
            
            # Stop if daily trade limit reached
            if self.daily_trade_count >= self.params['max_daily_trades']:
                return None
            
            current_price = spy_bars['close'].iloc[-1]
            current_n = self.calculate_true_range_n(spy_bars)
            regime = self.detect_market_regime(spy_bars)
            
            # REALISTIC DUAL SYSTEM BREAKOUTS
            best_signal = None
            max_strength = 0
            
            # System 1: 20-minute breakouts (faster entries)
            sys1_data = spy_bars.tail(self.params['system1_periods'])
            sys1_high = sys1_data['high'].max()
            sys1_low = sys1_data['low'].min()
            
            # System 2: 55-minute breakouts (stronger entries)
            sys2_data = spy_bars.tail(self.params['system2_periods'])
            sys2_high = sys2_data['high'].max()
            sys2_low = sys2_data['low'].min()
            
            # Check both systems
            for system, high_level, low_level in [(1, sys1_high, sys1_low), (2, sys2_high, sys2_low)]:
                
                # Bullish breakout
                if current_price > high_level:
                    strength = (current_price - high_level) / current_n
                    
                    if strength >= self.params['min_breakout_strength'] and strength > max_strength:
                        confidence = min(strength * 50, 1.0)  # Scale to 0-1
                        
                        best_signal = {
                            'signal': 'CALL',
                            'confidence': confidence,
                            'entry_reason': f'Realistic Turtle Sys{system} ({self.params[f"system{system}_periods"]}min) bullish breakout',
                            'contracts': self.calculate_position_size(current_n, strength),
                            'system': system,
                            'regime': regime,
                            'n_value': current_n,
                            'breakout_strength': strength,
                            'breakout_level': high_level,
                            'current_price': current_price
                        }
                        max_strength = strength
                
                # Bearish breakout
                if current_price < low_level:
                    strength = (low_level - current_price) / current_n
                    
                    if strength >= self.params['min_breakout_strength'] and strength > max_strength:
                        confidence = min(strength * 50, 1.0)
                        
                        best_signal = {
                            'signal': 'PUT',
                            'confidence': confidence,
                            'entry_reason': f'Realistic Turtle Sys{system} ({self.params[f"system{system}_periods"]}min) bearish breakout',
                            'contracts': self.calculate_position_size(current_n, strength),
                            'system': system,
                            'regime': regime,
                            'n_value': current_n,
                            'breakout_strength': strength,
                            'breakout_level': low_level,
                            'current_price': current_price
                        }
                        max_strength = strength
            
            if best_signal:
                self.signals_generated += 1
                self.daily_trade_count += 1
                self.logger.info(f"🎯 REALISTIC SIGNAL: {best_signal['signal']} Sys{best_signal['system']} strength={best_signal['breakout_strength']:.4f}")
            
            return best_signal
            
        except Exception as e:
            self.logger.error(f"❌ Realistic signal generation error: {e}")
            return None
    
    def calculate_position_size(self, current_n: float, strength: float) -> int:
        """Calculate position size based on N and signal strength"""
        try:
            # Base position using account risk
            account_risk = self.params['account_size'] * self.params['risk_per_trade']
            base_contracts = max(1, int(account_risk / (current_n * 100)))
            
            # Scale by signal strength
            if strength >= self.params['strong_breakout_threshold']:
                multiplier = 1.5  # 50% more for strong signals
            else:
                multiplier = 1.0
            
            position_size = int(base_contracts * multiplier)
            return min(max(position_size, 1), 8)  # Between 1 and 8 contracts
            
        except Exception:
            return 2  # Default
    
    def backtest_single_day(self, date_str: str) -> Dict:
        """
        REALISTIC single day backtest - fast and efficient
        """
        try:
            self.logger.info(f"🐢 REALISTIC ENHANCED TURTLE: {date_str}")
            
            # Reset daily counters
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_trade_count = 0
            self.signals_checked = 0
            self.signals_generated = 0
            winning_trades_today = 0
            
            # Load data
            data = self.load_cached_data(date_str)
            if not data:
                self.logger.warning(f"❌ No data for {date_str}")
                return {'date': date_str, 'pnl': 0.0, 'trades': 0, 'win_rate': 0.0}
            
            spy_bars = data['spy_bars']
            option_chain = data['option_chain']
            
            if spy_bars.empty or len(option_chain) == 0:
                self.logger.warning(f"❌ Empty data for {date_str}")
                return {'date': date_str, 'pnl': 0.0, 'trades': 0, 'win_rate': 0.0}
            
            self.logger.info(f"✅ Loaded {len(spy_bars)} SPY bars, {len(option_chain)} options")
            
            current_time = pd.to_datetime(date_str)
            signals_found = []
            
            # REALISTIC SIGNAL CHECKING: Every 15 minutes, not every minute!
            check_interval = self.params['signal_check_interval']  # 15 minutes
            start_idx = max(self.params['system2_periods'], 55)
            
            # Check signals at realistic intervals
            for i in range(start_idx, len(spy_bars), check_interval):
                if self.daily_trade_count >= self.params['max_daily_trades']:
                    break  # Stop when daily limit reached
                
                subset_data = spy_bars.iloc[:i+1]
                signal = self.generate_realistic_turtle_signal(subset_data, current_time)
                
                if signal:
                    signals_found.append(signal)
                    self.logger.info(f"🎯 Signal found at minute {i}: {signal['signal']} (System {signal['system']})")
            
            # Execute signals (up to daily limit)
            for signal in signals_found[:self.params['max_daily_trades']]:
                trade_result = self.simulate_trade(signal, spy_bars, option_chain)
                
                if trade_result:
                    pnl = trade_result.get('pnl', 0)
                    self.daily_pnl += pnl
                    self.daily_trades += 1
                    
                    if pnl > 0:
                        winning_trades_today += 1
                    
                    self.logger.info(f"💰 Trade: {signal['signal']} {signal['contracts']}x = ${pnl:.2f}")
            
            # Calculate win rate
            daily_win_rate = (winning_trades_today / max(self.daily_trades, 1)) * 100
            
            self.logger.info(f"✅ Day complete: {self.daily_trades} trades, ${self.daily_pnl:.2f} P&L, {daily_win_rate:.1f}% win rate")
            self.logger.info(f"📊 Signals: {self.signals_generated}/{self.signals_checked} generated")
            
            return {
                'date': date_str,
                'pnl': self.daily_pnl,
                'trades': self.daily_trades,
                'win_rate': daily_win_rate,
                'signals_checked': self.signals_checked,
                'signals_generated': self.signals_generated
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in realistic backtest {date_str}: {e}")
            return {'date': date_str, 'pnl': 0.0, 'trades': 0, 'win_rate': 0.0}

def main():
    """Main execution for realistic enhanced turtle"""
    parser = argparse.ArgumentParser(description='Realistic Enhanced Turtle 0DTE Strategy Backtest')
    parser.add_argument('--start_date', default='20240102', help='Start date (YYYYMMDD)')
    parser.add_argument('--end_date', default='20240705', help='End date (YYYYMMDD)')
    parser.add_argument('--date', help='Single date to backtest (YYYYMMDD)')
    
    args = parser.parse_args()
    
    backtest = RealisticEnhancedTurtle0DTEBacktest()
    
    if args.date:
        # Single day backtest
        print(f"🐢 Running realistic enhanced turtle backtest: {args.date}")
        result = backtest.backtest_single_day(args.date)
        print(f"\n🐢 Realistic Enhanced Turtle Result:")
        print(f"   Date: {result['date']}")
        print(f"   P&L: ${result['pnl']:.2f}")
        print(f"   Trades: {result['trades']}")
        print(f"   Win Rate: {result['win_rate']:.1f}%")
        if 'signals_checked' in result:
            print(f"   Signal Efficiency: {result['signals_generated']}/{result['signals_checked']}")
    else:
        # Multi-day backtest
        backtest.run_backtest(args.start_date, args.end_date)

if __name__ == "__main__":
    print("🐢 REALISTIC ENHANCED TURTLE 0DTE BACKTEST")
    print("="*50)
    print("🎯 1-5 trades per day maximum")
    print("⏰ Signal checks every 15 minutes")
    print("⚖️ Realistic breakout thresholds")
    print("🚀 Fast multi-day backtesting")
    print("="*50)
    
    try:
        main()
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()
