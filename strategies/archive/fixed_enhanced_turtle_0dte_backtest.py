#!/usr/bin/env python3
"""
FIXED Enhanced Turtle-Style 0DTE Strategy Backtest
=================================================

DEBUGGING FIXES APPLIED:
✅ Adjusted breakout thresholds to realistic levels
✅ Added comprehensive debug logging
✅ Fixed N calculation with proper bounds checking
✅ Simplified dual system to focus on working signals
✅ Added signal strength analysis
✅ Improved historical data handling

This version will generate trades and show detailed debugging info.
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

class FixedEnhancedTurtle0DTEBacktest(Turtle0DTEBacktest):
    """
    FIXED Enhanced turtle-style 0DTE backtest with debugging and realistic thresholds
    """
    
    def __init__(self, cache_dir: str = "../thetadata/cached_data"):
        # Initialize with the proven turtle framework
        super().__init__(cache_dir)
        
        # FIXED: More realistic parameters for 0DTE trading
        self.params.update({
            # ACCOUNT & RISK
            'account_size': 25000.0,
            'risk_per_trade': 0.02,             # 2% risk per trade (more aggressive for 0DTE)
            'max_portfolio_risk': 0.10,         # 10% total portfolio risk
            
            # BREAKOUT DETECTION - FIXED THRESHOLDS
            'min_breakout_strength': 0.001,     # 0.1% minimum breakout (much more realistic)
            'strong_breakout_threshold': 0.005, # 0.5% for strong signals
            'n_periods': 20,                    # ATR calculation periods
            
            # SIMPLIFIED DUAL SYSTEM
            'system1_periods': 20,              # 20-minute breakouts
            'system2_periods': 50,              # 50-minute breakouts (reduced from 55)
            'use_both_systems': True,           # Enable both systems
            
            # 0DTE SPECIFIC
            'min_time_to_expiry': 30,           # 30 minutes minimum
            'max_position_time': 90,            # 90 minutes maximum
            'volatility_filter': False,         # Disable for now to see all signals
            
            # DEBUG - OPTIMIZED FOR SPEED
            'debug_signals': False,             # Disable detailed signal debugging for speed
            'log_every_n_bars': 1000,          # Log every 1000 bars for summary only
        })
        
        # Enhanced tracking with debugging
        self.signal_attempts = 0
        self.signal_generated = 0
        self.breakout_attempts = {}
        self.n_value_history = []
        
        # Set up enhanced logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger.info("🐢 FIXED ENHANCED TURTLE 0DTE STRATEGY INITIALIZED")
        self.logger.info(f"🔧 DEBUG MODE: Detailed signal analysis enabled")
        self.logger.info(f"⚖️ Breakout threshold: {self.params['min_breakout_strength']:.3f} ({self.params['min_breakout_strength']*100:.2f}%)")
    
    def calculate_true_range_n(self, data: pd.DataFrame) -> float:
        """FIXED: Calculate turtle-style N value with proper bounds checking"""
        try:
            if len(data) < self.params['n_periods']:
                # Use simple volatility estimate for short data
                returns = data['close'].pct_change().dropna()
                if len(returns) > 5:
                    volatility = returns.std()
                    n_estimate = data['close'].iloc[-1] * volatility
                    self.logger.debug(f"🔧 Short data N estimate: {n_estimate:.4f}")
                    return max(n_estimate, 0.01)  # Minimum N value
                else:
                    return data['close'].iloc[-1] * 0.01  # 1% default
            
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
            
            # FIXED: Bounds checking and validation
            if pd.isna(n_value) or n_value <= 0:
                # Fallback calculation
                price_range = data['high'].iloc[-self.params['n_periods']:].max() - data['low'].iloc[-self.params['n_periods']:].min()
                n_value = price_range / self.params['n_periods']
                self.logger.debug(f"🔧 Fallback N calculation: {n_value:.4f}")
            
            # Reasonable bounds for SPY
            current_price = data['close'].iloc[-1]
            min_n = current_price * 0.001  # 0.1% minimum
            max_n = current_price * 0.05   # 5% maximum
            n_value = max(min(n_value, max_n), min_n)
            
            self.n_value_history.append(n_value)
            if len(self.n_value_history) > 100:
                self.n_value_history = self.n_value_history[-100:]  # Keep last 100
            
            return n_value
            
        except Exception as e:
            self.logger.warning(f"🔧 N calculation error: {e}")
            fallback_n = data['close'].iloc[-1] * 0.01
            self.logger.debug(f"🔧 Using fallback N: {fallback_n:.4f}")
            return fallback_n
    
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """SIMPLIFIED: Market regime detection"""
        try:
            if len(data) < 30:
                return 'UNKNOWN'
            
            prices = data['close']
            
            # Simple trend detection
            sma_10 = prices.rolling(10).mean().iloc[-1]
            sma_20 = prices.rolling(20).mean().iloc[-1]
            
            if sma_10 > sma_20 * 1.001:
                return 'BULLISH'
            elif sma_10 < sma_20 * 0.999:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception:
            return 'UNKNOWN'
    
    def generate_enhanced_turtle_signal(self, spy_bars: pd.DataFrame, current_time: datetime) -> Optional[Dict]:
        """
        FIXED: Enhanced turtle signal generation with comprehensive debugging
        """
        try:
            self.signal_attempts += 1
            
            # Minimum data requirement
            min_required = max(self.params['system1_periods'], self.params['system2_periods'])
            if len(spy_bars) < min_required:
                if self.params['debug_signals']:
                    self.logger.debug(f"🔧 Insufficient data: {len(spy_bars)} < {min_required}")
                return None
            
            current_price = spy_bars['close'].iloc[-1]
            current_n = self.calculate_true_range_n(spy_bars)
            regime = self.detect_market_regime(spy_bars)
            
            # Minimal debug logging for speed
            if len(spy_bars) % self.params['log_every_n_bars'] == 0:
                self.logger.info(f"📊 Progress: {len(spy_bars)} bars, {self.signal_attempts} attempts, {self.signal_generated} signals")
            
            best_signal = None
            max_strength = 0
            
            # SYSTEM 1: Short-term breakouts (20 periods)
            if len(spy_bars) >= self.params['system1_periods']:
                sys1_data = spy_bars.tail(self.params['system1_periods'])
                sys1_high = sys1_data['high'].max()
                sys1_low = sys1_data['low'].min()
                
                # Check bullish breakout
                if current_price > sys1_high:
                    strength = (current_price - sys1_high) / current_n
                    
                    if self.params['debug_signals']:
                        self.logger.info(f"🔧 SYS1 BULL: Price {current_price:.2f} > High {sys1_high:.2f}")
                        self.logger.info(f"🔧 Strength: {strength:.6f} (threshold: {self.params['min_breakout_strength']:.6f})")
                    
                    if strength >= self.params['min_breakout_strength'] and strength > max_strength:
                        confidence = min(strength * 100, 1.0)  # Convert to 0-1 scale
                        
                        best_signal = {
                            'signal': 'CALL',
                            'confidence': confidence,
                            'entry_reason': f'System 1 (20-period) bullish breakout - strength {strength:.4f}',
                            'contracts': self.calculate_position_size(current_n, strength),
                            'system': 1,
                            'regime': regime,
                            'n_value': current_n,
                            'breakout_strength': strength,
                            'breakout_level': sys1_high,
                            'current_price': current_price
                        }
                        max_strength = strength
                        
                        if self.params['debug_signals']:
                            self.logger.info(f"✅ SYS1 BULL SIGNAL: {best_signal}")
                
                # Check bearish breakout
                if current_price < sys1_low:
                    strength = (sys1_low - current_price) / current_n
                    
                    if self.params['debug_signals']:
                        self.logger.info(f"🔧 SYS1 BEAR: Price {current_price:.2f} < Low {sys1_low:.2f}")
                        self.logger.info(f"🔧 Strength: {strength:.6f} (threshold: {self.params['min_breakout_strength']:.6f})")
                    
                    if strength >= self.params['min_breakout_strength'] and strength > max_strength:
                        confidence = min(strength * 100, 1.0)
                        
                        best_signal = {
                            'signal': 'PUT',
                            'confidence': confidence,
                            'entry_reason': f'System 1 (20-period) bearish breakout - strength {strength:.4f}',
                            'contracts': self.calculate_position_size(current_n, strength),
                            'system': 1,
                            'regime': regime,
                            'n_value': current_n,
                            'breakout_strength': strength,
                            'breakout_level': sys1_low,
                            'current_price': current_price
                        }
                        max_strength = strength
                        
                        if self.params['debug_signals']:
                            self.logger.info(f"✅ SYS1 BEAR SIGNAL: {best_signal}")
            
            # SYSTEM 2: Longer-term breakouts (50 periods)
            if self.params['use_both_systems'] and len(spy_bars) >= self.params['system2_periods']:
                sys2_data = spy_bars.tail(self.params['system2_periods'])
                sys2_high = sys2_data['high'].max()
                sys2_low = sys2_data['low'].min()
                
                # Check bullish breakout
                if current_price > sys2_high:
                    strength = (current_price - sys2_high) / current_n
                    
                    if self.params['debug_signals']:
                        self.logger.info(f"🔧 SYS2 BULL: Price {current_price:.2f} > High {sys2_high:.2f}")
                        self.logger.info(f"🔧 Strength: {strength:.6f}")
                    
                    if strength >= self.params['min_breakout_strength'] and strength > max_strength:
                        confidence = min(strength * 100, 1.0)
                        
                        best_signal = {
                            'signal': 'CALL',
                            'confidence': confidence,
                            'entry_reason': f'System 2 (50-period) bullish breakout - strength {strength:.4f}',
                            'contracts': self.calculate_position_size(current_n, strength),
                            'system': 2,
                            'regime': regime,
                            'n_value': current_n,
                            'breakout_strength': strength,
                            'breakout_level': sys2_high,
                            'current_price': current_price
                        }
                        max_strength = strength
                        
                        if self.params['debug_signals']:
                            self.logger.info(f"✅ SYS2 BULL SIGNAL: {best_signal}")
                
                # Check bearish breakout
                if current_price < sys2_low:
                    strength = (sys2_low - current_price) / current_n
                    
                    if self.params['debug_signals']:
                        self.logger.info(f"🔧 SYS2 BEAR: Price {current_price:.2f} < Low {sys2_low:.2f}")
                        self.logger.info(f"🔧 Strength: {strength:.6f}")
                    
                    if strength >= self.params['min_breakout_strength'] and strength > max_strength:
                        confidence = min(strength * 100, 1.0)
                        
                        best_signal = {
                            'signal': 'PUT',
                            'confidence': confidence,
                            'entry_reason': f'System 2 (50-period) bearish breakout - strength {strength:.4f}',
                            'contracts': self.calculate_position_size(current_n, strength),
                            'system': 2,
                            'regime': regime,
                            'n_value': current_n,
                            'breakout_strength': strength,
                            'breakout_level': sys2_low,
                            'current_price': current_price
                        }
                        max_strength = strength
                        
                        if self.params['debug_signals']:
                            self.logger.info(f"✅ SYS2 BEAR SIGNAL: {best_signal}")
            
            # Final signal validation
            if best_signal:
                self.signal_generated += 1
                self.logger.info(f"🎯 SIGNAL GENERATED! System {best_signal['system']}, {best_signal['signal']}, Strength: {best_signal['breakout_strength']:.4f}")
                
                # Store breakout attempt for analysis
                attempt_key = f"{best_signal['system']}_{best_signal['signal']}"
                if attempt_key not in self.breakout_attempts:
                    self.breakout_attempts[attempt_key] = 0
                self.breakout_attempts[attempt_key] += 1
            
            # Periodic summary
            if self.signal_attempts % 100 == 0:
                self.logger.info(f"📊 SIGNAL SUMMARY: {self.signal_generated}/{self.signal_attempts} signals generated ({self.signal_generated/self.signal_attempts*100:.1f}%)")
                if self.n_value_history:
                    avg_n = sum(self.n_value_history) / len(self.n_value_history)
                    self.logger.info(f"📊 Average N: {avg_n:.4f}")
            
            return best_signal
            
        except Exception as e:
            self.logger.error(f"❌ Signal generation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_position_size(self, current_n: float, strength: float) -> int:
        """Calculate position size based on N and signal strength"""
        try:
            # Base position sizing using account risk
            account_risk = self.params['account_size'] * self.params['risk_per_trade']
            base_contracts = max(1, int(account_risk / (current_n * 100)))  # Assume $100 per point
            
            # Scale by signal strength
            if strength >= self.params['strong_breakout_threshold']:
                multiplier = 1.5  # 50% more for strong signals
            else:
                multiplier = 1.0
            
            position_size = int(base_contracts * multiplier)
            return min(max(position_size, 1), 10)  # Between 1 and 10 contracts
            
        except Exception:
            return 2  # Default position size
    
    def backtest_single_day(self, date_str: str) -> Dict:
        """
        FIXED: Single day backtest with enhanced debugging
        """
        try:
            self.logger.info(f"🐢 FIXED ENHANCED TURTLE BACKTEST: {date_str}")
            
            # Reset daily tracking
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.signal_attempts = 0
            self.signal_generated = 0
            winning_trades_today = 0
            
            # Load data using parent class method
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
            
            # Generate signal (simplified - use the spy_bars directly)
            current_time = pd.to_datetime(date_str)
            
            # Try multiple time points during the day for signals
            signals_found = []
            
            # Check every 100 bars for signals (faster)
            for i in range(max(self.params['system2_periods'], 50), len(spy_bars), 100):
                subset_data = spy_bars.iloc[:i+1]
                signal = self.generate_enhanced_turtle_signal(subset_data, current_time)
                
                if signal:
                    signals_found.append(signal)
                    self.logger.info(f"🎯 Found signal at bar {i}: {signal['signal']} (System {signal['system']})")
            
            # Execute the first valid signal
            if signals_found:
                signal = signals_found[0]
                
                # Simulate trade using parent class method
                trade_result = self.simulate_trade(signal, spy_bars, option_chain)
                
                if trade_result:
                    pnl = trade_result.get('pnl', 0)
                    self.daily_pnl += pnl
                    self.daily_trades += 1
                    
                    if pnl > 0:
                        winning_trades_today += 1
                    
                    self.logger.info(f"💰 Trade executed: {signal['signal']} {signal['contracts']} contracts = ${pnl:.2f}")
                    self.logger.info(f"🔧 Signal details: System {signal['system']}, Strength {signal['breakout_strength']:.4f}, N={signal['n_value']:.4f}")
            else:
                self.logger.info(f"📊 No signals generated on {date_str}")
                self.logger.info(f"📊 Signal attempts: {self.signal_attempts}")
                
                # Show some sample breakout analysis
                if len(spy_bars) >= 50:
                    sample_data = spy_bars.tail(50)
                    current_price = sample_data['close'].iloc[-1]
                    current_n = self.calculate_true_range_n(sample_data)
                    high_20 = sample_data['high'].tail(20).max()
                    low_20 = sample_data['low'].tail(20).min()
                    
                    bull_strength = (current_price - high_20) / current_n if current_price > high_20 else 0
                    bear_strength = (low_20 - current_price) / current_n if current_price < low_20 else 0
                    
                    self.logger.info(f"🔧 Sample analysis: Price={current_price:.2f}, N={current_n:.4f}")
                    self.logger.info(f"🔧 20-day High={high_20:.2f}, Low={low_20:.2f}")
                    self.logger.info(f"🔧 Bull strength={bull_strength:.6f}, Bear strength={bear_strength:.6f}")
                    self.logger.info(f"🔧 Threshold={self.params['min_breakout_strength']:.6f}")
            
            # Calculate win rate
            daily_win_rate = (winning_trades_today / max(self.daily_trades, 1)) * 100
            
            self.logger.info(f"✅ Day complete: {self.daily_trades} trades, ${self.daily_pnl:.2f} P&L, {daily_win_rate:.1f}% win rate")
            
            return {
                'date': date_str,
                'pnl': self.daily_pnl,
                'trades': self.daily_trades,
                'win_rate': daily_win_rate,
                'signals_attempted': self.signal_attempts,
                'signals_generated': self.signal_generated
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in backtest {date_str}: {e}")
            import traceback
            traceback.print_exc()
            return {'date': date_str, 'pnl': 0.0, 'trades': 0, 'win_rate': 0.0}

def main():
    """Main execution with enhanced debugging"""
    parser = argparse.ArgumentParser(description='Fixed Enhanced Turtle 0DTE Strategy Backtest')
    parser.add_argument('--start_date', default='20240102', help='Start date (YYYYMMDD)')
    parser.add_argument('--end_date', default='20240705', help='End date (YYYYMMDD)')
    parser.add_argument('--date', help='Single date to backtest (YYYYMMDD)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    backtest = FixedEnhancedTurtle0DTEBacktest()
    
    # Enable debug mode if requested
    if args.debug:
        backtest.params['debug_signals'] = True
        backtest.params['log_every_n_bars'] = 10
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.date:
        # Single day backtest with detailed analysis
        print(f"🔧 Running single day backtest with debugging: {args.date}")
        result = backtest.backtest_single_day(args.date)
        print(f"\n🐢 Fixed Enhanced Turtle Single Day Result:")
        print(f"   Date: {result['date']}")
        print(f"   P&L: ${result['pnl']:.2f}")
        print(f"   Trades: {result['trades']}")
        print(f"   Win Rate: {result['win_rate']:.1f}%")
        if 'signals_attempted' in result:
            print(f"   Signal Generation: {result['signals_generated']}/{result['signals_attempted']}")
    else:
        # Multi-day backtest
        backtest.run_backtest(args.start_date, args.end_date)

if __name__ == "__main__":
    print("🐢 FIXED ENHANCED TURTLE 0DTE BACKTEST")
    print("="*50)
    print("🔧 DEBUGGING ENABLED")
    print("⚖️ Realistic breakout thresholds")
    print("📊 Comprehensive signal analysis")
    print("="*50)
    
    try:
        main()
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()
