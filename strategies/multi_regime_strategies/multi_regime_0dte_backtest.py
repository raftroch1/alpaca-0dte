#!/usr/bin/env python3
"""
Multi-Regime 0DTE Options Strategy Backtest

Extends the proven live_ultra_aggressive_0dte_backtest.py framework
to implement sophisticated multi-regime options strategies.

Strategy Overview:
- High VIX → Iron Condor (volatility contraction)
- Low VIX → Diagonal Spread (time decay + directional)
- Moderate VIX + Bullish → Put Credit Spread
- Moderate VIX + Bearish → Call Credit Spread
- Moderate VIX + Neutral → Iron Butterfly
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, Tuple, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import proven backtest framework
from strategies.live_ultra_aggressive_0dte_backtest import LiveUltraAggressive0DTEBacktest


class MultiRegime0DTEBacktest(LiveUltraAggressive0DTEBacktest):
    """Multi-Regime 0DTE Options Strategy Backtest"""
    
    def __init__(self, cache_dir: str = "./thetadata/cached_data"):
        super().__init__(cache_dir)
        
        # Enhanced VIX thresholds
        self.low_vol_threshold = 15.0
        self.high_vol_threshold = 25.0
        
        # Risk management
        self.max_risk_per_trade = 0.02
        self.profit_target_pct = 0.50
        self.stop_loss_pct = 2.00
        
        # Daily limits
        self.daily_profit_target = 800.0
        self.daily_loss_limit = 400.0
        
        # Strategy performance tracking
        self.strategy_performance = {
            'IRON_CONDOR': {'trades': 0, 'pnl': 0, 'wins': 0},
            'DIAGONAL': {'trades': 0, 'pnl': 0, 'wins': 0},
            'PUT_CREDIT_SPREAD': {'trades': 0, 'pnl': 0, 'wins': 0},
            'CALL_CREDIT_SPREAD': {'trades': 0, 'pnl': 0, 'wins': 0},
            'IRON_BUTTERFLY': {'trades': 0, 'pnl': 0, 'wins': 0},
        }
        
        self.logger.info("🏛️ Multi-Regime 0DTE Backtest initialized")

    def simulate_vix(self, date: str) -> float:
        """Simulate VIX based on date patterns"""
        date_obj = datetime.strptime(date, '%Y%m%d')
        base_vix = 18.0
        
        # Seasonal variation
        day_of_year = date_obj.timetuple().tm_yday
        seasonal = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Deterministic randomness
        np.random.seed(int(date))
        stress = 1 + np.random.normal(0, 0.2)
        
        vix = base_vix * seasonal * stress
        return max(10.0, min(50.0, vix))

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI for momentum analysis"""
        if len(prices) < period + 1:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def calculate_momentum_score(self, spy_bars: pd.DataFrame, current_time: datetime) -> float:
        """Calculate momentum score (-1 to 1)"""
        try:
            recent_bars = spy_bars[spy_bars.index <= current_time].tail(20)
            
            if len(recent_bars) < 10:
                return 0.0
                
            # RSI momentum
            rsi = self.calculate_rsi(recent_bars['close'])
            rsi_score = (rsi - 50) / 50
            
            # Moving average momentum
            short_ma = recent_bars['close'].tail(5).mean()
            long_ma = recent_bars['close'].tail(15).mean()
            ma_score = np.tanh((short_ma / long_ma - 1) * 100)
            
            # Combined score
            momentum = 0.6 * rsi_score + 0.4 * ma_score
            return max(-1.0, min(1.0, momentum))
            
        except Exception as e:
            self.logger.warning(f"Error calculating momentum: {e}")
            return 0.0

    def analyze_market_regime(self, date: str, spy_bars: pd.DataFrame, 
                            current_time: datetime) -> Tuple[str, Dict]:
        """Analyze market regime for strategy selection"""
        vix = self.simulate_vix(date)
        momentum = self.calculate_momentum_score(spy_bars, current_time)
        spy_price = spy_bars[spy_bars.index <= current_time]['close'].iloc[-1]
        
        # Determine regimes
        if vix < self.low_vol_threshold:
            vix_regime = "LOW"
        elif vix > self.high_vol_threshold:
            vix_regime = "HIGH"
        else:
            vix_regime = "MODERATE"
            
        if momentum > 0.3:
            momentum_regime = "BULLISH"
        elif momentum < -0.3:
            momentum_regime = "BEARISH"
        else:
            momentum_regime = "NEUTRAL"
            
        # Strategy selection
        if vix_regime == "HIGH":
            strategy = "IRON_CONDOR"
        elif vix_regime == "LOW":
            strategy = "DIAGONAL"
        else:  # MODERATE
            if momentum_regime == "BULLISH":
                strategy = "PUT_CREDIT_SPREAD"
            elif momentum_regime == "BEARISH":
                strategy = "CALL_CREDIT_SPREAD"
            else:
                strategy = "IRON_BUTTERFLY"
                
        conditions = {
            'vix': vix,
            'vix_regime': vix_regime,
            'momentum_score': momentum,
            'momentum_regime': momentum_regime,
            'spy_price': spy_price
        }
        
        return strategy, conditions

    def generate_signal(self, spy_bars: pd.DataFrame, option_chain: pd.DataFrame,
                       current_time: datetime, date: str) -> Optional[Dict]:
        """Generate multi-regime trading signal"""
        try:
            strategy_type, conditions = self.analyze_market_regime(date, spy_bars, current_time)
            
            # Calculate confidence
            vix_strength = abs(conditions['vix'] - 20) / 20
            momentum_strength = abs(conditions['momentum_score'])
            confidence = min(1.0, (vix_strength + momentum_strength) / 2)
            
            if confidence < 0.1:
                return None
                
            signal = {
                'type': strategy_type,
                'spy_price': conditions['spy_price'],
                'confidence': confidence,
                'vix': conditions['vix'],
                'momentum_score': conditions['momentum_score'],
                'timestamp': current_time
            }
            
            self.logger.debug(f"🏛️ REGIME SIGNAL: {strategy_type} "
                            f"(conf: {confidence:.3f}, VIX: {conditions['vix']:.1f})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None

    def simulate_complex_strategy(self, signal: Dict, option_info: Dict, contracts: int,
                                option_chain: pd.DataFrame, spy_bars: pd.DataFrame,
                                entry_time: datetime) -> Dict:
        """Simulate complex multi-leg strategies"""
        strategy_type = signal['type']
        
        # Simplified simulation for backtesting
        if strategy_type == 'PUT_CREDIT_SPREAD':
            return self.simulate_credit_spread(signal, option_info, contracts, 
                                             spy_bars, entry_time, 'PUT')
        elif strategy_type == 'CALL_CREDIT_SPREAD':
            return self.simulate_credit_spread(signal, option_info, contracts,
                                             spy_bars, entry_time, 'CALL')
        elif strategy_type == 'IRON_CONDOR':
            return self.simulate_iron_condor(signal, option_info, contracts,
                                           spy_bars, entry_time)
        elif strategy_type == 'IRON_BUTTERFLY':
            return self.simulate_iron_butterfly(signal, option_info, contracts,
                                              spy_bars, entry_time)
        elif strategy_type == 'DIAGONAL':
            return self.simulate_diagonal(signal, option_info, contracts,
                                        spy_bars, entry_time)
        else:
            # Fallback to base simulation
            return self.simulate_trade(signal, option_info, contracts,
                                     option_chain, spy_bars, entry_time)

    def simulate_credit_spread(self, signal: Dict, option_info: Dict, contracts: int,
                             spy_bars: pd.DataFrame, entry_time: datetime, 
                             spread_type: str) -> Dict:
        """Simulate credit spread with realistic progression"""
        try:
            # Credit spread parameters
            entry_credit = 80 * contracts  # $0.80 credit per spread
            max_profit = entry_credit
            max_loss = 120 * contracts     # $2 spread - $0.80 credit
            
            current_time = entry_time
            end_time = entry_time + timedelta(hours=6)
            spy_entry = signal['spy_price']
            
            best_profit = 0
            current_pnl = 0
            exit_reason = "TIME_LIMIT"
            
            # Simulate progression
            while current_time <= end_time:
                current_bars = spy_bars[spy_bars.index <= current_time]
                if len(current_bars) == 0:
                    break
                    
                current_spy = current_bars['close'].iloc[-1]
                spy_move_pct = (current_spy / spy_entry - 1) * 100
                
                # Time decay factor
                time_elapsed = (current_time - entry_time).total_seconds() / 3600
                decay_factor = max(0.1, 1 - time_elapsed / 6.5)
                
                # Estimate P&L based on SPY movement and time decay
                if spread_type == 'PUT' and spy_move_pct > -1:
                    # PUT spread profits when SPY stays above strikes
                    current_pnl = entry_credit * (1 - decay_factor * 0.8)
                elif spread_type == 'CALL' and spy_move_pct < 1:
                    # CALL spread profits when SPY stays below strikes
                    current_pnl = entry_credit * (1 - decay_factor * 0.8)
                else:
                    # SPY moving against us
                    loss_factor = min(1.0, abs(spy_move_pct) / 2)
                    current_pnl = -max_loss * loss_factor
                    
                best_profit = max(best_profit, current_pnl)
                
                # Check exit conditions
                if current_pnl >= max_profit * self.profit_target_pct:
                    exit_reason = "PROFIT_TARGET"
                    break
                    
                if current_pnl <= -entry_credit * self.stop_loss_pct:
                    exit_reason = "STOP_LOSS"
                    break
                    
                current_time += timedelta(minutes=5)
                
            # Final results
            hold_minutes = (current_time - entry_time).total_seconds() / 60
            outcome = "WIN" if current_pnl > 0 else "LOSS"
            
            return {
                'signal': signal,
                'strategy_type': signal['type'],
                'contracts': contracts,
                'entry_credit': entry_credit,
                'pnl': current_pnl,
                'outcome': outcome,
                'exit_reason': exit_reason,
                'hold_time_minutes': hold_minutes,
                'spy_movement_pct': spy_move_pct if 'spy_move_pct' in locals() else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating credit spread: {e}")
            return self.create_failed_result(signal)

    def simulate_iron_condor(self, signal: Dict, option_info: Dict, contracts: int,
                           spy_bars: pd.DataFrame, entry_time: datetime) -> Dict:
        """Simulate Iron Condor (simplified)"""
        # Higher credit, higher max loss
        entry_credit = 150 * contracts
        max_loss = 350 * contracts
        
        # Use credit spread logic with tighter profit zone
        return self.simulate_range_bound_strategy(signal, contracts, entry_credit, 
                                                max_loss, spy_bars, entry_time)

    def simulate_iron_butterfly(self, signal: Dict, option_info: Dict, contracts: int,
                              spy_bars: pd.DataFrame, entry_time: datetime) -> Dict:
        """Simulate Iron Butterfly (simplified)"""
        # Moderate credit, needs SPY to stay near current price
        entry_credit = 100 * contracts
        max_loss = 300 * contracts
        
        return self.simulate_range_bound_strategy(signal, contracts, entry_credit,
                                                max_loss, spy_bars, entry_time)

    def simulate_diagonal(self, signal: Dict, option_info: Dict, contracts: int,
                        spy_bars: pd.DataFrame, entry_time: datetime) -> Dict:
        """Simulate Diagonal spread (simplified as directional)"""
        # Use base simulation with modified parameters
        return self.simulate_trade(signal, option_info, contracts,
                                 pd.DataFrame(), spy_bars, entry_time)

    def simulate_range_bound_strategy(self, signal: Dict, contracts: int, entry_credit: float,
                                    max_loss: float, spy_bars: pd.DataFrame, 
                                    entry_time: datetime) -> Dict:
        """Simulate range-bound strategies like IC and IB"""
        try:
            current_time = entry_time
            end_time = entry_time + timedelta(hours=6)
            spy_entry = signal['spy_price']
            
            current_pnl = 0
            exit_reason = "TIME_LIMIT"
            
            while current_time <= end_time:
                current_bars = spy_bars[spy_bars.index <= current_time]
                if len(current_bars) == 0:
                    break
                    
                current_spy = current_bars['close'].iloc[-1]
                spy_move_pct = abs((current_spy / spy_entry - 1) * 100)
                
                # Time decay benefits range-bound strategies
                time_elapsed = (current_time - entry_time).total_seconds() / 3600
                decay_benefit = (1 - time_elapsed / 6.5) * 0.7
                
                # Profit if SPY stays in range, loss if it moves too much
                if spy_move_pct < 1.0:  # Within 1% range
                    current_pnl = entry_credit * decay_benefit
                else:  # Outside range
                    loss_factor = min(1.0, spy_move_pct / 3.0)
                    current_pnl = -max_loss * loss_factor
                    
                # Check exits
                if current_pnl >= entry_credit * self.profit_target_pct:
                    exit_reason = "PROFIT_TARGET"
                    break
                    
                if current_pnl <= -entry_credit * self.stop_loss_pct:
                    exit_reason = "STOP_LOSS"
                    break
                    
                current_time += timedelta(minutes=5)
                
            hold_minutes = (current_time - entry_time).total_seconds() / 60
            outcome = "WIN" if current_pnl > 0 else "LOSS"
            
            return {
                'signal': signal,
                'strategy_type': signal['type'],
                'contracts': contracts,
                'entry_credit': entry_credit,
                'pnl': current_pnl,
                'outcome': outcome,
                'exit_reason': exit_reason,
                'hold_time_minutes': hold_minutes,
                'spy_movement_pct': spy_move_pct if 'spy_move_pct' in locals() else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating range strategy: {e}")
            return self.create_failed_result(signal)

    def create_failed_result(self, signal: Dict) -> Dict:
        """Create failed trade result"""
        return {
            'signal': signal,
            'strategy_type': signal.get('type', 'UNKNOWN'),
            'contracts': 0,
            'pnl': 0,
            'outcome': "FAILED",
            'exit_reason': "ERROR",
            'hold_time_minutes': 0,
            'spy_movement_pct': 0
        }

    def check_daily_risk_limits(self, daily_pnl: float, trade_count: int) -> bool:
        """Check daily risk limits"""
        if daily_pnl >= self.daily_profit_target:
            self.logger.warning(f"🎯 DAILY PROFIT TARGET: ${daily_pnl:.2f}")
            return False
            
        if daily_pnl <= -self.daily_loss_limit:
            self.logger.warning(f"🛑 DAILY LOSS LIMIT: ${daily_pnl:.2f}")
            return False
            
        return True

    def backtest_single_day(self, date: str) -> Dict:
        """Backtest single day with multi-regime logic"""
        self.logger.info(f"🏛️ MULTI-REGIME BACKTEST: {date}")
        
        # Load data using parent class method (proven logic) - EXACT same as turtle backtest
        data = self.load_cached_data(date)
        if not data or 'spy_bars' not in data or 'option_chain' not in data:
            self.logger.warning(f"⚠️ No data available for {date}")
            return {'date': date, 'pnl': 0.0, 'trades': 0, 'win_rate': 0.0}
            
        spy_bars = data['spy_bars']
        option_chain = data['option_chain']
        
        # Check if data is empty (handle both DataFrame and dict cases)
        if (hasattr(spy_bars, 'empty') and spy_bars.empty) or len(spy_bars) == 0 or \
           (hasattr(option_chain, 'empty') and option_chain.empty) or len(option_chain) == 0:
            self.logger.warning(f"⚠️ Empty data for {date}")
            return {'date': date, 'pnl': 0.0, 'trades': 0, 'win_rate': 0.0}
            
        self.logger.info(f"✅ Loaded {len(spy_bars)} SPY bars and {len(option_chain)} options for {date}")
        
        # Trading session
        market_open = datetime.strptime(f"{date} 09:30:00", "%Y%m%d %H:%M:%S")
        market_close = datetime.strptime(f"{date} 16:00:00", "%Y%m%d %H:%M:%S")
        
        daily_trades = []
        daily_pnl = 0.0
        trade_count = 0
        
        current_time = market_open
        
        while current_time < market_close:
            if not self.check_daily_risk_limits(daily_pnl, trade_count):
                self.logger.info(f"🏁 Daily goal reached - stopping trading")
                break
                
            # Generate signal
            signal = self.generate_signal(spy_bars, option_chain, current_time, date)
            
            if signal:
                # Find option
                option_info = self.find_best_option(option_chain, signal['spy_price'], 'call')
                
                if option_info:
                    # Position size (simplified)
                    contracts = max(1, min(int(signal['confidence'] * 5), 8))
                    
                    # Execute trade
                    trade_result = self.simulate_complex_strategy(
                        signal, option_info, contracts, option_chain, spy_bars, current_time
                    )
                    
                    if trade_result and trade_result.get('pnl', 0) != 0:
                        trade_count += 1
                        trade_pnl = trade_result['pnl']
                        daily_pnl += trade_pnl
                        daily_trades.append(trade_result)
                        
                        # Update strategy performance
                        strategy_type = trade_result['strategy_type']
                        if strategy_type in self.strategy_performance:
                            self.strategy_performance[strategy_type]['trades'] += 1
                            self.strategy_performance[strategy_type]['pnl'] += trade_pnl
                            if trade_pnl > 0:
                                self.strategy_performance[strategy_type]['wins'] += 1
                        
                        outcome_emoji = "📈" if trade_pnl > 0 else "📉"
                        self.logger.info(f"🏛️ Trade #{trade_count}: {outcome_emoji} "
                                       f"{trade_result['outcome']} - ${trade_pnl:.2f} | "
                                       f"Daily P&L: ${daily_pnl:.2f}")
                        
                        # Wait before next trade
                        current_time += timedelta(minutes=30)
                    else:
                        current_time += timedelta(minutes=5)
                else:
                    current_time += timedelta(minutes=5)
            else:
                current_time += timedelta(minutes=10)
                
        # Calculate win rate
        winning_trades = sum(1 for trade in daily_trades if trade['pnl'] > 0)
        win_rate = (winning_trades / len(daily_trades) * 100) if daily_trades else 0
        
        self.logger.info(f"✅ Day complete: {trade_count} trades, ${daily_pnl:.2f} P&L, "
                        f"{win_rate:.1f}% win rate")
        
        return {
            'date': date,
            'trades': trade_count,
            'pnl': daily_pnl,
            'win_rate': win_rate,
            'daily_trades': daily_trades
        }

    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """Get list of trading days by trying to load data for each date"""
        trading_days = []
        
        # Generate date range
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y%m%d')
            
            # Try to load data for this date
            try:
                data = self.load_cached_data(date_str)
                if data and not data.get('spy_bars', pd.DataFrame()).empty:
                    trading_days.append(date_str)
            except:
                pass  # Skip dates with no data
                
            current_dt += timedelta(days=1)
            
        self.logger.info(f"Found {len(trading_days)} trading days")
        return trading_days
    
    def save_results(self, results: Dict, filename_prefix: str):
        """Save backtest results to pickle file"""
        try:
            # Create results directory if it doesn't exist
            results_dir = "backtrader/results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.pkl"
            filepath = os.path.join(results_dir, filename)
            
            # Save results
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
                
            self.logger.info(f"💾 Results saved to: {filepath}")
            print(f"💾 Results saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def main(self):
        """Main execution method like turtle backtest"""
        import argparse
        
        parser = argparse.ArgumentParser(description='Multi-Regime 0DTE Strategy Backtest')
        parser.add_argument('--start_date', default='20240102', help='Start date (YYYYMMDD)')
        parser.add_argument('--end_date', default='20240705', help='End date (YYYYMMDD)')
        parser.add_argument('--date', help='Single date to backtest (YYYYMMDD)')
        
        args = parser.parse_args()
        
        self.logger.info("🏛️ Starting Multi-Regime 0DTE Strategy Backtest")
        
        if args.date:
            # Single day backtest
            self.logger.info(f"📅 Single day: {args.date}")
            result = self.backtest_single_day(args.date)
            print(f"Single day result: {result}")
        else:
            # Multi-day backtest using proven approach
            self.logger.info(f"📅 Period: {args.start_date} to {args.end_date}")
            
            # Generate date list like turtle backtest does
            start_dt = datetime.strptime(args.start_date, '%Y%m%d')
            end_dt = datetime.strptime(args.end_date, '%Y%m%d')
            
            all_results = []
            total_pnl = 0
            total_trades = 0
            winning_days = 0
            processed_days = 0
            
            current_dt = start_dt
            while current_dt <= end_dt:
                date_str = current_dt.strftime('%Y%m%d')
                
                # Try to backtest this day
                try:
                    day_result = self.backtest_single_day(date_str)
                    if day_result and 'pnl' in day_result:
                        all_results.append(day_result)
                        total_pnl += day_result['pnl']
                        total_trades += day_result.get('trades', 0)
                        processed_days += 1
                        
                        if day_result['pnl'] > 0:
                            winning_days += 1
                            
                        self.logger.info(f"📈 {date_str}: {day_result.get('trades', 0)} trades, ${day_result['pnl']:.2f} P&L")
                except Exception as e:
                    # Skip days with no data (weekends, holidays)
                    pass
                    
                current_dt += timedelta(days=1)
                
            # Print final results
            if processed_days > 0:
                avg_daily_pnl = total_pnl / processed_days
                profitable_day_rate = (winning_days / processed_days * 100)
                
                print("\n" + "="*80)
                print("🏛️ MULTI-REGIME 0DTE STRATEGY - BACKTEST RESULTS")
                print("="*80)
                print(f"📅 Trading Days: {processed_days}")
                print()
                print("💰 PERFORMANCE SUMMARY:")
                print(f"   Total P&L: ${total_pnl:.2f}")
                print(f"   Average Daily P&L: ${avg_daily_pnl:.2f}")
                print(f"   Max Daily Profit Target: ${self.daily_profit_target}")
                print(f"   Max Daily Loss Limit: ${self.daily_loss_limit}")
                print()
                print("📈 TRADE STATISTICS:")
                print(f"   Total Trades: {total_trades}")
                print(f"   Avg Trades/Day: {total_trades/processed_days:.1f}")
                print()
                print("📊 DAILY PERFORMANCE:")
                print(f"   Profitable Days: {profitable_day_rate:.1f}%")
                print()
                print("🎯 STRATEGY BREAKDOWN:")
                for strategy, perf in self.strategy_performance.items():
                    if perf['trades'] > 0:
                        win_rate = (perf['wins'] / perf['trades']) * 100
                        avg_pnl = perf['pnl'] / perf['trades']
                        print(f"   {strategy}: {perf['trades']} trades, "
                              f"{win_rate:.1f}% win rate, ${avg_pnl:.2f} avg P&L")
                print("="*80)
            else:
                print("❌ No trading days found with data")

    def print_final_results(self, total_pnl: float, avg_daily_pnl: float, 
                          total_trades: int, trading_days: int, 
                          profitable_day_rate: float, duration: float):
        """Print comprehensive final results"""
        print("\n" + "="*80)
        print("🏛️ MULTI-REGIME 0DTE STRATEGY - BACKTEST RESULTS")
        print("="*80)
        print(f"📅 Trading Days: {trading_days}")
        print(f"⚡ Backtest Duration: {duration:.2f} seconds")
        print()
        print("💰 PERFORMANCE SUMMARY:")
        print(f"   Total P&L: ${total_pnl:.2f}")
        print(f"   Average Daily P&L: ${avg_daily_pnl:.2f}")
        print(f"   Max Daily Profit Target: ${self.daily_profit_target}")
        print(f"   Max Daily Loss Limit: ${self.daily_loss_limit}")
        print()
        print("📈 TRADE STATISTICS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Avg Trades/Day: {total_trades/trading_days:.1f}")
        print()
        print("📊 DAILY PERFORMANCE:")
        print(f"   Profitable Days: {profitable_day_rate:.1f}%")
        print()
        print("🎯 STRATEGY BREAKDOWN:")
        for strategy, perf in self.strategy_performance.items():
            if perf['trades'] > 0:
                win_rate = (perf['wins'] / perf['trades']) * 100
                avg_pnl = perf['pnl'] / perf['trades']
                print(f"   {strategy}: {perf['trades']} trades, "
                      f"{win_rate:.1f}% win rate, ${avg_pnl:.2f} avg P&L")
        print("="*80)


if __name__ == "__main__":
    # Run backtest using the same pattern as turtle backtest
    backtest = MultiRegime0DTEBacktest(cache_dir='./thetadata/cached_data')
    backtest.main()
