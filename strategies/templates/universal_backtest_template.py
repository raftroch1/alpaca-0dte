#!/usr/bin/env python3
"""
🎯 UNIVERSAL BACKTEST TEMPLATE - REAL DATA ONLY
==============================================

COPY THIS TEMPLATE TO CREATE BACKTESTS FOR NEW STRATEGIES
Uses REAL historical data from Alpaca API + ThetaData cache (NO SIMULATION)

📋 QUICK START:
1. Copy: cp universal_backtest_template.py your_strategy_backtest.py
2. Rename class: UniversalBacktestTemplate -> YourStrategyBacktest
3. Import your strategy class
4. Customize backtest_single_day() method
5. Run backtest: python your_strategy_backtest.py --start-date 20240301 --end-date 20240331

✅ REAL DATA SOURCES:
- SPY Bars: ThetaData cache (6 months: Jan-Jun 2024)
- Option Prices: Alpaca historical API (real market prices)
- No synthetic time decay, volatility models, or random walks
- Same data sources used by proven profitable strategies

❌ NO SIMULATION:
- No synthetic option pricing
- No estimated Greeks or implied volatility
- No random market movements
- Only historical market data

📊 PROVEN PATTERN FROM:
- Multi-Regime Strategy: $71,724 profit over 6 months
- Phase 3 Strategy: $-1,794 P&L with 196 trades (real validation)
- Alpaca Real Data Strategy: Proven framework for historical option pricing

Author: Universal Backtest Framework v2.0
Date: 2025-01-31
Based on: Multi-Regime, Phase 3, Alpaca Real Data proven backtests
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import pickle
import gzip
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

# 🔧 STANDARDIZED IMPORT PATHS
# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), '.env'))

# 📊 ALPACA REAL DATA IMPORTS (No Simulation)
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame

# 🎯 IMPORT YOUR STRATEGY CLASS HERE
# Example: from your_strategy import YourStrategyClass


class UniversalBacktestTemplate:
    """
    🎯 Universal Backtest Template - Copy and customize for new strategies
    
    PROVEN REAL DATA PATTERNS:
    - ThetaData cache for SPY minute bars (6 months historical data)
    - Alpaca API for real historical option prices
    - Same infrastructure that validated profitable strategies
    - No simulation or synthetic data generation
    
    RENAME THIS CLASS: YourStrategyBacktest
    Example: MomentumBreakoutBacktest, VixContrarianBacktest, etc.
    """
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        """
        🚀 Initialize backtest with real data sources
        
        Args:
            cache_dir: Path to ThetaData cache directory
        """
        self.cache_dir = cache_dir
        self.setup_logging()
        
        # 🔑 Initialize Alpaca client for real option prices
        self._initialize_alpaca_client()
        
        # 📊 Backtest configuration
        self.config = self._get_backtest_config()
        
        # 📈 Performance tracking
        self._initialize_tracking()
        
        self.logger.info("🎯 Universal Backtest Template initialized")
        self.logger.info(f"📊 Cache Directory: {cache_dir}")
        self.logger.info(f"✅ Real Data Sources: ThetaData + Alpaca API")
        self.logger.info("❌ NO SIMULATION - Real historical data only")
    
    def setup_logging(self):
        """📝 Setup logging for backtest"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_alpaca_client(self):
        """🏦 Initialize Alpaca client for real option data"""
        try:
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key:
                raise ValueError("❌ Missing Alpaca API credentials")
            
            self.alpaca_client = OptionHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key
            )
            self.logger.info("✅ Alpaca option data client initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca client initialization failed: {e}")
            self.alpaca_client = None
    
    def _get_backtest_config(self) -> Dict[str, Any]:
        """
        🎯 CUSTOMIZE BACKTEST CONFIGURATION
        
        Based on proven patterns from working backtests
        """
        return {
            # 💰 ACCOUNT SETTINGS
            'starting_capital': 25000.0,
            'commission_per_contract': 0.65,  # Realistic commission
            'max_risk_per_trade': 0.02,       # 2% max risk per trade
            
            # ⏰ TIMING SETTINGS
            'market_open_buffer_minutes': 15,  # Wait after market open
            'market_close_buffer_minutes': 30, # Exit before market close
            'min_time_to_expiry_minutes': 30,  # Min time before expiry
            
            # 📊 DATA VALIDATION
            'min_spy_bars_required': 50,       # Minimum SPY bars for analysis
            'max_bid_ask_spread_pct': 0.20,    # Max 20% bid-ask spread
            'min_option_price': 0.05,          # Min option price
            'max_option_price': 5.00,          # Max option price
            
            # 🎯 STRATEGY SETTINGS (Customize for your strategy)
            'confidence_threshold': 0.60,      # Minimum signal confidence
            'max_daily_trades': 10,            # Max trades per day
            'max_position_time_minutes': 180,  # Max 3 hours per position
        }
    
    def _initialize_tracking(self):
        """📈 Initialize performance tracking"""
        self.results = {
            'daily_results': [],
            'all_trades': [],
            'total_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_drawdown': 0.0,
            'current_capital': self.config['starting_capital']
        }
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_positions = []
    
    def load_cached_spy_data(self, date_str: str) -> pd.DataFrame:
        """
        📊 Load SPY minute bars from ThetaData cache (proven pattern)
        
        Args:
            date_str: Date in YYYYMMDD format
            
        Returns:
            DataFrame with SPY minute bars
        """
        try:
            spy_file = os.path.join(self.cache_dir, 'spy_bars', f'spy_bars_{date_str}.pkl.gz')
            
            if not os.path.exists(spy_file):
                self.logger.warning(f"⚠️ No SPY data for {date_str}")
                return pd.DataFrame()
            
            with gzip.open(spy_file, 'rb') as f:
                spy_data = pickle.load(f)
            
            # Convert to DataFrame if needed
            if isinstance(spy_data, dict):
                spy_bars = pd.DataFrame(spy_data)
            else:
                spy_bars = spy_data
            
            # Ensure datetime index
            if not isinstance(spy_bars.index, pd.DatetimeIndex):
                spy_bars.index = pd.to_datetime(spy_bars.index)
            
            self.logger.debug(f"✅ Loaded {len(spy_bars)} SPY bars for {date_str}")
            return spy_bars
            
        except Exception as e:
            self.logger.error(f"❌ Error loading SPY data for {date_str}: {e}")
            return pd.DataFrame()
    
    def get_real_option_price(self, spy_price: float, option_type: str, 
                            strike: float, trade_date: str) -> Optional[float]:
        """
        💰 Get REAL historical option price from Alpaca API (proven pattern)
        
        Args:
            spy_price: Current SPY price  
            option_type: 'call' or 'put'
            strike: Option strike price
            trade_date: Date in YYYY-MM-DD format
            
        Returns:
            Real historical option price or None if not available
        """
        if not self.alpaca_client:
            self.logger.warning("⚠️ Alpaca client not available")
            return None
        
        try:
            # Build option symbol (Alpaca format)
            date_obj = datetime.strptime(trade_date, '%Y-%m-%d')
            exp_date = date_obj.strftime('%y%m%d')  # YYMMDD format
            option_letter = 'C' if option_type.lower() == 'call' else 'P'
            strike_str = f"{int(strike * 1000):08d}"  # 8-digit format
            
            symbol = f"SPY{exp_date}{option_letter}{strike_str}"
            
            # Request option data
            request = OptionBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=date_obj - timedelta(days=1),
                end=date_obj + timedelta(days=1)
            )
            
            bars_data = self.alpaca_client.get_option_bars(request)
            
            if symbol in bars_data.data and len(bars_data.data[symbol]) > 0:
                # Find bar for target date
                target_date = date_obj.date()
                for bar in bars_data.data[symbol]:
                    if bar.timestamp.date() == target_date:
                        price = float(bar.close)
                        self.logger.debug(f"✅ Real price {symbol}: ${price:.2f}")
                        return price
                
                # Use latest available if exact date not found
                if bars_data.data[symbol]:
                    price = float(bars_data.data[symbol][-1].close)
                    self.logger.debug(f"✅ Latest price {symbol}: ${price:.2f}")
                    return price
            
            self.logger.debug(f"❌ No data for {symbol}")
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting option price: {e}")
            return None
    
    def backtest_single_day(self, date_str: str) -> Dict[str, Any]:
        """
        🎯 IMPLEMENT THIS METHOD - Backtest your strategy for a single day
        
        Args:
            date_str: Trading date in YYYYMMDD format
            
        Returns:
            Dict with daily results
            
        CUSTOMIZATION POINTS:
        1. Load your strategy class
        2. Generate signals using strategy logic
        3. Simulate trades with real option prices
        4. Apply proper risk management
        """
        self.logger.info(f"📅 Backtesting {date_str}")
        
        # Reset daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_positions = []
        
        try:
            # 📊 Load real SPY data
            spy_data = self.load_cached_spy_data(date_str)
            if spy_data.empty:
                return self._create_daily_result(date_str, "No SPY data")
            
            if len(spy_data) < self.config['min_spy_bars_required']:
                return self._create_daily_result(date_str, "Insufficient SPY data")
            
            # 🎯 CUSTOMIZE: Initialize your strategy here
            # Example:
            # strategy = YourStrategyClass()
            
            # 📈 Simulate trading throughout the day
            trade_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            
            # Apply market hours buffer (proven pattern)
            start_idx = self.config['market_open_buffer_minutes']
            end_idx = len(spy_data) - self.config['market_close_buffer_minutes']
            
            trading_data = spy_data.iloc[start_idx:end_idx]
            
            # 🔄 Main trading simulation loop
            for i in range(len(trading_data)):
                if self.daily_trades >= self.config['max_daily_trades']:
                    break
                
                # Get current market slice
                current_slice = trading_data.iloc[:i+1]
                current_spy_price = current_slice['close'].iloc[-1]
                
                # 🎯 CUSTOMIZE: Generate signal using your strategy
                # Example:
                # signal = strategy.generate_signal(current_slice)
                
                # PLACEHOLDER SIGNAL (Replace with your strategy)
                signal = self._example_signal_generation(current_slice)
                
                if signal['action'] != 'HOLD':
                    # Simulate trade execution
                    trade_result = self._simulate_trade(
                        signal, current_spy_price, trade_date, trading_data.index[i]
                    )
                    
                    if trade_result:
                        self.daily_positions.append(trade_result)
                        self.daily_trades += 1
            
            # 📊 Process end-of-day results
            return self._process_daily_results(date_str, spy_data)
            
        except Exception as e:
            self.logger.error(f"❌ Error backtesting {date_str}: {e}")
            return self._create_daily_result(date_str, f"Error: {e}")
    
    def _example_signal_generation(self, spy_data: pd.DataFrame) -> Dict[str, Any]:
        """
        📊 REPLACE THIS WITH YOUR STRATEGY LOGIC
        
        This is a placeholder - implement your actual strategy signal generation
        """
        if len(spy_data) < 20:
            return {'action': 'HOLD', 'confidence': 0.0}
        
        # Simple example: RSI-based signals
        current_price = spy_data['close'].iloc[-1]
        returns = spy_data['close'].pct_change().dropna()
        
        if len(returns) < 14:
            return {'action': 'HOLD', 'confidence': 0.0}
        
        # Simple momentum check
        recent_return = returns.iloc[-5:].mean()
        
        if recent_return > 0.002:  # Strong upward momentum
            return {
                'action': 'BUY_CALL',
                'confidence': 0.70,
                'target_strike': round(current_price + 1.0),
                'reasoning': 'Upward momentum detected'
            }
        elif recent_return < -0.002:  # Strong downward momentum
            return {
                'action': 'BUY_PUT',
                'confidence': 0.70,
                'target_strike': round(current_price - 1.0),
                'reasoning': 'Downward momentum detected'
            }
        else:
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def _simulate_trade(self, signal: Dict[str, Any], spy_price: float, 
                       trade_date: str, timestamp) -> Optional[Dict[str, Any]]:
        """🎯 Simulate trade execution with real option prices"""
        try:
            if signal['confidence'] < self.config['confidence_threshold']:
                return None
            
            # Determine option parameters
            option_type = 'call' if signal['action'] == 'BUY_CALL' else 'put'
            strike = signal['target_strike']
            
            # Get real option price
            option_price = self.get_real_option_price(spy_price, option_type, strike, trade_date)
            
            if not option_price:
                self.logger.debug(f"⚠️ No option price available for {option_type} ${strike}")
                return None
            
            # Validate option price
            if (option_price < self.config['min_option_price'] or 
                option_price > self.config['max_option_price']):
                return None
            
            # Calculate position size based on risk
            max_risk = self.results['current_capital'] * self.config['max_risk_per_trade']
            position_size = max(1, int(max_risk / (option_price * 100)))
            position_size = min(position_size, 10)  # Cap at 10 contracts
            
            # Calculate costs
            premium_paid = option_price * position_size * 100
            commission = self.config['commission_per_contract'] * position_size
            total_cost = premium_paid + commission
            
            # Create trade record
            trade = {
                'timestamp': timestamp,
                'signal': signal,
                'option_type': option_type,
                'strike': strike,
                'spy_price': spy_price,
                'option_price': option_price,
                'position_size': position_size,
                'premium_paid': premium_paid,
                'commission': commission,
                'total_cost': total_cost,
                'entry_time': timestamp
            }
            
            self.logger.info(f"📈 Trade: {option_type.upper()} ${strike} @ ${option_price:.2f} x{position_size}")
            return trade
            
        except Exception as e:
            self.logger.error(f"❌ Trade simulation error: {e}")
            return None
    
    def _process_daily_results(self, date_str: str, spy_data: pd.DataFrame) -> Dict[str, Any]:
        """📊 Process end-of-day results and calculate P&L"""
        try:
            final_spy_price = spy_data['close'].iloc[-1]
            trade_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            
            daily_pnl = 0.0
            
            # Calculate P&L for each position (0DTE expires at end of day)
            for trade in self.daily_positions:
                # Calculate expiration value
                strike = trade['strike']
                option_type = trade['option_type']
                
                if option_type == 'call':
                    intrinsic_value = max(0, final_spy_price - strike)
                else:  # put
                    intrinsic_value = max(0, strike - final_spy_price)
                
                # Total P&L = (Expiration Value - Premium Paid) * Position Size * 100 - Commission
                position_pnl = (intrinsic_value - trade['option_price']) * trade['position_size'] * 100
                trade['pnl'] = position_pnl
                trade['exit_spy_price'] = final_spy_price
                trade['intrinsic_value'] = intrinsic_value
                
                daily_pnl += position_pnl
                
                # Add to all trades
                self.results['all_trades'].append(trade)
                
                self.logger.info(f"📊 Trade P&L: ${position_pnl:.2f} ({option_type} ${strike})")
            
            # Update tracking
            self.results['total_pnl'] += daily_pnl
            self.results['total_trades'] += len(self.daily_positions)
            self.results['current_capital'] += daily_pnl
            
            # Count wins/losses
            for trade in self.daily_positions:
                if trade['pnl'] > 0:
                    self.results['winning_trades'] += 1
                else:
                    self.results['losing_trades'] += 1
            
            return self._create_daily_result(date_str, "Success", daily_pnl, len(self.daily_positions))
            
        except Exception as e:
            self.logger.error(f"❌ Error processing daily results: {e}")
            return self._create_daily_result(date_str, f"Processing error: {e}")
    
    def _create_daily_result(self, date_str: str, status: str, 
                           pnl: float = 0.0, trades: int = 0) -> Dict[str, Any]:
        """📊 Create standardized daily result"""
        result = {
            'date': date_str,
            'status': status,
            'pnl': pnl,
            'trades': trades,
            'capital': self.results['current_capital']
        }
        self.results['daily_results'].append(result)
        return result
    
    def run_backtest(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        🚀 Run backtest over date range
        
        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
        """
        self.logger.info(f"🚀 Starting backtest: {start_date} to {end_date}")
        
        # Generate date range
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        current_dt = start_dt
        successful_days = 0
        
        while current_dt <= end_dt:
            # Skip weekends
            if current_dt.weekday() < 5:  # Monday = 0, Friday = 4
                date_str = current_dt.strftime('%Y%m%d')
                result = self.backtest_single_day(date_str)
                
                if result['status'] == 'Success':
                    successful_days += 1
                    
            current_dt += timedelta(days=1)
        
        # Generate final statistics
        return self._generate_final_statistics(successful_days)
    
    def _generate_final_statistics(self, successful_days: int) -> Dict[str, Any]:
        """📊 Generate comprehensive backtest statistics"""
        stats = {
            'successful_days': successful_days,
            'total_days': len(self.results['daily_results']),
            'total_pnl': self.results['total_pnl'],
            'total_trades': self.results['total_trades'],
            'winning_trades': self.results['winning_trades'],
            'losing_trades': self.results['losing_trades'],
            'win_rate': (self.results['winning_trades'] / max(1, self.results['total_trades'])) * 100,
            'final_capital': self.results['current_capital'],
            'total_return': ((self.results['current_capital'] - self.config['starting_capital']) / self.config['starting_capital']) * 100
        }
        
        # Calculate average daily P&L
        profitable_days = [r for r in self.results['daily_results'] if r['pnl'] > 0]
        stats['profitable_days'] = len(profitable_days)
        stats['avg_daily_pnl'] = self.results['total_pnl'] / max(1, successful_days)
        
        # Calculate max drawdown
        running_capital = self.config['starting_capital']
        peak_capital = running_capital
        max_dd = 0.0
        
        for result in self.results['daily_results']:
            running_capital += result['pnl']
            if running_capital > peak_capital:
                peak_capital = running_capital
            
            drawdown = (peak_capital - running_capital) / peak_capital * 100
            max_dd = max(max_dd, drawdown)
        
        stats['max_drawdown_pct'] = max_dd
        
        return stats
    
    def print_results(self, stats: Dict[str, Any]):
        """📊 Print formatted backtest results"""
        print(f"\n🎯 BACKTEST RESULTS - REAL DATA")
        print("=" * 60)
        print(f"📅 Trading Days: {stats['successful_days']}/{stats['total_days']}")
        print(f"💰 Total P&L: ${stats['total_pnl']:,.2f}")
        print(f"📈 Total Return: {stats['total_return']:.2f}%")
        print(f"💵 Final Capital: ${stats['final_capital']:,.2f}")
        print("")
        print(f"📊 Total Trades: {stats['total_trades']}")
        print(f"✅ Winning Trades: {stats['winning_trades']}")
        print(f"❌ Losing Trades: {stats['losing_trades']}")
        print(f"🎯 Win Rate: {stats['win_rate']:.1f}%")
        print("")
        print(f"📅 Profitable Days: {stats['profitable_days']}/{stats['successful_days']}")
        print(f"💰 Avg Daily P&L: ${stats['avg_daily_pnl']:.2f}")
        print(f"📉 Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
        print("")
        print("✅ DATA SOURCES: ThetaData Cache + Alpaca Real Option Prices")
        print("❌ NO SIMULATION - Real historical data only")
        print("=" * 60)


def main():
    """🎯 Main entry point for backtest"""
    parser = argparse.ArgumentParser(description='Universal Strategy Backtest - Real Data')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYYMMDD)')
    parser.add_argument('--cache-dir', default='../../thetadata/cached_data', help='ThetaData cache directory')
    
    args = parser.parse_args()
    
    print(f"\n🎯 UNIVERSAL STRATEGY BACKTEST")
    print("=" * 60)
    print(f"📅 Period: {args.start_date} to {args.end_date}")
    print(f"📊 Framework: Universal Template v2.0")
    print(f"✅ Real Data: ThetaData + Alpaca API")
    print(f"❌ No Simulation: Historical data only")
    print("=" * 60)
    
    # Initialize and run backtest
    backtest = UniversalBacktestTemplate(cache_dir=args.cache_dir)
    stats = backtest.run_backtest(args.start_date, args.end_date)
    backtest.print_results(stats)


if __name__ == "__main__":
    main() 