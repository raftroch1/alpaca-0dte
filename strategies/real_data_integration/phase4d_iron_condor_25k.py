#!/usr/bin/env python3
"""
🎪 PHASE 4D IRON CONDOR - 25K ACCOUNT OPTIMIZED
==============================================

Capital-efficient Iron Condor strategy based on @examples/options/options-iron-condor.ipynb
EXACT same filtering as balanced strategy but using 4-leg Iron Condors instead of bull put spreads.

🎪 IRON CONDOR STRUCTURE:
- Sell OTM Put + Buy further OTM Put (bull put spread)
- Sell OTM Call + Buy further OTM Call (bear call spread)  
- Profit if SPY stays BETWEEN inner strikes (wide profit zone)
- Max Risk = Larger of the two spread widths

💰 25K ACCOUNT OPTIMIZATION:
- Position sizing for $25,000 account
- Target $300/day revenue
- Risk management for small account

🔄 COMPARISON TARGET:
- Original Balanced (Naked Puts): 66.7% win rate, +$37.25
- Bull Put Spreads: 28.6% win rate, -$143.50
- Iron Condors: Testing for higher win rate + capital efficiency

Author: Strategy Development Framework  
Date: 2025-08-01
Version: Iron Condor 25K v1.0
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, time
import pickle
import gzip
import argparse
from dotenv import load_dotenv
from typing import Optional, Dict, List, Tuple
import math

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from alpaca.data import OptionHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest, StockBarsRequest, OptionLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce, ContractType
    from alpaca.trading.requests import MarketOrderRequest, OptionLegRequest, ClosePositionRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  Alpaca SDK not available")

class Phase4DIronCondor25K:
    """
    Iron Condor strategy for 25K account - capital efficient 4-leg strategy
    Following @examples/options/options-iron-condor.ipynb pattern for multi-leg execution
    """
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        self.cache_dir = cache_dir
        self.setup_logging()
        
        # IRON CONDOR Parameters - 25K Account Optimized
        self.params = {
            # Core strategy
            'strategy_type': 'iron_condor_25k',
            'wing_width': 2.0,               # $2 wings (capital efficient)
            'strike_buffer': 1.0,            # $1 buffer from current price
            'max_daily_trades': 1,
            
            # Strike Selection - KEEP BALANCED QUALITY
            'min_premium': 0.15,             # Minimum $0.15 total credit
            'max_premium': 3.00,             # Maximum $3.00 total credit
            'target_delta_range': (-0.25, -0.10),  # Same delta targeting
            
            # Volatility Filtering - KEEP PROTECTIVE FILTERING  
            'max_vix_threshold': 25,         # Same as balanced
            'max_daily_range': 5.0,          # Same as balanced
            'disaster_threshold': 8.0,       # Same as balanced
            
            # 25K ACCOUNT POSITION SIZING
            'base_contracts': 3,             # 3 Iron Condors per trade
            'max_contracts': 5,              # Max 5 condors ($1000 risk)
            'target_daily_pnl': 300,         # $300/day target
            
            # 25K RISK MANAGEMENT
            'max_loss_per_trade': 800,       # Max $800 loss per trade  
            'max_daily_loss': 1200,          # Max $1200 loss per day
            'profit_target_pct': 50,         # Take profit at 50%
            'stop_loss_multiple': 2.0,       # Stop at 2x premium
        }
        
        self.daily_pnl = 0
        self.last_trade_date = None
        self.setup_alpaca_clients()
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_alpaca_clients(self):
        """Setup Alpaca clients for real data"""
        try:
            load_dotenv()
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if api_key and secret_key and ALPACA_AVAILABLE:
                self.option_client = OptionHistoricalDataClient(
                    api_key=api_key,
                    secret_key=secret_key
                )
                self.stock_client = StockHistoricalDataClient(
                    api_key=api_key,
                    secret_key=secret_key
                )
                self.logger.info("✅ Alpaca clients established")
            else:
                self.option_client = None
                self.stock_client = None
                self.logger.error("❌ No Alpaca credentials")
                
        except Exception as e:
            self.logger.error(f"❌ Alpaca setup failed: {e}")
            self.option_client = None
            self.stock_client = None
    
    def load_spy_data(self, date_str: str) -> Optional[pd.DataFrame]:
        """Load SPY data for a date"""
        try:
            file_path = os.path.join(self.cache_dir, "spy_bars", f"spy_bars_{date_str}.pkl.gz")
            
            if not os.path.exists(file_path):
                self.logger.warning(f"No SPY data found for {date_str} at {file_path}")
                return None
                
            with gzip.open(file_path, 'rb') as f:
                spy_data = pickle.load(f)
            
            return spy_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading SPY data for {date_str}: {e}")
            return None
    
    def get_balanced_strike(self, spy_price: float, daily_range: float) -> float:
        """
        SAME strike selection as original balanced - keep signal quality intact
        """
        # Target ITM puts with meaningful premiums (same logic as original)
        if daily_range < 1.0:
            buffer = -0.25  # $0.25 ABOVE current SPY (ITM)
        elif daily_range < 3.0:
            buffer = 0.0   # Right at current SPY
        else:
            buffer = self.params['min_strike_buffer']

        optimal_strike = round(spy_price - buffer, 0)

        self.logger.info(f"🎯 BALANCED STRIKE SELECTION (SAME AS ORIGINAL):")
        self.logger.info(f"   SPY: ${spy_price:.2f}")
        self.logger.info(f"   Daily Range: {daily_range:.2f}%")
        self.logger.info(f"   Buffer: ${buffer:.2f}")
        self.logger.info(f"   Strike: ${optimal_strike}")

        return optimal_strike
    
    def get_real_option_price(self, symbol: str, date_str: str) -> Optional[float]:
        """Get real option price from Alpaca - FIXED VERSION"""
        if not self.option_client:
            return None
            
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            
            request = OptionBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=date_obj,
                end=date_obj + timedelta(days=1)
            )
            
            option_data = self.option_client.get_option_bars(request)
            
            if symbol in option_data.data and len(option_data.data[symbol]) > 0:
                last_bar = option_data.data[symbol][-1]
                price = float(last_bar.close)
                
                self.logger.info(f"✅ REAL {symbol}: ${price:.3f}")
                return price
            else:
                self.logger.warning(f"❌ No data for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching {symbol}: {e}")
            return None
    
    def get_call_option_price(self, spy_price: float, strike: float, date_str: str) -> Optional[float]:
        """Get call option price from Alpaca API"""
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            call_symbol = f"SPY{date_obj.strftime('%y%m%d')}C{int(strike*1000):08d}"
            
            call_price = self.get_real_option_price(call_symbol, date_str)
            if call_price:
                self.logger.info(f"📞 Call {call_symbol}: ${call_price:.3f}")
            return call_price
            
        except Exception as e:
            self.logger.error(f"❌ Error getting call price: {e}")
            return None

    def get_put_option_price(self, spy_price: float, strike: float, date_str: str) -> Optional[float]:
        """Get put option price from Alpaca API"""
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            put_symbol = f"SPY{date_obj.strftime('%y%m%d')}P{int(strike*1000):08d}"
            
            put_price = self.get_real_option_price(put_symbol, date_str)
            if put_price:
                self.logger.info(f"🔻 Put {put_symbol}: ${put_price:.3f}")
            return put_price
            
        except Exception as e:
            self.logger.error(f"❌ Error getting put price: {e}")
            return None

    def get_iron_condor_strikes(self, spy_price: float) -> Tuple[float, float, float, float]:
        """
        Calculate Iron Condor strike prices based on current SPY price
        Returns: (short_put_strike, long_put_strike, short_call_strike, long_call_strike)
        """
        # Short strikes closer to current price (collect premium)
        short_put_strike = spy_price - self.params['strike_buffer']
        short_call_strike = spy_price + self.params['strike_buffer']
        
        # Long strikes further away (protection wings)
        long_put_strike = short_put_strike - self.params['wing_width']
        long_call_strike = short_call_strike + self.params['wing_width']
        
        # Round to nearest $0.50 (SPY option standard)
        short_put_strike = round(short_put_strike * 2) / 2
        long_put_strike = round(long_put_strike * 2) / 2  
        short_call_strike = round(short_call_strike * 2) / 2
        long_call_strike = round(long_call_strike * 2) / 2
        
        return short_put_strike, long_put_strike, short_call_strike, long_call_strike
    
    def validate_trade_conditions(self, spy_data: pd.DataFrame, strike: float, premium: float) -> Tuple[bool, str]:
        """
        SAME validation as original balanced - keep ALL protective filtering
        """
        spy_open = spy_data['open'].iloc[0]
        spy_close = spy_data['close'].iloc[-1]
        spy_high = spy_data['high'].max()
        spy_low = spy_data['low'].min()
        daily_range = (spy_high - spy_low) / spy_open * 100
        
        # SAME premium validation as original
        if premium < self.params['min_premium']:
            return False, f"Premium ${premium:.3f} below minimum ${self.params['min_premium']:.3f}"
        
        if premium > self.params['max_premium']:
            return False, f"Premium ${premium:.3f} above maximum ${self.params['max_premium']:.3f}"
        
        # SAME volatility validation as original
        if daily_range > self.params['max_daily_range']:
            return False, f"High volatility: {daily_range:.2f}%"
        
        if daily_range > self.params['disaster_threshold']:
            return False, f"Disaster volatility: {daily_range:.2f}%"
        
        return True, "Valid trade"
    
    def calculate_iron_condor_outcome(self, spy_data: pd.DataFrame, strikes: Tuple, date_str: str) -> Dict:
        """Calculate IRON CONDOR outcome with 4-leg structure"""
        short_put_strike, long_put_strike, short_call_strike, long_call_strike = strikes
        spy_close = spy_data['close'].iloc[-1]
        spy_open = spy_data['open'].iloc[0]
        contracts = self.params['base_contracts']  # 3 contracts for 25K account
        
        # Get all four option prices
        short_put_price = self.get_put_option_price(spy_open, short_put_strike, date_str)
        long_put_price = self.get_put_option_price(spy_open, long_put_strike, date_str)
        short_call_price = self.get_call_option_price(spy_open, short_call_strike, date_str)
        long_call_price = self.get_call_option_price(spy_open, long_call_strike, date_str)
        
        if None in [short_put_price, long_put_price, short_call_price, long_call_price]:
            return {'strategy': 'iron_condor_failed', 'final_pnl': 0, 'outcome': 'PRICING_FAILED'}
        
        # Calculate net credit received (following @examples/ pattern)
        put_spread_credit = short_put_price - long_put_price
        call_spread_credit = short_call_price - long_call_price  
        total_credit = put_spread_credit + call_spread_credit
        
        # Gross premium collected
        gross_premium = total_credit * 100 * contracts
        
        # Trading costs (4 legs following @examples/)
        commission = 4.0 * contracts  # $4 per condor (4 legs × $1)
        bid_ask_cost = total_credit * 100 * contracts * 0.04  # 4% of premium
        slippage = total_credit * 100 * contracts * 0.01  # 1% slippage
        total_costs = commission + bid_ask_cost + slippage
        net_premium_received = gross_premium - total_costs
        
        # IRON CONDOR P&L CALCULATION AT EXPIRATION
        # Put spread P&L
        if spy_close <= long_put_strike:
            put_pnl = put_spread_credit - self.params['wing_width']  # Max loss
        elif spy_close >= short_put_strike:
            put_pnl = put_spread_credit  # Max profit
        else:
            put_pnl = put_spread_credit - (short_put_strike - spy_close)  # Partial loss
        
        # Call spread P&L  
        if spy_close >= long_call_strike:
            call_pnl = call_spread_credit - self.params['wing_width']  # Max loss
        elif spy_close <= short_call_strike:
            call_pnl = call_spread_credit  # Max profit
        else:
            call_pnl = call_spread_credit - (spy_close - short_call_strike)  # Partial loss
        
        # Total P&L
        total_spread_pnl = (put_pnl + call_pnl) * 100 * contracts
        final_pnl = total_spread_pnl - total_costs
        
        # Determine outcome
        if short_put_strike <= spy_close <= short_call_strike:
            outcome = 'MAX_PROFIT'  # Sweet spot
        elif spy_close <= long_put_strike or spy_close >= long_call_strike:
            outcome = 'MAX_LOSS'    # Beyond wings
        else:
            outcome = 'PARTIAL_LOSS'  # Hit one side
        
        self.logger.info(f"🎪 IRON CONDOR OUTCOME:")
        self.logger.info(f"   Put Spread: ${short_put_strike}/${long_put_strike} = ${put_spread_credit:.3f}")
        self.logger.info(f"   Call Spread: ${short_call_strike}/${long_call_strike} = ${call_spread_credit:.3f}")
        self.logger.info(f"   Total Credit: ${total_credit:.3f} x {contracts} = ${gross_premium:.2f}")
        self.logger.info(f"   Costs: ${total_costs:.2f}")
        self.logger.info(f"   Net Premium: ${net_premium_received:.2f}")
        self.logger.info(f"   SPY Close: ${spy_close:.2f}")
        self.logger.info(f"   Outcome: {outcome}")
        self.logger.info(f"   Final P&L: ${final_pnl:.2f}")
        
        return {
            'strategy': self.params['strategy_type'],
            'contracts': contracts,
            'short_put_strike': short_put_strike,
            'long_put_strike': long_put_strike,
            'short_call_strike': short_call_strike,
            'long_call_strike': long_call_strike,
            'put_spread_credit': put_spread_credit,
            'call_spread_credit': call_spread_credit,
            'total_credit': total_credit,
            'gross_premium': gross_premium,
            'total_costs': total_costs,
            'net_premium': net_premium_received,
            'spy_close': spy_close,
            'final_pnl': final_pnl,
            'outcome': outcome
        }
    
    def calculate_naked_put_fallback(self, spy_data: pd.DataFrame, strike: float, premium: float) -> Dict:
        """Fallback to naked put if protective put not available"""
        self.logger.warning("⚠️ Using naked put fallback - could not get protective put price")
        
        spy_close = spy_data['close'].iloc[-1]
        contracts = self.params['base_contracts']
        
        # Original naked put logic
        gross_premium = premium * 100 * contracts
        commission = 1.00 * contracts
        bid_ask_spread_cost = 0.02 * 100 * contracts
        slippage = 0.01 * 100 * contracts
        total_costs = commission + bid_ask_spread_cost + slippage
        net_premium_received = gross_premium - total_costs
        
        if spy_close > strike:
            final_pnl = net_premium_received
            outcome = 'NAKED_PUT_EXPIRED'
        else:
            intrinsic_value = (strike - spy_close) * 100 * contracts
            assignment_cost = intrinsic_value + (0.50 * contracts)
            final_pnl = net_premium_received - assignment_cost
            outcome = 'NAKED_PUT_ASSIGNED'
        
        return {
            'strategy': 'naked_put_fallback',
            'contracts': contracts,
            'strike': strike,
            'premium': premium,
            'gross_premium': gross_premium,
            'total_costs': total_costs,
            'net_premium': net_premium_received,
            'spy_close': spy_close,
            'final_pnl': final_pnl,
            'outcome': outcome
        }
    
    def run_single_day(self, date_str: str) -> Dict:
        """Run Iron Condor strategy for a single day"""
        if self.last_trade_date != date_str:
            self.daily_pnl = 0
            self.last_trade_date = date_str
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"🎪 IRON CONDOR 25K STRATEGY - {date_str}")
        self.logger.info(f"🎯 Capital efficient 4-leg strategy")
        self.logger.info(f"{'='*50}")
        
        spy_data = self.load_spy_data(date_str)
        if spy_data is None or spy_data.empty:
            return {'error': 'No SPY data available'}
        
        spy_open = spy_data['open'].iloc[0]
        spy_close = spy_data['close'].iloc[-1]
        spy_high = spy_data['high'].max()
        spy_low = spy_data['low'].min()
        daily_range = (spy_high - spy_low) / spy_open * 100
        
        # Get Iron Condor strikes based on SPY open price
        strikes = self.get_iron_condor_strikes(spy_open)
        short_put_strike, long_put_strike, short_call_strike, long_call_strike = strikes
        
        self.logger.info(f"🎪 IRON CONDOR SETUP:")
        self.logger.info(f"   SPY Open: ${spy_open:.2f}")
        self.logger.info(f"   Put Spread: ${short_put_strike}/${long_put_strike}")
        self.logger.info(f"   Call Spread: ${short_call_strike}/${long_call_strike}")
        self.logger.info(f"   Daily Range: {daily_range:.2f}%")
        
        # Check total credit minimum (use short put as proxy for validation)
        short_put_price = self.get_put_option_price(spy_open, short_put_strike, date_str)
        if short_put_price is None:
            return {'error': 'Could not get option prices'}
        
        # Estimate total credit for validation
        long_put_price = self.get_put_option_price(spy_open, long_put_strike, date_str)
        short_call_price = self.get_call_option_price(spy_open, short_call_strike, date_str)
        long_call_price = self.get_call_option_price(spy_open, long_call_strike, date_str)
        
        if None in [long_put_price, short_call_price, long_call_price]:
            return {'error': 'Could not get all option prices'}
        
        total_credit = (short_put_price - long_put_price) + (short_call_price - long_call_price)
        
        # SAME validation as original balanced (using total credit)
        is_valid, reason = self.validate_trade_conditions(spy_data, short_put_strike, total_credit)
        
        if not is_valid:
            self.logger.info(f"❌ {reason}")
            return {'no_trade': True, 'reason': reason, 'spy_close': spy_close}
        
        # Execute IRON CONDOR trade
        trade_result = self.calculate_iron_condor_outcome(spy_data, strikes, date_str)
        self.daily_pnl += trade_result['final_pnl']
        
        self.logger.info(f"🎪 IRON CONDOR: Total Credit ${total_credit:.3f}, Range {daily_range:.2f}%")
        self.logger.info(f"📊 CONDOR RESULT: ${trade_result['final_pnl']:.2f} ({trade_result['outcome']})")
        
        return {
            'success': True,
            'trade': trade_result,
            'spy_close': spy_close
        }

def main():
    """Main execution for balanced minimal scale strategy"""
    parser = argparse.ArgumentParser(description='Phase 4D Balanced Minimal Scale Strategy')
    parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    parser.add_argument('--cache-dir', default='../../thetadata/cached_data', help='Cache directory')
    
    args = parser.parse_args()
    
    print(f"�� Phase 4D Balanced Minimal Scale Strategy")
    print(f"📈 2x volume, same excellent signal quality")
    print(f"📅 Date: {args.date}")
    
    strategy = Phase4DIronCondor25K(cache_dir=args.cache_dir)
    result = strategy.run_single_day(args.date)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    elif 'no_trade' in result:
        print(f"📊 No trade: {result['reason']}")
    elif 'success' in result:
        trade = result['trade']
        print(f"✅ Trade executed: ${trade['final_pnl']:.2f} ({trade['strategy']})")

def run_multi_month_backtest():
    """Run Iron Condor backtest over 6 months to test capital efficiency"""
    import glob
    
    print("🚀 PHASE 4D IRON CONDOR 25K BACKTEST")
    print("📊 Running 6-month validation to test capital efficiency...")
    print("="*60)
    
    strategy = Phase4DIronCondor25K()
    
    # Get all available SPY data files
    spy_files = glob.glob("../../thetadata/cached_data/spy_bars/spy_bars_*.pkl.gz")
    spy_files.sort()
    
    results = []
    total_pnl = 0
    total_trades = 0
    winning_trades = 0
    
    for spy_file in spy_files[:130]:  # ~6 months of data
        date_str = spy_file.split('_')[-1].replace('.pkl.gz', '')
        
        try:
            result = strategy.run_single_day(date_str)
            
            if 'success' in result:
                trade = result['trade']
                total_pnl += trade['final_pnl']
                total_trades += 1
                
                if trade['final_pnl'] > 0:
                    winning_trades += 1
                
                results.append({
                    'date': date_str,
                    'pnl': trade['final_pnl'],
                    'outcome': trade['outcome'],
                    'strategy': trade['strategy']
                })
                
                print(f"✅ {date_str}: ${trade['final_pnl']:.2f} ({trade['outcome']})")
            elif 'no_trade' in result:
                print(f"⏭️ {date_str}: No trade ({result['reason']})")
            else:
                print(f"❌ {date_str}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ {date_str}: Error - {e}")
    
    print("\n" + "="*60)
    print("📊 IRON CONDOR 25K BACKTEST RESULTS:")
    print("="*60)
    print(f"🎪 Total Trades: {total_trades}")
    print(f"💰 Total P&L: ${total_pnl:.2f}")
    print(f"📊 Average P&L per Trade: ${total_pnl/max(total_trades,1):.2f}")
    print(f"📅 Average Daily P&L: ${total_pnl/max(len(spy_files[:130]),1):.2f}")
    print(f"🏆 Win Rate: {winning_trades/max(total_trades,1)*100:.1f}%")
    print(f"⚡ Execution Rate: {total_trades/len(spy_files[:130])*100:.1f}%")
    print("\n🎯 25K ACCOUNT TARGETS:")
    print(f"   📈 Daily P&L: ${total_pnl/max(len(spy_files[:130]),1):.2f}")
    print(f"   🎯 Daily Target: $300.00")
    print(f"   📊 Performance: {(total_pnl/max(len(spy_files[:130]),1))/300*100:.1f}% of target")
    print("\n🔄 COMPARISON:")
    print(f"   📊 Original Balanced (Naked Puts): 66.7% win rate, +$37.25")
    print(f"   🛡️ Bull Put Spreads: 28.6% win rate, -$143.50")
    print(f"   🎪 Iron Condors: {winning_trades/max(total_trades,1)*100:.1f}% win rate, ${total_pnl:.2f}")
    print("="*60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--backtest":
        run_multi_month_backtest()
    else:
        main()
