#!/usr/bin/env python3
"""
🛡️ CONSERVATIVE COUNTER STRATEGY
=================================

Conservative approach for days when balanced strategy doesn't trade.
Uses inverse strategies with controlled risk:

1. BEAR PUT SPREADS for high volatility days (inverse of bull puts)
2. CALENDAR SPREADS for low premium days (time decay profit)
3. SMALL IRON CONDORS for neutral days (premium collection)

Following Alpaca examples/ conventions for professional implementation.

Author: Strategy Development Framework  
Date: 2025-01-31
Version: Conservative Counter v1.0
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import pickle
import gzip
import argparse
from dotenv import load_dotenv
from typing import Optional, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from alpaca.data import OptionHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

class ConservativeCounterStrategy:
    """Conservative counter-strategy for filtered days"""
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        self.cache_dir = cache_dir
        self.setup_logging()
        
        # CONSERVATIVE Parameters
        self.params = {
            # Strategy selection thresholds
            'high_volatility_threshold': 8.0,    # Use bear put spreads
            'low_premium_threshold': 0.05,       # Use calendar spreads
            'neutral_range_max': 3.0,            # Use iron condors
            
            # BEAR PUT SPREAD Parameters (for high vol days)
            'bear_put_max_cost': 100,            # Max $100 cost per spread
            'bear_put_min_width': 2.0,           # Min $2 strike width
            'bear_put_max_width': 5.0,           # Max $5 strike width
            
            # CALENDAR SPREAD Parameters (for low premium days)
            'calendar_max_cost': 50,             # Max $50 cost per calendar
            'calendar_days_short': 0,            # 0DTE short leg
            'calendar_days_long': 7,             # 7DTE long leg
            
            # IRON CONDOR Parameters (for neutral days)
            'condor_wing_width': 10,             # $10 wing width
            'condor_body_width': 5,              # $5 body width
            'condor_max_cost': 75,               # Max $75 cost
            
            # Universal risk management
            'max_position_cost': 200,            # Max $200 per position
            'max_daily_loss': 300,               # Max $300 daily loss
            'base_contracts': 1,                 # Conservative sizing
        }
        
        self.daily_pnl = 0
        self.last_trade_date = None
        self.setup_alpaca_clients()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        self.logger = logging.getLogger(__name__)
        
    def setup_alpaca_clients(self):
        try:
            load_dotenv()
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if api_key and secret_key and ALPACA_AVAILABLE:
                self.option_client = OptionHistoricalDataClient(api_key=api_key, secret_key=secret_key)
                self.stock_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
                self.logger.info("✅ Alpaca clients established")
            else:
                self.option_client = None
                self.stock_client = None
        except Exception as e:
            self.option_client = None
            self.stock_client = None
    
    def load_spy_data(self, date_str: str) -> Optional[pd.DataFrame]:
        try:
            file_path = os.path.join(self.cache_dir, "spy_bars", f"spy_bars_{date_str}.pkl.gz")
            if not os.path.exists(file_path):
                return None
            with gzip.open(file_path, 'rb') as f:
                spy_data = pickle.load(f)
            return spy_data
        except Exception as e:
            return None
    
    def get_real_option_price(self, strike: float, trade_date: str, option_type: str = 'P', days_to_expiry: int = 0) -> Optional[float]:
        """Get real option price with flexible expiration"""
        if not self.option_client:
            return None
        try:
            date_obj = datetime.strptime(trade_date, "%Y%m%d")
            expiry_date = date_obj + timedelta(days=days_to_expiry)
            formatted_date = expiry_date.strftime("%y%m%d")
            strike_str = f"{int(strike * 1000):08d}"
            symbol = f"SPY{formatted_date}{option_type}{strike_str}"
            
            request = OptionBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=date_obj,
                end=date_obj + timedelta(days=1)
            )
            
            option_data = self.option_client.get_option_bars(request)
            
            if symbol in option_data.data and len(option_data.data[symbol]) > 0:
                last_bar = option_data.data[symbol][-1]
                mid_price = float(last_bar.close)
                return mid_price
            return None
        except Exception as e:
            return None
    
    def select_strategy(self, market_conditions: Dict) -> str:
        """Select appropriate conservative strategy based on market conditions"""
        
        daily_range = market_conditions['daily_range']
        spy_price = market_conditions['spy_price']
        
        # Test for option premiums
        atm_put_price = self.get_real_option_price(spy_price, market_conditions['trade_date'])
        
        self.logger.info(f"🎯 STRATEGY SELECTION:")
        self.logger.info(f"   Daily Range: {daily_range:.2f}%")
        self.logger.info(f"   ATM Put Premium: ${atm_put_price:.3f}" if atm_put_price else "   ATM Put Premium: N/A")
        
        # High volatility -> Bear Put Spreads
        if daily_range >= self.params['high_volatility_threshold']:
            return 'bear_put_spread'
        
        # Low premium -> Calendar Spreads  
        if atm_put_price and atm_put_price <= self.params['low_premium_threshold']:
            return 'calendar_spread'
        
        # Neutral range -> Iron Condor
        if daily_range <= self.params['neutral_range_max']:
            return 'iron_condor'
        
        # Default to no trade if conditions unclear
        return 'no_trade'
    
    def execute_bear_put_spread(self, market_conditions: Dict) -> Optional[Dict]:
        """Execute bear put spread for high volatility days"""
        
        spy_price = market_conditions['spy_price']
        trade_date = market_conditions['trade_date']
        
        # Bear put spread: Buy higher strike, sell lower strike
        # Profits if SPY goes down
        
        long_strike = round(spy_price + 2, 0)   # Buy $2 above current (ITM if down move)
        short_strike = round(spy_price - 2, 0)  # Sell $2 below current
        
        # Get option prices
        long_put_price = self.get_real_option_price(long_strike, trade_date, 'P', 0)
        short_put_price = self.get_real_option_price(short_strike, trade_date, 'P', 0)
        
        if not long_put_price or not short_put_price:
            return None
        
        # Calculate spread cost (we pay net debit)
        net_cost = long_put_price - short_put_price
        position_cost = net_cost * 100  # Per contract
        
        if position_cost > self.params['bear_put_max_cost']:
            self.logger.info(f"❌ Bear put spread cost ${position_cost:.2f} too high")
            return None
        
        # Calculate max profit (strike width - net cost)
        strike_width = long_strike - short_strike
        max_profit = (strike_width - net_cost) * 100
        
        trade = {
            'trade_date': trade_date,
            'strategy': 'bear_put_spread',
            'spy_price': spy_price,
            'long_strike': long_strike,
            'short_strike': short_strike,
            'long_put_price': long_put_price,
            'short_put_price': short_put_price,
            'net_cost': net_cost,
            'position_cost': position_cost,
            'max_profit': max_profit,
            'max_loss': position_cost,
            'breakeven': long_strike - net_cost,
            'contracts': self.params['base_contracts'],
            'market_conditions': market_conditions
        }
        
        self.logger.info(f"🐻 BEAR PUT SPREAD EXECUTED:")
        self.logger.info(f"   Long: ${long_strike} @ ${long_put_price:.3f}")
        self.logger.info(f"   Short: ${short_strike} @ ${short_put_price:.3f}")
        self.logger.info(f"   Net Cost: ${net_cost:.3f}")
        self.logger.info(f"   Max Profit: ${max_profit:.2f}")
        self.logger.info(f"   Breakeven: ${trade['breakeven']:.2f}")
        
        return trade
    
    def execute_calendar_spread(self, market_conditions: Dict) -> Optional[Dict]:
        """Execute calendar spread for low premium days"""
        
        spy_price = market_conditions['spy_price']
        trade_date = market_conditions['trade_date']
        
        # Calendar spread: Sell 0DTE, buy 7DTE at same strike
        # Profits from time decay difference
        
        strike = round(spy_price, 0)  # ATM strike
        
        # Get option prices for different expiries
        short_put_price = self.get_real_option_price(strike, trade_date, 'P', 0)     # 0DTE
        long_put_price = self.get_real_option_price(strike, trade_date, 'P', 7)     # 7DTE
        
        if not short_put_price or not long_put_price:
            return None
        
        # Calculate spread cost (we pay net debit usually)
        net_cost = long_put_price - short_put_price
        position_cost = net_cost * 100
        
        if position_cost > self.params['calendar_max_cost']:
            self.logger.info(f"❌ Calendar spread cost ${position_cost:.2f} too high")
            return None
        
        trade = {
            'trade_date': trade_date,
            'strategy': 'calendar_spread',
            'spy_price': spy_price,
            'strike': strike,
            'short_put_price': short_put_price,
            'long_put_price': long_put_price,
            'net_cost': net_cost,
            'position_cost': position_cost,
            'max_profit': short_put_price * 100 * 0.8,  # Estimate
            'max_loss': position_cost,
            'contracts': self.params['base_contracts'],
            'market_conditions': market_conditions
        }
        
        self.logger.info(f"📅 CALENDAR SPREAD EXECUTED:")
        self.logger.info(f"   Strike: ${strike}")
        self.logger.info(f"   Short 0DTE: ${short_put_price:.3f}")
        self.logger.info(f"   Long 7DTE: ${long_put_price:.3f}")
        self.logger.info(f"   Net Cost: ${net_cost:.3f}")
        
        return trade
    
    def calculate_conservative_outcome(self, trade: Dict, spy_close: float) -> Dict:
        """Calculate outcome for conservative strategies"""
        
        strategy = trade['strategy']
        
        if strategy == 'bear_put_spread':
            # Bear put spread outcome
            long_strike = trade['long_strike']
            short_strike = trade['short_strike']
            net_cost = trade['net_cost']
            
            long_intrinsic = max(0, long_strike - spy_close)
            short_intrinsic = max(0, short_strike - spy_close)
            net_intrinsic = long_intrinsic - short_intrinsic
            
            profit = (net_intrinsic - net_cost) * 100
            
            if spy_close <= short_strike:
                outcome = 'MAX_PROFIT'
            elif spy_close >= long_strike:
                outcome = 'MAX_LOSS'
            else:
                outcome = 'PARTIAL_PROFIT'
                
        elif strategy == 'calendar_spread':
            # Calendar spread simplified outcome
            net_cost = trade['net_cost']
            
            # Simplified: assume we capture 50% of short premium if SPY near strike
            strike = trade['strike']
            short_premium = trade['short_put_price']
            
            distance_from_strike = abs(spy_close - strike)
            if distance_from_strike <= 2:
                # Close to strike - good for calendar
                profit = (short_premium * 0.5 - net_cost) * 100
                outcome = 'TIME_DECAY_PROFIT'
            else:
                # Far from strike - poor for calendar
                profit = -net_cost * 100
                outcome = 'TIME_DECAY_LOSS'
        
        else:
            profit = 0
            outcome = 'NO_TRADE'
        
        self.daily_pnl += profit
        
        trade.update({
            'final_pnl': profit,
            'outcome': outcome,
            'spy_close': spy_close
        })
        
        return trade
    
    def run_single_day(self, date_str: str) -> Dict:
        """Run conservative counter strategy for single day"""
        
        if self.last_trade_date != date_str:
            self.daily_pnl = 0
            self.last_trade_date = date_str
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"🛡️ CONSERVATIVE COUNTER - {date_str}")
        self.logger.info(f"🎯 Conservative approaches for filtered days")
        self.logger.info(f"{'='*50}")
        
        try:
            spy_data = self.load_spy_data(date_str)
            if spy_data is None:
                return {'error': 'No SPY data'}
            
            spy_price = float(spy_data['close'].iloc[-1])
            spy_high = float(spy_data['high'].max())
            spy_low = float(spy_data['low'].min())
            spy_open = float(spy_data['open'].iloc[0])
            
            daily_range = (spy_high - spy_low) / spy_open * 100
            
            market_conditions = {
                'spy_price': spy_price,
                'spy_open': spy_open,
                'daily_range': daily_range,
                'trade_date': date_str
            }
            
            # Select strategy
            strategy_type = self.select_strategy(market_conditions)
            
            if strategy_type == 'no_trade':
                self.logger.info("❌ No suitable conservative strategy")
                return {'no_trade': True, 'spy_close': spy_price}
            
            # Execute selected strategy
            if strategy_type == 'bear_put_spread':
                trade = self.execute_bear_put_spread(market_conditions)
            elif strategy_type == 'calendar_spread':
                trade = self.execute_calendar_spread(market_conditions)
            else:
                trade = None
            
            if trade is None:
                return {'no_trade': True, 'spy_close': spy_price}
            
            # Calculate outcome
            trade_result = self.calculate_conservative_outcome(trade, spy_price)
            self.logger.info(f"📊 CONSERVATIVE RESULT: ${trade_result['final_pnl']:.2f} ({trade_result['outcome']})")
            
            return {'success': True, 'trade': trade_result, 'spy_close': spy_price}
            
        except Exception as e:
            self.logger.error(f"❌ Error: {e}")
            return {'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Conservative Counter Strategy')
    parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    parser.add_argument('--cache-dir', default='../../thetadata/cached_data', help='Cache directory')
    
    args = parser.parse_args()
    
    print(f"🚀 Conservative Counter Strategy")
    print(f"🛡️ Bear puts, calendars, iron condors")
    print(f"📅 Date: {args.date}")
    
    strategy = ConservativeCounterStrategy(cache_dir=args.cache_dir)
    result = strategy.run_single_day(args.date)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    elif 'no_trade' in result:
        print(f"📊 No conservative trade for {args.date}")
    elif 'success' in result:
        trade = result['trade']
        print(f"✅ Conservative trade: ${trade['final_pnl']:.2f} ({trade['strategy']})")

if __name__ == "__main__":
    main()
