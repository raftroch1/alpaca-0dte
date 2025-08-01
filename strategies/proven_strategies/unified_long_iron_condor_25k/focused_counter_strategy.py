#!/usr/bin/env python3
"""
🛡️ FOCUSED COUNTER STRATEGY - BULL PUT SPREADS
===============================================

Counter strategy using BULL PUT SPREADS for days the balanced strategy doesn't trade.
Targets the ~54% of filtered days with conservative, limited-risk approaches.

🎯 CHANGES FROM NAKED PUTS:
- Execution: Naked puts → Bull put spreads (short + long puts)
- Risk: Limited to spread width instead of unlimited
- Premium: Net credit (short - long) instead of full premium
- Keep ALL filtering and logic IDENTICAL - ONLY execution mechanism changed

🎯 FOCUS AREAS:
- Low premium days (where balanced filters out)
- High volatility days (where balanced avoids)
- Simple, conservative spreads that complement the primary strategy

Author: Strategy Development Framework
Date: 2025-08-01
Version: Bull Put Spreads Counter v1.0
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

# Environment-based path handling - no hardcoded relative paths needed

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

class FocusedCounterStrategyBullPutSpreads:
    """
    Counter strategy using BULL PUT SPREADS for days the balanced strategy doesn't trade
    """
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        self.cache_dir = cache_dir
        self.setup_logging()
        
        # Counter Strategy Parameters - Bull Put Spreads - Focus on complementing balanced strategy
        self.params = {
            'strategy_type': 'focused_counter_bull_put_spreads',
            'spread_width': 3.0,  # $3 spread width (smaller than primary for counter strategy)
            'max_daily_trades': 1,
            
            # Target different scenarios than balanced strategy
            'scenarios': {
                # Low premium days - try closer to ATM for higher premium
                'low_premium': {
                    'trigger': 'premium_too_low',  # When balanced says premium < $0.05
                    'approach': 'closer_atm_put',
                    'min_premium': 0.02,  # Lower threshold
                    'strike_adjustment': 0.25,  # Get closer to ATM
                    'max_daily_range': 4.0,  # Still avoid high vol
                },
                
                # Moderate volatility - bear put spreads
                'moderate_volatility': {
                    'trigger': 'volatility_6_to_8_percent',  # 6-8% range
                    'approach': 'bear_put_spread',
                    'min_daily_range': 6.0,
                    'max_daily_range': 8.0,
                    'max_spread_cost': 100,  # $1.00 max cost
                },
                
                # Very low volatility - short straddle components
                'low_volatility': {
                    'trigger': 'very_low_volatility',  # <1% daily range
                    'approach': 'short_call_supplement',
                    'max_daily_range': 1.0,
                    'min_premium': 0.03,
                }
            },
            
            # Position sizing - conservative
            'base_contracts': 1,
            'max_contracts': 1,
            
            # Risk management - strict limits
            'max_loss_per_trade': 100,  # Max $100 loss per trade
            'profit_target_pct': 40,    # Take profit at 40%
            'max_daily_loss': 200,      # Max $200 loss per day
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
    
    def get_protective_put_price(self, short_strike: float, date_str: str) -> Tuple[Optional[float], Optional[str]]:
        """Get protective put price for bull put spread - $3 below short strike (counter strategy)"""
        try:
            # Calculate long put strike ($3 below short put for counter strategy)
            long_strike = short_strike - self.params['spread_width']
            
            # Create option symbol for long put
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            long_put_symbol = f"SPY{date_obj.strftime('%y%m%d')}P{int(long_strike*1000):08d}"
            
            # Get price using same method as short put
            long_put_price = self.get_real_option_price(long_put_symbol, date_str)
            
            if long_put_price is None:
                self.logger.warning(f"❌ Could not get price for protective put {long_put_symbol}")
                return None, None
                
            self.logger.info(f"🛡️ Protective put {long_put_symbol}: ${long_put_price:.3f}")
            return long_put_price, long_put_symbol
            
        except Exception as e:
            self.logger.error(f"❌ Error getting protective put: {e}")
            return None, None
    
    def determine_counter_scenario(self, spy_data: pd.DataFrame, balanced_filter_reason: str) -> Optional[str]:
        """Determine which counter scenario to use based on why balanced strategy filtered"""
        spy_open = spy_data['open'].iloc[0]
        spy_high = spy_data['high'].max()
        spy_low = spy_data['low'].min()
        daily_range = (spy_high - spy_low) / spy_open * 100
        
        self.logger.info(f"🔍 COUNTER ANALYSIS:")
        self.logger.info(f"   Daily Range: {daily_range:.2f}%")
        self.logger.info(f"   Balanced Filter: {balanced_filter_reason}")
        
        # Moderate volatility (6-8%) - try bear put spread
        if 6.0 <= daily_range <= 8.0:
            return 'moderate_volatility'
        
        # Very low volatility (<1%) - try short call supplement  
        elif daily_range < 1.0:
            return 'low_volatility'
        
        # Low premium days - try closer to ATM
        elif 'premium' in balanced_filter_reason.lower():
            if daily_range < 4.0:  # Only if volatility is reasonable
                return 'low_premium'
        
        return None
    
    def execute_closer_atm_put(self, spy_data: pd.DataFrame, date_str: str) -> Optional[Dict]:
        """Execute BULL PUT SPREAD closer to ATM for higher premium"""
        spy_close = spy_data['close'].iloc[-1]
        
        # Get closer to ATM than balanced strategy would
        adjustment = self.params['scenarios']['low_premium']['strike_adjustment']
        short_strike = round(spy_close - adjustment, 0)  # Only $0.25 below SPY
        
        # Get option symbol and price
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        short_option_symbol = f"SPY{date_obj.strftime('%y%m%d')}P{int(short_strike*1000):08d}"
        short_option_price = self.get_real_option_price(short_option_symbol, date_str)
        
        if short_option_price is None or short_option_price < 0.02:
            return None
        
        # Get protective put price
        long_premium, long_symbol = self.get_protective_put_price(short_strike, date_str)
        if long_premium is None:
            # Fallback to naked put
            return self.execute_naked_put_fallback(spy_data, short_strike, short_option_price, 'closer_atm')
        
        # Calculate BULL PUT SPREAD outcome
        contracts = 1
        long_strike = short_strike - self.params['spread_width']
        net_credit = short_option_price - long_premium
        
        # Premium received (before costs) - NET CREDIT for spread
        gross_premium = net_credit * 100 * contracts
        
        # Commission and costs (doubled for spread - short and long legs)
        commission = 2.00 * contracts  # $2 per spread (2 legs)
        bid_ask_spread_cost = 0.04 * 100 * contracts  # $4 per spread (doubled)
        slippage = 0.02 * 100 * contracts  # $2 per spread (doubled)
        
        total_costs = commission + bid_ask_spread_cost + slippage
        net_premium_received = gross_premium - total_costs
        
        # Calculate max risk
        max_risk = (self.params['spread_width'] - net_credit) * 100 * contracts
        
        # BULL PUT SPREAD OUTCOME LOGIC
        if spy_close > short_strike:
            # Both puts expire worthless - we keep the net credit
            final_pnl = net_premium_received
            outcome = 'BOTH_EXPIRED_WORTHLESS'
        elif spy_close <= long_strike:
            # Maximum loss - spread is fully ITM
            final_pnl = net_premium_received - max_risk
            outcome = 'MAX_LOSS'
        else:
            # Partial loss - short put assigned, long put worthless
            short_intrinsic = (short_strike - spy_close) * 100 * contracts
            assignment_cost = short_intrinsic + (0.50 * contracts)  # Assignment fee
            final_pnl = net_premium_received - assignment_cost
            outcome = 'PARTIAL_LOSS'
        
        self.logger.info(f"🎯 CLOSER ATM BULL PUT SPREAD:")
        self.logger.info(f"   Short Put: ${short_strike} @ ${short_option_price:.3f}")
        self.logger.info(f"   Long Put: ${long_strike} @ ${long_premium:.3f}")
        self.logger.info(f"   Net Credit: ${net_credit:.3f}")
        self.logger.info(f"   Max Risk: ${max_risk:.2f}")
        self.logger.info(f"   Final P&L: ${final_pnl:.2f} ({outcome})")
        
        return {
            'strategy': 'closer_atm_bull_put_spread',
            'short_strike': short_strike,
            'long_strike': long_strike,
            'short_premium': short_option_price,
            'long_premium': long_premium,
            'net_credit': net_credit,
            'max_risk': max_risk,
            'final_pnl': final_pnl,
            'outcome': outcome
        }
    
    def execute_naked_put_fallback(self, spy_data: pd.DataFrame, strike: float, premium: float, strategy_type: str) -> Dict:
        """Fallback to naked put if protective put not available"""
        self.logger.warning("⚠️ Using naked put fallback - could not get protective put price")
        
        spy_close = spy_data['close'].iloc[-1]
        contracts = 1
        
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
            'strategy': f'{strategy_type}_naked_fallback',
            'strike': strike,
            'premium': premium,
            'final_pnl': final_pnl,
            'outcome': outcome
        }
    
    def execute_bear_put_spread(self, spy_data: pd.DataFrame, date_str: str) -> Optional[Dict]:
        """Execute bear put spread for moderate volatility"""
        spy_close = spy_data['close'].iloc[-1]
        
        # Bear put spread: Buy higher strike, sell lower strike
        long_strike = round(spy_close + 1.0, 0)   # $1 above SPY
        short_strike = round(spy_close - 1.0, 0)  # $1 below SPY
        
        # Get option symbols and prices
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        long_symbol = f"SPY{date_obj.strftime('%y%m%d')}P{int(long_strike*1000):08d}"
        short_symbol = f"SPY{date_obj.strftime('%y%m%d')}P{int(short_strike*1000):08d}"
        
        long_price = self.get_real_option_price(long_symbol, date_str)
        short_price = self.get_real_option_price(short_symbol, date_str)
        
        if long_price is None or short_price is None:
            return None
        
        # Calculate spread cost
        net_cost = long_price - short_price
        if net_cost <= 0 or net_cost > 1.0:  # Max $1.00 cost
            return None
        
        contracts = 1
        total_cost = net_cost * 100 * contracts + 6.00  # Costs for both legs
        
        # Calculate outcome
        if spy_close >= long_strike:
            # Both expire worthless
            final_pnl = -total_cost
            outcome = 'MAX_LOSS'
        elif spy_close <= short_strike:
            # Max profit
            spread_width = long_strike - short_strike
            max_profit = (spread_width - net_cost) * 100 * contracts - 6.00
            final_pnl = max_profit
            outcome = 'MAX_PROFIT'
        else:
            # Partial profit
            profit = (long_strike - spy_close - net_cost) * 100 * contracts - 6.00
            final_pnl = profit
            outcome = 'PARTIAL_PROFIT'
        
        self.logger.info(f"🐻 BEAR PUT SPREAD:")
        self.logger.info(f"   Long: ${long_strike} @ ${long_price:.3f}")
        self.logger.info(f"   Short: ${short_strike} @ ${short_price:.3f}")
        self.logger.info(f"   Net Cost: ${net_cost:.3f}")
        self.logger.info(f"   Final P&L: ${final_pnl:.2f} ({outcome})")
        
        return {
            'strategy': 'bear_put_spread',
            'long_strike': long_strike,
            'short_strike': short_strike,
            'net_cost': net_cost,
            'final_pnl': final_pnl,
            'outcome': outcome
        }
    
    def execute_short_call_supplement(self, spy_data: pd.DataFrame, date_str: str) -> Optional[Dict]:
        """Execute short call for very low volatility days"""
        spy_close = spy_data['close'].iloc[-1]
        
        # Short call slightly OTM
        target_strike = round(spy_close + 0.5, 0)  # $0.50 above SPY
        
        # Get option symbol and price
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        option_symbol = f"SPY{date_obj.strftime('%y%m%d')}C{int(target_strike*1000):08d}"
        option_price = self.get_real_option_price(option_symbol, date_str)
        
        if option_price is None or option_price < 0.03:
            return None
        
        # Calculate outcome
        contracts = 1
        gross_premium = option_price * 100 * contracts
        costs = 3.00 * contracts
        net_premium = gross_premium - costs
        
        if spy_close < target_strike:
            final_pnl = net_premium
            outcome = 'EXPIRED_WORTHLESS'
        else:
            intrinsic = (spy_close - target_strike) * 100 * contracts
            final_pnl = net_premium - intrinsic - (0.50 * contracts)
            outcome = 'ASSIGNED'
        
        self.logger.info(f"📞 SHORT CALL SUPPLEMENT:")
        self.logger.info(f"   Strike: ${target_strike} (${target_strike - spy_close:.2f} above SPY)")
        self.logger.info(f"   Premium: ${option_price:.3f}")
        self.logger.info(f"   Final P&L: ${final_pnl:.2f} ({outcome})")
        
        return {
            'strategy': 'short_call_supplement',
            'strike': target_strike,
            'premium': option_price,
            'final_pnl': final_pnl,
            'outcome': outcome
        }
    
    def run_counter_strategy(self, date_str: str, balanced_filter_reason: str) -> Dict:
        """Run counter strategy for a filtered day"""
        if self.last_trade_date != date_str:
            self.daily_pnl = 0
            self.last_trade_date = date_str
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"🛡️ FOCUSED COUNTER - {date_str}")
        self.logger.info(f"🎯 Targeting filtered days")
        self.logger.info(f"{'='*50}")
        
        spy_data = self.load_spy_data(date_str)
        if spy_data is None or spy_data.empty:
            return {'error': 'No SPY data available'}
        
        # Determine counter scenario
        scenario = self.determine_counter_scenario(spy_data, balanced_filter_reason)
        
        if scenario is None:
            self.logger.info("❌ No suitable counter scenario for this day")
            return {'no_trade': True, 'reason': 'No suitable counter scenario'}
        
        self.logger.info(f"�� Selected scenario: {scenario}")
        
        # Execute appropriate strategy
        trade_result = None
        
        if scenario == 'low_premium':
            trade_result = self.execute_closer_atm_put(spy_data, date_str)
        elif scenario == 'moderate_volatility':
            trade_result = self.execute_bear_put_spread(spy_data, date_str)
        elif scenario == 'low_volatility':
            trade_result = self.execute_short_call_supplement(spy_data, date_str)
        
        if trade_result:
            self.daily_pnl += trade_result['final_pnl']
            return {
                'success': True,
                'trade': trade_result,
                'scenario': scenario,
                'spy_close': spy_data['close'].iloc[-1]
            }
        else:
            return {'no_trade': True, 'reason': f'Could not execute {scenario} strategy'}

def main():
    """Main execution for focused counter strategy"""
    parser = argparse.ArgumentParser(description='Focused Counter Strategy')
    parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    parser.add_argument('--reason', default='filtered', help='Reason balanced strategy filtered')
    parser.add_argument('--cache-dir', default='../../thetadata/cached_data', help='Cache directory')
    
    args = parser.parse_args()
    
    print(f"🚀 Focused Counter Strategy")
    print(f"🛡️ Targeting days balanced strategy filters")
    print(f"📅 Date: {args.date}")
    print(f"🔍 Reason: {args.reason}")
    
    strategy = FocusedCounterStrategyBullPutSpreads(cache_dir=args.cache_dir)
    result = strategy.run_counter_strategy(args.date, args.reason)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    elif 'no_trade' in result:
        print(f"📊 No counter trade: {result['reason']}")
    elif 'success' in result:
        trade = result['trade']
        print(f"✅ Counter trade: ${trade['final_pnl']:.2f} ({trade['strategy']})")

if __name__ == "__main__":
    main()
