#!/usr/bin/env python3
"""
📈 PHASE 4D BALANCED - MINIMAL SCALE
=====================================

The balanced strategy with MINIMAL scaling - keeping signal quality intact.
Only minor adjustments for slightly higher execution rate while maintaining excellence.

🎯 CHANGES FROM ORIGINAL BALANCED:
- Position size: 1→2 contracts (simple volume increase)
- Max daily loss: $400→$500 (proportional to position increase)
- Keep ALL filtering and signal quality intact

Author: Strategy Development Framework  
Date: 2025-01-31
Version: Balanced Minimal Scale v1.0
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

class Phase4DBalancedMinimalScale:
    """
    Balanced strategy with minimal scaling - keeping ALL signal quality intact
    """
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        self.cache_dir = cache_dir
        self.setup_logging()
        
        # MINIMAL SCALE Parameters - Keep signal quality, just increase volume slightly
        self.params = {
            # Core strategy - UNCHANGED
            'strategy_type': 'balanced_minimal_scale_put_sale',
            'max_daily_trades': 1,
            
            # Strike Selection - KEEP ORIGINAL QUALITY
            'min_strike_buffer': 0.5,         # SAME as balanced
            'max_strike_buffer': 3.0,         # SAME as balanced  
            'target_delta_range': (-0.25, -0.10),  # SAME as balanced
            'min_premium': 0.05,              # SAME as balanced (quality threshold)
            'max_premium': 2.00,              # SAME as balanced
            
            # Volatility Filtering - KEEP PROTECTIVE FILTERING
            'max_vix_threshold': 25,          # SAME as balanced
            'max_daily_range': 5.0,           # SAME as balanced (protective)
            'disaster_threshold': 8.0,        # SAME as balanced
            
            # Position Sizing - ONLY CHANGE: Minimal volume increase
            'base_contracts': 2,              # 1→2 (simple 2x scale)
            'max_contracts': 2,               # Keep it simple
            
            # Risk Management - PROPORTIONAL to position increase
            'max_loss_per_trade': 200,        # SAME per contract
            'profit_target_pct': 50,          # SAME as balanced
            'stop_loss_multiple': 3.0,        # SAME as balanced
            'max_daily_loss': 500,            # $400→$500 (proportional)
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
    
    def calculate_trade_outcome(self, spy_data: pd.DataFrame, strike: float, premium: float) -> Dict:
        """Calculate trade outcome with realistic costs (same as original)"""
        spy_close = spy_data['close'].iloc[-1]
        contracts = self.params['base_contracts']  # Now 2 instead of 1
        
        # Premium received (before costs)
        gross_premium = premium * 100 * contracts
        
        # Commission and costs (per contract)
        commission = 1.00 * contracts  # $1 per contract
        bid_ask_spread_cost = 0.02 * 100 * contracts  # $2 per contract
        slippage = 0.01 * 100 * contracts  # $1 per contract
        
        total_costs = commission + bid_ask_spread_cost + slippage
        net_premium_received = gross_premium - total_costs
        
        # Determine outcome
        if spy_close > strike:
            # Option expires worthless - we keep the premium
            final_pnl = net_premium_received
            outcome = 'EXPIRED_WORTHLESS'
        else:
            # Option assigned
            intrinsic_value = (strike - spy_close) * 100 * contracts
            assignment_cost = intrinsic_value + (0.50 * contracts)  # $0.50 assignment fee per contract
            final_pnl = net_premium_received - assignment_cost
            outcome = 'ASSIGNED'
        
        self.logger.info(f"💰 TRADE OUTCOME:")
        self.logger.info(f"   Premium: ${premium:.3f} x {contracts} = ${gross_premium:.2f}")
        self.logger.info(f"   Costs: ${total_costs:.2f}")
        self.logger.info(f"   Net Premium: ${net_premium_received:.2f}")
        self.logger.info(f"   SPY Close: ${spy_close:.2f} vs Strike: ${strike:.0f}")
        self.logger.info(f"   Outcome: {outcome}")
        self.logger.info(f"   Final P&L: ${final_pnl:.2f}")
        
        return {
            'strategy': self.params['strategy_type'],
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
        """Run strategy for a single day (same logic as original balanced)"""
        if self.last_trade_date != date_str:
            self.daily_pnl = 0
            self.last_trade_date = date_str
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"📈 BALANCED MINIMAL SCALE - {date_str}")
        self.logger.info(f"🎯 2x volume, same signal quality")
        self.logger.info(f"{'='*50}")
        
        spy_data = self.load_spy_data(date_str)
        if spy_data is None or spy_data.empty:
            return {'error': 'No SPY data available'}
        
        spy_open = spy_data['open'].iloc[0]
        spy_close = spy_data['close'].iloc[-1]
        spy_high = spy_data['high'].max()
        spy_low = spy_data['low'].min()
        daily_range = (spy_high - spy_low) / spy_open * 100
        
        # Use SAME strike selection as original balanced
        target_strike = self.get_balanced_strike(spy_close, daily_range)
        
        # Get option symbol
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        option_symbol = f"SPY{date_obj.strftime('%y%m%d')}P{int(target_strike*1000):08d}"
        
        # Get real option price
        option_price = self.get_real_option_price(option_symbol, date_str)
        
        if option_price is None:
            return {'error': 'Could not get option price'}
        
        # SAME validation as original balanced
        is_valid, reason = self.validate_trade_conditions(spy_data, target_strike, option_price)
        
        if not is_valid:
            self.logger.info(f"❌ {reason}")
            return {'no_trade': True, 'reason': reason, 'spy_close': spy_close}
        
        # Execute trade
        trade_result = self.calculate_trade_outcome(spy_data, target_strike, option_price)
        self.daily_pnl += trade_result['final_pnl']
        
        self.logger.info(f"📈 MINIMAL SCALE TRADE: Strike ${target_strike}, Premium ${option_price:.3f}, Range {daily_range:.2f}%")
        self.logger.info(f"📊 MINIMAL SCALE RESULT: ${trade_result['final_pnl']:.2f} ({trade_result['outcome']})")
        
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
    
    strategy = Phase4DBalancedMinimalScale(cache_dir=args.cache_dir)
    result = strategy.run_single_day(args.date)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    elif 'no_trade' in result:
        print(f"📊 No trade: {result['reason']}")
    elif 'success' in result:
        trade = result['trade']
        print(f"✅ Trade executed: ${trade['final_pnl']:.2f} ({trade['strategy']})")

if __name__ == "__main__":
    main()
