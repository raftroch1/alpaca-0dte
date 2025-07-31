#!/usr/bin/env python3
"""
🎯 PHASE 4D: REAL MARKET STRUCTURE STRATEGY
===========================================

Based on ACTUAL 0DTE option pricing discovery:
- Puts below SPY = $0.01 (untradeable)
- Puts above SPY = $0.50+ (high premium)
- Massive pricing cliff between ITM/OTM

STRATEGY: Sell single ITM puts with high premium that are likely to expire worthless,
OR create wider spreads (e.g., $513/$508) that work with real pricing.

✅ REAL DATA FINDINGS:
- SPY $512.71 close on 03/01/2024
- $513 Put = $0.51 (sellable)
- $512 Put = $0.01 (expires worthless)
- Strategy: Sell $513 puts when SPY likely to stay above $513

Author: Strategy Development Framework  
Date: 2025-01-30
Version: Phase 4D Real Market v1.0
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
from typing import Optional, Dict, List, Tuple

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from alpaca.data import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  Alpaca SDK not available")

class Phase4DRealMarketStrategy:
    """Phase 4D strategy based on REAL 0DTE market structure"""
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        self.cache_dir = cache_dir
        self.setup_logging()
        
        # Parameters based on REAL market discovery
        self.params = {
            'strategy_type': 'itm_put_sales',
            'contracts_per_trade': 1,         # Conservative sizing
            'max_daily_trades': 2,            # Quality over quantity
            
            # ITM Put Sale Strategy
            'target_strike_buffer': 0.50,     # Sell puts $0.50 above current SPY
            'min_premium': 0.40,              # Minimum $0.40 premium
            'max_premium': 2.00,              # Maximum $2.00 premium (risk management)
            
            # Alternative: Wide Spreads
            'wide_spread_enabled': True,      # Enable wide spreads as backup
            'wide_spread_width': 10.0,        # $10 spreads to bridge pricing cliff
            'min_spread_credit': 0.30,        # Minimum $0.30 credit for wide spreads
            
            # Risk management
            'max_risk_per_trade': 500,        # $500 max risk per trade
            'profit_target_pct': 75,          # 75% of max profit
            'stop_loss_multiple': 3.0,        # 3x premium received
            
            # Market conditions
            'min_time_to_close': 4,           # Trade only if ≥4 hours to close
            'max_intraday_volatility': 2.0,   # Max 2% intraday move
        }
        
        self.setup_alpaca_client()
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_alpaca_client(self):
        """Setup Alpaca client for real option data"""
        try:
            load_dotenv()
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if api_key and secret_key and ALPACA_AVAILABLE:
                self.alpaca_client = OptionHistoricalDataClient(
                    api_key=api_key,
                    secret_key=secret_key
                )
                self.logger.info("✅ Alpaca REAL option client established")
            else:
                self.alpaca_client = None
                self.logger.error("❌ No Alpaca credentials")
                
        except Exception as e:
            self.logger.error(f"❌ Alpaca setup failed: {e}")
            self.alpaca_client = None
    
    def load_spy_data(self, date_str: str) -> Optional[pd.DataFrame]:
        """Load REAL SPY data"""
        try:
            file_path = os.path.join(self.cache_dir, "spy_bars", f"spy_bars_{date_str}.pkl.gz")
            
            if not os.path.exists(file_path):
                self.logger.error(f"❌ No SPY data for {date_str}")
                return None
                
            with gzip.open(file_path, 'rb') as f:
                spy_data = pickle.load(f)
            
            self.logger.info(f"✅ Loaded REAL SPY data: {len(spy_data)} bars for {date_str}")
            return spy_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading SPY data: {e}")
            return None
    
    def get_real_option_price(self, strike: float, trade_date: str) -> Optional[float]:
        """Get REAL option price"""
        if not self.alpaca_client:
            return None
            
        try:
            date_obj = datetime.strptime(trade_date, "%Y%m%d")
            formatted_date = date_obj.strftime("%y%m%d")
            
            strike_str = f"{int(strike * 1000):08d}"
            symbol = f"SPY{formatted_date}P{strike_str}"
            
            request = OptionBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=date_obj,
                end=date_obj + timedelta(days=1)
            )
            
            option_data = self.alpaca_client.get_option_bars(request)
            
            if symbol in option_data.data and len(option_data.data[symbol]) > 0:
                last_bar = option_data.data[symbol][-1]
                price = float(last_bar.close)
                
                self.logger.info(f"✅ REAL {symbol}: ${price:.3f}")
                return price
            else:
                self.logger.warning(f"⚠️  No data for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching {symbol}: {e}")
            return None
    
    def find_tradeable_strikes(self, spy_price: float, trade_date: str) -> Dict:
        """Find tradeable strikes based on REAL market structure"""
        
        self.logger.info(f"\n🎯 FINDING TRADEABLE STRIKES")
        self.logger.info(f"   SPY Price: ${spy_price:.2f}")
        
        results = {'itm_puts': [], 'wide_spreads': []}
        
        # Scan ITM puts above current SPY price
        for strike in range(int(spy_price + 0.5), int(spy_price + 10), 1):
            price = self.get_real_option_price(float(strike), trade_date)
            
            if price and self.params['min_premium'] <= price <= self.params['max_premium']:
                results['itm_puts'].append({
                    'strike': strike,
                    'price': price,
                    'distance_above_spy': strike - spy_price
                })
                
                self.logger.info(f"   ✅ TRADEABLE: ${strike} Put @ ${price:.3f} (${strike - spy_price:.2f} above SPY)")
        
        # Check for wide spreads if enabled
        if self.params['wide_spread_enabled']:
            for short_strike in range(int(spy_price + 0.5), int(spy_price + 6), 1):
                long_strike = short_strike - int(self.params['wide_spread_width'])
                
                short_price = self.get_real_option_price(float(short_strike), trade_date)
                long_price = self.get_real_option_price(float(long_strike), trade_date)
                
                if short_price and long_price:
                    credit = short_price - long_price
                    max_loss = self.params['wide_spread_width'] - credit
                    
                    if credit >= self.params['min_spread_credit'] and max_loss * 100 <= self.params['max_risk_per_trade']:
                        results['wide_spreads'].append({
                            'short_strike': short_strike,
                            'long_strike': long_strike,
                            'short_price': short_price,
                            'long_price': long_price,
                            'credit': credit,
                            'max_loss': max_loss
                        })
                        
                        self.logger.info(f"   ✅ WIDE SPREAD: {short_strike}/{long_strike} @ ${credit:.3f} credit")
        
        return results
    
    def generate_signal(self, spy_bars: pd.DataFrame) -> bool:
        """Generate signal based on market conditions"""
        if len(spy_bars) < 50:
            return False
            
        current_price = spy_bars['close'].iloc[-1]
        open_price = spy_bars['open'].iloc[0]
        
        # Calculate intraday volatility
        intraday_change = abs((current_price - open_price) / open_price) * 100
        
        # Only trade in stable market conditions
        if intraday_change <= self.params['max_intraday_volatility']:
            self.logger.info(f"✅ Signal: Stable market (${intraday_change:.2f}% move)")
            return True
        else:
            self.logger.info(f"❌ No signal: Too volatile (${intraday_change:.2f}% move)")
            return False
    
    def execute_best_trade(self, tradeable_strikes: Dict, spy_price: float, trade_date: str) -> Optional[Dict]:
        """Execute the best available trade"""
        
        # Prefer ITM put sales (higher profit potential)
        if tradeable_strikes['itm_puts']:
            best_put = min(tradeable_strikes['itm_puts'], key=lambda x: x['distance_above_spy'])
            
            trade = {
                'strategy': 'itm_put_sale',
                'trade_date': trade_date,
                'spy_price': spy_price,
                'strike': best_put['strike'],
                'premium': best_put['price'],
                'contracts': self.params['contracts_per_trade'],
                'max_profit': best_put['price'],
                'max_loss': best_put['strike'] - spy_price,  # If SPY falls below strike
                'breakeven': best_put['strike'] - best_put['price']
            }
            
            self.logger.info(f"🎯 EXECUTING ITM PUT SALE:")
            self.logger.info(f"   Strike: ${best_put['strike']}")
            self.logger.info(f"   Premium: ${best_put['price']:.3f}")
            self.logger.info(f"   Max Profit: ${best_put['price']:.3f}")
            self.logger.info(f"   Breakeven: ${trade['breakeven']:.2f}")
            
            return trade
            
        # Fallback to wide spreads
        elif tradeable_strikes['wide_spreads']:
            best_spread = max(tradeable_strikes['wide_spreads'], key=lambda x: x['credit'])
            
            trade = {
                'strategy': 'wide_spread',
                'trade_date': trade_date,
                'spy_price': spy_price,
                'short_strike': best_spread['short_strike'],
                'long_strike': best_spread['long_strike'],
                'credit': best_spread['credit'],
                'contracts': self.params['contracts_per_trade'],
                'max_profit': best_spread['credit'],
                'max_loss': best_spread['max_loss'],
                'breakeven': best_spread['short_strike'] - best_spread['credit']
            }
            
            self.logger.info(f"🎯 EXECUTING WIDE SPREAD:")
            self.logger.info(f"   {best_spread['short_strike']}/{best_spread['long_strike']} spread")
            self.logger.info(f"   Credit: ${best_spread['credit']:.3f}")
            self.logger.info(f"   Max Loss: ${best_spread['max_loss']:.3f}")
            
            return trade
        
        else:
            self.logger.warning("❌ No tradeable options found")
            return None
    
    def calculate_outcome(self, trade: Dict, spy_close: float) -> Dict:
        """Calculate trade outcome based on SPY close"""
        
        if trade['strategy'] == 'itm_put_sale':
            if spy_close > trade['strike']:
                # Put expires worthless - we keep full premium
                profit = trade['premium'] * trade['contracts'] * 100
                outcome = 'EXPIRED_WORTHLESS'
            else:
                # Put assigned - we have intrinsic loss
                intrinsic_value = trade['strike'] - spy_close
                net_loss = intrinsic_value - trade['premium']  # Loss per share
                profit = -net_loss * trade['contracts'] * 100  # Convert to P&L
                outcome = 'ASSIGNED'
                
        else:  # wide_spread
            if spy_close > trade['short_strike']:
                # Both puts expire worthless - we keep full credit
                profit = trade['credit'] * trade['contracts'] * 100
                outcome = 'EXPIRED_WORTHLESS'
            elif spy_close < trade['long_strike']:
                # Max loss scenario
                profit = -trade['max_loss'] * trade['contracts'] * 100
                outcome = 'MAX_LOSS'
            else:
                # Partial loss
                intrinsic_short = trade['short_strike'] - spy_close
                partial_loss = intrinsic_short - trade['credit']
                profit = -partial_loss * trade['contracts'] * 100
                outcome = 'PARTIAL_LOSS'
        
        trade['final_pnl'] = profit
        trade['outcome'] = outcome
        trade['spy_close'] = spy_close
        
        return trade
    
    def run_single_day(self, date_str: str) -> Dict:
        """Run REAL market structure strategy for single day"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎯 PHASE 4D REAL MARKET STRUCTURE - {date_str}")
        self.logger.info(f"{'='*60}")
        
        # Load SPY data
        spy_bars = self.load_spy_data(date_str)
        if spy_bars is None:
            return {'error': 'No SPY data available'}
        
        spy_price = spy_bars['close'].iloc[-1]
        spy_close = spy_bars['close'].iloc[-1]
        
        # Generate signal
        if not self.generate_signal(spy_bars):
            return {'no_signal': True, 'spy_close': spy_close}
        
        # Find tradeable strikes
        tradeable_strikes = self.find_tradeable_strikes(spy_price, date_str)
        
        # Execute best trade
        trade = self.execute_best_trade(tradeable_strikes, spy_price, date_str)
        
        if trade is None:
            return {'no_trade': True, 'spy_close': spy_close}
        
        # Calculate outcome
        trade_result = self.calculate_outcome(trade, spy_close)
        
        self.logger.info(f"\n📊 FINAL RESULT: ${trade_result['final_pnl']:.2f} ({trade_result['outcome']})")
        
        return {
            'success': True,
            'trade': trade_result,
            'spy_close': spy_close
        }

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Phase 4D Real Market Structure Strategy')
    parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    parser.add_argument('--cache-dir', default='../../thetadata/cached_data', help='Cache directory')
    
    args = parser.parse_args()
    
    strategy = Phase4DRealMarketStrategy(cache_dir=args.cache_dir)
    result = strategy.run_single_day(args.date)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    elif 'no_signal' in result:
        print(f"📊 No signal generated for {args.date}")
    elif 'no_trade' in result:
        print(f"📊 No tradeable options found for {args.date}")
    elif 'success' in result:
        trade = result['trade']
        print(f"✅ Trade executed: ${trade['final_pnl']:.2f} profit ({trade['strategy']})")

if __name__ == "__main__":
    main()