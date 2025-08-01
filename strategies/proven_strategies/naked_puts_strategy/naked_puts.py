#!/usr/bin/env python3
"""
🎯 PHASE 4D: BALANCED STRATEGY
==============================

Balanced version that finds the sweet spot between risk management and trade execution.
Based on lessons learned from the full month test where the optimized version was too conservative.

🔧 BALANCED APPROACH:
- Relaxed premium requirements (tradeable options)
- Reasonable volatility filters (avoid disasters only)
- Smarter strike selection (closer to ATM for meaningful premiums)
- Maintains core risk management principles

📊 TARGET: Execute ~50% of trading days while avoiding major losses

Author: Strategy Development Framework
Date: 2025-01-30
Version: Phase 4D Balanced v1.0
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
import math

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from alpaca.data import OptionHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  Alpaca SDK not available")

class Phase4DBalancedStrategy:
    """
    Balanced Phase 4D strategy - trades more frequently while maintaining risk control
    """
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        self.cache_dir = cache_dir
        self.setup_logging()
        
        # BALANCED Parameters - Trade execution focused
        self.params = {
            # Core strategy
            'strategy_type': 'balanced_put_sales',
            'max_daily_trades': 1,
            
            # RELAXED Strike Selection (closer to ATM for meaningful premiums)
            'min_strike_buffer': 0.5,         # Minimum $0.50 below SPY (much closer)
            'max_strike_buffer': 3.0,         # Maximum $3.00 below SPY  
            'target_delta_range': (-0.25, -0.10),  # 10-25 delta puts (closer to ATM)
            'min_premium': 0.05,              # Minimum $0.05 premium (RELAXED)
            'max_premium': 2.00,              # Maximum $2.00 premium
            
            # BALANCED VOLATILITY FILTERING (avoid disasters only)
            'max_vix_threshold': 30,          # Only avoid extreme VIX > 30
            'max_daily_range': 5.0,           # Only avoid extreme daily range > 5%
            'disaster_threshold': 8.0,        # Disaster = >8% daily range
            
            # FLEXIBLE POSITION SIZING
            'base_contracts': 1,              # Base position size
            'max_contracts': 2,               # Maximum contracts
            
            # BALANCED RISK MANAGEMENT
            'max_loss_per_trade': 200,        # Max $200 loss per trade
            'profit_target_pct': 50,          # Take profit at 50%
            'stop_loss_multiple': 3.0,        # Stop at 3x premium received
            'max_daily_loss': 400,            # Max $400 loss per day
        }
        
        # State tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
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
        """Load SPY data"""
        try:
            file_path = os.path.join(self.cache_dir, "spy_bars", f"spy_bars_{date_str}.pkl.gz")
            
            if not os.path.exists(file_path):
                return None
                
            with gzip.open(file_path, 'rb') as f:
                spy_data = pickle.load(f)
            
            return spy_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading SPY data: {e}")
            return None
    
    def check_market_conditions(self, date_str: str) -> Dict:
        """
        Check market conditions with balanced filtering
        """
        try:
            current_data = self.load_spy_data(date_str)
            if current_data is None:
                return {'favorable': False, 'reason': 'No data'}
            
            # Calculate daily metrics
            open_price = current_data['open'].iloc[0]
            close_price = current_data['close'].iloc[-1]
            high_price = current_data['high'].max()
            low_price = current_data['low'].min()
            
            daily_return = (close_price - open_price) / open_price * 100
            daily_range = (high_price - low_price) / open_price * 100
            
            # Only filter out extreme volatility (disasters)
            if daily_range > self.params['disaster_threshold']:
                return {
                    'favorable': False,
                    'reason': f'Disaster-level volatility: {daily_range:.2f}%',
                    'daily_return': daily_return,
                    'daily_range': daily_range,
                    'severity': 'DISASTER'
                }
            
            # Allow moderate volatility but note it
            volatility_level = 'LOW' if daily_range < 1.0 else 'MODERATE' if daily_range < 3.0 else 'HIGH'
            
            return {
                'favorable': True,
                'daily_return': daily_return,
                'daily_range': daily_range,
                'volatility_level': volatility_level,
                'market_sentiment': 'bullish' if daily_return > 0.5 else 'bearish' if daily_return < -0.5 else 'neutral'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error checking market conditions: {e}")
            return {'favorable': False, 'reason': f'Analysis error: {e}'}
    
    def get_balanced_strike(self, spy_price: float, daily_range: float) -> float:
        """
        Calculate balanced strike selection - target ITM puts with meaningful premiums
        Based on 0DTE pricing cliff analysis: OTM puts = $0.01, ITM puts = $0.50+
        """
        # For 0DTE, we need to go ITM to get meaningful premiums
        # Target slightly ITM puts that will likely expire worthless due to small moves
        
        if daily_range < 1.0:
            # Low volatility - go slightly ITM for premium
            buffer = -0.25  # $0.25 ABOVE current SPY (ITM)
        elif daily_range < 3.0:
            # Moderate volatility - slightly less ITM
            buffer = 0.0   # Right at current SPY
        else:
            # High volatility - stay OTM but close
            buffer = self.params['min_strike_buffer']
        
        optimal_strike = round(spy_price - buffer, 0)
        
        # Ensure we're targeting the pricing cliff efficiently
        # If SPY is $512.71, target $513 put (OTM with premium) or $512 put (ITM)
        
        self.logger.info(f"🎯 BALANCED STRIKE SELECTION (PRICING CLIFF AWARE):")
        self.logger.info(f"   SPY: ${spy_price:.2f}")
        self.logger.info(f"   Daily Range: {daily_range:.2f}%")
        self.logger.info(f"   Buffer: ${buffer:.2f}")
        self.logger.info(f"   Strike: ${optimal_strike}")
        self.logger.info(f"   Strategy: Target {'ITM' if buffer <= 0 else 'OTM'} puts with meaningful premiums")
        
        return optimal_strike
    
    def get_real_option_price(self, strike: float, trade_date: str) -> Optional[float]:
        """Get real option price from Alpaca"""
        if not self.option_client:
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
            
            option_data = self.option_client.get_option_bars(request)
            
            if symbol in option_data.data and len(option_data.data[symbol]) > 0:
                last_bar = option_data.data[symbol][-1]
                price = float(last_bar.close)
                
                self.logger.info(f"✅ REAL {symbol}: ${price:.3f}")
                return price
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching {symbol}: {e}")
            return None
    
    def validate_balanced_trade(self, market_conditions: Dict) -> Dict:
        """
        Balanced trade validation - only filter disasters
        """
        # Check daily loss limit
        if abs(self.daily_pnl) >= self.params['max_daily_loss']:
            return {'valid': False, 'reason': f'Daily loss limit: ${self.daily_pnl:.2f}'}
        
        # Check market conditions (only disasters filtered)
        if not market_conditions['favorable']:
            return {'valid': False, 'reason': market_conditions['reason']}
        
        return {'valid': True}
    
    def execute_balanced_trade(self, spy_data: pd.DataFrame, trade_date: str) -> Optional[Dict]:
        """
        Execute balanced trade with relaxed parameters
        """
        
        spy_price = spy_data['close'].iloc[-1]
        
        # Check market conditions
        market_conditions = self.check_market_conditions(trade_date)
        
        # Validate trade
        validation = self.validate_balanced_trade(market_conditions)
        if not validation['valid']:
            self.logger.info(f"❌ Trade validation failed: {validation['reason']}")
            return None
        
        # Get balanced strike
        daily_range = market_conditions.get('daily_range', 1.0)
        balanced_strike = self.get_balanced_strike(spy_price, daily_range)
        
        # Get real option price
        option_price = self.get_real_option_price(balanced_strike, trade_date)
        if option_price is None:
            self.logger.warning(f"❌ No option price for ${balanced_strike} strike")
            return None
        
        # Validate premium range (relaxed)
        if not (self.params['min_premium'] <= option_price <= self.params['max_premium']):
            self.logger.warning(f"❌ Premium ${option_price:.3f} outside range ${self.params['min_premium']}-${self.params['max_premium']}")
            return None
        
        # Position sizing (simple for now)
        position_size = self.params['base_contracts']
        
        # Calculate trade metrics
        max_loss_per_share = balanced_strike - spy_price  # Negative value
        max_loss_total = abs(max_loss_per_share) * position_size * 100
        
        # Risk check
        if max_loss_total > self.params['max_loss_per_trade']:
            self.logger.warning(f"❌ Max loss ${max_loss_total:.0f} exceeds limit ${self.params['max_loss_per_trade']}")
            return None
        
        # Create trade
        trade = {
            'trade_date': trade_date,
            'strategy': 'balanced_put_sale',
            'spy_price': spy_price,
            'strike': balanced_strike,
            'premium': option_price,
            'contracts': position_size,
            'max_profit': option_price,
            'max_loss': abs(max_loss_per_share),
            'breakeven': balanced_strike - option_price,
            'market_conditions': market_conditions,
            'position_rationale': f"Range:{daily_range:.2f}%, Buffer:${balanced_strike-spy_price:.2f}",
            'risk_reward_ratio': option_price / abs(max_loss_per_share) if max_loss_per_share != 0 else float('inf')
        }
        
        self.logger.info(f"🎯 BALANCED TRADE EXECUTED:")
        self.logger.info(f"   Strike: ${balanced_strike} (${balanced_strike-spy_price:.2f} below SPY)")
        self.logger.info(f"   Premium: ${option_price:.3f}")
        self.logger.info(f"   Max Profit: ${option_price * position_size * 100:.0f}")
        self.logger.info(f"   Max Loss: ${max_loss_total:.0f}")
        self.logger.info(f"   Risk/Reward: 1:{trade['risk_reward_ratio']:.2f}")
        
        return trade
    
    def calculate_trade_outcome(self, trade: Dict, spy_close: float) -> Dict:
        """Calculate trade outcome"""
        if spy_close > trade['strike']:
            # Put expires worthless - we keep full premium
            profit = trade['premium'] * trade['contracts'] * 100
            outcome = 'EXPIRED_WORTHLESS'
            
        else:
            # Put assigned - calculate loss
            intrinsic_value = trade['strike'] - spy_close
            net_loss = intrinsic_value - trade['premium']
            profit = -net_loss * trade['contracts'] * 100
            outcome = 'ASSIGNED'
        
        # Update tracking
        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        self.daily_pnl += profit
        
        trade['final_pnl'] = profit
        trade['outcome'] = outcome
        trade['spy_close'] = spy_close
        
        return trade
    
    def run_single_day(self, date_str: str) -> Dict:
        """Run balanced strategy for single day"""
        
        # Reset daily P&L if new day
        if self.last_trade_date != date_str:
            self.daily_pnl = 0
            self.last_trade_date = date_str
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎯 PHASE 4D BALANCED - {date_str}")
        self.logger.info(f"📊 Daily P&L: ${self.daily_pnl:.2f}")
        self.logger.info(f"{'='*60}")
        
        # Load SPY data
        spy_data = self.load_spy_data(date_str)
        if spy_data is None:
            return {'error': 'No SPY data available'}
        
        spy_close = spy_data['close'].iloc[-1]
        
        # Execute balanced trade
        trade = self.execute_balanced_trade(spy_data, date_str)
        
        if trade is None:
            return {'no_trade': True, 'spy_close': spy_close}
        
        # Calculate outcome
        trade_result = self.calculate_trade_outcome(trade, spy_close)
        
        self.logger.info(f"\n📊 BALANCED RESULT: ${trade_result['final_pnl']:.2f} ({trade_result['outcome']})")
        
        return {
            'success': True,
            'trade': trade_result,
            'spy_close': spy_close
        }

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Phase 4D Balanced Strategy')
    parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    parser.add_argument('--cache-dir', default='../../thetadata/cached_data', help='Cache directory')
    
    args = parser.parse_args()
    
    strategy = Phase4DBalancedStrategy(cache_dir=args.cache_dir)
    result = strategy.run_single_day(args.date)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    elif 'no_trade' in result:
        print(f"📊 No trade executed for {args.date}")
    elif 'success' in result:
        trade = result['trade']
        print(f"✅ Balanced trade: ${trade['final_pnl']:.2f} profit ({trade['strategy']})")

if __name__ == "__main__":
    main()