#!/usr/bin/env python3
"""
🔍 DIAGNOSTIC OPTION SCANNER
============================

Scans a range of option strikes to find which ones had meaningful premiums
on a given 0DTE expiration day. This helps us understand real market pricing.

This diagnostic tool will show us:
- What SPY actually closed at
- Which strikes had premiums > $0.05, $0.10, etc.
- The real bid/ask spreads in the market
- Where the tradeable options actually were
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

try:
    from alpaca.data import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  Alpaca SDK not available")

class OptionScanner:
    """Diagnostic scanner for 0DTE option premiums"""
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        self.cache_dir = cache_dir
        self.setup_logging()
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
        """Setup Alpaca client"""
        try:
            load_dotenv()
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if api_key and secret_key and ALPACA_AVAILABLE:
                self.alpaca_client = OptionHistoricalDataClient(
                    api_key=api_key,
                    secret_key=secret_key
                )
                self.logger.info("✅ Alpaca client established")
            else:
                self.alpaca_client = None
                self.logger.error("❌ No Alpaca credentials")
                
        except Exception as e:
            self.logger.error(f"❌ Alpaca setup failed: {e}")
            self.alpaca_client = None
    
    def load_spy_data(self, date_str: str):
        """Load SPY data to get actual market close"""
        try:
            file_path = os.path.join(self.cache_dir, "spy_bars", f"spy_bars_{date_str}.pkl.gz")
            
            with gzip.open(file_path, 'rb') as f:
                spy_data = pickle.load(f)
            
            return spy_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading SPY data: {e}")
            return None
    
    def get_option_price(self, strike: float, trade_date: str):
        """Get option price for specific strike"""
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
                return float(last_bar.close)
            else:
                return None
                
        except Exception as e:
            return None
    
    def scan_strikes(self, spy_price: float, trade_date: str, strike_range: int = 20):
        """Scan range of strikes to find meaningful premiums"""
        
        self.logger.info(f"\n🔍 SCANNING OPTION STRIKES for {trade_date}")
        self.logger.info(f"   SPY Reference Price: ${spy_price:.2f}")
        self.logger.info(f"   Scanning strikes from {spy_price - strike_range} to {spy_price + 5}")
        
        results = []
        
        # Scan strikes from deep ITM to OTM
        for strike in range(int(spy_price - strike_range), int(spy_price + 5), 1):
            price = self.get_option_price(float(strike), trade_date)
            
            if price is not None:
                moneyness = "ITM" if strike < spy_price else "OTM" if strike > spy_price else "ATM"
                distance = abs(strike - spy_price)
                
                results.append({
                    'strike': strike,
                    'price': price,
                    'moneyness': moneyness,
                    'distance': distance
                })
                
                # Color code based on premium level
                if price >= 0.50:
                    color = "🟢"  # High premium
                elif price >= 0.20:
                    color = "🟡"  # Medium premium
                elif price >= 0.05:
                    color = "🟠"  # Low but tradeable
                else:
                    color = "🔴"  # Very low/untradeable
                
                self.logger.info(f"   {color} ${strike} Put: ${price:.3f} ({moneyness}, ${distance:.2f} away)")
        
        return results
    
    def analyze_results(self, results, spy_close: float):
        """Analyze the scanning results"""
        
        self.logger.info(f"\n📊 ANALYSIS RESULTS:")
        self.logger.info(f"   SPY Close: ${spy_close:.2f}")
        
        # Find strikes with different premium levels
        high_premium = [r for r in results if r['price'] >= 0.50]
        medium_premium = [r for r in results if 0.20 <= r['price'] < 0.50]
        low_premium = [r for r in results if 0.05 <= r['price'] < 0.20]
        very_low = [r for r in results if r['price'] < 0.05]
        
        self.logger.info(f"   🟢 High Premium (≥$0.50): {len(high_premium)} strikes")
        if high_premium:
            for r in high_premium[:3]:  # Show first 3
                self.logger.info(f"      ${r['strike']} = ${r['price']:.3f}")
        
        self.logger.info(f"   🟡 Medium Premium ($0.20-$0.50): {len(medium_premium)} strikes")
        if medium_premium:
            for r in medium_premium[:3]:
                self.logger.info(f"      ${r['strike']} = ${r['price']:.3f}")
        
        self.logger.info(f"   🟠 Low Premium ($0.05-$0.20): {len(low_premium)} strikes")
        if low_premium:
            for r in low_premium[:3]:
                self.logger.info(f"      ${r['strike']} = ${r['price']:.3f}")
        
        self.logger.info(f"   🔴 Very Low (<$0.05): {len(very_low)} strikes")
        
        # Find the optimal spread candidates
        tradeable_strikes = [r for r in results if r['price'] >= 0.05]
        
        if len(tradeable_strikes) >= 2:
            self.logger.info(f"\n🎯 POTENTIAL SPREADS:")
            # Try different combinations for $5 spreads
            for i, short in enumerate(tradeable_strikes[:-4]):
                for long in tradeable_strikes[i+1:]:
                    if long['strike'] == short['strike'] - 5:  # $5 spread
                        credit = short['price'] - long['price']
                        max_loss = 5.0 - credit
                        
                        if credit > 0.05:  # Meaningful credit
                            self.logger.info(f"   📈 {short['strike']}/{long['strike']} Spread:")
                            self.logger.info(f"      Credit: ${credit:.3f}, Max Loss: ${max_loss:.3f}")
                            self.logger.info(f"      Short: ${short['price']:.3f}, Long: ${long['price']:.3f}")
        
    def run_diagnostic(self, date_str: str):
        """Run complete diagnostic scan"""
        # Load SPY data to get actual close
        spy_data = self.load_spy_data(date_str)
        if spy_data is None:
            return
        
        spy_close = spy_data['close'].iloc[-1]
        
        # Scan option strikes
        results = self.scan_strikes(spy_close, date_str, strike_range=15)
        
        # Analyze results
        self.analyze_results(results, spy_close)

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Diagnostic Option Scanner')
    parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    parser.add_argument('--cache-dir', default='../../thetadata/cached_data', help='Cache directory')
    
    args = parser.parse_args()
    
    scanner = OptionScanner(cache_dir=args.cache_dir)
    scanner.run_diagnostic(args.date)

if __name__ == "__main__":
    main()