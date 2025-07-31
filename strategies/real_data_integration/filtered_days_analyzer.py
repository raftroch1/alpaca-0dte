#!/usr/bin/env python3
"""
🔍 FILTERED DAYS ANALYZER
=========================

Analyzes the 67 days when balanced strategy didn't trade to identify 
counter-strategy opportunities.

Author: Strategy Development Framework
Date: 2025-01-31
Version: Filtered Days Analysis v1.0
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import pickle
import gzip
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

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

class FilteredDaysAnalyzer:
    """
    Analyze why balanced strategy filtered days and identify counter opportunities
    """
    
    def __init__(self, cache_dir: str = "../../thetadata/cached_data"):
        self.cache_dir = cache_dir
        self.setup_logging()
        self.setup_alpaca_clients()
        
        # Filter reason categories
        self.filter_categories = {
            'disaster_volatility': [],  # >8% daily range
            'low_premium': [],          # <$0.05 options
            'high_premium': [],         # >$2.00 options
            'missing_data': [],         # No option data
            'other': []                 # Other reasons
        }
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_alpaca_clients(self):
        """Setup Alpaca clients"""
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
                self.logger.warning("⚠️  No Alpaca credentials - limited analysis")
                
        except Exception as e:
            self.logger.error(f"❌ Alpaca setup failed: {e}")
            self.option_client = None
            self.stock_client = None
    
    def load_spy_data(self, date_str: str) -> Optional[pd.DataFrame]:
        """Load SPY data for a date"""
        try:
            file_path = os.path.join(self.cache_dir, "spy_bars", f"spy_bars_{date_str}.pkl.gz")
            
            if not os.path.exists(file_path):
                return None
                
            with gzip.open(file_path, 'rb') as f:
                spy_data = pickle.load(f)
            
            return spy_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading SPY data for {date_str}: {e}")
            return None
    
    def analyze_day_conditions(self, date_str: str) -> Dict:
        """Analyze market conditions for a specific day"""
        try:
            spy_data = self.load_spy_data(date_str)
            if spy_data is None:
                return {'date': date_str, 'error': 'No SPY data'}
            
            # Calculate basic metrics
            open_price = spy_data['open'].iloc[0]
            close_price = spy_data['close'].iloc[-1]
            high_price = spy_data['high'].max()
            low_price = spy_data['low'].min()
            
            daily_return = (close_price - open_price) / open_price * 100
            daily_range = (high_price - low_price) / open_price * 100
            
            # Volatility classification
            if daily_range > 8.0:
                volatility_class = 'DISASTER'
            elif daily_range > 5.0:
                volatility_class = 'HIGH'
            elif daily_range > 3.0:
                volatility_class = 'MODERATE'
            else:
                volatility_class = 'LOW'
            
            # Market sentiment
            if daily_return > 1.0:
                sentiment = 'STRONG_BULLISH'
            elif daily_return > 0.5:
                sentiment = 'BULLISH'
            elif daily_return > -0.5:
                sentiment = 'NEUTRAL'
            elif daily_return > -1.0:
                sentiment = 'BEARISH'
            else:
                sentiment = 'STRONG_BEARISH'
            
            return {
                'date': date_str,
                'spy_open': open_price,
                'spy_close': close_price,
                'spy_high': high_price,
                'spy_low': low_price,
                'daily_return': daily_return,
                'daily_range': daily_range,
                'volatility_class': volatility_class,
                'sentiment': sentiment,
                'volume': spy_data['volume'].sum()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing {date_str}: {e}")
            return {'date': date_str, 'error': str(e)}
    
    def categorize_filter_reason(self, day_analysis: Dict) -> str:
        """Categorize why a day was filtered"""
        
        # Check for disaster volatility
        if day_analysis.get('volatility_class') == 'DISASTER':
            return 'disaster_volatility'
        
        # Check for high volatility 
        if day_analysis.get('volatility_class') == 'HIGH':
            return 'high_volatility'
        
        # For now, categorize others as 'other'
        return 'other'
    
    def analyze_filtered_days(self, filtered_dates: List[str]) -> Dict:
        """Comprehensive analysis of all filtered days"""
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔍 ANALYZING {len(filtered_dates)} FILTERED DAYS")
        self.logger.info(f"{'='*80}")
        
        detailed_analysis = []
        
        for i, date_str in enumerate(filtered_dates, 1):
            self.logger.info(f"📊 Analyzing {date_str} ({i}/{len(filtered_dates)})")
            
            day_analysis = self.analyze_day_conditions(date_str)
            filter_reason = self.categorize_filter_reason(day_analysis)
            
            day_analysis['filter_reason'] = filter_reason
            detailed_analysis.append(day_analysis)
            
            # Categorize for summary
            self.filter_categories[filter_reason].append(day_analysis)
            
            # Log key findings
            if 'daily_range' in day_analysis:
                self.logger.info(f"   📈 {date_str}: {day_analysis['daily_range']:.2f}% range, {day_analysis['sentiment']}")
        
        # Generate report
        self.generate_filter_analysis_report(detailed_analysis)
        
        return {
            'detailed_analysis': detailed_analysis,
            'filter_categories': self.filter_categories,
            'summary_stats': self.calculate_filter_stats()
        }
    
    def calculate_filter_stats(self) -> Dict:
        """Calculate summary statistics for filtered days"""
        
        total_filtered = sum(len(days) for days in self.filter_categories.values())
        
        if total_filtered == 0:
            return {}
        
        stats = {
            'total_filtered_days': total_filtered,
            'disaster_volatility_pct': len(self.filter_categories['disaster_volatility']) / total_filtered * 100,
        }
        
        # Volatility analysis
        all_ranges = []
        all_returns = []
        
        for category_days in self.filter_categories.values():
            for day in category_days:
                if 'daily_range' in day:
                    all_ranges.append(day['daily_range'])
                if 'daily_return' in day:
                    all_returns.append(day['daily_return'])
        
        if all_ranges:
            stats['avg_daily_range'] = np.mean(all_ranges)
            stats['max_daily_range'] = np.max(all_ranges)
            stats['min_daily_range'] = np.min(all_ranges)
        
        if all_returns:
            stats['avg_daily_return'] = np.mean(all_returns)
            stats['volatility'] = np.std(all_returns)
        
        return stats
    
    def generate_filter_analysis_report(self, detailed_analysis: List[Dict]):
        """Generate comprehensive filter analysis report"""
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"📈 FILTERED DAYS ANALYSIS REPORT")
        self.logger.info(f"{'='*80}")
        
        stats = self.calculate_filter_stats()
        
        if stats:
            self.logger.info(f"\n📊 FILTER REASON BREAKDOWN:")
            self.logger.info(f"   💥 Disaster Volatility: {len(self.filter_categories['disaster_volatility'])} days ({stats.get('disaster_volatility_pct', 0):.1f}%)")
            
            if 'avg_daily_range' in stats:
                self.logger.info(f"\n📊 VOLATILITY ANALYSIS:")
                self.logger.info(f"   Average Daily Range: {stats['avg_daily_range']:.2f}%")
                self.logger.info(f"   Max Daily Range: {stats['max_daily_range']:.2f}%")
                self.logger.info(f"   Min Daily Range: {stats['min_daily_range']:.2f}%")
        
        # Counter-strategy opportunities
        self.identify_counter_opportunities()
    
    def identify_counter_opportunities(self):
        """Identify potential counter-strategy opportunities"""
        
        self.logger.info(f"\n🎯 COUNTER-STRATEGY OPPORTUNITIES:")
        
        # High volatility days
        disaster_days = len(self.filter_categories['disaster_volatility'])
        if disaster_days > 0:
            self.logger.info(f"\n💥 HIGH VOLATILITY STRATEGY ({disaster_days} days):")
            self.logger.info(f"   🎯 Approach: Long volatility strategies")
            self.logger.info(f"   💡 Logic: Big moves = profit from volatility")
            self.logger.info(f"   📊 Strategy: Straddles, strangles, or volatility spreads")
            self.logger.info(f"   ⚠️  Risk: High premiums, need significant moves")
        
        # Calculate potential opportunity
        total_other = sum(len(days) for key, days in self.filter_categories.items() if key != 'disaster_volatility')
        if total_other > 0:
            self.logger.info(f"\n📊 OTHER OPPORTUNITIES ({total_other} days):")
            self.logger.info(f"   🎯 Need deeper analysis to identify patterns")

def main():
    """Main execution for filtered days analysis"""
    
    analyzer = FilteredDaysAnalyzer()
    
    # High volatility filtered dates from our 6-month backtest
    high_vol_filtered_dates = [
        "20240305",  # Disaster volatility (10.15%)
        "20240320",  # Disaster volatility (11.45%)
        "20240508",  # Disaster volatility (11.98%)
        "20240510",  # Disaster volatility (9.68%)
        "20240603"   # Disaster volatility (9.98%)
    ]
    
    print("🔍 Analyzing high volatility filtered days...")
    results = analyzer.analyze_filtered_days(high_vol_filtered_dates)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Total filtered days analyzed: {len(high_vol_filtered_dates)}")

if __name__ == "__main__":
    main()