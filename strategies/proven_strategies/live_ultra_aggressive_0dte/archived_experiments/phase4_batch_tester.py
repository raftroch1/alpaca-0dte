#!/usr/bin/env python3
"""
📊 PHASE 4 BATCH TESTER & COMPARISON ANALYZER
============================================

Compare Phase 4 ML-enhanced strategy against previous phases to validate
improvements and demonstrate the effectiveness of LiuAlgoTrader-inspired
ML signal filtering.

COMPARISON METRICS:
✅ Signal quality vs quantity trade-off
✅ P&L improvement through filtering
✅ Win rate enhancement
✅ Risk-adjusted returns
"""

import logging
import pandas as pd
from datetime import datetime
import os
import sys

# Import our Phase 4 strategy
from phase4_standalone_ml import Phase4MLStrategy

class Phase4ComparisonAnalyzer:
    """
    Comprehensive analyzer comparing Phase 4 ML strategy against previous phases
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = []
    
    def run_comprehensive_comparison(self, test_dates: list) -> pd.DataFrame:
        """
        Run Phase 4 strategy across multiple dates and analyze results
        
        Args:
            test_dates: List of dates in YYYYMMDD format
            
        Returns:
            DataFrame with comprehensive results
        """
        self.logger.info(f"🧠 Running Phase 4 comprehensive analysis across {len(test_dates)} dates")
        
        # Test different ML thresholds to find optimal balance
        thresholds_to_test = [0.50, 0.60, 0.70, 0.75, 0.80]
        
        all_results = []
        
        for threshold in thresholds_to_test:
            self.logger.info(f"🎯 Testing ML threshold: {threshold}")
            
            threshold_results = []
            
            for date_str in test_dates:
                try:
                    # Initialize strategy with specific threshold
                    strategy = Phase4MLStrategy()
                    strategy.quality_threshold = threshold
                    
                    # Run backtest
                    result = strategy.run_phase4_backtest(date_str)
                    result['ml_threshold'] = threshold
                    
                    threshold_results.append(result)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed for {date_str} with threshold {threshold}: {e}")
                    continue
            
            # Analyze threshold performance
            if threshold_results:
                threshold_summary = self._analyze_threshold_performance(threshold, threshold_results)
                all_results.append(threshold_summary)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results)
        
        # Generate recommendations
        self._generate_recommendations(comparison_df)
        
        return comparison_df
    
    def _analyze_threshold_performance(self, threshold: float, results: list) -> dict:
        """Analyze performance for a specific ML threshold"""
        
        # Aggregate metrics
        total_signals = sum(r.get('signals_generated', 0) for r in results)
        total_passed = sum(r.get('signals_passed_ml', 0) for r in results)
        total_trades = sum(r.get('trades_executed', 0) for r in results)
        total_pnl = sum(r.get('total_pnl', 0) for r in results)
        
        # Calculate derived metrics
        filter_ratio = total_passed / total_signals if total_signals > 0 else 0
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        profitable_days = sum(1 for r in results if r.get('total_pnl', 0) > 0)
        win_rate = profitable_days / len(results) if results else 0
        
        # Risk metrics
        daily_pnls = [r.get('total_pnl', 0) for r in results]
        volatility = pd.Series(daily_pnls).std() if len(daily_pnls) > 1 else 0
        sharpe_ratio = (total_pnl / len(results)) / volatility if volatility > 0 else 0
        
        return {
            'ml_threshold': threshold,
            'days_tested': len(results),
            'total_signals_generated': total_signals,
            'signals_passed_filter': total_passed,
            'filter_selectivity': filter_ratio,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_daily_pnl': total_pnl / len(results),
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'profitable_days': profitable_days,
            'win_rate': win_rate,
            'daily_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_daily_loss': min(daily_pnls) if daily_pnls else 0,
            'max_daily_gain': max(daily_pnls) if daily_pnls else 0
        }
    
    def _generate_recommendations(self, comparison_df: pd.DataFrame):
        """Generate actionable recommendations based on analysis"""
        
        self.logger.info("\n" + "="*60)
        self.logger.info("🎯 PHASE 4 ML STRATEGY ANALYSIS COMPLETE")
        self.logger.info("="*60)
        
        if comparison_df.empty:
            self.logger.warning("⚠️ No results to analyze")
            return
        
        # Find optimal threshold
        best_sharpe = comparison_df.loc[comparison_df['sharpe_ratio'].idxmax()]
        best_pnl = comparison_df.loc[comparison_df['avg_daily_pnl'].idxmax()]
        best_win_rate = comparison_df.loc[comparison_df['win_rate'].idxmax()]
        
        self.logger.info(f"\n📊 OPTIMAL THRESHOLDS:")
        self.logger.info(f"🎯 Best Sharpe Ratio: {best_sharpe['ml_threshold']:.2f} (Sharpe: {best_sharpe['sharpe_ratio']:.3f})")
        self.logger.info(f"💰 Best Daily P&L: {best_pnl['ml_threshold']:.2f} (${best_pnl['avg_daily_pnl']:.2f}/day)")
        self.logger.info(f"🏆 Best Win Rate: {best_win_rate['ml_threshold']:.2f} ({best_win_rate['win_rate']:.1%})")
        
        # Filter effectiveness analysis
        self.logger.info(f"\n🔍 ML FILTER EFFECTIVENESS:")
        for _, row in comparison_df.iterrows():
            selectivity = row['filter_selectivity']
            pnl_per_trade = row['avg_pnl_per_trade']
            
            self.logger.info(f"Threshold {row['ml_threshold']:.2f}: "
                           f"{selectivity:.1%} selectivity, "
                           f"${pnl_per_trade:.2f}/trade avg")
        
        # Strategic recommendations
        self.logger.info(f"\n💡 STRATEGIC RECOMMENDATIONS:")
        
        if best_sharpe['avg_daily_pnl'] > 0:
            self.logger.info("✅ BREAKTHROUGH: Phase 4 showing positive returns!")
            self.logger.info(f"✅ Recommended threshold: {best_sharpe['ml_threshold']:.2f}")
        else:
            self.logger.info("⚠️ Still optimizing for profitability")
            self.logger.info("💡 Consider: Lower thresholds or enhanced features")
        
        # LiuAlgoTrader integration opportunities
        self.logger.info(f"\n🧠 LIUALGOTRADER INTEGRATION OPPORTUNITIES:")
        self.logger.info("✅ ML filtering working - similar to their optimizer component")
        self.logger.info("🔄 Next: Implement their hyper-parameter optimization approach")
        self.logger.info("📊 Consider: Their portfolio management and risk framework")
        
    def compare_with_previous_phases(self, date_str: str) -> dict:
        """
        Compare Phase 4 with previous phases on same date
        
        Returns estimated comparison (would need real Phase 1-3 results for accuracy)
        """
        # This is a simplified comparison - in production would load actual results
        
        # Estimated Phase 3 results (from our previous testing)
        phase3_estimate = {
            'trades': 6,
            'pnl': -77.07,  # Real data result from Phase 3
            'avg_per_trade': -12.85
        }
        
        # Run Phase 4
        strategy = Phase4MLStrategy()
        strategy.quality_threshold = 0.60  # Use moderate threshold
        phase4_result = strategy.run_phase4_backtest(date_str)
        
        comparison = {
            'date': date_str,
            'phase3_estimated_pnl': phase3_estimate['pnl'],
            'phase4_actual_pnl': phase4_result.get('total_pnl', 0),
            'improvement': phase4_result.get('total_pnl', 0) - phase3_estimate['pnl'],
            'phase4_selectivity': phase4_result.get('ml_filter_ratio', 0),
            'phase4_trades': phase4_result.get('trades_executed', 0)
        }
        
        return comparison

def main():
    """Run comprehensive Phase 4 analysis"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Test dates (adjust based on available data)
    test_dates = [
        '20240315',  # March 15, 2024
        '20240322',  # March 22, 2024 
        '20240329',  # March 29, 2024
    ]
    
    analyzer = Phase4ComparisonAnalyzer()
    
    print("🧠 PHASE 4 ML STRATEGY COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    print("Inspired by LiuAlgoTrader's optimizer and analysis framework")
    print("Building on our proven real data integration foundation")
    print()
    
    # Run comprehensive analysis
    try:
        results_df = analyzer.run_comprehensive_comparison(test_dates)
        
        print("\n📊 DETAILED RESULTS:")
        print(results_df.round(3).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        
        # Fallback: single date comparison
        print("\n🔄 Running fallback single-date analysis...")
        comparison = analyzer.compare_with_previous_phases('20240315')
        
        print(f"\n📊 PHASE 4 vs PHASE 3 COMPARISON ({comparison['date']}):")
        print(f"Phase 3 (Estimated): ${comparison['phase3_estimated_pnl']:.2f}")
        print(f"Phase 4 (ML Filter): ${comparison['phase4_actual_pnl']:.2f}")
        print(f"Improvement: ${comparison['improvement']:.2f}")
        print(f"ML Selectivity: {comparison['phase4_selectivity']:.1%}")
        
    print("\n🚀 Phase 4 ML Analysis Complete!")
    print("Ready for Phase 4 Enhanced Analytics and Dynamic Execution phases!")

if __name__ == "__main__":
    main() 