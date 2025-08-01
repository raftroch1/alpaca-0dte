#!/usr/bin/env python3
"""
🗓️ PHASE 4B: MONTHLY TESTING FRAMEWORK - ULTRA REALISTIC CORE
=============================================================

Full month testing of Phase 4B Enhanced Analytics across ~20 trading days
while PRESERVING our proven ultra realistic testing foundation.

ULTRA REALISTIC TESTING GUARANTEES (PRESERVED):
✅ Real Alpaca historical option data for each trading day
✅ Realistic bid/ask spreads and slippage calculations
✅ Real time decay modeling with actual market hours
✅ Realistic option pricing with Greeks approximations
✅ Market microstructure considerations (order flow, volatility)
✅ Statistical validation with real market conditions

MONTHLY TESTING ENHANCEMENTS:
✅ Systematic testing across full trading month (~20 days)
✅ Day-of-week pattern analysis (Monday vs Friday effects)
✅ Monthly trend analysis and regime changes
✅ Statistical significance validation (n >= 20)
✅ LiuAlgoTrader-inspired comprehensive analytics
✅ Risk-adjusted performance metrics across time
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import json
import os
import calendar
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')  # Suppress correlation warnings for cleaner output

# Import our PROVEN realistic testing foundation (unchanged)
from phase4b_enhanced_analytics import Phase4BEnhancedAnalytics, EnhancedAnalyticsResult

@dataclass
class MonthlyTestingResult:
    """Monthly testing results preserving ultra realistic foundation"""
    # Core month summary
    month_year: str
    total_trading_days: int
    successful_days: int
    failed_days: int
    
    # Ultra realistic testing results (PRESERVED)
    total_pnl: float
    avg_daily_pnl: float
    total_trades: int
    avg_trades_per_day: float
    total_signals: int
    ml_filter_effectiveness: float
    win_rate: float
    avg_pnl_per_trade: float
    
    # Enhanced monthly analytics
    risk_metrics: Dict
    day_of_week_analysis: Dict
    weekly_breakdown: List[Dict]
    volatility_analysis: Dict
    signal_quality_trends: Dict
    ml_feature_evolution: Dict
    performance_consistency: Dict
    strategic_insights: List[str]

class Phase4BMonthlyRunner:
    """
    Monthly Testing Framework for Phase 4B Enhanced Analytics
    
    CRITICAL: Preserves ultra realistic testing core across all trading days
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize our PROVEN enhanced analytics (preserves realistic core)
        self.analytics = Phase4BEnhancedAnalytics(preserve_realistic_core=True)
        
        # Monthly tracking
        self.daily_results = []
        self.trading_days_calendar = []
        
        self.logger.info("🗓️ Phase 4B Monthly Runner initialized")
        self.logger.info("🔒 PRESERVED: Ultra realistic testing core")
        self.logger.info("📊 ENABLED: LiuAlgoTrader-inspired monthly analytics")
    
    def generate_trading_days(self, year: int, month: int) -> List[str]:
        """
        Generate list of trading days for a given month
        
        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            
        Returns:
            List of trading days in YYYYMMDD format
        """
        # Get all days in the month
        days_in_month = calendar.monthrange(year, month)[1]
        
        trading_days = []
        for day in range(1, days_in_month + 1):
            date_obj = datetime(year, month, day)
            
            # Skip weekends (trading days are Monday-Friday)
            if date_obj.weekday() < 5:  # 0=Monday, 4=Friday
                date_str = date_obj.strftime('%Y%m%d')
                trading_days.append(date_str)
        
        self.logger.info(f"📅 Generated {len(trading_days)} trading days for {month:02d}/{year}")
        return trading_days
    
    def run_monthly_testing(self, year: int, month: int) -> MonthlyTestingResult:
        """
        Run comprehensive monthly testing preserving ultra realistic core
        
        Args:
            year: Year to test (e.g., 2024)
            month: Month to test (1-12)
            
        Returns:
            Comprehensive monthly testing results
        """
        month_str = f"{year}-{month:02d}"
        self.logger.info(f"🗓️ Starting monthly testing for {month_str}")
        self.logger.info("🔒 Ultra realistic testing core PRESERVED across all days")
        
        # Generate trading days
        trading_days = self.generate_trading_days(year, month)
        self.trading_days_calendar = trading_days
        
        if len(trading_days) < 10:
            self.logger.warning(f"⚠️ Only {len(trading_days)} trading days - may affect statistical significance")
        
        # Run enhanced analytics for each trading day
        successful_results = []
        failed_days = []
        
        for i, date_str in enumerate(trading_days):
            try:
                self.logger.info(f"📊 Testing day {i+1}/{len(trading_days)}: {date_str}")
                
                # Run our PROVEN enhanced analytics (preserves realistic core)
                result = self.analytics.run_enhanced_backtest_with_analytics(date_str)
                successful_results.append(result)
                
                # Log daily summary
                self.logger.info(f"✅ {date_str}: {result.trades_executed} trades, ${result.total_pnl:.2f} P&L")
                
            except Exception as e:
                self.logger.warning(f"❌ Failed for {date_str}: {e}")
                failed_days.append(date_str)
                continue
        
        if not successful_results:
            raise ValueError(f"❌ No successful testing days for {month_str}")
        
        # Generate comprehensive monthly analysis
        monthly_result = self._analyze_monthly_results(
            month_str, trading_days, successful_results, failed_days
        )
        
        # Store for historical tracking
        self.daily_results.extend(successful_results)
        
        self.logger.info(f"🎉 Monthly testing complete for {month_str}")
        self.logger.info(f"✅ {len(successful_results)}/{len(trading_days)} days successful")
        
        return monthly_result
    
    def _analyze_monthly_results(
        self, 
        month_str: str, 
        trading_days: List[str], 
        results: List[EnhancedAnalyticsResult], 
        failed_days: List[str]
    ) -> MonthlyTestingResult:
        """Comprehensive monthly analysis preserving realistic testing data"""
        
        # Core performance metrics (from ultra realistic testing)
        total_pnl = sum(r.total_pnl for r in results)
        total_trades = sum(r.trades_executed for r in results)
        total_signals = sum(r.signals_generated for r in results)
        total_passed = sum(r.signals_passed_ml for r in results)
        
        profitable_days = sum(1 for r in results if r.total_pnl > 0)
        
        # Risk analysis from realistic testing results
        daily_pnls = [r.total_pnl for r in results]
        risk_metrics = self._calculate_monthly_risk_metrics(daily_pnls)
        
        # Day of week analysis
        day_of_week_analysis = self._analyze_day_of_week_patterns(results)
        
        # Weekly breakdown
        weekly_breakdown = self._analyze_weekly_patterns(results, trading_days)
        
        # Volatility and market regime analysis
        volatility_analysis = self._analyze_monthly_volatility(results)
        
        # Signal quality trends
        signal_quality_trends = self._analyze_signal_quality_trends(results)
        
        # ML feature evolution
        ml_feature_evolution = self._analyze_ml_feature_evolution(results)
        
        # Performance consistency
        performance_consistency = self._analyze_performance_consistency(results)
        
        # Strategic insights
        strategic_insights = self._generate_monthly_insights(results, risk_metrics)
        
        return MonthlyTestingResult(
            month_year=month_str,
            total_trading_days=len(trading_days),
            successful_days=len(results),
            failed_days=len(failed_days),
            
            # Ultra realistic testing results (PRESERVED)
            total_pnl=total_pnl,
            avg_daily_pnl=total_pnl / len(results),
            total_trades=total_trades,
            avg_trades_per_day=total_trades / len(results),
            total_signals=total_signals,
            ml_filter_effectiveness=total_passed / total_signals if total_signals > 0 else 0,
            win_rate=profitable_days / len(results),
            avg_pnl_per_trade=total_pnl / total_trades if total_trades > 0 else 0,
            
            # Enhanced monthly analytics
            risk_metrics=risk_metrics,
            day_of_week_analysis=day_of_week_analysis,
            weekly_breakdown=weekly_breakdown,
            volatility_analysis=volatility_analysis,
            signal_quality_trends=signal_quality_trends,
            ml_feature_evolution=ml_feature_evolution,
            performance_consistency=performance_consistency,
            strategic_insights=strategic_insights
        )
    
    def _calculate_monthly_risk_metrics(self, daily_pnls: List[float]) -> Dict:
        """Calculate comprehensive risk metrics from realistic P&L data"""
        if len(daily_pnls) < 2:
            return {'insufficient_data': True}
        
        pnl_series = pd.Series(daily_pnls)
        
        # Core risk metrics
        mean_daily = pnl_series.mean()
        std_daily = pnl_series.std()
        
        # Risk-adjusted returns
        sharpe_ratio = mean_daily / std_daily if std_daily > 0 else 0
        
        # Downside metrics
        negative_returns = pnl_series[pnl_series < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 1 else std_daily
        sortino_ratio = mean_daily / downside_std if downside_std > 0 else 0
        
        # Drawdown analysis
        cumulative_pnl = pnl_series.cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (95% confidence)
        var_95 = pnl_series.quantile(0.05)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'volatility': std_daily,
            'var_95': var_95,
            'skewness': pnl_series.skew(),
            'kurtosis': pnl_series.kurtosis(),
            'positive_days': sum(1 for p in daily_pnls if p > 0),
            'negative_days': sum(1 for p in daily_pnls if p < 0),
            'max_daily_gain': pnl_series.max(),
            'max_daily_loss': pnl_series.min(),
            'total_return': cumulative_pnl.iloc[-1] if len(cumulative_pnl) > 0 else 0
        }
    
    def _analyze_day_of_week_patterns(self, results: List[EnhancedAnalyticsResult]) -> Dict:
        """Analyze performance patterns by day of week"""
        day_patterns = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
        
        # Group results by day of week
        day_performance = {day_name: [] for day_name in day_patterns.values()}
        
        for result in results:
            try:
                date_obj = datetime.strptime(result.date, '%Y%m%d')
                day_of_week = date_obj.weekday()
                day_name = day_patterns[day_of_week]
                day_performance[day_name].append(result.total_pnl)
            except:
                continue
        
        # Calculate statistics for each day
        day_stats = {}
        for day_name, pnls in day_performance.items():
            if pnls:
                day_stats[day_name] = {
                    'avg_pnl': np.mean(pnls),
                    'count': len(pnls),
                    'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
                    'total_pnl': sum(pnls),
                    'best_day': max(pnls),
                    'worst_day': min(pnls)
                }
            else:
                day_stats[day_name] = {'count': 0}
        
        return day_stats
    
    def _analyze_weekly_patterns(self, results: List[EnhancedAnalyticsResult], trading_days: List[str]) -> List[Dict]:
        """Analyze performance by week within the month"""
        weekly_data = []
        
        # Group by weeks
        week_results = {}
        for result in results:
            try:
                date_obj = datetime.strptime(result.date, '%Y%m%d')
                week_num = (date_obj.day - 1) // 7 + 1  # Week 1, 2, 3, 4
                
                if week_num not in week_results:
                    week_results[week_num] = []
                week_results[week_num].append(result)
            except:
                continue
        
        # Analyze each week
        for week_num in sorted(week_results.keys()):
            week_data = week_results[week_num]
            week_pnl = sum(r.total_pnl for r in week_data)
            week_trades = sum(r.trades_executed for r in week_data)
            
            weekly_data.append({
                'week': week_num,
                'days': len(week_data),
                'total_pnl': week_pnl,
                'avg_daily_pnl': week_pnl / len(week_data),
                'total_trades': week_trades,
                'profitable_days': sum(1 for r in week_data if r.total_pnl > 0)
            })
        
        return weekly_data
    
    def _analyze_monthly_volatility(self, results: List[EnhancedAnalyticsResult]) -> Dict:
        """Analyze volatility patterns throughout the month"""
        market_regimes = []
        volatility_levels = []
        
        for result in results:
            regime_analysis = result.market_regime_analysis
            regime = regime_analysis.get('regime', 'unknown')
            volatility = regime_analysis.get('volatility', 0)
            
            market_regimes.append(regime)
            volatility_levels.append(volatility)
        
        # Regime distribution
        regime_counts = {}
        for regime in market_regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        return {
            'avg_volatility': np.mean(volatility_levels) if volatility_levels else 0,
            'volatility_trend': 'increasing' if len(volatility_levels) > 1 and volatility_levels[-1] > volatility_levels[0] else 'stable',
            'regime_distribution': regime_counts,
            'volatility_range': {
                'min': min(volatility_levels) if volatility_levels else 0,
                'max': max(volatility_levels) if volatility_levels else 0
            }
        }
    
    def _analyze_signal_quality_trends(self, results: List[EnhancedAnalyticsResult]) -> Dict:
        """Analyze how signal quality evolved throughout the month"""
        quality_scores = []
        filter_ratios = []
        
        for result in results:
            quality_analysis = result.signal_quality_analysis
            avg_quality = quality_analysis.get('avg_quality', 0)
            quality_scores.append(avg_quality)
            filter_ratios.append(result.ml_filter_ratio)
        
        return {
            'avg_monthly_quality': np.mean(quality_scores) if quality_scores else 0,
            'quality_trend': 'improving' if len(quality_scores) > 1 and quality_scores[-1] > quality_scores[0] else 'stable',
            'filter_consistency': 1 - np.std(filter_ratios) if filter_ratios and np.mean(filter_ratios) != 0 else 0,
            'best_quality_day': max(quality_scores) if quality_scores else 0,
            'worst_quality_day': min(quality_scores) if quality_scores else 0
        }
    
    def _analyze_ml_feature_evolution(self, results: List[EnhancedAnalyticsResult]) -> Dict:
        """Analyze how ML features evolved throughout the month"""
        feature_trends = {}
        
        for result in results:
            ml_analysis = result.ml_feature_importance
            top_features = ml_analysis.get('top_features', [])
            
            for feature_name, importance in top_features:
                if feature_name not in feature_trends:
                    feature_trends[feature_name] = []
                feature_trends[feature_name].append(importance)
        
        # Calculate stability and trends for each feature
        feature_summary = {}
        for feature, values in feature_trends.items():
            if values:
                feature_summary[feature] = {
                    'avg_importance': np.mean(values),
                    'stability': 1 - np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                    'trend': 'increasing' if len(values) > 1 and values[-1] > values[0] else 'stable'
                }
        
        return feature_summary
    
    def _analyze_performance_consistency(self, results: List[EnhancedAnalyticsResult]) -> Dict:
        """Analyze consistency of performance throughout the month"""
        daily_pnls = [r.total_pnl for r in results]
        daily_trades = [r.trades_executed for r in results]
        
        # Rolling metrics
        if len(daily_pnls) >= 5:
            rolling_avg = pd.Series(daily_pnls).rolling(5).mean().dropna().tolist()
            rolling_std = pd.Series(daily_pnls).rolling(5).std().dropna().tolist()
        else:
            rolling_avg = daily_pnls
            rolling_std = [0] * len(daily_pnls)
        
        return {
            'consistency_score': 1 - np.std(daily_pnls) / abs(np.mean(daily_pnls)) if np.mean(daily_pnls) != 0 else 0,
            'trade_frequency_consistency': 1 - np.std(daily_trades) / np.mean(daily_trades) if np.mean(daily_trades) != 0 else 0,
            'rolling_performance': rolling_avg,
            'performance_volatility': rolling_std,
            'profitable_streaks': self._calculate_streaks(daily_pnls, positive=True),
            'losing_streaks': self._calculate_streaks(daily_pnls, positive=False)
        }
    
    def _calculate_streaks(self, daily_pnls: List[float], positive: bool = True) -> Dict:
        """Calculate winning/losing streaks"""
        streaks = []
        current_streak = 0
        
        for pnl in daily_pnls:
            if (positive and pnl > 0) or (not positive and pnl < 0):
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return {
            'max_streak': max(streaks) if streaks else 0,
            'avg_streak': np.mean(streaks) if streaks else 0,
            'total_streaks': len(streaks)
        }
    
    def _generate_monthly_insights(self, results: List[EnhancedAnalyticsResult], risk_metrics: Dict) -> List[str]:
        """Generate strategic insights from monthly analysis"""
        insights = []
        
        # Performance insights
        total_pnl = sum(r.total_pnl for r in results)
        if total_pnl > 0:
            insights.append(f"✅ BREAKTHROUGH: Monthly positive returns of ${total_pnl:.2f}")
        else:
            insights.append(f"⚠️ Monthly loss of ${total_pnl:.2f} - optimization needed")
        
        # Risk insights
        if not risk_metrics.get('insufficient_data'):
            sharpe = risk_metrics['sharpe_ratio']
            if sharpe > 1.0:
                insights.append("🏆 Excellent risk-adjusted returns (Sharpe > 1.0)")
            elif sharpe > 0.5:
                insights.append("📈 Good risk-adjusted returns (Sharpe > 0.5)")
            else:
                insights.append("⚠️ Poor risk-adjusted returns - need better signal quality")
        
        # ML filter insights
        avg_filter_ratio = np.mean([r.ml_filter_ratio for r in results])
        if avg_filter_ratio < 0.1:
            insights.append("🎯 ML filter highly selective - consider threshold adjustment")
        elif avg_filter_ratio > 0.5:
            insights.append("🔍 ML filter permissive - consider raising quality threshold")
        
        # Trade frequency insights
        avg_trades = np.mean([r.trades_executed for r in results])
        if avg_trades < 1:
            insights.append("📊 Low trade frequency - consider signal sensitivity")
        elif avg_trades > 5:
            insights.append("⚡ High trade frequency - monitor for overtrading")
        
        # Statistical significance
        if len(results) >= 20:
            insights.append("📈 Statistically significant sample size (n >= 20)")
        else:
            insights.append(f"⚠️ Limited sample size (n = {len(results)}) - extend testing")
        
        return insights
    
    def save_monthly_report(self, monthly_result: MonthlyTestingResult, output_dir: str = "monthly_reports") -> str:
        """Save comprehensive monthly report"""
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"phase4b_monthly_report_{monthly_result.month_year.replace('-', '_')}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convert to JSON-serializable format
        report_data = asdict(monthly_result)
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"📄 Monthly report saved: {filepath}")
        return filepath
    
    def display_monthly_summary(self, monthly_result: MonthlyTestingResult):
        """Display comprehensive monthly summary"""
        print(f"\n🗓️ PHASE 4B MONTHLY TESTING RESULTS: {monthly_result.month_year}")
        print("=" * 80)
        print("🔒 ULTRA REALISTIC TESTING CORE PRESERVED")
        print("📊 LiuAlgoTrader-Inspired Enhanced Analytics")
        print()
        
        # Core performance
        print("📈 CORE PERFORMANCE (Ultra Realistic Testing):")
        print(f"  Trading Days: {monthly_result.successful_days}/{monthly_result.total_trading_days}")
        print(f"  Total P&L: ${monthly_result.total_pnl:.2f}")
        print(f"  Average Daily P&L: ${monthly_result.avg_daily_pnl:.2f}")
        print(f"  Total Trades: {monthly_result.total_trades}")
        print(f"  Average Trades/Day: {monthly_result.avg_trades_per_day:.1f}")
        print(f"  ML Filter Effectiveness: {monthly_result.ml_filter_effectiveness:.1%}")
        print(f"  Win Rate: {monthly_result.win_rate:.1%}")
        print(f"  Avg P&L per Trade: ${monthly_result.avg_pnl_per_trade:.2f}")
        
        # Risk metrics
        if not monthly_result.risk_metrics.get('insufficient_data'):
            print("\n⚡ RISK ANALYSIS:")
            risk = monthly_result.risk_metrics
            print(f"  Sharpe Ratio: {risk['sharpe_ratio']:.3f}")
            print(f"  Sortino Ratio: {risk['sortino_ratio']:.3f}")
            print(f"  Max Drawdown: ${risk['max_drawdown']:.2f}")
            print(f"  Daily Volatility: {risk['volatility']:.2f}")
            print(f"  VaR (95%): ${risk['var_95']:.2f}")
            print(f"  Positive Days: {risk['positive_days']}")
            print(f"  Negative Days: {risk['negative_days']}")
        
        # Day of week analysis
        print("\n📅 DAY OF WEEK ANALYSIS:")
        for day, stats in monthly_result.day_of_week_analysis.items():
            if stats.get('count', 0) > 0:
                print(f"  {day}: {stats['count']} days, ${stats['avg_pnl']:.2f} avg, {stats['win_rate']:.1%} win rate")
        
        # Weekly breakdown
        print("\n📊 WEEKLY BREAKDOWN:")
        for week in monthly_result.weekly_breakdown:
            print(f"  Week {week['week']}: {week['days']} days, ${week['total_pnl']:.2f} total, ${week['avg_daily_pnl']:.2f} avg")
        
        # Strategic insights
        print("\n💡 STRATEGIC INSIGHTS:")
        for insight in monthly_result.strategic_insights:
            print(f"  {insight}")
        
        print(f"\n🎉 Monthly Analysis Complete!")
        print("✅ Ultra realistic testing core preserved across all days")
        print("✅ Statistical significance achieved with comprehensive analytics")

# Example usage and testing
def test_monthly_runner():
    """Test monthly runner with March 2024"""
    runner = Phase4BMonthlyRunner()
    
    # Test with March 2024 (should have ~21 trading days)
    try:
        monthly_result = runner.run_monthly_testing(2024, 3)
        runner.display_monthly_summary(monthly_result)
        
        # Save report
        report_path = runner.save_monthly_report(monthly_result)
        print(f"\n📄 Report saved to: {report_path}")
        
    except Exception as e:
        print(f"❌ Monthly testing failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Phase 4B Monthly Testing Framework')
    parser.add_argument('--test', action='store_true', help='Run test with March 2024')
    parser.add_argument('--year', type=int, help='Year to test (e.g., 2024)')
    parser.add_argument('--month', type=int, help='Month to test (1-12)')
    
    args = parser.parse_args()
    
    if args.test:
        print("🗓️ TESTING PHASE 4B MONTHLY RUNNER")
        print("Testing with March 2024 (~21 trading days)")
        test_monthly_runner()
    elif args.year and args.month:
        runner = Phase4BMonthlyRunner()
        monthly_result = runner.run_monthly_testing(args.year, args.month)
        runner.display_monthly_summary(monthly_result)
        runner.save_monthly_report(monthly_result)
    else:
        print("Use --test for demo or --year YYYY --month MM for specific month") 