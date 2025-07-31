#!/usr/bin/env python3
"""
🗓️ PHASE 4C: MONTHLY AGGRESSIVE TESTING - $300-500 DAILY TARGETS
================================================================

Full month testing of Phase 4C Aggressive Daily Targets across ~20 trading days
to validate the $300-500/day profit strategy on a $25K account.

AGGRESSIVE DAILY TARGET VALIDATION:
🎯 Target: $300-500/day (1.2-2% daily returns)
🎯 Account: $25,000 trading capital
🎯 Frequency: 8-15 trades/day (vs historical 0.1/day)
🎯 Position Size: $2,000-5,000 per trade
🎯 Win Rate Target: 60%+ for profitability

ULTRA REALISTIC TESTING PRESERVED:
✅ Real Alpaca historical option data for each trading day
✅ Realistic bid/ask spreads and slippage calculations
✅ Real time decay modeling with actual market hours
✅ Realistic option pricing with Greeks approximations
✅ Market microstructure considerations
✅ Commission and fee calculations

MONTHLY ANALYSIS FOCUS:
✅ Daily P&L distribution and target achievement rate
✅ Trade frequency and position sizing effectiveness
✅ Win rate analysis and loss pattern identification
✅ Market condition impact on strategy performance
✅ Optimization opportunities for profitability
✅ Risk-adjusted performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

# Import our aggressive daily targets strategy
from phase4c_aggressive_daily_targets import Phase4CAggressiveDailyTargets, DailyProfitSession

@dataclass
class MonthlyPerformanceReport:
    """Comprehensive monthly performance analysis for $300-500 daily targets"""
    # Core monthly metrics
    total_trading_days: int
    total_pnl: float
    avg_daily_pnl: float
    target_achievement_rate: float  # % of days hitting $300+ target
    
    # Trading activity
    total_trades: int
    avg_trades_per_day: float
    total_contracts_traded: int
    avg_position_size: float
    
    # Performance metrics
    win_rate: float
    avg_winning_trade: float
    avg_losing_trade: float
    best_day: float
    worst_day: float
    max_drawdown: float
    
    # Risk metrics
    daily_volatility: float
    sharpe_ratio: float
    daily_returns: List[float]
    
    # Pattern analysis
    day_of_week_performance: Dict[str, float]
    hourly_pnl_distribution: Dict[int, float]
    weekly_performance: List[Dict]
    
    # Optimization insights
    loss_causes: Dict[str, int]
    signal_quality_distribution: Dict[str, int]
    optimization_recommendations: List[str]

class Phase4CMonthlyRunner:
    """
    Orchestrates monthly testing for Phase 4C Aggressive Daily Targets,
    focusing on $300-500 daily profit validation.
    """
    
    def __init__(self, account_size: float = 25000):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.account_size = account_size
        self.daily_sessions: List[DailyProfitSession] = []
        self.aggressive_strategy = Phase4CAggressiveDailyTargets(account_size=account_size)
        
        self.logger.info("🗓️ Phase 4C Monthly Runner initialized")
        self.logger.info(f"💰 Account Size: ${account_size:,.0f}")
        self.logger.info("🎯 Testing $300-500 daily targets over full month")
        self.logger.info("🔒 PRESERVED: Ultra realistic testing core")
    
    def generate_trading_days(self, year: int, month: int) -> List[str]:
        """Generate list of trading days for the specified month"""
        # Start with first day of month
        start_date = datetime(year, month, 1)
        
        # Find last day of month
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends (Monday=0, Sunday=6)
            if current_date.weekday() < 5:  # Monday-Friday
                trading_days.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)
        
        self.logger.info(f"📅 Generated {len(trading_days)} trading days for {year}-{month:02d}")
        return trading_days
    
    def run_monthly_test(self, year: int = 2024, month: int = 3) -> MonthlyPerformanceReport:
        """
        Run comprehensive monthly test of aggressive daily targets
        
        Args:
            year: Year to test
            month: Month to test
            
        Returns:
            Complete monthly performance report
        """
        self.logger.info(f"🚀 Starting monthly test for {year}-{month:02d}")
        self.logger.info("🎯 Target: $300-500/day validation")
        
        trading_days = self.generate_trading_days(year, month)
        self.daily_sessions = []
        
        successful_days = 0
        failed_days = 0
        
        # Run strategy for each trading day
        for i, date_str in enumerate(trading_days, 1):
            self.logger.info(f"📊 Day {i}/{len(trading_days)}: {date_str}")
            
            try:
                session = self.aggressive_strategy.run_aggressive_daily_session(date_str)
                
                if session.trade_log and not any('error' in trade for trade in session.trade_log):
                    self.daily_sessions.append(session)
                    successful_days += 1
                    
                    # Log daily summary
                    target_status = "✅ TARGET" if session.current_pnl >= 300 else "❌ MISS"
                    self.logger.info(f"  {target_status}: ${session.current_pnl:.2f} ({session.trades_executed} trades)")
                else:
                    failed_days += 1
                    self.logger.warning(f"  ⚠️ FAILED: Data issues for {date_str}")
                    
            except Exception as e:
                failed_days += 1
                self.logger.error(f"  💥 ERROR: {date_str} - {e}")
        
        self.logger.info(f"📊 Monthly Test Complete: {successful_days} successful, {failed_days} failed")
        
        if not self.daily_sessions:
            raise ValueError("No successful trading days - cannot generate report")
        
        # Generate comprehensive report
        report = self._generate_monthly_report()
        
        return report
    
    def _generate_monthly_report(self) -> MonthlyPerformanceReport:
        """Generate comprehensive monthly performance report"""
        
        # Basic metrics
        total_days = len(self.daily_sessions)
        daily_pnls = [session.current_pnl for session in self.daily_sessions]
        total_pnl = sum(daily_pnls)
        avg_daily_pnl = total_pnl / total_days
        
        # Target achievement
        days_hitting_target = sum(1 for pnl in daily_pnls if pnl >= 300)
        target_achievement_rate = days_hitting_target / total_days
        
        # Trading activity
        total_trades = sum(session.trades_executed for session in self.daily_sessions)
        avg_trades_per_day = total_trades / total_days
        
        # Calculate total contracts and average position size
        total_contracts = 0
        total_contract_value = 0
        all_trade_pnls = []
        
        for session in self.daily_sessions:
            for trade in session.trade_log:
                if isinstance(trade, dict) and 'contracts' in trade:
                    contracts = trade.get('contracts', 0)
                    total_contracts += contracts
                    total_contract_value += contracts * 500  # Approximate $500 per contract
                    all_trade_pnls.append(trade.get('pnl', 0))
        
        avg_position_size = total_contract_value / max(total_trades, 1)
        
        # Performance metrics
        winning_trades = [pnl for pnl in all_trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in all_trade_pnls if pnl <= 0]
        
        win_rate = len(winning_trades) / max(len(all_trade_pnls), 1)
        avg_winning_trade = np.mean(winning_trades) if winning_trades else 0
        avg_losing_trade = np.mean(losing_trades) if losing_trades else 0
        
        best_day = max(daily_pnls)
        worst_day = min(daily_pnls)
        
        # Calculate drawdown
        cumulative_pnl = np.cumsum(daily_pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = cumulative_pnl - running_max
        max_drawdown = min(drawdowns)
        
        # Risk metrics
        daily_volatility = np.std(daily_pnls)
        daily_returns = [(pnl / self.account_size) * 100 for pnl in daily_pnls]
        avg_return = np.mean(daily_returns)
        return_std = np.std(daily_returns)
        sharpe_ratio = avg_return / max(return_std, 0.001) if return_std > 0 else 0
        
        # Day of week analysis
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_of_week_performance = {}
        
        for session in self.daily_sessions:
            date = datetime.strptime(session.date, '%Y%m%d')
            day_name = day_names[date.weekday()]
            if day_name not in day_of_week_performance:
                day_of_week_performance[day_name] = []
            day_of_week_performance[day_name].append(session.current_pnl)
        
        # Average by day of week
        day_of_week_avg = {day: np.mean(pnls) for day, pnls in day_of_week_performance.items()}
        
        # Hourly analysis
        hourly_pnl = {}
        for session in self.daily_sessions:
            for hour, pnl in session.hourly_pnl.items():
                if hour not in hourly_pnl:
                    hourly_pnl[hour] = []
                hourly_pnl[hour].append(pnl)
        
        hourly_pnl_avg = {hour: np.mean(pnls) for hour, pnls in hourly_pnl.items()}
        
        # Weekly performance
        weekly_performance = []
        sessions_by_week = {}
        
        for session in self.daily_sessions:
            date = datetime.strptime(session.date, '%Y%m%d')
            week_start = date - timedelta(days=date.weekday())
            week_key = week_start.strftime('%Y-%m-%d')
            
            if week_key not in sessions_by_week:
                sessions_by_week[week_key] = []
            sessions_by_week[week_key].append(session)
        
        for week_start, sessions in sessions_by_week.items():
            week_pnl = sum(s.current_pnl for s in sessions)
            week_trades = sum(s.trades_executed for s in sessions)
            weekly_performance.append({
                'week_start': week_start,
                'pnl': week_pnl,
                'trades': week_trades,
                'days': len(sessions)
            })
        
        # Loss cause analysis
        loss_causes = {'time_limit': 0, 'stop_loss': 0, 'profit_target': 0}
        signal_quality_dist = {'high': 0, 'medium': 0, 'low': 0}
        
        for session in self.daily_sessions:
            for trade in session.trade_log:
                if isinstance(trade, dict):
                    reason = trade.get('reason', 'unknown')
                    if reason in loss_causes:
                        loss_causes[reason] += 1
                    
                    quality = trade.get('quality_score', 0)
                    if quality >= 0.8:
                        signal_quality_dist['high'] += 1
                    elif quality >= 0.6:
                        signal_quality_dist['medium'] += 1
                    else:
                        signal_quality_dist['low'] += 1
        
        # Optimization recommendations
        recommendations = []
        
        if win_rate < 0.4:
            recommendations.append("🔄 CRITICAL: Win rate too low - consider inverse signal logic")
        
        if avg_daily_pnl < 0:
            recommendations.append("📈 URGENT: Negative daily returns - signal direction needs fixing")
        
        if target_achievement_rate < 0.2:
            recommendations.append("🎯 FOCUS: Low target achievement - increase position sizes or improve timing")
        
        if avg_trades_per_day < 8:
            recommendations.append("⚡ ACTIVITY: Increase signal frequency for more opportunities")
        
        if daily_volatility > 100:
            recommendations.append("🛡️ RISK: High volatility - implement better risk management")
        
        return MonthlyPerformanceReport(
            total_trading_days=total_days,
            total_pnl=total_pnl,
            avg_daily_pnl=avg_daily_pnl,
            target_achievement_rate=target_achievement_rate,
            total_trades=total_trades,
            avg_trades_per_day=avg_trades_per_day,
            total_contracts_traded=total_contracts,
            avg_position_size=avg_position_size,
            win_rate=win_rate,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            best_day=best_day,
            worst_day=worst_day,
            max_drawdown=max_drawdown,
            daily_volatility=daily_volatility,
            sharpe_ratio=sharpe_ratio,
            daily_returns=daily_returns,
            day_of_week_performance=day_of_week_avg,
            hourly_pnl_distribution=hourly_pnl_avg,
            weekly_performance=weekly_performance,
            loss_causes=loss_causes,
            signal_quality_distribution=signal_quality_dist,
            optimization_recommendations=recommendations
        )
    
    def display_monthly_report(self, report: MonthlyPerformanceReport):
        """Display comprehensive monthly performance report"""
        
        print(f"\n🗓️ PHASE 4C MONTHLY REPORT: AGGRESSIVE DAILY TARGETS")
        print("=" * 80)
        print("🎯 TARGET: $300-500/day on $25K account")
        print("🔒 ULTRA REALISTIC TESTING CORE PRESERVED")
        print()
        
        # Core Performance
        print("📊 MONTHLY PERFORMANCE SUMMARY:")
        print(f"  Trading Days: {report.total_trading_days}")
        print(f"  Total P&L: ${report.total_pnl:,.2f}")
        print(f"  Average Daily P&L: ${report.avg_daily_pnl:.2f}")
        print(f"  Target Achievement Rate: {report.target_achievement_rate:.1%}")
        monthly_return = (report.total_pnl / self.account_size) * 100
        print(f"  Monthly Return: {monthly_return:.2f}%")
        print()
        
        # Trading Activity
        print("⚡ TRADING ACTIVITY:")
        print(f"  Total Trades: {report.total_trades}")
        print(f"  Avg Trades/Day: {report.avg_trades_per_day:.1f}")
        print(f"  Total Contracts: {report.total_contracts_traded}")
        print(f"  Avg Position Size: ${report.avg_position_size:,.0f}")
        print()
        
        # Performance Metrics
        print("📈 PERFORMANCE ANALYSIS:")
        print(f"  Win Rate: {report.win_rate:.1%}")
        print(f"  Avg Winning Trade: ${report.avg_winning_trade:.2f}")
        print(f"  Avg Losing Trade: ${report.avg_losing_trade:.2f}")
        print(f"  Best Day: ${report.best_day:.2f}")
        print(f"  Worst Day: ${report.worst_day:.2f}")
        print(f"  Max Drawdown: ${report.max_drawdown:.2f}")
        print(f"  Daily Volatility: ${report.daily_volatility:.2f}")
        print(f"  Sharpe Ratio: {report.sharpe_ratio:.3f}")
        print()
        
        # Day of Week Analysis
        print("📅 DAY OF WEEK PERFORMANCE:")
        for day, avg_pnl in report.day_of_week_performance.items():
            print(f"  {day}: ${avg_pnl:.2f}")
        print()
        
        # Hourly Analysis
        print("⏰ HOURLY P&L DISTRIBUTION:")
        for hour in sorted(report.hourly_pnl_distribution.keys()):
            avg_pnl = report.hourly_pnl_distribution[hour]
            print(f"  {hour}:00 - ${avg_pnl:.2f}")
        print()
        
        # Weekly Breakdown
        print("📊 WEEKLY PERFORMANCE:")
        for i, week in enumerate(report.weekly_performance, 1):
            print(f"  Week {i}: ${week['pnl']:.2f} ({week['trades']} trades, {week['days']} days)")
        print()
        
        # Loss Analysis
        print("🔍 TRADE EXIT ANALYSIS:")
        total_exits = sum(report.loss_causes.values())
        for cause, count in report.loss_causes.items():
            pct = (count / max(total_exits, 1)) * 100
            print(f"  {cause.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        print()
        
        # Signal Quality
        print("🧠 SIGNAL QUALITY DISTRIBUTION:")
        total_signals = sum(report.signal_quality_distribution.values())
        for quality, count in report.signal_quality_distribution.items():
            pct = (count / max(total_signals, 1)) * 100
            print(f"  {quality.title()} Quality: {count} ({pct:.1f}%)")
        print()
        
        # Target Analysis
        print("🎯 DAILY TARGET ANALYSIS:")
        days_profitable = sum(1 for pnl in report.daily_returns if pnl > 0)
        profit_rate = days_profitable / report.total_trading_days
        days_300_plus = sum(1 for session in self.daily_sessions if session.current_pnl >= 300)
        days_500_plus = sum(1 for session in self.daily_sessions if session.current_pnl >= 500)
        
        print(f"  Profitable Days: {days_profitable}/{report.total_trading_days} ({profit_rate:.1%})")
        print(f"  Days ≥ $300: {days_300_plus}/{report.total_trading_days} ({days_300_plus/report.total_trading_days:.1%})")
        print(f"  Days ≥ $500: {days_500_plus}/{report.total_trading_days} ({days_500_plus/report.total_trading_days:.1%})")
        print()
        
        # Optimization Recommendations
        print("💡 OPTIMIZATION RECOMMENDATIONS:")
        for i, rec in enumerate(report.optimization_recommendations, 1):
            print(f"  {i}. {rec}")
        print()
        
        # Final Assessment
        print("🏆 FINAL ASSESSMENT:")
        if report.avg_daily_pnl >= 300:
            print("  ✅ EXCELLENT: Daily targets consistently achieved!")
        elif report.avg_daily_pnl > 0:
            print("  📈 POSITIVE: Profitable but needs optimization for targets")
        else:
            print("  ⚠️ NEEDS WORK: Strategy requires significant optimization")
        
        if report.target_achievement_rate >= 0.6:
            print("  🎯 TARGET SUCCESS: High target achievement rate")
        elif report.target_achievement_rate >= 0.3:
            print("  🎯 PARTIAL SUCCESS: Moderate target achievement")
        else:
            print("  🎯 TARGET MISS: Low target achievement - major optimization needed")

def run_march_2024_test():
    """Run comprehensive test for March 2024"""
    runner = Phase4CMonthlyRunner(account_size=25000)
    
    print("🚀 STARTING MARCH 2024 MONTHLY TEST")
    print("🎯 Validating $300-500 daily targets on $25K account")
    print("📊 Testing aggressive trading strategy across full month")
    print()
    
    report = runner.run_monthly_test(year=2024, month=3)
    runner.display_monthly_report(report)
    
    return report

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Phase 4C Monthly Testing')
    parser.add_argument('--year', type=int, default=2024, help='Year to test')
    parser.add_argument('--month', type=int, default=3, help='Month to test')
    parser.add_argument('--account', type=float, default=25000, help='Account size')
    
    args = parser.parse_args()
    
    runner = Phase4CMonthlyRunner(account_size=args.account)
    report = runner.run_monthly_test(year=args.year, month=args.month)
    runner.display_monthly_report(report) 