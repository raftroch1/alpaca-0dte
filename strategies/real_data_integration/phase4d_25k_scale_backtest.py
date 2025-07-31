"""
🚀 PHASE 4D 25K ACCOUNT SCALING BACKTEST
=======================================

TARGET: Validate $300/day performance with 25 contracts on $25k account
COMPARISON: Against current 2-contract performance (~$23.7/day)

RISK ANALYSIS:
- Position sizing appropriateness for $25k account
- Daily P&L volatility vs account size  
- Maximum drawdown tolerance
- Realistic execution with larger orders
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import List, Dict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from phase4d_balanced_25k_scale import Phase4DBalanced25k

def setup_logging():
    """Setup logging for the backtest"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def generate_test_dates(start_date: str, end_date: str) -> List[str]:
    """Generate trading dates between start and end"""
    dates = []
    current = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    while current <= end:
        # Only include weekdays (rough trading day filter)
        if current.weekday() < 5:
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    return dates

def run_comprehensive_backtest():
    """
    🚀 RUN 25K SCALING BACKTEST
    
    TEST PERIOD: 6 months (same as original validation)
    GOAL: Validate $300/day target achievability
    """
    logger = setup_logging()
    logger.info("🚀 Starting 25K Account Scaling Backtest")
    
    # Initialize strategy
    strategy = Phase4DBalanced25k()
    
    # Test shorter period first to validate scaling concept
    start_date = "20240301" 
    end_date = "20240331"   # 1 month for initial validation
    test_dates = generate_test_dates(start_date, end_date)
    
    logger.info(f"📅 Testing {len(test_dates)} potential trading days")
    logger.info(f"🎯 TARGET: $300/day average (vs current $23.7/day)")
    
    # Track results
    daily_results = []
    total_pnl = 0.0
    trades_executed = 0
    winning_trades = 0
    total_days = 0
    
    # Risk tracking for 25k account
    daily_pnls = []
    max_daily_loss = 0
    max_daily_gain = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    current_consecutive = 0
    
    # Process each date
    for i, date in enumerate(test_dates):
        try:
            logger.info(f"\n📊 Processing {date} ({i+1}/{len(test_dates)})")
            
            result = strategy.run_strategy(date)
            daily_results.append(result)
            
            pnl = result.get('pnl', 0.0)
            total_pnl += pnl
            total_days += 1
            
            if result.get('trade_executed', False):
                trades_executed += 1
                if pnl > 0:
                    winning_trades += 1
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            # Track daily P&L for risk analysis
            daily_pnls.append(pnl)
            max_daily_loss = min(max_daily_loss, pnl)
            max_daily_gain = max(max_daily_gain, pnl)
            
            # Progress update
            if (i + 1) % 20 == 0:
                avg_daily = total_pnl / total_days if total_days > 0 else 0
                execution_rate = trades_executed / total_days * 100 if total_days > 0 else 0
                logger.info(f"📈 Progress: {avg_daily:.1f}/day avg, {execution_rate:.1f}% execution")
                
        except Exception as e:
            logger.error(f"❌ Error processing {date}: {e}")
            daily_results.append({
                'date': date,
                'trade_executed': False,
                'pnl': 0.0,
                'error': str(e)
            })
    
    # 📊 COMPREHENSIVE ANALYSIS
    logger.info("\n" + "="*60)
    logger.info("🚀 25K ACCOUNT SCALING BACKTEST RESULTS")
    logger.info("="*60)
    
    # Basic Performance
    avg_daily_pnl = total_pnl / total_days if total_days > 0 else 0
    execution_rate = trades_executed / total_days * 100 if total_days > 0 else 0
    win_rate = winning_trades / trades_executed * 100 if trades_executed > 0 else 0
    
    logger.info(f"📅 Period: {start_date} to {end_date} ({total_days} days)")
    logger.info(f"💰 Total P&L: ${total_pnl:,.2f}")
    logger.info(f"📊 Average Daily P&L: ${avg_daily_pnl:.2f}")
    logger.info(f"🎯 Target Achievement: {avg_daily_pnl/300*100:.1f}% of $300/day goal")
    logger.info(f"⚡ Execution Rate: {execution_rate:.1f}% ({trades_executed}/{total_days})")
    logger.info(f"🏆 Win Rate: {win_rate:.1f}% ({winning_trades}/{trades_executed})")
    
    # Risk Analysis for 25K Account
    logger.info(f"\n🛡️ RISK ANALYSIS (25K Account Context):")
    logger.info(f"📈 Max Daily Gain: ${max_daily_gain:,.2f} ({max_daily_gain/25000*100:.2f}% of account)")
    logger.info(f"📉 Max Daily Loss: ${max_daily_loss:,.2f} ({abs(max_daily_loss)/25000*100:.2f}% of account)")
    logger.info(f"🔄 Max Consecutive Losses: {max_consecutive_losses}")
    
    # Daily P&L distribution
    daily_pnls_array = np.array(daily_pnls)
    pnl_std = np.std(daily_pnls_array)
    pnl_p95 = np.percentile(daily_pnls_array, 95)
    pnl_p5 = np.percentile(daily_pnls_array, 5)
    
    logger.info(f"📊 Daily P&L Std Dev: ${pnl_std:.2f}")
    logger.info(f"📊 95th Percentile: ${pnl_p95:.2f}")
    logger.info(f"📊 5th Percentile: ${pnl_p5:.2f}")
    
    # Account utilization analysis
    logger.info(f"\n💼 ACCOUNT UTILIZATION ANALYSIS:")
    avg_monthly = avg_daily_pnl * 21  # ~21 trading days per month
    annual_return_est = avg_monthly * 12 / 25000 * 100
    logger.info(f"📅 Estimated Monthly P&L: ${avg_monthly:,.2f}")
    logger.info(f"📈 Estimated Annual Return: {annual_return_est:.1f}%")
    
    # Comparison to original strategy
    original_daily = 23.7  # From 2-contract strategy
    scaling_factor = avg_daily_pnl / original_daily if original_daily > 0 else 0
    logger.info(f"\n🔄 SCALING ANALYSIS:")
    logger.info(f"📊 Original Strategy: ${original_daily:.2f}/day (2 contracts)")
    logger.info(f"📊 25K Strategy: ${avg_daily_pnl:.2f}/day (25 contracts)")
    logger.info(f"📊 Actual Scaling Factor: {scaling_factor:.1f}x (Target: 12.6x)")
    
    # Risk warnings for 25K account
    logger.info(f"\n⚠️ RISK ASSESSMENT:")
    max_loss_pct = abs(max_daily_loss) / 25000 * 100
    if max_loss_pct > 5:
        logger.warning(f"🚨 HIGH RISK: Max daily loss {max_loss_pct:.1f}% exceeds 5% of account")
    elif max_loss_pct > 3:
        logger.warning(f"⚠️ MODERATE RISK: Max daily loss {max_loss_pct:.1f}% exceeds 3% of account")
    else:
        logger.info(f"✅ ACCEPTABLE RISK: Max daily loss {max_loss_pct:.1f}% within tolerance")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save daily results
    df_daily = pd.DataFrame(daily_results)
    daily_file = f"phase4d_25k_scale_daily_{timestamp}.csv"
    df_daily.to_csv(daily_file, index=False)
    logger.info(f"💾 Daily results saved: {daily_file}")
    
    # Save summary
    summary = {
        'backtest_period': f"{start_date}_to_{end_date}",
        'total_days': total_days,
        'trades_executed': trades_executed,
        'total_pnl': total_pnl,
        'avg_daily_pnl': avg_daily_pnl,
        'target_achievement_pct': avg_daily_pnl/300*100,
        'execution_rate': execution_rate,
        'win_rate': win_rate,
        'max_daily_gain': max_daily_gain,
        'max_daily_loss': max_daily_loss,
        'max_consecutive_losses': max_consecutive_losses,
        'daily_pnl_std': pnl_std,
        'scaling_factor': scaling_factor,
        'max_loss_pct_of_account': max_loss_pct,
        'estimated_annual_return': annual_return_est
    }
    
    summary_file = f"phase4d_25k_scale_summary_{timestamp}.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    logger.info(f"💾 Summary saved: {summary_file}")
    
    # Final recommendation
    logger.info(f"\n🎯 FINAL ASSESSMENT:")
    if avg_daily_pnl >= 250:  # Within 83% of target
        logger.info(f"✅ STRONG: Achieving ${avg_daily_pnl:.0f}/day - Very close to $300 target!")
    elif avg_daily_pnl >= 200:  # Within 67% of target
        logger.info(f"✅ GOOD: Achieving ${avg_daily_pnl:.0f}/day - Promising results")
    elif avg_daily_pnl >= 150:  # Within 50% of target
        logger.info(f"⚠️ MODERATE: Achieving ${avg_daily_pnl:.0f}/day - Needs optimization")
    else:
        logger.info(f"❌ INSUFFICIENT: Only ${avg_daily_pnl:.0f}/day - Major scaling issues")
    
    if max_loss_pct <= 3:
        logger.info(f"✅ RISK ACCEPTABLE: Max loss {max_loss_pct:.1f}% manageable for 25K account")
    else:
        logger.info(f"⚠️ RISK CONCERN: Max loss {max_loss_pct:.1f}% high for account size")
    
    return summary

if __name__ == "__main__":
    run_comprehensive_backtest()