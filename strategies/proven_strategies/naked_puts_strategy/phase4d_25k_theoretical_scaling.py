"""
🚀 PHASE 4D - 25K THEORETICAL SCALING ANALYSIS
=============================================

APPROACH: Take the EXACT trades from the proven balanced strategy 
and scale them proportionally to validate $300/day target feasibility.

REAL TRADING CONDITIONS:
- Same strike selection as proven strategy
- Same market filtering
- Scaled position sizes (2 contracts → 25 contracts)
- Enhanced realistic costs for larger positions
- Risk management adjusted for 25K account
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Add project root to path  
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

class Phase4D25kTheoreticalScaling:
    """
    🎯 THEORETICAL SCALING ANALYSIS
    
    Takes proven balanced strategy trades and scales positions for 25K account
    """
    
    def __init__(self):
        self.setup_logging()
        
        # 25K SCALING PARAMETERS
        self.scaling_params = {
            # Position Scaling
            'original_contracts': 2,
            'scaled_contracts': 25,
            'scaling_factor': 12.5,  # 25/2
            
            # Enhanced Trading Costs for Larger Positions
            'commission_per_contract': 0.65,     # $0.65/contract (established)
            'bid_ask_spread_pct': 0.035,         # 3.5% for larger orders (vs 3% small)
            'slippage_pct': 0.008,               # 0.8% for larger orders (vs 0.5% small)
            'market_impact_threshold': 10,       # Additional impact for 10+ contracts
            'market_impact_pct': 0.005,          # 0.5% additional for large orders
            
            # Risk Management for 25K Account
            'max_daily_loss': 1500,              # $1500 daily stop
            'max_loss_per_trade': 1000,          # $1000 per trade stop
            'account_size': 25000,               # $25K account
            'max_risk_per_trade_pct': 4.0,       # 4% max risk per trade
        }
        
        self.logger.info("📊 Initialized 25K Theoretical Scaling Analysis")
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def load_proven_results(self, results_file: str) -> pd.DataFrame:
        """Load the proven balanced strategy results"""
        try:
            df = pd.read_csv(results_file)
            self.logger.info(f"📂 Loaded {len(df)} days of proven results")
            return df
        except Exception as e:
            self.logger.error(f"❌ Error loading results: {e}")
            return pd.DataFrame()
    
    def calculate_scaled_costs(self, gross_premium: float, contracts: int) -> Dict:
        """Calculate realistic costs for scaled positions"""
        
        # Base costs
        commission = self.scaling_params['commission_per_contract'] * contracts
        bid_ask_cost = gross_premium * self.scaling_params['bid_ask_spread_pct']
        slippage_cost = gross_premium * self.scaling_params['slippage_pct']
        
        # Market impact for large orders
        market_impact = 0.0
        if contracts >= self.scaling_params['market_impact_threshold']:
            market_impact = gross_premium * self.scaling_params['market_impact_pct']
            self.logger.debug(f"⚡ Market impact: ${market_impact:.2f} for {contracts} contracts")
        
        total_costs = commission + bid_ask_cost + slippage_cost + market_impact
        
        return {
            'commission': commission,
            'bid_ask_cost': bid_ask_cost,
            'slippage_cost': slippage_cost,
            'market_impact': market_impact,
            'total_costs': total_costs
        }
    
    def scale_trade_result(self, original_trade: Dict) -> Dict:
        """Scale an individual trade result to 25K account"""
        
        try:
            # Parse original trade details
            if isinstance(original_trade['trade_details'], str):
                import ast
                trade_details = ast.literal_eval(original_trade['trade_details'])
            else:
                trade_details = original_trade['trade_details']
            
            # Extract original trade parameters
            original_contracts = trade_details['contracts']
            strike = trade_details['strike']
            premium = trade_details['premium']
            spy_close = trade_details['spy_close']
            outcome = trade_details['outcome']
            
            # Scale position size
            scaled_contracts = self.scaling_params['scaled_contracts']
            scaling_factor = scaled_contracts / original_contracts
            
            # Calculate scaled gross premium
            scaled_gross_premium = premium * 100 * scaled_contracts
            
            # Calculate enhanced costs for larger position
            cost_breakdown = self.calculate_scaled_costs(scaled_gross_premium, scaled_contracts)
            scaled_net_premium = scaled_gross_premium - cost_breakdown['total_costs']
            
            # Calculate scaled outcome
            if "ASSIGNED" in outcome:
                # Calculate assignment cost with scaling
                assignment_cost = (strike - spy_close) * 100 * scaled_contracts
                scaled_final_pnl = scaled_net_premium - assignment_cost
            else:
                # Keep all premium
                scaled_final_pnl = scaled_net_premium
            
            # Apply risk management
            max_loss = self.scaling_params['max_loss_per_trade']
            if scaled_final_pnl < -max_loss:
                scaled_final_pnl = -max_loss
                outcome += "_RISK_STOPPED"
            
            return {
                'date': original_trade['date'],
                'trade_executed': True,
                'strategy_used': 'scaled_primary',
                'pnl': scaled_final_pnl,
                'spy_close': spy_close,
                'scaled_trade_details': {
                    'original_contracts': original_contracts,
                    'scaled_contracts': scaled_contracts,
                    'scaling_factor': scaling_factor,
                    'strike': strike,
                    'premium': premium,
                    'scaled_gross_premium': scaled_gross_premium,
                    'cost_breakdown': cost_breakdown,
                    'scaled_net_premium': scaled_net_premium,
                    'spy_close': spy_close,
                    'scaled_final_pnl': scaled_final_pnl,
                    'outcome': outcome,
                    'original_pnl': original_trade['pnl']
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error scaling trade: {e}")
            return {
                'date': original_trade['date'],
                'trade_executed': False,
                'pnl': 0.0,
                'error': f"scaling_error: {e}"
            }
    
    def run_theoretical_scaling_analysis(self, proven_results_file: str) -> Dict:
        """Run complete 25K scaling analysis"""
        
        self.logger.info("🚀 Starting 25K Theoretical Scaling Analysis")
        
        # Load proven results
        proven_df = self.load_proven_results(proven_results_file)
        if proven_df.empty:
            return {'error': 'Could not load proven results'}
        
        # Filter for executed trades only
        executed_trades = proven_df[proven_df['trade_executed'] == True].copy()
        self.logger.info(f"📊 Found {len(executed_trades)} executed trades to scale")
        
        # Scale each trade
        scaled_results = []
        total_scaled_pnl = 0.0
        winning_trades = 0
        
        for _, trade in executed_trades.iterrows():
            scaled_trade = self.scale_trade_result(trade.to_dict())
            scaled_results.append(scaled_trade)
            
            if scaled_trade['trade_executed']:
                pnl = scaled_trade['pnl']
                total_scaled_pnl += pnl
                if pnl > 0:
                    winning_trades += 1
                
                self.logger.info(f"📈 {scaled_trade['date']}: ${pnl:.0f} (original: ${trade['pnl']:.0f})")
        
        # Calculate performance metrics
        total_days = len(proven_df)
        execution_rate = len(executed_trades) / total_days * 100
        scaled_execution_rate = len([r for r in scaled_results if r['trade_executed']]) / total_days * 100
        win_rate = winning_trades / len(executed_trades) * 100 if executed_trades.empty == False else 0
        avg_daily_pnl = total_scaled_pnl / total_days
        
        # Risk analysis
        daily_pnls = [r['pnl'] for r in scaled_results]
        max_daily_gain = max(daily_pnls) if daily_pnls else 0
        max_daily_loss = min(daily_pnls) if daily_pnls else 0
        
        # Generate summary
        summary = {
            'analysis_type': '25K Theoretical Scaling',
            'period': f"{proven_df['date'].min()} to {proven_df['date'].max()}",
            'total_days': total_days,
            'original_total_pnl': float(proven_df['pnl'].sum()),
            'scaled_total_pnl': total_scaled_pnl,
            'scaling_factor': self.scaling_params['scaling_factor'],
            'original_avg_daily': float(proven_df['pnl'].mean()),
            'scaled_avg_daily': avg_daily_pnl,
            'target_daily': 300.0,
            'target_achievement_pct': (avg_daily_pnl / 300.0) * 100,
            'execution_rate': execution_rate,
            'win_rate': win_rate,
            'max_daily_gain': max_daily_gain,
            'max_daily_loss': max_daily_loss,
            'max_loss_pct_of_account': abs(max_daily_loss) / self.scaling_params['account_size'] * 100,
            'total_trades': len(executed_trades),
            'winning_trades': winning_trades,
            'scaling_params': self.scaling_params,
            'scaled_results': scaled_results
        }
        
        return summary
    
    def generate_scaling_report(self, summary: Dict) -> None:
        """Generate detailed scaling analysis report"""
        
        self.logger.info("\n" + "="*70)
        self.logger.info("🚀 25K ACCOUNT THEORETICAL SCALING REPORT")
        self.logger.info("="*70)
        
        # Basic Performance
        self.logger.info(f"📅 Analysis Period: {summary['period']}")
        self.logger.info(f"📊 Total Days: {summary['total_days']}")
        self.logger.info(f"📈 Total Trades: {summary['total_trades']}")
        
        self.logger.info(f"\n💰 SCALING PERFORMANCE:")
        self.logger.info(f"Original Total P&L: ${summary['original_total_pnl']:,.2f}")
        self.logger.info(f"Scaled Total P&L: ${summary['scaled_total_pnl']:,.2f}")
        self.logger.info(f"Scaling Factor: {summary['scaling_factor']:.1f}x")
        
        self.logger.info(f"\n📊 DAILY PERFORMANCE:")
        self.logger.info(f"Original Daily Avg: ${summary['original_avg_daily']:.2f}")
        self.logger.info(f"Scaled Daily Avg: ${summary['scaled_avg_daily']:.2f}")
        self.logger.info(f"🎯 Target Daily: $300.00")
        self.logger.info(f"🎯 Target Achievement: {summary['target_achievement_pct']:.1f}%")
        
        self.logger.info(f"\n⚡ EXECUTION METRICS:")
        self.logger.info(f"Execution Rate: {summary['execution_rate']:.1f}%")
        self.logger.info(f"Win Rate: {summary['win_rate']:.1f}%")
        
        self.logger.info(f"\n🛡️ RISK ANALYSIS (25K Account):")
        self.logger.info(f"Max Daily Gain: ${summary['max_daily_gain']:,.2f}")
        self.logger.info(f"Max Daily Loss: ${summary['max_daily_loss']:,.2f}")
        self.logger.info(f"Max Loss % of Account: {summary['max_loss_pct_of_account']:.2f}%")
        
        # Final assessment
        self.logger.info(f"\n🎯 FINAL ASSESSMENT:")
        if summary['target_achievement_pct'] >= 90:
            self.logger.info(f"✅ EXCELLENT: Achieving {summary['target_achievement_pct']:.0f}% of $300/day target!")
        elif summary['target_achievement_pct'] >= 75:
            self.logger.info(f"✅ VERY GOOD: Achieving {summary['target_achievement_pct']:.0f}% of $300/day target")
        elif summary['target_achievement_pct'] >= 50:
            self.logger.info(f"⚠️ MODERATE: Achieving {summary['target_achievement_pct']:.0f}% of $300/day target")
        else:
            self.logger.info(f"❌ INSUFFICIENT: Only {summary['target_achievement_pct']:.0f}% of $300/day target")
        
        if summary['max_loss_pct_of_account'] <= 4:
            self.logger.info(f"✅ RISK ACCEPTABLE: Max loss {summary['max_loss_pct_of_account']:.1f}% within 4% limit")
        else:
            self.logger.info(f"⚠️ RISK HIGH: Max loss {summary['max_loss_pct_of_account']:.1f}% exceeds 4% limit")

def main():
    """Run 25K theoretical scaling analysis"""
    
    analyzer = Phase4D25kTheoreticalScaling()
    
    # Use the proven balanced strategy results
    proven_results_file = "proper_unified_daily_20250731_142234.csv"
    
    if not os.path.exists(proven_results_file):
        print(f"❌ Could not find results file: {proven_results_file}")
        return
    
    # Run analysis
    summary = analyzer.run_theoretical_scaling_analysis(proven_results_file)
    
    if 'error' in summary:
        print(f"❌ Analysis failed: {summary['error']}")
        return
    
    # Generate report
    analyzer.generate_scaling_report(summary)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_df = pd.DataFrame(summary['scaled_results'])
    results_file = f"phase4d_25k_theoretical_scaling_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    
    # Save summary
    summary_file = f"phase4d_25k_theoretical_summary_{timestamp}.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\n💾 Results saved:")
    print(f"📊 Daily Results: {results_file}")  
    print(f"📋 Summary: {summary_file}")

if __name__ == "__main__":
    main()