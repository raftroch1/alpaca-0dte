#!/usr/bin/env python3
"""
📊 COUNTER STRATEGY RISK/REWARD ANALYSIS
========================================

Deep dive into win rate vs risk/reward profile of counter strategies.
Analyze if low win rate (30.8%) is compensated by favorable risk/reward.

Author: Strategy Development Framework
Date: 2025-08-01
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List

class CounterRiskRewardAnalysis:
    """
    Analyze risk/reward profile of counter strategies
    """
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        self.logger = logging.getLogger(__name__)
    
    def analyze_win_loss_distribution(self, trades_df: pd.DataFrame) -> Dict:
        """
        Analyze the distribution of wins vs losses
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 WIN/LOSS DISTRIBUTION ANALYSIS")
        self.logger.info(f"{'='*60}")
        
        # Separate winning and losing trades
        winning_trades = trades_df[trades_df['final_pnl'] > 0]
        losing_trades = trades_df[trades_df['final_pnl'] < 0]
        breakeven_trades = trades_df[trades_df['final_pnl'] == 0]
        
        # Basic statistics
        total_trades = len(trades_df)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        breakeven_count = len(breakeven_trades)
        
        win_rate = (win_count / total_trades) * 100
        loss_rate = (loss_count / total_trades) * 100
        
        # P&L statistics
        avg_winner = winning_trades['final_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loser = losing_trades['final_pnl'].mean() if len(losing_trades) > 0 else 0
        max_winner = winning_trades['final_pnl'].max() if len(winning_trades) > 0 else 0
        max_loser = losing_trades['final_pnl'].min() if len(losing_trades) > 0 else 0
        
        # Risk/Reward Ratio
        avg_risk_reward = abs(avg_winner / avg_loser) if avg_loser != 0 else float('inf')
        
        self.logger.info(f"📈 TRADE DISTRIBUTION:")
        self.logger.info(f"   Total Trades: {total_trades}")
        self.logger.info(f"   Winning Trades: {win_count} ({win_rate:.1f}%)")
        self.logger.info(f"   Losing Trades: {loss_count} ({loss_rate:.1f}%)")
        self.logger.info(f"   Breakeven Trades: {breakeven_count}")
        
        self.logger.info(f"\n💰 P&L ANALYSIS:")
        self.logger.info(f"   Average Winner: ${avg_winner:.2f}")
        self.logger.info(f"   Average Loser: ${avg_loser:.2f}")
        self.logger.info(f"   Max Winner: ${max_winner:.2f}")
        self.logger.info(f"   Max Loser: ${max_loser:.2f}")
        self.logger.info(f"   Risk/Reward Ratio: {avg_risk_reward:.2f}:1")
        
        # Analyze if this is a profitable system despite low win rate
        total_wins_pnl = winning_trades['final_pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses_pnl = losing_trades['final_pnl'].sum() if len(losing_trades) > 0 else 0
        net_pnl = total_wins_pnl + total_losses_pnl
        
        self.logger.info(f"\n🎯 PROFITABILITY ANALYSIS:")
        self.logger.info(f"   Total from Winners: ${total_wins_pnl:.2f}")
        self.logger.info(f"   Total from Losers: ${total_losses_pnl:.2f}")
        self.logger.info(f"   Net P&L: ${net_pnl:.2f}")
        self.logger.info(f"   Profit Factor: {abs(total_wins_pnl / total_losses_pnl):.2f}" if total_losses_pnl != 0 else "   Profit Factor: ∞")
        
        return {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'max_winner': max_winner,
            'max_loser': max_loser,
            'risk_reward_ratio': avg_risk_reward,
            'total_wins_pnl': total_wins_pnl,
            'total_losses_pnl': total_losses_pnl,
            'net_pnl': net_pnl,
            'profit_factor': abs(total_wins_pnl / total_losses_pnl) if total_losses_pnl != 0 else float('inf')
        }
    
    def analyze_by_strategy_type(self, trades_df: pd.DataFrame) -> Dict:
        """
        Break down risk/reward by strategy type
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🛡️ RISK/REWARD BY STRATEGY TYPE")
        self.logger.info(f"{'='*60}")
        
        strategy_analysis = {}
        
        for strategy in trades_df['strategy'].unique():
            strategy_trades = trades_df[trades_df['strategy'] == strategy]
            strategy_results = self.analyze_win_loss_distribution(strategy_trades)
            strategy_analysis[strategy] = strategy_results
            
            self.logger.info(f"\n📊 {strategy.upper()}:")
            self.logger.info(f"   Trades: {strategy_results['total_trades']}")
            self.logger.info(f"   Win Rate: {strategy_results['win_rate']:.1f}%")
            self.logger.info(f"   Avg Win: ${strategy_results['avg_winner']:.2f}")
            self.logger.info(f"   Avg Loss: ${strategy_results['avg_loser']:.2f}")
            self.logger.info(f"   Risk/Reward: {strategy_results['risk_reward_ratio']:.2f}:1")
            self.logger.info(f"   Profit Factor: {strategy_results['profit_factor']:.2f}")
        
        return strategy_analysis
    
    def analyze_risk_exposure(self, trades_df: pd.DataFrame) -> Dict:
        """
        Analyze actual risk exposure vs theoretical max risk
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"⚠️ RISK EXPOSURE ANALYSIS")
        self.logger.info(f"{'='*60}")
        
        # For counter strategies, analyze max risk vs actual loss
        if 'max_risk' in trades_df.columns:
            avg_max_risk = trades_df['max_risk'].mean()
            max_theoretical_risk = trades_df['max_risk'].max()
            
            # Compare to actual losses
            losing_trades = trades_df[trades_df['final_pnl'] < 0]
            if len(losing_trades) > 0:
                avg_actual_loss = abs(losing_trades['final_pnl'].mean())
                max_actual_loss = abs(losing_trades['final_pnl'].min())
                
                # Risk utilization
                avg_risk_utilization = (avg_actual_loss / avg_max_risk) * 100
                max_risk_utilization = (max_actual_loss / max_theoretical_risk) * 100
                
                self.logger.info(f"📊 THEORETICAL VS ACTUAL RISK:")
                self.logger.info(f"   Avg Max Risk: ${avg_max_risk:.2f}")
                self.logger.info(f"   Max Theoretical Risk: ${max_theoretical_risk:.2f}")
                self.logger.info(f"   Avg Actual Loss: ${avg_actual_loss:.2f}")
                self.logger.info(f"   Max Actual Loss: ${max_actual_loss:.2f}")
                self.logger.info(f"   Avg Risk Utilization: {avg_risk_utilization:.1f}%")
                self.logger.info(f"   Max Risk Utilization: {max_risk_utilization:.1f}%")
                
                return {
                    'avg_max_risk': avg_max_risk,
                    'max_theoretical_risk': max_theoretical_risk,
                    'avg_actual_loss': avg_actual_loss,
                    'max_actual_loss': max_actual_loss,
                    'avg_risk_utilization': avg_risk_utilization,
                    'max_risk_utilization': max_risk_utilization
                }
        
        return {}
    
    def calculate_expectancy(self, analysis_results: Dict) -> Dict:
        """
        Calculate mathematical expectancy and other key metrics
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🧮 MATHEMATICAL EXPECTANCY ANALYSIS")
        self.logger.info(f"{'='*60}")
        
        # Mathematical expectancy = (Win Rate × Avg Win) - (Loss Rate × |Avg Loss|)
        win_rate_decimal = analysis_results['win_rate'] / 100
        loss_rate_decimal = analysis_results['loss_rate'] / 100
        avg_winner = analysis_results['avg_winner']
        avg_loser = abs(analysis_results['avg_loser'])
        
        expectancy = (win_rate_decimal * avg_winner) - (loss_rate_decimal * avg_loser)
        expectancy_pct = (expectancy / avg_loser) * 100 if avg_loser > 0 else 0
        
        # Kelly Criterion for optimal position sizing
        win_prob = win_rate_decimal
        loss_prob = loss_rate_decimal
        win_loss_ratio = avg_winner / avg_loser if avg_loser > 0 else 0
        
        kelly_fraction = win_prob - (loss_prob / win_loss_ratio) if win_loss_ratio > 0 else 0
        kelly_fraction = max(0, kelly_fraction)  # Don't bet if Kelly is negative
        
        # Sharpe-like ratio for the strategy
        total_trades = analysis_results['total_trades']
        trade_returns = []  # We'd need individual trade data for this
        
        self.logger.info(f"📈 EXPECTANCY METRICS:")
        self.logger.info(f"   Mathematical Expectancy: ${expectancy:.2f} per trade")
        self.logger.info(f"   Expectancy %: {expectancy_pct:.2f}%")
        self.logger.info(f"   Win/Loss Ratio: {win_loss_ratio:.2f}")
        self.logger.info(f"   Kelly Fraction: {kelly_fraction:.3f} ({kelly_fraction*100:.1f}% of capital)")
        
        # Is this a viable strategy?
        if expectancy > 0:
            self.logger.info(f"✅ POSITIVE EXPECTANCY STRATEGY")
        else:
            self.logger.info(f"❌ NEGATIVE EXPECTANCY STRATEGY")
        
        if kelly_fraction > 0:
            self.logger.info(f"✅ KELLY SUGGESTS BETTING (optimal {kelly_fraction*100:.1f}% of capital)")
        else:
            self.logger.info(f"❌ KELLY SUGGESTS NO BETTING")
        
        return {
            'expectancy': expectancy,
            'expectancy_pct': expectancy_pct,
            'win_loss_ratio': win_loss_ratio,
            'kelly_fraction': kelly_fraction,
            'is_positive_expectancy': expectancy > 0,
            'kelly_suggests_betting': kelly_fraction > 0
        }
    
    def run_complete_analysis(self, trades_file: str = "proper_unified_trades_20250801_114942.csv") -> Dict:
        """
        Run complete risk/reward analysis
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"📊 COUNTER STRATEGY RISK/REWARD ANALYSIS")
        self.logger.info(f"🎯 Goal: Understand if 30.8% win rate is acceptable given risk/reward")
        self.logger.info(f"{'='*80}")
        
        # Load and filter counter strategy trades
        trades_df = pd.read_csv(trades_file)
        counter_strategies = ['closer_atm_bull_put_spread', 'bear_put_spread', 'short_call_supplement']
        counter_trades = trades_df[trades_df['strategy'].isin(counter_strategies)]
        
        self.logger.info(f"📊 Analyzing {len(counter_trades)} counter strategy trades...")
        
        # Overall analysis
        overall_analysis = self.analyze_win_loss_distribution(counter_trades)
        
        # By strategy type
        strategy_analysis = self.analyze_by_strategy_type(counter_trades)
        
        # Risk exposure
        risk_analysis = self.analyze_risk_exposure(counter_trades)
        
        # Mathematical expectancy
        expectancy_analysis = self.calculate_expectancy(overall_analysis)
        
        # Final verdict
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🎯 FINAL RISK/REWARD VERDICT")
        self.logger.info(f"{'='*80}")
        
        if expectancy_analysis['is_positive_expectancy'] and expectancy_analysis['kelly_suggests_betting']:
            self.logger.info(f"✅ VERDICT: Low win rate (30.8%) is ACCEPTABLE")
            self.logger.info(f"💡 REASON: High risk/reward ratio ({overall_analysis['risk_reward_ratio']:.2f}:1) creates positive expectancy")
            self.logger.info(f"🎯 STRATEGY: Continue with counter strategies but size positions carefully")
        else:
            self.logger.info(f"❌ VERDICT: Low win rate is PROBLEMATIC")
            self.logger.info(f"💡 REASON: Risk/reward ratio insufficient to overcome low win rate")
            self.logger.info(f"🎯 STRATEGY: Reconsider counter strategy parameters or avoid")
        
        return {
            'overall_analysis': overall_analysis,
            'strategy_analysis': strategy_analysis,
            'risk_analysis': risk_analysis,
            'expectancy_analysis': expectancy_analysis
        }

def main():
    analyzer = CounterRiskRewardAnalysis()
    results = analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()