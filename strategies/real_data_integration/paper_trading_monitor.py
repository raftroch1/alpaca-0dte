#!/usr/bin/env python3
"""
📊 PHASE 4D PAPER TRADING MONITOR
================================

Real-time monitoring dashboard for paper trading performance.
Validates backtest assumptions against live market execution.

FEATURES:
- Real-time P&L tracking
- Backtest vs live comparison
- Risk monitoring alerts
- Performance analytics
- Trade execution analysis
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

try:
    from alpaca.trading.client import TradingClient
    from config.trading_config import ALPACA_CONFIG
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

class PaperTradingMonitor:
    """Real-time paper trading monitor and analytics"""
    
    def __init__(self, account_size: str = "2k"):
        self.account_size = account_size
        self.setup_logging()
        self.setup_alpaca_client()
        
        # Performance tracking
        self.performance_data = {
            'session_start': datetime.now(),
            'account_size': account_size,
            'expected_daily_pnl': 34.38 if account_size == "2k" else 341.77,
            'backtest_execution_rate': 0.609,
            'backtest_win_rate': 0.623,
            'live_trades': [],
            'live_pnl_history': [],
            'alerts': [],
            'risk_metrics': {}
        }
        
        self.monitoring_active = False
        
    def setup_logging(self):
        """Setup monitoring logging"""
        self.logger = logging.getLogger(f"Monitor_{self.account_size}")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def setup_alpaca_client(self):
        """Initialize Alpaca client for monitoring"""
        if not ALPACA_AVAILABLE:
            self.logger.error("❌ Alpaca SDK not available")
            self.trading_client = None
            return
        
        try:
            self.trading_client = TradingClient(
                api_key=ALPACA_CONFIG['API_KEY'],
                secret_key=ALPACA_CONFIG['SECRET_KEY'],
                paper=True
            )
            
            # Test connection
            account = self.trading_client.get_account()
            self.logger.info(f"✅ Connected to paper account: ${account.equity}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Alpaca: {e}")
            self.trading_client = None
    
    def get_current_positions(self) -> List[Dict]:
        """Get current open positions"""
        try:
            if not self.trading_client:
                return []
            
            positions = self.trading_client.get_all_positions()
            
            position_data = []
            for pos in positions:
                position_data.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pnl': float(pos.unrealized_pnl),
                    'unrealized_pnl_pct': float(pos.unrealized_plpc) * 100,
                    'side': 'long' if float(pos.qty) > 0 else 'short'
                })
            
            return position_data
            
        except Exception as e:
            self.logger.error(f"❌ Error getting positions: {e}")
            return []
    
    def get_account_info(self) -> Optional[Dict]:
        """Get current account information"""
        try:
            if not self.trading_client:
                return None
            
            account = self.trading_client.get_account()
            
            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'last_equity': float(account.last_equity),
                'daytrade_count': int(account.daytrade_count),
                'daytrading_buying_power': float(account.daytrading_buying_power)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting account info: {e}")
            return None
    
    def calculate_daily_pnl(self, account_info: Dict) -> float:
        """Calculate current daily P&L"""
        try:
            current_equity = account_info['equity']
            last_equity = account_info['last_equity']
            
            daily_pnl = current_equity - last_equity
            return daily_pnl
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating daily P&L: {e}")
            return 0.0
    
    def analyze_performance_vs_backtest(self) -> Dict:
        """Analyze live performance against backtest expectations"""
        try:
            account_info = self.get_account_info()
            if not account_info:
                return {}
            
            daily_pnl = self.calculate_daily_pnl(account_info)
            expected_pnl = self.performance_data['expected_daily_pnl']
            
            # Calculate variance
            variance_pct = ((daily_pnl - expected_pnl) / expected_pnl * 100) if expected_pnl != 0 else 0
            
            # Update history
            self.performance_data['live_pnl_history'].append({
                'timestamp': datetime.now(),
                'pnl': daily_pnl,
                'expected_pnl': expected_pnl,
                'variance_pct': variance_pct
            })
            
            analysis = {
                'current_pnl': daily_pnl,
                'expected_pnl': expected_pnl,
                'variance_pct': variance_pct,
                'performance_status': self._get_performance_status(variance_pct),
                'account_equity': account_info['equity'],
                'positions_count': len(self.get_current_positions())
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing performance: {e}")
            return {}
    
    def _get_performance_status(self, variance_pct: float) -> str:
        """Get performance status based on variance"""
        if variance_pct > 20:
            return "OUTPERFORMING"
        elif variance_pct > -20:
            return "ON_TARGET"
        elif variance_pct > -50:
            return "UNDERPERFORMING"
        else:
            return "CONCERNING"
    
    def check_risk_alerts(self, account_info: Dict, positions: List[Dict]) -> List[Dict]:
        """Check for risk management alerts"""
        alerts = []
        
        try:
            daily_pnl = self.calculate_daily_pnl(account_info)
            
            # Daily loss limits
            max_daily_loss = 500 if self.account_size == "2k" else 1500
            if daily_pnl <= -max_daily_loss:
                alerts.append({
                    'type': 'DAILY_LOSS_LIMIT',
                    'severity': 'CRITICAL',
                    'message': f"Daily loss limit hit: ${daily_pnl:.2f}",
                    'timestamp': datetime.now()
                })
            elif daily_pnl <= -max_daily_loss * 0.8:
                alerts.append({
                    'type': 'DAILY_LOSS_WARNING',
                    'severity': 'WARNING',
                    'message': f"Approaching daily loss limit: ${daily_pnl:.2f}",
                    'timestamp': datetime.now()
                })
            
            # Position size alerts
            max_positions = 1  # We should only have 1 position at a time
            if len(positions) > max_positions:
                alerts.append({
                    'type': 'POSITION_COUNT',
                    'severity': 'WARNING',
                    'message': f"Too many positions: {len(positions)}",
                    'timestamp': datetime.now()
                })
            
            # Large unrealized loss alerts
            for pos in positions:
                max_loss_per_trade = 200 if self.account_size == "2k" else 1000
                if pos['unrealized_pnl'] <= -max_loss_per_trade:
                    alerts.append({
                        'type': 'POSITION_LOSS',
                        'severity': 'CRITICAL',
                        'message': f"Large loss in {pos['symbol']}: ${pos['unrealized_pnl']:.2f}",
                        'timestamp': datetime.now()
                    })
            
            # Store alerts
            self.performance_data['alerts'].extend(alerts)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"❌ Error checking risk alerts: {e}")
            return []
    
    def display_dashboard(self):
        """Display real-time monitoring dashboard"""
        try:
            # Clear screen (works on most terminals)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("="*80)
            print(f"📊 PHASE 4D PAPER TRADING MONITOR - {self.account_size.upper()} ACCOUNT")
            print("="*80)
            print(f"⏰ Update Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get current data
            account_info = self.get_account_info()
            positions = self.get_current_positions()
            analysis = self.analyze_performance_vs_backtest()
            alerts = self.check_risk_alerts(account_info, positions) if account_info else []
            
            if not account_info:
                print("❌ Unable to connect to Alpaca account")
                return
            
            # Account Summary
            print(f"\n💼 ACCOUNT SUMMARY:")
            print(f"   Equity: ${account_info['equity']:,.2f}")
            print(f"   Buying Power: ${account_info['buying_power']:,.2f}")
            print(f"   Portfolio Value: ${account_info['portfolio_value']:,.2f}")
            
            # Performance Analysis
            if analysis:
                print(f"\n📈 PERFORMANCE ANALYSIS:")
                print(f"   Current Daily P&L: ${analysis['current_pnl']:,.2f}")
                print(f"   Expected Daily P&L: ${analysis['expected_pnl']:,.2f}")
                print(f"   Variance: {analysis['variance_pct']:+.1f}%")
                print(f"   Status: {analysis['performance_status']}")
            
            # Current Positions
            print(f"\n📊 CURRENT POSITIONS ({len(positions)}):")
            if positions:
                for pos in positions:
                    side_icon = "📈" if pos['side'] == 'long' else "📉"
                    print(f"   {side_icon} {pos['symbol']}: {pos['qty']:+.0f} @ ${pos['avg_entry_price']:.3f}")
                    print(f"      Market Value: ${pos['market_value']:,.2f}")
                    print(f"      Unrealized P&L: ${pos['unrealized_pnl']:+,.2f} ({pos['unrealized_pnl_pct']:+.1f}%)")
            else:
                print("   No open positions")
            
            # Risk Alerts
            if alerts:
                print(f"\n🚨 RISK ALERTS ({len(alerts)}):")
                for alert in alerts[-5:]:  # Show last 5 alerts
                    severity_icon = "🔴" if alert['severity'] == 'CRITICAL' else "🟡"
                    print(f"   {severity_icon} {alert['type']}: {alert['message']}")
            else:
                print(f"\n✅ NO ACTIVE ALERTS")
            
            # Trading Statistics
            session_duration = datetime.now() - self.performance_data['session_start']
            print(f"\n📊 SESSION STATISTICS:")
            print(f"   Session Duration: {session_duration}")
            print(f"   Updates: {len(self.performance_data['live_pnl_history'])}")
            print(f"   Expected Execution Rate: {self.performance_data['backtest_execution_rate']*100:.1f}%")
            print(f"   Expected Win Rate: {self.performance_data['backtest_win_rate']*100:.1f}%")
            
            print("="*80)
            print("🔄 Updating every 30 seconds... Press Ctrl+C to stop")
            
        except Exception as e:
            self.logger.error(f"❌ Error displaying dashboard: {e}")
    
    def save_performance_report(self):
        """Save detailed performance report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"paper_trading_monitor_report_{self.account_size}_{timestamp}.json"
            
            # Prepare report data
            report_data = {
                **self.performance_data,
                'session_start': self.performance_data['session_start'].isoformat(),
                'final_analysis': self.analyze_performance_vs_backtest(),
                'final_account_info': self.get_account_info(),
                'final_positions': self.get_current_positions()
            }
            
            # Convert datetime objects to strings
            for pnl_entry in report_data['live_pnl_history']:
                pnl_entry['timestamp'] = pnl_entry['timestamp'].isoformat()
            
            for alert in report_data['alerts']:
                alert['timestamp'] = alert['timestamp'].isoformat()
            
            # Save report
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Performance report saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving report: {e}")
    
    async def run_monitoring(self):
        """Run continuous monitoring loop"""
        self.logger.info("🚀 Starting paper trading monitor")
        self.monitoring_active = True
        
        try:
            while self.monitoring_active:
                self.display_dashboard()
                
                # Wait 30 seconds between updates
                for _ in range(30):
                    if not self.monitoring_active:
                        break
                    await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Monitoring error: {e}")
        finally:
            self.monitoring_active = False
            self.save_performance_report()
            print("\n📊 Final report saved. Monitor stopped.")

def main():
    """Main entry point for monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 4D Paper Trading Monitor')
    parser.add_argument('--account', choices=['2k', '25k'], default='2k',
                       help='Account size to monitor (default: 2k)')
    
    args = parser.parse_args()
    
    if not ALPACA_AVAILABLE:
        print("❌ Alpaca SDK not available. Please install: pip install alpaca-py")
        return
    
    print(f"📊 Starting Phase 4D Paper Trading Monitor ({args.account.upper()} account)")
    print("🔄 Real-time monitoring dashboard will update every 30 seconds")
    print("🛑 Press Ctrl+C to stop monitoring and save final report")
    
    monitor = PaperTradingMonitor(args.account)
    
    try:
        asyncio.run(monitor.run_monitoring())
    except KeyboardInterrupt:
        print("\n✅ Monitoring session complete")

if __name__ == "__main__":
    main()