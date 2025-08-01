#!/usr/bin/env python3
"""
🚀 PHASE 4D PAPER TRADING LAUNCHER
=================================

Easy launcher for paper trading with real-time monitoring.
Validates backtest assumptions against live market performance.

USAGE:
    python paper_trading_launcher.py --account 2k    # 2-contract strategy  
    python paper_trading_launcher.py --account 25k   # 25-contract strategy

FEATURES:
- Live market data integration
- Real-time performance monitoring  
- Backtest vs live comparison
- Automatic position management
- Risk management alerts
- Daily performance reports
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
import signal
import threading
import time
from typing import Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from phase4d_paper_trading import Phase4DPaperTrading

class PaperTradingLauncher:
    """Enhanced launcher with monitoring and reporting"""
    
    def __init__(self, account_size: str):
        self.account_size = account_size
        self.strategy: Optional[Phase4DPaperTrading] = None
        self.monitoring_active = False
        self.setup_logging()
        
    def setup_logging(self):
        """Setup enhanced logging for paper trading"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"paper_trading_{self.account_size}_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"📋 Logging to: {log_filename}")
    
    def validate_environment(self) -> bool:
        """Validate trading environment and credentials"""
        self.logger.info("🔍 Validating trading environment...")
        
        # Check Alpaca credentials
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            self.logger.error("❌ Alpaca credentials not found")
            self.logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
            return False
        
        # Check market hours
        now = datetime.now()
        current_time = now.time()
        
        # Basic market hours check (9:30 AM - 4:00 PM ET)
        from datetime import time
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        if current_time < market_open or current_time > market_close:
            self.logger.warning("⚠️  Currently outside market hours")
            self.logger.info(f"Current time: {current_time}")
            self.logger.info(f"Market hours: {market_open} - {market_close} ET")
        
        self.logger.info("✅ Environment validation complete")
        return True
    
    def print_strategy_summary(self):
        """Print strategy configuration summary"""
        print("\n" + "="*60)
        print("🚀 PHASE 4D PAPER TRADING STRATEGY")
        print("="*60)
        print(f"📊 Account Size: {self.account_size.upper()}")
        
        if self.account_size == "25k":
            print(f"💰 Target Daily P&L: $341.77")
            print(f"📈 Position Size: 25 contracts")
            print(f"🛡️ Max Daily Loss: $1,500")
            print(f"🎯 Max Loss Per Trade: $1,000")
        else:
            print(f"💰 Target Daily P&L: $34.38")
            print(f"📈 Position Size: 2 contracts")
            print(f"🛡️ Max Daily Loss: $500")
            print(f"🎯 Max Loss Per Trade: $200")
        
        print(f"📊 Expected Execution Rate: 60.9%")
        print(f"🏆 Expected Win Rate: 62.3%")
        print(f"⚡ Strategy: ITM Put Sales (0DTE)")
        print(f"🎯 Market: SPY Options")
        print(f"📋 Mode: Paper Trading (Alpaca)")
        print("="*60)
    
    def start_monitoring_thread(self):
        """Start background monitoring thread"""
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        monitor_thread.start()
        self.logger.info("📊 Background monitoring started")
    
    def _monitor_performance(self):
        """Background performance monitoring"""
        while self.monitoring_active and self.strategy:
            try:
                # Monitor every 5 minutes
                time.sleep(300)
                
                if not self.strategy:
                    continue
                
                # Log current status
                current_pnl = self.strategy.daily_pnl
                trades_today = self.strategy.daily_trades
                expected_pnl = self.strategy.backtest_comparison['expected_daily_pnl']
                
                self.logger.info(f"📊 Performance Check:")
                self.logger.info(f"   Current P&L: ${current_pnl:.2f}")
                self.logger.info(f"   Expected P&L: ${expected_pnl:.2f}")
                self.logger.info(f"   Trades Today: {trades_today}")
                
                # Check for significant deviations
                if abs(current_pnl) > abs(expected_pnl) * 2:
                    self.logger.warning(f"⚠️  P&L deviation alert: {current_pnl:.2f} vs expected {expected_pnl:.2f}")
                
                # Check daily loss limit
                if current_pnl <= -self.strategy.params['max_daily_loss']:
                    self.logger.error(f"🚨 DAILY LOSS LIMIT HIT: ${current_pnl:.2f}")
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received shutdown signal ({signum})")
            self.shutdown_gracefully()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def shutdown_gracefully(self):
        """Graceful shutdown procedure"""
        self.logger.info("🛑 Initiating graceful shutdown...")
        
        # Stop monitoring
        self.monitoring_active = False
        
        # Stop trading
        if self.strategy:
            self.strategy.trading_active = False
            self.logger.info("📈 Trading stopped")
        
        self.logger.info("✅ Shutdown complete")
    
    async def run_paper_trading(self):
        """Main paper trading execution"""
        try:
            # Initialize strategy
            self.logger.info(f"🚀 Initializing {self.account_size.upper()} strategy...")
            self.strategy = Phase4DPaperTrading(account_size=self.account_size)
            
            if not self.strategy.trading_client:
                self.logger.error("❌ Failed to initialize trading client")
                return False
            
            # Start monitoring
            self.start_monitoring_thread()
            
            # Run trading session
            self.logger.info("🚀 Starting paper trading session...")
            await self.strategy.run_trading_session()
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Trading interrupted by user")
            return True
        except Exception as e:
            self.logger.error(f"❌ Trading session error: {e}")
            return False
        finally:
            # Generate final report
            if self.strategy:
                self.logger.info("📊 Generating final performance report...")
                self.strategy.generate_daily_report()
    
    def display_final_summary(self):
        """Display final trading session summary"""
        if not self.strategy:
            return
        
        print("\n" + "="*60)
        print("📊 TRADING SESSION SUMMARY")
        print("="*60)
        print(f"💰 Final P&L: ${self.strategy.daily_pnl:.2f}")
        print(f"📈 Trades Executed: {self.strategy.daily_trades}")
        print(f"🎯 Target Achievement: {(self.strategy.daily_pnl / self.strategy.backtest_comparison['expected_daily_pnl'] * 100):.1f}%")
        
        if self.strategy.orders:
            print(f"📋 Trade Details:")
            for order_id, trade in self.strategy.orders.items():
                status = trade.get('status', 'unknown')
                symbol = trade.get('symbol', 'N/A')
                pnl = trade.get('current_pnl', 0)
                print(f"   {order_id[:8]}: {symbol} → ${pnl:.2f} ({status})")
        
        print("="*60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Phase 4D Paper Trading Launcher')
    parser.add_argument('--account', choices=['2k', '25k'], required=True,
                       help='Account size: 2k (2 contracts) or 25k (25 contracts)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate environment, don\'t start trading')
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = PaperTradingLauncher(args.account)
    
    # Validate environment
    if not launcher.validate_environment():
        sys.exit(1)
    
    if args.validate_only:
        print("✅ Environment validation successful")
        return
    
    # Display strategy summary
    launcher.print_strategy_summary()
    
    # Confirm before starting
    print("\n🚀 Ready to start paper trading!")
    print("📋 This will execute real paper trades on Alpaca")
    print("🛑 Press Ctrl+C to stop trading at any time")
    
    confirm = input("\nStart trading? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Trading cancelled by user")
        return
    
    # Setup signal handlers
    launcher.setup_signal_handlers()
    
    # Run paper trading
    print("\n🚀 Starting paper trading session...")
    try:
        success = asyncio.run(launcher.run_paper_trading())
        
        if success:
            launcher.display_final_summary()
            print("✅ Trading session completed successfully")
        else:
            print("❌ Trading session ended with errors")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Trading session interrupted by user")
        launcher.display_final_summary()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()