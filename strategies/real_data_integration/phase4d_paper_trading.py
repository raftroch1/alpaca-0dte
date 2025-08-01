"""
🚀 PHASE 4D PAPER TRADING STRATEGY
==================================

LIVE PAPER TRADING implementation of our proven Phase 4D Balanced Strategy.
Follows EXACT backtest logic to validate real-world performance.

✅ EXACT BACKTEST MATCHING:
- Same strike selection algorithm  
- Same volatility filtering
- Same risk management rules
- Same position sizing (2 contracts → 25 contracts for 25K)
- Same cost calculations

✅ LIVE MARKET INTEGRATION:
- Real-time SPY prices
- Live option chain data
- Alpaca Paper Trading API
- Real execution timing

✅ PERFORMANCE VALIDATION:
- Track vs backtest expectations
- Monitor slippage differences
- Validate cost assumptions
- Compare daily P&L patterns
"""

import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
import signal
import json
from decimal import Decimal

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
    from alpaca.data.live import StockDataStream
    from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, OptionBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
    from config.trading_config import ALPACA_CONFIG
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  Alpaca SDK not available")

# Import our proven strategy logic
from phase4d_balanced_minimal_scale import Phase4DBalancedMinimalScale

class Phase4DPaperTrading(Phase4DBalancedMinimalScale):
    """
    🎯 LIVE PAPER TRADING STRATEGY
    
    Inherits exact logic from proven balanced strategy.
    Adds live market integration and paper trading execution.
    """
    
    def __init__(self, account_size: str = "2k"):
        """
        Initialize paper trading strategy
        
        Args:
            account_size: "2k" for 2-contract strategy, "25k" for 25-contract strategy
        """
        # Initialize parent strategy
        super().__init__()
        
        self.account_size = account_size
        self.setup_paper_trading()
        
        # Scale parameters based on account size
        if account_size == "25k":
            self.params['max_contracts'] = 25
            self.params['max_daily_loss'] = 1500
            self.params['max_loss_per_trade'] = 1000
            self.logger.info("🚀 Configured for 25K account (25 contracts)")
        else:
            self.params['max_contracts'] = 2
            self.params['max_daily_loss'] = 500
            self.params['max_loss_per_trade'] = 200
            self.logger.info("📊 Configured for 2K account (2 contracts)")
        
        # Live trading state
        self.trading_active = False
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.positions = {}
        self.orders = {}
        self.market_session = None
        
        # Performance tracking vs backtest
        self.backtest_comparison = {
            'expected_daily_pnl': 34.38 if account_size == "2k" else 341.77,
            'actual_daily_pnl': [],
            'execution_rate': 0.609,  # From backtest
            'actual_execution_rate': [],
            'variance_from_backtest': []
        }
        
        self.logger.info("✅ Phase 4D Paper Trading Strategy initialized")
    
    def setup_paper_trading(self):
        """Initialize Alpaca paper trading clients"""
        if not ALPACA_AVAILABLE:
            self.logger.error("❌ Alpaca SDK not available")
            return
        
        try:
            # Trading client for paper orders
            self.trading_client = TradingClient(
                api_key=ALPACA_CONFIG['API_KEY'],
                secret_key=ALPACA_CONFIG['SECRET_KEY'],
                paper=True  # PAPER TRADING MODE
            )
            
            # Get account info
            account = self.trading_client.get_account()
            self.logger.info(f"📋 Paper Account: ${account.equity} equity")
            self.logger.info(f"💰 Buying Power: ${account.buying_power}")
            
            self.logger.info("✅ Paper trading clients initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Paper trading setup failed: {e}")
            self.trading_client = None
    
    def get_live_spy_price(self) -> Optional[float]:
        """Get current live SPY price"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=["SPY"])
            quotes = self.stock_client.get_stock_latest_quote(request)
            
            if "SPY" in quotes:
                # Use midpoint of bid/ask for fair price
                bid = float(quotes["SPY"].bid_price)
                ask = float(quotes["SPY"].ask_price)
                mid_price = (bid + ask) / 2
                
                self.logger.info(f"📊 Live SPY: ${mid_price:.2f} (bid: ${bid:.2f}, ask: ${ask:.2f})")
                return mid_price
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting live SPY price: {e}")
            return None
    
    def get_live_option_chain(self, expiry_date: str) -> Optional[Dict]:
        """Get live 0DTE option chain for SPY"""
        try:
            # Format: SPY241231C00450000 (SPY + YYMMDD + C/P + strike*1000)
            today = datetime.now()
            exp_str = today.strftime("%y%m%d")
            
            # Get range of strikes around current SPY price
            spy_price = self.get_live_spy_price()
            if not spy_price:
                return None
            
            option_chain = {}
            
            # Check puts around SPY price (we trade puts)
            for strike_offset in range(-20, 5):  # Below current price mostly
                strike = round(spy_price + strike_offset)
                put_symbol = f"SPY{exp_str}P{strike:08.0f}"
                
                try:
                    # Get live option quote
                    request = StockLatestQuoteRequest(symbol_or_symbols=[put_symbol])
                    quotes = self.stock_client.get_stock_latest_quote(request)
                    
                    if put_symbol in quotes:
                        quote = quotes[put_symbol]
                        bid = float(quote.bid_price) if quote.bid_price else 0
                        ask = float(quote.ask_price) if quote.ask_price else 0
                        
                        if bid > 0 and ask > 0:
                            mid_price = (bid + ask) / 2
                            spread = (ask - bid) / mid_price if mid_price > 0 else 1.0
                            
                            option_chain[strike] = {
                                'symbol': put_symbol,
                                'strike': strike,
                                'bid': bid,
                                'ask': ask,
                                'mid': mid_price,
                                'spread_pct': spread * 100
                            }
                            
                except Exception as option_error:
                    # Some strikes may not exist
                    continue
            
            self.logger.info(f"📋 Found {len(option_chain)} live 0DTE put options")
            return option_chain
            
        except Exception as e:
            self.logger.error(f"❌ Error getting option chain: {e}")
            return None
    
    def live_strike_selection(self, spy_price: float, option_chain: Dict) -> Optional[Dict]:
        """
        🎯 EXACT SAME STRIKE SELECTION AS BACKTEST
        
        Uses identical logic to our proven balanced strategy
        """
        try:
            # Same buffer logic as backtest
            if spy_price > 500:
                buffer = 0.5
            elif spy_price > 400:
                buffer = 0.3
            else:
                buffer = 0.0
            
            # Target ITM puts (strikes above current SPY)
            target_strike = spy_price + buffer
            
            # Find closest available strike
            available_strikes = sorted(option_chain.keys())
            best_strike = min(available_strikes, key=lambda x: abs(x - target_strike))
            
            if best_strike in option_chain:
                option_info = option_chain[best_strike]
                
                # Same premium filter as backtest
                if option_info['mid'] >= self.params['min_premium']:
                    
                    self.logger.info(f"🎯 LIVE STRIKE SELECTED (same logic as backtest):")
                    self.logger.info(f"   SPY: ${spy_price:.2f}")
                    self.logger.info(f"   Target Strike: ${target_strike:.1f}")
                    self.logger.info(f"   Selected Strike: ${best_strike}")
                    self.logger.info(f"   Premium: ${option_info['mid']:.3f}")
                    self.logger.info(f"   Spread: {option_info['spread_pct']:.1f}%")
                    
                    return option_info
            
            self.logger.warning(f"❌ No suitable live strikes found (same criteria as backtest)")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error in live strike selection: {e}")
            return None
    
    def calculate_live_market_conditions(self) -> Dict:
        """Calculate live market conditions using same logic as backtest"""
        try:
            # Get recent SPY bars for volatility calculation
            request = StockBarsRequest(
                symbol_or_symbols=["SPY"],
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(hours=6),  # Last 6 hours
                end=datetime.now()
            )
            
            bars = self.stock_client.get_stock_bars(request)
            
            if "SPY" not in bars or len(bars["SPY"]) < 10:
                return {'vix_estimate': 15.0, 'daily_range': 0.0}
            
            spy_bars = bars["SPY"]
            
            # Calculate daily range (same as backtest)
            highs = [float(bar.high) for bar in spy_bars]
            lows = [float(bar.low) for bar in spy_bars]
            open_price = float(spy_bars[0].open)
            
            daily_high = max(highs)
            daily_low = min(lows)
            daily_range = ((daily_high - daily_low) / open_price) * 100
            
            # Calculate VIX estimate (same as backtest)
            closes = [float(bar.close) for bar in spy_bars]
            returns = []
            for i in range(1, len(closes)):
                if closes[i-1] > 0:
                    returns.append((closes[i] - closes[i-1]) / closes[i-1])
            
            if len(returns) >= 5:
                std_dev = np.std(returns)
                vix_estimate = std_dev * np.sqrt(252 * 390) * 100
                vix_estimate = max(10.0, min(50.0, vix_estimate))
            else:
                vix_estimate = 15.0
            
            self.logger.info(f"📊 Live Market Conditions:")
            self.logger.info(f"   Daily Range: {daily_range:.1f}%")
            self.logger.info(f"   VIX Estimate: {vix_estimate:.1f}")
            
            return {
                'daily_range': daily_range,
                'vix_estimate': vix_estimate
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating market conditions: {e}")
            return {'vix_estimate': 15.0, 'daily_range': 0.0}
    
    def should_trade_today(self, market_conditions: Dict) -> Tuple[bool, str]:
        """
        🛡️ EXACT SAME FILTERING AS BACKTEST
        
        Uses identical volatility and market condition filters
        """
        vix_estimate = market_conditions['vix_estimate']
        daily_range = market_conditions['daily_range']
        
        # Same filters as backtest
        if vix_estimate > self.params['max_vix_threshold']:
            return False, f"high_vix_{vix_estimate:.1f}"
        
        if daily_range > self.params['disaster_threshold']:
            return False, f"disaster_range_{daily_range:.1f}%"
        
        if daily_range > self.params['max_daily_range']:
            return False, f"high_range_{daily_range:.1f}%"
        
        # Check daily loss limit
        if self.daily_pnl <= -self.params['max_daily_loss']:
            return False, f"daily_loss_limit_{self.daily_pnl:.0f}"
        
        # Check max trades per day
        if self.daily_trades >= self.params['max_daily_trades']:
            return False, f"max_daily_trades_{self.daily_trades}"
        
        return True, "conditions_met"
    
    async def execute_paper_trade(self, option_info: Dict) -> Optional[Dict]:
        """
        📈 EXECUTE PAPER TRADE
        
        Places actual paper order on Alpaca matching backtest logic
        """
        try:
            if not self.trading_client:
                self.logger.error("❌ Trading client not available")
                return None
            
            symbol = option_info['symbol']
            contracts = self.params['max_contracts']
            entry_price = option_info['mid']
            
            # Create market order (same as we'd do in real trading)
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=contracts,
                side=OrderSide.SELL,  # We sell puts (collect premium)
                time_in_force=TimeInForce.DAY,
                client_order_id=f"phase4d_{datetime.now().strftime('%H%M%S')}"
            )
            
            # Submit the order
            order = self.trading_client.submit_order(order_request)
            
            # Calculate expected P&L using backtest logic
            expected_costs = self.calculate_realistic_costs(
                entry_price * 100 * contracts,  # Gross premium
                contracts
            )
            
            trade_info = {
                'order_id': order.id,
                'symbol': symbol,
                'contracts': contracts,
                'entry_price': entry_price,
                'strike': option_info['strike'],
                'entry_time': datetime.now(),
                'expected_costs': expected_costs,
                'gross_premium': entry_price * 100 * contracts,
                'status': 'submitted'
            }
            
            # Track the trade
            self.orders[order.id] = trade_info
            self.daily_trades += 1
            
            self.logger.info(f"🚀 PAPER TRADE EXECUTED:")
            self.logger.info(f"   📋 Order ID: {order.id}")
            self.logger.info(f"   📊 Symbol: {symbol}")
            self.logger.info(f"   📈 Contracts: {contracts}")
            self.logger.info(f"   💰 Entry Price: ${entry_price:.3f}")
            self.logger.info(f"   🎯 Strike: ${option_info['strike']}")
            self.logger.info(f"   💵 Gross Premium: ${trade_info['gross_premium']:.2f}")
            self.logger.info(f"   💸 Expected Costs: ${expected_costs['total_costs']:.2f}")
            
            return trade_info
            
        except Exception as e:
            self.logger.error(f"❌ Error executing paper trade: {e}")
            return None
    
    def calculate_realistic_costs(self, gross_premium: float, contracts: int) -> Dict:
        """
        💰 EXACT SAME COST CALCULATION AS BACKTEST
        
        Uses identical commission, slippage, and spread costs
        """
        # Same costs as backtest
        commission = self.params['commission_per_contract'] * contracts
        bid_ask_cost = gross_premium * self.params['bid_ask_spread_pct']
        slippage_cost = gross_premium * self.params['slippage_pct']
        
        # Market impact for larger positions (25K account)
        market_impact = 0.0
        if contracts >= 10:
            market_impact = gross_premium * 0.002
        
        total_costs = commission + bid_ask_cost + slippage_cost + market_impact
        
        return {
            'commission': commission,
            'bid_ask_cost': bid_ask_cost,
            'slippage_cost': slippage_cost,
            'market_impact': market_impact,
            'total_costs': total_costs
        }
    
    async def monitor_positions(self):
        """📊 Monitor open positions and manage exits"""
        try:
            if not self.trading_client:
                return
            
            # Get current positions
            positions = self.trading_client.get_all_positions()
            
            for position in positions:
                symbol = position.symbol
                qty = float(position.qty)
                current_price = float(position.market_value) / abs(qty) if qty != 0 else 0
                
                # Check if this is one of our tracked trades
                order_id = None
                for oid, trade_info in self.orders.items():
                    if trade_info['symbol'] == symbol and trade_info['status'] == 'filled':
                        order_id = oid
                        break
                
                if order_id:
                    self.logger.info(f"📊 Monitoring Position: {symbol}")
                    self.logger.info(f"   Qty: {qty}")
                    self.logger.info(f"   Current Price: ${current_price:.3f}")
                    
                    # Apply risk management (same as backtest)
                    await self.check_exit_conditions(order_id, current_price)
            
        except Exception as e:
            self.logger.error(f"❌ Error monitoring positions: {e}")
    
    async def check_exit_conditions(self, order_id: str, current_price: float):
        """🛡️ Risk management exits (same logic as backtest)"""
        try:
            trade_info = self.orders[order_id]
            entry_price = trade_info['entry_price']
            contracts = trade_info['contracts']
            
            # Calculate current P&L
            price_change = current_price - entry_price
            gross_pnl = -price_change * 100 * contracts  # We sold puts
            
            # Apply costs
            costs = trade_info['expected_costs']
            net_pnl = gross_pnl - costs['total_costs']
            
            # Check stop loss (same as backtest)
            max_loss = self.params['max_loss_per_trade']
            if net_pnl < -max_loss:
                await self.close_position(order_id, "stop_loss")
                return
            
            # Check profit target (same as backtest)
            net_premium = trade_info['gross_premium'] - costs['total_costs']
            profit_target = net_premium * (self.params['profit_target_pct'] / 100)
            
            if net_pnl > profit_target:
                await self.close_position(order_id, "profit_target")
                return
            
            # Update P&L tracking
            trade_info['current_pnl'] = net_pnl
            
        except Exception as e:
            self.logger.error(f"❌ Error checking exit conditions: {e}")
    
    async def close_position(self, order_id: str, reason: str):
        """Close position and record final P&L"""
        try:
            trade_info = self.orders[order_id]
            symbol = trade_info['symbol']
            contracts = trade_info['contracts']
            
            # Create closing order
            close_request = MarketOrderRequest(
                symbol=symbol,
                qty=contracts,
                side=OrderSide.BUY,  # Buy to close short put
                time_in_force=TimeInForce.DAY,
                client_order_id=f"close_{order_id[:8]}"
            )
            
            close_order = self.trading_client.submit_order(close_request)
            
            # Update trade tracking
            trade_info['close_order_id'] = close_order.id
            trade_info['close_reason'] = reason
            trade_info['close_time'] = datetime.now()
            trade_info['status'] = 'closed'
            
            # Update daily P&L
            final_pnl = trade_info.get('current_pnl', 0)
            self.daily_pnl += final_pnl
            
            self.logger.info(f"🔒 POSITION CLOSED:")
            self.logger.info(f"   📋 Trade ID: {order_id[:8]}")
            self.logger.info(f"   📊 Symbol: {symbol}")
            self.logger.info(f"   💰 Final P&L: ${final_pnl:.2f}")
            self.logger.info(f"   📝 Reason: {reason}")
            self.logger.info(f"   📈 Daily P&L: ${self.daily_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error closing position: {e}")
    
    async def run_trading_session(self):
        """
        🚀 MAIN TRADING LOOP
        
        Runs live paper trading session following exact backtest logic
        """
        self.logger.info("🚀 Starting Phase 4D Paper Trading Session")
        self.logger.info(f"💼 Account Size: {self.account_size}")
        self.logger.info(f"📊 Max Contracts: {self.params['max_contracts']}")
        
        self.trading_active = True
        
        try:
            while self.trading_active:
                current_time = datetime.now().time()
                
                # Trading hours check
                market_open = time(9, 30)  # 9:30 AM
                market_close = time(16, 0)  # 4:00 PM
                
                if current_time < market_open or current_time > market_close:
                    self.logger.info("📅 Outside market hours - waiting...")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue
                
                # Get live market data
                spy_price = self.get_live_spy_price()
                if not spy_price:
                    await asyncio.sleep(60)  # Wait 1 minute and retry
                    continue
                
                # Calculate market conditions (same as backtest)
                market_conditions = self.calculate_live_market_conditions()
                
                # Check if we should trade (same filters as backtest)
                should_trade, reason = self.should_trade_today(market_conditions)
                
                if not should_trade:
                    self.logger.info(f"🚫 No trade today: {reason}")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue
                
                # Get live option chain
                option_chain = self.get_live_option_chain(datetime.now().strftime("%Y%m%d"))
                if not option_chain:
                    await asyncio.sleep(60)
                    continue
                
                # Select strike (same logic as backtest)
                selected_option = self.live_strike_selection(spy_price, option_chain)
                if not selected_option:
                    await asyncio.sleep(60)
                    continue
                
                # Execute trade
                trade_result = await self.execute_paper_trade(selected_option)
                if trade_result:
                    self.logger.info("✅ Trade executed successfully")
                
                # Monitor positions
                await self.monitor_positions()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Trading session interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Trading session error: {e}")
        finally:
            self.trading_active = False
            await self.close_all_positions()
    
    async def close_all_positions(self):
        """Close all open positions at end of session"""
        try:
            if not self.trading_client:
                return
            
            positions = self.trading_client.get_all_positions()
            
            for position in positions:
                if float(position.qty) != 0:
                    self.logger.info(f"🔒 Closing position: {position.symbol}")
                    
                    close_request = MarketOrderRequest(
                        symbol=position.symbol,
                        qty=abs(float(position.qty)),
                        side=OrderSide.BUY if float(position.qty) < 0 else OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    self.trading_client.submit_order(close_request)
            
            self.logger.info("✅ All positions closed")
            
        except Exception as e:
            self.logger.error(f"❌ Error closing positions: {e}")
    
    def generate_daily_report(self):
        """Generate daily performance report vs backtest expectations"""
        try:
            expected_pnl = self.backtest_comparison['expected_daily_pnl']
            actual_pnl = self.daily_pnl
            variance = ((actual_pnl - expected_pnl) / expected_pnl * 100) if expected_pnl != 0 else 0
            
            # Update tracking
            self.backtest_comparison['actual_daily_pnl'].append(actual_pnl)
            self.backtest_comparison['variance_from_backtest'].append(variance)
            
            self.logger.info("\n" + "="*60)
            self.logger.info("📊 DAILY PERFORMANCE REPORT")
            self.logger.info("="*60)
            self.logger.info(f"💰 Expected P&L (backtest): ${expected_pnl:.2f}")
            self.logger.info(f"💰 Actual P&L (live): ${actual_pnl:.2f}")
            self.logger.info(f"📊 Variance: {variance:.1f}%")
            self.logger.info(f"📈 Trades Executed: {self.daily_trades}")
            self.logger.info(f"🎯 Account Size: {self.account_size}")
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"phase4d_paper_trading_report_{timestamp}.json"
            
            report_data = {
                'date': datetime.now().isoformat(),
                'account_size': self.account_size,
                'expected_pnl': expected_pnl,
                'actual_pnl': actual_pnl,
                'variance_pct': variance,
                'trades_executed': self.daily_trades,
                'orders': {k: {**v, 'entry_time': v['entry_time'].isoformat() if 'entry_time' in v else None} 
                          for k, v in self.orders.items()}
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating report: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Shutdown signal received. Closing positions...")
    # Set global flag to stop trading
    import sys
    sys.exit(0)

async def main():
    """Main entry point for paper trading"""
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Phase 4D Paper Trading Strategy")
    print("Choose account size:")
    print("1. 2K Account (2 contracts)")
    print("2. 25K Account (25 contracts)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        account_size = "25k"
        print("💰 Configured for 25K account (25 contracts)")
    else:
        account_size = "2k"
        print("📊 Configured for 2K account (2 contracts)")
    
    # Initialize strategy
    strategy = Phase4DPaperTrading(account_size=account_size)
    
    if not strategy.trading_client:
        print("❌ Failed to initialize trading client. Check API credentials.")
        return
    
    print("✅ Strategy initialized. Starting trading session...")
    print("🛑 Press Ctrl+C to stop trading and generate final report")
    
    try:
        # Run trading session
        await strategy.run_trading_session()
    finally:
        # Generate final report
        strategy.generate_daily_report()
        print("📊 Final report generated. Session complete.")

if __name__ == "__main__":
    if not ALPACA_AVAILABLE:
        print("❌ Alpaca SDK not available. Please install: pip install alpaca-py")
    else:
        asyncio.run(main())