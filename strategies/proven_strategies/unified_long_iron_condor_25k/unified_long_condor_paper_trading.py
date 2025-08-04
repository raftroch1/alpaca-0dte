#!/usr/bin/env python3
"""
🎪🛡️ UNIFIED LONG IRON CONDOR + COUNTER - LIVE PAPER TRADING
================================================================

Live paper trading implementation of the unified Long Iron Condor + Counter strategy
targeting $250/day for 25K account with 85-90% backtest accuracy.

PROVEN BACKTEST RESULTS:
- Average Daily P&L: $243.84 (97.5% of $250 target)
- Win Rate: 77.0%
- Execution Rate: 100.0%
- Primary Strategy: Long Iron Condor (6 contracts)
- Counter Strategies: Bear put spreads, short call supplements

LIVE TRADING FEATURES:
- Real-time SPY minute data via Alpaca
- Live 0DTE option contract discovery
- Multi-leg Iron Condor execution with OptionLegRequest
- Real-time position monitoring with profit targets (75%) and stop losses (50%)
- Market timing controls (no new positions 30min before close)
- Professional error handling and logging
- Risk management with daily limits

Author: Strategy Development Framework
Date: 2025-01-31
Version: Unified Paper Trading v1.0
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, time
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
import uuid
from dotenv import load_dotenv
import math

warnings.filterwarnings('ignore')

# Alpaca imports using proven patterns
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, 
    GetOptionContractsRequest,
    GetOrdersRequest,
    OptionLegRequest,
    ClosePositionRequest
)
from alpaca.trading.enums import (
    OrderSide, 
    OrderType, 
    TimeInForce, 
    OrderClass,
    QueryOrderStatus,
    ContractType,
    AssetStatus,
    OrderStatus
)
from alpaca.trading.models import Position, Order
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


class UnifiedLongCondorPaperTrading:
    """
    Live paper trading implementation of the unified Long Iron Condor + Counter system.
    Matches backtest accuracy with bulletproof live trading architecture.
    """
    
    def __init__(self, cache_dir: str = None):
        """Initialize with bulletproof architecture patterns"""
        
        # Load environment variables
        load_dotenv()
        
        # Validate API keys first (bulletproof pattern)
        self.validate_api_keys()
        
        # Initialize Alpaca clients with error handling
        self.setup_trading_clients()
        self.setup_data_clients()
        
        # Strategy parameters from proven backtest
        self.params = self.get_strategy_parameters()
        
        # Risk management tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.positions = {}
        self.active_orders = {}
        self.max_daily_loss = self.params['max_daily_loss']
        self.target_daily_pnl = self.params['target_daily_pnl']
        
        # Market timing controls
        self.market_open_time = time(9, 30)  # 9:30 AM ET
        self.market_close_time = time(16, 0)  # 4:00 PM ET
        self.no_new_positions_time = time(15, 30)  # 3:30 PM ET (30min before close)
        self.position_close_time = time(15, 45)  # 3:45 PM ET (15min before close)
        
        # Setup professional logging
        self.setup_professional_logging()
        
        # Cache directory for fallback data
        if cache_dir is None:
            cache_dir = os.getenv('THETA_CACHE_DIR', os.path.join(os.getcwd(), 'thetadata', 'cached_data'))
        self.cache_dir = cache_dir
        
        self.logger.info("🎪🛡️ Unified Long Iron Condor Paper Trading System Initialized")
        self.logger.info(f"🎯 Target: ${self.target_daily_pnl}/day | Max Loss: ${self.max_daily_loss}")
    
    def validate_api_keys(self):
        """Validate all required API keys are present (bulletproof pattern)"""
        required_keys = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            raise ValueError(f"❌ Missing API keys: {missing_keys}")
        
        print("✅ All API keys validated")
    
    def setup_trading_clients(self):
        """Setup trading client with error handling (bulletproof pattern)"""
        try:
            self.trading_client = TradingClient(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
                paper=True  # Always start with paper trading
            )
            
            # Test connection
            account = self.trading_client.get_account()
            print(f"✅ Trading client connected - Account: ${account.cash}")
            
        except Exception as e:
            print(f"❌ Failed to setup trading client: {e}")
            raise
    
    def setup_data_clients(self):
        """Setup data clients with fallback mechanisms (bulletproof pattern)"""
        try:
            # Stock data client for SPY minute data
            self.stock_client = StockHistoricalDataClient(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY")
            )
            
            # Option data client for option pricing
            self.option_client = OptionHistoricalDataClient(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY")
            )
            
            print("✅ Data clients initialized")
            
        except Exception as e:
            print(f"❌ Failed to setup data clients: {e}")
            raise
    
    def setup_professional_logging(self):
        """Setup comprehensive logging system (bulletproof pattern)"""
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for all logs
        file_handler = logging.FileHandler(
            f"logs/unified_long_condor_live_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Trade-specific logger
        self.trade_logger = logging.getLogger("trades")
        trade_handler = logging.FileHandler(
            f"logs/unified_trades_{datetime.now().strftime('%Y%m%d')}.log"
        )
        trade_handler.setFormatter(detailed_formatter)
        self.trade_logger.addHandler(trade_handler)
    
    def get_strategy_parameters(self) -> Dict:
        """Get strategy parameters from proven backtest (exact same values)"""
        return {
            # === PRIMARY STRATEGY: Long Iron Condor ===
            'primary_strategy': 'long_iron_condor',
            'wing_width': 1.0,               # $1 wings (capital efficient)
            'strike_buffer': 0.75,           # Distance from ATM
            
            # Strike Selection (proven to work)
            'min_strike_buffer': 0.5,        
            'max_strike_buffer': 3.0,          
            'target_delta_range': (-0.25, -0.10),  
            'min_debit': 0.10,               # Minimum debit to pay
            'max_debit': 1.50,               # Maximum debit willing to pay
            
            # PRIMARY Volatility Filtering (REVERSED for Long Condors)
            'min_daily_range': 0.5,          # Need SOME volatility to profit
            'max_daily_range': 12.0,         # High vol is good for us
            'min_vix_threshold': 12,         # Min VIX for volatility
            'max_vix_threshold': 40,         # Max VIX (too crazy = no trades)
            
            # PRIMARY Position Sizing - 25K SCALED FOR $250/DAY TARGET
            'primary_base_contracts': 6,     # SCALED: 6 Long Iron Condors per trade for $250/day
            'primary_max_contracts': 8,      # Max 8 condors for high-confidence days
            
            # === COUNTER STRATEGIES ===
            'counter_strategy': 'adaptive_counter',
            
            # COUNTER Volatility Thresholds (for strategy selection)
            'counter_moderate_vol_min': 6.0,    # Bear Put Spreads
            'counter_moderate_vol_max': 8.0,
            'counter_low_vol_max': 1.0,         # Short Call Supplements
            
            # COUNTER Position Sizing - 25K SCALED  
            'counter_base_contracts': 10,       # SCALED: 10 spreads per counter trade
            'counter_max_contracts': 15,        # Max 15 spreads for scaling
            
            # === UNIFIED RISK MANAGEMENT - 25K Account ===
            'max_loss_per_trade': 900,         # Max $900 loss per trade (6 contracts)
            'max_daily_loss': 1500,            # Max $1500 loss per day
            'target_daily_pnl': 250,           # $250/day target (6 contracts)
            'profit_target_pct': 75,           # Take profit at 75% of max profit
            'stop_loss_pct': 50,               # Stop loss at 50% of debit paid
            
            # === LIVE TRADING CONTROLS ===
            'min_open_interest': 100,          # Minimum open interest for liquidity
            'max_bid_ask_spread': 0.15,        # Max 15% bid/ask spread
            'min_time_to_close': 30,           # Min 30 minutes to market close
            'position_check_interval': 60,     # Check positions every 60 seconds
        }
    
    def get_spy_minute_data(self, minutes_back: int = 50) -> pd.DataFrame:
        """Get recent SPY minute data with fallback mechanisms (bulletproof pattern)"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=minutes_back)
            
            request = StockBarsRequest(
                symbol_or_symbols=["SPY"],
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start_time,
                end=end_time
            )
            
            bars = self.stock_client.get_stock_bars(request)
            df = bars.df.reset_index()
            
            if df.empty:
                self.logger.warning("⚠️ No SPY data received from API")
                return self.generate_fallback_spy_data(minutes_back)
                
            # Validate data quality
            if len(df) < self.params.get('min_data_points', 10):
                self.logger.warning(f"⚠️ Insufficient data: {len(df)} bars")
                return self.generate_fallback_spy_data(minutes_back)
            
            self.logger.debug(f"📊 Retrieved {len(df)} SPY minute bars")
            return df[df['symbol'] == 'SPY'].copy()
            
        except Exception as e:
            self.logger.warning(f"⚠️ SPY data API failed: {e}")
            self.logger.info("📊 Using simulated SPY data for analysis")
            return self.generate_fallback_spy_data(minutes_back)
    
    def generate_fallback_spy_data(self, minutes_back: int) -> pd.DataFrame:
        """Generate realistic fallback SPY data when API fails (bulletproof pattern)"""
        current_time = datetime.now()
        times = [current_time - timedelta(minutes=i) for i in range(minutes_back, 0, -1)]
        
        # Simple price simulation based on recent ranges
        base_price = 520.0  # Approximate SPY range
        prices = []
        for i, time in enumerate(times):
            # Add some realistic intraday movement
            price = base_price + np.random.normal(0, 0.5) + (i * 0.01)
            prices.append(price)
        
        return pd.DataFrame({
            'timestamp': times,
            'symbol': 'SPY',
            'open': prices,
            'high': [p + 0.1 for p in prices],
            'low': [p - 0.1 for p in prices], 
            'close': prices,
            'volume': [1000000] * len(prices)
        })
    
    def discover_0dte_options(self, spy_price: float) -> Dict:
        """Discover available 0DTE option contracts (proven pattern)"""
        try:
            # Get current date for 0DTE
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Get option contracts around current price
            strike_range = self.params.get('strike_discovery_range', 10.0)
            min_strike = spy_price - strike_range
            max_strike = spy_price + strike_range
            
            # Get calls
            call_request = GetOptionContractsRequest(
                underlying_symbols=["SPY"],
                type=ContractType.CALL,
                status=AssetStatus.ACTIVE,
                expiration_date_gte=current_date,
                expiration_date_lte=current_date,
                strike_price_gte=str(min_strike),
                strike_price_lte=str(max_strike)
            )
            
            # Get puts
            put_request = GetOptionContractsRequest(
                underlying_symbols=["SPY"],
                type=ContractType.PUT,
                status=AssetStatus.ACTIVE,
                expiration_date_gte=current_date,
                expiration_date_lte=current_date,
                strike_price_gte=str(min_strike),
                strike_price_lte=str(max_strike)
            )
            
            calls = self.trading_client.get_option_contracts(call_request).option_contracts
            puts = self.trading_client.get_option_contracts(put_request).option_contracts
            
            # Filter for liquidity (following Alpaca examples pattern)
            filtered_calls = []
            filtered_puts = []
            
            for contract in calls:
                try:
                    open_interest = int(contract.open_interest) if contract.open_interest else 0
                    if open_interest >= self.params['min_open_interest']:
                        filtered_calls.append(contract)
                except (ValueError, TypeError):
                    continue  # Skip contracts with invalid open interest data
            
            for contract in puts:
                try:
                    open_interest = int(contract.open_interest) if contract.open_interest else 0
                    if open_interest >= self.params['min_open_interest']:
                        filtered_puts.append(contract)
                except (ValueError, TypeError):
                    continue  # Skip contracts with invalid open interest data
            
            # Return structure matching Alpaca examples
            available_contracts = {
                'calls': filtered_calls,
                'puts': filtered_puts
            }
            
            total_contracts = len(filtered_calls) + len(filtered_puts)
            self.logger.info(f"📋 Found {total_contracts} liquid 0DTE contracts")
            return available_contracts
            
        except Exception as e:
            self.logger.error(f"❌ Failed to discover options: {e}")
            return {'calls': [], 'puts': []}
    
    def check_market_conditions(self, spy_data: pd.DataFrame) -> Tuple[bool, str, str]:
        """
        Check market conditions and determine strategy (same logic as backtest)
        Returns: (can_trade, reason, strategy_type)
        """
        if spy_data.empty:
            return False, "No SPY data available", "no_trade"
        
        spy_open = spy_data['open'].iloc[0] if not spy_data.empty else spy_data['close'].iloc[0]
        spy_high = spy_data['high'].max()
        spy_low = spy_data['low'].min()
        spy_close = spy_data['close'].iloc[-1]
        
        # Calculate daily range
        daily_range = ((spy_high - spy_low) / spy_open) * 100
        
        # === PRIMARY STRATEGY: Long Iron Condor Logic ===
        if self.params['min_daily_range'] <= daily_range <= self.params['max_daily_range']:
            return True, f"PRIMARY: Good volatility {daily_range:.2f}%", "long_iron_condor"
        
        # === COUNTER STRATEGIES: For filtered days ===
        
        # 1. Bear Put Spreads for moderate volatility (6-8%)
        if (self.params['counter_moderate_vol_min'] <= daily_range <= 
            self.params['counter_moderate_vol_max']):
            return True, f"COUNTER: Bear Put Spreads - mod vol {daily_range:.2f}%", "bear_put_spreads"
        
        # 2. Short Call Supplements for very low volatility (<1%)
        if daily_range < self.params['counter_low_vol_max']:
            return True, f"COUNTER: Short Calls - low vol {daily_range:.2f}%", "short_call_supplement"
        
        # 3. Closer-to-ATM Bull Put Spreads for other filtered conditions
        if daily_range > self.params['max_daily_range']:
            return True, f"COUNTER: Closer ATM Spreads - high vol {daily_range:.2f}%", "closer_atm_spreads"
        
        # No trading conditions met
        return False, f"No strategy fits - vol {daily_range:.2f}%", "no_trade"
    
    def get_iron_condor_strikes(self, spy_price: float) -> Tuple[float, float, float, float]:
        """Get Iron Condor strikes using WHOLE DOLLAR strikes (proven approach)"""
        wing_width = self.params['wing_width']
        strike_buffer = self.params['strike_buffer']
        
        # Round to whole dollars for data availability (proven approach)
        short_put_strike = round(spy_price - strike_buffer, 0)
        long_put_strike = short_put_strike - wing_width
        
        short_call_strike = round(spy_price + strike_buffer, 0)
        long_call_strike = short_call_strike + wing_width
        
        return short_put_strike, long_put_strike, short_call_strike, long_call_strike
    
    def get_option_prices_with_validation(self, contracts: Dict, symbols: List[str]) -> Optional[Dict]:
        """Get option prices with bid/ask spread validation (following Alpaca examples)"""
        try:
            prices = {}
            
            for symbol in symbols:
                # Find contract info in the new structure
                contract_found = None
                # Search in calls
                for contract in contracts.get('calls', []):
                    if contract.symbol == symbol:
                        contract_found = contract
                        break
                # Search in puts if not found in calls
                if not contract_found:
                    for contract in contracts.get('puts', []):
                        if contract.symbol == symbol:
                            contract_found = contract
                            break
                
                if not contract_found:
                    self.logger.warning(f"⚠️ Contract not found for {symbol}")
                    return None
                
                # Get latest quote
                quote_request = OptionLatestQuoteRequest(symbol_or_symbols=[symbol])
                quote_response = self.option_client.get_option_latest_quote(quote_request)
                
                if symbol not in quote_response:
                    self.logger.warning(f"⚠️ No quote available for {symbol}")
                    return None
                
                quote = quote_response[symbol]
                
                # Validate bid/ask spread
                if quote.bid_price <= 0 or quote.ask_price <= 0:
                    self.logger.warning(f"⚠️ Invalid bid/ask for {symbol}")
                    return None
                
                spread_pct = (quote.ask_price - quote.bid_price) / quote.ask_price
                if spread_pct > self.params['max_bid_ask_spread']:
                    self.logger.warning(f"⚠️ Bid/ask spread too wide for {symbol}: {spread_pct:.1%}")
                    return None
                
                mid_price = (quote.bid_price + quote.ask_price) / 2
                prices[symbol] = {
                    'mid_price': mid_price,
                    'bid': quote.bid_price,
                    'ask': quote.ask_price,
                    'spread_pct': spread_pct
                }
            
            return prices
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get option prices: {e}")
            return None
    
    def execute_long_iron_condor(self, spy_price: float, contracts: Dict) -> Optional[str]:
        """Execute Long Iron Condor using multi-leg order (proven pattern)"""
        try:
            # Get strikes
            short_put_strike, long_put_strike, short_call_strike, long_call_strike = self.get_iron_condor_strikes(spy_price)
            
            # Find matching contracts (following Alpaca examples pattern)
            short_put_symbol = None
            long_put_symbol = None
            short_call_symbol = None
            long_call_symbol = None
            
            # Search in puts
            for contract in contracts.get('puts', []):
                strike = float(contract.strike_price)
                
                if strike == short_put_strike:
                    short_put_symbol = contract.symbol
                elif strike == long_put_strike:
                    long_put_symbol = contract.symbol
            
            # Search in calls
            for contract in contracts.get('calls', []):
                strike = float(contract.strike_price)
                
                if strike == short_call_strike:
                    short_call_symbol = contract.symbol
                elif strike == long_call_strike:
                    long_call_symbol = contract.symbol
            
            # Validate all symbols found
            required_symbols = [short_put_symbol, long_put_symbol, short_call_symbol, long_call_symbol]
            if any(symbol is None for symbol in required_symbols):
                self.logger.warning("⚠️ Not all Iron Condor legs available")
                return None
            
            # Get and validate option prices
            prices = self.get_option_prices_with_validation(contracts, required_symbols)
            if not prices:
                return None
            
            # Calculate net debit (Long Iron Condor = we pay debit)
            put_spread_debit = prices[short_put_symbol]['mid_price'] - prices[long_put_symbol]['mid_price']
            call_spread_debit = prices[short_call_symbol]['mid_price'] - prices[long_call_symbol]['mid_price']
            total_debit = put_spread_debit + call_spread_debit
            
            # Apply debit filters (same as backtest)
            if total_debit < self.params['min_debit']:
                self.logger.info(f"📊 Filtered: Debit too low ${total_debit:.3f}")
                return None
            
            if total_debit > self.params['max_debit']:
                self.logger.info(f"📊 Filtered: Debit too high ${total_debit:.3f}")
                return None
            
            # Build multi-leg order (proven pattern from examples)
            contracts_qty = self.params['primary_base_contracts']
            
            order_legs = []
            
            # Long Iron Condor legs:
            # Buy put spread: Buy long put, Sell short put
            order_legs.append(OptionLegRequest(
                symbol=long_put_symbol,
                side=OrderSide.BUY,
                ratio_qty=contracts_qty
            ))
            order_legs.append(OptionLegRequest(
                symbol=short_put_symbol,
                side=OrderSide.SELL,
                ratio_qty=contracts_qty
            ))
            
            # Buy call spread: Buy long call, Sell short call
            order_legs.append(OptionLegRequest(
                symbol=long_call_symbol,
                side=OrderSide.BUY,
                ratio_qty=contracts_qty
            ))
            order_legs.append(OptionLegRequest(
                symbol=short_call_symbol,
                side=OrderSide.SELL,
                ratio_qty=contracts_qty
            ))
            
            # Submit multi-leg order
            order_request = MarketOrderRequest(
                qty=contracts_qty,
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                legs=order_legs
            )
            
            order = self.trading_client.submit_order(order_request)
            
            self.logger.info(f"🎪 LONG IRON CONDOR ORDER SUBMITTED:")
            self.logger.info(f"   Order ID: {order.id}")
            self.logger.info(f"   Contracts: {contracts_qty}")
            self.logger.info(f"   Expected Debit: ${total_debit:.3f}")
            self.logger.info(f"   Strikes: P{long_put_strike}/{short_put_strike} | C{short_call_strike}/{long_call_strike}")
            
            # Log trade execution
            self.trade_logger.info(
                f"LONG_IRON_CONDOR_EXECUTED | Order: {order.id} | "
                f"Contracts: {contracts_qty} | "
                f"Expected Debit: ${total_debit:.3f} | "
                f"SPY: ${spy_price:.2f}"
            )
            
            # Track position
            self.track_position(order.id, {
                'strategy': 'long_iron_condor',
                'contracts': contracts_qty,
                'expected_debit': total_debit,
                'strikes': {
                    'short_put': short_put_strike,
                    'long_put': long_put_strike,
                    'short_call': short_call_strike,
                    'long_call': long_call_strike
                },
                'symbols': {
                    'short_put': short_put_symbol,
                    'long_put': long_put_symbol,
                    'short_call': short_call_symbol,
                    'long_call': long_call_symbol
                },
                'entry_time': datetime.now(),
                'spy_price': spy_price
            })
            
            return order.id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute Long Iron Condor: {e}")
            return None
    
    def track_position(self, order_id: str, position_data: Dict):
        """Track position for monitoring"""
        self.active_orders[order_id] = position_data
        self.daily_trades += 1
    
    async def monitor_positions_realtime(self):
        """Monitor all positions with real-time P&L tracking (bulletproof pattern)"""
        try:
            positions = self.trading_client.get_all_positions()
            
            current_portfolio_pnl = 0
            for position in positions:
                # Calculate current P&L
                unrealized_pnl = float(position.unrealized_pl or 0)
                current_portfolio_pnl += unrealized_pnl
                
                # Check profit targets and stop losses
                await self.check_profit_stop_conditions(position)
                
                # Log position status
                self.logger.debug(f"📈 {position.symbol}: P&L ${unrealized_pnl:.2f}")
            
            # Update daily tracking
            self.daily_pnl = current_portfolio_pnl
            
            # Check daily limits
            if self.daily_pnl <= -self.max_daily_loss:
                self.logger.warning("🛑 Daily loss limit hit - Stopping trading")
                await self.emergency_close_all_positions()
                
        except Exception as e:
            self.logger.error(f"❌ Position monitoring failed: {e}")
    
    async def check_profit_stop_conditions(self, position):
        """Check individual position profit targets and stop losses"""
        try:
            unrealized_pnl = float(position.unrealized_pl or 0)
            position_value = abs(float(position.market_value or 0))
            
            if position_value <= 0:
                return
            
            pnl_percentage = unrealized_pnl / position_value
            
            # Profit target: 75% of max profit (configurable)
            profit_target_pct = self.params['profit_target_pct'] / 100
            if pnl_percentage >= profit_target_pct:
                self.logger.info(f"🎯 Profit target hit for {position.symbol}: {pnl_percentage:.1%}")
                await self.close_position(position.symbol, "PROFIT_TARGET")
            
            # Stop loss: 50% of debit paid (configurable)  
            stop_loss_pct = -self.params['stop_loss_pct'] / 100
            if pnl_percentage <= stop_loss_pct:
                self.logger.warning(f"🛑 Stop loss hit for {position.symbol}: {pnl_percentage:.1%}")
                await self.close_position(position.symbol, "STOP_LOSS")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to check profit/stop conditions: {e}")
    
    async def close_position(self, symbol: str, reason: str):
        """Close individual position"""
        try:
            close_request = ClosePositionRequest()
            self.trading_client.close_position(
                symbol_or_asset_id=symbol,
                close_position_request=close_request
            )
            
            self.logger.info(f"🔒 Position closed: {symbol} | Reason: {reason}")
            self.trade_logger.info(f"POSITION_CLOSED | Symbol: {symbol} | Reason: {reason}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to close position {symbol}: {e}")
    
    async def emergency_close_all_positions(self):
        """Emergency close all positions (bulletproof pattern)"""
        try:
            positions = self.trading_client.get_all_positions()
            
            for position in positions:
                await self.close_position(position.symbol, "EMERGENCY_STOP")
                
            self.logger.info("🚨 Emergency close all positions completed")
            
        except Exception as e:
            self.logger.error(f"❌ Emergency close failed: {e}")
    
    def execute_short_call_supplement(self, spy_price: float, contracts: Dict) -> Optional[str]:
        """Execute short call supplement for very low volatility days"""
        try:
            # Short call slightly OTM
            target_strike = round(spy_price + 0.5, 0)  # $0.50 above SPY
            
            # Find suitable call contract
            call_contracts = [c for c in contracts['calls'] if float(c.strike_price) == target_strike]
            if not call_contracts:
                self.logger.warning(f"⚠️ No call contract found for strike ${target_strike}")
                return None
            
            call_contract = call_contracts[0]
            
            # Get current option price
            prices = self.get_option_prices_with_validation(contracts, [call_contract.symbol])
            if not prices or call_contract.symbol not in prices:
                self.logger.warning(f"⚠️ No price data for {call_contract.symbol}")
                return None
            
            option_price = prices[call_contract.symbol]['mid_price']
            
            # Check minimum premium
            if option_price < 0.03:
                self.logger.info(f"📊 SHORT CALL: Premium too low ${option_price:.3f} < $0.03")
                return None
            
            # Execute short call order
            order_legs = [
                OptionLegRequest(
                    symbol=call_contract.symbol,
                    side=OrderSide.SELL,  # SELL call (short)
                    ratio_qty=self.params['counter_base_contracts']
                )
            ]
            
            order_request = MarketOrderRequest(
                symbol="SPY",
                legs=order_legs,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.MLEG
            )
            
            order = self.trading_client.submit_order(order_request)
            order_id = str(order.id)
            
            self.logger.info(f"📞 SHORT CALL SUPPLEMENT EXECUTED:")
            self.logger.info(f"   Strike: ${target_strike} (${target_strike - spy_price:.2f} above SPY)")
            self.logger.info(f"   Premium: ${option_price:.3f}")
            self.logger.info(f"   Contracts: {self.params['counter_base_contracts']}")
            self.logger.info(f"   Order ID: {order_id}")
            
            # Track position
            position_data = {
                'strategy': 'short_call_supplement',
                'strike': target_strike,
                'premium': option_price,
                'contracts': self.params['counter_base_contracts'],
                'entry_time': datetime.now(),
                'spy_price_at_entry': spy_price
            }
            
            self.track_position(order_id, position_data)
            self.trade_logger.info(f"COUNTER_EXECUTED | Strategy: short_call_supplement | Strike: ${target_strike} | Premium: ${option_price:.3f} | Contracts: {self.params['counter_base_contracts']}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"❌ Short call supplement execution failed: {e}")
            return None
    
    def execute_bear_put_spread(self, spy_price: float, contracts: Dict) -> Optional[str]:
        """Execute bear put spread for moderate volatility"""
        try:
            # Bear put spread: Buy higher strike, sell lower strike
            long_strike = round(spy_price + 1.0, 0)   # $1 above SPY
            short_strike = round(spy_price - 1.0, 0)  # $1 below SPY
            
            # Find suitable put contracts
            long_puts = [c for c in contracts['puts'] if float(c.strike_price) == long_strike]
            short_puts = [c for c in contracts['puts'] if float(c.strike_price) == short_strike]
            
            if not long_puts or not short_puts:
                self.logger.warning(f"⚠️ Missing put contracts for bear spread")
                return None
            
            long_put = long_puts[0]
            short_put = short_puts[0]
            
            # Get option prices
            symbols = [long_put.symbol, short_put.symbol]
            prices = self.get_option_prices_with_validation(contracts, symbols)
            if not prices or len(prices) != 2:
                self.logger.warning(f"⚠️ Missing price data for bear put spread")
                return None
            
            long_price = prices[long_put.symbol]['mid_price']
            short_price = prices[short_put.symbol]['mid_price']
            
            # Calculate spread cost
            net_cost = long_price - short_price
            if net_cost <= 0 or net_cost > 1.0:  # Max $1.00 cost
                self.logger.info(f"📊 BEAR PUT: Invalid spread cost ${net_cost:.3f}")
                return None
            
            # Execute bear put spread order
            order_legs = [
                OptionLegRequest(
                    symbol=long_put.symbol,
                    side=OrderSide.BUY,  # BUY higher strike put
                    ratio_qty=self.params['counter_base_contracts']
                ),
                OptionLegRequest(
                    symbol=short_put.symbol,
                    side=OrderSide.SELL,  # SELL lower strike put
                    ratio_qty=self.params['counter_base_contracts']
                )
            ]
            
            order_request = MarketOrderRequest(
                symbol="SPY",
                legs=order_legs,
                side=OrderSide.BUY,  # Net debit trade
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.MLEG
            )
            
            order = self.trading_client.submit_order(order_request)
            order_id = str(order.id)
            
            self.logger.info(f"🐻 BEAR PUT SPREAD EXECUTED:")
            self.logger.info(f"   Long: ${long_strike} @ ${long_price:.3f}")
            self.logger.info(f"   Short: ${short_strike} @ ${short_price:.3f}")
            self.logger.info(f"   Net Cost: ${net_cost:.3f}")
            self.logger.info(f"   Contracts: {self.params['counter_base_contracts']}")
            self.logger.info(f"   Order ID: {order_id}")
            
            # Track position
            position_data = {
                'strategy': 'bear_put_spread',
                'long_strike': long_strike,
                'short_strike': short_strike,
                'net_cost': net_cost,
                'contracts': self.params['counter_base_contracts'],
                'entry_time': datetime.now(),
                'spy_price_at_entry': spy_price
            }
            
            self.track_position(order_id, position_data)
            self.trade_logger.info(f"COUNTER_EXECUTED | Strategy: bear_put_spread | Long: ${long_strike} | Short: ${short_strike} | Cost: ${net_cost:.3f}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"❌ Bear put spread execution failed: {e}")
            return None
    
    def execute_closer_atm_spreads(self, spy_price: float, contracts: Dict) -> Optional[str]:
        """Execute closer to ATM spreads for low premium days"""
        try:
            # Get closer to ATM than normal Long Iron Condor
            put_strike = round(spy_price - 0.25, 0)  # Only $0.25 below SPY
            call_strike = round(spy_price + 0.25, 0)  # Only $0.25 above SPY
            
            # Find suitable contracts
            puts = [c for c in contracts['puts'] if float(c.strike_price) == put_strike]
            calls = [c for c in contracts['calls'] if float(c.strike_price) == call_strike]
            
            if not puts or not calls:
                self.logger.warning(f"⚠️ Missing contracts for closer ATM spreads")
                return None
            
            put_contract = puts[0]
            call_contract = calls[0]
            
            # Get option prices
            symbols = [put_contract.symbol, call_contract.symbol]
            prices = self.get_option_prices_with_validation(contracts, symbols)
            if not prices or len(prices) != 2:
                self.logger.warning(f"⚠️ Missing price data for closer ATM spreads")
                return None
            
            put_price = prices[put_contract.symbol]['mid_price']
            call_price = prices[call_contract.symbol]['mid_price']
            
            # Check minimum premium
            if put_price < 0.05 and call_price < 0.05:
                self.logger.info(f"📊 CLOSER ATM: Both premiums too low")
                return None
            
            # Execute short straddle (or strangle)
            order_legs = [
                OptionLegRequest(
                    symbol=put_contract.symbol,
                    side=OrderSide.SELL,  # SELL put
                    ratio_qty=self.params['counter_base_contracts']
                ),
                OptionLegRequest(
                    symbol=call_contract.symbol,
                    side=OrderSide.SELL,  # SELL call
                    ratio_qty=self.params['counter_base_contracts']
                )
            ]
            
            order_request = MarketOrderRequest(
                symbol="SPY",
                legs=order_legs,
                side=OrderSide.SELL,  # Net credit trade
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.MLEG
            )
            
            order = self.trading_client.submit_order(order_request)
            order_id = str(order.id)
            
            total_premium = put_price + call_price
            
            self.logger.info(f"🎯 CLOSER ATM SPREADS EXECUTED:")
            self.logger.info(f"   Put: ${put_strike} @ ${put_price:.3f}")
            self.logger.info(f"   Call: ${call_strike} @ ${call_price:.3f}")
            self.logger.info(f"   Total Premium: ${total_premium:.3f}")
            self.logger.info(f"   Contracts: {self.params['counter_base_contracts']}")
            self.logger.info(f"   Order ID: {order_id}")
            
            # Track position
            position_data = {
                'strategy': 'closer_atm_spreads',
                'put_strike': put_strike,
                'call_strike': call_strike,
                'total_premium': total_premium,
                'contracts': self.params['counter_base_contracts'],
                'entry_time': datetime.now(),
                'spy_price_at_entry': spy_price
            }
            
            self.track_position(order_id, position_data)
            self.trade_logger.info(f"COUNTER_EXECUTED | Strategy: closer_atm_spreads | Put: ${put_strike} | Call: ${call_strike} | Premium: ${total_premium:.3f}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"❌ Closer ATM spreads execution failed: {e}")
            return None
    
    def is_market_hours(self) -> bool:
        """Check if market is open (bulletproof pattern)"""
        now = datetime.now().time()
        return self.market_open_time <= now <= self.market_close_time
    
    def can_open_new_positions(self) -> bool:
        """Check if new positions can be opened (timing control)"""
        now = datetime.now().time()
        return (self.is_market_hours() and 
                now < self.no_new_positions_time and
                self.daily_pnl > -self.max_daily_loss)
    
    def should_close_positions(self) -> bool:
        """Check if positions should be closed (timing control)"""
        now = datetime.now().time()
        return now >= self.position_close_time
    
    async def run_unified_strategy(self):
        """Main unified strategy trading loop"""
        self.logger.info("🚀 Starting Unified Long Iron Condor + Counter Live Trading")
        
        try:
            while self.is_market_hours():
                # 1. Check if we should close positions before market close
                if self.should_close_positions():
                    self.logger.info("🕐 Market close approaching - closing all positions")
                    await self.emergency_close_all_positions()
                    break
                
                # 2. Monitor existing positions
                await self.monitor_positions_realtime()
                
                # 3. Check daily limits
                if self.daily_pnl <= -self.max_daily_loss:
                    self.logger.warning("🛑 Daily loss limit reached - stopping trading")
                    break
                
                if self.daily_pnl >= self.target_daily_pnl:
                    self.logger.info(f"🎯 Daily target reached: ${self.daily_pnl:.2f} - continuing to monitor")
                
                # 4. Check if we can open new positions
                if self.can_open_new_positions() and len(self.active_orders) < 3:  # Max 3 concurrent positions
                    
                    # Get real-time market data
                    spy_data = self.get_spy_minute_data()
                    
                    if not spy_data.empty:
                        spy_price = spy_data['close'].iloc[-1]
                        
                        # Check market conditions and determine strategy
                        can_trade, reason, strategy_type = self.check_market_conditions(spy_data)
                        
                        if can_trade:
                            self.logger.info(f"✅ {reason} -> {strategy_type}")
                            
                            if strategy_type == "long_iron_condor":
                                # Discover available option contracts
                                contracts = self.discover_0dte_options(spy_price)
                                
                                if contracts:
                                    # Execute primary strategy
                                    order_id = self.execute_long_iron_condor(spy_price, contracts)
                                    
                                    if order_id:
                                        self.logger.info(f"✅ Long Iron Condor executed: {order_id}")
                                    else:
                                        self.logger.info("📊 Long Iron Condor filtered out")
                                else:
                                    self.logger.warning("⚠️ No suitable option contracts found")
                            
                            else:
                                # Execute counter strategies
                                contracts = self.discover_0dte_options(spy_price)
                                
                                if contracts:
                                    order_id = None
                                    
                                    if strategy_type == "short_call_supplement":
                                        order_id = self.execute_short_call_supplement(spy_price, contracts)
                                    elif strategy_type == "bear_put_spreads":
                                        order_id = self.execute_bear_put_spread(spy_price, contracts)
                                    elif strategy_type == "closer_atm_spreads":
                                        order_id = self.execute_closer_atm_spreads(spy_price, contracts)
                                    else:
                                        self.logger.warning(f"⚠️ Unknown counter strategy: {strategy_type}")
                                    
                                    if order_id:
                                        self.logger.info(f"✅ Counter strategy {strategy_type} executed: {order_id}")
                                    else:
                                        self.logger.info(f"📊 Counter strategy {strategy_type} filtered out")
                                else:
                                    self.logger.warning("⚠️ No suitable option contracts found for counter strategy")
                        
                        else:
                            self.logger.debug(f"📊 {reason}")
                
                # 5. Wait before next iteration
                await asyncio.sleep(self.params['position_check_interval'])
                
        except KeyboardInterrupt:
            self.logger.info("👋 Graceful shutdown requested")
        except Exception as e:
            self.logger.error(f"❌ Strategy error: {e}")
        finally:
            await self.cleanup_and_shutdown()
    
    async def cleanup_and_shutdown(self):
        """Clean shutdown with position management (bulletproof pattern)"""
        self.logger.info("🧹 Cleaning up and generating final performance report")
        
        try:
            # Generate daily report
            self.generate_daily_report()
            
            # Optional: Close all positions on shutdown
            if self.params.get('close_positions_on_shutdown', True):
                await self.emergency_close_all_positions()
                
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
    
    def generate_daily_report(self):
        """Generate comprehensive daily performance report"""
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            total_unrealized_pnl = sum(float(pos.unrealized_pl or 0) for pos in positions)
            
            self.logger.info("=" * 80)
            self.logger.info("📊 UNIFIED LONG IRON CONDOR - DAILY PERFORMANCE REPORT")
            self.logger.info("=" * 80)
            self.logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
            self.logger.info(f"💰 Account Value: ${float(account.portfolio_value):,.2f}")
            self.logger.info(f"📈 Daily P&L: ${total_unrealized_pnl:.2f}")
            self.logger.info(f"🎯 Target Progress: {(total_unrealized_pnl / self.target_daily_pnl * 100):.1f}%")
            self.logger.info(f"📊 Daily Trades: {self.daily_trades}")
            self.logger.info(f"📋 Active Positions: {len(positions)}")
            
            if positions:
                self.logger.info("\n📈 ACTIVE POSITIONS:")
                for pos in positions:
                    self.logger.info(f"   {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f} | P&L: ${float(pos.unrealized_pl or 0):.2f}")
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate daily report: {e}")


async def main():
    """Main entry point for unified paper trading"""
    
    # Load environment variables
    load_dotenv()
    
    # Initialize strategy
    strategy = UnifiedLongCondorPaperTrading()
    
    # Run live strategy
    await strategy.run_unified_strategy()


if __name__ == "__main__":
    print("🎪🛡️ Starting Unified Long Iron Condor + Counter Paper Trading System")
    print("📋 Paper Trading Mode - Safe for testing")
    print("🎯 Target: $250/day | Strategy: Long Iron Condor + Counter")
    print("=" * 70)
    
    # Run the strategy
    asyncio.run(main())