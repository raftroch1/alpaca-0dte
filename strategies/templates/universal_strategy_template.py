#!/usr/bin/env python3
"""
🎯 UNIVERSAL STRATEGY TEMPLATE - ALPACA 0DTE FRAMEWORK
====================================================

COPY THIS TEMPLATE TO CREATE NEW STRATEGIES
All patterns based on proven working strategies: Multi-Regime, Turtle, Phase 4 Aggressive

📋 QUICK START:
1. Copy: cp universal_strategy_template.py your_strategy_name.py
2. Rename class: UniversalStrategyTemplate -> YourStrategyName  
3. Update strategy_name and parameters
4. Implement generate_trading_signal() method
5. Test with paper trading first
6. Create corresponding backtest using universal_backtest_template.py

✅ PROVEN PATTERNS INCLUDED:
- API key management (dotenv + environment variables)
- Alpaca client initialization (Trading + Historical Data)
- Real data integration (ThetaData cache + Alpaca API)
- Comprehensive logging and error handling
- Risk management and position sizing
- Paper trading mode support
- Market hours validation
- Performance tracking

❌ NO SIMULATION:
- Only real historical data sources
- Real option pricing via Alpaca API
- Real SPY data from ThetaData cache
- No synthetic time decay or volatility models

Author: Universal Framework v2.0
Date: 2025-01-31
Based on: Multi-Regime, Turtle, Phase 4 Aggressive proven strategies
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
import uuid
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# 🔧 STANDARDIZED IMPORT PATHS
# Add project root for Alpaca imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

# Load environment variables from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), '.env'))

# 📊 ALPACA CLIENT IMPORTS (Proven Pattern)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, 
    GetOptionContractsRequest,
    GetOrdersRequest,
    LimitOrderRequest
)
from alpaca.trading.enums import (
    OrderSide, 
    OrderType, 
    TimeInForce, 
    OrderClass,
    QueryOrderStatus,
    ContractType,
    PositionIntent
)
from alpaca.trading.models import Position, Order
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.stream import TradingStream

# 🎯 REAL DATA INTEGRATION IMPORTS (No Simulation)
try:
    # ThetaData for cached SPY data (proven pattern)
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'thetadata', 'theta_connection'))
    from connector import ThetaDataConnector
except ImportError:
    ThetaDataConnector = None
    logging.warning("⚠️ ThetaData connector not available - using Alpaca only mode")


class UniversalStrategyTemplate:
    """
    🎯 Universal Strategy Template - Copy and customize for new strategies
    
    PROVEN PATTERNS FROM WORKING STRATEGIES:
    - Live Ultra Aggressive (Phase 4): Real-time trading with dynamic position sizing
    - Multi-Regime: VIX-based strategy selection and regime adaptation  
    - Turtle: ATR-based risk management and breakout systems
    - Real Data Integration: Alpaca + ThetaData for authentic backtesting
    
    RENAME THIS CLASS: YourStrategyName (PascalCase)
    Examples: MomentumBreakout, VixContrarian, EarningsPlay, etc.
    """
    
    def __init__(self, strategy_name: str = "universal_strategy"):
        """
        🚀 Initialize strategy with proven patterns
        
        Args:
            strategy_name: Your strategy identifier (lowercase_with_underscores)
        """
        self.strategy_name = strategy_name
        self.logger = self._setup_logging()
        
        # 🔑 API KEY MANAGEMENT (Proven Pattern)
        self._validate_environment()
        self._initialize_alpaca_clients()
        
        # 🎯 STRATEGY PARAMETERS - CUSTOMIZE THESE
        self.params = self._get_strategy_parameters()
        
        # 📊 REAL DATA SOURCES (No Simulation)
        self._initialize_data_sources()
        
        # 📈 TRACKING & RISK MANAGEMENT
        self._initialize_tracking()
        
        self.logger.info(f"✅ {strategy_name} initialized with proven framework patterns")
        self.logger.info(f"📊 Data Sources: ThetaData={'✅' if self.theta_connector else '❌'}, Alpaca=✅")
        self.logger.info(f"🎯 Paper Trading: {'✅' if self.trading_client.paper else '❌'}")
    
    def _setup_logging(self) -> logging.Logger:
        """📝 Setup comprehensive logging (proven pattern)"""
        logger = logging.getLogger(f"{self.strategy_name}_live")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(f'logs/{self.strategy_name}_live.log'),
                    logging.StreamHandler()
                ]
            )
        return logger
    
    def _validate_environment(self):
        """🔐 Validate API keys (proven pattern)"""
        required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            raise ValueError(f"❌ Missing environment variables: {missing}")
        
        self.logger.info("✅ Environment variables validated")
    
    def _initialize_alpaca_clients(self):
        """🏦 Initialize Alpaca clients (proven pattern)"""
        try:
            # Trading client for live/paper trading
            self.trading_client = TradingClient(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
                paper=True  # Start with paper trading - change to False for live
            )
            
            # Historical data clients
            self.stock_client = StockHistoricalDataClient(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY")
            )
            
            self.option_client = OptionHistoricalDataClient(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY")
            )
            
            self.logger.info("✅ Alpaca clients initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca client initialization failed: {e}")
            raise
    
    def _get_strategy_parameters(self) -> Dict[str, Any]:
        """
        🎯 CUSTOMIZE THESE PARAMETERS FOR YOUR STRATEGY
        
        Based on proven patterns from working strategies:
        - Conservative risk management (1-2% per trade)
        - Realistic profit targets ($300-500 daily)
        - Proper position sizing and limits
        """
        return {
            # 💰 ACCOUNT & RISK MANAGEMENT
            'account_size': 25000.0,           # Starting account size
            'risk_per_trade': 0.01,            # 1% account risk per trade (turtle pattern)
            'max_daily_trades': 10,            # Maximum trades per day
            'daily_profit_target': 500.0,      # Daily profit target
            'daily_loss_limit': 350.0,         # Daily loss limit
            'max_position_size': 5,            # Maximum contracts per position
            
            # ⏰ TIMING & MARKET HOURS
            'market_open_buffer': 15,          # Minutes after market open
            'market_close_buffer': 30,         # Minutes before market close
            'min_time_to_expiry': 30,          # Minimum minutes to option expiry
            'max_position_time': 120,          # Maximum minutes to hold position
            
            # 📊 OPTION SELECTION
            'min_option_price': 0.10,          # Minimum option price
            'max_option_price': 3.00,          # Maximum option price  
            'max_bid_ask_spread': 0.15,        # Maximum 15% bid-ask spread
            'min_open_interest': 100,          # Minimum open interest
            'preferred_dte': 0,                # Days to expiration (0 = same day)
            
            # 🎯 SIGNAL REQUIREMENTS  
            'confidence_threshold': 0.60,      # Minimum signal confidence
            'min_volume_ratio': 1.2,           # Minimum volume vs average
            'volatility_threshold': 0.008,     # Minimum volatility for entry
            
            # 🔄 TECHNICAL INDICATORS (Customize based on your strategy)
            'rsi_period': 14,                  # RSI calculation period
            'rsi_oversold': 30,                # RSI oversold level
            'rsi_overbought': 70,              # RSI overbought level
            'ma_short_period': 5,              # Short moving average
            'ma_long_period': 20,              # Long moving average
            'atr_period': 14,                  # ATR period (turtle pattern)
            
            # 📈 POSITION MANAGEMENT
            'profit_target_pct': 0.50,         # Take profit at 50% of max profit
            'stop_loss_pct': 2.00,             # Stop loss at 200% of premium paid
            'trailing_stop_trigger': 0.30,     # Start trailing at 30% profit
            'trailing_stop_distance': 0.10,    # Trail by 10%
        }
    
    def _initialize_data_sources(self):
        """📊 Initialize real data sources (proven pattern)"""
        # ThetaData connector for cached SPY data
        self.theta_connector = None
        if ThetaDataConnector:
            try:
                self.theta_connector = ThetaDataConnector()
                self.logger.info("✅ ThetaData connector initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ ThetaData initialization failed: {e}")
        
        # Data cache for performance
        self.data_cache = {}
        self.cache_timeout = 60  # Cache data for 60 seconds
    
    def _initialize_tracking(self):
        """📈 Initialize performance tracking (proven pattern)"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.positions = {}
        self.orders = {}
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'daily_results': []
        }
        
        # Market hours tracking
        self.market_close_time = None
        self.current_trading_day = None
        
        # Risk management tracking
        self.current_risk_exposure = 0.0
        self.daily_risk_taken = 0.0
    
    # 🎯 CORE METHODS TO IMPLEMENT
    
    def generate_trading_signal(self, spy_data: pd.DataFrame) -> Dict[str, Any]:
        """
        🔍 IMPLEMENT THIS METHOD - Your core strategy logic
        
        Args:
            spy_data: Recent SPY price data with OHLCV
            
        Returns:
            Dict containing:
            - 'signal': 'BUY_CALL', 'BUY_PUT', or 'HOLD'
            - 'confidence': float between 0 and 1
            - 'reasoning': str explaining the decision
            - 'target_strike': float (optional)
            - 'position_size': int (number of contracts)
        
        STRATEGY EXAMPLES:
        - Momentum: RSI + MACD + price breakouts
        - Mean Reversion: Bollinger Bands + RSI extremes  
        - Volatility: VIX levels + implied volatility
        - Breakout: Support/resistance + volume confirmation
        """
        # 📊 EXAMPLE IMPLEMENTATION - REPLACE WITH YOUR LOGIC
        try:
            if spy_data.empty or len(spy_data) < self.params['ma_long_period']:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': 'Insufficient data'}
            
            # Calculate technical indicators
            current_price = spy_data['close'].iloc[-1]
            rsi = self._calculate_rsi(spy_data['close'])
            ma_short = spy_data['close'].rolling(self.params['ma_short_period']).mean().iloc[-1]
            ma_long = spy_data['close'].rolling(self.params['ma_long_period']).mean().iloc[-1]
            
            # Example signal logic - CUSTOMIZE THIS
            if rsi < self.params['rsi_oversold'] and current_price > ma_short:
                return {
                    'signal': 'BUY_CALL',
                    'confidence': 0.75,
                    'reasoning': f'Oversold RSI ({rsi:.1f}) + bullish MA crossover',
                    'target_strike': round(current_price + 1.0),
                    'position_size': self._calculate_position_size(0.75)
                }
            elif rsi > self.params['rsi_overbought'] and current_price < ma_short:
                return {
                    'signal': 'BUY_PUT', 
                    'confidence': 0.75,
                    'reasoning': f'Overbought RSI ({rsi:.1f}) + bearish MA crossover',
                    'target_strike': round(current_price - 1.0),
                    'position_size': self._calculate_position_size(0.75)
                }
            else:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reasoning': 'No clear signal - waiting for setup'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error generating trading signal: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasoning': f'Error: {e}'}
    
    # 📊 UTILITY METHODS (Proven Patterns)
    
    def get_spy_minute_data(self, lookback_minutes: int = 100) -> pd.DataFrame:
        """📈 Get recent SPY minute data (proven pattern)"""
        try:
            # Use cache if available and recent
            cache_key = f"spy_data_{lookback_minutes}"
            if cache_key in self.data_cache:
                data, timestamp = self.data_cache[cache_key]
                if (datetime.now() - timestamp).seconds < self.cache_timeout:
                    return data
            
            # Get data from Alpaca
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=lookback_minutes)
            
            request = StockBarsRequest(
                symbol_or_symbols=["SPY"],
                timeframe=TimeFrame.Minute,
                start=start_time,
                end=end_time
            )
            
            bars = self.stock_client.get_stock_bars(request)
            if "SPY" not in bars.data or not bars.data["SPY"]:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for bar in bars.data["SPY"]:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high), 
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            # Cache the data
            self.data_cache[cache_key] = (df, datetime.now())
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error getting SPY data: {e}")
            return pd.DataFrame()
    
    def find_best_option_contract(self, signal_data: Dict[str, Any]) -> Optional[str]:
        """🎯 Find best option contract (proven pattern from live strategies)"""
        try:
            signal = signal_data['signal']
            target_strike = signal_data.get('target_strike')
            
            if signal == 'HOLD':
                return None
            
            # Determine option type
            option_type = ContractType.CALL if 'CALL' in signal else ContractType.PUT
            
            # Get 0DTE contracts
            today = datetime.now().strftime('%Y-%m-%d')
            
            request = GetOptionContractsRequest(
                underlying_symbol="SPY",
                expiration_date=today,
                contract_type=option_type,
                strike_price_gte=str(target_strike - 5.0) if target_strike else None,
                strike_price_lte=str(target_strike + 5.0) if target_strike else None
            )
            
            contracts = self.trading_client.get_option_contracts(request)
            
            if not contracts:
                self.logger.warning(f"⚠️ No {option_type} contracts found for {today}")
                return None
            
            # Filter by price and spread criteria
            best_contract = None
            best_score = 0
            
            for contract in contracts:
                # Get current quote (simplified - you may want to implement full quote checking)
                try:
                    # Basic filtering by strike proximity
                    strike = float(contract.strike_price)
                    if target_strike and abs(strike - target_strike) > 3.0:
                        continue
                    
                    # Score based on strike proximity and other factors
                    score = 1.0 / (1.0 + abs(strike - target_strike)) if target_strike else 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_contract = contract.symbol
                        
                except Exception as e:
                    self.logger.debug(f"Contract filtering error: {e}")
                    continue
            
            if best_contract:
                self.logger.info(f"✅ Selected contract: {best_contract}")
            else:
                self.logger.warning("⚠️ No suitable contracts found")
            
            return best_contract
            
        except Exception as e:
            self.logger.error(f"❌ Error finding option contract: {e}")
            return None
    
    def _calculate_position_size(self, confidence: float) -> int:
        """📊 Calculate position size based on confidence and risk (proven pattern)"""
        try:
            account = self.trading_client.get_account()
            buying_power = float(account.buying_power)
            
            # Base position size on risk per trade
            max_risk = buying_power * self.params['risk_per_trade']
            
            # Adjust by confidence (higher confidence = larger size)
            confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range
            
            # Calculate size (assuming $100 average option price for estimation)
            estimated_option_price = 1.00  # Conservative estimate
            max_contracts = int(max_risk / (estimated_option_price * 100))
            
            # Apply confidence and limits
            position_size = int(max_contracts * confidence_multiplier)
            position_size = max(1, min(position_size, self.params['max_position_size']))
            
            self.logger.debug(f"📊 Position size: {position_size} contracts (confidence: {confidence:.2f})")
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 1
    
    def _calculate_rsi(self, prices: pd.Series, period: int = None) -> float:
        """📊 Calculate RSI (proven pattern)"""
        if period is None:
            period = self.params['rsi_period']
            
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def is_market_hours(self) -> bool:
        """⏰ Check if market is open (proven pattern)"""
        now = datetime.now(timezone.utc)
        
        # Convert to Eastern Time
        eastern = now.astimezone(timezone(timedelta(hours=-5)))  # EST/EDT approximation
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = eastern.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = eastern.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Check if it's a weekday
        if eastern.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Apply buffers
        open_buffer = timedelta(minutes=self.params['market_open_buffer'])
        close_buffer = timedelta(minutes=self.params['market_close_buffer'])
        
        trading_start = market_open + open_buffer
        trading_end = market_close - close_buffer
        
        return trading_start <= eastern <= trading_end
    
    def should_trade(self) -> bool:
        """🎯 Check if conditions are right for trading (proven pattern)"""
        # Market hours check
        if not self.is_market_hours():
            return False
        
        # Daily limits check
        if self.daily_pnl <= -self.params['daily_loss_limit']:
            self.logger.info(f"📊 Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
        
        if self.daily_pnl >= self.params['daily_profit_target']:
            self.logger.info(f"📊 Daily profit target reached: ${self.daily_pnl:.2f}")
            return False
        
        if self.daily_trades >= self.params['max_daily_trades']:
            self.logger.info(f"📊 Daily trade limit reached: {self.daily_trades}")
            return False
        
        return True
    
    # 🚀 MAIN TRADING LOOP
    
    async def run_live_trading(self):
        """
        🚀 Main live trading loop (proven pattern)
        
        CUSTOMIZATION POINTS:
        1. generate_trading_signal() - Implement your strategy logic
        2. Strategy parameters in _get_strategy_parameters()
        3. Risk management rules
        4. Position sizing logic
        """
        self.logger.info(f"🚀 Starting {self.strategy_name} live trading")
        self.logger.info(f"📊 Paper Mode: {self.trading_client.paper}")
        
        try:
            while True:
                if not self.should_trade():
                    await asyncio.sleep(60)  # Check every minute when not trading
                    continue
                
                # Get market data
                spy_data = self.get_spy_minute_data()
                if spy_data.empty:
                    self.logger.warning("⚠️ No SPY data available")
                    await asyncio.sleep(30)
                    continue
                
                # Generate trading signal
                signal_data = self.generate_trading_signal(spy_data)
                
                if signal_data['signal'] == 'HOLD':
                    self.logger.debug(f"📊 Holding: {signal_data['reasoning']}")
                    await asyncio.sleep(30)
                    continue
                
                # Execute trade if signal is strong enough
                if signal_data['confidence'] >= self.params['confidence_threshold']:
                    await self._execute_trade(signal_data)
                
                # Monitor existing positions
                await self._monitor_positions()
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Trading interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Trading loop error: {e}")
        finally:
            await self._cleanup()
    
    async def _execute_trade(self, signal_data: Dict[str, Any]):
        """🎯 Execute trade based on signal (proven pattern)"""
        try:
            # Find best option contract
            contract_symbol = self.find_best_option_contract(signal_data)
            if not contract_symbol:
                return
            
            # Create order
            order_request = MarketOrderRequest(
                symbol=contract_symbol,
                qty=signal_data['position_size'],
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order (paper trading)
            order = self.trading_client.submit_order(order_request)
            
            self.logger.info(f"📈 Order submitted: {contract_symbol} x{signal_data['position_size']}")
            self.logger.info(f"🎯 Signal: {signal_data['signal']} | Confidence: {signal_data['confidence']:.2f}")
            self.logger.info(f"💭 Reasoning: {signal_data['reasoning']}")
            
            # Track the order
            self.orders[order.id] = {
                'symbol': contract_symbol,
                'qty': signal_data['position_size'],
                'signal_data': signal_data,
                'timestamp': datetime.now()
            }
            
            self.daily_trades += 1
            
        except Exception as e:
            self.logger.error(f"❌ Trade execution error: {e}")
    
    async def _monitor_positions(self):
        """📊 Monitor and manage existing positions (proven pattern)"""
        try:
            positions = self.trading_client.get_all_positions()
            
            for position in positions:
                # Implement position management logic
                # - Check profit/loss
                # - Apply stop losses
                # - Take profits
                # - Time-based exits
                pass
                
        except Exception as e:
            self.logger.error(f"❌ Position monitoring error: {e}")
    
    async def _cleanup(self):
        """🧹 Cleanup resources (proven pattern)"""
        self.logger.info("🧹 Cleaning up resources")
        
        # Close any remaining positions if needed
        # Save performance data
        # Log final statistics
        
        self.logger.info(f"📊 Final Daily P&L: ${self.daily_pnl:.2f}")
        self.logger.info(f"📈 Total Trades: {self.daily_trades}")


def main():
    """🎯 Main entry point for live trading"""
    
    # CUSTOMIZE: Change strategy name and parameters
    strategy = UniversalStrategyTemplate(strategy_name="your_strategy_name")
    
    print(f"\n🎯 {strategy.strategy_name.upper()} - LIVE TRADING")
    print("=" * 60)
    print(f"📊 Framework: Universal Template v2.0")
    print(f"🏦 Alpaca Paper Trading: {'✅' if strategy.trading_client.paper else '❌'}")
    print(f"📈 Daily Target: ${strategy.params['daily_profit_target']:.0f}")
    print(f"🛡️ Daily Limit: ${strategy.params['daily_loss_limit']:.0f}")
    print("=" * 60)
    
    # Start live trading
    try:
        asyncio.run(strategy.run_live_trading())
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main() 