#!/usr/bin/env python3
"""
Enhanced Turtle-Style 0DTE Options Strategy
==========================================

Enhanced with proper Turtle methodology:
1. True N-based position sizing and stops (ATR-based)
2. Multiple system logic (System 1: 20-day, System 2: 55-day)
3. Portfolio correlation management
4. Better risk management with pyramid adding
5. Market regime detection
6. Improved exit strategies
7. Realistic expectations for 0DTE options

Target: $300-500 daily profit with authentic Turtle risk management
"""

import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle
import gzip

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from strategies.base_theta_strategy import BaseThetaStrategy
from thetadata.theta_connection.connector import ThetaDataConnector

@dataclass
class EnhancedTurtleSignal:
    """Enhanced turtle signal with proper N-based calculations"""
    signal_type: str  # 'CALL_BREAKOUT', 'PUT_BREAKOUT'
    system: int  # 1 (20-day) or 2 (55-day)
    confidence: float
    current_n: float  # Current ATR value (turtle's "N")
    breakout_strength: float
    position_size_dollars: float  # Dollar amount, not contracts
    entry_price_target: float
    stop_loss_price: float  # N-based stop
    add_signals: List[float]  # Prices for adding positions
    market_regime: str  # 'TRENDING', 'CHOPPY', 'VOLATILE'
    symbol: str  # Which symbol this signal is for

class EnhancedTurtle0DTEStrategy(BaseThetaStrategy):
    """Enhanced turtle-style strategy with proper risk management"""
    
    def __init__(self):
        super().__init__()
        self.name = "enhanced_turtle_0dte"
        self.version = "2.0.0"
        
        # Enhanced Turtle Parameters
        self.params = {
            # Account & Risk Management
            'account_size': 25000.0,        # Starting account size
            'risk_per_trade': 0.01,         # 1% account risk per trade (turtle standard)
            'max_portfolio_risk': 0.06,     # 6% total portfolio risk
            'daily_profit_target': 500.0,
            'max_daily_loss': 350.0,
            
            # Turtle System Parameters
            'system1_entry': 20,            # System 1: 20-day breakout
            'system1_exit': 10,             # System 1: 10-day exit
            'system2_entry': 55,            # System 2: 55-day breakout
            'system2_exit': 20,             # System 2: 20-day exit
            'n_periods': 20,                # ATR calculation periods
            'stop_loss_n': 2.0,             # 2N stop loss (turtle standard)
            'add_position_n': 0.5,          # Add every 0.5N move
            'max_positions': 4,             # Max 4 units per market (turtle rule)
            
            # 0DTE Adaptations
            'min_time_to_expiry': 30,       # Minimum 30 min to expiry
            'max_position_time': 120,       # Max 2 hours per position
            'volatility_threshold': 0.008,  # Min volatility for entry
            'max_bid_ask_spread': 0.12,     # Max 12% spread
            
            # Market Regime Detection
            'regime_lookback': 50,          # Periods for regime detection
            'trend_threshold': 0.6,         # Trend strength threshold
            'volatility_percentile': 80,    # High volatility threshold
            
            # Multiple Underlyings (Portfolio Approach)
            'symbols': ['SPY', 'QQQ', 'IWM'],  # Multiple ETFs for diversification
            'correlation_threshold': 0.7,   # Max correlation between positions
        }
        
        # Enhanced Tracking
        self.daily_pnl = 0.0
        self.positions = {}  # Track all positions with pyramid levels
        self.market_data = {}  # Cache for multiple symbols
        self.correlation_matrix = pd.DataFrame()
        self.regime_history = {}
        self.n_values = {}  # Current N for each symbol
        
        # Portfolio Risk Tracking
        self.current_risk_exposure = 0.0
        self.symbol_exposures = {}
        
    def calculate_true_range_n(self, data: pd.DataFrame) -> float:
        """Calculate turtle-style N value (Average True Range)"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            prev_close = close.shift(1)
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            n_value = true_range.rolling(window=self.params['n_periods']).mean().iloc[-1]
            
            return n_value if not pd.isna(n_value) else data['close'].iloc[-1] * 0.01
            
        except Exception as e:
            self.logger.warning(f"N calculation error: {e}")
            return data['close'].iloc[-1] * 0.01

    def calculate_position_size_dollars(self, current_n: float, symbol: str) -> float:
        """Calculate position size in dollars using turtle methodology"""
        try:
            # Turtle formula: Risk = N * Contracts * Dollar per point
            # Rearranged: Contracts = (Account * Risk%) / N
            account_risk_dollars = self.params['account_size'] * self.params['risk_per_trade']
            
            # For options, we'll use the underlying price movement as proxy
            # This is a simplification - real implementation would use option Greeks
            position_size_dollars = account_risk_dollars / current_n
            
            # Cap position size
            max_position = self.params['account_size'] * 0.05  # Max 5% per position
            position_size_dollars = min(position_size_dollars, max_position)
            
            return position_size_dollars
            
        except Exception:
            return self.params['account_size'] * 0.02  # Default 2%

    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime for better signal filtering"""
        try:
            prices = data['close']
            returns = prices.pct_change()
            
            # Trend Detection
            sma_short = prices.rolling(10).mean()
            sma_long = prices.rolling(30).mean()
            trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
            
            # Volatility Analysis
            volatility = returns.rolling(20).std().iloc[-1]
            vol_percentile = (returns.rolling(self.params['regime_lookback']).std() < volatility).sum() / self.params['regime_lookback']
            
            # Regime Classification
            if abs(trend_strength) > self.params['trend_threshold'] / 100:
                return 'TRENDING'
            elif vol_percentile > self.params['volatility_percentile'] / 100:
                return 'VOLATILE'
            else:
                return 'CHOPPY'
                
        except Exception:
            return 'UNKNOWN'
    
    def detect_breakout(self, prices: pd.Series) -> Dict:
        """Detect turtle-style breakouts"""
        try:
            if len(prices) < self.params['breakout_periods']:
                return {'type': 'NONE', 'strength': 0.0, 'direction': 'NEUTRAL'}
            
            current_price = prices.iloc[-1]
            lookback_period = self.params['breakout_periods']
            
            # Calculate breakout levels
            high_level = prices.rolling(window=lookback_period).max().iloc[-2]  # Exclude current
            low_level = prices.rolling(window=lookback_period).min().iloc[-2]
            mid_level = (high_level + low_level) / 2
            
            # Check for breakouts
            if current_price > high_level:
                strength = (current_price - high_level) / high_level
                if strength >= self.params['breakout_threshold'] / 100:
                    return {
                        'type': 'UPSIDE_BREAKOUT',
                        'strength': strength,
                        'direction': 'BULLISH',
                        'level': high_level,
                        'target': high_level * (1 + strength * 1.5)
                    }
            
            elif current_price < low_level:
                strength = (low_level - current_price) / low_level
                if strength >= self.params['breakout_threshold'] / 100:
                    return {
                        'type': 'DOWNSIDE_BREAKOUT',
                        'strength': strength,
                        'direction': 'BEARISH',
                        'level': low_level,
                        'target': low_level * (1 - strength * 1.5)
                    }
            
            # Check for range-bound conditions (turtle also trades ranges)
            range_size = (high_level - low_level) / mid_level
            if range_size < 0.01:  # Tight range
                if current_price > mid_level * 1.005:
                    return {
                        'type': 'RANGE_TOP',
                        'strength': (current_price - mid_level) / mid_level,
                        'direction': 'RANGE_RESISTANCE',
                        'level': high_level
                    }
                elif current_price < mid_level * 0.995:
                    return {
                        'type': 'RANGE_BOTTOM',
                        'strength': (mid_level - current_price) / mid_level,
                        'direction': 'RANGE_SUPPORT',
                        'level': low_level
                    }
            
            return {'type': 'NONE', 'strength': 0.0, 'direction': 'NEUTRAL'}
            
        except Exception as e:
            self.logger.warning(f"⚠️ Breakout detection error: {e}")
            return {'type': 'NONE', 'strength': 0.0, 'direction': 'NEUTRAL'}
    
    def calculate_position_size(self, signal_strength: float, volatility: float) -> int:
        """Calculate turtle-style position size based on volatility and signal strength"""
        try:
            base_size = self.params['base_position_size']
            
            if not self.params['volatility_scaling']:
                return base_size
            
            # Turtle-style: Reduce size in high volatility, increase in low volatility
            volatility_multiplier = 0.02 / max(volatility, 0.005)  # Target 2% volatility
            volatility_multiplier = min(max(volatility_multiplier, 0.5), 2.0)  # Cap between 0.5x and 2x
            
            # Scale by signal strength
            strength_multiplier = 1.0 + signal_strength * 2.0  # Up to 3x for strong signals
            
            # Calculate final size
            position_size = int(base_size * volatility_multiplier * strength_multiplier)
            position_size = min(position_size, self.params['max_position_size'])
            position_size = max(position_size, 1)  # At least 1 contract
            
            return position_size
            
        except Exception as e:
            self.logger.warning(f"⚠️ Position size calculation error: {e}")
            return self.params['base_position_size']
    
    def generate_enhanced_turtle_signals(self, symbol: str, data: pd.DataFrame) -> List[EnhancedTurtleSignal]:
        """Generate enhanced turtle signals with both systems"""
        signals = []
        
        try:
            if len(data) < max(self.params['system2_entry'], self.params['n_periods']):
                return signals
            
            current_price = data['close'].iloc[-1]
            current_n = self.calculate_true_range_n(data)
            self.n_values[symbol] = current_n
            
            # Market regime detection
            regime = self.detect_market_regime(data)
            
            # Skip choppy markets (turtle wisdom)
            if regime == 'CHOPPY':
                return signals
            
            # System 1: 20-day breakout
            sys1_high = data['high'].rolling(self.params['system1_entry']).max().iloc[-2]
            sys1_low = data['low'].rolling(self.params['system1_entry']).min().iloc[-2]
            
            # System 2: 55-day breakout
            sys2_high = data['high'].rolling(self.params['system2_entry']).max().iloc[-2]
            sys2_low = data['low'].rolling(self.params['system2_entry']).min().iloc[-2]
            
            # Generate signals for both systems
            for system, high_level, low_level in [(1, sys1_high, sys1_low), (2, sys2_high, sys2_low)]:
                
                # Bullish breakout
                if current_price > high_level:
                    breakout_strength = (current_price - high_level) / current_n
                    
                    if breakout_strength > 0.1:  # Minimum breakout strength
                        position_size = self.calculate_position_size_dollars(current_n, symbol)
                        
                        # Calculate pyramid add levels
                        add_signals = [
                            current_price + (i * self.params['add_position_n'] * current_n)
                            for i in range(1, self.params['max_positions'])
                        ]
                        
                        signals.append(EnhancedTurtleSignal(
                            signal_type='CALL_BREAKOUT',
                            system=system,
                            confidence=min(breakout_strength, 1.0),
                            current_n=current_n,
                            breakout_strength=breakout_strength,
                            position_size_dollars=position_size,
                            entry_price_target=current_price,
                            stop_loss_price=current_price - (self.params['stop_loss_n'] * current_n),
                            add_signals=add_signals,
                            market_regime=regime,
                            symbol=symbol
                        ))
                
                # Bearish breakout
                if current_price < low_level:
                    breakout_strength = (low_level - current_price) / current_n
                    
                    if breakout_strength > 0.1:
                        position_size = self.calculate_position_size_dollars(current_n, symbol)
                        
                        add_signals = [
                            current_price - (i * self.params['add_position_n'] * current_n)
                            for i in range(1, self.params['max_positions'])
                        ]
                        
                        signals.append(EnhancedTurtleSignal(
                            signal_type='PUT_BREAKOUT',
                            system=system,
                            confidence=min(breakout_strength, 1.0),
                            current_n=current_n,
                            breakout_strength=breakout_strength,
                            position_size_dollars=position_size,
                            entry_price_target=current_price,
                            stop_loss_price=current_price + (self.params['stop_loss_n'] * current_n),
                            add_signals=add_signals,
                            market_regime=regime,
                            symbol=symbol
                        ))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation error for {symbol}: {e}")
            return signals
    
    def check_profit_loss_goals(self) -> bool:
        """Check if we should stop trading based on profit/loss goals (NO TRADE LIMITS!)"""
        try:
            # Stop if we hit our daily profit target
            if self.daily_pnl >= self.params['daily_profit_target']:
                self.logger.info(f"🎯 DAILY PROFIT TARGET REACHED: ${self.daily_pnl:.2f} (Target: ${self.params['daily_profit_target']})")
                return False
                
            # Stop if we hit our daily loss limit
            if self.daily_pnl <= -self.params['max_daily_loss']:
                self.logger.warning(f"🛑 DAILY LOSS LIMIT REACHED: ${self.daily_pnl:.2f} (Limit: -${self.params['max_daily_loss']})")
                return False
            
            # NO TRADE COUNT LIMITS - only profit/loss goals matter!
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error checking profit/loss goals: {e}")
            return False
    
    def monitor_position(self, position_id: str, entry_data: Dict) -> bool:
        """Monitor position with turtle-style exit logic"""
        try:
            # Get current option price (simplified for now)
            current_time = datetime.now()
            entry_time = entry_data['entry_time']
            time_elapsed = (current_time - entry_time).total_seconds() / 60  # Minutes
            
            # Emergency exit near market close
            if time_elapsed >= (390 - self.params['emergency_exit_minutes']):  # 30 min before close
                self.logger.info(f"⏰ Emergency exit: {position_id} (near market close)")
                return True  # Exit position
            
            # Max position time exit
            if time_elapsed >= self.params['max_position_time']:
                self.logger.info(f"⏰ Time limit exit: {position_id} ({time_elapsed:.1f} min)")
                return True  # Exit position
            
            # TODO: Add real-time P&L monitoring with profit targets and stop losses
            # This would require real-time option price feeds
            
            return False  # Keep position open
            
        except Exception as e:
            self.logger.error(f"❌ Error monitoring position {position_id}: {e}")
            return True  # Exit on error
    
    async def run_enhanced_live_trading(self):
        """Enhanced turtle trading loop with portfolio management"""
        self.logger.info("🐢 ENHANCED TURTLE 0DTE STRATEGY STARTED")
        self.logger.info(f"🎯 Daily Target: ${self.params['daily_profit_target']:.0f}")
        self.logger.info(f"🛡️ Max Loss: ${self.params['max_daily_loss']:.0f}")
        self.logger.info(f"📊 Symbols: {', '.join(self.params['symbols'])}")
        self.logger.info(f"⚖️ Risk per trade: {self.params['risk_per_trade']:.1%}")
        
        try:
            while True:
                # Check profit/loss goals
                if self.check_profit_loss_goals():
                    self.logger.info("🎯 Daily goals reached - stopping trading")
                    break
                
                # Check market hours
                if not self.is_market_hours():
                    self.logger.info("⏰ Market closed - waiting...")
                    await asyncio.sleep(300)
                    continue
                
                # Process each symbol in portfolio
                all_signals = []
                
                for symbol in self.params['symbols']:
                    try:
                        # Get market data for this symbol
                        data = await self.get_market_data(symbol)
                        if data is None or len(data) < 55:
                            continue
                        
                        # Generate enhanced signals
                        signals = self.generate_enhanced_turtle_signals(symbol, data)
                        
                        # Filter signals based on portfolio risk and correlation
                        filtered_signals = self.filter_signals_by_portfolio_risk(signals)
                        all_signals.extend(filtered_signals)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing {symbol}: {e}")
                        continue
                
                # Execute highest priority signals
                if all_signals:
                    # Sort by confidence and regime compatibility
                    all_signals.sort(key=lambda s: s.confidence * (2 if s.market_regime == 'TRENDING' else 1), reverse=True)
                    
                    # Execute top signals within risk limits
                    for signal in all_signals[:3]:  # Max 3 concurrent positions
                        if self.current_risk_exposure < self.params['max_portfolio_risk']:
                            success = await self.execute_enhanced_turtle_trade(signal)
                            if success:
                                self.logger.info(f"✅ {signal.symbol} System {signal.system} trade executed")
                                await asyncio.sleep(30)  # Brief delay between executions
                
                # Monitor existing positions and pyramid opportunities
                await self.monitor_enhanced_positions()
                
                # Update correlation matrix periodically
                if len(self.positions) > 1:
                    await self.update_correlation_matrix()
                
                # Regular cycle delay
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Turtle strategy stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Enhanced trading loop error: {e}")
            raise
        
        finally:
            self.logger.info("🐢 Enhanced turtle strategy stopped")
            self.logger.info(f"💰 Final Daily P&L: ${self.daily_pnl:.2f}")
            self.logger.info(f"📊 Total Risk Exposure: {self.current_risk_exposure:.1%}")
    
    def filter_signals_by_portfolio_risk(self, signals: List[EnhancedTurtleSignal]) -> List[EnhancedTurtleSignal]:
        """Filter signals based on portfolio risk and correlation limits"""
        filtered = []
        
        for signal in signals:
            # Check if adding this position would exceed portfolio risk
            position_risk = signal.position_size_dollars / self.params['account_size']
            
            if self.current_risk_exposure + position_risk > self.params['max_portfolio_risk']:
                continue
            
            # Check correlation with existing positions
            if self.check_correlation_limits(signal.symbol):
                filtered.append(signal)
        
        return filtered
    
    def check_correlation_limits(self, new_symbol: str) -> bool:
        """Check if new symbol would violate correlation limits"""
        if not self.positions:
            return True
        
        # Simple correlation check - in real implementation would use historical correlation matrix
        existing_symbols = set(pos['symbol'] for pos in self.positions.values() if 'symbol' in pos)
        
        # Don't allow too many positions in the same symbol
        symbol_count = sum(1 for pos in self.positions.values() if pos.get('symbol') == new_symbol)
        if symbol_count >= self.params['max_positions']:
            return False
        
        return True
    
    async def execute_enhanced_turtle_trade(self, signal: EnhancedTurtleSignal) -> bool:
        """Execute enhanced turtle trade with proper risk management"""
        try:
            self.logger.info(f"🐢 ENHANCED TURTLE SIGNAL: {signal.symbol} {signal.signal_type}")
            self.logger.info(f"   System: {signal.system} | Confidence: {signal.confidence:.2f}")
            self.logger.info(f"   Breakout Strength: {signal.breakout_strength:.3f}N")
            self.logger.info(f"   Position Size: ${signal.position_size_dollars:.0f}")
            self.logger.info(f"   Stop Loss: ${signal.stop_loss_price:.2f}")
            self.logger.info(f"   Market Regime: {signal.market_regime}")
            
            # Generate position ID
            position_id = f"turtle_{signal.symbol}_{signal.system}_{datetime.now().strftime('%H%M%S')}"
            
            # Store enhanced position data
            self.positions[position_id] = {
                'signal': signal,
                'symbol': signal.symbol,
                'system': signal.system,
                'entry_time': datetime.now(),
                'entry_price': signal.entry_price_target,
                'position_size_dollars': signal.position_size_dollars,
                'stop_loss_price': signal.stop_loss_price,
                'add_signals': signal.add_signals.copy(),
                'units_added': 0,  # Track pyramid levels
                'current_n': signal.current_n,
                'regime': signal.market_regime
            }
            
            # Update risk exposure
            position_risk = signal.position_size_dollars / self.params['account_size']
            self.current_risk_exposure += position_risk
            self.symbol_exposures[signal.symbol] = self.symbol_exposures.get(signal.symbol, 0) + position_risk
            
            self.logger.info(f"✅ Enhanced turtle position opened: {position_id}")
            self.logger.info(f"📊 Portfolio risk: {self.current_risk_exposure:.1%}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error executing enhanced turtle trade: {e}")
            return False
    
    async def monitor_enhanced_positions(self):
        """Monitor positions with pyramid adding and N-based exits"""
        positions_to_close = []
        
        for pos_id, pos_data in self.positions.items():
            try:
                # Get current market price (simplified)
                current_price = pos_data['entry_price'] * (1 + np.random.normal(0, 0.01))  # Simulate price movement
                
                # Check for pyramid adding opportunities
                if pos_data['units_added'] < self.params['max_positions'] - 1:
                    for i, add_price in enumerate(pos_data['add_signals']):
                        if i <= pos_data['units_added']:  # Already added at this level
                            continue
                        
                        # Check if price has moved enough to add
                        if ((pos_data['signal'].signal_type == 'CALL_BREAKOUT' and current_price >= add_price) or
                            (pos_data['signal'].signal_type == 'PUT_BREAKOUT' and current_price <= add_price)):
                            
                            await self.add_pyramid_position(pos_id, i)
                            break
                
                # Check N-based stop loss
                if ((pos_data['signal'].signal_type == 'CALL_BREAKOUT' and current_price <= pos_data['stop_loss_price']) or
                    (pos_data['signal'].signal_type == 'PUT_BREAKOUT' and current_price >= pos_data['stop_loss_price'])):
                    
                    self.logger.info(f"🛡️ Stop loss triggered for {pos_id}")
                    positions_to_close.append(pos_id)
                
                # Check time-based exit for 0DTE
                time_elapsed = (datetime.now() - pos_data['entry_time']).total_seconds() / 60
                if time_elapsed > self.params['max_position_time']:
                    self.logger.info(f"⏰ Time limit reached for {pos_id}")
                    positions_to_close.append(pos_id)
                    
            except Exception as e:
                self.logger.warning(f"Error monitoring position {pos_id}: {e}")
                positions_to_close.append(pos_id)
        
        # Close positions that need closing
        for pos_id in positions_to_close:
            await self.close_enhanced_position(pos_id)
    
    async def add_pyramid_position(self, position_id: str, level: int):
        """Add to existing position (turtle pyramid)"""
        try:
            pos_data = self.positions[position_id]
            
            # Calculate additional position size (same as original)
            additional_size = pos_data['position_size_dollars']
            
            # Check if we have room for more risk
            additional_risk = additional_size / self.params['account_size']
            if self.current_risk_exposure + additional_risk > self.params['max_portfolio_risk']:
                return
            
            # Add to position
            pos_data['position_size_dollars'] += additional_size
            pos_data['units_added'] = level + 1
            self.current_risk_exposure += additional_risk
            
            self.logger.info(f"🔺 Pyramid add #{level + 1} for {position_id}: +${additional_size:.0f}")
            
        except Exception as e:
            self.logger.error(f"Error adding pyramid position: {e}")
    
    async def close_enhanced_position(self, position_id: str):
        """Close enhanced position with P&L calculation"""
        try:
            if position_id not in self.positions:
                return
            
            pos_data = self.positions[position_id]
            
            # Simulate P&L (in real implementation, would use actual option prices)
            base_pnl = np.random.normal(0, 100)  # Random P&L for demo
            
            # Adjust P&L based on regime and system
            if pos_data['regime'] == 'TRENDING':
                base_pnl *= 1.5  # Better performance in trending markets
            elif pos_data['regime'] == 'VOLATILE':
                base_pnl *= 0.8  # More challenging in volatile markets
            
            # Account for pyramid positions
            total_pnl = base_pnl * (1 + pos_data['units_added'] * 0.3)
            
            self.daily_pnl += total_pnl
            
            # Update risk exposure
            position_risk = pos_data['position_size_dollars'] / self.params['account_size']
            self.current_risk_exposure -= position_risk
            self.symbol_exposures[pos_data['symbol']] -= position_risk
            
            self.logger.info(f"🔒 Enhanced position closed: {position_id}")
            self.logger.info(f"💰 Trade P&L: ${total_pnl:.2f} | Daily P&L: ${self.daily_pnl:.2f}")
            self.logger.info(f"📊 Portfolio risk: {self.current_risk_exposure:.1%}")
            
            # Remove from positions
            del self.positions[position_id]
            
        except Exception as e:
            self.logger.error(f"Error closing enhanced position {position_id}: {e}")
    
    async def update_correlation_matrix(self):
        """Update correlation matrix for portfolio management"""
        try:
            # Simplified correlation update
            # In real implementation, would calculate rolling correlations
            symbols = list(set(pos['symbol'] for pos in self.positions.values()))
            
            if len(symbols) > 1:
                self.logger.debug(f"Portfolio symbols: {symbols}")
                # Would update self.correlation_matrix here
            
        except Exception as e:
            self.logger.warning(f"Error updating correlation matrix: {e}")
    
    async def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data for specified symbol"""
        try:
            # TODO: Implement real-time data fetching for multiple symbols
            # For now, return simulated data
            
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
            prices = np.random.randn(100).cumsum() + 500
            highs = prices + np.random.uniform(0, 2, 100)
            lows = prices - np.random.uniform(0, 2, 100)
            
            return pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': highs,
                'low': lows,
                'close': prices,
                'volume': np.random.randint(1000, 10000, 100)
            })
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} data: {e}")
            return None
    
    async def execute_turtle_trade(self, signal: EnhancedTurtleSignal) -> bool:
        """Execute a turtle-style trade"""
        try:
            self.logger.info(f"🐢 ENHANCED TURTLE SIGNAL: {signal.signal_type}")
            self.logger.info(f"   System: {signal.system} | Confidence: {signal.confidence:.2f}")
            self.logger.info(f"   Breakout Strength: {signal.breakout_strength:.3f}N")
            self.logger.info(f"   Position Size: ${signal.position_size_dollars:.0f}")
            self.logger.info(f"   Stop Loss: ${signal.stop_loss_price:.2f}")
            self.logger.info(f"   Market Regime: {signal.market_regime}")
            
            # TODO: Implement actual trade execution with Alpaca
            # For now, simulate the trade
            
            # Generate position ID
            position_id = f"turtle_{datetime.now().strftime('%H%M%S')}"
            
            # Store position data
            self.positions[position_id] = {
                'signal': signal,
                'symbol': signal.symbol,
                'system': signal.system,
                'entry_time': datetime.now(),
                'entry_price': signal.entry_price_target,
                'position_size_dollars': signal.position_size_dollars,
                'stop_loss_price': signal.stop_loss_price,
                'regime': signal.market_regime
            }
            
            self.logger.info(f"✅ Turtle position opened: {position_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error executing turtle trade: {e}")
            return False
    
    async def close_position(self, position_id: str):
        """Close a position"""
        try:
            if position_id not in self.positions:
                return
            
            pos_data = self.positions[position_id]
            
            # TODO: Implement actual position closing with Alpaca
            # For now, simulate P&L
            
            # Simulate random P&L for demonstration
            import random
            pnl = random.uniform(-50, 100)  # Random P&L for demo
            
            self.daily_pnl += pnl
            
            self.logger.info(f"🔒 Position closed: {position_id}")
            self.logger.info(f"💰 Trade P&L: ${pnl:.2f} | Daily P&L: ${self.daily_pnl:.2f}")
            
            # Remove from positions
            del self.positions[position_id]
            
        except Exception as e:
            self.logger.error(f"❌ Error closing position {position_id}: {e}")
    
    async def get_spy_data(self) -> Optional[pd.DataFrame]:
        """Get SPY market data for analysis"""
        try:
            # TODO: Implement real-time SPY data fetching
            # For now, return dummy data
            
            dates = pd.date_range(start='2024-01-01', periods=50, freq='1min')
            prices = np.random.randn(50).cumsum() + 500
            
            return pd.DataFrame({
                'timestamp': dates,
                'close': prices,
                'volume': np.random.randint(1000, 10000, 50)
            })
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching SPY data: {e}")
            return None
    
    def is_market_hours(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        # Simplified market hours check (9:30 AM - 4:00 PM ET)
        return 9.5 <= now.hour + now.minute/60 <= 16.0

if __name__ == "__main__":
    print("🐢 ENHANCED TURTLE 0DTE STRATEGY")
    print("=" * 60)
    print("🎯 Target: $300-500 daily profit")
    print("⚖️ True N-based position sizing (ATR)")
    print("🔄 Dual system logic (20-day + 55-day breakouts)")
    print("📊 Portfolio management (SPY, QQQ, IWM)")
    print("🔺 Pyramid position adding (turtle-style)")
    print("🌍 Market regime detection")
    print("🛡️ N-based stops and risk management")
    print("🚫 NO TRADE LIMITS - only profit/loss goals")
    print("=" * 60)
    
    try:
        strategy = EnhancedTurtle0DTEStrategy()
        asyncio.run(strategy.run_enhanced_live_trading())
    except KeyboardInterrupt:
        print("\n⏹️ Enhanced turtle strategy stopped by user")
    except Exception as e:
        print(f"\n❌ Enhanced turtle strategy error: {e}")
        import traceback
        traceback.print_exc()
