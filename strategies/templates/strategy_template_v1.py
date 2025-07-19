#!/usr/bin/env python3
"""
STRATEGY TEMPLATE - Copy this file to create new strategies
============================================================

This template provides the basic structure for all 0DTE option strategies.
Copy this file and rename it following the naming convention:
[strategy_name]_v1.py

Example: momentum_scalper_v1.py, mean_reversion_v1.py, etc.

REQUIRED STEPS:
1. Copy this file: cp strategy_template_v1.py your_strategy_v1.py
2. Rename the class: StrategyTemplate -> YourStrategy
3. Implement all abstract methods
4. Add your strategy logic
5. Create corresponding backtest file
6. Test with cached data first
7. Run full backtest with real ThetaData
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

# Import the base strategy class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_theta_strategy import BaseThetaStrategy

class StrategyTemplate(BaseThetaStrategy):
    """
    Template for creating new 0DTE option strategies.
    
    RENAME THIS CLASS to match your strategy name in PascalCase.
    Example: MomentumScalper, MeanReversion, VixContrarian, etc.
    """
    
    def __init__(self, strategy_name: str = "strategy_template"):
        """
        Initialize the strategy with custom parameters.
        
        Args:
            strategy_name: Name of your strategy (used for logging)
        """
        super().__init__(strategy_name)
        
        # 🎯 STRATEGY-SPECIFIC PARAMETERS
        # Customize these parameters for your strategy
        self.confidence_threshold = 0.60  # Minimum confidence to trade
        self.max_daily_trades = 10        # Maximum trades per day
        self.min_option_price = 0.30      # Minimum option price to consider
        self.max_option_price = 2.50      # Maximum option price to consider
        self.position_size_base = 1       # Base position size
        self.risk_per_trade = 0.02        # 2% risk per trade
        
        # 📊 TECHNICAL INDICATORS SETTINGS
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.ma_short_period = 5
        self.ma_long_period = 20
        
        # 🎯 OPTION SELECTION SETTINGS
        self.preferred_dte = 0            # Days to expiration (0 = same day)
        self.strike_selection = "ATM"     # "ATM", "OTM", "ITM"
        self.otm_distance = 2.0           # Distance from current price for OTM
        
        self.logger.info(f"✅ {strategy_name} initialized with custom parameters")
    
    def analyze_market_conditions(self) -> Dict[str, Any]:
        """
        🔍 IMPLEMENT THIS METHOD - Analyze market conditions and generate signals.
        
        This is where you implement your core strategy logic.
        Analyze SPY price action, technical indicators, market conditions, etc.
        
        Returns:
            Dict containing:
            - 'signal': 'BUY', 'SELL', or 'HOLD'
            - 'confidence': float between 0 and 1
            - 'reasoning': str explaining the decision
            - 'option_type': 'CALL' or 'PUT' (if signal is not HOLD)
            - 'target_strike': float (if signal is not HOLD)
        
        Example implementation patterns:
        - Momentum: RSI + MACD + price breakouts
        - Mean reversion: Bollinger Bands + RSI extremes
        - Volatility: VIX levels + implied volatility
        - News/Events: Economic calendar + market reactions
        """
        try:
            # 📊 GET CURRENT MARKET DATA
            spy_data = self.get_current_spy_data()
            if spy_data is None or spy_data.empty:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reasoning': 'No SPY data available'
                }
            
            current_price = spy_data['close'].iloc[-1]
            
            # 🔍 EXAMPLE ANALYSIS - REPLACE WITH YOUR LOGIC
            # This is just a template - implement your actual strategy here
            
            # Calculate technical indicators
            rsi = self.calculate_rsi(spy_data, self.rsi_period)
            ma_short = spy_data['close'].rolling(self.ma_short_period).mean().iloc[-1]
            ma_long = spy_data['close'].rolling(self.ma_long_period).mean().iloc[-1]
            
            # Example signal logic (REPLACE THIS)
            if rsi < self.rsi_oversold and current_price < ma_long:
                return {
                    'signal': 'BUY',
                    'confidence': 0.75,
                    'reasoning': f'RSI oversold ({rsi:.1f}) + price below long MA',
                    'option_type': 'CALL',
                    'target_strike': self.select_strike(current_price, 'CALL')
                }
            elif rsi > self.rsi_overbought and current_price > ma_long:
                return {
                    'signal': 'SELL',
                    'confidence': 0.75,
                    'reasoning': f'RSI overbought ({rsi:.1f}) + price above long MA',
                    'option_type': 'PUT',
                    'target_strike': self.select_strike(current_price, 'PUT')
                }
            else:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.4,
                    'reasoning': f'No clear signal (RSI: {rsi:.1f}, Price: {current_price:.2f})'
                }
                
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reasoning': f'Analysis error: {str(e)}'
            }
    
    def execute_strategy(self, market_data: Dict) -> Optional[Dict]:
        """
        🎯 IMPLEMENT THIS METHOD - Execute trades based on market analysis.
        
        This method receives the market analysis and decides whether to execute a trade.
        It should handle option selection, price validation, and trade execution.
        
        Args:
            market_data: Dict containing current market data and analysis
            
        Returns:
            Trade details dict if trade executed, None otherwise
            
        Trade dict should contain:
        - 'action': 'BUY' or 'SELL'
        - 'option_type': 'CALL' or 'PUT'
        - 'strike': float
        - 'quantity': int
        - 'price': float
        - 'timestamp': datetime
        - 'reasoning': str
        """
        try:
            # Get market analysis
            analysis = self.analyze_market_conditions()
            
            # Check if we should trade
            if analysis['signal'] == 'HOLD':
                return None
            
            if analysis['confidence'] < self.confidence_threshold:
                self.logger.info(f"Signal confidence too low: {analysis['confidence']:.2f}")
                return None
            
            # Check daily trade limit
            if self.total_trades >= self.max_daily_trades:
                self.logger.info(f"Daily trade limit reached: {self.total_trades}")
                return None
            
            # 🎯 OPTION SELECTION AND VALIDATION
            option_type = analysis['option_type']
            strike = analysis['target_strike']
            
            # Get option price from ThetaData
            option_price = self.get_option_price(
                symbol='SPY',
                strike=strike,
                option_type=option_type,
                expiry_date=self.get_current_expiry_date()
            )
            
            if option_price is None:
                self.logger.warning(f"No option price available for {strike} {option_type}")
                return None
            
            # Validate option price range
            if option_price < self.min_option_price or option_price > self.max_option_price:
                self.logger.info(f"Option price outside range: ${option_price:.2f}")
                return None
            
            # 📊 POSITION SIZING
            position_size = self.calculate_position_size(analysis['confidence'])
            
            # 🎯 EXECUTE TRADE
            trade_details = {
                'action': 'BUY',  # Always buying options in this template
                'option_type': option_type,
                'strike': strike,
                'quantity': position_size,
                'price': option_price,
                'timestamp': datetime.now(),
                'reasoning': analysis['reasoning'],
                'confidence': analysis['confidence']
            }
            
            # Log the trade
            self.logger.info(f"🎯 TRADE EXECUTED: {trade_details}")
            
            # Update counters
            self.total_trades += 1
            
            return trade_details
            
        except Exception as e:
            self.logger.error(f"Error executing strategy: {e}")
            return None
    
    def calculate_position_size(self, signal_strength: float) -> int:
        """
        💰 IMPLEMENT THIS METHOD - Calculate position size based on signal strength.
        
        This method determines how many contracts to trade based on:
        - Signal confidence/strength
        - Risk management rules
        - Account size
        - Current portfolio exposure
        
        Args:
            signal_strength: Confidence level (0.0 to 1.0)
            
        Returns:
            Number of contracts to trade
        """
        try:
            # 🎯 EXAMPLE POSITION SIZING - CUSTOMIZE THIS
            
            # Base position size
            base_size = self.position_size_base
            
            # Scale with signal strength
            scaled_size = int(base_size * signal_strength)
            
            # Apply risk management limits
            max_size = 5  # Maximum contracts per trade
            min_size = 1  # Minimum contracts per trade
            
            position_size = max(min_size, min(scaled_size, max_size))
            
            self.logger.info(f"Position size calculated: {position_size} contracts (strength: {signal_strength:.2f})")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1  # Default to 1 contract
    
    # 🛠️ HELPER METHODS - Customize these for your strategy
    
    def select_strike(self, current_price: float, option_type: str) -> float:
        """Select option strike based on strategy settings"""
        if self.strike_selection == "ATM":
            return round(current_price)
        elif self.strike_selection == "OTM":
            if option_type == "CALL":
                return round(current_price + self.otm_distance)
            else:  # PUT
                return round(current_price - self.otm_distance)
        elif self.strike_selection == "ITM":
            if option_type == "CALL":
                return round(current_price - self.otm_distance)
            else:  # PUT
                return round(current_price + self.otm_distance)
        else:
            return round(current_price)  # Default to ATM
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return 50.0  # Neutral RSI if calculation fails
    
    def get_current_expiry_date(self) -> str:
        """Get current expiry date for 0DTE options"""
        # For 0DTE, expiry is same day
        # Format: YYYYMMDD
        return datetime.now().strftime('%Y%m%d')

# 🎯 MAIN EXECUTION - For testing the strategy
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Strategy Template')
    parser.add_argument('--date', type=str, help='Date to run strategy (YYYYMMDD)')
    parser.add_argument('--use_cached_data', action='store_true', help='Use cached data for testing')
    
    args = parser.parse_args()
    
    # Initialize strategy
    strategy = StrategyTemplate("strategy_template_test")
    
    print("🎯 STRATEGY TEMPLATE TEST")
    print("=" * 50)
    print("This is a template file. Copy and customize it for your strategy.")
    print("=" * 50)
    
    # Test market analysis
    analysis = strategy.analyze_market_conditions()
    print(f"📊 Market Analysis: {analysis}")
    
    # Test strategy execution
    trade = strategy.execute_strategy({})
    if trade:
        print(f"🎯 Trade Generated: {trade}")
    else:
        print("❌ No trade generated")
    
    print("\n✅ Template test completed. Now customize this file for your strategy!")
