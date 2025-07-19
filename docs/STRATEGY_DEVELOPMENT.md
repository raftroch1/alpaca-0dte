# Strategy Development Guide - Alpaca 0DTE Framework

## 🎯 Overview
This guide provides step-by-step instructions for developing new trading strategies using the Alpaca 0DTE framework. All strategies must follow the established patterns and inherit from `BaseThetaStrategy`.

## 📋 Prerequisites

### Environment Setup
```bash
# 1. Activate the Conda environment
conda activate Alpaca_Options

# 2. Verify ThetaData connection
python thetadata/theta_connection/connector.py

# 3. Test framework components
python config/trading_config.py
```

### Required Knowledge
- Python 3.8+ programming
- Options trading fundamentals
- Basic understanding of ThetaData API
- Familiarity with Alpaca SDK

## 🚀 Quick Start: Creating Your First Strategy

### Step 1: Copy the Base Template
```bash
# Navigate to strategies directory
cd strategies/

# Copy base template with your strategy name
cp base_theta_strategy.py my_strategy_v1.py
```

### Step 2: Implement Required Methods
Edit `my_strategy_v1.py` and implement these abstract methods:

```python
class MyStrategy(BaseThetaStrategy):
    def analyze_market_conditions(self) -> Dict[str, Any]:
        """
        Analyze current market conditions and return signals.
        
        Returns:
            Dict containing:
            - 'signal': 'BUY', 'SELL', or 'HOLD'
            - 'confidence': float between 0 and 1
            - 'reasoning': str explaining the decision
        """
        # Your market analysis logic here
        pass
    
    def execute_strategy(self, market_data: Dict) -> Optional[Dict]:
        """
        Execute the trading strategy based on market conditions.
        
        Args:
            market_data: Current market data from ThetaData
            
        Returns:
            Trade details dict or None if no trade
        """
        # Your strategy execution logic here
        pass
    
    def calculate_position_size(self, signal_strength: float) -> int:
        """
        Calculate position size based on signal strength and risk management.
        
        Args:
            signal_strength: Confidence level (0.0 to 1.0)
            
        Returns:
            Number of contracts to trade
        """
        # Your position sizing logic here
        pass
```

### Step 3: Create Corresponding Backtest
```bash
# Copy backtest template
cp backtrader/run_v2_real_backtest.py backtrader/my_strategy_backtest.py
```

### Step 4: Test with Cached Data
```bash
# Run with cached data for fast iteration
python strategies/my_strategy_v1.py --use_cached_data --date 20250717
```

### Step 5: Run Full Backtest
```bash
# Run complete backtest with real ThetaData
python backtrader/my_strategy_backtest.py --start_date 20250701 --end_date 20250717
```

## 🎯 Strategy Development Patterns

### 1. Market Analysis Patterns

#### Momentum Strategy
```python
def analyze_market_conditions(self) -> Dict[str, Any]:
    """Example momentum analysis"""
    spy_data = self.get_current_spy_data()
    
    # Calculate momentum indicators
    rsi = self.calculate_rsi(spy_data, period=14)
    macd = self.calculate_macd(spy_data)
    
    # Generate signal
    if rsi > 70 and macd['signal'] == 'SELL':
        return {
            'signal': 'SELL',
            'confidence': 0.75,
            'reasoning': f'RSI overbought ({rsi:.2f}) + MACD sell signal'
        }
    elif rsi < 30 and macd['signal'] == 'BUY':
        return {
            'signal': 'BUY',
            'confidence': 0.75,
            'reasoning': f'RSI oversold ({rsi:.2f}) + MACD buy signal'
        }
    else:
        return {
            'signal': 'HOLD',
            'confidence': 0.5,
            'reasoning': 'No clear momentum signal'
        }
```

#### Mean Reversion Strategy
```python
def analyze_market_conditions(self) -> Dict[str, Any]:
    """Example mean reversion analysis"""
    spy_data = self.get_current_spy_data()
    
    # Calculate mean reversion indicators
    bollinger_bands = self.calculate_bollinger_bands(spy_data)
    current_price = spy_data['close'].iloc[-1]
    
    # Check for mean reversion opportunities
    if current_price > bollinger_bands['upper']:
        return {
            'signal': 'SELL',
            'confidence': 0.8,
            'reasoning': f'Price above upper Bollinger Band ({current_price:.2f} > {bollinger_bands["upper"]:.2f})'
        }
    elif current_price < bollinger_bands['lower']:
        return {
            'signal': 'BUY',
            'confidence': 0.8,
            'reasoning': f'Price below lower Bollinger Band ({current_price:.2f} < {bollinger_bands["lower"]:.2f})'
        }
    else:
        return {
            'signal': 'HOLD',
            'confidence': 0.4,
            'reasoning': 'Price within normal range'
        }
```

### 2. Option Selection Patterns

#### ATM (At-The-Money) Selection
```python
def select_option_strike(self, spy_price: float, option_type: str) -> float:
    """Select ATM option strike"""
    # Round to nearest strike (usually $1 increments for SPY)
    atm_strike = round(spy_price)
    
    if option_type == 'CALL':
        return atm_strike
    else:  # PUT
        return atm_strike
```

#### OTM (Out-of-The-Money) Selection
```python
def select_option_strike(self, spy_price: float, option_type: str, otm_distance: float = 2.0) -> float:
    """Select OTM option strike"""
    if option_type == 'CALL':
        return round(spy_price + otm_distance)
    else:  # PUT
        return round(spy_price - otm_distance)
```

### 3. Risk Management Patterns

#### Fixed Position Size
```python
def calculate_position_size(self, signal_strength: float) -> int:
    """Fixed position size regardless of signal strength"""
    return 1  # Always trade 1 contract
```

#### Signal-Based Position Size
```python
def calculate_position_size(self, signal_strength: float) -> int:
    """Scale position size based on signal confidence"""
    base_size = 1
    max_size = 5
    
    # Scale position size with confidence
    size = int(base_size + (max_size - base_size) * signal_strength)
    return min(size, max_size)
```

#### Kelly Criterion Position Size
```python
def calculate_position_size(self, signal_strength: float) -> int:
    """Kelly criterion-based position sizing"""
    win_rate = self.get_historical_win_rate()
    avg_win = self.get_average_win()
    avg_loss = self.get_average_loss()
    
    if avg_loss == 0:
        return 1
    
    # Kelly fraction
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
    
    # Scale by signal strength
    position_fraction = kelly_fraction * signal_strength
    
    # Convert to number of contracts
    account_value = self.get_account_value()
    option_price = self.get_current_option_price()
    
    if option_price > 0:
        max_contracts = int((account_value * position_fraction) / (option_price * 100))
        return max(1, max_contracts)
    
    return 1
```

## 📊 Testing and Validation

### 1. Unit Testing
Create test files for your strategy components:

```python
# tests/test_my_strategy.py
import unittest
from strategies.my_strategy_v1 import MyStrategy

class TestMyStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MyStrategy()
    
    def test_analyze_market_conditions(self):
        # Test market analysis logic
        pass
    
    def test_calculate_position_size(self):
        # Test position sizing logic
        pass
    
    def test_risk_management(self):
        # Test risk management rules
        pass
```

### 2. Backtesting Validation
```python
# Validate backtest results
def validate_backtest_results(results):
    """Validate that backtest results meet minimum standards"""
    checks = {
        'total_trades': results['total_trades'] > 10,
        'win_rate': results['win_rate'] > 0.40,
        'max_drawdown': results['max_drawdown'] < 0.20,
        'sharpe_ratio': results['sharpe_ratio'] > 0.5
    }
    
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}: {results[check]}")
    
    return all(checks.values())
```

### 3. Paper Trading Validation
```python
# Test strategy with paper trading before live deployment
def run_paper_trading_test(strategy, duration_days=7):
    """Run strategy in paper trading mode for validation"""
    # Implementation for paper trading test
    pass
```

## 🔧 Advanced Features

### 1. Multi-Timeframe Analysis
```python
def analyze_multiple_timeframes(self) -> Dict[str, Any]:
    """Analyze multiple timeframes for better signals"""
    timeframes = ['1Min', '5Min', '15Min', '1H']
    signals = {}
    
    for tf in timeframes:
        spy_data = self.get_spy_data(timeframe=tf)
        signals[tf] = self.analyze_timeframe(spy_data)
    
    # Combine signals from multiple timeframes
    return self.combine_timeframe_signals(signals)
```

### 2. Volatility-Based Adjustments
```python
def adjust_for_volatility(self, base_signal: Dict) -> Dict:
    """Adjust strategy based on current volatility"""
    vix = self.get_vix_data()
    
    if vix > 30:  # High volatility
        base_signal['confidence'] *= 0.8  # Reduce confidence
    elif vix < 15:  # Low volatility
        base_signal['confidence'] *= 1.2  # Increase confidence
    
    return base_signal
```

### 3. Market Regime Detection
```python
def detect_market_regime(self) -> str:
    """Detect current market regime"""
    spy_data = self.get_spy_data(period=20)
    
    # Calculate regime indicators
    volatility = spy_data['close'].pct_change().std() * np.sqrt(252)
    trend = self.calculate_trend_strength(spy_data)
    
    if volatility > 0.25:
        return 'HIGH_VOLATILITY'
    elif trend > 0.7:
        return 'TRENDING'
    elif trend < -0.7:
        return 'DECLINING'
    else:
        return 'SIDEWAYS'
```

## 📈 Performance Optimization

### 1. Data Caching
```python
@lru_cache(maxsize=128)
def get_cached_indicator(self, symbol: str, period: int, indicator_type: str):
    """Cache expensive indicator calculations"""
    # Implementation with caching
    pass
```

### 2. Vectorized Calculations
```python
def calculate_indicators_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
    """Use vectorized operations for better performance"""
    # Use pandas/numpy vectorized operations instead of loops
    data['rsi'] = self.calculate_rsi_vectorized(data['close'])
    data['macd'] = self.calculate_macd_vectorized(data['close'])
    return data
```

### 3. Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor

def analyze_multiple_symbols(self, symbols: List[str]) -> Dict:
    """Analyze multiple symbols in parallel"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(self.analyze_symbol, symbol): symbol 
                  for symbol in symbols}
        
        results = {}
        for future in futures:
            symbol = futures[future]
            results[symbol] = future.result()
        
        return results
```

## 🚨 Common Pitfalls and Solutions

### 1. Look-Ahead Bias
```python
# ❌ WRONG: Using future data
def analyze_market_conditions(self):
    data = self.get_spy_data()
    future_price = data['close'].iloc[-1]  # This is future data!
    
# ✅ CORRECT: Only use past data
def analyze_market_conditions(self):
    data = self.get_spy_data()
    current_price = data['close'].iloc[-2]  # Use previous bar
```

### 2. Survivorship Bias
```python
# ✅ Include delisted options and expired contracts
def get_option_data(self, date: str):
    # Include all options, not just currently active ones
    return self.theta_connector.get_all_options(date, include_expired=True)
```

### 3. Data Snooping
```python
# ✅ Use separate datasets for development and validation
def validate_strategy(self):
    # Use out-of-sample data for final validation
    train_period = ('2024-01-01', '2024-06-30')
    test_period = ('2024-07-01', '2024-12-31')
    
    # Develop on train_period, validate on test_period
```

## 📋 Strategy Checklist

Before deploying a new strategy:

- [ ] All abstract methods implemented
- [ ] Proper error handling for missing data
- [ ] Risk management rules enforced
- [ ] Logging configured correctly
- [ ] Unit tests written and passing
- [ ] Backtest results validated
- [ ] Paper trading test completed
- [ ] Performance metrics acceptable
- [ ] Code reviewed and documented
- [ ] Version control updated

## 🔍 Debugging and Troubleshooting

### Common Issues and Solutions

1. **ThetaData Connection Issues**
   ```bash
   # Test connection
   python thetadata/theta_connection/connector.py
   
   # Check ThetaData Terminal is running
   curl http://127.0.0.1:25510/v2/list/exch
   ```

2. **Missing Option Data**
   ```python
   # Always check data availability
   if option_data is None or option_data.empty:
       self.logger.warning(f"No option data available for {date}")
       return None
   ```

3. **Strategy Not Executing**
   ```python
   # Add debug logging
   self.logger.info(f"Market conditions: {market_analysis}")
   self.logger.info(f"Signal generated: {signal}")
   ```

4. **Performance Issues**
   ```python
   # Profile your code
   import cProfile
   cProfile.run('your_strategy_function()')
   ```

## 📚 Additional Resources

- [ThetaData API Documentation](https://thetadata.net/docs)
- [Alpaca API Documentation](https://alpaca.markets/docs)
- [Backtrader Documentation](https://www.backtrader.com/docu/)
- [Options Trading Fundamentals](https://www.optionseducation.org/)

## 🤝 Contributing

When contributing new strategies or improvements:

1. Follow the established naming conventions
2. Include comprehensive tests
3. Document your strategy logic
4. Provide backtest results
5. Update this guide if adding new patterns
