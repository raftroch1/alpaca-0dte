# 🎯 UNIVERSAL FRAMEWORK GUIDE
## Creating Profitable 0DTE Strategies with Real Data

**Version**: 2.0  
**Date**: 2025-01-31  
**Based on**: Multi-Regime, Turtle, Phase 4 Aggressive proven strategies  

---

## 📋 QUICK START

### 1. Create New Strategy
```bash
# Copy template
cp templates/universal_strategy_template.py strategies/your_strategy.py

# Copy backtest template  
cp templates/universal_backtest_template.py strategies/your_strategy_backtest.py
```

### 2. Customize Strategy
```python
# Rename class and implement generate_trading_signal()
class YourStrategy(UniversalStrategyTemplate):
    def generate_trading_signal(self, spy_data):
        # Your strategy logic here
        return {'signal': 'BUY_CALL', 'confidence': 0.75}
```

### 3. Test with Real Data
```bash
# Run backtest with cached data
python your_strategy_backtest.py --start-date 20240301 --end-date 20240331

# Start paper trading
python your_strategy.py
```

---

## ✅ PROVEN PATTERNS CODIFIED

### 🔑 API Key Management
```python
# Standardized pattern from all working strategies
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
```

### 🏦 Alpaca Client Initialization  
```python
# Pattern from Live Ultra Aggressive, Multi-Regime, Turtle
self.trading_client = TradingClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_SECRET_KEY"),
    paper=True  # Start with paper trading
)

self.stock_client = StockHistoricalDataClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_SECRET_KEY")
)

self.option_client = OptionHistoricalDataClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_SECRET_KEY")
)
```

### 📊 Real Data Integration
```python
# ThetaData cache pattern (6 months: Jan-Jun 2024)
spy_file = os.path.join(cache_dir, 'spy_bars', f'spy_bars_{date_str}.pkl.gz')
with gzip.open(spy_file, 'rb') as f:
    spy_data = pickle.load(f)

# Alpaca real option prices pattern
symbol = f"SPY{exp_date}{option_letter}{strike_str}"
request = OptionBarsRequest(symbol_or_symbols=[symbol], ...)
bars = self.alpaca_client.get_option_bars(request)
```

### 🎯 Strategy Structure Pattern
```python
class YourStrategy:
    def __init__(self):
        self._validate_environment()      # API keys
        self._initialize_alpaca_clients() # Trading & data clients
        self.params = self._get_strategy_parameters()  # Config
        self._initialize_tracking()       # Performance tracking
    
    def generate_trading_signal(self, spy_data):
        # Core strategy logic - IMPLEMENT THIS
        pass
    
    async def run_live_trading(self):
        # Main trading loop with proven patterns
        pass
```

---

## 📊 FRAMEWORK COMPONENTS

### 1. Universal Strategy Template
**File**: `templates/universal_strategy_template.py`

**Features**:
- ✅ Proven API key management patterns
- ✅ Standardized Alpaca client initialization
- ✅ Real-time data integration (no simulation)
- ✅ Comprehensive risk management
- ✅ Performance tracking and logging
- ✅ Market hours validation
- ✅ Position sizing and limits

**Customization Points**:
- `generate_trading_signal()` - Your core strategy logic
- `_get_strategy_parameters()` - Strategy-specific parameters
- Signal confidence thresholds and risk settings

### 2. Universal Backtest Template  
**File**: `templates/universal_backtest_template.py`

**Features**:
- ✅ Real historical data from ThetaData cache + Alpaca API
- ✅ No simulation or synthetic pricing
- ✅ Comprehensive performance statistics
- ✅ Proven patterns from Multi-Regime ($71k profit) validation
- ✅ Daily P&L tracking and risk management
- ✅ Commission and slippage modeling

**Data Sources**:
- **SPY Bars**: ThetaData cache (6 months: Jan-Jun 2024)
- **Option Prices**: Alpaca historical API (real market data)
- **No Simulation**: Only actual historical market data

### 3. Real Data Integration
**Pattern**: Extends `AlpacaRealDataStrategy` proven framework

**Key Methods**:
- `load_cached_spy_data()` - ThetaData cache loading
- `get_real_option_price()` - Alpaca historical option pricing
- `validate_option_data()` - Data quality checks

---

## 🎯 STRATEGY DEVELOPMENT WORKFLOW

### Phase 1: Strategy Design
1. **Copy Templates**: Start with universal templates
2. **Define Strategy**: Implement `generate_trading_signal()`
3. **Set Parameters**: Configure risk, timing, signal thresholds
4. **Add Indicators**: RSI, MACD, moving averages, etc.

### Phase 2: Backtesting  
1. **Single Day Test**: Test with one day of cached data
2. **Weekly Test**: Validate over 1 week period
3. **Monthly Test**: Full month validation with real data
4. **Parameter Optimization**: Tune based on real results

### Phase 3: Paper Trading
1. **Initialize Paper Mode**: `paper=True` in TradingClient
2. **Monitor Performance**: Track against backtest results
3. **Risk Validation**: Ensure real trading matches expectations
4. **Fine Tuning**: Adjust based on live market conditions

### Phase 4: Live Trading (When Profitable)
1. **Set Capital Allocation**: Start with small position sizes
2. **Enable Live Mode**: `paper=False` in TradingClient  
3. **Monitor Closely**: Track daily P&L vs targets
4. **Scale Gradually**: Increase size as confidence grows

---

## 📈 PROVEN STRATEGY EXAMPLES

### 1. Multi-Regime Strategy  
**Results**: $71,724 profit over 6 months  
**Pattern**: VIX-based strategy selection
```python
# VIX thresholds for regime detection
if vix < 15: strategy = "diagonal_spread"
elif vix > 25: strategy = "iron_condor"  
else: strategy = "credit_spread"
```

### 2. Turtle Strategy
**Pattern**: ATR-based breakout system
```python
# Turtle position sizing with True N (ATR)
current_n = calculate_atr(spy_data, 20)
position_size = (account_size * 0.01) / (current_n * 100)
stop_loss = entry_price - (2.0 * current_n)
```

### 3. Phase 4 Aggressive
**Pattern**: High-frequency momentum trading
```python
# Dynamic position sizing based on confidence
base_size = 1
confidence_multiplier = 0.5 + (confidence * 0.5)
position_size = int(base_size * confidence_multiplier)
```

---

## 🔧 CONFIGURATION PATTERNS

### Risk Management (Proven Settings)
```python
'risk_per_trade': 0.01,        # 1% account risk per trade
'daily_loss_limit': 350.0,     # $350 daily stop loss
'daily_profit_target': 500.0,  # $500 daily target
'max_daily_trades': 10,        # Trade frequency limit
'max_position_size': 5,        # Contract size limit
```

### Option Selection (Validated Criteria)
```python
'min_option_price': 0.10,      # Minimum premium
'max_option_price': 3.00,      # Maximum premium
'max_bid_ask_spread': 0.15,    # Max 15% spread
'min_open_interest': 100,      # Liquidity requirement
'preferred_dte': 0,            # 0DTE same-day expiry
```

### Market Timing (Market Hours Patterns)
```python
'market_open_buffer': 15,      # Wait 15min after open
'market_close_buffer': 30,     # Exit 30min before close
'min_time_to_expiry': 30,      # Min 30min to expiry
'max_position_time': 120,      # Max 2hr hold time
```

---

## 📊 DATA SOURCES & VALIDATION

### ThetaData Cache (Primary SPY Data)
- **Coverage**: 6 months (Jan-Jun 2024)
- **Resolution**: Minute bars
- **File Format**: `.pkl.gz` compressed pickle
- **Location**: `thetadata/cached_data/spy_bars/`
- **Validation**: Used by profitable Multi-Regime strategy

### Alpaca Historical API (Option Prices)
- **Coverage**: February 2024 onwards  
- **Data Type**: Real historical option prices
- **Format**: OHLCV bars via `OptionHistoricalDataClient`
- **Validation**: Used by `AlpacaRealDataStrategy` framework

### No Simulation Sources
- ❌ **No synthetic time decay models**
- ❌ **No random walk option pricing**  
- ❌ **No estimated Greeks or volatility**
- ❌ **No Monte Carlo simulations**

---

## 🛠️ DEVELOPMENT CHECKLIST

### ✅ Strategy Development
- [ ] Copy universal templates to new strategy files
- [ ] Rename classes to match strategy name
- [ ] Implement `generate_trading_signal()` method
- [ ] Configure strategy parameters in `_get_strategy_parameters()`
- [ ] Add technical indicators and signal logic
- [ ] Test signal generation with sample data

### ✅ Backtesting Validation  
- [ ] Test with single day of cached data
- [ ] Validate option price retrieval from Alpaca
- [ ] Run weekly backtest (5 trading days)
- [ ] Run monthly backtest (20+ trading days)
- [ ] Analyze win rate, P&L, and drawdown statistics
- [ ] Compare results to proven strategy benchmarks

### ✅ Paper Trading Preparation
- [ ] Verify Alpaca API credentials in `.env` file
- [ ] Confirm paper trading mode (`paper=True`)
- [ ] Test real-time data feeds and market hours
- [ ] Validate order submission and position tracking
- [ ] Set up logging and performance monitoring
- [ ] Test emergency stop and cleanup procedures

### ✅ Risk Management Validation
- [ ] Verify daily loss limits are enforced
- [ ] Test position sizing calculations
- [ ] Validate market hours and timing buffers
- [ ] Confirm option selection criteria
- [ ] Test bid-ask spread filtering
- [ ] Verify commission and slippage calculations

---

## 📚 EXAMPLES & REFERENCES

### Strategy Implementation Examples
```python
# Momentum Strategy Example
def generate_trading_signal(self, spy_data):
    rsi = self._calculate_rsi(spy_data['close'])
    macd = self._calculate_macd(spy_data['close'])
    
    if rsi < 30 and macd > 0:
        return {
            'signal': 'BUY_CALL',
            'confidence': 0.75,
            'reasoning': 'Oversold + bullish MACD'
        }
```

### Backtest Usage Examples
```bash
# Single month test
python strategy_backtest.py --start-date 20240301 --end-date 20240331

# Quarter test  
python strategy_backtest.py --start-date 20240101 --end-date 20240331

# Custom cache directory
python strategy_backtest.py --start-date 20240301 --end-date 20240331 --cache-dir /path/to/cache
```

### Paper Trading Examples
```bash
# Start paper trading
python your_strategy.py

# Monitor logs
tail -f logs/your_strategy_live.log
```

---

## 🚨 CRITICAL RULES

### 1. NO SIMULATION ARCHITECTURE
- ❌ **Never use synthetic option pricing**
- ❌ **Never estimate time decay or Greeks**
- ❌ **Never use random walk or Monte Carlo**
- ✅ **Always use real historical data sources**

### 2. DATA SOURCE PRIORITY
1. **ThetaData Cache** (SPY minute bars)
2. **Alpaca Historical API** (option prices)  
3. **No other sources** (no simulation fallbacks)

### 3. FRAMEWORK REQUIREMENTS  
- ✅ **Always extend proven base templates**
- ✅ **Always use standardized import paths**
- ✅ **Always load API keys from environment**
- ✅ **Always implement comprehensive logging**

### 4. TESTING REQUIREMENTS
- ✅ **Always backtest with real cached data first**
- ✅ **Always validate with paper trading second**  
- ✅ **Never skip to live trading without validation**
- ✅ **Always compare to proven strategy benchmarks**

---

## 🎯 SUCCESS METRICS

### Backtest Validation Targets
- **Win Rate**: >65% (based on proven strategies)
- **Daily P&L**: $300-500 target (realistic for $25k account)
- **Max Drawdown**: <15% (risk management validation)
- **Profitable Days**: >60% of trading days

### Paper Trading Validation  
- **Performance Match**: Within 20% of backtest results
- **Risk Adherence**: All limits properly enforced
- **Execution Quality**: Orders filled at expected prices
- **Stability**: No system errors or crashes

### Live Trading Success
- **Consistent Profitability**: Meet daily targets >50% of days
- **Risk Control**: Never exceed daily loss limits
- **Scalability**: Increase position sizes as confidence grows
- **Adaptability**: Adjust to changing market conditions

---

## 📞 SUPPORT & TROUBLESHOOTING

### Common Issues

**1. Import Errors**
- Check `sys.path.append()` statements match project structure
- Verify all required packages are installed
- Confirm Python environment is activated

**2. API Connection Issues**  
- Verify `.env` file contains valid Alpaca credentials
- Check network connectivity and API status
- Confirm paper trading mode for testing

**3. Data Loading Errors**
- Verify cache directory path is correct
- Check ThetaData cache files exist for target dates
- Confirm file permissions allow reading

**4. Backtest Inconsistencies**
- Validate date ranges match available cached data
- Check option symbol formatting for Alpaca API
- Verify commission and slippage settings

### Framework Updates
- Check `PROJECT_STRUCTURE.md` for latest patterns
- Review working strategy examples for new techniques
- Monitor performance of proven strategies for benchmark updates

---

**🎯 Ready to build profitable strategies with real data validation!** 