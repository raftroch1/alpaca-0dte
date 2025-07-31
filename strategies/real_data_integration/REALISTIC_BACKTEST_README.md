# Realistic 0DTE Options Backtest Framework 🎯

## Overview

This framework provides a **comprehensive, realistic backtesting system** for 0DTE options trading that incorporates all real-world trading conditions. Unlike simple backtests that ignore trading costs and market microstructure, this system models:

- ✅ **Real market data** via [Alpaca Historical Option Data API](https://docs.alpaca.markets/docs/historical-option-data)
- ✅ **Realistic trading costs** (commissions, fees, slippage)
- ✅ **Bid/ask spreads** and market microstructure
- ✅ **Order execution modeling** with fill probabilities
- ✅ **Market impact** and liquidity constraints
- ✅ **Supply/demand dynamics** simulation

## 🚨 Why This Framework is Critical

**The original backtest was completely broken**, showing results like:
- **CLAIMED**: +$2,294 daily profits, 95% win rate
- **ACTUAL**: -$374 daily losses, 0% win rate, hit loss limits

This happened because the backtest used completely different logic than the live strategy. Our framework ensures **exact alignment** between backtest and live implementation.

## 📂 Framework Components

### 1. **`realistic_0dte_backtest_v2.py`** - Main Backtest Engine
- Comprehensive backtesting framework with real market conditions
- Alpaca API integration for historical data
- Realistic order execution simulation
- Position tracking and P&L calculation

### 2. **`realistic_trading_config.py`** - Trading Cost Models
- **Commission Structure**: $0.65 base + $0.50/contract + regulatory fees
- **Bid/Ask Spreads**: Dynamic spread modeling based on:
  - Volatility (wider spreads with higher IV)
  - Time to expiry (spreads explode near close)
  - Volume (narrower spreads with higher volume)
  - Moneyness (wider for OTM options)
- **Slippage Models**: Market impact, liquidity penalties
- **Execution Probabilities**: Realistic fill rates and partial fills

### 3. **`strategy_alignment_validator.py`** - Strategy Validation
- Compares live strategy vs backtest implementations
- Identifies mismatched parameters and logic
- Ensures signal generation is identical
- Validates risk management alignment

## 🎯 Realistic Trading Costs (Example)

For a typical 5-contract trade worth $2,500:

```python
Commission breakdown:
- Base commission: $0.65
- Per-contract fees: $2.50 (5 × $0.50)
- Regulatory fees: $0.65
- Exchange fees: $0.75
- TOTAL: $4.55 per trade
```

Plus:
- **Bid/Ask Spread**: $0.05 - $0.50 per contract
- **Slippage**: $0.02 - $0.15 per contract (market orders)
- **Market Impact**: Increases with position size

## 📊 Realistic Market Conditions

### Bid/Ask Spreads
```
Time Period          | Typical Spread
Market Open (9:30)   | $0.10 - $0.20
Regular Hours        | $0.05 - $0.15  
Final Hour (15:00)   | $0.15 - $0.30
Final 5 Minutes      | $0.30 - $0.50+
```

### Order Execution Probabilities
```
Order Type          | Fill Probability
Market Orders       | 95-98%
Limit Orders        | 60-70%
Large Orders (10+)  | 70% (potential partial fills)
High Volatility     | -15% fill probability
Low Volume          | -20% fill probability
```

### Volume Patterns (Realistic for SPY 0DTE)
```
Time        | Volume/Min
09:30       | 500 contracts
12:00       | 100 contracts (lunch lull)
15:30       | 800 contracts (ramp up)
15:55       | 2000+ contracts (panic)
```

## 🔧 Setup Instructions

### 1. Install Dependencies
```bash
pip install alpaca-py pandas numpy
```

### 2. Set Alpaca API Keys
```bash
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
```

### 3. Test Configuration
```bash
python realistic_trading_config.py
```

### 4. Validate Strategy Alignment
```bash
python strategy_alignment_validator.py
```

## ⚠️ Current Status & Next Steps

### ✅ Completed
- [x] Realistic trading cost models
- [x] Bid/ask spread simulation
- [x] Slippage and market impact modeling
- [x] Order execution engine
- [x] Strategy alignment validator
- [x] Alpaca API integration framework

### 🚧 In Progress
- [ ] **CRITICAL**: Extract exact signal generation logic from live strategy
- [ ] **CRITICAL**: Implement identical position sizing algorithms
- [ ] **CRITICAL**: Match risk management parameters exactly

### 📋 To Do
- [ ] Integrate real Alpaca historical option data
- [ ] Implement proper option pricing models (Black-Scholes)
- [ ] Add Greeks calculation for realistic option behavior
- [ ] Create comprehensive test suite
- [ ] Run validation against multiple dates

## 🎯 How to Fix the Current Issues

### Issue 1: Missing Signal Generation Logic
**Problem**: Backtest uses simple price change detection, live strategy uses complex multi-factor analysis.

**Solution**: Extract the exact `generate_trading_signal()` method from `live_ultra_aggressive_0dte.py` and implement it in the backtest.

### Issue 2: Parameter Misalignment  
**Problem**: Different risk limits, position sizes, and thresholds.

**Solution**: Use the validator to identify ALL parameter differences and align them exactly.

### Issue 3: Trading Conditions
**Problem**: Entry/exit logic differs between implementations.

**Solution**: Copy the exact conditional logic from live strategy to backtest.

## 📈 Expected Realistic Results

With proper alignment and realistic costs, expect:

- **Win Rate**: 45-55% (typical for 0DTE options)
- **Average Win**: $30-50 per contract
- **Average Loss**: $20-40 per contract  
- **Transaction Costs**: $5-15 per round trip
- **Daily P&L**: Highly variable, often negative after costs

## 🚨 Critical Warnings

1. **Never trust backtest results without proper validation**
2. **Transaction costs can easily eliminate profits**
3. **Bid/ask spreads widen dramatically near expiry**
4. **Partial fills and failed orders are common**
5. **Market conditions change rapidly for 0DTE options**

## 📞 Implementation Priority

### Phase 1: Strategy Alignment (CRITICAL)
1. Fix signal generation logic mismatch
2. Align all parameters exactly
3. Validate with alignment tool

### Phase 2: Real Data Integration
1. Implement Alpaca option data fetching
2. Add proper option pricing models
3. Test with historical data

### Phase 3: Advanced Features
1. Greeks modeling
2. Volatility surface interpolation
3. Multi-day backtesting

## 🎯 Success Criteria

The backtest is complete when:
- ✅ Strategy alignment validator shows 0 errors
- ✅ Signal generation logic is identical
- ✅ All parameters match exactly
- ✅ Results are consistent across multiple test dates
- ✅ Performance matches expected realistic ranges

## 💡 Key Insights

1. **Realistic costs dramatically impact profitability**
2. **0DTE options have extreme bid/ask spreads near expiry**
3. **Order execution becomes difficult in final minutes**
4. **Position sizing must account for all transaction costs**
5. **Strategy validation is essential for reliable results**

---

**Next Action**: Run the strategy alignment validator, fix all errors, then test with real market data.

Remember: **A backtest is only as good as its alignment with reality!** 🎯 