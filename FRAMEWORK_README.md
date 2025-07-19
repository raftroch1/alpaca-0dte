# 🎯 Alpaca 0DTE Trading Framework - Complete Template

## 📋 Project Summary

This robust trading framework extends the official Alpaca Python SDK with specialized tools for 0DTE (Zero Days to Expiration) options trading using ThetaData as the market data provider. The framework is designed for rapid strategy development, backtesting, and live trading with SPY options.

## ✅ What's Included

### 🏗️ Core Infrastructure
- **Complete project structure** with proper directory organization
- **Robust connectivity testing** for all framework components
- **Centralized configuration** system for easy customization
- **Template-based strategy development** with inheritance patterns
- **Comprehensive logging and performance tracking**

### 📊 Data Integration
- **ThetaData connectivity** with real options market data
- **Data caching system** for fast strategy iteration
- **Pre-cached SPY data** (8 trading days of minute bars + option chains)
- **Alpaca SDK integration** for live trading capabilities

### 🎯 Strategy Framework
- **BaseThetaStrategy** abstract class for consistent development
- **Strategy template** with implementation examples
- **Cached strategy runner** for fast testing
- **Real-time execution engine** for live trading

### 🧪 Testing & Validation
- **Comprehensive connectivity test suite**
- **Backtrader integration** for robust backtesting
- **Paper trading validation** before live deployment
- **Performance metrics and analysis tools**

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate the conda environment
conda activate Alpaca_Options

# Start ThetaData Terminal
java -jar thetadata/theta_connection/ThetaTerminal.jar

# Verify everything works
python test_framework_connectivity.py
```

### 2. Create Your First Strategy
```bash
# Copy the strategy template
cp strategies/templates/strategy_template_v1.py strategies/my_strategy_v1.py

# Edit and implement your trading logic
# Test with cached data
python strategies/my_strategy_v1.py --use_cached_data
```

### 3. Run Backtests
```bash
# Create corresponding backtest
cp backtrader/run_v2_real_backtest.py backtrader/my_strategy_backtest.py

# Run full backtest with real data
python backtrader/my_strategy_backtest.py
```

## 📁 Project Structure Overview

```
alpaca-0dte/
├── 🚫 alpaca/                    # Original Alpaca SDK (DO NOT MODIFY)
├── ✅ thetadata/                 # ThetaData integration
│   ├── theta_connection/         # Core connectivity
│   ├── cached_data/             # Market data cache
│   └── tests/                   # Testing utilities
├── ✅ strategies/                # Strategy development
│   ├── base_theta_strategy.py   # Abstract base class
│   ├── templates/               # Strategy templates
│   └── logs/                    # Execution logs
├── ✅ backtrader/                # Backtesting framework
│   └── results/                 # Backtest results
├── ✅ config/                    # Configuration files
└── ✅ docs/                      # Documentation
```

## 🎯 Strategy Development Rules

### 1. **Inheritance Requirement**
- All strategies MUST inherit from `BaseThetaStrategy`
- Implement required abstract methods:
  - `analyze_market_conditions()`
  - `execute_strategy()`
  - `calculate_position_size()`

### 2. **Naming Conventions**
- Base strategies: `[strategy_name]_v1.py`
- Improved versions: `[strategy_name]_v2.py`, etc.
- Class names: PascalCase (e.g., `MomentumScalper`)

### 3. **Data Requirements**
- **ONLY use real ThetaData** - NO simulation fallback
- Proper error handling for missing option data
- Skip trades when real data unavailable
- Validate ThetaData connection before execution

### 4. **Logging Standards**
- Automatic logging to `strategies/logs/`
- Filename format: `{strategy_name}_{version}_{timestamp}.log`
- CSV results: `{strategy_name}_{version}_{timestamp}_trades.csv`

## 🔧 Core Components

### ThetaData Integration
- **Connector**: `thetadata/theta_connection/connector.py`
- **Data Collector**: `thetadata/theta_connection/thetadata_collector.py`
- **Option Chain Fetcher**: `thetadata/theta_connection/option_chain_fetcher.py`

### Strategy Framework
- **Base Strategy**: `strategies/base_theta_strategy.py`
- **Strategy Runner**: `strategies/cached_strategy_runner.py`
- **Template**: `strategies/templates/strategy_template_v1.py`

### Configuration
- **Trading Config**: `config/trading_config.py`
- **Environment**: `config/environment.yml`
- **Requirements**: `config/requirements.txt`

## 📊 Testing Results

The framework includes a comprehensive connectivity test that validates:

✅ **Python Environment** (3.8+ with all dependencies)  
✅ **Alpaca SDK** (data client, requests, timeframe)  
✅ **ThetaData Connection** (real-time options data)  
✅ **Strategy Framework** (base class, runner, template)  
✅ **Directory Structure** (all required folders)  
✅ **Data Caching System** (SPY bars + option chains)  
✅ **Backtrader Framework** (backtesting capabilities)  
⚠️ **Configuration** (requires Alpaca API credentials)

**Result**: 7/8 tests passing (missing API credentials expected)

## 🚨 Critical Rules

### 1. **Core SDK Protection**
- **NEVER modify files in the `alpaca/` directory**
- All custom functionality goes in designated folders
- Use imports to access core Alpaca functionality

### 2. **Environment Requirements**
- **ALWAYS use Conda environment**: `Alpaca_Options`
- ThetaData Terminal must be running on `localhost:25510`
- Proper API credentials configured in `.env` file

### 3. **Development Workflow**
1. Activate conda environment
2. Start ThetaData Terminal
3. Copy strategy template
4. Implement required methods
5. Test with cached data
6. Run full backtest
7. Validate results

## 📚 Documentation Files

- **`PROJECT_STRUCTURE.md`** - Detailed project organization
- **`SETUP_INSTRUCTIONS.md`** - Step-by-step setup guide
- **`docs/STRATEGY_DEVELOPMENT.md`** - Comprehensive strategy guide
- **`.windsurfrulesfile`** - Development rules and guidelines

## 🎯 Key Features

### ✅ Template-Based Development
- Consistent strategy structure across all implementations
- Abstract base class ensures proper inheritance
- Pre-built examples and patterns

### ✅ Real Data Integration
- ThetaData for accurate options pricing
- Cached data for fast iteration
- No simulation fallback (real data only)

### ✅ Robust Testing
- Comprehensive connectivity validation
- Backtrader integration for backtesting
- Performance metrics and analysis

### ✅ Production Ready
- Proper logging and error handling
- Risk management frameworks
- Live trading capabilities

## 🚀 Next Steps

1. **Setup Environment**: Follow `SETUP_INSTRUCTIONS.md`
2. **Run Connectivity Test**: `python test_framework_connectivity.py`
3. **Study Examples**: Review existing strategies
4. **Create First Strategy**: Copy template and implement logic
5. **Test and Iterate**: Use cached data for fast development
6. **Deploy**: Run backtests and validate performance

## 🤝 Development Guidelines

### Before Creating a Strategy:
- [ ] Environment activated and tested
- [ ] ThetaData Terminal running
- [ ] API credentials configured
- [ ] Template copied and renamed
- [ ] Required methods understood

### During Development:
- [ ] Follow naming conventions
- [ ] Implement all abstract methods
- [ ] Add proper error handling
- [ ] Include comprehensive logging
- [ ] Test with cached data first

### Before Deployment:
- [ ] Full backtest completed
- [ ] Performance metrics acceptable
- [ ] Risk management validated
- [ ] Code reviewed and documented

## 🎉 Success Metrics

A successful framework implementation should achieve:

- **✅ All connectivity tests passing**
- **✅ Strategies inherit from BaseThetaStrategy**
- **✅ Real ThetaData integration working**
- **✅ Backtesting producing valid results**
- **✅ Logging and performance tracking active**
- **✅ No modifications to core Alpaca SDK**

---

**🎯 This framework provides a complete, production-ready template for developing 0DTE options trading strategies with real market data and robust testing capabilities.**

**Happy Trading! 🚀**
