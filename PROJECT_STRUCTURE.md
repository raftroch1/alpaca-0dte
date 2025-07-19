# Alpaca 0DTE Trading Framework - Project Structure

## 🎯 Overview
This project extends the official Alpaca Python SDK with a specialized framework for 0DTE (Zero Days to Expiration) options trading using ThetaData as the market data provider. The framework is designed for rapid strategy development, backtesting, and live trading with SPY options.

## 📁 Project Structure

```
alpaca-0dte/
├── alpaca/                          # 🚫 CORE SDK - DO NOT MODIFY
│   └── [original alpaca-py files]   # Original Alpaca Python SDK
├── thetadata/                       # ✅ ThetaData Integration
│   ├── theta_connection/            # Core ThetaData connectivity
│   │   ├── __init__.py
│   │   ├── connector.py             # Main ThetaData API connector
│   │   ├── thetadata_collector.py   # Market data caching system
│   │   ├── option_chain_fetcher.py  # Option chain utilities
│   │   ├── utils.py                 # ThetaData utility functions
│   │   ├── ThetaTerminal.jar        # ThetaData Terminal app
│   │   └── test_theta_integration.py
│   ├── cached_data/                 # Cached market data
│   │   ├── spy_bars/               # SPY minute bar data
│   │   └── option_chains/          # Option chain data
│   └── tests/                      # ThetaData testing suite
├── strategies/                      # ✅ Strategy Development
│   ├── base_theta_strategy.py      # Abstract base strategy template
│   ├── cached_strategy_runner.py   # Fast cached strategy execution
│   ├── demo_cached_strategy.py     # Working demo strategy
│   ├── live_ultra_aggressive_0dte.py # Live trading strategy
│   ├── logs/                       # Strategy execution logs
│   └── templates/                  # Strategy templates (to be created)
├── backtrader/                     # ✅ Backtesting Framework
│   ├── multi_day_cached_backtest.py
│   ├── run_v2_real_backtest.py
│   └── results/                    # Backtest results (to be created)
├── config/                         # ✅ Configuration (to be created)
│   ├── environment.yml             # Conda environment specification
│   ├── requirements.txt            # Python dependencies
│   └── trading_config.py           # Trading configuration
└── docs/                          # ✅ Documentation (to be created)
    ├── STRATEGY_DEVELOPMENT.md     # Strategy development guide
    ├── THETADATA_SETUP.md         # ThetaData setup instructions
    └── API_REFERENCE.md           # API reference documentation
```

## 🔧 Core Components

### 1. ThetaData Integration (`/thetadata/`)
- **Purpose**: Provides robust connection to ThetaData API for real options market data
- **Key Files**:
  - `connector.py`: Reusable ThetaData API connector with caching
  - `thetadata_collector.py`: Market data collection and caching system
  - `option_chain_fetcher.py`: Option chain fetching utilities
- **Data Storage**: Compressed pickle files in `cached_data/` for fast loading

### 2. Strategy Framework (`/strategies/`)
- **Purpose**: Template-based strategy development with standardized logging and execution
- **Key Files**:
  - `base_theta_strategy.py`: Abstract base class for all strategies
  - `cached_strategy_runner.py`: Fast execution engine using cached data
- **Requirements**: All strategies MUST inherit from `BaseThetaStrategy`

### 3. Backtesting Framework (`/backtrader/`)
- **Purpose**: Real data backtesting using ThetaData
- **Key Files**:
  - `multi_day_cached_backtest.py`: Multi-day strategy comparison
  - `run_v2_real_backtest.py`: Real data backtesting runner

## 🚨 Critical Rules

### 1. Core SDK Protection
- **NEVER modify any files in the `alpaca/` directory**
- All custom functionality goes in `thetadata/`, `strategies/`, or `backtrader/`
- Use imports to access core Alpaca functionality

### 2. Environment Requirements
- **ALWAYS use Conda environment**: `Alpaca_Options`
- Activate before any development: `conda activate Alpaca_Options`

### 3. Data Requirements
- **ONLY use real ThetaData** - NO simulation fallback
- Proper error handling for missing option data
- Skip trades when real data unavailable (don't simulate)
- Validate ThetaData connection before strategy execution

## 🎯 Strategy Development Workflow

### 1. Strategy Creation Process
1. Inherit from `BaseThetaStrategy` in `/strategies/base_theta_strategy.py`
2. Implement required abstract methods:
   - `analyze_market_conditions()`
   - `execute_strategy()`
   - `calculate_position_size()`
3. Create corresponding backtest file in `/backtrader/`
4. Test with cached data using `cached_strategy_runner.py`
5. Run full backtest with real ThetaData

### 2. Naming Conventions
- **Base strategies**: `[strategy_name]_v1.py`
- **Improved versions**: `[strategy_name]_v2.py`, `[strategy_name]_v3.py`
- **Class names**: PascalCase (e.g., `VixContrarianStrategy`)
- **Log files**: `{strategy_name}_{version}_{timestamp}.log`

### 3. Versioning Rules
- Keep ALL working versions for comparison
- Create new version file when improving strategy
- Document changes between versions in docstring
- Include version comparison in commit messages

## 📊 Logging & Performance Tracking

### Automatic Logging
- **Location**: `strategies/logs/` folder
- **Filename format**: `{strategy_name}_{version}_{timestamp}.log`
- **CSV results**: `{strategy_name}_{version}_{timestamp}_trades.csv`
- **Output**: Both file and console logging

### Performance Metrics
- Win rate and profit/loss tracking
- Trade execution timing
- Data availability validation
- Error handling and recovery

## 🔌 ThetaData Connection

### Connection Requirements
- ThetaData Terminal must be running locally
- Default endpoint: `http://127.0.0.1:25510`
- Connection validation before strategy execution
- Proper error handling for API failures

### Data Caching
- SPY minute bars cached in `thetadata/cached_data/spy_bars/`
- Option chains cached in `thetadata/cached_data/option_chains/`
- Compressed pickle format for fast loading
- Date-based file organization

## 🚀 Quick Start

1. **Environment Setup**:
   ```bash
   conda activate Alpaca_Options
   cd /path/to/alpaca-0dte
   ```

2. **Test ThetaData Connection**:
   ```bash
   python thetadata/theta_connection/connector.py
   ```

3. **Run Demo Strategy**:
   ```bash
   python strategies/demo_cached_strategy.py
   ```

4. **Create New Strategy**:
   ```bash
   cp strategies/base_theta_strategy.py strategies/my_strategy_v1.py
   # Edit and implement required methods
   ```

## 📋 Development Checklist

Before creating a new strategy:
- [ ] Conda environment `Alpaca_Options` activated
- [ ] ThetaData Terminal running and connected
- [ ] Base strategy template copied and renamed
- [ ] Required abstract methods implemented
- [ ] Logging configuration set up
- [ ] Backtest file created
- [ ] Error handling implemented
- [ ] Data validation added

## 🔍 Troubleshooting

### Common Issues
1. **ThetaData Connection Failed**: Ensure ThetaData Terminal is running
2. **Import Errors**: Verify Conda environment is activated
3. **Missing Data**: Check cached data availability for target dates
4. **Strategy Errors**: Ensure all abstract methods are implemented

### Debug Tools
- `thetadata/theta_connection/test_theta_integration.py`
- Connection validation in `connector.py`
- Logging output in `strategies/logs/`
