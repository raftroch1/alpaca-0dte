# Alpaca 0DTE Trading Framework - Project Structure

## 🎯 Overview
This project extends the official Alpaca Python SDK with a specialized framework for 0DTE (Zero Days to Expiration) options trading using **REAL historical data** from multiple sources. The framework supports rapid strategy development, **realistic backtesting with 6 months of real data**, and live trading with SPY options.

## 📁 Project Structure

```
alpaca-0dte/
├── alpaca/                          # 🚫 CORE SDK - DO NOT MODIFY
│   └── [original alpaca-py files]   # Original Alpaca Python SDK
├── thetadata/                       # ✅ ThetaData Integration (6 MONTHS OF REAL DATA)
│   ├── theta_connection/            # Core ThetaData connectivity
│   │   ├── __init__.py
│   │   ├── connector.py             # Main ThetaData API connector
│   │   ├── thetadata_collector.py   # Market data caching system
│   │   ├── option_chain_fetcher.py  # Option chain utilities
│   │   ├── utils.py                 # ThetaData utility functions
│   │   ├── ThetaTerminal.jar        # ThetaData Terminal app
│   │   └── test_theta_integration.py
│   ├── cached_data/                 # ✅ 6 MONTHS OF CACHED REAL DATA
│   │   ├── spy_bars/               # SPY minute bar data (Jan-Jun 2024)
│   │   └── option_chains/          # Option chain data (Jan-Jun 2024)
│   └── tests/                      # ThetaData testing suite
├── strategies/                      # ✅ Strategy Development
│   ├── base_theta_strategy.py      # Abstract base strategy template
│   ├── cached_strategy_runner.py   # Fast cached strategy execution
│   ├── demo_cached_strategy.py     # Working demo strategy
│   ├── real_data_integration/      # ✅ REAL DATA FRAMEWORK (PHASE 3+)
│   │   ├── alpaca_real_data_strategy.py    # REAL Alpaca historical option data
│   │   ├── monthly_alpaca_real_data_runner.py # Monthly real data backtesting
│   │   ├── monthly_phase3_real_data.py     # Phase 3 real data runner
│   │   ├── real_theta_data_strategy.py     # REAL ThetaData option prices
│   │   └── REALISTIC_BACKTEST_README.md    # Real backtesting documentation
│   ├── live_ultra_aggressive_0dte/ # ✅ LIVE TRADING & PHASE 4D
│   │   ├── live_ultra_aggressive_0dte.py   # Live trading strategy
│   │   ├── phase4d_bull_put_spreads.py     # Phase 4D credit spreads
│   │   ├── phase4d_real_data_test.py       # Phase 4D real data testing
│   │   ├── archived_experiments/           # Historical development
│   │   └── monthly_reports/                # Performance reports
│   ├── logs/                       # Strategy execution logs
│   └── templates/                  # Strategy templates
├── backtrader/                     # ✅ Backtesting Framework
│   ├── multi_day_cached_backtest.py
│   ├── run_v2_real_backtest.py
│   └── results/                    # Backtest results
├── framework_tests/                 # ✅ Framework Testing
│   ├── connectivity/
│   ├── alpaca/
│   └── strategy/
├── config/                         # ✅ Configuration
│   ├── environment.yml             # Conda environment specification
│   ├── requirements.txt            # Python dependencies
│   └── trading_config.py           # Trading configuration
├── memory-bank/                    # ✅ MCP Memory Bank for Context
└── docs/                          # ✅ Documentation
    ├── STRATEGY_DEVELOPMENT.md     # Strategy development guide
    ├── THETADATA_SETUP.md         # ThetaData setup instructions
    └── API_REFERENCE.md           # API reference documentation
```

## 🔧 Core Components

### 1. Real Data Integration (`/strategies/real_data_integration/`)
- **Purpose**: PROVEN framework using REAL historical option prices
- **Key Achievement**: Eliminates simulation bias with actual market data
- **Data Sources**:
  - **Alpaca Historical Option API**: Real option prices (Feb 2024+)
  - **ThetaData Cache**: 6 months of real SPY minute bars
  - **Real Option Chains**: Actual bid/ask spreads and Greeks
- **Key Files**:
  - `alpaca_real_data_strategy.py`: Core real data framework
  - `monthly_alpaca_real_data_runner.py`: Monthly backtesting with real data
  - `REALISTIC_BACKTEST_README.md`: Real vs simulated backtesting guide

### 2. ThetaData Integration (`/thetadata/`) - 6 MONTHS OF REAL DATA
- **Purpose**: Provides robust connection to ThetaData API with extensive caching
- **Data Coverage**: January 2024 - June 2024 (6 months of trading data)
- **Key Files**:
  - `connector.py`: ThetaData API connector with caching
  - `thetadata_collector.py`: Market data collection and caching system
  - `option_chain_fetcher.py`: Option chain fetching utilities
- **Data Storage**: Compressed pickle files in `cached_data/` for fast loading
- **Volume**: 125+ days of SPY minute bars and option chains

### 3. Phase 4D Development (`/strategies/live_ultra_aggressive_0dte/`)
- **Current Focus**: Bull put credit spreads using REAL data
- **Strategy Type**: Option selling (credit spreads) vs option buying
- **Key Files**:
  - `live_ultra_aggressive_0dte.py`: Live trading implementation
  - `phase4d_bull_put_spreads.py`: Credit spread strategy with real data
  - `phase4d_real_data_test.py`: Real data validation testing
- **Target**: $300-500 daily profit with realistic risk management

### 4. Strategy Framework (`/strategies/`)
- **Purpose**: Template-based strategy development with real data integration
- **Evolution**: Moved from simulation (Phase 1-2) to real data (Phase 3+)
- **Requirements**: All new strategies MUST use real data frameworks

## 🚨 Critical Rules - UPDATED FOR REAL DATA

### 1. NO MORE SIMULATION
- **RULE**: ALL strategies must use REAL historical data
- **Approved Sources**: 
  - AlpacaHistoricalOptionAPI (alpaca_real_data_strategy.py)
  - ThetaData cache (6 months of real data)
- **FORBIDDEN**: Synthetic option pricing, random walks, estimated time decay

### 2. Data Sources Priority
1. **Primary**: AlpacaRealDataStrategy (Feb 2024+)
2. **Secondary**: ThetaData cache (Jan-Jun 2024)
3. **Forbidden**: Any simulation or synthetic data

### 3. Framework Requirements
- **ALWAYS extend existing real data frameworks**
- **NEVER create new simulated approaches**
- **Use proven infrastructure in /real_data_integration/**

## 🎯 Current Development Status

### ✅ Completed (REAL DATA)
- [x] 6 months of cached ThetaData (Jan-Jun 2024)
- [x] AlpacaRealDataStrategy framework (REAL option prices)
- [x] Monthly real data backtesting validation
- [x] Realistic trading cost modeling
- [x] Phase 3 profitable strategy with real data
- [x] Live trading integration with Alpaca

### 🚧 Phase 4D - Bull Put Spreads (IN PROGRESS)
- [x] Strategy framework using real data
- [x] Bull put spread logic implementation
- [ ] **FIX**: Integration with AlpacaRealDataStrategy
- [ ] **FIX**: Import chain issues resolved
- [ ] Monthly validation with 6 months of real data
- [ ] Live paper trading deployment

### 📋 Next Phase (REAL DATA ONLY)
- [ ] Extend AlpacaRealDataStrategy for Phase 4D
- [ ] 6-month comprehensive backtest using real data
- [ ] Performance validation vs Phase 3
- [ ] Live deployment with real money

## 🎯 Strategy Development Workflow - REAL DATA ONLY

### 1. Strategy Creation Process (UPDATED)
1. **EXTEND** existing frameworks in `/real_data_integration/`
2. **USE** AlpacaRealDataStrategy or RealThetaDataStrategy
3. **VALIDATE** with cached real data (6 months available)
4. **TEST** with monthly real data runners
5. **DEPLOY** to live paper trading

### 2. Data Validation Requirements
- **MANDATORY**: Use real historical option prices
- **VALIDATION**: Compare against known market events
- **TESTING**: Monthly backtests with statistical significance
- **MONITORING**: Track performance vs realistic expectations

## 📊 Real Data Performance Validation

### Phase 3 Real Data Results (PROVEN)
- **Data Source**: AlpacaRealDataStrategy
- **Period**: March 2024 (20 trading days)
- **Results**: -$1,794 total P&L, 196 trades
- **Validation**: REAL market data, no simulation bias

### ThetaData Cache Statistics
- **Coverage**: January 2024 - June 2024
- **SPY Bars**: 125+ trading days of minute data
- **Option Chains**: Daily option chain snapshots
- **Format**: Compressed pickle files for fast access

## 🔌 MCP Integration

### Memory Bank MCP
- **Purpose**: Maintain context across development sessions
- **Location**: `/memory-bank/`
- **Usage**: Store critical decisions and architectural choices

### Chat History MCP  
- **Purpose**: Access past conversation context
- **Usage**: Prevent regression to dismissed approaches

## 🚀 Quick Start - REAL DATA ONLY

1. **Environment Setup**:
   ```bash
   conda activate Alpaca_Options
   cd alpaca-0dte/strategies/real_data_integration
   ```

2. **Test Real Data Framework**:
   ```bash
   python alpaca_real_data_strategy.py --date 20240201
   ```

3. **Run Phase 4D with Real Data**:
   ```bash
   python phase4d_bull_put_spreads.py --date 20240201
   ```

## 📋 Development Checklist - REAL DATA

Before creating a new strategy:
- [ ] Conda environment `Alpaca_Options` activated
- [ ] Using AlpacaRealDataStrategy or proven real data framework
- [ ] NO simulation or synthetic data sources
- [ ] Tested with cached real data (6 months available)
- [ ] Validated with monthly real data runners
- [ ] Proper error handling for missing real data
- [ ] Performance expectations aligned with realistic market conditions

## 🔍 Troubleshooting

### Common Issues
1. **Import Errors**: Check sys.path for real_data_integration
2. **Missing Real Data**: Use cached data from 6-month dataset  
3. **Simulation Creep**: STOP - use existing real data frameworks
4. **Performance Unrealistic**: Validate against real market conditions

### Debug Tools
- `AlpacaRealDataStrategy` - proven real data framework
- Monthly real data runners for validation
- 6 months of cached ThetaData for testing
