# Alpaca 0DTE Trading Framework - Setup Instructions

## 🎯 Quick Setup Guide

### 1. Environment Setup

#### Create Conda Environment
```bash
# Create the environment from the provided file
conda env create -f config/environment.yml

# Activate the environment
conda activate Alpaca_Options

# Verify installation
python --version  # Should be 3.10+
```

#### Alternative: Manual Environment Setup
```bash
# Create environment manually
conda create -n Alpaca_Options python=3.10
conda activate Alpaca_Options

# Install dependencies
pip install -r config/requirements.txt
```

### 2. ThetaData Setup

#### Start ThetaData Terminal
```bash
# Navigate to project directory
cd /path/to/alpaca-0dte

# Start ThetaData Terminal (required for options data)
java -jar thetadata/theta_connection/ThetaTerminal.jar
```

**Important**: 
- ThetaData Terminal must be running before using any strategies
- Enter your ThetaData credentials when prompted
- The terminal runs on `localhost:25510` by default

### 3. Alpaca API Configuration

#### Set Environment Variables
Create a `.env` file in the project root:

```bash
# Copy the example and edit with your credentials
cp .env.example .env
```

Edit `.env` file:
```bash
# Alpaca API Credentials
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here

# Trading Environment (paper or live)
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading
# ALPACA_BASE_URL=https://api.alpaca.markets      # Live trading

# Data URL
ALPACA_DATA_URL=https://data.alpaca.markets
```

#### Get Alpaca API Keys
1. Sign up at [Alpaca Markets](https://alpaca.markets/)
2. Go to your dashboard
3. Generate API keys for paper trading (recommended for testing)
4. Copy keys to your `.env` file

### 4. Verify Setup

#### Run Connectivity Test
```bash
# Ensure conda environment is activated
conda activate Alpaca_Options

# Run comprehensive connectivity test
python test_framework_connectivity.py
```

**Expected Results**:
- ✅ 8/8 tests should pass (with proper API credentials)
- ✅ ThetaData connection successful
- ✅ All framework components working

#### Test Individual Components
```bash
# Test ThetaData connection only
python thetadata/theta_connection/connector.py

# Test configuration
python config/trading_config.py

# Test strategy template
python strategies/templates/strategy_template_v1.py
```

## 🚀 First Strategy Development

### 1. Copy Strategy Template
```bash
# Navigate to strategies directory
cd strategies/

# Copy template with your strategy name
cp templates/strategy_template_v1.py my_first_strategy_v1.py
```

### 2. Customize Your Strategy
Edit `my_first_strategy_v1.py`:

1. **Rename the class**: `StrategyTemplate` → `MyFirstStrategy`
2. **Implement required methods**:
   - `analyze_market_conditions()`
   - `execute_strategy()`
   - `calculate_position_size()`
3. **Add your trading logic**

### 3. Test Your Strategy
```bash
# Test with cached data (fast)
python my_first_strategy_v1.py --use_cached_data --date 20250717

# Run full backtest (slower, uses real ThetaData)
python ../backtrader/run_v2_real_backtest.py --strategy my_first_strategy_v1
```

## 📊 Data Management

### Cached Data
The framework includes pre-cached SPY options data:
- **Location**: `thetadata/cached_data/`
- **SPY Bars**: `spy_bars/` (minute-level data)
- **Option Chains**: `option_chains/` (0DTE options)
- **Format**: Compressed pickle files

### Collect New Data
```bash
# Collect data for specific date
python thetadata/theta_connection/thetadata_collector.py --date 20250717

# Collect data for date range
python thetadata/theta_connection/thetadata_collector.py --start_date 20250701 --end_date 20250717
```

## 🔧 Development Workflow

### Daily Development Routine
```bash
# 1. Activate environment
conda activate Alpaca_Options

# 2. Start ThetaData Terminal
java -jar thetadata/theta_connection/ThetaTerminal.jar &

# 3. Verify connectivity
python test_framework_connectivity.py

# 4. Develop your strategy
# Edit your strategy file...

# 5. Test with cached data
python strategies/your_strategy_v1.py --use_cached_data

# 6. Run backtest
python backtrader/your_strategy_backtest.py
```

### Strategy Versioning
```bash
# When improving a strategy, create new version
cp my_strategy_v1.py my_strategy_v2.py

# Update class name and version in the new file
# Keep all versions for comparison
```

## 🚨 Troubleshooting

### Common Issues

#### 1. ThetaData Connection Failed
```bash
# Check if ThetaData Terminal is running
curl http://127.0.0.1:25510/v2/list/exch

# If not running, start it:
java -jar thetadata/theta_connection/ThetaTerminal.jar
```

#### 2. Import Errors
```bash
# Ensure conda environment is activated
conda activate Alpaca_Options

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Reinstall dependencies if needed
pip install -r config/requirements.txt
```

#### 3. Missing API Credentials
```bash
# Check .env file exists and has correct format
cat .env

# Verify environment variables are loaded
python -c "import os; print(os.getenv('ALPACA_API_KEY'))"
```

#### 4. Permission Issues
```bash
# Fix file permissions
chmod +x thetadata/theta_connection/ThetaTerminal.jar
chmod -R 755 strategies/
chmod -R 755 thetadata/
```

### Debug Mode
```bash
# Run with debug logging
export PYTHONPATH=$PWD:$PYTHONPATH
python -v your_script.py  # Verbose import debugging
```

## 📋 Pre-Development Checklist

Before starting development:

- [ ] Conda environment `Alpaca_Options` activated
- [ ] ThetaData Terminal running and connected
- [ ] Alpaca API credentials configured in `.env`
- [ ] Connectivity test passing (8/8 tests)
- [ ] Project structure understood
- [ ] Strategy template copied and renamed
- [ ] Development workflow clear

## 🎯 Next Steps

1. **Read the Documentation**:
   - `PROJECT_STRUCTURE.md` - Overall project organization
   - `docs/STRATEGY_DEVELOPMENT.md` - Detailed strategy guide
   - `.windsurfrulesfile` - Development rules and guidelines

2. **Explore Examples**:
   - `strategies/demo_cached_strategy.py` - Working demo
   - `strategies/live_ultra_aggressive_0dte.py` - Live strategy example

3. **Start Development**:
   - Copy strategy template
   - Implement your trading logic
   - Test with cached data
   - Run backtests
   - Iterate and improve

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the connectivity test output
3. Examine log files in `strategies/logs/`
4. Verify all prerequisites are met

**Happy Trading! 🚀**
