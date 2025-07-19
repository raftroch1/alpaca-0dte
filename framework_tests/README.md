# Framework Testing Suite - Organized Structure

This directory contains all testing and validation scripts for the Alpaca 0DTE Options Trading Framework, properly organized and separate from the core Alpaca SDK.

## 📁 Test Organization

### `/connectivity/` - Framework Connectivity Tests
Tests that validate the overall framework connectivity and integration between components.

- **`test_framework_connectivity.py`** - Comprehensive framework connectivity validation
  - Tests Python environment and dependencies
  - Validates Alpaca SDK integration
  - Checks ThetaData connection and Terminal status
  - Verifies strategy framework components
  - Tests configuration loading and validation
  - Validates directory structure and cached data

### `/alpaca/` - Alpaca API & SDK Tests
Tests specifically focused on Alpaca API connectivity, authentication, and SDK functionality.

- **`test_alpaca_api.py`** - Basic Alpaca API endpoint testing
- **`test_alpaca_sdk_proper.py`** - Proper Alpaca SDK usage validation
- **`test_trading_specific.py`** - Trading-specific API investigation
- **`diagnose_alpaca_account.py`** - Account diagnostics and troubleshooting

### `/strategy/` - Strategy Testing
Tests for strategy initialization, components, and functionality.

- **`test_strategy_startup.py`** - Strategy initialization and startup validation

## 🚀 Running Tests

### Quick Test Suite
```bash
# Run all tests
python framework_tests/run_all_tests.py

# Quick mode (skip long-running tests)
python framework_tests/run_all_tests.py --quick
```

### Category-Specific Tests
```bash
# Only connectivity tests
python framework_tests/run_all_tests.py --connectivity

# Only Alpaca tests
python framework_tests/run_all_tests.py --alpaca

# Only strategy tests
python framework_tests/run_all_tests.py --strategy
```

### Individual Tests
```bash
# Framework connectivity
python framework_tests/connectivity/test_framework_connectivity.py

# Strategy startup
python framework_tests/strategy/test_strategy_startup.py

# Alpaca diagnostics
python framework_tests/alpaca/diagnose_alpaca_account.py
```

## 📊 Test Categories Explained

### 1. Connectivity Tests
These tests validate that all framework components can communicate properly:
- ThetaData Terminal connection
- Alpaca API authentication
- Strategy framework imports
- Configuration loading
- Data cache accessibility

### 2. Alpaca Tests
These tests focus on Alpaca-specific functionality:
- API key validation
- SDK initialization
- Trading client connectivity
- Data client functionality
- Account status verification

### 3. Strategy Tests
These tests validate strategy-specific functionality:
- Strategy class initialization
- Parameter loading
- Risk management systems
- Market hours detection
- Technical indicator calculations

## 🎯 Expected Results

### ✅ Successful Test Run
When all tests pass, you should see:
- Framework connectivity: ✅ Working
- Alpaca integration: ✅ Connected
- Strategy components: ✅ Functional
- ThetaData: ✅ Connected (when Terminal is running)

### ⚠️ Expected Warnings
Some tests may show warnings outside market hours:
- SPY data retrieval may fail (normal outside trading hours)
- Market status will show "closed" (expected)
- Some Alpaca endpoints may return 403 (normal for certain operations)

## 🔧 Organization Benefits

This organized structure provides:
- **Clear separation** from core Alpaca SDK files
- **Logical categorization** of different test types
- **Easy maintenance** and updates
- **Scalable structure** for adding new tests
- **Compliance** with framework rules (no core file modifications)

## 📝 Adding New Tests

When adding new tests:
1. Place them in the appropriate category folder
2. Follow the naming convention: `test_*.py`
3. Update the `run_all_tests.py` script if needed
4. Document the test purpose in this README
