#!/usr/bin/env python3
"""
Framework Connectivity Test Suite
=================================

This script tests all core components of the Alpaca 0DTE trading framework
to ensure proper connectivity and functionality before development.

Run this script after setting up the environment to validate:
- ThetaData connection
- Alpaca SDK imports
- Strategy framework
- Configuration files
- Directory structure
- Data caching system

Usage:
    conda activate Alpaca_Options
    python test_framework_connectivity.py
"""

import sys
import os
import traceback
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_test_header(test_name: str):
    """Print formatted test header"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {test_name}")
    print(f"{'='*60}")

def print_result(test_name: str, success: bool, message: str = ""):
    """Print test result with formatting"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}")
    if message:
        print(f"   {message}")

def test_environment():
    """Test Python environment and basic imports"""
    print_test_header("Python Environment")
    
    try:
        # Test Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            print_result("Python Version", True, f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            print_result("Python Version", False, f"Python {python_version.major}.{python_version.minor} < 3.8")
            return False
        
        # Test essential imports
        import pandas as pd
        print_result("Pandas Import", True, f"Version {pd.__version__}")
        
        import numpy as np
        print_result("NumPy Import", True, f"Version {np.__version__}")
        
        import requests
        print_result("Requests Import", True, f"Version {requests.__version__}")
        
        return True
        
    except Exception as e:
        print_result("Environment Test", False, str(e))
        return False

def test_alpaca_sdk():
    """Test Alpaca SDK imports and basic functionality"""
    print_test_header("Alpaca SDK")
    
    try:
        # Test core Alpaca imports
        from alpaca.data.historical.stock import StockHistoricalDataClient
        print_result("Alpaca Data Client Import", True)
        
        from alpaca.data.requests import StockBarsRequest
        print_result("Alpaca Requests Import", True)
        
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        print_result("Alpaca TimeFrame Import", True)
        
        # Test broker imports
        try:
            from alpaca.broker.client import BrokerClient
            print_result("Alpaca Broker Client Import", True)
        except ImportError:
            print_result("Alpaca Broker Client Import", False, "Broker client not available (may be normal)")
        
        return True
        
    except Exception as e:
        print_result("Alpaca SDK Test", False, str(e))
        traceback.print_exc()
        return False

def test_thetadata_connection():
    """Test ThetaData connectivity"""
    print_test_header("ThetaData Connection")
    
    try:
        # Import ThetaData connector
        from thetadata.theta_connection.connector import ThetaDataConnector
        print_result("ThetaData Connector Import", True)
        
        # Test connection
        connector = ThetaDataConnector()
        connection_ok = connector.test_connection()
        
        if connection_ok:
            print_result("ThetaData Connection", True, "Connected to ThetaData Terminal")
        else:
            print_result("ThetaData Connection", False, "Cannot connect to ThetaData Terminal")
            print("   Make sure ThetaData Terminal is running on localhost:25510")
        
        return connection_ok
        
    except Exception as e:
        print_result("ThetaData Test", False, str(e))
        print("   Make sure ThetaData files are in the correct location")
        return False

def test_strategy_framework():
    """Test strategy framework components"""
    print_test_header("Strategy Framework")
    
    try:
        # Test base strategy import
        from strategies.base_theta_strategy import BaseThetaStrategy
        print_result("Base Strategy Import", True)
        
        # Test strategy runner import
        from strategies.cached_strategy_runner import CachedStrategyRunner
        print_result("Strategy Runner Import", True)
        
        # Test template import
        from strategies.templates.strategy_template_v1 import StrategyTemplate
        print_result("Strategy Template Import", True)
        
        # Test instantiation
        template = StrategyTemplate("test_strategy")
        print_result("Strategy Template Instantiation", True)
        
        return True
        
    except Exception as e:
        print_result("Strategy Framework Test", False, str(e))
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration files"""
    print_test_header("Configuration")
    
    try:
        # Test trading config import
        from config.trading_config import TradingConfig
        print_result("Trading Config Import", True)
        
        # Test configuration validation
        config_valid = TradingConfig.validate_config()
        print_result("Configuration Validation", config_valid)
        
        # Test strategy config generation
        strategy_config = TradingConfig.get_strategy_config("test_strategy")
        print_result("Strategy Config Generation", True)
        
        return config_valid
        
    except Exception as e:
        print_result("Configuration Test", False, str(e))
        traceback.print_exc()
        return False

def test_directory_structure():
    """Test required directory structure"""
    print_test_header("Directory Structure")
    
    required_dirs = [
        'strategies/logs',
        'strategies/templates',
        'backtrader/results',
        'thetadata/cached_data',
        'thetadata/theta_connection',
        'config',
        'docs'
    ]
    
    all_exist = True
    
    for directory in required_dirs:
        exists = os.path.exists(directory)
        print_result(f"Directory: {directory}", exists)
        if not exists:
            all_exist = False
    
    return all_exist

def test_data_caching():
    """Test data caching system"""
    print_test_header("Data Caching System")
    
    try:
        # Test data collector import
        from thetadata.theta_connection.thetadata_collector import ThetaDataCollector
        print_result("Data Collector Import", True)
        
        # Test instantiation
        collector = ThetaDataCollector("test_cache")
        print_result("Data Collector Instantiation", True)
        
        # Check if cached data exists
        cache_dir = "thetadata/cached_data"
        spy_bars_dir = f"{cache_dir}/spy_bars"
        option_chains_dir = f"{cache_dir}/option_chains"
        
        spy_files = len(os.listdir(spy_bars_dir)) if os.path.exists(spy_bars_dir) else 0
        option_files = len(os.listdir(option_chains_dir)) if os.path.exists(option_chains_dir) else 0
        
        print_result("SPY Bars Cache", spy_files > 0, f"{spy_files} files found")
        print_result("Option Chains Cache", option_files > 0, f"{option_files} files found")
        
        return True
        
    except Exception as e:
        print_result("Data Caching Test", False, str(e))
        return False

def test_backtrader_framework():
    """Test backtrader framework"""
    print_test_header("Backtrader Framework")
    
    try:
        # Test backtrader import
        import backtrader as bt
        print_result("Backtrader Import", True)
        
        # Test backtest files exist
        backtest_files = [
            'backtrader/multi_day_cached_backtest.py',
            'backtrader/run_v2_real_backtest.py'
        ]
        
        all_exist = True
        for file_path in backtest_files:
            exists = os.path.exists(file_path)
            print_result(f"Backtest File: {os.path.basename(file_path)}", exists)
            if not exists:
                all_exist = False
        
        return all_exist
        
    except Exception as e:
        print_result("Backtrader Test", False, str(e))
        return False

def run_comprehensive_test():
    """Run all connectivity tests"""
    print("🎯 ALPACA 0DTE FRAMEWORK CONNECTIVITY TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run all tests
    tests = [
        ("Environment", test_environment),
        ("Alpaca SDK", test_alpaca_sdk),
        ("ThetaData Connection", test_thetadata_connection),
        ("Strategy Framework", test_strategy_framework),
        ("Configuration", test_configuration),
        ("Directory Structure", test_directory_structure),
        ("Data Caching", test_data_caching),
        ("Backtrader Framework", test_backtrader_framework)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_result(test_name, False, f"Unexpected error: {str(e)}")
            results[test_name] = False
    
    # Print summary
    print_test_header("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Framework is ready for development.")
        print("\n🚀 Next steps:")
        print("1. Copy strategy template: cp strategies/templates/strategy_template_v1.py strategies/my_strategy_v1.py")
        print("2. Implement your strategy logic")
        print("3. Test with cached data")
        print("4. Run full backtest")
    else:
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")
        print("\n🔧 Common fixes:")
        print("- Ensure ThetaData Terminal is running")
        print("- Check conda environment is activated: conda activate Alpaca_Options")
        print("- Verify all dependencies are installed")
        print("- Check file permissions and paths")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
