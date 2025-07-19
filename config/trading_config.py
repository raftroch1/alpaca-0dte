#!/usr/bin/env python3
"""
Trading Configuration for Alpaca 0DTE Framework
===============================================

Centralized configuration for all trading strategies and connections.
This file contains default settings that can be overridden by individual strategies.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TradingConfig:
    """Centralized configuration for the trading framework"""
    
    # 🔌 ThetaData Configuration
    THETADATA_CONFIG = {
        'base_url': 'http://127.0.0.1:25510',
        'timeout': 10,
        'max_retries': 3,
        'retry_delay': 1.0,
        'cache_enabled': True
    }
    
    # 📊 Alpaca Configuration
    ALPACA_CONFIG = {
        'api_key': os.getenv('ALPACA_API_KEY'),
        'secret_key': os.getenv('ALPACA_SECRET_KEY'),
        'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),  # Paper trading by default
        'data_url': os.getenv('ALPACA_DATA_URL', 'https://data.alpaca.markets')
    }
    
    # 🎯 Strategy Default Parameters
    STRATEGY_DEFAULTS = {
        'symbol': 'SPY',
        'max_daily_trades': 15,
        'confidence_threshold': 0.55,
        'min_option_price': 0.50,
        'max_option_price': 3.00,
        'position_size_pct': 0.02,  # 2% of portfolio per trade
        'max_portfolio_risk': 0.20,  # 20% max portfolio risk
        'stop_loss_pct': 0.50,  # 50% stop loss
        'take_profit_pct': 1.00,  # 100% take profit
    }
    
    # 📁 Directory Configuration
    DIRECTORIES = {
        'cache_dir': 'thetadata/cached_data',
        'logs_dir': 'strategies/logs',
        'results_dir': 'backtrader/results',
        'templates_dir': 'strategies/templates'
    }
    
    # 📝 Logging Configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_enabled': True,
        'console_enabled': True,
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }
    
    # ⏰ Market Hours Configuration (Eastern Time)
    MARKET_HOURS = {
        'market_open': '09:30',
        'market_close': '16:00',
        'pre_market_start': '04:00',
        'after_hours_end': '20:00'
    }
    
    # 🔄 Data Collection Settings
    DATA_COLLECTION = {
        'spy_timeframe': '1Min',
        'option_expiry_days': [0, 1, 2, 3, 7, 14, 30],  # DTE options to collect
        'strikes_range': 50,  # Number of strikes above/below current price
        'cache_compression': True,
        'cache_format': 'pickle'
    }
    
    # 🧪 Backtesting Configuration
    BACKTEST_CONFIG = {
        'initial_cash': 100000,  # $100k starting capital
        'commission': 0.65,  # Per contract commission
        'slippage': 0.01,  # 1 cent slippage
        'margin_requirement': 0.25,  # 25% margin requirement
        'risk_free_rate': 0.05  # 5% risk-free rate for metrics
    }
    
    # 🚨 Risk Management
    RISK_MANAGEMENT = {
        'max_position_size': 0.05,  # 5% max position size
        'max_daily_loss': 0.02,  # 2% max daily loss
        'max_drawdown': 0.10,  # 10% max drawdown
        'correlation_limit': 0.7,  # Max correlation between positions
        'var_confidence': 0.95,  # VaR confidence level
        'stress_test_enabled': True
    }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        errors = []
        
        # Check Alpaca credentials
        if not cls.ALPACA_CONFIG['api_key']:
            errors.append("ALPACA_API_KEY environment variable not set")
        if not cls.ALPACA_CONFIG['secret_key']:
            errors.append("ALPACA_SECRET_KEY environment variable not set")
        
        # Check directories exist
        for dir_name, dir_path in cls.DIRECTORIES.items():
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"✅ Created directory: {dir_path}")
                except Exception as e:
                    errors.append(f"Cannot create directory {dir_path}: {e}")
        
        if errors:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("✅ Configuration validation passed")
        return True
    
    @classmethod
    def get_strategy_config(cls, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy"""
        import pandas as pd
        
        config = {
            'strategy_name': strategy_name,
            'thetadata': cls.THETADATA_CONFIG.copy(),
            'alpaca': cls.ALPACA_CONFIG.copy(),
            'defaults': cls.STRATEGY_DEFAULTS.copy(),
            'directories': cls.DIRECTORIES.copy(),
            'logging': cls.LOGGING_CONFIG.copy(),
            'market_hours': cls.MARKET_HOURS.copy(),
            'backtest': cls.BACKTEST_CONFIG.copy(),
            'risk_management': cls.RISK_MANAGEMENT.copy()
        }
        
        # Add strategy-specific log file path
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        config['log_file'] = f"{cls.DIRECTORIES['logs_dir']}/{strategy_name}_{timestamp}.log"
        config['csv_file'] = f"{cls.DIRECTORIES['logs_dir']}/{strategy_name}_{timestamp}_trades.csv"
        
        return config
    
    @classmethod
    def print_config_summary(cls):
        """Print a summary of the current configuration"""
        print("\n" + "="*60)
        print("🎯 ALPACA 0DTE TRADING FRAMEWORK CONFIGURATION")
        print("="*60)
        print(f"📊 ThetaData Endpoint: {cls.THETADATA_CONFIG['base_url']}")
        print(f"🏦 Alpaca Environment: {'PAPER' if 'paper' in cls.ALPACA_CONFIG['base_url'] else 'LIVE'}")
        print(f"💰 Initial Capital: ${cls.BACKTEST_CONFIG['initial_cash']:,}")
        print(f"📈 Max Daily Trades: {cls.STRATEGY_DEFAULTS['max_daily_trades']}")
        print(f"🎯 Default Symbol: {cls.STRATEGY_DEFAULTS['symbol']}")
        print(f"📁 Cache Directory: {cls.DIRECTORIES['cache_dir']}")
        print(f"📝 Logs Directory: {cls.DIRECTORIES['logs_dir']}")
        print("="*60)

if __name__ == "__main__":
    # Test configuration
    TradingConfig.print_config_summary()
    TradingConfig.validate_config()
