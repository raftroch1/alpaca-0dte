#!/usr/bin/env python3
"""
🔧 PHASE 4D: QUICK SETUP
=======================

Quick setup for Phase 4D paper trading environment.
"""

import os

def create_sample_env():
    """Create sample .env file"""
    env_content = """# Alpaca Paper Trading API Credentials
# Get these from: https://app.alpaca.markets/paper/dashboard/overview
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here

# Phase 4D Strategy Settings
DAILY_TARGET=365.00
MAX_TRADES=8
MAX_LOSS=2000.00
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env template")
        print("📝 Please edit .env with your actual Alpaca paper trading credentials")
        return False
    else:
        print("✅ .env file already exists")
        return True

def main():
    """Quick setup main function"""
    print("🔧 PHASE 4D: QUICK SETUP")
    print("=" * 30)
    
    # Create .env if needed
    env_exists = create_sample_env()
    
    print("\n📦 Required packages:")
    print("   pip install pandas numpy alpaca-py python-dotenv")
    
    print("\n🌐 Get Alpaca Paper Trading credentials:")
    print("   https://app.alpaca.markets/paper/dashboard/overview")
    
    if env_exists:
        print("\n🚀 Ready to start paper trading!")
        print("   python start_paper_trading.py")
    else:
        print("\n📝 Next steps:")
        print("   1. Edit .env with your actual credentials")
        print("   2. Run: python start_paper_trading.py")
    
    print("=" * 30)

if __name__ == "__main__":
    main()
