#!/usr/bin/env python3
"""
Environment Setup Script for Proven Strategies

This script helps set up the required environment variables for running
the proven strategies from any location without import issues.
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up environment variables for proven strategies"""
    
    # Get the project root (assuming this script is in strategies/proven_strategies/)
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent.parent.parent  # Go up 3 levels to reach Alpaca_0dte
    
    # Define the key paths
    theta_cache_dir = project_root / "alpaca-0dte" / "thetadata" / "cached_data"
    
    print("🔧 Setting up environment variables for proven strategies...")
    print(f"📁 Project root: {project_root}")
    print(f"📊 ThetaData cache: {theta_cache_dir}")
    
    # Set environment variables
    os.environ['THETA_CACHE_DIR'] = str(theta_cache_dir)
    os.environ['PROJECT_ROOT'] = str(project_root)
    
    # Verify the paths exist
    if not theta_cache_dir.exists():
        print(f"⚠️  Warning: ThetaData cache directory doesn't exist: {theta_cache_dir}")
        print("   Make sure you have the thetadata folder with cached_data")
    else:
        print("✅ ThetaData cache directory found")
    
    # Check for .env file
    env_file = project_root / "alpaca-0dte" / ".env"
    if env_file.exists():
        print("✅ .env file found for Alpaca API keys")
    else:
        print("⚠️  Warning: .env file not found. Make sure you have Alpaca API keys configured")
    
    print("\n🎯 Environment setup complete!")
    print("\nEnvironment variables set:")
    print(f"  THETA_CACHE_DIR = {os.environ.get('THETA_CACHE_DIR')}")
    print(f"  PROJECT_ROOT = {os.environ.get('PROJECT_ROOT')}")
    
    return True

def generate_env_script():
    """Generate a shell script to set environment variables"""
    
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent.parent.parent
    theta_cache_dir = project_root / "alpaca-0dte" / "thetadata" / "cached_data"
    
    script_content = f"""#!/bin/bash
# Environment setup for proven strategies
# Source this file: source setup_env.sh

export THETA_CACHE_DIR="{theta_cache_dir}"
export PROJECT_ROOT="{project_root}"

echo "✅ Environment variables set for proven strategies"
echo "  THETA_CACHE_DIR = $THETA_CACHE_DIR"
echo "  PROJECT_ROOT = $PROJECT_ROOT"
"""
    
    script_path = current_dir / "setup_env.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(script_path, 0o755)
    
    print(f"📝 Generated shell script: {script_path}")
    print("   Usage: source setup_env.sh")

if __name__ == "__main__":
    # Auto-setup when run directly
    setup_environment()
    generate_env_script()
    
    print("\n🚀 Now you can run any proven strategy from any location!")
    print("\nExample usage:")
    print("  cd strategies/proven_strategies/unified_long_iron_condor_25k/")
    print("  python strategy.py --start-date 20240301 --end-date 20240705")