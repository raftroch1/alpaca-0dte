#!/usr/bin/env python3
"""
Debug script to test data loading directly
"""
import sys
import os
sys.path.append('..')
sys.path.append('../thetadata')
sys.path.append('../thetadata/theta_connection')

from thetadata_collector import ThetaDataCollector

def test_data_loading():
    """Test data loading with different cache directory paths"""
    
    # Test absolute path (where we know files exist)
    abs_cache_dir = "/Users/devops/Desktop/coding projects/windsurf/Alpaca_0dte/alpaca-0dte/thetadata/cached_data"
    print(f"🧪 Testing absolute path: {abs_cache_dir}")
    
    collector = ThetaDataCollector(abs_cache_dir)
    
    # Test file path construction
    spy_path = collector.get_cache_path("spy_bars", "20240102")
    option_path = collector.get_cache_path("option_chains", "20240102")
    
    print(f"📁 SPY path: {spy_path}")
    print(f"📁 Option path: {option_path}")
    print(f"✅ SPY exists: {os.path.exists(spy_path)}")
    print(f"✅ Option exists: {os.path.exists(option_path)}")
    
    # Test loading
    print("\n🔄 Testing data loading...")
    spy_data = collector.load_from_cache("spy_bars", "20240102")
    option_data = collector.load_from_cache("option_chains", "20240102")
    
    print(f"📊 SPY data loaded: {spy_data is not None}")
    print(f"📊 Option data loaded: {option_data is not None}")
    
    if spy_data is not None:
        print(f"📈 SPY bars count: {len(spy_data)}")
    if option_data is not None:
        print(f"📋 Option chains count: {len(option_data)}")

if __name__ == "__main__":
    test_data_loading()
