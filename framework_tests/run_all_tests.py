#!/usr/bin/env python3
"""
Framework Test Runner - Organized Testing Suite
===============================================

Runs all framework tests in the proper order with organized structure.
Tests are now properly organized into subdirectories for better maintenance.

Usage:
    python framework_tests/run_all_tests.py
    python framework_tests/run_all_tests.py --quick    # Skip long-running tests
    python framework_tests/run_all_tests.py --alpaca   # Only Alpaca tests
    python framework_tests/run_all_tests.py --strategy # Only strategy tests
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test_script(script_path, description):
    """Run a test script and return success status"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Running: {script_path}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              cwd=project_root, 
                              capture_output=True, 
                              text=True, 
                              timeout=120)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT (>120s)")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run framework tests')
    parser.add_argument('--quick', action='store_true', help='Skip long-running tests')
    parser.add_argument('--alpaca', action='store_true', help='Only run Alpaca tests')
    parser.add_argument('--strategy', action='store_true', help='Only run strategy tests')
    parser.add_argument('--connectivity', action='store_true', help='Only run connectivity tests')
    args = parser.parse_args()
    
    print("🚀 ALPACA 0DTE FRAMEWORK - ORGANIZED TEST SUITE")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print(f"Test Organization: framework_tests/ (separate from core SDK)")
    
    # Define test categories
    test_categories = {
        'connectivity': [
            ('framework_tests/connectivity/test_framework_connectivity.py', 'Framework Connectivity Test')
        ],
        'alpaca': [
            ('framework_tests/alpaca/test_alpaca_api.py', 'Alpaca API Basic Test'),
            ('framework_tests/alpaca/test_alpaca_sdk_proper.py', 'Alpaca SDK Proper Test'),
            ('framework_tests/alpaca/test_trading_specific.py', 'Alpaca Trading Investigation'),
            ('framework_tests/alpaca/diagnose_alpaca_account.py', 'Alpaca Account Diagnostics')
        ],
        'strategy': [
            ('framework_tests/strategy/test_strategy_startup.py', 'Strategy Startup Test')
        ]
    }
    
    # Determine which tests to run
    tests_to_run = []
    
    if args.alpaca:
        tests_to_run.extend(test_categories['alpaca'])
    elif args.strategy:
        tests_to_run.extend(test_categories['strategy'])
    elif args.connectivity:
        tests_to_run.extend(test_categories['connectivity'])
    else:
        # Run all tests in logical order
        tests_to_run.extend(test_categories['connectivity'])
        tests_to_run.extend(test_categories['alpaca'])
        tests_to_run.extend(test_categories['strategy'])
    
    # Skip certain tests in quick mode
    if args.quick:
        tests_to_run = [t for t in tests_to_run if 'diagnose' not in t[0]]
    
    # Run tests
    results = []
    for script_path, description in tests_to_run:
        full_path = project_root / script_path
        if full_path.exists():
            success = run_test_script(str(full_path), description)
            results.append((description, success))
        else:
            print(f"⚠️  Test file not found: {script_path}")
            results.append((description, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {description}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Framework is ready for production!")
        print("\n🚀 Next Steps:")
        print("   1. Start ThetaData Terminal if not running")
        print("   2. Run your live strategy: python strategies/live_ultra_aggressive_0dte.py")
        print("   3. Monitor logs and performance")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        print("   Some failures may be expected (e.g., outside market hours)")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
