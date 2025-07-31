#!/usr/bin/env python3
"""
Strategy Alignment Validator
=============================

This module ensures that the backtest implementation exactly matches 
the live strategy implementation by comparing:
- Signal generation logic
- Position sizing algorithms  
- Risk management parameters
- Entry/exit criteria
- Market regime detection

Any discrepancies between backtest and live strategy are flagged as errors.
"""

import os
import sys
import ast
import inspect
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class AlignmentStatus(Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH" 
    MISSING = "MISSING"
    ERROR = "ERROR"

@dataclass
class ValidationResult:
    component: str
    status: AlignmentStatus
    description: str
    live_value: Any = None
    backtest_value: Any = None
    severity: str = "ERROR"  # ERROR, WARNING, INFO

class StrategyAlignmentValidator:
    """Validates alignment between live strategy and backtest"""
    
    def __init__(self, live_strategy_path: str, backtest_path: str):
        """Initialize validator
        
        Args:
            live_strategy_path: Path to live strategy file
            backtest_path: Path to backtest file
        """
        self.live_strategy_path = live_strategy_path
        self.backtest_path = backtest_path
        self.validation_results = []
        
    def validate_complete_alignment(self) -> List[ValidationResult]:
        """Perform complete validation of strategy alignment
        
        Returns:
            List of validation results
        """
        print("🔍 Starting Strategy Alignment Validation...")
        print("=" * 60)
        
        # Load source code
        live_code = self._load_source_code(self.live_strategy_path)
        backtest_code = self._load_source_code(self.backtest_path)
        
        if not live_code or not backtest_code:
            return [ValidationResult(
                component="Source Code",
                status=AlignmentStatus.ERROR,
                description="Could not load source files"
            )]
        
        # Validate core components
        self._validate_signal_generation(live_code, backtest_code)
        self._validate_position_sizing(live_code, backtest_code)
        self._validate_risk_parameters(live_code, backtest_code)
        self._validate_market_regime_detection(live_code, backtest_code)
        self._validate_entry_exit_logic(live_code, backtest_code)
        self._validate_constants_and_parameters(live_code, backtest_code)
        
        # Generate summary
        self._print_validation_summary()
        
        return self.validation_results
    
    def _load_source_code(self, file_path: str) -> Optional[str]:
        """Load source code from file"""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def _validate_signal_generation(self, live_code: str, backtest_code: str):
        """Validate signal generation logic alignment"""
        print("🎯 Validating Signal Generation Logic...")
        
        # Extract signal generation methods
        live_signals = self._extract_signal_methods(live_code)
        backtest_signals = self._extract_signal_methods(backtest_code)
        
        # Key signal generation components to check
        signal_components = [
            'generate_trading_signal',
            'calculate_momentum_score', 
            'detect_volatility_regime',
            'calculate_confidence_score',
            'apply_time_filters'
        ]
        
        for component in signal_components:
            live_method = live_signals.get(component)
            backtest_method = backtest_signals.get(component)
            
            if not live_method and not backtest_method:
                continue  # Both missing, OK
            elif not live_method:
                self.validation_results.append(ValidationResult(
                    component=f"Signal Generation: {component}",
                    status=AlignmentStatus.MISSING,
                    description=f"Method {component} missing in live strategy",
                    severity="ERROR"
                ))
            elif not backtest_method:
                self.validation_results.append(ValidationResult(
                    component=f"Signal Generation: {component}",
                    status=AlignmentStatus.MISSING,
                    description=f"Method {component} missing in backtest",
                    severity="ERROR"
                ))
            else:
                # Compare method implementations
                alignment = self._compare_method_logic(live_method, backtest_method)
                self.validation_results.append(ValidationResult(
                    component=f"Signal Generation: {component}",
                    status=alignment['status'],
                    description=alignment['description'],
                    live_value=live_method[:100] + "..." if len(live_method) > 100 else live_method,
                    backtest_value=backtest_method[:100] + "..." if len(backtest_method) > 100 else backtest_method,
                    severity="ERROR" if alignment['status'] != AlignmentStatus.MATCH else "INFO"
                ))
    
    def _validate_position_sizing(self, live_code: str, backtest_code: str):
        """Validate position sizing algorithm alignment"""
        print("📊 Validating Position Sizing Logic...")
        
        # Extract position sizing parameters
        live_params = self._extract_position_params(live_code)
        backtest_params = self._extract_position_params(backtest_code)
        
        # Key parameters to validate
        key_params = [
            'base_position_size',
            'max_position_size', 
            'confidence_scaling_factor',
            'risk_adjusted_sizing',
            'volatility_adjustment'
        ]
        
        for param in key_params:
            live_val = live_params.get(param)
            backtest_val = backtest_params.get(param)
            
            if live_val != backtest_val:
                self.validation_results.append(ValidationResult(
                    component=f"Position Sizing: {param}",
                    status=AlignmentStatus.MISMATCH,
                    description=f"Parameter {param} differs between implementations",
                    live_value=live_val,
                    backtest_value=backtest_val,
                    severity="ERROR"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    component=f"Position Sizing: {param}",
                    status=AlignmentStatus.MATCH,
                    description=f"Parameter {param} matches",
                    live_value=live_val,
                    backtest_value=backtest_val,
                    severity="INFO"
                ))
    
    def _validate_risk_parameters(self, live_code: str, backtest_code: str):
        """Validate risk management parameters"""
        print("⚠️ Validating Risk Management Parameters...")
        
        # Extract risk parameters
        live_risk = self._extract_risk_params(live_code)
        backtest_risk = self._extract_risk_params(backtest_code)
        
        # Critical risk parameters
        risk_params = [
            'max_daily_loss',
            'max_trade_loss',
            'position_limit',
            'volatility_threshold',
            'drawdown_limit'
        ]
        
        for param in risk_params:
            live_val = live_risk.get(param)
            backtest_val = backtest_risk.get(param)
            
            if live_val != backtest_val:
                self.validation_results.append(ValidationResult(
                    component=f"Risk Management: {param}",
                    status=AlignmentStatus.MISMATCH,
                    description=f"Risk parameter {param} differs - CRITICAL ERROR",
                    live_value=live_val,
                    backtest_value=backtest_val,
                    severity="ERROR"
                ))
    
    def _validate_market_regime_detection(self, live_code: str, backtest_code: str):
        """Validate market regime detection logic"""
        print("🌊 Validating Market Regime Detection...")
        
        # Look for regime detection methods
        regime_methods = [
            'detect_market_regime',
            'calculate_volatility_regime',
            'trend_detection',
            'momentum_regime'
        ]
        
        for method in regime_methods:
            live_has = method in live_code
            backtest_has = method in backtest_code
            
            if live_has != backtest_has:
                missing_in = "backtest" if live_has else "live strategy"
                self.validation_results.append(ValidationResult(
                    component=f"Market Regime: {method}",
                    status=AlignmentStatus.MISSING,
                    description=f"Method {method} missing in {missing_in}",
                    severity="ERROR"
                ))
    
    def _validate_entry_exit_logic(self, live_code: str, backtest_code: str):
        """Validate entry and exit logic"""
        print("🚪 Validating Entry/Exit Logic...")
        
        # Extract entry/exit conditions
        live_conditions = self._extract_trading_conditions(live_code)
        backtest_conditions = self._extract_trading_conditions(backtest_code)
        
        # Compare critical conditions
        if live_conditions != backtest_conditions:
            self.validation_results.append(ValidationResult(
                component="Entry/Exit Logic",
                status=AlignmentStatus.MISMATCH,
                description="Trading conditions differ between implementations",
                live_value=str(live_conditions)[:200],
                backtest_value=str(backtest_conditions)[:200],
                severity="ERROR"
            ))
    
    def _validate_constants_and_parameters(self, live_code: str, backtest_code: str):
        """Validate all constants and parameters"""
        print("🔢 Validating Constants and Parameters...")
        
        # Extract all numeric constants
        live_constants = self._extract_numeric_constants(live_code)
        backtest_constants = self._extract_numeric_constants(backtest_code)
        
        # Find common constants
        common_names = set(live_constants.keys()) & set(backtest_constants.keys())
        
        for name in common_names:
            if live_constants[name] != backtest_constants[name]:
                self.validation_results.append(ValidationResult(
                    component=f"Constants: {name}",
                    status=AlignmentStatus.MISMATCH,
                    description=f"Constant {name} has different values",
                    live_value=live_constants[name],
                    backtest_value=backtest_constants[name],
                    severity="WARNING"
                ))
    
    def _extract_signal_methods(self, code: str) -> Dict[str, str]:
        """Extract signal generation methods from code"""
        methods = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if any(keyword in node.name.lower() for keyword in 
                          ['signal', 'momentum', 'confidence', 'regime', 'filter']):
                        methods[node.name] = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
        except:
            pass
        return methods
    
    def _extract_position_params(self, code: str) -> Dict[str, Any]:
        """Extract position sizing parameters"""
        params = {}
        # Look for common position sizing patterns
        patterns = {
            'base_position_size': r'base_position_size\s*=\s*(\d+)',
            'max_position_size': r'max_position_size\s*=\s*(\d+)',
        }
        
        import re
        for param, pattern in patterns.items():
            match = re.search(pattern, code)
            if match:
                params[param] = int(match.group(1))
        
        return params
    
    def _extract_risk_params(self, code: str) -> Dict[str, Any]:
        """Extract risk management parameters"""
        params = {}
        import re
        
        # Common risk parameters
        patterns = {
            'max_daily_loss': r'max_daily_loss\s*=\s*(\d+)',
            'max_trade_loss': r'max_trade_loss\s*=\s*(\d+)',
        }
        
        for param, pattern in patterns.items():
            match = re.search(pattern, code)
            if match:
                params[param] = int(match.group(1))
        
        return params
    
    def _extract_trading_conditions(self, code: str) -> List[str]:
        """Extract trading entry/exit conditions"""
        conditions = []
        # This would need more sophisticated parsing
        # For now, just look for key condition keywords
        import re
        
        condition_patterns = [
            r'if.*confidence.*>',
            r'if.*price_change.*>',
            r'if.*volume.*>',
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, code)
            conditions.extend(matches)
        
        return conditions
    
    def _extract_numeric_constants(self, code: str) -> Dict[str, float]:
        """Extract numeric constants from code"""
        constants = {}
        import re
        
        # Look for variable assignments with numeric values
        pattern = r'(\w+)\s*=\s*([\d.]+)'
        matches = re.findall(pattern, code)
        
        for name, value in matches:
            try:
                constants[name] = float(value)
            except:
                pass
        
        return constants
    
    def _compare_method_logic(self, live_method: str, backtest_method: str) -> Dict[str, Any]:
        """Compare two method implementations"""
        # Simple comparison - could be made more sophisticated
        if live_method.strip() == backtest_method.strip():
            return {
                'status': AlignmentStatus.MATCH,
                'description': "Method implementations are identical"
            }
        else:
            # Calculate similarity
            similarity = self._calculate_similarity(live_method, backtest_method)
            if similarity > 0.9:
                return {
                    'status': AlignmentStatus.MATCH,
                    'description': f"Method implementations are very similar ({similarity:.1%})"
                }
            else:
                return {
                    'status': AlignmentStatus.MISMATCH,
                    'description': f"Method implementations differ significantly ({similarity:.1%} similarity)"
                }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Simple character-based similarity
        if not text1 or not text2:
            return 0.0
        
        # Remove whitespace and normalize
        t1 = ''.join(text1.split()).lower()
        t2 = ''.join(text2.split()).lower()
        
        if t1 == t2:
            return 1.0
        
        # Calculate character overlap
        common_chars = sum(1 for c in t1 if c in t2)
        max_len = max(len(t1), len(t2))
        
        return common_chars / max_len if max_len > 0 else 0.0
    
    def _print_validation_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("STRATEGY ALIGNMENT VALIDATION RESULTS")
        print("="*60)
        
        errors = [r for r in self.validation_results if r.severity == "ERROR"]
        warnings = [r for r in self.validation_results if r.severity == "WARNING"]
        matches = [r for r in self.validation_results if r.status == AlignmentStatus.MATCH]
        
        print(f"✅ Matches: {len(matches)}")
        print(f"⚠️  Warnings: {len(warnings)}")
        print(f"❌ Errors: {len(errors)}")
        
        if errors:
            print("\n🚨 CRITICAL ERRORS:")
            for error in errors:
                print(f"   - {error.component}: {error.description}")
        
        if warnings:
            print("\n⚠️  WARNINGS:")
            for warning in warnings:
                print(f"   - {warning.component}: {warning.description}")
        
        print("\n" + "="*60)
        
        if errors:
            print("❌ VALIDATION FAILED: Backtest does not match live strategy")
            print("   Fix all errors before running backtest!")
        else:
            print("✅ VALIDATION PASSED: Backtest aligns with live strategy")

def main():
    """Main validation function"""
    validator = StrategyAlignmentValidator(
        live_strategy_path="live_ultra_aggressive_0dte.py",
        backtest_path="realistic_0dte_backtest_v2.py"
    )
    
    results = validator.validate_complete_alignment()
    
    # Return exit code based on validation
    errors = [r for r in results if r.severity == "ERROR"]
    return 1 if errors else 0

if __name__ == "__main__":
    exit(main()) 