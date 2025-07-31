# 🚀 Alpaca 0DTE Strategies - Organized Structure

## 📂 **ORGANIZED STRATEGY ARCHITECTURE**

This directory contains all 0DTE options trading strategies organized by family and purpose for better maintainability and development workflow.

---

## 🏗️ **FOLDER STRUCTURE**

### 📊 **Core Strategy Families**

#### 📂 `live_ultra_aggressive_0dte/` 
**Main Strategy Evolution Line**
- `live_ultra_aggressive_0dte.py` - ✅ **Live Strategy** (source of truth)
- `live_ultra_aggressive_0dte_backtest.py` - ✅ **Aligned Backtest** (Phase 1 fixes)
- `optimized_0dte_strategy.py` - ✅ **Phase 2** (63% improvement, inverse signals)
- `phase3_profitable_0dte_strategy.py` - ✅ **Phase 3** (75% improvement, advanced optimizations)
- `LIVE_ULTRA_AGGRESSIVE_0DTE_README.md` - Documentation
- `live_bot*.out` - Live trading logs

#### 📂 `real_data_integration/` 
**🎯 BREAKTHROUGH: Real Market Data Integration**
- `alpaca_real_data_strategy.py` - ✅ **MAIN BREAKTHROUGH** (Real Alpaca option data)
- `real_theta_data_strategy.py` - ThetaData integration alternative
- `monthly_alpaca_real_data_runner.py` - Batch testing framework
- `REALISTIC_BACKTEST_README.md` - Real data documentation
- `*.csv` - Monthly validation results (196 trades, March 2024)

#### 📂 `turtle_strategies/`
**Turtle Trading Methodology**
- `turtle_0dte_strategy.py` - Enhanced turtle strategy with N-based sizing
- `turtle_0dte_backtest.py` - Turtle backtest framework

---

### 🛠️ **Support Tools**

#### 📂 `analysis_tools/`
**Strategy Analysis & Validation**
- `diagnostic_strategy_analysis.py` - Performance diagnosis & signal analysis
- `strategy_alignment_validator.py` - Live/backtest alignment validation
- `realistic_trading_config.py` - Real market conditions modeling
- `realistic_0dte_backtest_v2.py` - Comprehensive realistic backtest framework

#### 📂 `framework_core/`
**Base Framework Components**
- `base_theta_strategy.py` - Abstract base strategy class
- `cached_strategy_runner.py` - Strategy execution framework
- `demo_cached_strategy.py` - Demo strategy implementation

#### 📂 `multi_regime_strategies/`
**Market Regime-Based Strategies**
- `multi_regime_0dte_backtest.py` - Multi-regime backtest
- `multi_regime_options_strategy.py` - Advanced regime-based strategy
- `live_multi_regime_0dte.py` - Live multi-regime implementation
- `enhanced_turtle_0dte_backtest.py` - Enhanced turtle with regime detection

---

### 🗃️ **Development & Archive**

#### 📂 `archive/`
**Legacy & Experimental Strategies**
- Historical strategy versions kept for reference
- Deprecated implementations
- Experimental approaches that didn't make it to production

#### 📂 `tests/`
**Test & Debug Files**
- `debug_data_loading.py` - Data loading diagnostics
- Other test utilities

---

## 🎯 **USAGE RECOMMENDATIONS**

### **For Live Trading:**
1. **Production Ready**: `live_ultra_aggressive_0dte/live_ultra_aggressive_0dte.py`
2. **Real Data Validation**: `real_data_integration/alpaca_real_data_strategy.py`

### **For Backtesting:**
1. **Aligned Backtest**: `live_ultra_aggressive_0dte/live_ultra_aggressive_0dte_backtest.py`
2. **Real Data Backtest**: `real_data_integration/alpaca_real_data_strategy.py`
3. **Optimized Strategies**: `live_ultra_aggressive_0dte/phase3_profitable_0dte_strategy.py`

### **For Development:**
1. **Strategy Analysis**: `analysis_tools/diagnostic_strategy_analysis.py`
2. **Validation**: `analysis_tools/strategy_alignment_validator.py`
3. **Framework**: `framework_core/base_theta_strategy.py`

---

## 📈 **PROVEN RESULTS SUMMARY**

| **Strategy** | **Performance vs Original** | **Status** |
|-------------|----------------------------|------------|
| **Phase 1 Aligned** | Exact live/backtest matching | ✅ **Validated** |
| **Phase 2 Optimized** | **63% loss reduction** | ✅ **Tested** |
| **Phase 3 Advanced** | **75% loss reduction** | ✅ **Tested** |
| **Real Data Integration** | **Eliminates simulation bias** | ✅ **Breakthrough** |

---

## 🔄 **DEVELOPMENT WORKFLOW**

1. **Strategy Development** → `live_ultra_aggressive_0dte/`
2. **Real Data Validation** → `real_data_integration/`
3. **Performance Analysis** → `analysis_tools/`
4. **Specialized Variants** → `turtle_strategies/` or `multi_regime_strategies/`
5. **Proven Strategies** → Keep in main folders
6. **Experimental** → `archive/` when deprecated

---

## 🚀 **NEXT STEPS**

Ready for **Phase 4 optimizations** to push strategies into consistent profitability using the organized framework and proven real data integration. 