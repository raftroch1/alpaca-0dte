# 🛠️ Alpaca 0DTE Tools Collection

## 📂 **ORGANIZED TOOLS ARCHITECTURE**

Essential tools for development, debugging, production, and maintenance of the Alpaca 0DTE trading framework.

---

## 🏭 **PRODUCTION TOOLS** (`production/`)

### 🔄 `run_strategy_daemon.sh`
**Production-grade daemon runner for live trading**
- ✅ Runs strategies as background services
- ✅ Automatic restart on failures  
- ✅ Process monitoring with PID management
- ✅ Comprehensive logging
- ✅ Conda environment activation

**Usage:**
```bash
# Start daemon
./tools/production/run_strategy_daemon.sh start

# Stop daemon  
./tools/production/run_strategy_daemon.sh stop

# Check status
./tools/production/run_strategy_daemon.sh status
```

### 📊 `monitor_signals.py`
**Real-time strategy monitoring and diagnostics**
- ✅ Live signal generation tracking
- ✅ Trading decision monitoring
- ✅ Risk management action alerts
- ✅ Performance metrics display
- ✅ Log file analysis

**Usage:**
```bash
python tools/production/monitor_signals.py
```

---

## 🔍 **DEBUG TOOLS** (`debug/`)

### 📈 `debug_spy_data.py`
**SPY data retrieval diagnostics**
- ✅ Tests exact SPY data access like live strategy
- ✅ API connectivity validation
- ✅ Data quality verification
- ✅ Timestamp and frequency analysis
- ✅ Signal generation troubleshooting

**Usage:**
```bash
python tools/debug/debug_spy_data.py
```

### 🔧 `test_live_alpaca_data.py`
**Live market data troubleshooting**
- ✅ Market hours data verification
- ✅ 403 error diagnosis
- ✅ Real-time data access testing
- ✅ Subscription validation
- ✅ Data feed troubleshooting

**Usage:**
```bash
python tools/debug/test_live_alpaca_data.py
```

---

## ⚙️ **SETUP TOOLS** (`setup/`)

### 🔑 `test_api.py`
**Alpaca API connectivity validation**
- ✅ API key verification
- ✅ Trading permissions check
- ✅ Option contract access test
- ✅ Account status validation
- ✅ Environment configuration test

**Usage:**
```bash
python tools/setup/test_api.py
```

---

## 🔄 **ALTERNATIVE APPROACHES** (`alternatives/`)

### 💰 `test_delayed_data_strategy.py`
**Free delayed data testing (15-min delay)**
- ✅ Cost-effective strategy testing
- ✅ No real-time data subscription needed
- ✅ Strategy logic validation
- ✅ Development environment testing
- ✅ Educational purposes

**Usage:**
```bash
python tools/alternatives/test_delayed_data_strategy.py
```

---

## 📜 **SCRIPTS** (`scripts/`)

### 📖 `generate-docs.sh`
**Documentation generation script**
- ✅ Automated documentation building
- ✅ API reference generation
- ✅ Strategy documentation updates

---

## 🎯 **USAGE SCENARIOS**

### **🚀 Production Deployment:**
1. **Setup**: `tools/setup/test_api.py` - Validate environment
2. **Launch**: `tools/production/run_strategy_daemon.sh start` - Start daemon
3. **Monitor**: `tools/production/monitor_signals.py` - Watch activity

### **🐛 Debugging Issues:**
1. **Data Issues**: `tools/debug/debug_spy_data.py` - Check SPY data
2. **Live Data**: `tools/debug/test_live_alpaca_data.py` - Test real-time access
3. **API Issues**: `tools/setup/test_api.py` - Validate connectivity

### **💡 Development & Testing:**
1. **Free Testing**: `tools/alternatives/test_delayed_data_strategy.py`
2. **Environment Setup**: `tools/setup/test_api.py`
3. **Monitor Development**: `tools/production/monitor_signals.py`

---

## 🏆 **TOOL BENEFITS**

### ✅ **Production Ready:**
- Robust daemon management for 24/7 trading
- Real-time monitoring and alerting
- Automatic restart capabilities

### ✅ **Development Friendly:**
- Comprehensive debugging utilities
- Cost-effective testing options
- Easy environment validation

### ✅ **Maintainable:**
- Organized by purpose and usage
- Clear documentation and examples
- Reusable across different strategies

---

## 🔧 **REQUIREMENTS**

- **Python Environment**: Alpaca_Options conda environment
- **API Keys**: Alpaca API credentials in `.env`
- **Permissions**: Option trading enabled on Alpaca account
- **System**: macOS/Linux for shell scripts

---

## 🚀 **NEXT STEPS**

These tools provide a complete ecosystem for:
1. **Production trading** with monitoring and reliability
2. **Development debugging** with comprehensive diagnostics  
3. **Cost-effective testing** with alternative data sources
4. **Easy deployment** with automated setup validation

Ready to support all phases of the 0DTE strategy development and deployment! 🎯 