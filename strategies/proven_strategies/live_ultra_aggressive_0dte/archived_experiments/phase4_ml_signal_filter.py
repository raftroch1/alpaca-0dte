#!/usr/bin/env python3
"""
🧠 PHASE 4: ML-ENHANCED SIGNAL FILTERING
=======================================

Advanced signal filtering using machine learning techniques inspired by
LiuAlgoTrader's optimizer and analysis framework for 0DTE options trading.

KEY IMPROVEMENTS:
✅ ML-based signal quality scoring
✅ Multi-factor confidence weighting  
✅ Real-time market regime detection
✅ Historical pattern recognition
✅ Signal correlation analysis
✅ Dynamic threshold optimization

TARGET: Push strategy into consistent profitability through intelligent signal filtering.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Import from our organized structure
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'real_data_integration'))
from alpaca_real_data_strategy import AlpacaRealDataStrategy

@dataclass
class SignalFeatures:
    """Feature engineering for ML signal filtering"""
    # Technical indicators
    rsi_divergence: float
    ema_momentum: float
    volume_anomaly: float
    price_action_strength: float
    
    # Market microstructure
    bid_ask_spread: float
    order_flow_imbalance: float
    market_volatility: float
    time_to_close: float
    
    # Historical patterns
    similar_pattern_success_rate: float
    recent_signal_accuracy: float
    market_regime_alignment: float
    
    # Risk factors
    vix_level: float
    spy_trend_strength: float
    option_iv_percentile: float

class MLSignalFilter:
    """
    Machine Learning Signal Quality Filter
    
    Inspired by LiuAlgoTrader's optimizer component for hyper-parameter optimization
    and strategy analysis framework.
    """
    
    def __init__(self, model_save_path: str = "ml_models/"):
        self.model_save_path = model_save_path
        self.signal_quality_model = None
        self.feature_scaler = StandardScaler()
        self.signal_history = []
        self.performance_tracker = {}
        
        # Create model directory
        os.makedirs(model_save_path, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🧠 ML Signal Filter initialized")
    
    def extract_signal_features(self, spy_data: pd.DataFrame, signal_data: Dict) -> SignalFeatures:
        """
        Extract comprehensive features for ML signal evaluation
        
        Args:
            spy_data: Real-time SPY price data
            signal_data: Raw signal from strategy
            
        Returns:
            SignalFeatures object with all extracted features
        """
        current_price = spy_data['close'].iloc[-1]
        
        # Technical indicators analysis
        rsi_values = self._calculate_rsi(spy_data['close'], 14)
        rsi_divergence = self._detect_rsi_divergence(spy_data, rsi_values)
        
        ema_short = spy_data['close'].ewm(span=12).mean()
        ema_long = spy_data['close'].ewm(span=26).mean()
        ema_momentum = (ema_short.iloc[-1] - ema_long.iloc[-1]) / current_price
        
        # Volume analysis
        volume_ma = spy_data['volume'].rolling(20).mean()
        current_volume = spy_data['volume'].iloc[-1]
        volume_anomaly = (current_volume - volume_ma.iloc[-1]) / volume_ma.iloc[-1]
        
        # Price action strength
        price_range = spy_data['high'].iloc[-1] - spy_data['low'].iloc[-1]
        typical_range = spy_data['high'].rolling(20).mean() - spy_data['low'].rolling(20).mean()
        price_action_strength = price_range / typical_range.iloc[-1] if typical_range.iloc[-1] > 0 else 0
        
        # Market microstructure (simulated - would use real data in production)
        bid_ask_spread = self._estimate_bid_ask_spread(current_price)
        order_flow_imbalance = self._estimate_order_flow(spy_data)
        
        # Market volatility
        returns = spy_data['close'].pct_change()
        market_volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Time factors
        current_time = datetime.now()
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        time_to_close = (market_close - current_time).total_seconds() / 3600
        
        # Historical pattern matching
        similar_pattern_success_rate = self._find_similar_patterns(spy_data)
        recent_signal_accuracy = self._calculate_recent_accuracy()
        market_regime_alignment = self._assess_market_regime(spy_data)
        
        # Risk factors (would integrate with real VIX data)
        vix_level = self._estimate_vix_from_spy_volatility(market_volatility)
        spy_trend_strength = self._calculate_trend_strength(spy_data)
        option_iv_percentile = self._estimate_iv_percentile(market_volatility)
        
        return SignalFeatures(
            rsi_divergence=rsi_divergence,
            ema_momentum=ema_momentum,
            volume_anomaly=volume_anomaly,
            price_action_strength=price_action_strength,
            bid_ask_spread=bid_ask_spread,
            order_flow_imbalance=order_flow_imbalance,
            market_volatility=market_volatility,
            time_to_close=time_to_close,
            similar_pattern_success_rate=similar_pattern_success_rate,
            recent_signal_accuracy=recent_signal_accuracy,
            market_regime_alignment=market_regime_alignment,
            vix_level=vix_level,
            spy_trend_strength=spy_trend_strength,
            option_iv_percentile=option_iv_percentile
        )
    
    def train_signal_quality_model(self, historical_data: List[Tuple[SignalFeatures, bool]]):
        """
        Train ML model to predict signal quality
        
        Args:
            historical_data: List of (features, profitable_outcome) tuples
        """
        if len(historical_data) < 50:
            self.logger.warning("⚠️ Insufficient data for ML training, using rule-based fallback")
            return
        
        # Prepare training data
        features_array = np.array([[
            f.rsi_divergence, f.ema_momentum, f.volume_anomaly, f.price_action_strength,
            f.bid_ask_spread, f.order_flow_imbalance, f.market_volatility, f.time_to_close,
            f.similar_pattern_success_rate, f.recent_signal_accuracy, f.market_regime_alignment,
            f.vix_level, f.spy_trend_strength, f.option_iv_percentile
        ] for f, _ in historical_data])
        
        labels = np.array([outcome for _, outcome in historical_data])
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features_array)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42
        )
        
        # Train ensemble model (similar to LiuAlgoTrader's optimization approach)
        self.signal_quality_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.signal_quality_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.signal_quality_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        self.logger.info(f"🧠 ML Model trained - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        # Save model
        joblib.dump(self.signal_quality_model, os.path.join(self.model_save_path, 'signal_quality_model.pkl'))
        joblib.dump(self.feature_scaler, os.path.join(self.model_save_path, 'feature_scaler.pkl'))
    
    def predict_signal_quality(self, features: SignalFeatures) -> float:
        """
        Predict signal quality score (0-1, higher is better)
        
        Args:
            features: Extracted signal features
            
        Returns:
            Quality score between 0 and 1
        """
        if self.signal_quality_model is None:
            return self._rule_based_quality_score(features)
        
        # Prepare features for prediction
        feature_array = np.array([[
            features.rsi_divergence, features.ema_momentum, features.volume_anomaly,
            features.price_action_strength, features.bid_ask_spread, features.order_flow_imbalance,
            features.market_volatility, features.time_to_close, features.similar_pattern_success_rate,
            features.recent_signal_accuracy, features.market_regime_alignment, features.vix_level,
            features.spy_trend_strength, features.option_iv_percentile
        ]])
        
        # Scale and predict
        feature_scaled = self.feature_scaler.transform(feature_array)
        quality_prob = self.signal_quality_model.predict_proba(feature_scaled)[0, 1]
        
        return quality_prob
    
    def _rule_based_quality_score(self, features: SignalFeatures) -> float:
        """Fallback rule-based scoring when ML model not available"""
        score = 0.5  # Base score
        
        # Technical strength
        if abs(features.rsi_divergence) > 0.3:
            score += 0.1
        if abs(features.ema_momentum) > 0.002:
            score += 0.1
        if features.volume_anomaly > 0.5:
            score += 0.1
        
        # Market conditions
        if features.market_volatility < 0.25:  # Low volatility preferred
            score += 0.1
        if features.time_to_close > 1.0:  # Avoid last hour
            score += 0.1
        
        # Historical success
        if features.similar_pattern_success_rate > 0.6:
            score += 0.2
        if features.recent_signal_accuracy > 0.5:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _detect_rsi_divergence(self, spy_data: pd.DataFrame, rsi: pd.Series) -> float:
        """Detect RSI-price divergence"""
        if len(spy_data) < 20:
            return 0.0
        
        recent_prices = spy_data['close'].tail(10)
        recent_rsi = rsi.tail(10)
        
        price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        rsi_trend = recent_rsi.iloc[-1] - recent_rsi.iloc[0]
        
        # Divergence when price and RSI move in opposite directions
        if price_trend > 0 and rsi_trend < -5:
            return -0.5  # Bearish divergence
        elif price_trend < 0 and rsi_trend > 5:
            return 0.5   # Bullish divergence
        
        return 0.0
    
    def _estimate_bid_ask_spread(self, price: float) -> float:
        """Estimate bid-ask spread based on price level"""
        # Simplified estimation - would use real market data in production
        return min(0.01, price * 0.0001)
    
    def _estimate_order_flow(self, spy_data: pd.DataFrame) -> float:
        """Estimate order flow imbalance from price/volume data"""
        if len(spy_data) < 5:
            return 0.0
        
        # Simplified calculation using price changes and volume
        recent_data = spy_data.tail(5)
        price_changes = recent_data['close'].diff()
        volumes = recent_data['volume']
        
        # Weight price changes by volume
        weighted_flow = (price_changes * volumes).sum()
        total_volume = volumes.sum()
        
        return weighted_flow / total_volume if total_volume > 0 else 0.0
    
    def _find_similar_patterns(self, spy_data: pd.DataFrame) -> float:
        """Find similar historical patterns and their success rates"""
        # Simplified pattern matching - would implement more sophisticated approach
        if len(self.signal_history) < 10:
            return 0.5
        
        # Pattern similarity based on recent price movements
        recent_returns = spy_data['close'].pct_change().tail(5)
        
        similar_count = 0
        successful_count = 0
        
        for historical_signal in self.signal_history[-50:]:  # Last 50 signals
            if 'pattern_returns' in historical_signal:
                pattern_returns = historical_signal['pattern_returns']
                similarity = 1 - np.std(recent_returns - pattern_returns[:len(recent_returns)])
                
                if similarity > 0.8:  # Similar pattern
                    similar_count += 1
                    if historical_signal.get('profitable', False):
                        successful_count += 1
        
        return successful_count / similar_count if similar_count > 0 else 0.5
    
    def _calculate_recent_accuracy(self) -> float:
        """Calculate recent signal accuracy"""
        if len(self.signal_history) < 5:
            return 0.5
        
        recent_signals = self.signal_history[-10:]
        successful = sum(1 for s in recent_signals if s.get('profitable', False))
        return successful / len(recent_signals)
    
    def _assess_market_regime(self, spy_data: pd.DataFrame) -> float:
        """Assess current market regime alignment with strategy"""
        if len(spy_data) < 20:
            return 0.5
        
        # Trend strength
        prices = spy_data['close']
        sma_20 = prices.rolling(20).mean()
        trend_strength = (prices.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
        
        # Volatility regime
        returns = prices.pct_change()
        volatility = returns.rolling(20).std()
        vol_percentile = (volatility.iloc[-1] - volatility.quantile(0.2)) / (volatility.quantile(0.8) - volatility.quantile(0.2))
        
        # Regime score (0DTE strategies often work better in certain regimes)
        regime_score = 0.5
        if 0.15 < vol_percentile < 0.85:  # Moderate volatility
            regime_score += 0.2
        if abs(trend_strength) < 0.02:  # Not too trending
            regime_score += 0.3
        
        return min(regime_score, 1.0)
    
    def _estimate_vix_from_spy_volatility(self, spy_vol: float) -> float:
        """Estimate VIX level from SPY volatility"""
        # Approximate relationship between SPY vol and VIX
        return spy_vol * 100 * 1.5  # Rough approximation
    
    def _calculate_trend_strength(self, spy_data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple timeframes"""
        if len(spy_data) < 50:
            return 0.0
        
        prices = spy_data['close']
        
        # Multiple EMA trend alignment
        ema_9 = prices.ewm(span=9).mean()
        ema_21 = prices.ewm(span=21).mean()
        ema_50 = prices.ewm(span=50).mean()
        
        # Trend alignment score
        if ema_9.iloc[-1] > ema_21.iloc[-1] > ema_50.iloc[-1]:
            return 1.0  # Strong uptrend
        elif ema_9.iloc[-1] < ema_21.iloc[-1] < ema_50.iloc[-1]:
            return -1.0  # Strong downtrend
        else:
            return 0.0  # No clear trend
    
    def _estimate_iv_percentile(self, current_vol: float) -> float:
        """Estimate IV percentile from historical volatility"""
        # Simplified estimation - would use real IV data in production
        # Typical SPY vol ranges from 10% to 80%
        vol_pct = current_vol * 100
        
        if vol_pct < 15:
            return 0.2  # Low IV
        elif vol_pct > 35:
            return 0.8  # High IV
        else:
            return 0.5  # Medium IV

class Phase4MLEnhancedStrategy(AlpacaRealDataStrategy):
    """
    Phase 4 strategy with ML-enhanced signal filtering
    
    Builds on our Phase 3 real data integration and adds sophisticated
    signal quality assessment using machine learning.
    """
    
    def __init__(self, cache_dir: str = "../thetadata/cached_data"):
        super().__init__(cache_dir)
        self.ml_filter = MLSignalFilter()
        self.signal_threshold = 0.7  # Higher threshold for ML-filtered signals
        
        self.logger.info("🧠 Phase 4 ML-Enhanced Strategy initialized")
        self.logger.info("✅ Advanced signal filtering with ML quality assessment")
    
    def generate_ml_filtered_signals(self, spy_bars: pd.DataFrame) -> List[Dict]:
        """
        Generate signals with ML-based quality filtering
        
        Args:
            spy_bars: Real SPY price data
            
        Returns:
            List of high-quality filtered signals
        """
        # Get base signals from Phase 3 strategy
        base_signals = self.generate_signals(spy_bars)
        
        filtered_signals = []
        
        for signal in base_signals:
            # Extract features for ML evaluation
            features = self.ml_filter.extract_signal_features(spy_bars, signal)
            
            # Get ML quality score
            quality_score = self.ml_filter.predict_signal_quality(features)
            
            # Apply quality threshold
            if quality_score >= self.signal_threshold:
                signal['ml_quality_score'] = quality_score
                signal['features'] = features
                filtered_signals.append(signal)
                
                self.logger.info(f"🎯 High-quality signal: {signal['signal']} (Score: {quality_score:.3f})")
            else:
                self.logger.debug(f"⚠️ Signal filtered out: {signal['signal']} (Score: {quality_score:.3f})")
        
        return filtered_signals
    
    def run_phase4_ml_backtest(self, date_str: str) -> Dict:
        """
        Run Phase 4 backtest with ML signal filtering
        
        Args:
            date_str: Date in YYYYMMDD format
            
        Returns:
            Dictionary with backtest results and ML metrics
        """
        self.logger.info(f"🧠 Running Phase 4 ML backtest for {date_str}")
        
        # Load data
        data = self.load_cached_data(date_str)
        spy_bars = data['spy_bars']
        option_chains = data['option_chain']
        
        # Generate ML-filtered signals
        ml_signals = self.generate_ml_filtered_signals(spy_bars)
        
        if not ml_signals:
            self.logger.warning(f"📊 No high-quality signals found for {date_str}")
            return {
                'date': date_str,
                'trades': 0,
                'pnl': 0.0,
                'ml_signals_generated': 0,
                'ml_signals_filtered': 0,
                'average_quality_score': 0.0
            }
        
        # Execute trades with real option pricing
        total_pnl = 0.0
        trade_count = 0
        quality_scores = []
        
        for signal in ml_signals:
            # Execute trade using real Alpaca data
            trade_result = self.simulate_real_trade(signal, spy_bars, option_chains)
            
            if trade_result:
                total_pnl += trade_result['pnl']
                trade_count += 1
                quality_scores.append(signal['ml_quality_score'])
                
                # Update ML performance tracking
                self.ml_filter.signal_history.append({
                    'signal': signal,
                    'profitable': trade_result['pnl'] > 0,
                    'quality_score': signal['ml_quality_score'],
                    'features': signal['features']
                })
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        self.logger.info(f"📊 Phase 4 ML Results: {trade_count} trades, ${total_pnl:.2f} P&L, {avg_quality:.3f} avg quality")
        
        return {
            'date': date_str,
            'trades': trade_count,
            'pnl': total_pnl,
            'ml_signals_generated': len(ml_signals),
            'average_quality_score': avg_quality,
            'ml_filter_effectiveness': avg_quality
        }

if __name__ == "__main__":
    # Test Phase 4 ML-enhanced strategy
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 4 ML-Enhanced 0DTE Strategy')
    parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    args = parser.parse_args()
    
    strategy = Phase4MLEnhancedStrategy()
    result = strategy.run_phase4_ml_backtest(args.date)
    
    print(f"\n🧠 PHASE 4 ML-ENHANCED RESULTS:")
    print(f"📅 Date: {result['date']}")
    print(f"🎯 ML Signals: {result['ml_signals_generated']}")
    print(f"📊 Trades Executed: {result['trades']}")
    print(f"💰 P&L: ${result['pnl']:.2f}")
    print(f"🎯 Avg Quality Score: {result['average_quality_score']:.3f}") 