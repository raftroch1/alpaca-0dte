#!/usr/bin/env python3
"""
Realistic Trading Configuration for 0DTE Options
===============================================

This module defines all realistic trading costs, market microstructure parameters,
and execution modeling parameters based on actual market conditions for 0DTE SPY options.

References:
- Alpaca commission structure: https://docs.alpaca.markets/docs/historical-option-data
- CBOE SPY options market data and statistics
- Academic studies on options market microstructure
- Real-world trading experience with 0DTE options
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd

@dataclass
class CommissionStructure:
    """Realistic commission structure for options trading"""
    
    # Alpaca commission structure (as of 2024)
    base_commission: float = 0.65  # Base commission per trade
    per_contract_fee: float = 0.50  # Per contract fee
    
    # Regulatory fees (realistic estimates)
    sec_fee_rate: float = 0.0000221  # SEC fee (per dollar of trade value)
    finra_taf: float = 0.000145  # FINRA Trading Activity Fee
    occ_fee: float = 0.04  # OCC clearing fee per contract
    
    # Exchange fees (varies by exchange, using CBOE estimates)
    exchange_fee_per_contract: float = 0.15
    
    def calculate_total_commission(self, quantity: int, trade_value: float) -> float:
        """Calculate total commission for a trade
        
        Args:
            quantity: Number of contracts
            trade_value: Total trade value in dollars
            
        Returns:
            Total commission and fees
        """
        total = (
            self.base_commission +
            (quantity * self.per_contract_fee) +
            (trade_value * self.sec_fee_rate) +
            (trade_value * self.finra_taf) +
            (quantity * self.occ_fee) +
            (quantity * self.exchange_fee_per_contract)
        )
        return round(total, 2)

@dataclass
class BidAskSpreadModel:
    """Realistic bid/ask spread modeling for 0DTE options"""
    
    # Base spread parameters (in dollars)
    min_spread: float = 0.01  # Minimum spread (1 cent)
    max_spread: float = 0.50  # Maximum spread for illiquid options
    
    # Spread factors
    volatility_factor: float = 0.3  # Spread widens with volatility
    time_to_expiry_factor: float = 2.0  # Spread widens as expiry approaches
    moneyness_factor: float = 1.5  # Spread widens for OTM options
    volume_factor: float = 0.8  # Spread narrows with higher volume
    
    # Market hours impact
    market_open_spread_multiplier: float = 2.0  # Wider spreads at open
    market_close_spread_multiplier: float = 3.0  # Much wider spreads near close
    
    def calculate_spread(self, 
                        underlying_price: float,
                        strike: float,
                        time_to_expiry_minutes: int,
                        implied_vol: float,
                        volume: int,
                        time_of_day: str) -> float:
        """Calculate realistic bid/ask spread
        
        Args:
            underlying_price: Current underlying price
            strike: Option strike price
            time_to_expiry_minutes: Minutes until expiry
            implied_vol: Implied volatility
            volume: Recent volume
            time_of_day: "open", "regular", "close"
            
        Returns:
            Spread in dollars
        """
        # Base spread
        base = self.min_spread
        
        # Adjust for volatility
        vol_adjustment = implied_vol * self.volatility_factor
        
        # Adjust for time to expiry (wider spreads as expiry approaches)
        time_adjustment = max(0, self.time_to_expiry_factor * (1 - time_to_expiry_minutes / 60))
        
        # Adjust for moneyness
        moneyness = abs(underlying_price - strike) / underlying_price
        moneyness_adjustment = moneyness * self.moneyness_factor
        
        # Adjust for volume (narrower spreads with higher volume)
        volume_adjustment = max(0, self.volume_factor * (1 - min(volume / 1000, 1)))
        
        # Calculate total spread
        spread = base + vol_adjustment + time_adjustment + moneyness_adjustment + volume_adjustment
        
        # Apply time of day multipliers
        if time_of_day == "open":
            spread *= self.market_open_spread_multiplier
        elif time_of_day == "close":
            spread *= self.market_close_spread_multiplier
        
        return min(spread, self.max_spread)

@dataclass
class SlippageModel:
    """Realistic slippage modeling for options orders"""
    
    # Base slippage (in basis points)
    base_slippage_bps: int = 5  # 0.05% base slippage
    
    # Market impact factors
    market_impact_per_contract: float = 0.01  # Additional slippage per contract
    urgency_multiplier: float = 2.0  # Market orders have higher slippage
    
    # Liquidity factors
    low_volume_threshold: int = 50  # Contracts per minute
    low_volume_penalty: float = 0.02  # Additional slippage for low volume
    
    # Time factors
    time_decay_multiplier: float = 1.5  # Higher slippage near expiry
    
    def calculate_slippage(self,
                         order_quantity: int,
                         is_market_order: bool,
                         recent_volume: int,
                         time_to_expiry_minutes: int,
                         option_price: float) -> float:
        """Calculate realistic slippage for an order
        
        Args:
            order_quantity: Number of contracts to trade
            is_market_order: True if market order, False if limit
            recent_volume: Recent volume (contracts per minute)
            time_to_expiry_minutes: Minutes until expiry
            option_price: Current option price
            
        Returns:
            Slippage in dollars per contract
        """
        # Base slippage
        base_slippage = (self.base_slippage_bps / 10000) * option_price
        
        # Market impact
        market_impact = order_quantity * self.market_impact_per_contract
        
        # Order type adjustment
        if is_market_order:
            urgency_adjustment = base_slippage * self.urgency_multiplier
        else:
            urgency_adjustment = 0
        
        # Liquidity penalty
        if recent_volume < self.low_volume_threshold:
            liquidity_penalty = option_price * self.low_volume_penalty
        else:
            liquidity_penalty = 0
        
        # Time decay factor
        time_factor = max(1, self.time_decay_multiplier * (1 - time_to_expiry_minutes / 60))
        
        total_slippage = (base_slippage + market_impact + urgency_adjustment + liquidity_penalty) * time_factor
        
        return round(total_slippage, 4)

@dataclass
class OrderExecutionModel:
    """Realistic order execution probabilities and timing"""
    
    # Fill probabilities
    market_order_fill_prob: float = 0.98  # Market orders almost always fill
    limit_order_fill_prob_base: float = 0.65  # Base limit order fill probability
    
    # Partial fill thresholds
    partial_fill_threshold: int = 10  # Contracts above which partial fills occur
    partial_fill_probability: float = 0.25  # Probability of partial fill
    partial_fill_ratio_range: Tuple[float, float] = (0.3, 0.8)  # Partial fill ratio range
    
    # Execution timing
    market_order_delay_seconds: Tuple[float, float] = (0.1, 0.5)  # Market order delay range
    limit_order_delay_seconds: Tuple[float, float] = (1.0, 30.0)  # Limit order delay range
    
    # Market condition factors
    high_volatility_fill_reduction: float = 0.15  # Reduced fill prob in high vol
    low_liquidity_fill_reduction: float = 0.20  # Reduced fill prob in low liquidity
    
    def calculate_fill_probability(self,
                                 is_market_order: bool,
                                 order_quantity: int,
                                 current_volume: int,
                                 implied_vol: float) -> float:
        """Calculate order fill probability
        
        Args:
            is_market_order: True if market order
            order_quantity: Number of contracts
            current_volume: Current market volume
            implied_vol: Current implied volatility
            
        Returns:
            Fill probability (0.0 to 1.0)
        """
        if is_market_order:
            base_prob = self.market_order_fill_prob
        else:
            base_prob = self.limit_order_fill_prob_base
        
        # Adjust for high volatility
        if implied_vol > 0.5:  # 50% IV threshold
            base_prob -= self.high_volatility_fill_reduction
        
        # Adjust for low liquidity
        if current_volume < 100:  # Low volume threshold
            base_prob -= self.low_liquidity_fill_reduction
        
        return max(0.1, min(1.0, base_prob))  # Clamp between 10% and 100%

@dataclass
class MarketDataRealism:
    """Parameters for realistic market data simulation"""
    
    # Volume patterns (realistic for SPY 0DTE options)
    typical_volume_per_minute: Dict[str, int] = None
    volume_volatility: float = 0.3  # Volume standard deviation factor
    
    # Bid/ask quote frequency
    quote_updates_per_minute: int = 60  # Number of quote updates
    quote_volatility: float = 0.02  # Quote price volatility
    
    # Market gaps and jumps
    gap_probability: float = 0.001  # Probability of price gap per minute
    max_gap_size: float = 0.10  # Maximum gap size as fraction of price
    
    def __post_init__(self):
        if self.typical_volume_per_minute is None:
            # Realistic volume patterns throughout the day
            self.typical_volume_per_minute = {
                "09:30": 500,  # Market open - high volume
                "10:00": 200,  # Early morning
                "11:00": 150,  # Mid morning
                "12:00": 100,  # Lunch time - low volume
                "13:00": 120,  # Early afternoon
                "14:00": 180,  # Mid afternoon
                "15:00": 300,  # Late afternoon pickup
                "15:30": 800,  # Final 30 minutes - very high volume
                "15:45": 1200, # Final 15 minutes - extreme volume
                "15:55": 2000  # Final 5 minutes - massive volume
            }

@dataclass
class RealismConfig:
    """Complete realistic trading configuration"""
    
    # Core components
    commissions: CommissionStructure = None
    spreads: BidAskSpreadModel = None
    slippage: SlippageModel = None
    execution: OrderExecutionModel = None
    market_data: MarketDataRealism = None
    
    # Risk management constraints
    max_position_size: int = 50  # Maximum contracts per position
    max_daily_trades: int = 100  # Maximum trades per day
    
    # Account constraints
    day_trading_buying_power_factor: float = 4.0  # 4:1 leverage for day trading
    pattern_day_trader_requirements: bool = True  # Require $25k minimum
    
    def __post_init__(self):
        if self.commissions is None:
            self.commissions = CommissionStructure()
        if self.spreads is None:
            self.spreads = BidAskSpreadModel()
        if self.slippage is None:
            self.slippage = SlippageModel()
        if self.execution is None:
            self.execution = OrderExecutionModel()
        if self.market_data is None:
            self.market_data = MarketDataRealism()

# Predefined realistic configurations
ULTRA_REALISTIC_CONFIG = RealismConfig(
    commissions=CommissionStructure(
        base_commission=0.65,
        per_contract_fee=0.50
    ),
    spreads=BidAskSpreadModel(
        min_spread=0.01,
        max_spread=0.50,
        volatility_factor=0.3,
        market_close_spread_multiplier=5.0  # Very wide spreads near close
    ),
    slippage=SlippageModel(
        base_slippage_bps=10,  # Higher realistic slippage
        market_impact_per_contract=0.02
    ),
    execution=OrderExecutionModel(
        market_order_fill_prob=0.95,
        limit_order_fill_prob_base=0.60  # Lower fill probability
    )
)

CONSERVATIVE_REALISTIC_CONFIG = RealismConfig(
    commissions=CommissionStructure(
        base_commission=0.65,
        per_contract_fee=0.50
    ),
    spreads=BidAskSpreadModel(
        min_spread=0.01,
        max_spread=0.30,
        market_close_spread_multiplier=3.0
    ),
    slippage=SlippageModel(
        base_slippage_bps=5,
        market_impact_per_contract=0.01
    ),
    execution=OrderExecutionModel(
        market_order_fill_prob=0.98,
        limit_order_fill_prob_base=0.70
    )
)

# Option Greeks estimation (for realistic option pricing)
def estimate_option_greeks(underlying_price: float,
                         strike: float,
                         time_to_expiry_minutes: int,
                         risk_free_rate: float = 0.05,
                         implied_vol: float = 0.5) -> Dict[str, float]:
    """Estimate option Greeks for realistic pricing
    
    Note: This is a simplified estimation. In practice, you would use
    a proper options pricing model like Black-Scholes.
    """
    import math
    
    # Convert time to years
    T = time_to_expiry_minutes / (252 * 24 * 60)  # Trading minutes in a year
    
    # Simplified estimates (real implementation would use proper formulas)
    moneyness = underlying_price / strike
    
    # Rough approximations for 0DTE options
    delta = 0.5 if abs(moneyness - 1) < 0.01 else (0.8 if moneyness > 1 else 0.2)
    gamma = max(0, 10 * (1 - abs(moneyness - 1)) * math.sqrt(T))
    theta = -50 * math.sqrt(T)  # Time decay accelerates
    vega = underlying_price * 0.1 * math.sqrt(T)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }

# Market hours and trading session definitions
MARKET_SESSIONS = {
    'pre_market': ('04:00', '09:30'),
    'regular_hours': ('09:30', '16:00'),
    'after_hours': ('16:00', '20:00')
}

def get_time_of_day_category(current_time: str) -> str:
    """Categorize time of day for spread calculations
    
    Args:
        current_time: Time in HH:MM format
        
    Returns:
        'open', 'regular', or 'close'
    """
    from datetime import datetime
    
    time_obj = datetime.strptime(current_time, '%H:%M').time()
    
    if current_time <= '10:00':
        return 'open'
    elif current_time >= '15:30':
        return 'close' 
    else:
        return 'regular'

if __name__ == "__main__":
    # Test the configuration
    config = ULTRA_REALISTIC_CONFIG
    
    print("🎯 Realistic Trading Configuration Test")
    print("=" * 50)
    
    # Test commission calculation
    commission = config.commissions.calculate_total_commission(
        quantity=5, 
        trade_value=2500
    )
    print(f"Commission for 5 contracts ($2500): ${commission:.2f}")
    
    # Test spread calculation
    spread = config.spreads.calculate_spread(
        underlying_price=450.0,
        strike=450.0,
        time_to_expiry_minutes=30,
        implied_vol=0.5,
        volume=100,
        time_of_day='close'
    )
    print(f"Bid/ask spread near close: ${spread:.2f}")
    
    # Test slippage calculation
    slippage = config.slippage.calculate_slippage(
        order_quantity=3,
        is_market_order=True,
        recent_volume=50,
        time_to_expiry_minutes=15,
        option_price=2.50
    )
    print(f"Slippage for market order: ${slippage:.4f} per contract")
    
    print("\n✅ Configuration test completed successfully!") 