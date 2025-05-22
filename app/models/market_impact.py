import numpy as np
import numba

class AlmgrenChrissModel:
    """
    Almgren-Chriss market impact model implementation.
    Impact = η * (normalized_size)^α + γ * (normalized_size)^β
    """
    def __init__(self, 
                 alpha=1, 
                 beta=1, 
                 gamma=0.05, 
                 eta=0.05, 
                 volatility=0.3, 
                 risk_aversion=0.001):
        """
        Initialize the Almgren-Chriss model with parameters.
        
        Parameters:
        - alpha: Exponent for temporary impact function
        - beta: Exponent for permanent impact function
        - gamma: Scale parameter for permanent impact
        - eta: Scale parameter for temporary impact
        - volatility: Market volatility
        - risk_aversion: Risk aversion parameter
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.volatility = volatility
        self.risk_aversion = risk_aversion
    
    def temporary_impact(self, volume):
        """Calculate temporary market impact."""
        return _numba_temporary_impact(volume, self.eta, self.alpha)
    
    def permanent_impact(self, volume):
        """Calculate permanent market impact."""
        return _numba_permanent_impact(volume, self.gamma, self.beta)
    
    def calculate_market_impact(self, order_size, price, order_book_data):
        """
        Calculate the market impact for a given order size.
        
        Parameters:
        - order_size: Size of the order in base currency units
        - price: Current market price
        - order_book_data: Current order book state (bids/asks)
        
        Returns:
        - Estimated market impact in percentage and absolute terms
        """
        # Extract relevant data from order book
        asks_array, bids_array = self._prepare_orderbook_arrays(order_book_data)
        
        # Calculate spread and depth using Numba-optimized functions
        avg_spread = _numba_calculate_avg_spread(asks_array, bids_array)
        avg_depth = _numba_calculate_avg_depth(asks_array, bids_array, 10)
        
        # Calculate impact using Numba-optimized core calculation
        impact_results = _numba_market_impact_calc(
            order_size, 
            price, 
            avg_spread, 
            avg_depth, 
            self.eta, 
            self.alpha, 
            self.gamma, 
            self.beta
        )
        
        return {
            "impact_percentage": impact_results[0],
            "impact_absolute": impact_results[1],
            "temporary_impact": impact_results[2],
            "permanent_impact": impact_results[3]
        }
    
    def _prepare_orderbook_arrays(self, order_book_data):
        """Convert orderbook data to NumPy arrays for Numba processing."""
        asks = order_book_data["asks"]
        bids = order_book_data["bids"]
        
        # Convert to numpy arrays for better performance with Numba
        asks_array = np.array([[float(level[0]), float(level[1])] for level in asks])
        bids_array = np.array([[float(level[0]), float(level[1])] for level in bids])
        
        return asks_array, bids_array
    
    def _calculate_avg_spread(self, order_book_data):
        """Calculate average spread from order book data."""
        asks = order_book_data["asks"]
        bids = order_book_data["bids"]
        
        if not asks or not bids:
            return 0
            
        best_ask = float(asks[0][0])
        best_bid = float(bids[0][0])
        
        return (best_ask - best_bid) / ((best_ask + best_bid) / 2)
    
    def _calculate_avg_depth(self, order_book_data, levels=10):
        """Calculate average market depth from order book data."""
        asks = order_book_data["asks"]
        bids = order_book_data["bids"]
        
        ask_depth = sum(float(level[1]) for level in asks[:min(levels, len(asks))])
        bid_depth = sum(float(level[1]) for level in bids[:min(levels, len(bids))])
        
        return (ask_depth + bid_depth) / 2


# Numba-optimized functions
@numba.jit(nopython=True)
def _numba_temporary_impact(volume, eta, alpha):
    """Numba-optimized temporary impact calculation."""
    return eta * (volume ** alpha)

@numba.jit(nopython=True)
def _numba_permanent_impact(volume, gamma, beta):
    """Numba-optimized permanent impact calculation."""
    return gamma * (volume ** beta)

@numba.jit(nopython=True)
def _numba_calculate_avg_spread(asks_array, bids_array):
    """Numba-optimized spread calculation."""
    if asks_array.size == 0 or bids_array.size == 0:
        return 0.0
        
    best_ask = asks_array[0, 0]
    best_bid = bids_array[0, 0]
    
    return (best_ask - best_bid) / ((best_ask + best_bid) / 2)

@numba.jit(nopython=True)
def _numba_calculate_avg_depth(asks_array, bids_array, levels):
    """Numba-optimized depth calculation."""
    ask_depth = 0.0
    bid_depth = 0.0
    
    # Calculate ask depth
    ask_levels = min(levels, asks_array.shape[0])
    for i in range(ask_levels):
        ask_depth += asks_array[i, 1]
    
    # Calculate bid depth
    bid_levels = min(levels, bids_array.shape[0])
    for i in range(bid_levels):
        bid_depth += bids_array[i, 1]
    
    return (ask_depth + bid_depth) / 2

@numba.jit(nopython=True)
def _numba_market_impact_calc(order_size, price, avg_spread, avg_depth, eta, alpha, gamma, beta):
    """
    Numba-optimized core market impact calculation.
    
    Returns: (impact_percentage, impact_absolute, temporary_impact, permanent_impact)
    """
    # Prevent division by zero
    if avg_depth <= 0:
        normalized_size = 1.0
    else:
        normalized_size = order_size / avg_depth
    
    # Calculate components
    temp_impact = eta * (normalized_size ** alpha)
    perm_impact = gamma * (normalized_size ** beta)
    
    # Calculate total impact
    total_impact = temp_impact + perm_impact
    
    # Convert to percentage
    temp_impact_pct = temp_impact * 100
    perm_impact_pct = perm_impact * 100
    impact_percentage = total_impact * 100
    
    # Ensure minimum impact
    impact_percentage = max(0.005, impact_percentage)
    
    # Calculate absolute impact
    impact_absolute = impact_percentage / 100 * price * order_size
    
    return impact_percentage, impact_absolute, temp_impact_pct, perm_impact_pct