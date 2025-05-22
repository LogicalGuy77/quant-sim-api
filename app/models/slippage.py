import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression, QuantileRegressor
import logging

logger = logging.getLogger(__name__) # Initialize logger for this module

class SlippageModel:
    """
    Model for estimating slippage based on order book data.
    """
    def __init__(self, regression_type="linear"):
        """
        Initialize slippage model.
        
        Parameters:
        - regression_type: Type of regression model to use ('linear' or 'quantile')
        """
        self.regression_type = regression_type
        self.model = self._initialize_model()
        self.is_trained = False # Training not implemented in this scope
        
    def _initialize_model(self):
        """Initialize the regression model based on specified type."""
        if self.regression_type == "linear":
            return LinearRegression()
        elif self.regression_type == "quantile":
            # Using a common quantile for slippage (e.g., 75th or 90th percentile)
            return QuantileRegressor(quantile=0.75, solver='highs') # Specify solver for newer scikit-learn
        else:
            raise ValueError(f"Unsupported regression type: {self.regression_type}")
    
    def train(self, features, targets):
        """
        Train the slippage model with historical data.
        (Note: Training feature preparation and usage is not fully implemented here)
        Parameters:
        - features: Array of feature vectors 
        - targets: Array of observed slippage values
        """
        self.model.fit(features, targets)
        self.is_trained = True
    
    def estimate_slippage(self, order_size_base, order_book_data, volatility):
        """
        Estimate slippage. Uses heuristic model.
        'order_size_base' is the order quantity in the base asset.
        'volatility' is expected as a decimal (e.g., 0.1 for 10%).
        """
        # Currently, always using the heuristic model
        return self._heuristic_slippage(order_size_base, order_book_data, volatility)
    
    def _heuristic_slippage(self, order_size_base, order_book_data, volatility):
        """
        Simplified heuristic model for slippage.
        """
        if not order_book_data or \
           not order_book_data.get("asks") or not order_book_data.get("bids") or \
           not order_book_data["asks"] or not order_book_data["bids"]:
            logger.warning("SlippageModel: Order book data is missing, incomplete, or empty for heuristic calculation.")
            return 0.01 # Default to minimum if data is bad

        spread_percentage = self._calculate_spread(order_book_data) # This is already a percentage e.g. 0.01 for 0.01%
        
        # Use a more restrictive depth for heuristic, e.g., average of top 2 ask/bid levels' quantities
        depth_base_L2 = self._calculate_depth(order_book_data, levels=2) # Depth in base asset units

        if depth_base_L2 <= 0: # Changed from == 0 to <=0 for robustness
            logger.warning(f"SlippageModel: Calculated depth (L2) is {depth_base_L2}. Defaulting slippage.")
            return 0.01 

        normalized_size = min(1.0, order_size_base / depth_base_L2)

        # Revised slippage formula for better sensitivity:
        # Component 1: Cost from crossing the spread / consuming immediate liquidity
        # 'spread_percentage' is e.g. 0.01 for 0.01%.
        # If normalized_size is 0.1 (10% of L2 depth), and C1 is 5, this part is 0.1 * 0.01% * 5 = 0.005%
        C1_spread_factor = 5.0  # Multiplier for spread component
        slippage_pct_consumption = normalized_size * spread_percentage * C1_spread_factor

        # Component 2: Cost from volatility during execution, scaled by size
        # 'volatility' is a decimal (e.g., 0.1 for 10%).
        # If vol=0.1, norm_size=0.1, C2=0.5, this part is 0.1 * 0.1 * 0.5 * 100 = 0.5%
        C2_volatility_factor = 0.5 # Multiplier for volatility component
        slippage_pct_volatility = volatility * normalized_size * C2_volatility_factor * 100.0 # Convert decimal contribution to percentage points

        slippage_pct_calculated = slippage_pct_consumption + slippage_pct_volatility

        logger.debug(
            f"SlippageHeuristic: order_size_base={order_size_base:.4f}, "
            f"spread={spread_percentage:.4f}%, depth_base_L2={depth_base_L2:.4f}, "
            f"normalized_size={normalized_size:.6f}, volatility_param={volatility:.4f}, "
            f"C1_spread_factor={C1_spread_factor}, C2_volatility_factor={C2_volatility_factor}, "
            f"slippage_consumption_contrib={slippage_pct_consumption:.6f}%, "
            f"slippage_volatility_contrib={slippage_pct_volatility:.6f}%, "
            f"calc_slippage_pct_before_min={slippage_pct_calculated:.6f}%"
        )
        
        # Ensure a minimum slippage, but allow the calculated value if it's higher.
        return max(0.01, slippage_pct_calculated)

    def _calculate_spread(self, order_book_data):
        """Calculate spread as a percentage. Returns 0.0 on error or if data is insufficient."""
        try:
            if not order_book_data or \
               not order_book_data.get("asks") or not order_book_data.get("bids") or \
               not order_book_data["asks"] or not order_book_data["bids"]:
                logger.debug("SlippageModel._calculate_spread: Insufficient order book data.")
                return 0.01 # Default to a small spread if data is bad

            best_ask_price = float(order_book_data["asks"][0][0])
            best_bid_price = float(order_book_data["bids"][0][0])
            
            if best_bid_price <= 0 or best_ask_price <= best_bid_price: # Added checks for valid prices
                 logger.debug(f"SlippageModel._calculate_spread: Invalid best_ask/best_bid prices: Ask={best_ask_price}, Bid={best_bid_price}")
                 return 0.01 # Default if prices are invalid

            mid_price = (best_ask_price + best_bid_price) / 2.0
            if mid_price == 0:
                logger.debug("SlippageModel._calculate_spread: Mid price is zero.")
                return 0.01 # Default

            spread_percentage = ((best_ask_price - best_bid_price) / mid_price) * 100.0
            return spread_percentage
        except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
            logger.warning(f"SlippageModel: Error calculating spread: {e}. Orderbook sample: asks[:1]={order_book_data.get('asks', [])[:1]}, bids[:1]={order_book_data.get('bids', [])[:1]}")
            return 0.01 # Default to a small spread on error to avoid large downstream errors

    def _calculate_depth(self, order_book_data, levels=10):
        """
        Calculate average cumulative quantity for a certain number of order book levels 
        on one side (e.g., average of ask depth and bid depth for 'levels').
        Returns depth in base asset units.
        """
        ask_qty_sum = 0.0
        bid_qty_sum = 0.0
        
        if order_book_data and order_book_data.get("asks"):
            for level_data in order_book_data["asks"][:levels]:
                try:
                    ask_qty_sum += float(level_data[1]) # Quantity is at index 1
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"SlippageModel: Malformed ask level in depth calculation: {level_data}, error: {e}")
        
        if order_book_data and order_book_data.get("bids"):
            for level_data in order_book_data["bids"][:levels]:
                try:
                    bid_qty_sum += float(level_data[1]) # Quantity is at index 1
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"SlippageModel: Malformed bid level in depth calculation: {level_data}, error: {e}")
        
        # Average depth of the two sides for the specified number of levels
        # If one side is empty, this will still give a measure based on the other side.
        if ask_qty_sum > 0 and bid_qty_sum > 0:
            avg_depth_one_side = (ask_qty_sum + bid_qty_sum) / 2.0
        elif ask_qty_sum > 0:
            avg_depth_one_side = ask_qty_sum # Or consider it half, depending on model needs
        elif bid_qty_sum > 0:
            avg_depth_one_side = bid_qty_sum # Or consider it half
        else:
            avg_depth_one_side = 0.0
            
        return avg_depth_one_side

class MakerTakerModel:
    """
    Model for predicting maker/taker proportion for a given order.
    """
    def __init__(self):
        # This would be a logistic regression model in practice
        # For now, we'll use a simple heuristic
        pass
    
    def predict_maker_proportion(self, order_size, order_book_data, volatility):
        """
        Predict the proportion of an order that will be executed as maker.
        
        Parameters:
        - order_size: Size of the order
        - order_book_data: Current order book state
        - volatility: Current market volatility
        
        Returns:
        - Estimated maker proportion (0.0-1.0)
        """
        # Simple heuristic: larger orders and higher volatility reduce maker proportion
        # In practice, this would be a trained logistic regression model
        
        # Extract relevant metrics from order book
        depth = self._calculate_depth(order_book_data)
        spread = self._calculate_spread(order_book_data)
        
        # Normalize order size relative to market depth
        size_factor = min(1.0, order_size / depth) if depth > 0 else 1.0
        
        # Higher volatility reduces maker proportion
        volatility_factor = min(1.0, volatility / 0.5)  # Normalize volatility
        
        # Wider spreads may increase maker proportion
        spread_factor = min(1.0, 0.01 / spread) if spread > 0 else 0.5
        
        # Calculate maker proportion (inverse relationship with size and volatility)
        base_maker_proportion = 0.7  # Base assumption
        maker_proportion = base_maker_proportion * (1 - size_factor * 0.5) * (1 - volatility_factor * 0.7) * spread_factor
        
        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, maker_proportion))
    
    def _calculate_depth(self, order_book_data, levels=5):
        """Calculate market depth from order book data."""
        asks = order_book_data["asks"]
        bids = order_book_data["bids"]
        
        ask_depth = sum(float(level[1]) for level in asks[:min(levels, len(asks))])
        bid_depth = sum(float(level[1]) for level in bids[:min(levels, len(bids))])
        
        return (ask_depth + bid_depth) / 2
    
    def _calculate_spread(self, order_book_data):
        """Calculate the bid-ask spread from order book data."""
        asks = order_book_data["asks"]
        bids = order_book_data["bids"]
        
        if not asks or not bids:
            return 0
            
        best_ask = float(asks[0][0])
        best_bid = float(bids[0][0])
        
        return (best_ask - best_bid) / ((best_ask + best_bid) / 2)