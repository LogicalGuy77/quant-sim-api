from app.config import OKX_FEE_TIERS

class FeeModel:
    """
    Model for calculating trading fees based on exchange fee structure.
    """
    def __init__(self):
        self.fee_tiers = OKX_FEE_TIERS
    
    def calculate_fees(self, order_value, fee_tier, maker_taker_proportion=None):
        """
        Calculate trading fees for a given order.
        
        Parameters:
        - order_value: Value of the order in quote currency (e.g., USDT)
        - fee_tier: User's fee tier on the exchange
        - maker_taker_proportion: Proportion of order expected to be maker vs taker 
                                 (0.0 = all taker, 1.0 = all maker)
                                 
        Returns:
        - Dictionary with fee details
        """
        # Default to worst case (all taker) if proportion not provided
        if maker_taker_proportion is None:
            maker_taker_proportion = 0.0
            
        # Get fee rates for the specified tier
        if fee_tier not in self.fee_tiers:
            fee_tier = list(self.fee_tiers.keys())[0]  # Default to first tier
            
        maker_fee_rate = self.fee_tiers[fee_tier]["maker"]
        taker_fee_rate = self.fee_tiers[fee_tier]["taker"]
        
        # Calculate maker and taker portions
        maker_portion = order_value * maker_taker_proportion
        taker_portion = order_value * (1 - maker_taker_proportion)
        
        # Calculate fees
        maker_fee = maker_portion * maker_fee_rate
        taker_fee = taker_portion * taker_fee_rate
        total_fee = maker_fee + taker_fee
        effective_fee_rate = total_fee / order_value if order_value > 0 else 0
        
        return {
            "maker_fee": maker_fee,
            "taker_fee": taker_fee,
            "total_fee": total_fee,
            "effective_fee_rate": effective_fee_rate,
            "maker_proportion": maker_taker_proportion,
            "taker_proportion": 1 - maker_taker_proportion,
            "fee_tier": fee_tier
        }

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