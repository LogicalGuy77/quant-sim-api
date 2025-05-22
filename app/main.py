import asyncio
import time
import logging
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any
import json
from datetime import datetime 

from app.config import WEBSOCKET_ENDPOINT, SUPPORTED_ASSETS
from app.services.websocket_service import WebSocketService
from app.models.market_impact import AlmgrenChrissModel
from app.models.slippage import SlippageModel, MakerTakerModel
from app.models.fees import FeeModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) 

# Initialize FastAPI app
app = FastAPI(title="Trade Cost Simulator API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
market_impact_model = AlmgrenChrissModel()
slippage_model = SlippageModel()
maker_taker_model = MakerTakerModel()
fee_model = FeeModel()

# Shared state
latest_order_books = {}
connected_clients = set()

# Pydantic models for request/response
class SimulationRequest(BaseModel):
    exchange: str
    asset: str
    order_type: str
    quantity: float
    volatility: float
    fee_tier: str

class SimulationResponse(BaseModel):
    slippage: float
    fees: Dict[str, float]
    market_impact: float
    net_cost: float
    maker_taker_proportion: float
    internal_latency: float
    timestamp: float

# WebSocket service
websocket_service = None

async def broadcast_updates():
    """Broadcast updates to all connected WebSocket clients."""
    message = {
        "type": "update",
        "timestamp": time.time(),
        "data": {k: v for k, v in latest_order_books.items()}
    }
    
    disconnected_clients = set()
    for client in connected_clients:
        try:
            await client.send_text(json.dumps(message))
        except Exception:
            disconnected_clients.add(client)
    
    # Remove disconnected clients
    connected_clients.difference_update(disconnected_clients)

# Store continuous simulation results
continuous_metrics = {}
# Store historical metrics (last 100 ticks)
continuous_metrics_history = {}
# Store the parameters from the last successful simulation, keyed by asset_key (e.g., "okx:btc-usdt-swap")
last_simulation_params_store: Dict[str, SimulationRequest] = {}

async def process_orderbook(orderbook_data):
    """Process orderbook data and update shared state."""
    if not orderbook_data:
        return
        
    exchange = orderbook_data.get("exchange")
    symbol = orderbook_data.get("symbol")
    
    if not exchange or not symbol:
        return
    
    key = f"{exchange.lower()}:{symbol.lower()}"
    latest_order_books[key] = orderbook_data
    
    # Update metrics with each new tick
    await update_metrics(key, orderbook_data)
    
    # Notify connected clients about new data
    if connected_clients:
        await broadcast_updates()
        
    # Also broadcast metrics updates
    await broadcast_metrics_updates()

async def update_metrics(key, orderbook_data):
    """Update metrics based on the latest simulation parameters for this asset, or defaults."""
    global last_simulation_params_store, continuous_metrics, continuous_metrics_history
    global maker_taker_model, fee_model, slippage_model, market_impact_model # Ensure models are accessible

    # Default parameters if no simulation has been run for this asset yet
    default_quantity_usd = 100.0
    default_volatility = 0.1
    default_fee_tier = "VIP0"

    current_sim_request: Optional[SimulationRequest] = last_simulation_params_store.get(key)

    quantity_usd_to_use = default_quantity_usd
    volatility_to_use = default_volatility
    fee_tier_to_use = default_fee_tier

    if current_sim_request:
        quantity_usd_to_use = current_sim_request.quantity
        volatility_to_use = current_sim_request.volatility
        fee_tier_to_use = current_sim_request.fee_tier
        logger.debug(f"Using stored sim params for {key}: Q={quantity_usd_to_use}, V={volatility_to_use}, F={fee_tier_to_use}")
    else:
        logger.debug(f"Using default params for {key}: Q={quantity_usd_to_use}, V={volatility_to_use}, F={fee_tier_to_use}")
    
    try:
        # Start timing
        start_time = time.perf_counter()
        
        # Calculate mid price
        mid_price = calculate_mid_price(orderbook_data)

        if mid_price <= 0:
            logger.warning(f"Mid price is {mid_price} for {key}, cannot calculate metrics accurately.")
            # Set error state or minimal metrics if mid_price is invalid
            continuous_metrics[key] = {
                "timestamp": time.time(), "error": "Invalid mid_price", "mid_price": mid_price,
                "slippage": 0.0, "fees": {"maker_fee": 0.0, "taker_fee": 0.0, "total_fee": 0.0, "effective_rate": 0.0},
                "market_impact": 0.0, "net_cost": 0.0, "maker_taker_proportion": 0.0, "internal_latency": 0.0
            }
            return
        
        # order_value is always in quote currency (USD)
        order_value_quote = quantity_usd_to_use
        # order_size_base is in the base asset's units (e.g., BTC quantity)
        order_size_base = order_value_quote / mid_price
        
        # Estimate maker/taker proportion
        maker_proportion = maker_taker_model.predict_maker_proportion(
            order_size_base, # Use base asset quantity
            orderbook_data,
            volatility_to_use
        )
        
        # Calculate fees
        fees_result = fee_model.calculate_fees(
            order_value_quote, # Fee calculation uses quote currency value
            fee_tier_to_use,
            maker_proportion
        )
        
        # Calculate slippage
        slippage_percentage = slippage_model.estimate_slippage(
            order_size_base, # Use base asset quantity
            orderbook_data,
            volatility_to_use
        )
        slippage_absolute = (slippage_percentage / 100) * order_value_quote
        
        # Calculate market impact
        impact_result = market_impact_model.calculate_market_impact(
            order_size_base, # Use base asset quantity
            mid_price,
            orderbook_data
        )
        # impact_result["impact_absolute"] is already in quote currency value
        market_impact_absolute = impact_result["impact_absolute"]
        
        # Calculate net cost in quote currency
        net_cost_quote = slippage_absolute + fees_result["total_fee"] + market_impact_absolute
        # Net cost as a percentage of order value for MetricsDisplay
        net_cost_percentage = (net_cost_quote / order_value_quote) * 100 if order_value_quote > 0 else 0.0
        
        # Calculate latency
        end_time = time.perf_counter()
        processing_latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Store metrics
        timestamp = time.time()
        continuous_metrics[key] = {
            "timestamp": timestamp,
            "slippage": slippage_percentage,
            "fees": {
                "maker_fee": fees_result["maker_fee"],
                "taker_fee": fees_result["taker_fee"],
                "total_fee": fees_result["total_fee"],
                "effective_rate": fees_result["effective_fee_rate"]
            },
            "market_impact": impact_result["impact_percentage"],
            "net_cost": net_cost_percentage, # Storing net_cost as percentage
            "maker_taker_proportion": maker_proportion,
            "internal_latency": processing_latency,
            "mid_price": mid_price,
            "debug_params_used": { # Optional: for debugging
                "quantity_usd": quantity_usd_to_use,
                "volatility": volatility_to_use,
                "fee_tier": fee_tier_to_use
            }
        }
        
        # Store historical data (keep last 100 ticks)
        if key not in continuous_metrics_history:
            continuous_metrics_history[key] = []
        
        continuous_metrics_history[key].append({**continuous_metrics[key]})
        if len(continuous_metrics_history[key]) > 100:
            continuous_metrics_history[key].pop(0)
            
    except Exception as e:
        logger.error(f"Error updating metrics for {key} with quantity {quantity_usd_to_use}: {str(e)}", exc_info=True)
        # Store error state
        continuous_metrics[key] = {
            "timestamp": time.time(), "error": str(e), "mid_price": calculate_mid_price(orderbook_data) if orderbook_data else 0,
            "slippage": 0.0, "fees": {"maker_fee": 0.0, "taker_fee": 0.0, "total_fee": 0.0, "effective_rate": 0.0},
            "market_impact": 0.0, "net_cost": 0.0, "maker_taker_proportion": 0.0, "internal_latency": 0.0
        }

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    global websocket_service
    
    # Initialize WebSocket service
    websocket_service = WebSocketService(
        WEBSOCKET_ENDPOINT,
        callback=process_orderbook
    )
    
    # Start WebSocket listener in the background
    asyncio.create_task(websocket_service.listen())
    logger.info("WebSocket service started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    if websocket_service:
        await websocket_service.disconnect()
    logger.info("Application shutting down")

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"status": "ok", "message": "Trade Cost Simulator API"}

@app.get("/assets")
def get_supported_assets():
    """Get list of supported assets."""
    return {"assets": SUPPORTED_ASSETS}

@app.post("/simulate", response_model=SimulationResponse)
async def simulate_trade(request: SimulationRequest):
    """
    Simulate trade execution and calculate expected costs.
    """
    # Start timing for latency measurement
    start_time = time.perf_counter()

    if request.exchange.lower() != "okx":
        raise HTTPException(status_code=400, detail="Only OKX exchange is supported")
    
    if request.asset not in SUPPORTED_ASSETS:
        raise HTTPException(status_code=400, detail=f"Asset not supported. Supported assets: {SUPPORTED_ASSETS}")
    
    if request.order_type.lower() != "market":
        raise HTTPException(status_code=400, detail="Only market orders are supported currently")
    
    key = f"{request.exchange.lower()}:{request.asset.lower()}"
    orderbook_data = latest_order_books.get(key)
    
    if not orderbook_data:
        raise HTTPException(status_code=503, detail="No order book data available. Please try again later.")
    
    mid_price = calculate_mid_price(orderbook_data)
    if mid_price <= 0:
        raise HTTPException(status_code=503, detail="Invalid mid_price from order book data.")

    order_value_quote = request.quantity  # Assuming quantity is already in USD equivalent
    order_size_base = order_value_quote / mid_price # Convert USD value to asset quantity
    
    maker_proportion = maker_taker_model.predict_maker_proportion(
        order_size_base,
        orderbook_data,
        request.volatility
    )
    
    fees_result = fee_model.calculate_fees(
        order_value_quote,
        request.fee_tier,
        maker_proportion
    )
    
    slippage_percentage = slippage_model.estimate_slippage(
        order_size_base,
        orderbook_data,
        request.volatility
    )
    slippage_absolute = (slippage_percentage / 100) * order_value_quote
    
    impact_result = market_impact_model.calculate_market_impact(
        order_size_base,
        mid_price,
        orderbook_data
    )
    market_impact_absolute = impact_result["impact_absolute"]
    
    net_cost_quote = slippage_absolute + fees_result["total_fee"] + market_impact_absolute
    
    end_time = time.perf_counter()
    processing_latency = (end_time - start_time) * 1000
    
    response = SimulationResponse(
        slippage=slippage_percentage,
        fees={
            "maker_fee": fees_result["maker_fee"],
            "taker_fee": fees_result["taker_fee"],
            "total_fee": fees_result["total_fee"],
            "effective_rate": fees_result["effective_fee_rate"]
        },
        market_impact=impact_result["impact_percentage"],
        net_cost=net_cost_quote, # SimulationResponse returns absolute net cost
        maker_taker_proportion=maker_proportion,
        internal_latency=processing_latency,
        timestamp=time.time()
    )
    
    # Store the successful simulation request parameters for the given asset
    asset_key_for_store = f"{request.exchange.lower()}:{request.asset.lower()}"
    last_simulation_params_store[asset_key_for_store] = request.model_copy(deep=True) # Store a deep copy
    logger.info(f"Stored simulation parameters for {asset_key_for_store}: {last_simulation_params_store[asset_key_for_store]}")
    
    return response

@app.get("/latency")
def get_latency_stats():
    """Get latency statistics for WebSocket processing."""
    if not websocket_service:
        raise HTTPException(status_code=503, detail="WebSocket service not initialized")
    
    return websocket_service.get_latency_stats()

@app.get("/status")
def get_status():
    """Get WebSocket connection status and available orderbooks."""
    global websocket_service, latest_order_books
    
    current_time = time.time()
    last_message_time = None
    ws_is_connected_flag = False

    if websocket_service:
        last_message_time = websocket_service.last_message_time
        ws_is_connected_flag = websocket_service.is_connected
    
    is_actively_connected = False
    if ws_is_connected_flag:
        if last_message_time and (current_time - last_message_time) < 10: # 10-second threshold
            is_actively_connected = True
    
    has_recent_data_activity = False
    if last_message_time and (current_time - last_message_time) < 15: # 15-second threshold for any data
        has_recent_data_activity = True
    
    orderbook_data_for_response = {}
    if latest_order_books: # Check if latest_order_books is not None or empty
        for key, ob_item in latest_order_books.items():
            try:
                if "timestamp" in ob_item and ob_item["timestamp"] is not None:
                    item_unix_ts = None
                    ts_value = ob_item["timestamp"]

                    if isinstance(ts_value, str):
                        # Attempt to parse ISO string, assuming it might have 'Z'
                        parsed_dt = datetime.fromisoformat(ts_value.replace('Z', '+00:00'))
                        item_unix_ts = parsed_dt.timestamp()
                    elif isinstance(ts_value, (int, float)):
                        # Assume it's already a Unix timestamp (in seconds)
                        item_unix_ts = float(ts_value)
                    
                    if item_unix_ts and (current_time - item_unix_ts) < 60:  # Only include data from the last 60 seconds
                        
                        top_ask_val = None
                        if ob_item.get("asks") and len(ob_item["asks"]) > 0 and len(ob_item["asks"][0]) > 0:
                            top_ask_val = ob_item["asks"][0][0]

                        top_bid_val = None
                        if ob_item.get("bids") and len(ob_item["bids"]) > 0 and len(ob_item["bids"][0]) > 0:
                            top_bid_val = ob_item["bids"][0][0]

                        orderbook_data_for_response[key] = {
                            "timestamp": ts_value, # Send original timestamp back
                            "ask_levels": len(ob_item.get("asks", [])),
                            "bid_levels": len(ob_item.get("bids", [])),
                            "top_ask": top_ask_val,
                            "top_bid": top_bid_val
                        }
                else:
                    logger.warning(f"Order book item for {key} missing timestamp or timestamp is None.")
            except Exception as e: # Catch a broader range of exceptions during item processing
                logger.error(f"Error processing order book item {key}: {str(e)}. Data: {ob_item}")
    
    # Determine final websocket_connected status
    final_websocket_connected_status = is_actively_connected or \
                                       (has_recent_data_activity and len(orderbook_data_for_response) > 0)

    status_payload = {
        "websocket_connected": final_websocket_connected_status,
        "available_orderbooks": orderbook_data_for_response,
        "last_message_time": last_message_time, # This is from websocket_service
        "current_time": current_time,
        "debug_info": { # Added for easier debugging from the response
            "websocket_service_initialized": websocket_service is not None,
            "websocket_service_is_connected_flag": ws_is_connected_flag,
            "raw_last_message_time_from_service": last_message_time,
            "calculated_is_actively_connected": is_actively_connected,
            "calculated_has_recent_data_activity": has_recent_data_activity,
            "count_latest_order_books_source": len(latest_order_books) if latest_order_books else 0,
            "count_filtered_orderbook_data_for_response": len(orderbook_data_for_response)
        }
    }
    
    try:
        # Log the exact payload being returned. Ensure it's JSON serializable.
        logger.info(f"/status endpoint returning: {json.dumps(status_payload, indent=2, default=str)}")
    except Exception as e:
        logger.error(f"Error serializing status_payload for logging: {str(e)}")
        # Fallback logging if json.dumps fails
        logger.info(f"/status endpoint returning (partial, serialization error): websocket_connected={status_payload['websocket_connected']}")

    return status_payload

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    connected_clients.add(websocket)
    
    try:
        # Send initial data
        if latest_order_books:
            initial_data = {
                "type": "initial",
                "timestamp": time.time(),
                "data": {k: v for k, v in latest_order_books.items()}
            }
            await websocket.send_text(json.dumps(initial_data))
        
        # Keep connection open and handle client messages
        while True:
            message = await websocket.receive_text()
            try:
                message_data = json.loads(message)
                
                # Handle ping-pong for keepalive
                if message_data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
                else:
                    # Echo other messages for debugging
                    await websocket.send_text(f"Received: {message}")
            except json.JSONDecodeError:
                # Not JSON, just echo back
                await websocket.send_text(f"Received: {message}")
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected")
        if websocket in connected_clients:
            connected_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

def calculate_mid_price(orderbook_data):
    """Calculate mid price from orderbook data."""
    asks = orderbook_data.get("asks", [])
    bids = orderbook_data.get("bids", [])
    
    if not asks or not bids:
        return 0
        
    best_ask = float(asks[0][0])
    best_bid = float(bids[0][0])
    
    return (best_ask + best_bid) / 2


@app.get("/continuous-metrics/{asset}")
def get_continuous_metrics(asset: str):
    """Get continuously updated metrics for a specific asset."""
    for key in continuous_metrics:
        if asset.lower() in key:
            return continuous_metrics[key]
    
    raise HTTPException(status_code=404, detail=f"No continuous metrics found for {asset}")

@app.get("/continuous-metrics-history/{asset}")
def get_continuous_metrics_history(asset: str):
    """Get historical metrics for a specific asset (last 100 ticks)."""
    for key in continuous_metrics_history:
        if asset.lower() in key:
            return continuous_metrics_history[key]
    
    raise HTTPException(status_code=404, detail=f"No continuous metrics history found for {asset}")

@app.websocket("/ws/metrics/{asset}")
async def websocket_metrics_endpoint(websocket: WebSocket, asset: str):
    """WebSocket endpoint for streaming real-time metrics."""
    await websocket.accept()
    
    key_pattern = asset.lower()
    connected_metrics_clients.setdefault(key_pattern, set()).add(websocket)
    
    try:
        while True:
            # Keep connection open and handle client messages
            message = await websocket.receive_text()
            await websocket.send_text(f"Received: {message}")
            
    except WebSocketDisconnect:
        if key_pattern in connected_metrics_clients and websocket in connected_metrics_clients[key_pattern]:
            connected_metrics_clients[key_pattern].remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket metrics error: {str(e)}")
        if key_pattern in connected_metrics_clients and websocket in connected_metrics_clients[key_pattern]:
            connected_metrics_clients[key_pattern].remove(websocket)

# Map of connected clients interested in metrics by asset
connected_metrics_clients = {}

async def broadcast_metrics_updates():
    """Broadcast metrics updates to interested WebSocket clients."""
    for key, metrics in continuous_metrics.items():
        for pattern, clients in connected_metrics_clients.items():
            if pattern in key:
                message = {
                    "type": "metrics_update",
                    "asset": key,
                    "data": metrics
                }
                
                disconnected_clients = set()
                for client in clients:
                    try:
                        await client.send_text(json.dumps(message))
                    except Exception:
                        disconnected_clients.add(client)
                
                # Remove disconnected clients
                clients.difference_update(disconnected_clients)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)