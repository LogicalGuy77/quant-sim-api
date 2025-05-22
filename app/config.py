import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OKX API Credentials - now loaded from environment variables
OKX_API_KEY = os.getenv("OKX_API_KEY", "")
OKX_SECRET_KEY = os.getenv("OKX_SECRET_KEY", "")
OKX_IP = os.getenv("OKX_IP", "")
OKX_API_KEY_NAME = os.getenv("OKX_API_KEY_NAME", "goquant")
OKX_API_PERMISSIONS = os.getenv("OKX_API_PERMISSIONS", "Read")

# WebSocket endpoint
WEBSOCKET_ENDPOINT = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"

# Supported spot assets
SUPPORTED_ASSETS = ["BTC-USDT-SWAP"]

# Fee tiers for OKX (actual values from OKX docs would be needed)
OKX_FEE_TIERS = {
    "VIP0": {"maker": 0.0008, "taker": 0.0010},
    "VIP1": {"maker": 0.0007, "taker": 0.0009},
    "VIP2": {"maker": 0.0006, "taker": 0.0008},
    "VIP3": {"maker": 0.0005, "taker": 0.0007},
    "VIP4": {"maker": 0.0004, "taker": 0.0006},
    "VIP5": {"maker": 0.0003, "taker": 0.0005}
}