import json
import asyncio
import websockets
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketService:
    """
    Service for handling WebSocket connections to order book data streams.
    """
    def __init__(self, websocket_url, callback=None):
        """
        Initialize the WebSocket service.
        
        Parameters:
        - websocket_url: URL of the WebSocket endpoint
        - callback: Function to call with each received message
        """
        self.websocket_url = websocket_url
        self.callback = callback
        self.websocket = None
        self.is_connected = False
        self.last_message_time = None
        self.processing_times = []
        self.reconnect_delay = 1  # Initial reconnect delay in seconds
        self.max_reconnect_delay = 60  # Maximum reconnect delay
    
    async def connect(self):
        """Establish WebSocket connection."""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            self.is_connected = True
            self.reconnect_delay = 1  # Reset reconnect delay on successful connection
            logger.info(f"Connected to WebSocket at {self.websocket_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {str(e)}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Close WebSocket connection."""
        if self.websocket and self.is_connected:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from WebSocket")
    
    async def listen(self):
        """
        Listen for messages on the WebSocket connection.
        Process each message and calculate latency metrics.
        """
        while True:
            if not self.is_connected:
                try:
                    success = await self.connect()
                    if not success:
                        # Exponential backoff for reconnection attempts
                        await asyncio.sleep(self.reconnect_delay)
                        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                        continue
                except Exception as e:
                    logger.error(f"Error during reconnection: {str(e)}")
                    await asyncio.sleep(self.reconnect_delay)
                    continue
            
            try:
                # Wait for message
                message = await self.websocket.recv()
                receive_time = time.time()
                
                # Parse message
                parsed_message = self._parse_message(message)
                
                # Calculate processing latency
                if parsed_message and "timestamp" in parsed_message:
                    message_time = self._parse_timestamp(parsed_message["timestamp"])
                    latency = (receive_time - message_time) * 1000  # Convert to milliseconds
                    self.processing_times.append(latency)
                    
                    # Keep only the last 1000 measurements
                    if len(self.processing_times) > 1000:
                        self.processing_times.pop(0)
                
                # Process message through callback
                if self.callback and parsed_message:
                    process_start = time.time()
                    await self.callback(parsed_message)
                    process_end = time.time()
                    processing_time = (process_end - process_start) * 1000  # ms
                    
                    # Log processing time
                    logger.debug(f"Message processing time: {processing_time:.2f}ms")
                
                self.last_message_time = receive_time
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.is_connected = False
                await asyncio.sleep(self.reconnect_delay)
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                # Don't disconnect on processing errors
    
    def _parse_message(self, message):
        """Parse WebSocket message into a Python object."""
        try:
            return json.loads(message)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {str(e)}")
            return None
    
    def _parse_timestamp(self, timestamp_str):
        """Parse ISO timestamp string into Unix timestamp."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.timestamp()
        except ValueError:
            logger.error(f"Failed to parse timestamp: {timestamp_str}")
            return time.time()  # Fallback to current time
    
    def get_latency_stats(self):
        """Get statistics about message processing latency."""
        if not self.processing_times:
            return {
                "min_latency": 0,
                "max_latency": 0,
                "avg_latency": 0,
                "p95_latency": 0,
                "p99_latency": 0,
                "sample_count": 0
            }
        
        # Calculate statistics
        import numpy as np
        times = np.array(self.processing_times)
        
        return {
            "min_latency": float(np.min(times)),
            "max_latency": float(np.max(times)),
            "avg_latency": float(np.mean(times)),
            "p95_latency": float(np.percentile(times, 95)),
            "p99_latency": float(np.percentile(times, 99)),
            "sample_count": len(times)
        }