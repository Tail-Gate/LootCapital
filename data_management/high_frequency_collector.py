import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import ccxt
import asyncio
import aiohttp
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import os

class HighFrequencyCollector:
    """
    Collects high-frequency market data from crypto exchanges.
    Handles OHLCV data and order book snapshots.
    """
    
    def __init__(
        self,
        exchange_id: str = 'binance',
        symbol: str = 'ETH/USD',
        interval: str = '15m',
        order_book_depth: int = 10,
        data_dir: str = 'data/high_frequency',
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        """
        Initialize the high-frequency data collector.
        
        Args:
            exchange_id: Exchange ID (e.g., 'binance', 'coinbase')
            symbol: Trading pair symbol
            interval: OHLCV interval
            order_book_depth: Number of order book levels to collect
            data_dir: Directory to store collected data
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.interval = interval
        self.order_book_depth = order_book_depth
        self.data_dir = Path(data_dir)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exchange
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # OKX uses 'swap' for perpetual futures
                'defaultContractType': 'perpetual'
            }
        })
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize data queues
        self.ohlcv_queue = Queue()
        self.order_book_queue = Queue()
        
        # Initialize data storage
        self.ohlcv_data = pd.DataFrame()
        self.order_book_data = pd.DataFrame()
        
        # Initialize locks for thread safety
        self.ohlcv_lock = threading.Lock()
        self.order_book_lock = threading.Lock()
    
    async def fetch_ohlcv(self, since: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from the exchange.
        
        Args:
            since: Timestamp in milliseconds to fetch data from
            
        Returns:
            DataFrame with OHLCV data
        """
        for attempt in range(self.max_retries):
            try:
                # Use fetch_ohlcv with proper parameters for OKX
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.interval,
                    since=since,
                    limit=1000  # OKX allows up to 1000 candles per request
                )
                
                if not ohlcv:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
                
            except Exception as e:
                self.logger.error(f"Error fetching OHLCV data: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
    
    async def fetch_order_book(self) -> Dict:
        """
        Fetch order book snapshot from the exchange.
        
        Returns:
            Dictionary with order book data
        """
        for attempt in range(self.max_retries):
            try:
                # Use synchronous fetch_order_book since CCXT doesn't support async
                order_book = self.exchange.fetch_order_book(
                    symbol=self.symbol,
                    limit=self.order_book_depth
                )
                
                # Extract top levels
                bids = order_book['bids'][:self.order_book_depth]
                asks = order_book['asks'][:self.order_book_depth]
                
                return {
                    'timestamp': datetime.now(),
                    'bids': bids,
                    'asks': asks
                }
                
            except Exception as e:
                self.logger.error(f"Error fetching order book: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
    
    async def collect_data(self, duration_minutes: int = 60):
        """
        Collect data for a specified duration.
        
        Args:
            duration_minutes: Duration to collect data for in minutes
        """
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            try:
                # Fetch OHLCV data
                ohlcv_data = await self.fetch_ohlcv()
                with self.ohlcv_lock:
                    self.ohlcv_data = pd.concat([self.ohlcv_data, ohlcv_data])
                    self.ohlcv_queue.put(ohlcv_data)
                
                # Fetch order book data
                order_book_data = await self.fetch_order_book()
                with self.order_book_lock:
                    self.order_book_data = pd.concat([
                        self.order_book_data,
                        pd.DataFrame([order_book_data])
                    ])
                    self.order_book_queue.put(order_book_data)
                
                # Save data periodically
                if len(self.ohlcv_data) % 100 == 0:
                    self.save_data()
                
                # Wait for next interval
                await asyncio.sleep(60)  # Wait for 1 minute
                
            except Exception as e:
                self.logger.error(f"Error in data collection: {str(e)}")
                await asyncio.sleep(self.retry_delay)
    
    def save_data(self):
        """Save collected data to disk."""
        try:
            # Save OHLCV data
            ohlcv_path = self.data_dir / f"{self.symbol.replace('/', '_')}_ohlcv_{self.interval}.csv"
            with self.ohlcv_lock:
                self.ohlcv_data.to_csv(ohlcv_path)
            
            # Save order book data
            order_book_path = self.data_dir / f"{self.symbol.replace('/', '_')}_orderbook.json"
            with self.order_book_lock:
                self.order_book_data.to_json(order_book_path, orient='records')
            
            self.logger.info(f"Data saved to {self.data_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load previously collected data from disk.
        
        Returns:
            Tuple of (OHLCV data, Order book data)
        """
        try:
            # Load OHLCV data
            ohlcv_path = self.data_dir / f"{self.symbol.replace('/', '_')}_ohlcv_{self.interval}.csv"
            if ohlcv_path.exists():
                ohlcv_data = pd.read_csv(ohlcv_path, index_col='timestamp', parse_dates=True)
            else:
                ohlcv_data = pd.DataFrame()
            
            # Load order book data
            order_book_path = self.data_dir / f"{self.symbol.replace('/', '_')}_orderbook.json"
            if order_book_path.exists():
                order_book_data = pd.read_json(order_book_path)
            else:
                order_book_data = pd.DataFrame()
            
            return ohlcv_data, order_book_data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def validate_data(self) -> Dict[str, bool]:
        """
        Validate collected data for quality and completeness.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'ohlcv_complete': False,
            'order_book_complete': False,
            'data_quality': False
        }
        
        try:
            # Check OHLCV data
            if not self.ohlcv_data.empty:
                # Check for missing values
                missing_values = self.ohlcv_data.isnull().sum().sum()
                validation_results['ohlcv_complete'] = missing_values == 0
                
                # Check for data quality
                price_valid = all(self.ohlcv_data['high'] >= self.ohlcv_data['low'])
                volume_valid = all(self.ohlcv_data['volume'] >= 0)
                validation_results['data_quality'] = price_valid and volume_valid
            
            # Check order book data
            if not self.order_book_data.empty:
                # Check for missing values
                missing_values = self.order_book_data.isnull().sum().sum()
                validation_results['order_book_complete'] = missing_values == 0
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            return validation_results
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Save any remaining data
            self.save_data()
            
            # Clear queues
            while not self.ohlcv_queue.empty():
                self.ohlcv_queue.get()
            while not self.order_book_queue.empty():
                self.order_book_queue.get()
            
            # Clear data
            with self.ohlcv_lock:
                self.ohlcv_data = pd.DataFrame()
            with self.order_book_lock:
                self.order_book_data = pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise 