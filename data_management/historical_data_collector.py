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
from .high_frequency_collector import HighFrequencyCollector

class HistoricalDataCollector(HighFrequencyCollector):
    """
    Collects historical market data from crypto exchanges.
    Handles OHLCV data with support for long-term historical collection.
    """
    
    def __init__(
        self,
        exchange_id: str = 'okx',
        symbol: str = 'ETH-USDT-SWAP',
        interval: str = '15m',
        data_dir: str = 'data/historical',
        max_retries: int = 3,
        retry_delay: int = 5,
        batch_size: int = 1000,  # OKX allows up to 1000 candles per request
        max_workers: int = 4     # Number of parallel workers
    ):
        """
        Initialize the historical data collector.
        
        Args:
            exchange_id: Exchange ID (e.g., 'okx', 'binance)
            symbol: Trading pair symbol (e.g., 'ETH-USDT-SWAP' for OKX)
            interval: OHLCV interval
            data_dir: Directory to store collected data
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            batch_size: Number of candles to fetch per request
            max_workers: Maximum number of parallel workers
        """
        super().__init__(
            exchange_id=exchange_id,
            symbol=symbol,
            interval=interval,
            data_dir=data_dir,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize progress tracking
        self.progress = {
            'start_time': None,
            'end_time': None,
            'current_time': None,
            'total_candles': 0,
            'collected_candles': 0,
            'last_update_time': None,
            'candles_per_second': 0
        }
    
    def estimate_remaining_time(self) -> str:
        """
        Estimate remaining time based on current collection speed.
        
        Returns:
            String with estimated time remaining
        """
        if not self.progress['last_update_time'] or self.progress['candles_per_second'] == 0:
            return "Calculating..."
        
        remaining_candles = self.progress['total_candles'] - self.progress['collected_candles']
        if remaining_candles <= 0:
            return "Complete"
        
        seconds_remaining = remaining_candles / self.progress['candles_per_second']
        
        if seconds_remaining < 60:
            return f"{int(seconds_remaining)} seconds"
        elif seconds_remaining < 3600:
            return f"{int(seconds_remaining / 60)} minutes"
        else:
            hours = int(seconds_remaining / 3600)
            minutes = int((seconds_remaining % 3600) / 60)
            return f"{hours} hours, {minutes} minutes"
    
    def calculate_progress_percentage(self) -> float:
        """
        Calculate the percentage of data collected.
        
        Returns:
            Float between 0 and 100 representing progress percentage
        """
        if self.progress['total_candles'] == 0:
            return 0.0
        return (self.progress['collected_candles'] / self.progress['total_candles']) * 100
    
    async def fetch_historical_ohlcv(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a specific time range.
        
        Args:
            start_time: Start time for data collection
            end_time: End time for data collection
            
        Returns:
            DataFrame with historical OHLCV data
        """
        all_data = []
        current_time = start_time
        last_update_time = time.time()
        candles_since_last_update = 0
        
        while current_time < end_time:
            try:
                # Convert datetime to milliseconds timestamp
                since = int(current_time.timestamp() * 1000)
                
                # Calculate end time for this batch
                batch_end = min(
                    current_time + timedelta(minutes=self.batch_size * 15),  # 15 minutes per candle
                    end_time
                )
                batch_end_ts = int(batch_end.timestamp() * 1000)
                
                # Fetch batch of OHLCV data
                ohlcv = await self.fetch_ohlcv(since=since)
                
                if ohlcv.empty:
                    break
                
                all_data.append(ohlcv)
                
                # Update current time to the last timestamp + interval
                current_time = ohlcv.index[-1] + pd.Timedelta(self.interval)
                
                # Update progress
                self.progress['current_time'] = current_time
                self.progress['collected_candles'] += len(ohlcv)
                
                # Update collection speed metrics
                current_time_sec = time.time()
                time_diff = current_time_sec - last_update_time
                if time_diff >= 5:  # Update speed every 5 seconds
                    candles_since_last_update += len(ohlcv)
                    self.progress['candles_per_second'] = candles_since_last_update / time_diff
                    self.progress['last_update_time'] = current_time_sec
                    candles_since_last_update = 0
                    last_update_time = current_time_sec
                    
                    # Log progress
                    progress_pct = self.calculate_progress_percentage()
                    remaining_time = self.estimate_remaining_time()
                    self.logger.info(
                        f"Progress: {progress_pct:.1f}% | "
                        f"Collected: {self.progress['collected_candles']}/{self.progress['total_candles']} candles | "
                        f"Speed: {self.progress['candles_per_second']:.1f} candles/sec | "
                        f"Remaining: {remaining_time}"
                    )
                
                # Save progress after each batch
                self.save_progress()
                
                # Save data periodically (every 10 batches)
                if len(all_data) % 10 == 0:
                    combined_data = pd.concat(all_data)
                    self.ohlcv_data = combined_data
                    self.save_data()
                    self.logger.info(f"Saved {len(combined_data)} candles to disk")
                
                # Respect rate limits - OKX has a rate limit of 20 requests per second
                await asyncio.sleep(0.1)  # Slightly more conservative than 20 requests per second
                
            except Exception as e:
                self.logger.error(f"Error fetching historical data: {str(e)}")
                # Save progress before retrying
                self.save_progress()
                await asyncio.sleep(self.retry_delay)
        
        # Combine all data
        if all_data:
            return pd.concat(all_data)
        return pd.DataFrame()
    
    async def collect_historical_data(
        self,
        start_time: datetime,
        end_time: datetime = None,
        save_interval: int = 1000
    ):
        """
        Collect historical OHLCV and order book data.
        
        Args:
            start_time: Start time for data collection
            end_time: End time for data collection (defaults to now)
            save_interval: Number of candles to collect before saving
        """
        if end_time is None:
            end_time = datetime.now()
            
        self.progress['start_time'] = start_time
        self.progress['end_time'] = end_time
        self.progress['current_time'] = start_time
        
        # Convert to timestamps
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        # Initialize data storage
        ohlcv_data = []
        order_book_data = []
        
        while self.progress['current_time'] < end_time:
            try:
                # Fetch OHLCV data
                ohlcv_batch = await self.fetch_ohlcv(since=start_timestamp)
                if not ohlcv_batch.empty:
                    ohlcv_data.append(ohlcv_batch)
                    
                    # Fetch order book data for each OHLCV timestamp
                    for timestamp in ohlcv_batch.index:
                        try:
                            # Add delay to respect rate limits
                            await asyncio.sleep(0.1)  # 100ms delay between requests
                            
                            order_book = await self.fetch_order_book()
                            order_book['ohlcv_timestamp'] = timestamp
                            order_book_data.append(order_book)
                            
                            # Log progress
                            self.logger.info(f"Collected order book for {timestamp}")
                            
                        except Exception as e:
                            self.logger.error(f"Error collecting order book for {timestamp}: {str(e)}")
                            continue
                    
                    # Update progress
                    self.progress['current_time'] = ohlcv_batch.index[-1]
                    self.progress['collected_candles'] += len(ohlcv_batch)
                    
                    # Save data periodically
                    if len(ohlcv_data) >= save_interval:
                        self._save_batch(ohlcv_data, order_book_data)
                        ohlcv_data = []
                        order_book_data = []
                
                # Update start timestamp for next batch
                start_timestamp = int(self.progress['current_time'].timestamp() * 1000) + 1
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error collecting historical data: {str(e)}")
                await asyncio.sleep(self.retry_delay)
        
        # Save any remaining data
        if ohlcv_data:
            self._save_batch(ohlcv_data, order_book_data)
    
    def _save_batch(self, ohlcv_data: List[pd.DataFrame], order_book_data: List[Dict]):
        """Save a batch of collected data."""
        try:
            # Combine OHLCV data
            ohlcv_df = pd.concat(ohlcv_data)
            
            # Convert order book data to DataFrame
            order_book_df = pd.DataFrame(order_book_data)
            
            # Save OHLCV data
            ohlcv_path = self.data_dir / f"{self.symbol}_ohlcv_{self.interval}.csv"
            ohlcv_df.to_csv(ohlcv_path)
            
            # Save order book data with proper formatting
            order_book_path = self.data_dir / f"{self.symbol}_orderbook.json"
            
            # Convert timestamp to datetime for better readability
            if 'timestamp' in order_book_df.columns:
                order_book_df['timestamp'] = pd.to_datetime(order_book_df['timestamp'], unit='ms')
            if 'ohlcv_timestamp' in order_book_df.columns:
                order_book_df['ohlcv_timestamp'] = pd.to_datetime(order_book_df['ohlcv_timestamp'])
            
            # Save with proper formatting
            order_book_df.to_json(
                order_book_path,
                orient='records',
                date_format='iso',
                indent=2
            )
            
            self.logger.info(f"Saved batch of {len(ohlcv_df)} OHLCV records and {len(order_book_df)} order book records")
            
        except Exception as e:
            self.logger.error(f"Error saving batch: {str(e)}")
    
    def save_progress(self):
        """Save collection progress to disk."""
        progress_path = self.data_dir / f"{self.symbol.replace('-', '_')}_progress.json"
        with open(progress_path, 'w') as f:
            json.dump({
                'start_time': self.progress['start_time'].isoformat() if self.progress['start_time'] else None,
                'end_time': self.progress['end_time'].isoformat() if self.progress['end_time'] else None,
                'current_time': self.progress['current_time'].isoformat() if self.progress['current_time'] else None,
                'total_candles': self.progress['total_candles'],
                'collected_candles': self.progress['collected_candles']
            }, f)
    
    def load_progress(self) -> Dict:
        """Load collection progress from disk."""
        progress_path = self.data_dir / f"{self.symbol.replace('-', '_')}_progress.json"
        if progress_path.exists():
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                # Convert ISO format strings back to datetime
                for key in ['start_time', 'end_time', 'current_time']:
                    if progress[key]:
                        progress[key] = datetime.fromisoformat(progress[key])
                return progress
        return self.progress
    
    def validate_historical_data(self) -> Dict[str, bool]:
        """
        Validate historical data for quality and completeness.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'data_complete': False,
            'data_quality': False,
            'time_continuity': False
        }
        
        try:
            if not self.ohlcv_data.empty:
                # Check for missing values
                missing_values = self.ohlcv_data.isnull().sum().sum()
                validation_results['data_quality'] = missing_values == 0
                
                # Check time continuity
                expected_interval = pd.Timedelta(self.interval)
                time_diffs = self.ohlcv_data.index.to_series().diff()
                validation_results['time_continuity'] = all(
                    diff == expected_interval for diff in time_diffs[1:]
                )
                
                # Check data completeness
                expected_candles = self.progress['total_candles']
                actual_candles = len(self.ohlcv_data)
                validation_results['data_complete'] = abs(
                    actual_candles - expected_candles
                ) <= 1  # Allow for 1 candle difference due to timezone issues
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating historical data: {str(e)}")
            return validation_results
    
    def load_data(self) -> pd.DataFrame:
        """Load collected data from disk."""
        data_path = self.data_dir / f"{self.symbol.replace('-', '_')}_data.csv"
        if data_path.exists():
            return pd.read_csv(data_path, index_col=0, parse_dates=True)
        return pd.DataFrame()  # Return empty DataFrame instead of tuple

    def load_orderbook_data(self) -> pd.DataFrame:
        """Load collected order book data from disk."""
        order_book_path = self.data_dir / f"{self.symbol.replace('-', '_')}_orderbook.json"
        if order_book_path.exists():
            try:
                # Load order book data
                order_book_data = pd.read_json(order_book_path)
                
                # Convert timestamp to datetime if it exists
                if 'timestamp' in order_book_data.columns:
                    order_book_data['timestamp'] = pd.to_datetime(order_book_data['timestamp'])
                    order_book_data.set_index('timestamp', inplace=True)
                
                return order_book_data
            except Exception as e:
                self.logger.error(f"Error loading order book data: {str(e)}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    async def fetch_order_book(self) -> Dict:
        """
        Fetch order book snapshot from the exchange.
        
        Returns:
            Dictionary with order book data including:
            - timestamp: Exchange timestamp in milliseconds
            - bids: List of [price, amount] for buy orders
            - asks: List of [price, amount] for sell orders
        """
        for attempt in range(self.max_retries):
            try:
                # Check if exchange supports order book fetching
                if not self.exchange.has['fetchOrderBook']:
                    raise NotImplementedError(f"Exchange {self.exchange_id} does not support order book fetching")

                # Fetch order book with specified depth
                order_book = self.exchange.fetch_order_book(
                    symbol=self.symbol,
                    limit=self.order_book_depth
                )
                
                # Log raw order book data
                self.logger.info(f"Raw order book data: {json.dumps(order_book, indent=2)}")
                
                # Extract and validate data
                timestamp = order_book.get('timestamp')
                if timestamp is None:
                    timestamp = int(datetime.now().timestamp() * 1000)
                
                bids = order_book.get('bids', [])
                asks = order_book.get('asks', [])
                
                # Log bids and asks structure
                self.logger.info(f"Bids structure (first 3): {bids[:3]}")
                self.logger.info(f"Asks structure (first 3): {asks[:3]}")
                
                # Validate order book data
                if not bids or not asks:
                    raise ValueError("Empty order book received")
                
                # Handle OKX's order book format which includes additional information
                # OKX returns [price, amount, order_id, timestamp] for each level
                validated_bids = []
                validated_asks = []
                
                for bid in bids[:self.order_book_depth]:
                    if len(bid) >= 2:  # Ensure we have at least price and amount
                        validated_bids.append([float(bid[0]), float(bid[1])])
                
                for ask in asks[:self.order_book_depth]:
                    if len(ask) >= 2:  # Ensure we have at least price and amount
                        validated_asks.append([float(ask[0]), float(ask[1])])
                
                # Log validated data
                self.logger.info(f"Validated bids (first 3): {validated_bids[:3]}")
                self.logger.info(f"Validated asks (first 3): {validated_asks[:3]}")
                
                return {
                    'timestamp': timestamp,
                    'bids': validated_bids,
                    'asks': validated_asks,
                    'exchange_timestamp': order_book.get('datetime'),
                    'nonce': order_book.get('nonce')
                }
                
            except Exception as e:
                self.logger.error(f"Error fetching order book: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
    
    async def collect_orderbook_data(
        self,
        start_time: datetime,
        end_time: datetime,
        save_interval: int = 100
    ):
        """
        Collect order book data for a specific time period using WebSocket.
        
        Args:
            start_time: Start time for data collection
            end_time: End time for data collection
            save_interval: Number of snapshots to collect before saving
        """
        try:
            # Initialize WebSocket connection
            ws_url = self.exchange.urls['ws']
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    # Subscribe to order book updates
                    subscribe_message = {
                        "op": "subscribe",
                        "args": [{
                            "channel": "books",
                            "instId": self.symbol
                        }]
                    }
                    await ws.send_json(subscribe_message)
                    
                    # Initialize data storage
                    order_book_data = []
                    collection_count = 0
                    
                    # Start collection
                    self.logger.info(f"Starting order book collection from {start_time} to {end_time}")
                    
                    while datetime.now() < end_time:
                        try:
                            # Receive order book update
                            msg = await ws.receive_json()
                            
                            if 'data' in msg:
                                order_book = {
                                    'timestamp': int(datetime.now().timestamp() * 1000),
                                    'bids': msg['data'][0]['bids'],
                                    'asks': msg['data'][0]['asks'],
                                    'exchange_timestamp': msg.get('ts')
                                }
                                
                                order_book_data.append(order_book)
                                collection_count += 1
                                
                                # Log progress
                                if collection_count % 10 == 0:
                                    self.logger.info(f"Collected {collection_count} order book snapshots")
                                
                                # Save data periodically
                                if len(order_book_data) >= save_interval:
                                    self._save_batch([], order_book_data)
                                    order_book_data = []
                                    self.logger.info("Saved batch of order book data")
                            
                        except Exception as e:
                            self.logger.error(f"Error receiving order book data: {str(e)}")
                            await asyncio.sleep(self.retry_delay)
                    
                    # Save any remaining data
                    if order_book_data:
                        self._save_batch([], order_book_data)
                    
                    self.logger.info(f"Order book collection completed. Total snapshots: {collection_count}")
                    
        except Exception as e:
            self.logger.error(f"Error in order book collection: {str(e)}")
            raise 

    async def fetch_trade_history(self, since: int, limit: int = 1000) -> pd.DataFrame:
        """Fetch trade history from the exchange.
        
        Args:
            since: Timestamp in milliseconds to fetch trades from
            limit: Maximum number of trades to fetch
            
        Returns:
            DataFrame containing trade history
        """
        try:
            # Use run_in_executor to run the synchronous fetch_trades in a thread pool
            trades = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_trades(
                    symbol=self.symbol,
                    since=since,
                    limit=limit
                )
            )
            
            if not trades:
                return pd.DataFrame()
            
            # Convert to DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'price', 'amount', 'side']
            if not all(col in trades_df.columns for col in required_columns):
                raise ValueError(f"Missing required columns in trade data. Required: {required_columns}")
            
            # Convert timestamp to datetime
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], unit='ms')
            
            # Sort by timestamp
            trades_df = trades_df.sort_values('timestamp')
            
            return trades_df
            
        except Exception as e:
            self.logger.error(f"Error fetching trade history: {str(e)}")
            raise

    def reconstruct_orderbook(
        self,
        trades_df: pd.DataFrame,
        depth: int = 20
    ) -> Dict:
        """
        Reconstruct order book state from trade history.
        
        Args:
            trades_df: DataFrame with trade history
            depth: Depth of order book to reconstruct
            
        Returns:
            Dictionary with reconstructed order book state
        """
        try:
            # Log input trade data structure
            self.logger.info(f"Input trades DataFrame structure:")
            self.logger.info(f"Columns: {trades_df.columns.tolist()}")
            self.logger.info(f"First few trades:\n{trades_df.head().to_string()}")
            
            # Initialize order book
            bids = {}  # price -> amount
            asks = {}  # price -> amount
            
            # Process trades in chronological order
            for _, trade in trades_df.iterrows():
                price = float(trade['price'])
                amount = float(trade['amount'])
                side = trade['side']
                
                if side == 'buy':
                    # Update asks (seller's side)
                    asks[price] = asks.get(price, 0) + amount
                else:
                    # Update bids (buyer's side)
                    bids[price] = bids.get(price, 0) + amount
            
            # Log intermediate state
            self.logger.info(f"Intermediate state - Number of bid levels: {len(bids)}")
            self.logger.info(f"Intermediate state - Number of ask levels: {len(asks)}")
            
            # Convert to sorted lists
            sorted_bids = sorted(
                [[price, amount] for price, amount in bids.items() if amount > 0],
                key=lambda x: x[0],
                reverse=True
            )[:depth]
            
            sorted_asks = sorted(
                [[price, amount] for price, amount in asks.items() if amount > 0],
                key=lambda x: x[0]
            )[:depth]
            
            # Log final reconstructed order book
            self.logger.info(f"Reconstructed order book - First 3 bid levels: {sorted_bids[:3]}")
            self.logger.info(f"Reconstructed order book - First 3 ask levels: {sorted_asks[:3]}")
            
            # Get the last timestamp from the trades DataFrame
            last_timestamp = trades_df['timestamp'].iloc[-1]
            if isinstance(last_timestamp, pd.Timestamp):
                timestamp_ms = int(last_timestamp.timestamp() * 1000)
            else:
                # If it's already a Unix timestamp in milliseconds
                timestamp_ms = int(last_timestamp)
            
            return {
                'timestamp': timestamp_ms,
                'bids': sorted_bids,
                'asks': sorted_asks
            }
            
        except Exception as e:
            self.logger.error(f"Error reconstructing order book: {str(e)}")
            raise

    async def collect_historical_orderbook(
        self,
        start_time: datetime,
        end_time: datetime,
        save_interval: int = 100
    ):
        """Collect historical order book data by reconstructing from trade history.
        
        Args:
            start_time: Start time for data collection
            end_time: End time for data collection
            save_interval: Number of snapshots to collect before saving
        """
        self.logger.info(f"Starting historical order book collection from {start_time} to {end_time}")
        
        current_time = start_time
        order_book_data = []
        snapshot_count = 0
        
        while current_time < end_time:
            try:
                # Convert current time to milliseconds timestamp
                since = int(current_time.timestamp() * 1000)
                
                # Fetch trade history for this period
                trades_df = await self.fetch_trade_history(since=since)
                
                if not trades_df.empty:
                    # Reconstruct order book state
                    order_book = self.reconstruct_orderbook(trades_df)
                    
                    # Add timestamp
                    order_book['timestamp'] = current_time
                    
                    # Add to collection
                    order_book_data.append(order_book)
                    snapshot_count += 1
                    
                    # Save batch if we've reached the save interval
                    if snapshot_count >= save_interval:
                        self._save_batch([], order_book_data)
                        order_book_data = []
                        snapshot_count = 0
                        self.logger.info(f"Saved batch of {save_interval} order book snapshots")
                
                # Move to next 15-minute interval
                current_time += timedelta(minutes=15)
                
                # Respect rate limits
                await asyncio.sleep(0.1)  # 100ms delay between requests
                
            except Exception as e:
                self.logger.error(f"Error collecting historical order book: {str(e)}")
                await asyncio.sleep(self.retry_delay)
                continue
        
        # Save any remaining data
        if order_book_data:
            self._save_batch([], order_book_data)
            self.logger.info(f"Saved final batch of {len(order_book_data)} order book snapshots")
        
        self.logger.info("Historical order book collection completed") 