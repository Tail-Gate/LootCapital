import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import os

class MarketData:
    """
    Class for handling market data operations.
    """
    
    def __init__(self, data_source_path=None):
        """
        Initialize the MarketData class.
        
        Args:
            data_source_path: Path to local data source directory.
                             Defaults to 'data' for local data
        """
        self.data_cache = {}
        # Set default local path if none provided
        self.data_source_path = data_source_path or 'data'
        
        # Ensure the data directory exists
        if not os.path.exists(self.data_source_path):
            os.makedirs(self.data_source_path, exist_ok=True)
            print(f"Created data directory: {self.data_source_path}")
        
    def get_data(self, symbol: Union[str, List[str]], start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Get market data for one or more symbols.
        
        Args:
            symbol: The trading symbol(s) (e.g., 'BTC', 'ETH' or ['BTC', 'ETH'])
            start_time: Optional start time for data range
            end_time: Optional end time for data range
            
        Returns:
            DataFrame or dictionary of DataFrames containing market data
        """
        # Handle single symbol
        if isinstance(symbol, str):
            # Only use cached data if no specific time range is requested
            if symbol in self.data_cache and start_time is None and end_time is None:
                return self.data_cache[symbol]
            
            # Try to load real data first, fall back to synthetic if not available
            data = self._load_real_data(symbol, start_time, end_time)
            if data is None:
                data = self._generate_synthetic_data(symbol, start_time, end_time)
            
            # Only cache if no specific time range was requested
            if start_time is None and end_time is None:
                self.data_cache[symbol] = data
            return data
            
        # Handle multiple symbols
        elif isinstance(symbol, list):
            result = {}
            for sym in symbol:
                result[sym] = self.get_data(sym, start_time, end_time)
            return result
            
        else:
            raise TypeError("symbol must be a string or list of strings")
    
    def _load_real_data(self, symbol: str, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Load real market data from CSV files (local only).
        
        Args:
            symbol: The trading symbol
            start_time: Optional start time for data range
            end_time: Optional end time for data range
            
        Returns:
            DataFrame containing real market data, or None if not available
        """
        # Map symbols to file names
        symbol_to_file = {
            'ETH/USD': 'ETH-USDT-SWAP_ohlcv_15m.csv',
            'ETH/USDT': 'ETH-USDT-SWAP_ohlcv_15m.csv',
            'BTC/USD': 'BTC-USDT-SWAP_ohlcv_15m.csv',
            'BTC/USDT': 'BTC-USDT-SWAP_ohlcv_15m.csv',
        }
        
        # Check if we have a mapping for this symbol
        if symbol not in symbol_to_file:
            print(f"No real data file mapping found for {symbol}, using synthetic data")
            return None
        
        file_name = symbol_to_file[symbol]
        
        # Local path
        file_path = os.path.join(self.data_source_path, 'historical', file_name)
        print(f"Attempting to load from local: {file_path}")
        
        try:
            # Load the CSV file
            print(f"Loading real data from: {file_path}")
            
            # Use regular pandas read_csv for local files
            data = pd.read_csv(file_path)
            
            # Convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
            
            # Filter by time range if provided
            if start_time is not None:
                data = data[data.index >= start_time]
            if end_time is not None:
                data = data[data.index <= end_time]
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                print(f"Missing required columns in {file_path}, using synthetic data")
                return None
            
            print(f"Loaded {len(data)} rows of real data for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error loading real data from {file_path}: {e}")
            return None
            
    def _generate_synthetic_data(self, symbol: str, start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate synthetic market data for testing.
        
        Args:
            symbol: The trading symbol
            start_time: Optional start time for data range
            end_time: Optional end time for data range
            
        Returns:
            DataFrame containing synthetic market data
        """
        # Set default time range if not provided
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=30)
            
        # Generate time index
        time_index = pd.date_range(start=start_time, end=end_time, freq='1h')
        
        # Generate synthetic price data
        np.random.seed(hash(symbol) % 10000)  # Use symbol as seed for reproducibility
        returns = np.random.normal(0.0001, 0.02, len(time_index))
        price = 100 * (1 + returns).cumprod()
        
        # Generate synthetic volume data
        volume = np.random.lognormal(10, 1, len(time_index))
        
        # Create DataFrame
        data = pd.DataFrame({
            'open': price * (1 + np.random.normal(0, 0.001, len(time_index))),
            'high': price * (1 + np.abs(np.random.normal(0, 0.002, len(time_index)))),
            'low': price * (1 - np.abs(np.random.normal(0, 0.002, len(time_index)))),
            'close': price,
            'volume': volume
        }, index=time_index)
        
        return data
        
    def get_latest_data(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
        """
        Get the most recent market data for a symbol.
        
        Args:
            symbol: The trading symbol
            lookback: Number of periods to look back
            
        Returns:
            DataFrame containing recent market data
        """
        data = self.get_data(symbol)
        return data.iloc[-lookback:]
        
    def update_data(self, symbol: str, new_data: pd.DataFrame) -> None:
        """
        Update market data for a symbol.
        
        Args:
            symbol: The trading symbol
            new_data: New market data to add
        """
        if symbol in self.data_cache:
            self.data_cache[symbol] = pd.concat([self.data_cache[symbol], new_data])
        else:
            self.data_cache[symbol] = new_data 

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest close price for a symbol.

        Args:
            symbol: The trading symbol

        Returns:
            The latest close price, or None if not available
        """
        data = self.get_data(symbol)
        if data is not None and not data.empty and 'close' in data.columns:
            return data['close'].iloc[-1]
        return None 