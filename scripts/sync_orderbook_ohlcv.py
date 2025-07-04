import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.order_book_features import calculate_order_flow_imbalance, calculate_volume_pressure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sync_orderbook_ohlcv.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_data(data_dir: str = 'data/historical') -> tuple:
    """
    Load OHLCV and order book data from the specified directory.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Tuple of (OHLCV data, Order book data)
    """
    data_path = Path(data_dir)
    
    # Load OHLCV data
    ohlcv_path = data_path / "ETH-USDT-SWAP_ohlcv_15m.csv"
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV data not found at {ohlcv_path}")
    
    ohlcv_data = pd.read_csv(ohlcv_path, index_col='timestamp', parse_dates=True)
    logger.info(f"Loaded {len(ohlcv_data)} OHLCV records")
    
    # Load order book data
    orderbook_path = data_path / "ETH-USDT-SWAP_orderbook.json"
    if not orderbook_path.exists():
        raise FileNotFoundError(f"Order book data not found at {orderbook_path}")
    
    orderbook_data = pd.read_json(orderbook_path)
    logger.info(f"Loaded {len(orderbook_data)} order book records")
    
    return ohlcv_data, orderbook_data

def process_orderbook_data(orderbook_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw order book data into a format suitable for analysis.
    
    Args:
        orderbook_data: Raw order book data
        
    Returns:
        Processed order book data
    """
    # Convert timestamp to datetime if it's not already
    if not isinstance(orderbook_data['timestamp'].iloc[0], datetime):
        orderbook_data['timestamp'] = pd.to_datetime(orderbook_data['timestamp'])
    
    # Set timestamp as index
    orderbook_data.set_index('timestamp', inplace=True)
    
    # Extract bid and ask prices and volumes
    processed_data = pd.DataFrame(index=orderbook_data.index)
    
    # Process bids
    processed_data['bid_prices'] = orderbook_data['bids'].apply(lambda x: [price for price, _ in x])
    processed_data['bid_volumes'] = orderbook_data['bids'].apply(lambda x: [vol for _, vol in x])
    
    # Process asks
    processed_data['ask_prices'] = orderbook_data['asks'].apply(lambda x: [price for price, _ in x])
    processed_data['ask_volumes'] = orderbook_data['asks'].apply(lambda x: [vol for _, vol in x])
    
    # Calculate total volumes
    processed_data['bid_volume_total'] = processed_data['bid_volumes'].apply(sum)
    processed_data['ask_volume_total'] = processed_data['ask_volumes'].apply(sum)
    
    # Calculate spread
    processed_data['spread'] = processed_data['ask_prices'].apply(lambda x: x[0]) - processed_data['bid_prices'].apply(lambda x: x[0])
    
    # Calculate order book imbalance
    processed_data['order_imbalance'] = (processed_data['bid_volume_total'] - processed_data['ask_volume_total']) / \
                                      (processed_data['bid_volume_total'] + processed_data['ask_volume_total'])
    
    return processed_data

def synchronize_data(ohlcv_data: pd.DataFrame, orderbook_data: pd.DataFrame) -> pd.DataFrame:
    """
    Synchronize OHLCV and order book data.
    
    Args:
        ohlcv_data: OHLCV data
        orderbook_data: Processed order book data
        
    Returns:
        Synchronized DataFrame
    """
    # Resample order book data to match OHLCV intervals
    orderbook_resampled = orderbook_data.resample('15T').agg({
        'bid_volume_total': 'mean',
        'ask_volume_total': 'mean',
        'spread': 'mean',
        'order_imbalance': 'mean'
    })
    
    # Merge with OHLCV data
    synchronized_data = pd.merge(
        ohlcv_data,
        orderbook_resampled,
        left_index=True,
        right_index=True,
        how='left'
    )
    
    # Fill missing values with forward fill
    synchronized_data.fillna(method='ffill', inplace=True)
    
    # Calculate additional features
    synchronized_data['volume_imbalance'] = (synchronized_data['bid_volume_total'] - synchronized_data['ask_volume_total']) / \
                                          (synchronized_data['bid_volume_total'] + synchronized_data['ask_volume_total'])
    
    synchronized_data['market_depth'] = synchronized_data['bid_volume_total'] + synchronized_data['ask_volume_total']
    
    return synchronized_data

def main():
    try:
        # Load data
        logger.info("Loading data...")
        ohlcv_data, orderbook_data = load_data()
        
        # Process order book data
        logger.info("Processing order book data...")
        processed_orderbook = process_orderbook_data(orderbook_data)
        
        # Synchronize data
        logger.info("Synchronizing data...")
        synchronized_data = synchronize_data(ohlcv_data, processed_orderbook)
        
        # Save synchronized data
        output_path = Path('data/historical/ETH-USDT-SWAP_synchronized.csv')
        synchronized_data.to_csv(output_path)
        logger.info(f"Saved synchronized data to {output_path}")
        
        # Print data summary
        logger.info("\nData Summary:")
        logger.info(f"Total records: {len(synchronized_data)}")
        logger.info(f"Date range: {synchronized_data.index.min()} to {synchronized_data.index.max()}")
        logger.info(f"Missing values: {synchronized_data.isnull().sum().sum()}")
        
    except Exception as e:
        logger.error(f"Error in data synchronization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 