import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import tempfile
import shutil
from data_management.high_frequency_collector import HighFrequencyCollector

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def collector(temp_data_dir):
    """Create a HighFrequencyCollector instance for testing."""
    return HighFrequencyCollector(
        exchange_id='binance',
        symbol='ETH/USD',
        interval='15m',
        order_book_depth=10,
        data_dir=temp_data_dir,
        max_retries=2,
        retry_delay=1
    )

@pytest.mark.asyncio
async def test_fetch_ohlcv(collector):
    """Test OHLCV data fetching."""
    # Fetch OHLCV data
    ohlcv_data = await collector.fetch_ohlcv()
    
    # Check DataFrame structure
    assert isinstance(ohlcv_data, pd.DataFrame)
    assert all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    # Check data types
    assert isinstance(ohlcv_data.index, pd.DatetimeIndex)
    assert all(ohlcv_data[col].dtype in ['float64', 'int64'] for col in ['open', 'high', 'low', 'close', 'volume'])
    
    # Check data validity
    assert len(ohlcv_data) > 0
    assert all(ohlcv_data['high'] >= ohlcv_data['low'])
    assert all(ohlcv_data['high'] >= ohlcv_data['open'])
    assert all(ohlcv_data['high'] >= ohlcv_data['close'])
    assert all(ohlcv_data['low'] <= ohlcv_data['open'])
    assert all(ohlcv_data['low'] <= ohlcv_data['close'])
    assert all(ohlcv_data['volume'] >= 0)

@pytest.mark.asyncio
async def test_fetch_order_book(collector):
    """Test order book data fetching."""
    # Fetch order book data
    order_book_data = await collector.fetch_order_book()
    
    # Check data structure
    assert isinstance(order_book_data, dict)
    assert 'timestamp' in order_book_data
    assert 'bids' in order_book_data
    assert 'asks' in order_book_data
    
    # Check data types
    assert isinstance(order_book_data['timestamp'], datetime)
    assert isinstance(order_book_data['bids'], list)
    assert isinstance(order_book_data['asks'], list)
    
    # Check data validity
    assert len(order_book_data['bids']) <= collector.order_book_depth
    assert len(order_book_data['asks']) <= collector.order_book_depth
    assert all(len(bid) == 2 for bid in order_book_data['bids'])
    assert all(len(ask) == 2 for ask in order_book_data['asks'])
    assert all(bid[0] < ask[0] for bid, ask in zip(order_book_data['bids'], order_book_data['asks']))

@pytest.mark.asyncio
async def test_collect_data(collector):
    """Test data collection for a short duration."""
    # Collect data for 2 minutes
    collection_task = asyncio.create_task(collector.collect_data(duration_minutes=2))
    await asyncio.sleep(120)  # Wait for collection to complete
    
    # Check collected data
    assert not collector.ohlcv_data.empty
    assert not collector.order_book_data.empty
    
    # Validate data
    validation_results = collector.validate_data()
    assert validation_results['ohlcv_complete']
    assert validation_results['order_book_complete']
    assert validation_results['data_quality']

def test_save_and_load_data(collector):
    """Test saving and loading data."""
    # Create some test data
    collector.ohlcv_data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [102, 103, 104],
        'low': [99, 100, 101],
        'close': [101, 102, 103],
        'volume': [1000, 1100, 1200]
    }, index=pd.date_range(start='2024-01-01', periods=3, freq='15min'))
    
    collector.order_book_data = pd.DataFrame([{
        'timestamp': datetime.now(),
        'bids': [[100, 1000], [99, 2000]],
        'asks': [[101, 1000], [102, 2000]]
    }])
    
    # Save data
    collector.save_data()
    
    # Clear data
    collector.ohlcv_data = pd.DataFrame()
    collector.order_book_data = pd.DataFrame()
    
    # Load data
    ohlcv_data, order_book_data = collector.load_data()
    
    # Check loaded data
    assert not ohlcv_data.empty
    assert not order_book_data.empty
    assert all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert all(col in order_book_data.columns for col in ['timestamp', 'bids', 'asks'])

def test_cleanup(collector):
    """Test cleanup functionality."""
    # Add some data
    collector.ohlcv_queue.put(pd.DataFrame())
    collector.order_book_queue.put(pd.DataFrame())
    collector.ohlcv_data = pd.DataFrame({'test': [1, 2, 3]})
    collector.order_book_data = pd.DataFrame({'test': [1, 2, 3]})
    
    # Run cleanup
    collector.cleanup()
    
    # Check cleanup results
    assert collector.ohlcv_queue.empty()
    assert collector.order_book_queue.empty()
    assert collector.ohlcv_data.empty
    assert collector.order_book_data.empty 