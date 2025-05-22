import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.synthetic_data_generator import SyntheticDataGenerator

@pytest.fixture
def data_generator():
    """Create a SyntheticDataGenerator instance for testing."""
    return SyntheticDataGenerator(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        base_price=100.0,
        volatility=0.02,
        trend_strength=0.1,
        volume_base=1000.0,
        seed=42
    )

def test_generate_ohlcv(data_generator):
    """Test OHLCV data generation."""
    # Generate data
    ohlcv_data = data_generator.generate_ohlcv(interval_minutes=15)
    
    # Check DataFrame structure
    assert isinstance(ohlcv_data, pd.DataFrame)
    assert all(col in ohlcv_data.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Check data types
    assert ohlcv_data['timestamp'].dtype == 'datetime64[ns]'
    assert all(ohlcv_data[col].dtype in ['float64', 'int64'] for col in ['open', 'high', 'low', 'close', 'volume'])
    
    # Check data validity
    assert len(ohlcv_data) > 0
    assert all(ohlcv_data['high'] >= ohlcv_data['low'])
    assert all(ohlcv_data['high'] >= ohlcv_data['open'])
    assert all(ohlcv_data['high'] >= ohlcv_data['close'])
    assert all(ohlcv_data['low'] <= ohlcv_data['open'])
    assert all(ohlcv_data['low'] <= ohlcv_data['close'])
    assert all(ohlcv_data['volume'] > 0)

def test_generate_order_book(data_generator):
    """Test order book data generation."""
    # Generate base data
    ohlcv_data = data_generator.generate_ohlcv(interval_minutes=15)
    
    # Generate order book
    order_book_data = data_generator.generate_order_book(ohlcv_data, n_levels=5)
    
    # Check DataFrame structure
    assert isinstance(order_book_data, pd.DataFrame)
    assert all(col in order_book_data.columns for col in ['timestamp', 'bid_prices', 'bid_volumes', 'ask_prices', 'ask_volumes'])
    
    # Check data types
    assert order_book_data['timestamp'].dtype == 'datetime64[ns]'
    assert all(isinstance(order_book_data[col].iloc[0], np.ndarray) for col in ['bid_prices', 'bid_volumes', 'ask_prices', 'ask_volumes'])
    
    # Check data validity
    assert len(order_book_data) == len(ohlcv_data)
    assert all(len(prices) == 5 for prices in order_book_data['bid_prices'])
    assert all(len(prices) == 5 for prices in order_book_data['ask_prices'])
    assert all(all(vol > 0) for vol in order_book_data['bid_volumes'])
    assert all(all(vol > 0) for vol in order_book_data['ask_volumes'])
    assert all(all(bid < ask) for bid, ask in zip(order_book_data['bid_prices'], order_book_data['ask_prices']))

def test_generate_features(data_generator):
    """Test feature generation."""
    # Generate base data
    ohlcv_data = data_generator.generate_ohlcv(interval_minutes=15)
    order_book_data = data_generator.generate_order_book(ohlcv_data)
    
    # Generate features
    features = data_generator.generate_features(ohlcv_data, order_book_data)
    
    # Check DataFrame structure
    assert isinstance(features, pd.DataFrame)
    expected_features = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'returns', 'rolling_mean', 'rolling_std', 'volume_ma',
                        'volume_std', 'bid_ask_spread', 'volume_imbalance',
                        'price_momentum', 'volume_momentum']
    assert all(col in features.columns for col in expected_features)
    
    # Check data validity
    assert len(features) == len(ohlcv_data)
    assert not features.isnull().any().any()  # No NaN values
    assert all(features['volume'] > 0)
    assert all(features['bid_ask_spread'] > 0)

def test_generate_target(data_generator):
    """Test target generation."""
    # Generate base data and features
    ohlcv_data = data_generator.generate_ohlcv(interval_minutes=15)
    order_book_data = data_generator.generate_order_book(ohlcv_data)
    features = data_generator.generate_features(ohlcv_data, order_book_data)
    
    # Generate target
    target = data_generator.generate_target(features, lookahead=5, threshold=0.001)
    
    # Check Series structure
    assert isinstance(target, pd.Series)
    assert target.dtype in ['int64', 'int32']
    
    # Check data validity
    assert len(target) == len(features)
    assert all(target.isin([0, 1]))  # Binary values
    assert not target.isnull().any()  # No NaN values

def test_generate_dataset(data_generator):
    """Test complete dataset generation."""
    # Generate complete dataset
    ohlcv_data, order_book_data, features, target = data_generator.generate_dataset(
        interval_minutes=15,
        n_levels=5,
        lookahead=5,
        threshold=0.001
    )
    
    # Check all components
    assert isinstance(ohlcv_data, pd.DataFrame)
    assert isinstance(order_book_data, pd.DataFrame)
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
    
    # Check data consistency
    assert len(ohlcv_data) == len(order_book_data) == len(features) == len(target)
    assert all(ohlcv_data['timestamp'] == order_book_data['timestamp'])
    assert all(ohlcv_data['timestamp'] == features['timestamp'])

def test_reproducibility(data_generator):
    """Test data generation reproducibility with same seed."""
    # Generate two datasets with same seed
    data1 = data_generator.generate_dataset(seed=42)
    data2 = data_generator.generate_dataset(seed=42)
    
    # Check that they are identical
    for df1, df2 in zip(data1, data2):
        pd.testing.assert_frame_equal(df1, df2)

def test_different_seeds(data_generator):
    """Test data generation with different seeds."""
    # Generate two datasets with different seeds
    data1 = data_generator.generate_dataset(seed=42)
    data2 = data_generator.generate_dataset(seed=43)
    
    # Check that they are different
    for df1, df2 in zip(data1, data2):
        assert not df1.equals(df2) 